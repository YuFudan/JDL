import pickle
import matplotlib.pyplot as plt
from math import ceil
from pyproj import Proj
from eviltransform import gcj2wgs, wgs2gcj
from shapely.geometry import Polygon, Point


projector = Proj("+proj=utm +zone=50 +ellps=WGS84 +datum=WGS84 +units=m +no_defs")

buildings = pickle.load(open("data/buildings_all_refined.pkl", "rb"))
for bd in buildings:
    bd["poly"] = Polygon([projector(*p) for p in bd["points"]])


def upscale_traj(points, sample_gap=10):
    """
    将轨迹点升采样到采样间隔不大于10s
    """
    points_new = [points[0]]
    last_x, last_y, last_t = points[0]
    for x, y, t in points[1:]:
        if t - last_t > sample_gap:
            section_num = ceil((t - last_t) / sample_gap)
            delta_x, delta_y, delta_t = x - last_x, y - last_y, t - last_t
            for i in range(1, section_num):
                p = i / section_num 
                points_new.append((last_x + p*delta_x, last_y + p*delta_y, last_t + p*delta_t))
        points_new.append((x, y, t))
        last_x, last_y, last_t = x, y, t
    return points_new


def find_points_indoor(traj_points, distance_gate=1):
    """
    找出在室内的轨迹点的时间(轨迹点是平面投影坐标)
    """
    ts = []
    for x, y, t in traj_points:
        for bd in buildings:
            if Point((x, y)).distance(bd["poly"]) < distance_gate:
                ts.append(t)
                break
    return ts


def plot_f(indoor_t, odrs_t, pts_t, min_t, max_t):
    """
    indoor_t: 在室内的时间
    odr_t: 所有订单完成时间
    pts_t: 所有在室内的轨迹点时间(可考虑先将轨迹点升采样到10s间隔看会不会有更好效果)
    """
    fig = plt.figure(figsize=(12, 10))

    ax1 = plt.subplot(3, 1, 1, xlim=(min_t, max_t), ylim=(0,1))
    for t_min, t_max in indoor_t:
        ax1.hlines(y=0.2, xmin=t_min, xmax=t_max)
    ax1.set_title('indoor_t')

    ax2 = plt.subplot(3, 1, 2, xlim=(min_t, max_t), ylim=(0,1)) 
    ax2.vlines(odrs_t, ymin=0, ymax=1)
    ax2.set_title('odr_t')

    ax3 = plt.subplot(3, 1, 3, xlim=(min_t, max_t), ylim=(0,1))
    ax3.vlines(pts_t, ymin=0, ymax=1)
    ax3.set_title('pts_t')

    plt.savefig("figure/test.png")


if __name__ == "__main__":
    # 输入轨迹 traj_points: [(lon, lat, t), ...]
    traj_points = [
        (116.428933,39.968672,0), 
        (116.428611,39.968898,100),
        (116.428,39.968898, 110),
        (116.428155,39.968898,130),
        (116.42742,39.968803,190),
        (116.42683,39.968795,250),
        (116.426487,39.968766,255),
        (116.42668,39.968799,270),
        (116.426423,39.968766,280),
        (116.427018,39.968396,340)
    ]

    # 将轨迹转为平面投影坐标
    traj_points = [(*projector(*gcj2wgs(lat, lon)[::-1]), t) for lon, lat, t in traj_points]  # 输入轨迹GCJ坐标
    # traj_points = [(*projector(lon, lat), t) for lon, lat, t in traj_points]  # 输入轨迹是WGS坐标
    
    # 将轨迹点升采样到时间间隔<10s
    traj_points = upscale_traj(traj_points, sample_gap=10) 

    # 找出在室内的轨迹点的时间
    pts_t = find_points_indoor(traj_points, distance_gate=1)

    # 订单完成时间
    odrs_t = [120, 125, 260, 265]

    # IOD数据在室内的时间
    indoor_t = [(100,150), (240, 290)]

    # 比较三个数据的时间
    min_t = min(min(t for tt in indoor_t for t in tt), min(odrs_t), min(pts_t))
    max_t = max(max(t for tt in indoor_t for t in tt), max(odrs_t), max(pts_t))
    plot_f(indoor_t=indoor_t, odrs_t=odrs_t, pts_t=pts_t, min_t=min_t, max_t=max_t)
