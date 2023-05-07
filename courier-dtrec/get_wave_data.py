import pickle
import random
from collections import Counter
from copy import deepcopy
from math import ceil
from math import pi as PI

import folium
import matplotlib.pyplot as plt
import numpy as np
from coord_convert.transform import wgs2gcj
from matplotlib.ticker import MultipleLocator
from shapely.geometry import LineString, MultiPoint, Point, Polygon
from tqdm import tqdm
from matplotlib.font_manager import FontProperties

from constants import *

NUM_TOLERANCE = 5          # 容忍波两端也不一定完全是平的的数量
START_END_IGNORE = 6       # 若第一波开始前的单量很少, 或最后一波结束后的单量很少, 不再认为那里有trivial的start或end
START_END_SHIFT_GAP = 1800 # 最开始和最后面的两端, 可能发生零星的完成, 不希望这个影响对开始或结束时间的判断
WAVE_GAP = 3600            # 波两端需要往外走多远才能发生一定量的送货量改变
WAVE_PORTION = 0.07        # 波之间发生收货的比例
WAVE_EXTEND = 900          # 波时间范围的延展
TRAJ_GAP = 120             # 轨迹点间隔阈值, 判断轨迹覆盖时间轴的范围
TRAJ_COVER_GATE = 0.7      # 筛选轨迹覆盖比例理想的波
FILTER_WAVE_LENGTH = 1800  # 筛选时长较长的波
FILTER_WAVE_ORDER_NUM = 20 # 筛选单量较多的波
FILTER_WAVE_NOISE = 0.15   # 筛选轨迹点噪声较少的波
NEAR_STA_DIS2 = 120 ** 2   # 删除波两端在站附近的时段, 以及在站附近的驻留点

DIS_REGION = 100           # 过滤路区外的轨迹点
V_NOISE = 12               # 过滤噪声点速度

TRAJ_UPSAMPLE = 5          # 轨迹升采样时间间隔
D_STAY = 30                # 驻留点提取空间阈值
T_STAY = 60                # 驻留点提取时间阈值
D_STAY_MERGE = 45          # 驻留点合并空间阈值
T_STAY_MERGE = 60          # 驻留点合并时间阈值
FLAT_GATE = 0.0225         # 过滤掉很扁的驻留点(面积/周长**2, 该阈值相当于1:9的矩形)
POINT_AREA = 0.1           # 完全静止驻留点的默认面积
POINT_LENGTH = 2 * PI * (POINT_AREA / PI) ** 0.5  # 完全静止驻留点的默认周长(假设为圆形)
GRID_LEN = 300             # 驻留点特征, 空间栅格边长
SLOT_LEN = 3600            # 驻留点特征, 时间片长

random.seed(233)


def cum(ts, for_plot=False):
    """
    将一堆事件的发生时间统计为[(t, 累积发生次数)]的形式 
    """
    num0 = len([t for t in ts if t <= 0])
    t_nums = sorted(list(Counter([t for t in ts if t > 0]).items()), key=lambda x:x[0])
    if for_plot: 
        points = [(0, num0)]
    else:
        points = []
    cnt = num0
    for t, n in t_nums:
        if for_plot:
            points.append((t, cnt))
        cnt += n
        points.append((t, cnt))
    return points


def find_cum_at_t(points, t_target):
    if t_target < points[0][0]:
        return 0
    for i, (t, n) in enumerate(points):
        if t == t_target:
            return n
        if t > t_target:
            return points[i-1][1]
    return points[-1][1]


def get_waves(orders):
    """
    判断波的起止时间
    """
    start_times = [o["start_time"] for o in orders if o["type"] == ORDER_DELIVER]
    finish_times = [o["finish_time"] for o in orders]
    receive_points = cum(start_times)
    finish_points = cum(finish_times)
    
    # 顺序遍历, 判断t1是否是某波的结束
    ends = []
    jump_idx = 0
    last_n1 = 0  # 上一个end处的送货量
    last_end = None
    for i, (t1, n1) in enumerate(finish_points):  
        if i < jump_idx:
            continue
        if i == len(finish_points) - 1:  
            break
        for j in range(i+1, len(finish_points)):
            t2, n2 = finish_points[j]  # 往后找到订单完成数增加 达到一定阈值的t2(考虑到在站里也会有零星的完成)
            if n2 - n1 >= NUM_TOLERANCE:
                break
        else:
            # 最后一波结束的时间: 如果订单数增加5个的t2距离t1已经很久, 且上一个end到t1发生了一定量的送货
            if t2 - t1 > WAVE_GAP and n1 - last_n1 > len(orders) * (WAVE_PORTION + 0.03):
                last_end = t1
            break
        
        # debug
        # if t2 - t1 > WAVE_GAP:
        #     print(time_conventer(t1), time_conventer(t2), n1 - last_n1, find_cum_at_t(receive_points, t2) - find_cum_at_t(receive_points, t1))

        # 如果订单数增加5个的t2距离t1已经很久, 且上一个end到t1发生了一定量的送货, 且t1到t2内发生了一定量的收货, 则t1为某波的结束时间
        if t2 - t1 > WAVE_GAP and n1 - last_n1 > len(orders) * WAVE_PORTION and \
          find_cum_at_t(receive_points, t2) - find_cum_at_t(receive_points, t1) > len(orders) * WAVE_PORTION:
            ends.append(t1)
            last_n1 = n1
            jump_idx = j + 1  # 若t1是end, 下一个end应该至少在t2之后
        else:
            jump_idx = i + 1  # 若t1不是end, 继续往后找
    if ends:
        n_last_finish = find_cum_at_t(finish_points, ends[-1])
        # 若上面找出来的最后一波结束后, 剩下的完成数还比较多, 说面后面还有一波
        if finish_points[-1][1] - n_last_finish > START_END_IGNORE:
            if last_end is not None:
                ends.append(last_end)
            else:
                ends.append(finish_points[-1][0])
    
    # 逆序遍历, 判断t1是否是某波的开始
    starts = []
    jump_idx = len(finish_points)-1
    next_n1 = len(orders)  # 后一个start处的送货量
    first_start = None
    for i in range(len(finish_points)-1, -1, -1):  
        t1, n1 = finish_points[i]
        if i > jump_idx:
            continue
        if i == 0:
            break
        for j in range(i-1, -1, -1):
            t2, n2 = finish_points[j]  # 往前找到订单完成数减少 达到一定阈值的t2(考虑到在站里也会有零星的完成)
            if n1 - n2 >= NUM_TOLERANCE:
                break
        else:
            # 第一波开始的时间: 如果订单数减少5个的t2距离t1已经很久, 且t1到下一个start发生了一定量的送货
            if t2 - t1 > WAVE_GAP and next_n1 - n1 > len(orders) * (WAVE_PORTION + 0.03):
                first_start = t1
            break
        # 如果订单数减少5个的t2距离t1已经很久, 且t1到下一个start发生了一定量的送货, 且t2到t1内发生了一定量的收货, 则t1为某波的开始时间
        if t1 - t2 > WAVE_GAP and next_n1 - n1 > len(orders) * WAVE_PORTION and \
          find_cum_at_t(receive_points, t1) - find_cum_at_t(receive_points, t2) > len(orders) * WAVE_PORTION:
            starts.append(t1)
            next_n1 = n1
            jump_idx = j - 1  # 若t1是start, 上一个start应该至少在t2之前
        else:
            jump_idx = i - 1  # 若t1不是start, 继续往前找
    starts = starts[::-1]
    if starts:
        n_first_start = find_cum_at_t(finish_points, starts[0] - 1)
        # 若上面找出来的第一波开始前, 完成数还比较多, 说面前面还有一波
        if n_first_start > START_END_IGNORE:
            if first_start is not None:
                starts = [first_start] + starts
            else:
                starts = [finish_points[0][0]] + starts
    
    if len(starts) == 0 and len(ends) == 0 and len(orders) > 30:  # 只有一波
        starts = [first_start if first_start is not None else finish_points[0][0]]
        ends = [last_end if last_end is not None else finish_points[-1][0]]

    try:
        assert len(starts) == len(ends)
        waves = list(zip(starts, ends))
        last_e = -999
        for s, e in waves:
            assert e > s
            assert s > last_e
            last_e = e
    except:
        # print("Assert failed")
        # print([round(s / 3600, 1) for s in starts], [round(e / 3600, 1) for e in ends])
        if len(starts) < len(ends):
            starts += [0] * (len(ends) - len(starts))
        else:
            ends += [0] * (len(starts) - len(ends))
        return list(zip(starts, ends)), None  # 供画图debug用
    
    return waves


def plot_wave(orig_data):
    """
    画出 收货数, 完成数 随时间上升的曲线
    收货数只包括deliver
    完成数为全量
    同时画出波的起止时间, 以验证起止时间判断是否准确
    """
    sample_num = min(len(orig_data), 40)
    plt.figure(figsize=(6, sample_num*2))
    for idx, x in enumerate(random.sample(orig_data, sample_num)):
        orders = x["orders"]
        waves = get_waves(orders)
        if len(waves) and waves[-1] is None:
            # print(x["cid"], x["date"])
            waves = waves[0]
        start_times = [o["start_time"] for o in orders if o["type"] == ORDER_DELIVER]
        finish_times = [o["finish_time"] for o in orders]
        start_ts, start_ns = zip(*cum(start_times, for_plot=True))
        finish_ts, finish_ns = zip(*cum(finish_times, for_plot=True))
        start_ts = [t / 3600 for t in start_ts]
        finish_ts = [t / 3600 for t in finish_ts]
        plt.subplot(sample_num, 1, idx + 1)
        plt.xlim((0, 24))
        plt.gca().xaxis.set_major_locator(plt.MultipleLocator(1))
        plt.plot(start_ts, start_ns, label="r" + "_" + str(x["cid"]))
        plt.plot(finish_ts, finish_ns, label="f" + "_" + str(x["date"]))
        # plt.plot(start_ts, start_ns, label="Receive Num")
        # plt.plot(finish_ts, finish_ns, label="Finish Num")

        waves_long = [[s, e] for s, e in waves if e - s > FILTER_WAVE_LENGTH]
        waves_traj = []
        for i, (s, e) in enumerate(waves_long):
            if i == 0:
                if i + 1 < len(waves_long):
                    mid = (waves_long[i+1][0] + e) / 2  # 这波结束和下波开始的中点
                    waves_traj.append([max(s - WAVE_EXTEND, 0), min(e + WAVE_EXTEND, mid, 86399)])
                    last_mid = mid
                else:
                    waves_traj.append([max(s - WAVE_EXTEND, 0), min(e + WAVE_EXTEND, 86399)])
            else:
                if i + 1 < len(waves_long):
                    mid = (waves_long[i+1][0] + e) / 2
                    waves_traj.append([max(s - WAVE_EXTEND, last_mid, 0), min(e + WAVE_EXTEND, mid, 86399)])
                    last_mid = mid
                else:
                    waves_traj.append([max(s - WAVE_EXTEND, last_mid, 0), min(e + WAVE_EXTEND, 86399)])

        for s, e in waves:
            plt.axvline(s / 3600, 0, len(orders), c="black", ls="--", lw=1)
            plt.axvline(e / 3600, 0, len(orders), c="gray", ls="--", lw=1)
        for s, e in waves_traj:
            plt.axvline(s / 3600, 0, len(orders), c="black", lw=1)# , ls=":")
            plt.axvline(e / 3600, 0, len(orders), c="gray", lw=1)#, ls=":")
        plt.legend()

    plt.savefig("figure/wave.png")


def get_traj_tm_cover(traj):
    """
    计算轨迹在时间轴上的覆盖范围(轨迹点间隔过大处视为未覆盖)
    """
    ts = [p[-1] for p in traj]
    cover = 0
    for t1, t2 in zip(ts, ts[1:]):
        d = t2 - t1
        if d < TRAJ_GAP:
            cover += d
    return cover


def preprocess_orders(orders):
    for odr in orders:
        assert isinstance(odr["building_id"], int)
        floor = int(odr["floor"])
        if odr["building_id"] == -1:
            if floor == -1:
                floor = random.choice([1, 2, 3, 4, 5])
        else:
            bd = buildings[odr["building_id"]]
            if floor == -1:  # 不确定楼层的订单
                if bd.get("floor", None):  # 知道总层数, 随机抽
                    floor = random.choice(range(1, bd["floor"] + 1))
                else:
                    if bd["is_elevator"]:  # 不知道总层数, 根据楼是否为电梯楼随机拍
                        floor = random.choice([6, 7, 8, 9, 10])
                    else:
                        floor = random.choice([1, 2, 3, 4, 5])
            else:
                if bd.get("floor", None):
                    floor = min(floor, bd["floor"])
        floor = max(1, min(20, floor))
        odr["floor"] = floor
        unit = int(odr["unit"])
        if unit == -1:
            unit = 1
        unit = max(1, min(10, unit))
        odr["unit"] = unit
        assert odr["finish_time"] > odr["start_time"]
    return orders


def t2color(t):
    """
    用连续渐变的颜色来展现轨迹点时间
    """
    # cs = [237, 94, 6]   # 红 起始颜色
    # ce = [133, 63, 255]  # 蓝 结束颜色
    cs = [255, 75, 0]
    ce = [0, 75, 255]
    a = 1 - t
    b = 1 - a
    c = [round(a * xs + b * xe) for xs, xe in zip(cs, ce)]
    return "#" + "".join([f"{x:02X}" for x in c])  # 转为16进制表示的matplotlib颜色字符串


def cal_v(p1, p2):
    (x1, y1, t1), (x2, y2, t2) = p1, p2
    return ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5 / (t2 - t1)


def cal_dis(p1, p2):
    (x1, y1), (x2, y2) = p1, p2
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5


def denoise(traj, do_plot=False, plot_path="0figure/test_denoise.html"):
    """
    轨迹点去噪
    """
    # 过滤路区外的点
    to_remove = []
    for i, p in enumerate(traj):
        p = Point(p[:2])
        for r in regions.values():
            if p.distance(r["poly"]) < DIS_REGION:
                break
        else:
            to_remove.append(i)
    traj = [p for i, p in enumerate(traj) if i not in set(to_remove)]

    noise_idxs = []
    traj_filter = []
    # 启动: 找到第一个不是噪声点的点: 与其后2个点速度在V_NOISE内
    for i, p1 in enumerate(traj):
        for j in range(i + 1, i + 3):
            p2 = traj[j]
            if cal_v(p1, p2) >= V_NOISE:
                noise_idxs.append(i)
                break
        else:
            traj_filter.append(p1)
            break
    jump_idx = i + 1
    # 过滤噪声点: 与前1个点速度在V_NOISE内
    for i, p in enumerate(traj):
        if i < jump_idx:
            continue
        if cal_v(traj_filter[-1], p) < V_NOISE:
            traj_filter.append(p)
        else:
            noise_idxs.append(i)
    # for i, (p1, p2) in enumerate(zip(traj, traj[1:])):
    #     print(i, round(cal_v(p1, p2), 1), time_conventer(p1[-1]), time_conventer(p2[-1]))
    # print(noise_idxs)
    # exit()
    print("noise:", len(noise_idxs), "/", len(traj))
    if not do_plot:
        return traj_filter, len(noise_idxs) / len(traj)

    m = folium.Map(
        location=[LAT_CEN, LON_CEN],
        control_scale=True,
        tiles='http://webrd02.is.autonavi.com/appmaptile?lang=zh_cn&size=1&scale=1&style=8&x={x}&y={y}&z={z}',
        attr='高德底图',
        zoom_start=20,
    )
    folium.LatLngPopup().add_to(m)
    for b in buildings.values():
        color = "orange" if b["is_elevator"] else "black"
        folium.PolyLine(
            locations=[wgs2gcj(*p)[::-1] for p in b["points"]],
            opacity=0.8,
            weight=2,
            color=color
        ).add_to(m)
    st, et = traj[0][-1], traj[-1][-1]
    for i, (x, y, t) in enumerate(traj):
        r = 2 if i not in noise_idxs else 4
        c = t2color((t - st) / (et - st))
        folium.CircleMarker(
            location=wgs2gcj(*projector(x, y, inverse=True))[::-1],
            radius=r,
            color=c,
            fill=True,
            opacity=0.8,
            popup=time_conventer(t),
        ).add_to(m)
    folium.PolyLine(
        locations=[wgs2gcj(*projector(*p[:2], inverse=True))[::-1] for p in traj_filter],
        color="gray",
        weight=1,
        opacity=0.8,
    ).add_to(m)
    m.save(plot_path)
    return traj_filter, len(noise_idxs) / len(traj)


def get_stay_points(traj, do_ref=True, do_plot=False, plot_path="0figure/test_stay.html"):
    """
    升采样 5s
    提取驻留点
    {
        "se": (s, e),  # 驻留点 在 升采样后的轨迹点中的 idx
        "traj"         # 轨迹[s:e+1]
        "point"        # 平均坐标
        "trange"       # 起止时间
    }
    """
    def upscale_traj(traj, sample_gap=5):
        """
        将轨迹点升采样
        """
        traj_new = [traj[0]]
        last_x, last_y, last_t = traj[0]
        for x, y, t in traj[1:]:
            if t - last_t > sample_gap:
                section_num = ceil((t - last_t) / sample_gap)
                delta_x, delta_y, delta_t = x - last_x, y - last_y, t - last_t
                for i in range(1, section_num):
                    p = i / section_num 
                    traj_new.append((last_x + p*delta_x, last_y + p*delta_y, last_t + p*delta_t))
            traj_new.append((x, y, t))
            last_x, last_y, last_t = x, y, t
        return traj_new
    
    # 升采样
    traj = upscale_traj(traj, sample_gap=TRAJ_UPSAMPLE)
    # 提驻留点
    stays = []
    jump_idx = 0
    for i, (x, y, t) in enumerate(traj):
        if i < jump_idx:
            continue
        for j in range(i+1, len(traj)):
            if traj[j][-1] - t >= T_STAY:  # 往后找到第一个间隔大于T的点j
                break
        else:  # 找到末尾也没有, 直接结束
            break
        for k in range(j, i, -1):  # 其间的点k均需满足到i的距离小于D
            x1, y1 = traj[k][:2]
            if (x - x1) ** 2 + (y - y1) ** 2 > D_STAY ** 2:
                break
        else:  # 说明从i到j均为驻留点, 继续往j后面找, 直到距离i的距离>D
            if j == len(traj) - 1:
                k = j + 1
            else:
                for k in range(j+1, len(traj)):
                    x1, y1 = traj[k][:2]
                    if (x - x1) ** 2 + (y - y1) ** 2 > D_STAY ** 2:
                        break
                else:  # 遍历到最后一点, 也在距离内, 说明最后一点也是 
                    k += 1
            if not do_ref:
                stays.append([i, k-1])  # start end idx
                jump_idx = k  # 下次从驻留点之后开始找
            else:
                # 考虑到驻留点的起点往往还是边缘点, 用平均点和平均半径再次调整驻留点范围
                ps = [p[:2] for p in traj[i:k]]
                xs, ys = zip(*ps)
                center = (np.mean(xs), np.mean(ys))
                r = np.mean([cal_dis(p, center) for p in ps])
                r = max(D_STAY / 2, r)  # 新的r不要太小
                assert r <= D_STAY
                # 去掉头部一段
                for j in range(i, k):  
                    if cal_dis(traj[j][:2], center) <= r:
                        break
                else:
                    assert False
                i_new = j
                # 延伸尾部一段
                if k == len(traj):
                    k_new = k
                else:
                    for j in range(k, len(traj)):
                        if cal_dis(traj[j][:2], center) > r:
                            break
                    else:
                        j += 1
                    k_new = j
                assert i_new < k_new
                # 再次计算平均点和平均半径
                ps = [p[:2] for p in traj[i_new:k_new]]
                xs, ys = zip(*ps)
                center = (np.mean(xs), np.mean(ys))
                r = np.mean([cal_dis(p, center) for p in ps])
                r = max(D_STAY / 2, r)  # 新的r不要太小
                assert r <= D_STAY
                # 去掉尾部一段
                if k_new - i_new > 1:
                    for j in range(k_new - 1, i_new, -1):  
                        if cal_dis(traj[j][:2], center) <= r:
                            break
                    else:
                        assert False
                    k_new = j + 1
                assert i_new < k_new

                # 再次计算平均点和平均半径
                ps = [p[:2] for p in traj[i_new:k_new]]
                xs, ys = zip(*ps)
                center = (np.mean(xs), np.mean(ys))
                r = np.mean([cal_dis(p, center) for p in ps])
                r = max(D_STAY * 2 / 3, r)  # 新的r不要太小
                assert r <= D_STAY
                # 去掉头部一段
                for j in range(i_new, k_new):  
                    if cal_dis(traj[j][:2], center) <= r:
                        break
                else:
                    assert False
                i_new = j
                # 去掉尾部一段
                if k_new - i_new > 1:
                    for j in range(k_new - 1, i_new, -1):  
                        if cal_dis(traj[j][:2], center) <= r:
                            break
                    else:
                        assert False
                    k_new = j + 1
                
                if traj[k_new - 1][-1] - traj[i_new][-1] > T_STAY * 2 / 3:
                    stays.append([i_new, k_new-1])
                    jump_idx = k_new
                else:
                    print("short stay after process", round(traj[k_new - 1][-1] - traj[i_new][-1]))
            
    # 去除站点附近的驻留点
    to_remove = []
    for i, (s, e) in enumerate(stays):
        ps = traj[s:e+1]
        xs, ys, _ = zip(*ps)
        x, y = np.mean(xs), np.mean(ys)
        if (x - X_STA) ** 2 + (y - Y_STA) ** 2 < NEAR_STA_DIS2:
            to_remove.append(i)
    stays = [x for i, x in enumerate(stays) if i not in to_remove]
    orig_stay_num = len(stays)

    # 合并驻留点: 两相邻驻留点间时间间隔不大, 且合起来后可视为空间距离阈值稍大些的一个驻留点
    if do_ref and len(stays) > 1:
        stays_merge = [stays[0]]
        x, y = traj[stays[0][0]][:2]  # 上一段驻留点的起始位置
        last_e = stays[0][1]          # 上一段驻留点的结束idx
        last_t = traj[last_e][-1]     # 上一段驻留点的结束时间
        for s, e in stays[1:]:
            if traj[s][-1] - last_t < T_STAY_MERGE:
                for k in range(last_e + 1, e + 1):  # 从上一段结束到这一段结束, 距离小于阈值才合并
                    x1, y1 = traj[k][:2]
                    if (x - x1) ** 2 + (y - y1) ** 2 > D_STAY_MERGE ** 2:
                        stays_merge.append([s, e])
                        x, y = traj[s][:2]
                        last_e = e
                        last_t = traj[e][-1]
                        break
                else:  # 成功合并
                    stays_merge[-1][1] = e
                    last_e = e
                    last_t = traj[e][-1]
            else:
                stays_merge.append([s, e])
                x, y = traj[s][:2]
                last_e = e
                last_t = traj[e][-1]
        stays = stays_merge
    merged_stay_num = len(stays)

    # 检查合法
    last_e = -1
    for s, e in stays:
        assert e > s
        assert s > last_e, (s, last_e)
        last_e = e

    stays = [{"se": se} for se in stays]
    for i, x in enumerate(stays):
        s, e = x["se"]
        ps = traj[s:e+1]
        x["traj"] = ps
        xs, ys, ts = zip(*ps)
        x_avg, y_avg, t_avg = np.mean(xs), np.mean(ys), np.mean(ts)
        x["point"] = (round(x_avg, 6), round(y_avg, 6), t_avg)
        x["trange"] = (ts[0], ts[-1])
    stay_idxs = set(sum([list(range(x["se"][0], x["se"][1]+1)) for x in stays], []))
    print("stay points:", orig_stay_num, merged_stay_num, len(stay_idxs), "/", len(traj))
        
    if not do_plot:
        return traj, stays

    m = folium.Map(
        location=[LAT_CEN, LON_CEN],
        control_scale=True,
        tiles='http://webrd02.is.autonavi.com/appmaptile?lang=zh_cn&size=1&scale=1&style=8&x={x}&y={y}&z={z}',
        attr='高德底图',
        zoom_start=20,
    )
    folium.LatLngPopup().add_to(m)
    for b in buildings.values():
        color = "orange" if b["is_elevator"] else "black"
        folium.PolyLine(
            locations=[wgs2gcj(*p)[::-1] for p in b["points"]],
            opacity=0.8,
            weight=2,
            color=color
        ).add_to(m)
    st, et = traj[0][-1], traj[-1][-1]
    for i, (x, y, t) in enumerate(traj):
        if i % 3 != 0:  # and i not in stay_idxs:  # 点太密画出来看不清
            continue
        r = 1 if i not in stay_idxs else 2
        c = t2color((t - st) / (et - st))
        folium.CircleMarker(
            location=wgs2gcj(*projector(x, y, inverse=True))[::-1],
            radius=r,
            color=c,
            fill=True,
            opacity=0.8,
            popup=time_conventer(t),
        ).add_to(m)
    folium.PolyLine(
        locations=[wgs2gcj(*projector(*p[:2], inverse=True))[::-1] for p in traj],
        color="gray",
        weight=1,
        opacity=0.8,
    ).add_to(m)
    for x in stays:
        folium.Polygon(
            locations=[wgs2gcj(*projector(*p, inverse=True))[::-1] for p in x["poly"]],
            popup=(round(x["trange"][1] - x["trange"][0],1)),
            color="green",
            weight=1,
            fill=True,
            opacity=0.3,
        ).add_to(m)
        xx, yy, t = x["point"]
        folium.CircleMarker(
            location=wgs2gcj(*projector(xx, yy, inverse=True))[::-1],
            radius=3,
            color="black",
            fill=True,
            opacity=0.8,
            popup=(time_conventer(t)),
        ).add_to(m)

    m.save(plot_path)

    return traj, stays


def read_wave_data(orig_data, do_ref=True, do_plot=False):
    wave_data = []  # 以一波为基本单位的数据
    fail_waves = 0  # 统计因各种原因过滤掉的wave
    short_wave = 0
    long_wave = 0
    bad_traj_tm = 0
    bad_traj_noise = 0
    good_traj = 0
    for data in tqdm(orig_data):
        waves = get_waves(data["orders"])  # 原始wave范围比较紧
        if not waves or waves[-1] is None:
            print("fail wave:", data["cid"], data["date"])
            fail_waves += 1
            continue
        waves_long = [[s, e] for s, e in waves if e - s > FILTER_WAVE_LENGTH]  # 丢弃时长过短的wave
        # 在wave的两端删除在站附近的时段
        traj = data["traj"]
        for i, (s, e) in enumerate(waves_long):
            s_new = s
            for x, y, t in traj:
                if s < t < s + 1800:  # 避免删掉太多
                    if (x - X_STA) ** 2 + (y - Y_STA) ** 2 < NEAR_STA_DIS2:
                        s_new = t
                    else:
                        break
            e_new = e
            for x, y, t in traj[::-1]:
                if e - 1800 < t < e:  # 避免删掉太多
                    if (x - X_STA) ** 2 + (y - Y_STA) ** 2 < NEAR_STA_DIS2:
                        e_new = t
                    else:
                        break
            waves_long[i] = [s_new, e_new]
        waves_long = [[s, e] for s, e in waves_long if e - s > FILTER_WAVE_LENGTH]  # 丢弃时长过短的wave
        short_wave += len(waves) - len(waves_long)
        long_wave += len(waves_long)

        # 手动指定1101波的起止时间
        if data["date"] == 1:
            if data["cid"] == 21777999:    # 李文凯 电梯
                waves_long = [(7 * 3600 + 50 * 60, 14 * 3600 + 0 * 60), (16 * 3600 + 20 * 60, 19 * 3600 + 20 * 60)]
            elif data["cid"] == 22626330:  # 王茂林 楼梯
                waves_long = [(8 * 3600 + 10 * 60, 12 * 3600 + 0 * 60), (16 * 3600 + 30 * 60, 18 * 3600 + 20 * 60)]
            elif data["cid"] == 21173943:  # 王云飞 写字楼 工业园
                waves_long = [(7 * 3600 + 40 * 60, 14 * 3600 + 30 * 60), (16 * 3600 + 0 * 60, 19 * 3600 + 0 * 60)]
            elif data["cid"] == 22602631:  # 肖明江
                waves_long = [(8 * 3600 + 0 * 60, 11 * 3600 + 50 * 60), (16 * 3600 + 40 * 60, 17 * 3600 + 30 * 60)]

        waves_traj = []                 # 取轨迹的范围, 放松一点 (作为最终研究的时间范围)
        waves_order = []                # 取订单的范围, 更松 (取这波结束和下波开始的中点为分割)
        for i, (s, e) in enumerate(waves_long):
            if i == 0:
                if i + 1 < len(waves_long):
                    mid = (waves_long[i+1][0] + e) / 2  # 这波结束和下波开始的中点
                    waves_traj.append([max(s - WAVE_EXTEND, 0), min(e + WAVE_EXTEND, mid, 86399)])
                    waves_order.append([0, mid])
                    last_mid = mid
                else:
                    waves_traj.append([max(s - WAVE_EXTEND, 0), min(e + WAVE_EXTEND, 86399)])
                    waves_order.append([0, 86399])
            else:
                if i + 1 < len(waves_long):
                    mid = (waves_long[i+1][0] + e) / 2
                    waves_traj.append([max(s - WAVE_EXTEND, last_mid, 0), min(e + WAVE_EXTEND, mid, 86399)])
                    waves_order.append([last_mid, mid])
                    last_mid = mid
                else:
                    waves_traj.append([max(s - WAVE_EXTEND, last_mid, 0), min(e + WAVE_EXTEND, 86399)])
                    waves_order.append([last_mid, 86399])
        
        last_e = -999
        for s, e in waves_traj:
            assert e > s
            assert s >= last_e
            last_e = e
        last_e = -999
        for s, e in waves_order:
            assert e > s
            assert s >= last_e
            last_e = e
        assert len(waves_long) == len(waves_traj) == len(waves_order)

        cid = data["cid"]
        date = data["date"]
        orders = data["orders"]
        traj = data["traj"]

        wave_idx = 0
        for (s, e), (st, et), (so, eo) in zip(waves_long, waves_traj, waves_order):
            tj = [p for p in traj if st < p[-1] <= et]
            # print(time_conventer(st), time_conventer(et), round(get_traj_tm_cover(tj) / (e - s), 2))

            # if data["date"] == 1 and data["cid"] == 22602631:
            #     print(get_traj_tm_cover(tj) / (e - s) < TRAJ_COVER_GATE, get_traj_tm_cover(tj) / (e - s))
            if get_traj_tm_cover(tj) / (e - s) < TRAJ_COVER_GATE:
                bad_traj_tm += 1
                continue
            
            tj_denoise, noise_portion = denoise(tj, do_plot, f"figure/denoise_{cid}_{date}_{wave_idx}.html")
            # if data["date"] == 1 and data["cid"] == 22602631:
            #     print(noise_portion > FILTER_WAVE_NOISE, noise_portion)
            if noise_portion > FILTER_WAVE_NOISE:
                bad_traj_noise += 1
                continue
            
            good_traj += 1
            tj_new, stays = get_stay_points(tj_denoise, do_ref, do_plot, f"figure/stay_{cid}_{date}_{wave_idx}.html")
            odrs = [o for o in orders if so < o["finish_time"] <= eo]
            odrs.sort(key=lambda x: x["finish_time"])  # 订单按时间排序
            # odrs_orig = deepcopy(odrs)
            odrs = preprocess_orders(odrs)   # 将楼层为-1的订单随机到楼层
            wave_data.append({
                "cid": cid,
                "date": date,
                "orders": odrs,  
                # "orders_orig": odrs_orig,
                "traj": tj_new,
                "stays": stays,
                # "traj_orig": tj,
                "wave": (s, e),              # 算法判断出来的原始起止时间
                "wave_traj": (st, et),       # 向两边各拓展15min后的起止时间(也是取轨迹数据的时间范围)
                "wave_order": (so, eo),
                "wave_idx": wave_idx,
                "is_morning": st < 12 * 3600
            })
            wave_idx += 1
    print("fail_waves:", fail_waves, "/", len(orig_data))
    print("short_wave:", short_wave, "/", short_wave + long_wave)
    print("bad_traj_tm:", bad_traj_tm, "/", bad_traj_tm + bad_traj_noise + good_traj)
    print("bad_traj_noise:", bad_traj_noise, "/", bad_traj_tm + bad_traj_noise + good_traj)
    return wave_data


if __name__ == "__main__":
    # orig_data = pickle.load(open("data/orig_data.pkl", "rb"))
 
    # for x in orig_data:
    #     if x["cid"] == 22626330:
    #         print(x["date"], len(x["orders"]), len(x["traj"]))
    # exit()

    # # debug: 查看某一天对波起止位置的判断情况
    # for x in orig_data:
    #     if x["cid"] == 20485832 and x["date"] == 828:
    #         print(x["cid"], x["date"])
    #         plot_wave([x])
    #         exit()
    # exit()

    # # debug: 查看某小哥波的过滤情况
    # orig_data = [x for x in orig_data if x["cid"] == 22607367]
    # read_wave_data(orig_data, do_plot=False)
    # exit()

    # # debug: 查看所有波的判断情况
    # plot_wave(orig_data)
    # exit()

    # # debug: 调试去噪和驻留点提取的效果
    # wave_data = pickle.load(open("0data/wave_data.pkl", "rb"))
    # tcid, tdate = 20937077, 10
    # xs = [x for x in wave_data if x["cid"] == tcid and x["date"] == tdate]
    # for x in xs:
    #     traj, _ = denoise(x["traj"], True, f"0figure/denoise_{tcid}_{tdate}_{x['wave_idx']}.html")
    #     get_stay_points(traj, True, f"0figure/stay_{tcid}_{tdate}_{x['wave_idx']}.html")
    # exit()

    orig_data = pickle.load(open("data/orig_data.pkl", "rb"))
    wave_data = read_wave_data(orig_data, do_ref=True,do_plot=False)
    pickle.dump(wave_data, open("data/wave_data.pkl", "wb"))

    orig_data = pickle.load(open("data/orig_data.pkl", "rb"))
    wave_data = read_wave_data(orig_data, do_ref=False,do_plot=False)
    pickle.dump(wave_data, open("data/wave_data_no_stay_ref.pkl", "wb"))  # 不校正驻留点

    morning, afternoon = 0, 0
    for x in wave_data:
        if x["is_morning"]:
            afternoon += 1
        else:
            morning += 1
    print("morning waves:", morning)
    print("afternoon waves:", afternoon)
