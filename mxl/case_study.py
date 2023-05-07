"""无法复现之前结果, 仍暂时使用/validate/case_study.py"""
import random
import folium
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from collections import defaultdict
from constants_all import *
from shapely.geometry import LineString
from math import ceil


def t2color(t):
    """用连续渐变的颜色来展现轨迹点时间"""
    # cs = [237, 94, 6]   # 红 起始颜色
    # ce = [133, 63, 255]  # 蓝 结束颜色
    cs = [255, 75, 0]
    ce = [0, 75, 255]
    a = 1 - t
    b = 1 - a
    c = [round(a * xs + b * xe) for xs, xe in zip(cs, ce)]
    return "#" + "".join([f"{x:02X}" for x in c])  # 转为16进制表示的matplotlib颜色字符串


def find_traj_at_t(traj, t):
    if not traj[0][-1] <= t <= traj[-1][-1]:
        print("t not in traj")
        exit()
    for (x1, y1, t1), (x2, y2, t2) in zip(traj, traj[1:]):
        if t1 <= t <= t2:
            a = (t2 - t) / (t2 - t1)
            b = (t - t1) / (t2 - t1)
            return x1 * a + x2 * b, y1 * a + y2 * b
    assert False


def xy2loc(xy):
    return wgs2gcj(*projector(*xy, inverse=True))[::-1]


def plot_gt(wave):
    traj = wave["traj"]
    stays = wave["stays"]
    stay_idxs = set(sum([list(range(x["se"][0], x["se"][1]+1)) for x in stays], []))
    odrs = wave["orders"]
    bid2odrts = defaultdict(list)
    for o in odrs:
        bid2odrts[o["building_id"]].append(o["finish_time"])
    for v in bid2odrts.values():
        v.sort()
    st, et = traj[0][-1], traj[-1][-1]
    odrs = [o for o in odrs if st <= o["finish_time"] <= et]
    ts = [o["finish_time"] for o in odrs]
    t_min, t_max = min(ts) - 30, max(ts) + 30
    traj = [p for p in traj if t_min <= p[-1] <= t_max]

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
        odrts = bid2odrts[b["id"]]
        weight = 3 if odrts else 1
        folium.PolyLine(
            locations=[wgs2gcj(*p)[::-1] for p in b["points"]],
            opacity=0.8,
            weight=weight,
            color=color,
            popup=[time_conventer(t) for t in odrts]
        ).add_to(m)


    st, et = traj[0][-1], traj[-1][-1]
    for i, (x, y, t) in enumerate(traj):
        if i % 3 != 0:
            continue
        r = 1  # if i not in stay_idxs else 2
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
    # for x in stays:
    #     # folium.Polygon(
    #     #     locations=[wgs2gcj(*projector(*p, inverse=True))[::-1] for p in x["poly"]],
    #     #     popup=(time_conventer(x["trange"][0]), time_conventer(x["trange"][1]), round(x["trange"][1] - x["trange"][0],1)),
    #     #     color="green",
    #     #     weight=1,
    #     #     fill=True,
    #     #     opacity=0.3,
    #     # ).add_to(m)
    #     xx, yy, t = x["point"]
    #     folium.CircleMarker(
    #         location=wgs2gcj(*projector(xx, yy, inverse=True))[::-1],
    #         radius=4,
    #         color="black",
    #         fill=True,
    #         opacity=0.8,
    #         popup=(time_conventer(x["trange"][0]), time_conventer(x["trange"][1]), round(x["trange"][1] - x["trange"][0],1)),
    #     ).add_to(m)

    # xy2oids = defaultdict(list)
    # for o in odrs:
    #     if st <= o["finish_time"] <= et:
    #         x, y = find_traj_at_t(traj, o["finish_time"])
    #         x, y = round(x), round(y)
    #         xy2oids[(x, y)].append(o["id"])
    # random_offset = 10
    # for (x, y), oids in xy2oids.items():
    #     for oid in oids:
    #         folium.Marker(
    #             xy2loc((x + random.uniform(-random_offset, random_offset),y + random.uniform(-random_offset, random_offset))),
    #             icon=folium.DivIcon(
    #                 icon_size=(20,6),
    #                 icon_anchor=(10,3),
    #                 html=f'<div style="font-size: 10pt" text-align="center">{int(oid)}</div>',
    #                 )
    #             ).add_to(m)
    #         y += 5

    m.save(f"figure/gt_wave_{cid2name[wave['cid']]}_{wave['date']}_{wave['wave_idx']}.html")


def plot_sim(actions):
    cid, date, wid = actions["cid"], actions["date"], actions["wave_idx"]
    actions = actions["actions"]
    T_SAMPLE = 4

    def get_traj(xys, t1, t2):
        """按一定采样率插值得到轨迹"""
        traj = [[*xys[0], t1]]
        p2 = [*xys[-1], t2]
        n = int((t2 - t1) / T_SAMPLE)
        if n == 0:
            traj.append(p2)
            return traj
        line = LineString(xys)
        s_step = line.length / (n + 1)
        t_step = (t2 - t1) / (n + 1)
        for i in range(n):
            s = s_step * (i + 1)
            t = t1 + t_step * (i + 1)
            xy = line.interpolate(s).coords[:][0]
            traj.append([*xy, t])
        traj.append(p2)
        return traj

    odrs = []
    traj = []
    for a in actions:
        if a["type"] in ACTION_ORDER:
            o = a["target_orders"][0]
            o["finish_time"] = a["end_time"]
            odrs.append(o)
        elif a["type"] in ACTION_MOVE:
            gpss = a["gps"]
            xys = [projector(*p) for p in gpss]
            t1, t2 = a["start_time"], a["end_time"]
            traj += get_traj(xys, t1, t2)
    bid2odrts = defaultdict(list)
    for o in odrs:
        bid2odrts[o["building_id"]].append(o["finish_time"])
    for v in bid2odrts.values():
        v.sort()

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
        odrts = bid2odrts[b["id"]]
        weight = 3 if odrts else 1
        folium.PolyLine(
            locations=[wgs2gcj(*p)[::-1] for p in b["points"]],
            opacity=0.8,
            weight=weight,
            color=color,
            popup=[time_conventer(t) for t in odrts]
        ).add_to(m)

    st, et = traj[0][-1], traj[-1][-1]
    for x, y, t in traj:
        r = 1
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

    # xy2oids = defaultdict(list)
    # for o in odrs:
    #     if st <= o["finish_time"] <= et:
    #         x, y = find_traj_at_t(traj, o["finish_time"])
    #         x, y = round(x, 6), round(y, 6)
    #         xy2oids[(x, y)].append(o["id"])
    # for (x, y), oids in xy2oids.items():
    #     for oid in oids:
    #         folium.Marker(
    #             xy2loc((x,y)),
    #             icon=folium.DivIcon(
    #                 icon_size=(20,6),
    #                 icon_anchor=(10,3),
    #                 html=f'<div style="font-size: 10pt" text-align="center">{int(oid)}</div>',
    #                 )
    #             ).add_to(m)
    #         y += 5
    m.save(f"figure/sim_wave_{cid2name[cid]}_{date}_{wid}.html")


def compare_taxis(wave, actions):
    actions = actions["actions"]
    traj = wave["traj"]
    t_min, t_max = int(traj[0][-1] / 3600), ceil(traj[-1][-1] / 3600)
    plt.figure(figsize=(30, 2))
    x_min, x_max = int(min(t_min, actions[0]["start_time"]/3600)), int(max(t_max, actions[-1]["end_time"]/3600)) + 1
    plt.xlim((x_min, x_max))
    ts = np.arange(x_min, x_max, 1/6)
    ls = [time_conventer(3600*t)[:-3] for t in ts]
    # plt.gca().xaxis.set_major_locator(MultipleLocator(0.1))
    ax = plt.gca()
    ax.set_xticks(ts)
    ax.set_xticklabels(ls)
    plt.ylim((0.7, 1.4))

    t2odrs = defaultdict(list)
    for o in wave["orders"]:
        t2odrs[int(o["finish_time"])].append(o)
    t_odrs = list(t2odrs.items())
    groups = []
    group = [t_odrs[0]]  # 避免文字重叠
    for t, odrs in t_odrs[1:]:
        if t <= group[-1][0] + 30:
            group.append((t, odrs))
        else:
            groups.append(group)
            group = [(t, odrs)]
    groups.append(group)
    oid2ty1 = {}
    for group in groups:
        y = 1.02
        for t, odrs in group:
            x, xmin, xmax = t / 3600, (t - 5) / 3600, (t + 5) / 3600
            for o in odrs:
                # if o["id"] == 2336:
                #     print(2336)
                #     t2336 = t
                #     continue
                plt.hlines(y=y, xmin=xmin, xmax=xmax, color="red")
                oid2ty1[o["id"]] = [t, y]
                y += 0.03
                plt.text(x=x, y=y, s=str(int(o["id"])), color="black", ha="center", size=4)
                y += 0.03
    for x in wave["stays"]:
        t1, t2 = x["traj"][0][-1], x["traj"][-1][-1]
        # if t1 < t2336 < t2:
        #     t2 -= 180
        plt.hlines(y=1.0, xmin=t1 / 3600, xmax=t2 / 3600)
        
        
    # ===================================== #

    stay_ranges = []
    t1 = actions[0]["start_time"]
    for a in actions:
        if a["type"] in ACTION_MOVE:
            t2 = a["start_time"]
            if t2 > t1:
                stay_ranges.append([t1, t2])
            t1 = a["end_time"]
    t2 = actions[-1]["end_time"]
    if t2 > t1:
        stay_ranges.append([t1, t2])

    orders = []
    for a in actions:
        if a["type"] in ACTION_ORDER:
            o = a["target_orders"][0]
            o["finish_time"] = a["end_time"]
            orders.append(o)
    t2odrs = defaultdict(list)
    for o in orders:
        t2odrs[int(o["finish_time"])].append(o)
    t_odrs = list(t2odrs.items())
    groups = []
    group = [t_odrs[0]]  # 避免文字重叠
    for t, odrs in t_odrs[1:]:
        if t <= group[-1][0] + 30:
            group.append((t, odrs))
        else:
            groups.append(group)
            group = [(t, odrs)]
    groups.append(group)
    oid2ty2 = {}
    for group in groups:
        y = 0.88
        for t, odrs in group:
            x, xmin, xmax = t / 3600, (t - 5) / 3600, (t + 5) / 3600
            for o in odrs:
                plt.hlines(y=y, xmin=xmin, xmax=xmax, color="red")
                oid2ty2[o["id"]] = [t, y]
                y -= 0.03
                plt.text(x=x, y=y, s=str(int(o["id"])), color="black", ha="center", size=4)
                y -= 0.03
    for t1, t2 in stay_ranges:
        plt.hlines(y=0.9, xmin=t1 / 3600, xmax=t2 / 3600)

    for oid, (t1, y1) in oid2ty1.items():
        t2, y2 = oid2ty2[oid]
        if abs(t1 - t2) < 600:
            plt.plot([t1/3600, t2/3600], [y1, y2], linestyle="--", linewidth=0.1, color="gray")

    plt.savefig(f"figure/gt_sim_wave_{cid2name[wave['cid']]}_{wave['date']}_{wave['wave_idx']}.pdf")


def main():
    waves = pickle.load(open("data/eval_datas_21_10_nofilter.pkl", "rb"))[1]
    actions = pickle.load(open(f"data/sim_actions_nofilter.pkl", "rb"))

    # for cid, date, wid in [(21495458, "2022-8-24", 1)]:
    for cid, date, wid in [(21777999, "2022-8-27", 1)]:
        w = [w for w in waves[cid] if w["date"] == date and w["wave_idx"] == wid][0]
        min_oid = min(o["id"] for o in w["orders"])
        for o in w["orders"]:
            o["id"] -= min_oid
        action = [a for a in actions[cid] if a["date"] == date and a["wave_idx"] == wid][0]
        for a in action["actions"]:
            if a["type"] in ACTION_ORDER:
                for o in a["target_orders"]:
                    o["id"] -= min_oid
        # plot_gt(w)
        # plot_sim(action)
        compare_taxis(w, action)


if __name__ == "__main__":
    main()
