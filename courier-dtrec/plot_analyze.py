import pickle
import random
from collections import defaultdict
from copy import deepcopy
from math import ceil
from pprint import pprint

import folium
import matplotlib.pyplot as plt
import numpy as np
from coord_convert.transform import wgs2gcj
from matplotlib.font_manager import FontProperties
from matplotlib.pyplot import MultipleLocator

from constants import *

random.seed(233)

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


def plot_wave_odr_stay(wave):
    """
    在时间轴上对比订单完成时间和驻留点时间
    """
    traj = wave["traj"]
    t_min, t_max = int(traj[0][-1] / 3600), ceil(traj[-1][-1] / 3600)
    plt.figure(figsize=(30, 2))
    plt.xlim((t_min, t_max))
    plt.ylim((0, 2))
    plt.gca().xaxis.set_major_locator(MultipleLocator(1))
    for o in wave["orders"]:
        t1, t2 = o["finish_time"] - 10, o["finish_time"] + 10
        plt.hlines(y=1, xmin=t1 / 3600, xmax=t2 / 3600)
        plt.text(x=(t1 + t2) / 7200, y=1.1, s=str(o["building_id"]), color="black", ha="center", size=2)
    for x in wave["stays"]:
        t1, t2 = x["traj"][0][-1], x["traj"][-1][-1]
        plt.hlines(y=0.9, xmin=t1 / 3600, xmax=t2 / 3600)
        plt.text(x=(t1 + t2) / 7200, y=0.8, s=str(round(t2 - t1)), color="black", ha="center", size=2)
    plt.savefig(f"figure/odr_stay_{cid2name[wave['cid']]}_{wave['date']}_{wave['wave_idx']}.pdf")


def folium_wave_odr_stay(wave):
    traj = wave["traj"]
    stays = wave["stays"]
    stay_idxs = set(sum([list(range(x["se"][0], x["se"][1]+1)) for x in stays], []))

    odrs = wave["orders"]
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
    for i, (x, y, t) in enumerate(traj):
        if i % 3 != 0:
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
        # folium.Polygon(
        #     locations=[wgs2gcj(*projector(*p, inverse=True))[::-1] for p in x["poly"]],
        #     popup=(time_conventer(x["trange"][0]), time_conventer(x["trange"][1]), round(x["trange"][1] - x["trange"][0],1)),
        #     color="green",
        #     weight=1,
        #     fill=True,
        #     opacity=0.3,
        # ).add_to(m)
        xx, yy, t = x["point"]
        folium.CircleMarker(
            location=wgs2gcj(*projector(xx, yy, inverse=True))[::-1],
            radius=4,
            color="black",
            fill=True,
            opacity=0.8,
            popup=(time_conventer(x["trange"][0]), time_conventer(x["trange"][1]), round(x["trange"][1] - x["trange"][0],1)),
        ).add_to(m)

    m.save(f"figure/odr_stay_{cid2name[wave['cid']]}_{wave['date']}_{wave['wave_idx']}.html")


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


def find_traj_idx_at_t(traj, t):
    if not traj[0][-1] <= t <= traj[-1][-1]:
        print("t not in traj")
        exit()
    for i, p in enumerate(traj):
        if p[-1] > t:
            return i - 1
        elif p[-1] == t:
            return i


def folium_wave_with_gt(wave, actions, events):
    """
    画图分析一波中的轨迹, 订单, 驻留点 与 真值标注
    发现真值标注的坐标可能不准确, 改为使用真值标注的时间所在的轨迹坐标
    """
    traj = wave["traj"]
    tj_st, tj_et = traj[0][-1], traj[-1][-1]
    stays = wave["stays"]
    stay_idxs = set(sum([list(range(x["se"][0], x["se"][1]+1)) for x in stays], []))
    
    odrs = wave["orders"]
    bid2odrts = defaultdict(list)
    for o in odrs:
        bid2odrts[o["building_id"]].append(o["finish_time"])
    for v in bid2odrts.values():
        v.sort()

    actions = {a["idx"]: a for a in actions}
    events = [[actions[i] for i in range(e[0], e[-1]+1)] for e in events]
    events = [e for e in events if tj_st < e[0]["st"] and e[-1]["et"] < tj_et]

    # 找到action时间范围内的轨迹, 以更新真值坐标
    for e in events:
        for a in e:
            a["sxy"] = find_traj_at_t(traj, a["st"])
            a["exy"] = find_traj_at_t(traj, a["et"])
            x1, y1, t1 = *a["sxy"], a["st"]
            x2, y2, t2 = *a["exy"], a["et"]
            a["ave_point"] = ((x1 + x2) / 2, (y1 + y2) / 2, (t1 + t2) / 2)

    # 底图
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

    # 轨迹
    for i, (x, y, t) in enumerate(traj):
        if i in stay_idxs:
            if i % 3 != 0:
                continue
        else:
            if i % 3 != 0:
                continue
        r = 1 if i not in stay_idxs else 1
        c = t2color((t - tj_st) / (tj_et - tj_st))
        folium.CircleMarker(
            location=wgs2gcj(*projector(x, y, inverse=True))[::-1],
            radius=r,
            color=c,
            fill=True,
            opacity=0.8,
            popup=time_conventer(t),
        ).add_to(m)
    # folium.PolyLine(
    #     locations=[wgs2gcj(*projector(*p[:2], inverse=True))[::-1] for p in traj],
    #     color="gray",
    #     weight=1,
    #     opacity=0.8,
    # ).add_to(m)

    # 订单组
    groups = wave["groups"]
    for i, g in enumerate(groups):
        x, y = g["address"]
        x, y = x + random.uniform(0, 5), y + random.uniform(0, 5)  # 避免点重合
        location = wgs2gcj(*projector(x, y, inverse=True))[::-1]
        g["location"] = location
        folium.CircleMarker(
            location=location,
            radius=4,
            color="red",
            fill=True,
            opacity=0.8,
            popup=(i, [time_conventer(o["finish_time"]) for o in g["orders"]]),
        ).add_to(m)

    # 驻留点
    for x in stays:
        t1, t2 = x["trange"]
        for e in events:  # 在真值工作期间内的驻留点加粗显示
            a, b = e[0]["st"], e[-1]["et"]
            if not (b < t1 or a > t2):
                weight = 1.5
                opacity = 0.8
                break
        else:
            weight = 1
            opacity = 0.3
        folium.PolyLine(
            locations=[wgs2gcj(*projector(*p, inverse=True))[::-1] for p in x["poly"]],
            popup=(time_conventer(x["trange"][0]), time_conventer(x["trange"][1]), round(x["trange"][1] - x["trange"][0],1)),
            color="green",
            weight=weight,
            # fill=True,
            opacity=opacity,
        ).add_to(m)
        xx, yy, t = x["point"]
        location = wgs2gcj(*projector(xx, yy, inverse=True))[::-1]
        folium.CircleMarker(
            location=location,
            radius=4,
            color="green",
            fill=True,
            opacity=opacity,
            popup=(time_conventer(x["trange"][0]), time_conventer(x["trange"][1]), round(x["trange"][1] - x["trange"][0],1)),
        ).add_to(m)
        # 与订单组匹配
        for gid, (ss, st) in zip(x["gids_matched"], x["match_scores"]):
            g = groups[gid]
            folium.PolyLine(
                locations=[location, g["location"]],
                weight=1.5,
                color="red",
                opacity=0.8,
                popup=(gid, round(100*ss, 1), round(100*st, 1))
            ).add_to(m)

    # 真值行为
    for e in events:
        locations = []
        acts = []
        for a in e:
            x, y, t = a["ave_point"]
            # c = t2color((t - tj_st) / (tj_et - tj_st))
            c = "black"
            t1, t2 = a["st"], a["et"]
            location = wgs2gcj(*projector(x, y, inverse=True))[::-1]
            folium.CircleMarker(
                location=location,
                radius=3,
                color=c,
                fill=True,
                opacity=0.7,
                popup=(a["act"], a["idx"], time_conventer(t1), time_conventer(t2), a["smsg"])
            ).add_to(m)
            locations.append(location)
            acts.append(a["act"])
                
        t1, t2 = e[0]["st"], e[-1]["et"]
        t = (t1 + t2) / 2
        c = t2color((t - tj_st) / (tj_et - tj_st))
        folium.PolyLine(
            locations=locations,
            color="black",
            # color=c,
            weight=1.5,
            opacity=1,
            popup=(acts, time_conventer(t1), time_conventer(t2), e[0]["idx"], e[-1]["idx"])
        ).add_to(m)
        # i, j = find_traj_idx_at_t(traj, e[0]["st"]), find_traj_idx_at_t(traj, e[-1]["et"])
        # tj = traj[i: j+2]
        # folium.PolyLine(
        #     locations=[wgs2gcj(*projector(*p[:2], inverse=True))[::-1] for p in tj],
        #     color="gray",
        #     weight=1,
        #     opacity=0.8,
        #     popup=(acts, time_conventer(e[0]["st"]), time_conventer(e[-1]["et"]), e[0]["idx"], e[-1]["idx"])
        # ).add_to(m)

        # 动作序号文本
        lats, lons = zip(*locations)
        if len(e) == 1:
            txt = str(e[0]["idx"])
        else:
            txt = str(e[0]["idx"]) + "," + str(e[-1]["idx"])
        has_pick = False
        for a in e:
            for k in ["act", "smsg"]:
                if "揽" in a[k]:
                    has_pick = True
                    break
            if has_pick:
                break
        font_size = 12 if has_pick else 10
        folium.Marker(
            [np.mean(lats)-0.00003, np.mean(lons)],
            icon=folium.DivIcon(
                icon_size=(40,10),
                icon_anchor=(20,5),
                html=f'<div style="font-size: {font_size}pt" text-align="center">{txt}</div>',
                )
            ).add_to(m)
    m.save(f"figure/wave_gt_{wave['cid']}_{wave['date']}_{wave['wave_idx']}.html")


def plot_wave_with_gt(wave, actions, events):
    """
    通过画图辅助判断 真值行为与订单的对应关系
    """
    traj = wave["traj"]
    tj_st, tj_et = traj[0][-1], traj[-1][-1]
    actions = {a["idx"]: a for a in actions}
    events = [[actions[i] for i in range(e[0], e[-1]+1)] for e in events]
    events = [e for e in events if tj_st < e[0]["st"] and e[-1]["et"] < tj_et]

    font = FontProperties(fname=r"msyh.ttc")
  
    plt.figure(figsize=(30, 2))
    plt.xlim((int(traj[0][-1] / 3600), ceil(traj[-1][-1] / 3600)))
    plt.ylim((0, 2))
    plt.gca().xaxis.set_major_locator(MultipleLocator(1))
    for g in wave["groups"]:
        t1, t2 = g["orders"][0]["finish_time"] - 10, g["orders"][-1]["finish_time"] + 10
        plt.hlines(y=1.2, xmin=t1 / 3600, xmax=t2 / 3600)
        plt.text(x=(t1 + t2) / 7200, y=1.3, s=str(g["address_id"]), color="black", ha="center", size=2)
    for o in wave["orders"]:
        t1, t2 = o["finish_time"] - 10, o["finish_time"] + 10
        plt.hlines(y=1, xmin=t1 / 3600, xmax=t2 / 3600)
        plt.text(x=(t1 + t2) / 7200, y=1.1, s=str(o["building_id"]), color="black", ha="center", size=2)  # TODO:
    for e in events:
        t1, t2 = e[0]["st"], e[-1]["et"]
        if t1 == t2:
            t1 -= 10
            t2 += 10
        else:
            t1 += 10
            t2 -= 10
        plt.hlines(y=0.8, xmin=t1 / 3600, xmax=t2 / 3600)
        plt.text(x=(t1 + t2) / 7200, y=0.9, s=",".join([a["act"] for a in e]), color="black", ha="center", size=2, fontproperties=font)
    groups = wave["groups"]
    for x in wave["stays"]:
        t1, t2 = x["traj"][0][-1], x["traj"][-1][-1]
        plt.hlines(y=0.7, xmin=t1 / 3600, xmax=t2 / 3600)
        plt.text(x=(t1 + t2) / 7200, y=0.6, s=str(round(t2 - t1)), color="black", ha="center", size=2)
        plt.text(x=(t1 + t2) / 7200, y=0.5, s=str([groups[gid]["address_id"] for gid in x["gids_matched"]]), color="black", ha="center", size=2)
    plt.savefig(f"figure/wave_gt_{wave['cid']}_{wave['date']}_{wave['wave_idx']}.pdf")


def label_action_with_order(wave, actions, target_oids):
    """
    通过画图, 人工标注action上的order
    """
    orders = wave["orders"]
    oid2odr = {o["id"]: o for o in orders}
    st, et = wave["wave"]
    traj = wave["traj"]
    tj_st, tj_et = traj[0][-1], traj[-1][-1]
    stays = wave["stays"]
    stay_idxs = set(sum([list(range(x["se"][0], x["se"][1]+1)) for x in stays], []))
    # 截取在波时段内的真值
    actions = deepcopy(actions)
    for i, a in enumerate(actions):
        if a["et"] > st:
            break
    for j in range(len(actions)-1, -1, -1):
        a = actions[j]
        if a["st"] < et:
            break
    actions = actions[i:j+1]

    # 空间图
    m = get_base_map()
    # 轨迹
    for i, (x, y, t) in enumerate(traj):
        if i in stay_idxs:
            if i % 3 != 0:
                continue
        else:
            if i % 3 != 0:
                continue
        r = 1 if i not in stay_idxs else 1
        c = t2color((t - tj_st) / (tj_et - tj_st))
        folium.CircleMarker(
            location=wgs2gcj(*projector(x, y, inverse=True))[::-1],
            radius=r,
            color=c,
            fill=True,
            opacity=0.5,
            popup=time_conventer(t),
        ).add_to(m)
    # stays
    for x in stays:
        t1, t2 = x["trange"]
        for a in actions:  # 在真值工作期间内的驻留点加粗显示
            a, b = a["st"], a["et"]
            if not (b < t1 or a > t2):
                weight = 1
                opacity = 0.5
                break
        else:
            weight = 1
            opacity = 0.3
        folium.PolyLine(
            locations=[wgs2gcj(*projector(*p, inverse=True))[::-1] for p in x["poly"]],
            popup=(time_conventer(x["trange"][0]), time_conventer(x["trange"][1]), round(x["trange"][1] - x["trange"][0],1)),
            color="green",
            weight=weight,
            # fill=True,
            opacity=opacity,
        ).add_to(m)
        # xx, yy, t = x["point"]
        # location = wgs2gcj(*projector(xx, yy, inverse=True))[::-1]
        # folium.CircleMarker(
        #     location=location,
        #     radius=4,
        #     color="green",
        #     fill=True,
        #     opacity=opacity,
        #     popup=(time_conventer(x["trange"][0]), time_conventer(x["trange"][1]), round(x["trange"][1] - x["trange"][0],1)),
        # ).add_to(m)
    # actions
    for a in actions:
        if not a["act"] == WORK:
            continue
        t1, t2 = a["st"], a["et"]
        t = (t1 + t2) / 2
        sxy = find_traj_at_t(traj, t1)
        exy = find_traj_at_t(traj, t2)
        axy = find_traj_at_t(traj, t)
        locs = [wgs2gcj(*projector(*xy, inverse=True))[::-1] for xy in [sxy, axy, exy]]
        rs = [2,3,2]
        cs = ["red", "black", "blue"]
        idxs = [str(x["idx"]) for x in a["actions_orig"]]
        txt = ",".join(idxs if len(idxs) < 3 else [idxs[0], idxs[-1]])
        popup = (txt, time_conventer(t1), time_conventer(t), time_conventer(t2), " ".join([x["act"] for x in a["actions_orig"]]))
        for loc, r, c in zip(locs, rs, cs):
            folium.CircleMarker(
                location=loc,
                radius=r,
                color=c,
                fill=True,
                opacity=0.7,
                popup=popup,
            ).add_to(m)
        folium.PolyLine(
            locations=locs,
            color="black",
            weight=2,
            opacity=0.7,
            popup=popup,
        ).add_to(m)
        lats, lons = zip(*locs)
        folium.Marker(
            [np.mean(lats)-0.00003, np.mean(lons)],
            icon=folium.DivIcon(
                icon_size=(40,10),
                icon_anchor=(20,5),
                html=f'<div style="font-size: 10pt" text-align="center">{txt}</div>',
                )
            ).add_to(m)
    # orders
    for oid in target_oids:
        o = oid2odr[oid]
        xy_cs = [(o["xy"], "black")]
        if o["building_id"] != -1:
            xy_cs.append((buildings[o["building_id"]]["gate_xy"], "orange"))
        if o["address_xy"]:
            xy_cs.append((o["address_xy"], "blue"))
        for x in o["loc_scores"]:
            xy_cs.append((x[0], "green"))
        print(o["loc_scores"])
        exit()
        
        loc_cs = [(xy2loc(xy), c) for xy, c in xy_cs][::-1]
        axy = np.mean(np.array([x[0] for x in xy_cs]), axis=0)
        aloc = xy2loc(axy)
        popup = (o["type"], o["id"], time_conventer(o["finish_time"]))
        folium.CircleMarker(
            location=aloc,
            radius=5,
            color="red",
            fill=True,
            opacity=0.7,
            popup=popup,
        ).add_to(m)
        for loc, c in loc_cs:
            folium.CircleMarker(
                location=loc,
                radius=4,
                color=c,
                fill=True,
                opacity=0.6,
                popup=popup,
            ).add_to(m)
            folium.PolyLine(
                locations=[loc, aloc],
                color=c,
                weight=2,
                opacity=0.8,
                popup=popup,
            ).add_to(m)
        lat, lon = aloc
        folium.Marker(
            [lat-0.00003, lon],
            icon=folium.DivIcon(
                icon_size=(40,10),
                icon_anchor=(20,5),
                html=f'<div style="font-size: 15pt" text-align="center">{o["id"]}</div>',
                )
            ).add_to(m)
    m.save(f"0figure_label_mxl/spatial_{cid2name[wave['cid']]}_{wave['wave_idx']}.html")
    if True:
        return

    # 时间图
    plt.cla()
    plt.figure(figsize=(60, 2))
    plt.xlim((int(min(st, orders[0]["finish_time"]) / 3600), ceil(max(et, orders[-1]["finish_time"]) / 3600)))
    plt.ylim((0, 2))
    plt.gca().xaxis.set_major_locator(MultipleLocator(0.1))
    font = FontProperties(fname=r"msyh.ttc")
    color_map = {
        WORK: "red",
        REST: "green",
        OTHER: "blue",
        IGNORE: "gray"
    }
    # 画actions
    for a in actions:
        color = color_map[a["act"]]
        idxs = [x["idx"] for x in a["actions_orig"]]
        act_msg = [x["act"] + " " + x["smsg"] for x in a["actions_orig"]]
        t1, t2 = a["st"], a["et"]
        plt.hlines(y=1, xmin=t1 / 3600, xmax=t2 / 3600, color=color)
        x = (t1 + t2) / 7200
        y = 0.9
        for idx in idxs:
            plt.text(x=x, y=y, s=str(idx), color="black", ha="center", size=2)
            y -= 0.05
        for string in act_msg:
            plt.text(x=x, y=y, s=string, color="black", ha="center", size=2, fontproperties=font)
            y -= 0.1
    # 画orders
    t2odrs = defaultdict(list)
    for o in orders:
        t2odrs[int(o["finish_time"])].append(o)
    t_odrs = list(t2odrs.items())
    groups = []
    group = [t_odrs[0]]  # 避免文字重叠
    for t, odrs in t_odrs[1:]:
        if t <= group[-1][0] + 20:
            group.append((t, odrs))
        else:
            groups.append(group)
            group = [(t, odrs)]
    groups.append(group)
    for group in groups:
        y = 1.1
        for t, odrs in group:
            x, xmin, xmax = t / 3600, (t - 5) / 3600, (t + 5) / 3600
            for o in odrs:
                plt.hlines(y=y, xmin=xmin, xmax=xmax, color="red")
                y += 0.02
                plt.text(x=x, y=y, s=str(o["id"]), color="black", ha="center", size=1.5)
                y += 0.05

    plt.savefig(f"0figure_label_mxl/time_{cid2name[wave['cid']]}_{wave['wave_idx']}.pdf")


if __name__ == "__main__":
    train_data, test_data = pickle.load(open("0data_mxl/train_test_data.pkl", "rb"))

    # # step 0: 画轨迹和驻留点
    # # cid = 21777999  # 李文凯 电梯
    # # cid = 22626330  # 王茂林 楼梯 只有6波训练数据
    # # cid = 21173943  # 王云飞 写字楼 工业园 摆摊
    # cid = 22602631  # 肖明江 电梯楼 1101第2波轨迹缺失大
    # test_waves = test_data[cid]
    # for w in test_waves:
    #     folium_wave_odr_stay(w)
    # exit()

    # step 1: 标注粗粒度真值, 跑0match_baseline.py

    # step 2: 标注细粒度真值, 画感兴趣订单
    cid2label_actions = pickle.load(open("0ground_truth_mxl/gt_label_actions.pkl", "rb"))
    cid = 21777999  # 李文凯 电梯
    # cid = 22626330  # 王茂林 楼梯 只有6波训练数据
    # cid = 21173943  # 王云飞 写字楼 工业园 摆摊
    # cid = 22602631  # 肖明江 电梯楼 1101第2波轨迹缺失大
    for w in test_data[cid][1:2]:
        label_action_with_order(w, cid2label_actions[cid], target_oids=[2122])
        exit()
