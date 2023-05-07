import json
import pickle
import random
from collections import Counter, defaultdict
from copy import deepcopy
from math import ceil
from pprint import pprint

import numpy as np
from eviltransform import gcj2wgs
from networkx import shortest_path
from shapely.geometry import LineString, Point, Polygon

from constants_mw import *
from utils import pprint_actions

random.seed(233)

# 统计订单丢失的原因
throw_odr_match2cut = 0  # 订单时间在原始轨迹时间范围外
throw_odr_overtime = 0   # 无关联轨迹点订单, 按物理规则模拟送单, 模拟用时超过真实约束
throw_odr_iot_overlap = 0 # 有关联轨迹点的订单, 按关联轨迹点推算iot, 但多个单间iot有交叠且无法合理处理
throw_odr_iot_short = 0  # iot时间间隔过短
throw_odr_cannot_insert = 0  # 无关联轨迹点的订单无法插入有关联轨迹点的订单的iot之外的时间


def find_nearest_node(xy):
    """
    找给定坐标点的最近路网节点(非楼门非快递站)
    """
    x, y = xy
    tmp = [
        [nid, (nx-x)**2 + (ny-y)**2] 
        for nid, (nx, ny) in intersections
    ]
    nid, dis2 = min(tmp, key=lambda x:x[-1])
    return nid, dis2**0.5


def get_travel_path_t(start_nid=None, start_xy=None, end_nid=None, end_xy=None, v=V_WALK):
    """
    找od间的最短路径, od可以给node_id, 也可以给坐标
    """    
    if start_nid:  # 起点是node
        onid, gps_pre, xy_pre, dis_pre = start_nid, [], [], 0
    else:          # 起点是一个坐标
        onid, dis_pre = find_nearest_node(start_xy)
        gps_pre, xy_pre = [projector(*start_xy, inverse=True)], [start_xy]
    if end_nid:    # 终点是node
        dnid, gps_post, xy_post, dis_post = end_nid, [], [], 0
    else:          # 终点是一个坐标
        dnid, dis_post = find_nearest_node(end_xy)
        gps_post, xy_post = [projector(*end_xy, inverse=True)], [end_xy]

    if onid != dnid:
        path_nodes = shortest_path(G, onid, dnid, "length")
        path_edges = [G.edges[u, v] for u, v in zip(path_nodes, path_nodes[1:])]
        path_gps = path_edges[0]["gps"] + sum(
            [x["gps"][1:] for x in path_edges[1:]], []
        )
        path_xy = path_edges[0]["xy"] + sum(
            [x["xy"][1:] for x in path_edges[1:]], []
        )
        length = sum(edge["length"] for edge in path_edges)
        use_time = length / v
    else:
        path_gps, path_xy, use_time = [], [], 0.0

    path_gps = gps_pre + path_gps + gps_post
    path_xy = xy_pre + path_xy + xy_post
    use_time += (dis_pre + dis_post) / v

    return path_gps, path_xy, use_time


def filter_traj_points(traj_points, related_rids, dis_gate=200):
    """
    过滤掉距离所有订单所在楼所在路区距离都大于阈值的轨迹点
    """
    pts_filtered = []
    idx_filtered = []
    polys = [regions[rid]["poly"] for rid in related_rids]
    for i, p in enumerate(traj_points):
        point = Point(p[:2])
        for poly in polys:
            if point.distance(poly) < dis_gate:
                pts_filtered.append(p)
                idx_filtered.append(i)
                break
    return pts_filtered, idx_filtered


def cut_traj(points_filtered, idx_filtered, points_orig, tm_gap_gate=600):
    """
    按照时间间隔, 将轨迹切分成 由时间接近的过滤后的轨迹点构成的多段cut
    并记录时间不在cuts范围里的原始轨迹点(即第一段前head, 最后一段后tail, 每两段间inter)
    方便检查小哥在快递站的行为, 以及处理小哥并没有出路区(并没有过滤掉轨迹点)只是因为轨迹点采样间隔本来就很大而被切段的情况
    输出的cuts在轨迹点上是首尾相接的
    """
    cuts = []
    if idx_filtered[0] > 0:
        cuts.append({
            "type": "head",  # head型的轨迹段, 由若干个(>0)被过滤掉的原始轨迹点, 以及末尾一个没被过滤掉的轨迹点构成
            "points": points_orig[:idx_filtered[0]+1]
        })
    one_cut = [points_filtered[0]]
    tm_last = points_filtered[0][-1]
    idx_last = idx_filtered[0]
    for point, idx in zip(points_filtered[1:], idx_filtered[1:]):
        tm = point[-1]
        if tm - tm_last > tm_gap_gate:
            cuts.append({
                "type": "cut",  # cut型的轨迹段, 均由没被过滤掉的轨迹点构成
                "points": one_cut
            })
            # if idx + 1 - idx_last > 2:
            if True:  # TODO:
                cuts.append({
                    "type": "inter", # inter型的轨迹段, 由首尾两个没被过滤掉的轨迹点及中间若干个(>=0)被过滤掉的原始轨迹点构成
                    "points": points_orig[idx_last:idx+1]
                })
            one_cut = [point]
        else:
            one_cut.append(point)
        tm_last = tm
        idx_last = idx
    cuts.append({
        "type": "cut", 
        "points": one_cut
    })
    if idx_filtered[-1] < len(points_orig) - 1:
        cuts.append({
            "type": "tail",  # tail型的轨迹段, 由开头一个没被过滤掉的轨迹点, 以及若干个(>0)被过滤掉的原始轨迹点构成
            "points": points_orig[idx_filtered[-1]:]
        })
    
    # 将相邻的"cut"型轨迹段合并, 避免只是单纯地因为中间10min没有任何轨迹点而被断开
    cuts.sort(key=lambda x:x["points"][0][-1])
    cuts_new = []
    one_merge = []
    for cut in cuts:
        if cut["type"] != "cut":
            if len(one_merge) == 1:
                cuts_new.append(one_merge[0])
                one_merge = []
            elif len(one_merge) > 1:
                cuts_new.append({
                    "type": "cut",
                    "points": one_merge[0]["points"] + sum([x["points"][1:] for x in one_merge[1:]], [])
                })
                one_merge = []
            cuts_new.append(cut)
        else:
            one_merge.append(cut)
    if len(one_merge) == 1:
        cuts_new.append(one_merge[0])
    elif len(one_merge) > 1:
        cuts_new.append({
            "type": "cut",
            "points": one_merge[0]["points"] + sum([x["points"][1:] for x in one_merge[1:]], [])
        })

    return cuts_new


def match_order_to_cut(orders, cuts, tm_gap_gate=60):
    """将订单分配到轨迹段"""
    global throw_odr_match2cut
    match_results = []

    # step1 优先考虑将订单分配给正常的由过滤后的轨迹点构成的段(type=cut)
    t_ranges = [
        (i, (cut["points"][0][-1], cut["points"][-1][-1])) 
        for i, cut in enumerate(cuts)
        if cut["type"] == "cut" and len(cut["points"]) > 1
    ]
    # 订单完成时间在某cutx型轨迹段的时间范围内的属于这段
    undecided = []
    for i, order in enumerate(orders):  
        t = order["finish_time"]
        for j, (ts, te) in t_ranges:
            if ts <= t <= te:
                match_results.append((i, j))
                break
        else:
            undecided.append(i)
    # 还剩下的订单，若其完成时间离某段轨迹的时间范围<1min，分给这段轨迹
    still_undecided = []
    for i in undecided:  
        t = orders[i]["finish_time"]
        tmp = [(j, t, ts, te, min(abs(ts-t), abs(te-t))) for j, (ts, te) in t_ranges]
        tmp = [x for x in tmp if x[-1] < tm_gap_gate]
        if tmp:
            j, t, ts, te, _ = min(tmp, key=lambda x:x[-1])
            # 修改订单完成时间
            if abs(t-ts) < abs(t-te):
                orders[i]["finish_time"] = ts
            else:
                orders[i]["finish_time"] = te
            match_results.append((i, j))
        else:
            still_undecided.append(i)

    # step2 仍剩下的订单直接按时间塞到head, tail, inter类型的段
    t_ranges = [
        (i, (cut["points"][0][-1], cut["points"][-1][-1])) 
        for i, cut in enumerate(cuts)
        if cut["type"] != "cut"
    ]
    for i in still_undecided:
        t = orders[i]["finish_time"]
        for j, (ts, te) in t_ranges:
            if ts <= t <= te:
                match_results.append((i, j))
                break

    # 至此, 只要订单finish_time不是在整条原始轨迹的时间范围之外, 都能匹配到某段轨迹中

    # 记录每段轨迹的所有订单
    for cut in cuts:
        cut["orders"] = []
    for i, j in match_results:
        cuts[j]["orders"].append(orders[i])
    assert sum(len(c["orders"]) for c in cuts) == len(match_results)

    # 统计丢弃的订单数
    throw_odr_match2cut += len(orders) - len(match_results)

    return cuts


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


def find_near_station_times(pts, nearstation_gate):
    """
    返回在快递站附近的轨迹点的时间
    若与多个站有近的轨迹点, 返回最多的那个站
    """
    if not pts:
        return None, None
    sid2ts = defaultdict(list)
    gate2 = nearstation_gate ** 2
    for x, y, t in pts:
        for s in stations.values():
            x2, y2 = s["xy"]
            if (x - x2) ** 2 + (y - y2) ** 2 < gate2:
                sid2ts[s["id"]].append(t)
    if sid2ts:
        sid, ts = max(list(sid2ts.items()), key=lambda x:x[1])
        return sid, ts
    else:
        return None, None


def gen_instation_actions(sid, start_time, end_time):
    """
    生成在营业部里的行为
    在生成actions的过程中, 先直接生成一个粗粒度的行为填充整段在快递站里的时间, 生成完之后, 后处理时再细化
    """
    return [{
        "type": ACTION_ARRANGE,
        "start_time": start_time,
        "end_time": end_time,
        "station_id": sid,
        "gps": stations[sid]["gps"],
        "xy": stations[sid]["xy"],
        "target_orders": [],
    }]


def gen_detail_instation_actions(sid, start_time, end_time, receive_times):
    """
    细化在营业部里的行为
    在开早会的时间段内开早会, 在收货时间段内理货拣货, 其它时间休息
    """
    receive_gap_gate = 600 # 若收货时间间隔超过此阈值, 认为在开早会或休息, 否则都算在理货拣货
    single_receive_t = 10  # 若有孤立的收货时间(距离其它收货时间都超过600s), 则[t-10, t]作为理货拣货的时间段
    if receive_times:
        receive_times = list(set(receive_times))
        receive_times.sort()
        assert start_time <= receive_times[0] <= receive_times[-1] <= end_time
        receive_ranges = [[receive_times[0], receive_times[0]]]
        for t in receive_times[1:]:
            if t - receive_ranges[-1][-1] < receive_gap_gate:
                receive_ranges[-1][-1] = t
            else:
                receive_ranges.append([t, t])
        for x in receive_ranges:
            if x[0] == x[1]:
                x[0] = max(start_time, x[0] - single_receive_t)
        receive_ranges = [x for x in receive_ranges if x[0] < x[1]]
        receive_ranges_new = []
        for ts, te in receive_ranges:
            if T_MEETING_START <= ts and te <= T_MEETING_END:  # 收货时间被早会时间覆盖, 则丢弃
                pass
            elif ts < T_MEETING_START and T_MEETING_END < te:  # 早会时间被收货时间覆盖, 则中间挖掉, 拆成两段
                receive_ranges_new.append([ts, T_MEETING_START])
                receive_ranges_new.append([T_MEETING_END, te])
            elif ts < T_MEETING_START and T_MEETING_START < te <= T_MEETING_END:  # 后半段挖掉
                receive_ranges_new.append([ts, T_MEETING_START])
            elif T_MEETING_START <= ts < T_MEETING_END and T_MEETING_END < te:  # 前半段挖掉
                receive_ranges_new.append([T_MEETING_END, te])
            else:
                receive_ranges_new.append([ts, te])
        action_ranges = [(ACTION_ARRANGE, x) for x in receive_ranges_new]
    else:
        action_ranges = []
    if end_time > T_MEETING_START and start_time < T_MEETING_END:
        action_ranges.append((ACTION_MEETING, [max(T_MEETING_START, start_time), min(T_MEETING_END, end_time)]))
    action_ranges.sort(key=lambda x:x[1][0])
    
    actions = []
    last_te = start_time
    for atp, (ts, te) in action_ranges:
        assert ts >= last_te and te > ts
        if ts > last_te:
            actions.append({
                "type": ACTION_REST,
                "start_time": last_te,
                "end_time": ts
            })
        actions.append({
            "type": atp,
            "start_time": ts,
            "end_time": te
        })
        last_te = te
    if last_te < end_time:
        actions.append({
            "type": ACTION_REST,
            "start_time": last_te,
            "end_time": end_time
        })
    for a in actions:
        a["station_id"] = sid
        a["gps"] = stations[sid]["gps"]
        a["xy"] = stations[sid]["xy"]
        a["target_orders"] = []
    return actions


def gen_inbd_actions(orders, start_time, end_time=None):
    """
    给定在某栋楼一次性送的单, 及开始时间, 假定小哥已经到达楼门, 按简单物理逻辑生成上下楼/上门/揽派件等行为
    若还给定end_time, 则按比例放缩每个action的时间, 使得满足end_time约束
    """
    actions = []

    bid = orders[0]["building_id"]
    is_elevator = buildings[bid]["is_elevator"]
    time_axis = start_time

    unit2odrs = defaultdict(list)
    for odr in orders:
        unit2odrs[odr["unit"]].append(odr)
    for unit, odrs in unit2odrs.items():
        floor2odrs = defaultdict(list)
        for odr in odrs:
            floor2odrs[odr["floor"]].append(odr)
        floor_odrs = sorted(list(floor2odrs.items()), key=lambda x:x[0])
        last_floor = 1
        already_wait = False  # 是否等过第一次电梯

        for floor, odrs in floor_odrs:
            # 上楼
            up_num = floor - last_floor
            if up_num > 0:
                if is_elevator:
                    t_wait = T_ELEVATOR_WAIT2 if already_wait else T_ELEVATOR_WAIT1
                    use_time = t_wait + T_ELEVATOR_FLOOR * up_num
                    action_type = ACTION_ELEVATOR
                    already_wait = True
                else:
                    use_time = T_STAIR_UP * up_num
                    action_type = ACTION_UPSTAIR
                actions.append({
                    "type": action_type,
                    "start_time": time_axis,
                    "end_time": time_axis + use_time,
                    "num": up_num,
                    "building": bid,
                    "unit": unit,
                    "from": last_floor,
                    "to": floor,
                    "target_orders": odrs,
                })
                time_axis += use_time
                last_floor = floor
            # 送单
            for odr in odrs:
                actions.append({
                    "type": ACTION_TODOOR,
                    "start_time": time_axis,
                    "end_time": time_axis + T_TODOOR,
                    "building": bid,
                    "unit": unit,
                    "floor": floor,
                    "target_orders": [odr],
                })
                t_order = ORDER2T[odr["type"]]
                actions.append({
                    "type": ORDER2ACTION[odr["type"]],
                    "start_time": time_axis + T_TODOOR,
                    "end_time": time_axis + T_TODOOR + t_order,
                    "building": bid,
                    "unit": unit,
                    "floor": floor,
                    "target_orders": [odr],
                })  
                time_axis += T_TODOOR + t_order
        # 下楼
        down_num = last_floor - 1
        if down_num > 0:
            if is_elevator:
                use_time = T_ELEVATOR_WAIT2 + T_ELEVATOR_FLOOR * down_num
                action_type = ACTION_ELEVATOR
            else:
                use_time = T_STAIR_DOWN * down_num
                action_type = ACTION_DOWNSTAIR
            actions.append({
                "type": action_type,
                "start_time": time_axis,
                "end_time": time_axis + use_time,
                "num": -down_num,
                "building": bid,
                "unit": unit,
                "from": last_floor,
                "to": 1,
                "target_orders": [],
            })
            time_axis += use_time
            last_floor = 1

    if end_time:  # 考虑满足end_time约束
        assert end_time > start_time
        if end_time > actions[-1]["end_time"]:  # 若模拟时间比约束用时更短, 在前面插入一段卸货
            t_rest = end_time - actions[-1]["end_time"]
            for a in actions:  # 原本的action往后挪
                a["start_time"] += t_rest
                a["end_time"] += t_rest
            action_rest = {
                "type": ACTION_DISCHARGE,
                "start_time": start_time,
                "end_time": start_time + t_rest,
                "building": bid,
                "target_orders": orders,
            }
            actions = [action_rest] + actions
        elif end_time < actions[-1]["end_time"]:  # 若模拟用时比约束用时长, 按比例缩短各个动作的用时
            p = (end_time - start_time) / (actions[-1]["end_time"] - start_time)
            last_end_time = start_time
            for action in actions:
                action["end_time"] = last_end_time + (action["end_time"] - action["start_time"]) * p
                action["start_time"] = last_end_time
                last_end_time = action["end_time"]
        
    assert len(orders) == len([x for x in actions if x["type"] in {ACTION_CPICK, ACTION_BPICK, ACTION_DELIVER}])

    return actions


def gen_bds_actions(bid_odrs, start_time):
    """
    [(bid, odrs), ...]代表小哥送单序列, 假设小哥初始已经在第一栋楼门, 按简单物理逻辑生成其行为
    """
    actions = []
    time_axis = start_time
    last_bid = bid_odrs[0][0]
    
    for i, (bid, odrs) in enumerate(bid_odrs):
        # 在楼间移动
        if i > 0:
            if bid != last_bid:
                path_gps, path_xy, use_time = get_travel_path_t(
                    start_nid=buildings[last_bid]["gate_id"], 
                    end_nid=buildings[bid]["gate_id"], 
                    v=V_WALK)
                actions.append(
                    {
                        "type": ACTION_WALK,
                        "start_time": time_axis,
                        "end_time": time_axis + use_time,
                        "start_building": last_bid,
                        "end_building": bid,
                        "gps": path_gps,
                        "xy": path_xy,
                        "target_orders": odrs,
                    }
                )
                time_axis += use_time
        # 在楼内的行为
        actions += gen_inbd_actions(
            orders=odrs, 
            start_time=time_axis
        )
        time_axis = actions[-1]["end_time"]
        last_bid = bid

    return actions


def gen_no_odr_actions(start_time, end_time=None, start_nid=None, start_xy=None, end_nid=None, end_xy=None):
    """
    对于没有订单的一段轨迹, 按简单物理逻辑生成其移动行为
    并考虑满足end_time时间约束: 若移动完成时小于end_time, 补一段休息行为, 否则直接认为移动在end_time结束
    """
    if not end_time is None:
        assert start_time < end_time
    actions = []

    action_type, v = ACTION_WALK, V_WALK
    if start_nid:  # 起点是node
        if start_nid in station_ids:
            assert end_nid not in station_ids
            action_type, v = ACTION_FROMSTATION, V_CAR
    if end_nid:  # 终点是node
        if end_nid in station_ids:
            action_type, v = ACTION_TOSTATION, V_CAR
    
    # 从起点直接去终点
    path_gps, path_xy, use_time = get_travel_path_t(
        start_nid=start_nid,
        start_xy=start_xy,
        end_nid=end_nid,
        end_xy=end_xy,
        v=v)
    actions.append({
        "type": action_type,
        "start_time": start_time,
        "end_time": start_time + use_time,
        "gps": path_gps,
        "xy": path_xy,
        "target_orders": [],
    })
    if start_nid:
        actions[-1]["start_node"] = start_nid
    else:
        actions[-1]["start_xy"] = start_xy
    if end_nid:
        actions[-1]["end_node"] = end_nid
    else:
        actions[-1]["end_xy"] = end_xy

    # 考虑end_time时间约束
    if not end_time is None:
        if actions[-1]["end_time"] < end_time:
            actions.append({
                "type": ACTION_REST,
                "start_time": actions[-1]["end_time"],
                "end_time": end_time,
                "target_orders": [],
            })
            if end_nid and end_nid in station_ids:
                actions[-1]["station_id"] = end_nid
        else:
            actions[-1]["end_time"] = end_time

    return actions


def gen_has_odr_actions(orders, start_time, end_time=None, start_nid=None, start_xy=None, end_nid=None, end_xy=None, follow_order_seq=False):
    """
    对于有订单的一段轨迹, 按简单物理逻辑生成其行为
    若发现生成的行为超出end_time时间约束, 则放弃其中订单最少的楼的所有单, 重新尝试, 直至满足时间约束或所有订单均被放弃
    """
    global throw_odr_overtime
    if not end_time is None:
        assert start_time < end_time
    actions = []

    if follow_order_seq:
        # 按订单顺序访问楼
        bid_odrs = []
        for odr in orders:
            bid = odr["building_id"]
            if bid_odrs and bid_odrs[-1][0] == bid:
                bid_odrs[-1][1].append(odr)
            else:
                bid_odrs.append([bid, [odr]])
    else:
        # 按楼距起点的距离决定小哥去楼的顺序(不考虑订单完成时间, 去每栋楼时送完该楼所有订单)
        bid2odrs = defaultdict(list)
        for odr in orders:
            bid2odrs[odr["building_id"]].append(odr)
        nid = start_nid if start_nid else find_nearest_node(start_xy)[0]
        start_p = Point(G.nodes[nid]["xy"])
        bid_odrs_diss = [
            (bid, odrs, start_p.distance(buildings[bid]["point"]))
            for bid, odrs in bid2odrs.items()
        ]
        bid_odrs_diss.sort(key=lambda x:x[-1])
        bid_odrs = [x[:2] for x in bid_odrs_diss]
    
    # 从起点到第一栋楼
    first_bid = bid_odrs[0][0]
    if start_nid:  # 起点是node
        if start_nid in station_ids:
            action_type, v = ACTION_FROMSTATION, V_CAR
        else:
            action_type, v = ACTION_WALK, V_WALK
        path_gps, path_xy, use_time = get_travel_path_t(
            start_nid=start_nid,
            end_nid=buildings[first_bid]["gate_id"],
            v=v)
        actions.append({
            "type": action_type,
            "start_time": start_time,
            "end_time": start_time + use_time,
            "start_node": start_nid,
            "end_building": first_bid,
            "gps": path_gps,
            "xy": path_xy,
            "target_orders": bid_odrs[0][1],
        })
    else:  # 起点是一个坐标
        path_gps, path_xy, use_time = get_travel_path_t(
            start_xy=start_xy,
            end_nid=buildings[first_bid]["gate_id"],
            v=V_WALK)
        actions.append({
            "type": ACTION_WALK,
            "start_time": start_time,
            "end_time": start_time + use_time,
            "start_xy": start_xy,
            "end_building": first_bid,
            "gps": path_gps,
            "xy": path_xy,
            "target_orders": bid_odrs[0][1],
        })
    time_axis = start_time + use_time

    # 在楼中送单及楼间移动的action
    actions += gen_bds_actions(
        bid_odrs=bid_odrs,
        start_time=time_axis
    )

    # 从最后一栋楼到终点
    last_bid = bid_odrs[-1][0]
    last_action_end_time = actions[-1]["end_time"]
    if end_nid:  # 终点是node
        if end_nid in station_ids:
            action_type, v = ACTION_TOSTATION, V_CAR
        else:
            action_type, v = ACTION_WALK, V_WALK
        path_gps, path_xy, use_time = get_travel_path_t(
            start_nid=buildings[last_bid]["gate_id"], 
            end_nid=end_nid, 
            v=v)
        actions.append({
            "type": action_type,
            "start_time": last_action_end_time,
            "end_time": last_action_end_time + use_time,
            "start_building": last_bid,
            "end_node": end_nid,
            "gps": path_gps,
            "xy": path_xy,
            "target_orders": [],
        })
    else:  # 终点是一个坐标
        path_gps, path_xy, use_time = get_travel_path_t(
            start_nid=buildings[last_bid]["gate_id"],
            end_xy=end_xy,
            v=V_WALK)
        actions.append({
            "type": ACTION_WALK,
            "start_time": last_action_end_time,
            "end_time": last_action_end_time + use_time,
            "start_building": last_bid,
            "end_xy": end_xy,
            "gps": path_gps,
            "xy": path_xy,
            "target_orders": [],
        })
    
    if end_time is None:
        return actions

    # 考虑行为的结束时间应为end_time的约束
    last_action_end_time = actions[-1]["end_time"]
    if last_action_end_time < end_time:
        actions.append({
            "type": ACTION_REST,
            "start_time": last_action_end_time,
            "end_time": end_time,
            "target_orders": [],
        })
        if end_nid and end_nid in station_ids:
            actions[-1]["station_id"] = end_nid
    elif last_action_end_time > end_time:
        if actions[-1]["start_time"] < end_time:
            actions[-1]["end_time"] = end_time
        else:
            p = (end_time - start_time) / (last_action_end_time - start_time)
            if p > P_SHORT:  # 若超时不多, 按比例缩放每个动作的用时
                last_end_time = start_time
                for action in actions:
                    action["end_time"] = last_end_time + (action["end_time"] - action["start_time"]) * p
                    action["start_time"] = last_end_time
                    last_end_time = action["end_time"]
            else:        # 若超时过多, 放弃订单数最少的楼中的订单, 重新尝试
                bodrs = sorted(list(bid2odrs.values()), key=lambda x:len(x))
                throw_odr_overtime += len(bodrs[0])
                next_try_odrs = [odr for odrs in bodrs[1:] for odr in odrs]
                if not next_try_odrs:
                    return gen_no_odr_actions(
                        start_time=start_time, 
                        end_time=end_time, 
                        start_nid=start_nid, 
                        start_xy=start_xy, 
                        end_nid=end_nid, 
                        end_xy=end_xy
                    )
                else:
                    return gen_has_odr_actions(  # 递归
                        orders=next_try_odrs,
                        start_time=start_time, 
                        end_time=end_time, 
                        start_nid=start_nid, 
                        start_xy=start_xy, 
                        end_nid=end_nid, 
                        end_xy=end_xy
                    )

    return actions


def gen_odrs_actions(orders, start_time, end_time, start_nid=None, start_xy=None, end_nid=None, end_xy=None):
    """
    把上面两个函数包装一下
    """
    if orders:
        return gen_has_odr_actions(orders, start_time, end_time, start_nid, start_xy, end_nid, end_xy)
    else:
        return gen_no_odr_actions(start_time, end_time, start_nid, start_xy, end_nid, end_xy)


def gen_bds_iot_actions(odrs_iot):
    """
    [(odr_list, [ti, to]), ...]代表小哥送单序列, 假设小哥初始已经在第一栋楼门, 按ti_to约束生成其行为
    """
    actions = []
    last_bid = odrs_iot[0][0][0]["building_id"]
    last_to = None

    for i, (odrs, (ti, to)) in enumerate(odrs_iot):
        bid = odrs[0]["building_id"]
        # 在楼间移动
        if i > 0:  # i == 0时直接进楼
            if bid != last_bid:
                path_gps, path_xy, t_exp = get_travel_path_t(
                    start_nid=buildings[last_bid]["gate_id"], 
                    end_nid=buildings[bid]["gate_id"], 
                    v=V_WALK)
                if last_to + t_exp > ti:
                    walk_end_time = ti
                    discharge_end_time = None
                elif last_to + t_exp < ti:  # 若在楼间移动时间过长, 在后面插入一段卸货
                    walk_end_time = last_to + t_exp
                    discharge_end_time = ti
                actions.append(
                    {
                        "type": ACTION_WALK,
                        "start_time": last_to,
                        "end_time": walk_end_time,
                        "start_building": last_bid,
                        "end_building": bid,
                        "gps": path_gps,
                        "xy": path_xy,
                        "target_orders": odrs,
                    }
                )
                if discharge_end_time:
                    actions.append(
                        {
                            "type": ACTION_DISCHARGE,
                            "start_time": walk_end_time,
                            "end_time": discharge_end_time,
                            "building": bid,
                            "target_orders": odrs,
                        }
                    )
            elif ti > last_to:  # 下一楼是同一栋楼, 直接在原地卸货
                actions.append({
                    "type": ACTION_DISCHARGE,
                    "start_time": last_to,
                    "end_time": ti,
                    "building": bid,
                    "target_orders": odrs,
                })
        # 在楼内的行为
        actions += gen_inbd_actions(
            orders=odrs, 
            start_time=ti, 
            end_time=to
        )
        last_to = to
        last_bid = bid

    return actions


def gen_odrs_iot_actions(odrs_iot, start_time, end_time, start_xy, end_xy):
    """
    对于已知订单(一次进楼一次性送的所有单已经聚合)以及进出楼时间, 生成带时间约束的行为
    """
    assert start_time < end_time
    actions = []

    first_bid = odrs_iot[0][0][0]["building_id"]
    first_ti = odrs_iot[0][1][0]
    last_bid = odrs_iot[-1][0][0]["building_id"]
    last_to = odrs_iot[-1][1][1]

    # 从起点到第一栋楼
    path_gps, path_xy, t_walk = get_travel_path_t(
        start_xy=start_xy, 
        end_nid=buildings[first_bid]["gate_id"], 
        v=V_WALK)
    if first_ti - start_time < t_walk:
        t_walk = first_ti - start_time
        t_discharge = 0
    else:  # 时间有富余, 卸货
        t_discharge = first_ti - start_time - t_walk 
    actions.append({
        "type": ACTION_WALK,
        "start_time": start_time,
        "end_time": start_time + t_walk,
        "start_xy": start_xy,
        "end_building": first_bid,
        "gps": path_gps,
        "xy": path_xy,
        "target_orders": odrs_iot[0][0],
    })
    if t_discharge > 0:
        actions.append({
            "type": ACTION_DISCHARGE,
            "start_time": start_time + t_walk,
            "end_time": first_ti,
            "building": first_bid,
            "target_orders": odrs_iot[0][0],
        })

    # 在楼中送单及楼间移动的action
    actions += gen_bds_iot_actions(odrs_iot)

    # 从最后一栋楼到终点
    path_gps, path_xy, t_travel = get_travel_path_t(
        start_nid=buildings[last_bid]["gate_id"], 
        end_xy=end_xy, 
        v=V_WALK)
    if end_time - last_to < t_travel:
        t_travel = end_time - last_to
        t_rest = 0
    else:  # 时间有富余, 休息
        t_rest = end_time - last_to - t_travel 
    actions.append({
        "type": ACTION_WALK,
        "start_time": last_to,
        "end_time": last_to + t_travel,
        "start_building": last_bid,
        "end_xy": end_xy,
        "gps": path_gps,
        "xy": path_xy,
        "target_orders": [],
    })
    if t_rest > 0:
        actions.append({
            "type": ACTION_REST,
            "start_time": last_to + t_travel,
            "end_time": end_time,
            "target_orders": [],
        })

    return actions


def infer_iot(odrs, pts=None, min_max_t=None):
    """
    根据订单和关联轨迹点, 推测进出楼时间
    若没有关联轨迹点, 根据订单完成时间推测
    可以给一个iot最小值和最大值的限制min_max_t
    限制之后, 若导致在楼内的时间缩短到0.7倍以下, 认为失败
    """
    global throw_odr_iot_short
    actions = gen_inbd_actions(odrs, 0)
    t_exp_use = actions[-1]["end_time"]  # 预期的在楼内的用时
    if pts:  # 给定关联轨迹点, 计算加权平均的轨迹点时间, 前后半个预期用时即为推测的进出楼时间
        bid = odrs[0]["building_id"]
        total_t = 0
        total_w = 0
        for p in pts:
            t = p[-2]
            labels = p[-1]
            for l, w in labels:
                if l == bid:
                    total_t += t * max(w, 0.1)
                    total_w += max(w, 0.1)
                    break
            else:  # 由于向两边延拓时只有当连续2个label不含bid才停止, 故关联轨迹点中也可能含有label不含bid的
                total_t += t * 0.1
                total_w += 0.1
        t_ave = total_t / total_w
        if min_max_t:
            t_min, t_max = min_max_t
            ti, to = max(t_min, t_ave - t_exp_use / 2), min(t_max, t_ave + t_exp_use / 2)
            if to - ti > P_SHORT * t_exp_use:
                return ti, to
            else:
                throw_odr_iot_short += len(odrs)
                return None, None
        else:
            return t_ave - t_exp_use / 2, t_ave + t_exp_use / 2

    else:  # 没有关联轨迹点, 根据订单完成时间推算进出楼时间
        assert len(odrs) == 1
        tf = odrs[0]["finish_time"]
        for action in actions[::-1]:
            if action["type"] != ACTION_ELEVATOR and action["type"] != ACTION_DOWNSTAIR:  # 找到最后一个不是坐电梯或下楼的动作, 即应该是订单完成时
                t_exp_f = action["end_time"]
                break
        if min_max_t:
            t_min, t_max = min_max_t
            ti, to = max(t_min, tf - t_exp_f), min(t_max, tf - t_exp_f + t_exp_use)
            if to - ti > P_SHORT * t_exp_use:
                return ti, to
            else:
                throw_odr_iot_short += len(odrs)
                return None, None
        else:
            return tf - t_exp_f, tf - t_exp_f + t_exp_use


def merge_list_min_max(data, merge_gate=0):
    """
    对于[(list_last, [min_last, max_last]), (list, [min, max])...]型的数据, 
    当min <= last_max + merge_gate时, 合并为[(list_last + list, [min_last, max(max, max_last)]), ...]
    """
    if not data:
        return []
    data.sort(key=lambda x:x[1][0])
    data_merged = []
    one_merge = data[0]
    l_last, (mi_last, ma_last) = one_merge
    for l, (mi, ma) in data[1:]:
        if mi <= ma_last + merge_gate:
            one_merge = (l_last + l, [mi_last, max(ma, ma_last)])
        else:
            data_merged.append(one_merge)
            one_merge = (l, [mi, ma])
        l_last, (mi_last, ma_last) = one_merge
    data_merged.append(one_merge)

    return data_merged


def tackle_odrs_iot_overlap(odrs_iot):
    """
    处理[(odr_list, [ti, to]), ...]数据中ti, to发生交叠的情况
    """
    global throw_odr_iot_overlap
    if not odrs_iot:
        return []
    odrs_iot.sort(key=lambda x:x[1][0])
    oid2deltat_orig = {}  # 记录处理之前的iot
    for odrs, (ti, to) in odrs_iot:
        oid2deltat_orig[tuple(o["id"] for o in odrs)] = to - ti
    odrs_iot_refined = [odrs_iot[0]]
    ti_last, to_last = odrs_iot_refined[0][1]
    for odrs, (ti, to) in odrs_iot[1:]:
        overlap = to_last - ti + T_OVERLAP  # 至少间隔10s
        if overlap <= 0:
            odrs_iot_refined.append((odrs, [ti, to]))
            ti_last, to_last = ti, to
        else:
            a = to_last - ti_last
            b = to - ti
            if overlap < (a + b) / 2:  # to_last往前缩, ti往后移, 但不能导致在楼内的时间减少一半以上
                ta = a / (a + b) * overlap  # 两段iot按时长比例分配overlap
                tb = overlap - ta
                to_last -= ta
                odrs_iot_refined[-1][1][1] = to_last
                ti += tb
                odrs_iot_refined.append((odrs, [ti, to]))
                ti_last, to_last = ti, to
            else:  # 无法通过移动时间的方式处理交叠, 放弃二者中订单更少的那个
                if len(odrs) <= len(odrs_iot_refined[-1][0]):
                    throw_odr_iot_overlap += len(odrs)
                else:
                    throw_odr_iot_overlap += len(odrs_iot_refined[-1][0])
                    odrs_iot_refined[-1] = (odrs, [ti, to])
                    ti_last, to_last = ti, to
    # 虽然上面在循环里挪的时候有考虑到不能挪一半以上, 但保不准同一段被挪了多次, 因此再次过滤
    odrs_iot_refined_filtered = []
    for odrs, (ti, to) in odrs_iot_refined:
        if to - ti > 0.5 * oid2deltat_orig[tuple(o["id"] for o in odrs)]:
            odrs_iot_refined_filtered.append((odrs, [ti, to]))
        else:
            throw_odr_iot_overlap += len(odrs)
    return odrs_iot_refined_filtered


def gen_odrs_pts_actions(orders, pts, iod_data=None, sample_gap=10, outbd_gate=30, nearbd_gate=5, nearodr_gate=120, relateodr_gate=120):
    """
    生成cut型轨迹段的行为
    对于有订单, 且相应有在路区内的轨迹点的一段轨迹, 按尽量符合轨迹点时空信息和订单完成时间的方式, 生成行为
    而上面几个函数均按简单物理逻辑, 即默认的步行/上楼速度等生成, 并不考虑与订单完成时间相符
    sample_gap: 轨迹点升采样时间间隔
    outbd_gate: 与所有楼距离超过此阈值的轨迹点, 标记为在楼外
    nearbd_gate: 与楼距离小于此阈值的轨迹点, 标记为在这栋楼内
    nearodr_gate: 与楼距离在nearbd_gate和outbd_gate之间的, 若时间在楼内某订单完成时间的此阈值内, 标记为在这栋楼内, 否则标记为在楼外
    relateodr_gate: 找与订单相关联的轨迹点: 在该楼内, 且时间相差在此阈值内的轨迹点
    """
    cut_start_time, cut_end_time = pts[0][-1], pts[-1][-1]
    assert cut_end_time > cut_start_time
    if not orders:
        return gen_no_odr_actions(
            start_time=cut_start_time, 
            end_time=cut_end_time, 
            start_nid=None, 
            start_xy=pts[0][:2], 
            end_nid=None, 
            end_xy=pts[-1][:2]
        )

    def dis2weight(dis):
        """nearbd_gate < dis < outbd_gate, 0.4 < w < 1"""
        nonlocal nearbd_gate, outbd_gate
        a = outbd_gate - dis
        b = dis - nearbd_gate
        return (a + 0.4*b) / (a + b)

    def gap2weight(gap):
        """nearodr_gate/2 < gap < nearodr_gate, 0.4 < w <1"""
        nonlocal nearodr_gate
        if gap <= nearodr_gate / 2:
            return 1.0
        a = nearodr_gate - gap
        b = gap - nearodr_gate / 2
        return (a + 0.4*b) / (a + b)

    def get_odrs_relate_pts(pts, odrs, iod_data=None):
        """
        给每个订单找相关联轨迹点
        同一楼中相关联轨迹点有交叠的不同订单被合并, 视为一次进楼送多单
        并记录找不到相关联轨迹点的订单
        若有IOD数据, 使用IOD数据校准给点打label的过程
        """
        input_odrs_num = len(odrs)

        pts = upscale_traj(pts, sample_gap=sample_gap)  # 将轨迹点升采样到时间间隔<10s
        bid2odrs = defaultdict(list)
        for odr in odrs:
            bid2odrs[odr["building_id"]].append(odr)
        
        # 根据轨迹点与所有楼（只看这些订单对应的楼）的距离, 将轨迹点打上标签
        pts_labeled = []
        bds = [buildings[i] for i in bid2odrs.keys()]  # 与所有订单对应的所有楼
        for x, y, t in pts:
            p = Point((x, y))
            dis_bid_s = []
            for bd in bds:
                dis = p.distance(bd["poly"])
                if dis == 0:
                    pts_labeled.append([x, y, t, [(bd["id"], 1)]])  # 在楼内的, 标签为building_id, 标签还有一个置信度权重
                    break
                dis_bid_s.append((dis, bd["id"]))
            else:
                dis, bid = min(dis_bid_s, key=lambda x:x[0])
                if dis > outbd_gate:  # 距离所有楼的距离都>outbd_gate的, 标签为-1, 表示在楼外
                    pts_labeled.append([x, y, t, [(-1, 1)]])  # label为-1, 置信度为1
                elif dis < nearbd_gate:  # 与楼的最小距离<nearbd_gate的, 标签为building_id
                    pts_labeled.append([x, y, t, [(bid, 1)]])  # label为bid, 置信度为1
                else:  # 找与之距离<outbd_gate的所有楼, 若楼内有订单时间在轨迹点时间附近, 标签为building_id, 否则为-1
                    near_dis_bid_s = [x for x in dis_bid_s if x[0] < outbd_gate]
                    tmp = []
                    for dis, bid in near_dis_bid_s:
                        gap = min(abs(t - odr["finish_time"]) for odr in bid2odrs[bid])
                        if gap < nearodr_gate:
                            tmp.append((dis, gap, bid))
                    if tmp:
                        label = [(bid, dis2weight(dis)*gap2weight(gap)) for dis, gap, bid in tmp]
                        pts_labeled.append([x, y, t, label])  # 可能会与多个楼有置信度
                    else:
                        pts_labeled.append([x, y, t, [(-1, 1)]])
        
        def get_io_status(t):
            """
            判断t时刻小哥的状态 'in'|'out'|'unknown'
            取t正负5s时间范围内的所有iod_data, 
            若其中'in'多于'out', 返回'in'; 
            若'out'多于'in', 返回'out';
            其它情况, 例如没有数据, 或只有semi/unknown, 或'in'和'out'一样多, 返回'unknown'
            """
            nonlocal iod_data
            if 'in多于out':
                return 'in'
            elif 'out多于in':
                return 'out'
            else:
                return 'unknown'

        # 根据IOD数据校正轨迹点的label
        if iod_data is not None:
            pts_labeled_new = []
            outbd_gate2 = 2 * outbd_gate
            for x, y, t, label in pts_labeled:
                ios = get_io_status(t)
                if ios == "in":  # iod认为在楼里
                    if label[0][0] == -1:  # 原本认为在楼外
                        p = Point((x, y))
                        dis_bid_s = [(p.distance(bd["poly"]), bd["id"]) for bd in bds]
                        label_new = [
                            (bid, (outbd_gate2 - dis) / outbd_gate2)
                            for dis, bid in dis_bid_s
                            if dis < outbd_gate2  # 距离阈值放宽到2倍, 判断在哪栋楼
                        ]
                        if not label_new:
                            pts_labeled_new.append((x, y, t, [(-1, 1)]))
                        else:
                            pts_labeled_new.append((x, y, t, label_new))
                    else:
                        label_new = [(bid, min(w * 1.5, 1)) for bid, w in label]  # 已经认为在楼内, 增大权重
                        pts_labeled_new.append((x, y, t, label_new))
                elif ios == "out":  # iod认为在楼外
                    if not label[0][0] == -1:  # 原本认为在楼里, 权重降低
                        label_new = [(bid, w * 0.5) for bid, w in label]
                        pts_labeled_new.append((x, y, t, label_new))
                else:  # iod无法判断, 保持不变
                    pts_labeled_new.append((x, y, t, label))
            pts = pts_labeled_new
        else:
            pts = pts_labeled

        # 遍历楼, 找与订单相关联的轨迹点: 在订单完成时间附近, 且标签为订单所在楼id的轨迹点
        all_odrs_relate_pts_merged = []
        undecided_odrs = []  # 找不到相关联轨迹点的订单
        label2pts = defaultdict(list)
        for i, p in enumerate(pts):
            for l, w in p[-1]:
                label2pts[l].append((i, p))
        for bid, odrs in bid2odrs.items():
            # 找与每个订单相关联的轨迹点
            odrs_relate_pts = []  # 某楼里各订单及相关联轨迹点
            pts_inbd = label2pts[bid]
            for odr in odrs:
                t = odr["finish_time"]
                pts_relateodr = [(i, p) for i, p in pts_inbd if abs(t-p[-2]) < relateodr_gate]  # 找订单完成时间附近的所有标签为楼id的轨迹点
                if pts_relateodr:
                    # 找到的轨迹点进一步在时间轴上向两边拓展：即把与这些轨迹点时间相邻的、标签为楼id的轨迹点也收进来
                    i_min = min(pts_relateodr, key=lambda x:x[0])[0]
                    i_max = max(pts_relateodr, key=lambda x:x[0])[0]
                    while i_min > 0:
                        if bid in {x[0] for x in pts[i_min - 1][-1]}:
                            i_min -= 1
                        elif i_min > 1 and bid in {x[0] for x in pts[i_min - 2][-1]}:  # 只有当之后连续2个点标签不为bid才停止拓展
                            i_min -= 1
                        else:
                            break
                    while i_max < len(pts) - 1:
                        if bid in {x[0] for x in pts[i_max + 1][-1]}:
                            i_max += 1
                        elif i_max < len(pts) - 2 and bid in {x[0] for x in pts[i_max + 2][-1]}:
                            i_max += 1
                        else:
                            break
                    odrs_relate_pts.append(([odr], [i_min, i_max]))
                else:
                    undecided_odrs.append(odr)

            # 同一楼中的订单, 根据关联轨迹点的交叠进行合并
            all_odrs_relate_pts_merged += merge_list_min_max(odrs_relate_pts)

        all_odrs_relate_pts_merged.sort(key=lambda x:x[1][0])  # [(odrs, [i_min, i_max]), ...]
        assert len(undecided_odrs) + sum([len(x[0]) for x in all_odrs_relate_pts_merged]) == input_odrs_num
        return pts, all_odrs_relate_pts_merged, undecided_odrs

    pts_labeled, odrs_relate_pts, undecided_odrs = get_odrs_relate_pts(pts, orders, iod_data)

    def get_odrs_iot(odrs_relate_pts, undecided_odrs, pts, min_max_t):
        """
        对于有关联轨迹点的订单, 根据订单信息及关联轨迹点, 推断进楼/出楼时间
        对于没有关联轨迹点的订单, 根据订单信息及订单完成时间, 推断进楼/出楼时间
        处理了进楼/出楼时间发生交叠的情况
        在优先保证有关联轨迹点的订单之下, 尽可能在剩余的时间内插入没有关联轨迹点的订单
        """
        global throw_odr_iot_short
        global throw_odr_cannot_insert
        if odrs_relate_pts:
            # 推断有关联轨迹点的订单的iot
            odrs_iot = []
            for odrs, (i_min, i_max) in odrs_relate_pts:
                ti, to = infer_iot(odrs=odrs, pts=pts[i_min:i_max + 1], min_max_t=min_max_t)
                if ti:
                    odrs_iot.append((odrs, [ti, to]))
            # 合并同一楼中iot有交叠或接近的订单
            all_odrs_iot_merged = []
            bid2odrs_iot = defaultdict(list)
            for x in odrs_iot:
                bid2odrs_iot[x[0][0]["building_id"]].append(x)
            for onebd_odrs_iot in bid2odrs_iot.values():
                all_odrs_iot_merged += merge_list_min_max(onebd_odrs_iot, merge_gate=60)
            # 多个楼的放一起后, 处理iot发生交叠的情况
            odrs_iot_refined = tackle_odrs_iot_overlap(all_odrs_iot_merged)
            odrs_iot_refined.sort(key=lambda x:x[1][0])

        else:
            odrs_iot_refined = []
        
        if undecided_odrs:
            # 推断没有关联轨迹点的订单的iot
            ud_odrs_iot = []
            for odr in undecided_odrs:
                ti, to = infer_iot(odrs=[odr], min_max_t=min_max_t)
                if ti:
                    ud_odrs_iot.append(([odr], [ti, to]))
            ud_odrs_iot.sort(key=lambda x:x[1][0])
            # 合并同一楼中iot有交叠或接近的订单
            all_ud_odrs_iot_merged = []
            bid2ud_odrs_iot = defaultdict(list)
            for x in ud_odrs_iot:
                bid2ud_odrs_iot[x[0][0]["building_id"]].append(x)
            for onebd_ud_odrs_iot in bid2ud_odrs_iot.values():
                all_ud_odrs_iot_merged += merge_list_min_max(onebd_ud_odrs_iot, merge_gate=60)
            all_ud_odrs_iot_merged.sort(key=lambda x:x[1][0])
            # 筛选出与有关联轨迹点的订单的iot不交叠(且至少间隔10s)的
            ud_odrs_iot_filtered = []
            for odrs, (ti, to) in all_ud_odrs_iot_merged:
                for _, (ti2, to2) in odrs_iot_refined:
                    if to >= ti2 - T_OVERLAP and ti <= to2 + T_OVERLAP:  # 至少间隔10s
                        throw_odr_cannot_insert += len(odrs)
                        break
                else:
                    ud_odrs_iot_filtered.append((odrs, [ti, to]))
            # 多个楼的放一起后, 处理iot发生交叠的情况
            ud_odrs_iot_filtered_refined = tackle_odrs_iot_overlap(ud_odrs_iot_filtered)
            ud_odrs_iot_filtered_refined.sort(key=lambda x:x[1][0])
        else:
            ud_odrs_iot_filtered_refined = []
        
        # 尝试在有关联轨迹点的订单的iot之外的时间里, 插入没有关联轨迹点的订单
        for odrs, (ti, to) in ud_odrs_iot_filtered_refined:
            for _, (ti2, to2) in odrs_iot_refined:
                if to >= ti2 - T_OVERLAP and ti <= to2 + T_OVERLAP:  # 至少间隔10s
                    throw_odr_cannot_insert += len(odrs)
                    break
            else:
                odrs_iot_refined.append((odrs, [ti, to]))
        odrs_iot_refined.sort(key=lambda x:x[1][0])
        
        # 最终过滤, to-ti的时长需大于按简单物理规则用时的0.7倍
        odrs_iot_refined_filtered = []
        for odrs, (ti, to) in odrs_iot_refined:
            if to - ti > P_SHORT * gen_inbd_actions(odrs, 0)[-1]["end_time"]:
                odrs_iot_refined_filtered.append((odrs, [ti, to]))
            else:
                throw_odr_iot_short += len(odrs)

        # 确保没有交叠
        odrs_iot_refined_filtered = tackle_odrs_iot_overlap(odrs_iot_refined_filtered)

        # 检查iot合法性
        last_to = -1e10
        for odrs, (ti, to) in odrs_iot_refined_filtered:
            assert to > ti
            assert ti > last_to
            last_to = to
    
        return odrs_iot_refined_filtered

    t_margin = 5
    while 2 * t_margin >= cut_end_time - cut_start_time:
        t_margin /= 2
    odrs_iot = get_odrs_iot(
        odrs_relate_pts=odrs_relate_pts, 
        undecided_odrs=undecided_odrs,
        pts=pts_labeled,
        min_max_t=(cut_start_time + t_margin, cut_end_time - t_margin)
    )

    if odrs_iot:
        actions = gen_odrs_iot_actions(
            odrs_iot=odrs_iot,
            start_time=cut_start_time,
            end_time=cut_end_time,
            start_xy=pts[0][:2],
            end_xy=pts[-1][:2],
        )
    else:
        actions = gen_no_odr_actions(
            start_time=cut_start_time, 
            end_time=cut_end_time, 
            start_nid=None, 
            start_xy=pts[0][:2], 
            end_nid=None, 
            end_xy=pts[-1][:2]
        )

    return actions


def infer_actions(cuts, iod_data, nearstation_gate, sample_gap, outbd_gate, nearbd_gate, nearodr_gate, relateodr_gate):
    """
    主函数
    """
    actions = []
    for cut in cuts:
        if len(cut["points"]) < 2:
            assert len(cut["orders"]) == 0
    cuts = [cut for cut in cuts if len(cut["points"]) >= 2]

    for cut_idx, cut in enumerate(cuts):
        pts = cut["points"]
        odrs = cut["orders"]

        onecut_actions = []

        # head型轨迹段
        if cut["type"] == "head":
            sid, near_ts = find_near_station_times(pts[:-1], nearstation_gate)
            if near_ts:  # 从快递站出发
                te = max(near_ts)  # 离开站的时间
                # 进入站的时间是第一个 在站附近, 且其后连续两个点在站附近 的点
                idxs = []
                for t in near_ts:
                    for i, p in enumerate(pts):
                        if p[-1] == t:
                            idxs.append(i)
                            break
                for i, idx in enumerate(idxs):
                    if i+1 < len(idxs) and idxs[i+1] != idx + 1:
                        continue
                    if i+2 < len(idxs) and idxs[i+2] != idx + 2:
                        continue
                    break
                ts = max(5.5*3600, pts[idx][-1])
                ts = min(te - 600, ts)  # 进入站的时间不早于5点半, 且至少比离开时早10min
                onecut_actions += gen_instation_actions(sid, ts, te)
                start_time = te
                start_nid = sid
                start_xy = None
            else:  # 从坐标点出发
                start_time = pts[0][-1]
                start_nid = None
                start_xy = pts[0][:2]
            # 从快递站出发后/从路区外 到 路区内第一个轨迹点 的行为
            assert start_time < pts[-1][-1] 
            onecut_actions += gen_odrs_actions(
                orders=odrs,
                start_time=start_time, 
                end_time=pts[-1][-1], 
                start_nid=start_nid, 
                start_xy=start_xy, 
                end_nid=None, 
                end_xy=pts[-1][:2]
            )

        # tail型轨迹段
        elif cut["type"] == "tail":
            sid, near_ts = find_near_station_times(pts[1:], nearstation_gate)
            if near_ts:  # 去快递站: 快递站附近的轨迹点的最晚时间之后的行为被忽略, 均认为在快递站理货拣货
                end_time = min(near_ts)  # 到达快递站的时间
                end_time = end_time + min(1800, (pts[-1][-1] - end_time) / 3)  # 到快递站的时间推迟一点, 留一点时间送单
                end_nid = sid
                end_xy = None
            else:  # 去坐标点
                end_time = pts[-1][-1]
                end_nid = None
                end_xy = pts[-1][:2]
            # 从路区内最后一个轨迹点 到 快递站/路区外 的行为
            onecut_actions += gen_odrs_actions(
                orders=odrs,
                start_time=pts[0][-1], 
                end_time=end_time, 
                start_nid=None, 
                start_xy=pts[0][:2], 
                end_nid=end_nid, 
                end_xy=end_xy
            )
            if near_ts and end_time < pts[-1][-1]:  # 小哥到达快递站后, 还有一段时间, 认为在理货拣货
                if onecut_actions[-1]["type"] == ACTION_REST:  # 若到站后由于时间没花完而补了一个休息动作, 将其融进拣货理货中
                    end_time = onecut_actions[-1]["start_time"]
                    onecut_actions.pop()
                final_time  = min(23 * 3600, pts[-1][-1])
                final_time = max(end_time + 600, final_time)  # 行为结束时间不晚于23点, 且至少比进站时晚10min
                onecut_actions += gen_instation_actions(sid, end_time, final_time)         

        # inter型轨迹段
        elif cut["type"] == "inter":
            sid, near_ts = find_near_station_times(pts[1:-1], nearstation_gate)
            if near_ts:  # 中途去过快递站
                ts, te = min(near_ts), max(near_ts)
                if te - ts < 60:  # 在快递站待的时间太短, 考虑往两边挪挪, 但不得超出首尾轨迹点时间范围
                    t_margin = min(30, (pts[-1][-1] - pts[0][-1] - 1) / 2)
                    ts = max(pts[0][-1] + t_margin, (ts + te) / 2 - 30)
                    te = min(pts[-1][-1] - t_margin, (ts + te) / 2 + 30)
                t = (ts + te) / 2
                part1_odrs = [odr for odr in odrs if odr["finish_time"] <= t]
                part2_odrs = [odr for odr in odrs if odr["finish_time"] > t]
                part1_actions = gen_odrs_actions(
                    orders=part1_odrs,
                    start_time=pts[0][-1], 
                    end_time=ts, 
                    start_nid=None, 
                    start_xy=pts[0][:2], 
                    end_nid=sid, 
                    end_xy=None
                )
                part2_actions = gen_odrs_actions(
                    orders=part2_odrs,
                    start_time=te, 
                    end_time=pts[-1][-1], 
                    start_nid=sid, 
                    start_xy=None, 
                    end_nid=None, 
                    end_xy=pts[-1][:2]
                )
                if part1_actions[-1]["type"] == ACTION_REST:  # 若到站后由于时间没花完而补了一个休息动作, 将其融进拣货理货中
                    ts = part1_actions[-1]["start_time"]
                    part1_actions.pop()
                onecut_actions += part1_actions
                onecut_actions += gen_instation_actions(sid, ts, te)
                onecut_actions += part2_actions
            else:  # 中途没去过快递站
                onecut_actions += gen_odrs_actions(
                    orders=odrs,
                    start_time=pts[0][-1], 
                    end_time=pts[-1][-1], 
                    start_nid=None, 
                    start_xy=pts[0][:2], 
                    end_nid=None, 
                    end_xy=pts[-1][:2],
                )

        # cut型轨迹段
        elif cut["type"] == "cut":
            onecut_actions += gen_odrs_pts_actions(
                orders=odrs, 
                pts=pts,
                iod_data=iod_data,
                sample_gap=sample_gap,
                outbd_gate=outbd_gate,
                nearbd_gate=nearbd_gate,
                nearodr_gate=nearodr_gate,
                relateodr_gate=relateodr_gate,
            )

        # 最后一段轨迹结束, 若小哥不在站里, 强制让小哥回站再下班
        if cut_idx == len(cuts) - 1:
            if cut["type"] == "tail":
                if not near_ts:
                    if onecut_actions and onecut_actions[-1]["type"] == ACTION_REST:
                        start_time = onecut_actions[-1]["start_time"]
                        onecut_actions.pop()
                    else:
                        start_time = pts[-1][-1]
                    onecut_actions += gen_no_odr_actions(
                        start_time=start_time, 
                        end_time=None, 
                        start_xy=pts[-1][:2], 
                        end_nid=DEFAULT_SID
                    )
            elif cut["type"] in {"cut", "inter"}:
                if onecut_actions and onecut_actions[-1]["type"] == ACTION_REST:
                    start_time = onecut_actions[-1]["start_time"]
                    onecut_actions.pop()
                else:
                    start_time = pts[-1][-1]
                onecut_actions += gen_no_odr_actions(
                    start_time=start_time, 
                    end_time=None, 
                    start_xy=pts[-1][:2], 
                    end_nid=DEFAULT_SID
                )
            else:
                assert False, "The last part of traj is 'head'!"

        actions += onecut_actions

    return actions


def check_results(actions):
    """
    检查action序列的合法性
    """
    last_te = None
    for i, action in enumerate(actions):
        assert action["end_time"] > action["start_time"], "时长不为正"
        if last_te:
            assert abs(action["start_time"] - last_te) < 0.1, "时间不首尾相接"
        last_te = action["end_time"]
    
    def get_start_end_pos(action):
        if "start_xy" in action:
            start_pos = action["start_xy"]
        elif "start_node" in action:
            start_pos = G.nodes[action["start_node"]]["xy"]
        elif "start_building" in action:
            start_pos = buildings[action["start_building"]]["gate_xy"]
        if "end_xy" in action:
            end_pos = action["end_xy"]
        elif "end_node" in action:
            end_pos = G.nodes[action["end_node"]]["xy"]
        elif "end_building" in action:
            end_pos = buildings[action["end_building"]]["gate_xy"]
        return start_pos, end_pos

    def get_dis(p1, p2):
        x1, y1 = p1
        x2, y2 = p2
        return (x1 - x2) ** 2 + (y1 - y2) ** 2

    last_pos = None
    tmp = []
    for action in actions:
        tmp.append(action)
        if action["type"] in ACTION_MOVE:
            start_pos, end_pos = get_start_end_pos(action)
            if last_pos:
                try:
                    assert get_dis(last_pos, start_pos) < 1, "位置不首尾相接"
                    tmp = [action]
                except:
                    print("位置不首尾相接")
                    pprint_actions(tmp)
                    exit()
            last_pos = end_pos
    
    for action in actions:
        assert "support_points" in action
        assert "target_orders" in action
        if action["type"] in ACTION_ORDER:
            assert len(action["target_orders"]) == 1
            for k in ["floor", "unit", "building"]:
                assert k in action
        elif action["type"] in ACTION_MOVE:
            assert action["xy"], "移动路径为空" 
        elif action["type"] in ACTION_FLOOR:
            for k in ["from", "to", "unit", "building"]:
                assert k in action
        elif action["type"] == ACTION_TODOOR:
            for k in ["floor", "unit", "building"]:
                assert k in action


def postprocess(actions, traj_points_orig, do_refine_instation=True):
    """后处理actions, 转为标准数字格式, 补充一些字段, 细化在站里的行为"""
    def pkl2json(actions):
        """将action中的所有np.float64, np.int64转为float, int, 以便可以json"""
        for a in actions:
            for k, v in a.items():
                if isinstance(v, np.float64):
                    a[k] = float(v)
                elif isinstance(v, np.int64):
                    a[k] = int(v)
                elif isinstance(v, list):
                    v_new = []
                    for x in v:
                        if isinstance(x, tuple) or isinstance(x, list):
                            x = [float(y) for y in x]
                        elif isinstance(x, dict):
                            for kk, vv in x.items():
                                if isinstance(vv, np.float64):
                                    x[kk] = float(vv)
                                elif isinstance(vv, np.int64):
                                    x[kk] = int(vv)
                        v_new.append(x)
                    a[k] = v_new
                elif isinstance(v, dict):
                    for kk, vv in v.items():
                        if isinstance(vv, np.float64):
                            v[kk] = float(vv)
                        elif isinstance(vv, np.int64):
                            v[kk] = int(vv)
                    a[k] = v
        return actions

    actions = pkl2json(actions)

    # 计算移动路线长度
    for action in actions:
        if action["type"] in ACTION_MOVE:
            action["length"] = LineString(action["xy"]).length

    def refine_instation_actions(actions):
        """
        在生成actions的过程中, 先直接用ACTION_ARRANGE填充整段在营业部里的时间, 现在再细化
        首先调整在营业部里的时间:
            若两个ACTION_ARRANGE之间时间间隔比较短且中间夹的action没有完成订单的, 则直接将中间的时间也吸收进在营业部里的时间
        然后找所有派送单的start_time, 并尝试将它们匹配到在营业部的时段中:
            若时间落在营业部时段外, 如果离最近时段的时间范围里是在rest或者没有action, 那么进一步延展在营业部里的时间段
            还是匹配不上的, 说明是在路区里送的时候才想起来点收货, 直接把它放到它之前最近的在站里的时间段里, start_time修改为那段里的平均收货时间
        """
        def absorb_between_instation(actions):
            """
            若两个ACTION_ARRANGE之间时间间隔比较短且中间夹的action没有完成订单的
            将中间的时间也吸收进在营业部里的时间
            """
            arrange_idxs = [i for i, a in enumerate(actions) if a["type"] == ACTION_ARRANGE]
            for i, j in zip(arrange_idxs, arrange_idxs[1:]):
                if actions[j]["start_time"] - actions[i]["end_time"] < 3600:
                    for a in actions[i+1:j]:
                        if a["type"] in ACTION_ORDER:
                            break
                    else:  # 吸收夹在中间的action
                        actions[i]["end_time"] = actions[j]["end_time"]
                        actions = actions[:i+1] + actions[j+1:]
                        return absorb_between_instation(actions)
            return actions

        actions = absorb_between_instation(actions)
        
        # 在把派件单的收货时间匹配到在营业部的时间的过程中, 进一步调整在营业部的时间
        aidx_ts_te = [(i, a["start_time"], a["end_time"]) for i, a in enumerate(actions)]
        oid_ts = [
            (a["target_orders"][0]["id"], a["target_orders"][0]["start_time"])
            for a in actions if a["type"] == ACTION_DELIVER
        ]
        aidx2td = defaultdict(list)  # 需要进一步调整起止时间的在营业部的时段
        aidx2ts = defaultdict(list)  # 记录每个在营业部的时段, 匹配到的收货时间
        oid_ts_tochange = []         # 记录匹配不到任何营业部时段的收货时间
        for oid, t in oid_ts:
            if t < 5 * 3600:
                # print(f"收货时间不理想: {t}")
                oid_ts_tochange.append((oid, t))
                continue
            aidx_td = []
            for i, ts, te in aidx_ts_te:
                if ts <= t <= te:
                    aidx2ts[i].append(t)
                    break
                else:
                    aidx_td.append((i, t - ts if t < ts else t - te))
            else:  # 收货时间未落在任何在营业部里的时段内
                aidx_td.sort(key=lambda x:abs(x[1]))
                for i, td in aidx_td:
                    if td < 0:
                        if i == 0:
                            aidx2td[i].append(td)
                            aidx2ts[i].append(t)
                            break
                        elif actions[i-1]["type"] == ACTION_REST and actions[i-1]["start_time"] < t:
                            aidx2td[i].append(td)
                            aidx2ts[i].append(t)
                            break
                    else:
                        if i == len(actions) - 1:
                            aidx2td[i].append(td)
                            aidx2ts[i].append(t)
                            break
                        elif actions[i+1]["type"] == ACTION_REST and actions[i+1]["end_time"] > t:
                            aidx2td[i].append(td)
                            aidx2ts[i].append(t)
                            break
                else:   # 无法通过调整营业部的时段范围来使得能够匹配上的收货时间
                    oid_ts_tochange.append((oid, t))
        # 调整在营业部的时段
        for i, tds in aidx2td.items():
            a = [t for t in tds if t < 0]
            if a:
                a = min(a)
                if i == 0:
                    actions[i]["start_time"] += a
                else:
                    actions[i-1]["end_time"] += a
                    actions[i]["start_time"] += a
            a = [t for t in tds if t > 0]
            if a:
                a = max(a)
                if i == len(actions) - 1:
                    actions[i]["end_time"] += a
                else:
                    actions[i]["end_time"] += a
                    actions[i+1]["start_time"] += a
        # 调整收货时间
        oid2ts_new = {}
        aidx_range = [[i, (a["start_time"], a["end_time"])] for i, a in enumerate(actions) if a["type"] == ACTION_ARRANGE]
        aidx2ts_range = {i:(min(ts), max(ts)) for i, ts in aidx2ts.items()}
        for oid, t in oid_ts_tochange:
            for i, (_, (ts, _)) in enumerate(aidx_range):
                if ts > t:
                    break
            aidx, (ts, te) = aidx_range[max(i-1, 0)]  # 无法匹配的收货时间, 改到上一次在营业部里的时间里
            if aidx in aidx2ts_range:
                ts, te = aidx2ts_range[aidx]
            ts_new = random.uniform(ts, te)
            oid2ts_new[oid] = ts_new
            aidx2ts[aidx].append(ts_new)

        for action in actions:
            for o in action["target_orders"]:
                if o["type"] == ORDER_DELIVER and o["id"] in oid2ts_new:
                    o["start_time"] = oid2ts_new[o["id"]]
        # 细化在站里的行为
        actions_new = []
        for i, a in enumerate(actions):
            if a["type"] == ACTION_ARRANGE:
                actions_new += gen_detail_instation_actions(
                    sid=a["station_id"], 
                    start_time=a["start_time"], 
                    end_time=a["end_time"],
                    receive_times=aidx2ts.get(i, []),
                )
            else:
                actions_new.append(a)
        return actions_new

    # 细化在站里的行为
    if do_refine_instation:
        actions = refine_instation_actions(actions)

    # 设置订单开始时间, 确保订单在最早成为target_orders之前产生(满足因果)
    # 记录订单的以下几个时间: 最早成为target的时间, 开始被送的时间, 实际被完成的时间
    oid2t_target = defaultdict(list)
    oid2t_serving = {}
    oid2t_served = {}
    for action in actions:
        for o in action["target_orders"]:
            oid2t_target[o["id"]].append(action["start_time"])
        if action["type"] in ACTION_ORDER:
            for o in action["target_orders"]:
                oid2t_serving[o["id"]] = action["start_time"]
                oid2t_served[o["id"]] = action["end_time"]
    oid2t_target = {o : min(ts) for o, ts in oid2t_target.items()}
    for action in actions:
        for o in action["target_orders"]:
            o["target_time"] = oid2t_target.get(o["id"], None)
            o["serving_time"] = oid2t_serving.get(o["id"], None)
            o["served_time"] = oid2t_served.get(o["id"], None)
            if not o["target_time"] is None:
                if o["target_time"] < o["start_time"]:
                    # print("adjust order start_time ahead of target_time", o["type"])
                    t_advance = T_DELIVER_ADVANCE if o["type"] == ORDER_DELIVER else T_PICK_ADVANCE
                    o["start_time"] = min(o["start_time"], o["target_time"] - t_advance)

    # 检查订单不同状态的时间合法性
    for action in actions:
        for o in action["target_orders"]:
            ts = []
            for k in ["start_time", "target_time", "serving_time", "served_time"]:
                t = o.get(k, None)
                if t:
                    ts.append(t)
            if len(ts) > 1:
                for t1, t2 in zip(ts, ts[1:]):
                    assert t2 >= t1

    def calculate_var(act):
        """计算某个action对累计统计指标产生的影响"""
        atype = act["type"]
        var = {}
        if atype == ACTION_WALK:  # 路区和站间往返不计入移动距离
            var = {"traveled_length": act["length"]}
        elif atype == ACTION_UPSTAIR:  # 坐电梯或下楼不计爬楼层数
            var = {"climbed_floors": act["num"]}
        elif atype == ACTION_DELIVER:
            on_time = act["end_time"] <= act["target_orders"][0]["ddl_time"]
            return {
                "delivered_orders": 1, 
                "delivered_on_time": 1 if on_time else 0,
            }
        elif atype == ACTION_CPICK:
            on_time = act["end_time"] <= act["target_orders"][0]["ddl_time"]
            return {
                "cpicked_orders": 1, 
                "cpicked_on_time": 1 if on_time else 0,
            }
        elif atype == ACTION_BPICK:
            on_time = act["end_time"] <= act["target_orders"][0]["ddl_time"]
            return {
                "bpicked_orders": 1, 
                "bpicked_on_time": 1 if on_time else 0,
            }
        # work_time
        if "station_id" not in act:
            if atype in [ACTION_REST, ACTION_DISCHARGE] and act["end_time"] - act["start_time"] > 10 * 60:
                var["work_time"] = (act["end_time"] - act["start_time"]) * 0.2
            else:
                var["work_time"] = act["end_time"] - act["start_time"]
        elif atype == ACTION_ARRANGE:
            var["work_time"] = act["end_time"] - act["start_time"]
        return var
    
    # 补充每个action开始时的累计统计指标
    vars_maintain = {
        "traveled_length": 0.0,
        "climbed_floors": 0,
        "delivered_orders": 0,
        "delivered_on_time": 0,
        "cpicked_orders": 0,
        "cpicked_on_time": 0,
        "bpicked_orders": 0,
        "bpicked_on_time": 0,
        "work_time": 0.0
    }
    # 为了体现最后一个action对status的影响, 最后添加一个空action, 因为status都是以action开始时来算的
    actions.append({
        "type": ACTION_REST,
        "start_time": actions[-1]["end_time"],
        "end_time": actions[-1]["end_time"] + 0.1,
        "target_orders": [],
    })
    for action in actions:
        action["status"] = deepcopy(vars_maintain)
        action["status"].pop("work_time")
        efficiency = (vars_maintain["delivered_orders"] + vars_maintain["cpicked_orders"] + vars_maintain["bpicked_orders"]) \
            / (vars_maintain["work_time"] + 1e-12) * 3600
        efficiency = max(0, min(efficiency, 28.3))
        action["status"]["efficiency"] = efficiency
        for k, v in calculate_var(action).items():
            vars_maintain[k] += v

    # status补充未完成订单数
    def calculate_order_status(odr, t):
        """计算订单在t时刻的完成状态"""
        if t < odr["start_time"]:
            return ORDER_UNSTART
        elif t < odr["target_time"]:
            return ORDER_WAITING
        elif t < odr["serving_time"]:
            return ORDER_TARGETING
        elif t < odr["served_time"]:
            return ORDER_SERVING
        else:
            return ORDER_SERVED
    orders = [o for a in actions if a["type"] in ACTION_ORDER for o in a["target_orders"]]
    oid2odr = {o["id"]: o for o in orders}
    for action in actions:
        oid2status = {oid: calculate_order_status(odr, action["start_time"]) for oid, odr in oid2odr.items()}
        started = [oid2odr[oid]["type"] for oid, s in oid2status.items() if s != ORDER_UNSTART]
        counter = Counter(started)
        deliver_all = counter[ORDER_DELIVER]
        cpick_all = counter[ORDER_CPICK]
        bpick_all = counter[ORDER_BPICK]
        action["status"]["deliver_todo"] = deliver_all - action["status"]["delivered_orders"]
        action["status"]["cpick_todo"] = cpick_all - action["status"]["cpicked_orders"]
        action["status"]["bpick_todo"] = bpick_all - action["status"]["bpicked_orders"]

    # 增加go_for_picking字段
    cpick_set = set()  # 所有已成为target但未完成的C揽单
    for action in actions:
        for odr in action["target_orders"]:
            if odr["type"] == ORDER_CPICK:
                cpick_set.add(odr["id"])
        action["go_for_picking"] = True if cpick_set else False
        if action["type"] == ACTION_CPICK:
            cpick_set.remove(action["target_orders"][0]["id"])

    # 给action加上support_points字段
    i, j = 0, 0
    pts = [(*projector(x, y, inverse=True), t) for x, y, t in traj_points_orig]
    for a in actions:
        ts, te = a["start_time"], a["end_time"]
        while i < len(pts):  # i是第一个t>=ts的点, 若最后一点仍<ts, i越界
            if pts[i][-1] < ts:
                i += 1
            else:
                break
        while j < len(pts):  # j是第一个t>te的点, 若最后一个点仍<=te, j越界
            if pts[j][-1] <= te:
                j += 1
            else:
                break
        a["support_points"] = pts[i:j]
    
    return actions


def main(traj_points_orig, orders, iod_data=None):
    """程序入口"""
    global throw_odr_match2cut, throw_odr_overtime, throw_odr_iot_overlap, throw_odr_iot_short, throw_odr_cannot_insert
    throw_odr_match2cut = 0
    throw_odr_overtime = 0
    throw_odr_iot_overlap = 0
    throw_odr_iot_short = 0
    throw_odr_cannot_insert = 0 

    def preprocess(traj_points_orig, orders):
        """预处理输入的轨迹和订单数据, 并做合法性检查"""
        # 预处理轨迹数据
        traj_points_orig.sort(key=lambda x:x[-1])
        tmp = []
        one_merge = [traj_points_orig[0]]
        last_t = traj_points_orig[0][-1]
        for p in traj_points_orig:  # 对于同一个时间多个不同位置的轨迹点, 合并为位置取平均的一个点
            if p[-1] == last_t:
                one_merge.append(p)
            else:
                if len(one_merge) == 1:
                    tmp.append(one_merge[0])
                else:
                    lon, lat, t = zip(*one_merge)
                    tmp.append((np.mean(lon), np.mean(lat), t[0]))
                one_merge = [p]
            last_t = p[-1]
        if len(one_merge) == 1:
            tmp.append(one_merge[0])
        else:
            lon, lat, t = zip(*one_merge)
            tmp.append((np.mean(lon), np.mean(lat), t[0]))
        traj_points_orig = [[*projector(lon, lat), t] for lon, lat, t in tmp]  # 投影到平面xy坐标
        traj_points_orig = [[float(lon), float(lat), float(t)] for lon, lat, t in traj_points_orig]  # 转成标准数据格式
        # 预处理订单数据
        oids = []
        for odr in orders:
            oids.append(odr["id"])
            assert isinstance(odr["building_id"], int)
            floor = int(odr["floor"])
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
            assert 1 <= floor <= 30
            odr["floor"] = floor
            unit = int(odr["unit"])
            assert 1 <= unit <= 10
            odr["unit"] = unit
            # if odr["start_time"] < 5 * 3600:  # 历史遗留的派件, 可能来自历史日期
            #     odr["start_time"] = min(7.5 * 3600, odr["finish_time"] - 600)
            assert odr["finish_time"] > odr["start_time"]
        assert len(oids) == len(set(oids))
        for odr in orders:  # 转成标准数据格式
            for k, v in odr.items():
                if isinstance(v, np.int64):
                    odr[k] = int(v)
                elif isinstance(v, np.float64):
                    odr[k] = float(v)
                else:
                    assert isinstance(v, str) or isinstance(v, int) or isinstance(v, float) or isinstance(v, bool)
        # 找出所有订单所在楼所在路区
        related_rids = []
        for odr in orders:
            related_rids.append(buildings[odr["building_id"]]["region"])
        related_rids = set(related_rids)

        return traj_points_orig, orders, related_rids

    # 预处理
    traj_points_orig, orders, related_rids = preprocess(traj_points_orig, orders)

    # 过滤掉距离 所有订单所在楼所在的路区 都太远的轨迹点
    traj_points_filtered, idx_filtered = filter_traj_points(traj_points_orig, related_rids, dis_gate=150)
    print("orig traj points:", len(traj_points_orig))
    print("traj points after filtering:", len(traj_points_filtered))
    if not idx_filtered:
        print("no traj points after filtering!")
        return []

    # 按时间间隔阈值将轨迹切段
    cuts = cut_traj(traj_points_filtered, idx_filtered, traj_points_orig, tm_gap_gate=600)
    print("cut num of each type:", list(dict(Counter(c["type"] for c in cuts)).items()))

    # 将订单分配到轨迹段
    cuts = match_order_to_cut(orders, cuts, tm_gap_gate=60)

    # 处理小哥轨迹缺失的问题: 大量订单的完成时间在轨迹的时间范围外
    if throw_odr_match2cut > len(orders) * 0.3:
        ts = [o["finish_time"] for o in orders if 0 < o["finish_time"] < 86400]
        t_min, t_max = min(ts), max(ts)
        x, y, t = traj_points_orig[0]
        if t_min < t:
            traj_points_orig = [(x, y, t_min - 1)] + traj_points_orig
        x, y, t = traj_points_orig[-1]
        if t_max > t:
            traj_points_orig = traj_points_orig + [(x, y, t_max + 1)]
        traj_points_orig = [(*projector(x, y, inverse=True), t) for x, y, t in traj_points_orig]
        print("too many odrs out of traj time range!")
        return main(traj_points_orig, orders)  # 递归

    # 推断小哥行为
    actions = infer_actions(
        cuts=cuts,
        iod_data=iod_data,
        nearstation_gate=200,
        sample_gap=10,
        outbd_gate=30,
        nearbd_gate=5,
        nearodr_gate=120,
        relateodr_gate=120
    )
    print("action num:", len(actions))
    print("action time range:", actions[-1]["end_time"] - actions[0]["start_time"])
    print("action start/end time:", actions[0]["start_time"], actions[-1]["end_time"])

    # 报告订单丢失情况
    print("orig orders:", len(orders))
    print("throw_odr_match2cut:", throw_odr_match2cut)
    print("throw_odr_overtime:", throw_odr_overtime)
    print("throw_odr_iot_overlap:", throw_odr_iot_overlap)
    print("throw_odr_cannot_insert:", throw_odr_cannot_insert)
    print("throw_odr_iot_short:", throw_odr_iot_short)
    total_orders = len([1 for a in actions if a["type"] in ACTION_ORDER])
    assert len(orders) - total_orders == \
        throw_odr_match2cut + throw_odr_overtime + throw_odr_iot_overlap + throw_odr_cannot_insert + throw_odr_iot_short

    actions = postprocess(actions, traj_points_orig)

    # 检查行为序列合法性
    check_results(actions)

    # 避免由于浮点数精度问题带来的action时间不首尾相接(check_results中已经检查过事实上是首尾相接)
    last_te = actions[0]["start_time"]
    for action in actions:
        action["start_time"] = last_te
        last_te = action["end_time"]

    return actions


def get_bd_status(many_courier_actions):
    """记录楼里各种类型的订单数量"""
    bid2events = defaultdict(list)  # 每栋楼里, 订单开始或完成时, 记录一个event, (t, odr_type, 开始/完成)
    for actions in many_courier_actions:
        for a in actions:
            if a["type"] in ACTION_ORDER:
                o = a["target_orders"][0]
                bid2events[o["building_id"]].append((o["start_time"], o["type"], 1))
                bid2events[o["building_id"]].append((o["served_time"], o["type"], -1))
    bid2status = {bid:[] for bid in buildings}
    for bid in buildings:
        events = bid2events.get(bid, [])
        if events:
            events_before0 = [x for x in events if x[0] <= 0]
            status0 = {
                "time": 0,
                ORDER_DELIVER: 0,
                ORDER_CPICK: 0,
                ORDER_BPICK: 0
            }
            for e in events_before0:
                status0[e[1]] += e[2]
            bid2status[bid].append(status0)

            events_after0 = [x for x in events if x[0] > 0]
            t2events = defaultdict(list)
            for e in events_after0:
                t2events[e[0]].append(e[1:])
            t_events = sorted(list(t2events.items()), key=lambda x:x[0])
            for t, events in t_events:
                s = deepcopy(bid2status[bid][-1])
                s["time"] = t
                for a, b in events:
                    s[a] += b
                bid2status[bid].append(s)
        else:
            bid2status[bid].append({
                "time": 0,
                ORDER_DELIVER: 0,
                ORDER_CPICK: 0,
                ORDER_BPICK: 0
            })

    return bid2status


def manify_3D_actions():
    """调整3D两个路区小哥前3单, 使得比较好看"""
    data = pickle.load(open(f"data/actions_recover.pkl", "rb"))
    data_new = []
    for cid, actions in data:
        if int(cid) == 22284086:  # 龙正业
            # print(len(actions))
            # print(len([a for a in actions if a["type"] in ACTION_ORDER]))
            # pprint_actions(actions[:20])

            traj_points_orig = sum([a["support_points"] for a in actions], [])
            traj_points_orig = list(set(traj_points_orig))
            traj_points_orig.sort(key=lambda x:x[-1])

            actions_part1 = actions[:4]

            t = actions[4]["end_time"] - actions[4]["start_time"]  # 去掉去路区前的一个rest
            actions[5]["start_time"] -= t    # 去路区提前
            actions[5]["end_time"] -= t
            actions_part1.append(actions[5]) # 到路区及之前
            actions_part2 = actions[9:]      # 送第一个单及之后
            
            start_time = actions_part1[-1]["end_time"]
            start_xy = actions_part1[-1]["end_xy"]
            end_building = actions_part2[0]["building"]
            end_time = actions_part2[0]["start_time"]
            t_delta = end_time - start_time
            print(start_time, end_time, t_delta)

            orders_manify = [
                {
                    'building_id': 757,
                    'ddl_time': 86399.0,
                    'finish_time': start_time + t_delta / 3,
                    'floor': 5,
                    'id': 233332,
                    'start_time': 29162.0,
                    'type': 'deliver',
                    'unit': 1
                },
                {
                    'building_id': 757,
                    'ddl_time': 86399.0,
                    'finish_time': start_time + t_delta / 3,
                    'floor': 8,
                    'id': 233333,
                    'start_time': 29162.0,
                    'type': 'deliver',
                    'unit': 1
                },
                {
                    'building_id': 757,
                    'ddl_time': 86399.0,
                    'finish_time': start_time + t_delta / 3,
                    'floor': 10,
                    'id': 233334,
                    'start_time': 29162.0,
                    'type': 'deliver',
                    'unit': 2
                },
                {
                    'building_id': 755,
                    'ddl_time': 86399.0,
                    'finish_time': start_time + t_delta * 2 / 3,
                    'floor': 3,
                    'id': 233335,
                    'start_time': 29162.0,
                    'type': 'deliver',
                    'unit': 1
                },
                {
                    'building_id': 755,
                    'ddl_time': 86399.0,
                    'finish_time': start_time + t_delta * 2 / 3,
                    'floor': 4,
                    'id': 233336,
                    'start_time': 29162.0,
                    'type': 'deliver',
                    'unit': 2
                },
                {
                    'building_id': 753,
                    'ddl_time': 86399.0,
                    'finish_time': start_time + t_delta,
                    'floor': 2,
                    'id': 233337,
                    'start_time': 29162.0,
                    'type': 'deliver',
                    'unit': 1
                },
                {
                    'building_id': 753,
                    'ddl_time': 86399.0,
                    'finish_time': start_time + t_delta,
                    'floor': 3,
                    'id': 233338,
                    'start_time': 29162.0,
                    'type': 'deliver',
                    'unit': 1
                },
            ]

            actions_add = gen_has_odr_actions(
                orders=orders_manify, 
                start_time=start_time, 
                # end_time = end_time,
                start_xy=start_xy, 
                end_nid=buildings[orders_manify[-1]["building_id"]]["gate_id"],
                follow_order_seq=True
            )
            actions_add = actions_add[:-1]
            last_te = actions_add[0]["start_time"]
            for a in actions_add:
                if a["type"] == ACTION_ELEVATOR:
                    a["end_time"] = last_te + a["end_time"] - a["start_time"] + 10
                elif a["type"] == ACTION_DELIVER:
                    a["end_time"] = last_te + a["end_time"] - a["start_time"] + 14
                else:
                    a["end_time"] = last_te + a["end_time"] - a["start_time"]
                a["start_time"] = last_te
                last_te = a["end_time"]
            print(actions_add[0]["start_time"], actions_add[-1]["end_time"], actions_add[-1]["end_time"] - actions_add[0]["start_time"])

            path_gps, path_xy, use_time = get_travel_path_t(
                start_nid=buildings[orders_manify[-1]["building_id"]]["gate_id"], 
                end_nid=buildings[end_building]["gate_id"], 
                v=V_WALK)
            time_axis = actions_add[-1]["end_time"]
            actions_add.append(
                {
                    "type": ACTION_WALK,
                    "start_time": time_axis,
                    "end_time": time_axis + use_time,
                    "start_building": orders_manify[-1]["building_id"],
                    "end_building": end_building,
                    "gps": path_gps,
                    "xy": path_xy,
                    "target_orders": actions[13]["target_orders"],
                }
            )
            time_axis += use_time
            print(actions_add[-1]["start_time"], actions_add[-1]["end_time"])
            
            actions_add.append({
                "type": ACTION_DISCHARGE,
                "start_time": time_axis,
                "end_time": end_time,
                "building": end_building,
                "target_orders": actions[13]["target_orders"],
            })
            
            actions = actions_part1 + actions_add + actions_part2
            actions = postprocess(actions[:-1], traj_points_orig, do_refine_instation=False)
            check_results(actions)
        elif int(cid) == 22346500:  # 余祺祥
            # print(len(actions))
            # print(len([a for a in actions if a["type"] in ACTION_ORDER]))
            # pprint_actions(actions[:20])
            # exit()

            traj_points_orig = sum([a["support_points"] for a in actions], [])
            traj_points_orig = list(set(traj_points_orig))
            traj_points_orig.sort(key=lambda x:x[-1])

            actions_part1 = actions[:8]
            t = actions[8]["end_time"] - actions[8]["start_time"]  # 去掉去路区前的一个rest
            actions[9]["start_time"] -= t    # 去路区提前
            actions[9]["end_time"] -= t
            actions_part1.append(actions[9]) # 到路区及之前
            actions_part2 = actions[13:]     # 送第一个单及之后
            
            start_time = actions_part1[-1]["end_time"]
            start_xy = actions_part1[-1]["end_xy"]
            end_building = actions_part2[0]["building"]
            end_time = actions_part2[0]["start_time"]
            t_delta = end_time - start_time
            print(start_time, end_time, t_delta)

            orders_manify = [
                {
                    'building_id': 161,
                    'ddl_time': 86399.0,
                    'finish_time': start_time + t_delta / 3,
                    'floor': 3,
                    'id': 233339,
                    'start_time': 29162.0,
                    'type': 'deliver',
                    'unit': 1
                },
                {
                    'building_id': 161,
                    'ddl_time': 86399.0,
                    'finish_time': start_time + t_delta / 3,
                    'floor': 6,
                    'id': 233340,
                    'start_time': 29162.0,
                    'type': 'deliver',
                    'unit': 1
                },
                # {
                #     'building_id': 161,
                #     'ddl_time': 86399.0,
                #     'finish_time': start_time + t_delta / 3,
                #     'floor': 8,
                #     'id': 233341,
                #     'start_time': 29162.0,
                #     'type': 'deliver',
                #     'unit': 2
                # },
                {
                    'building_id': 154,
                    'ddl_time': 86399.0,
                    'finish_time': start_time + t_delta * 2 / 3,
                    'floor': 2,
                    'id': 233342,
                    'start_time': 29162.0,
                    'type': 'deliver',
                    'unit': 1
                },
                # {
                #     'building_id': 154,
                #     'ddl_time': 86399.0,
                #     'finish_time': start_time + t_delta * 2 / 3,
                #     'floor': 3,
                #     'id': 233343,
                #     'start_time': 29162.0,
                #     'type': 'deliver',
                #     'unit': 1
                # },
            ]

            actions_add = gen_has_odr_actions(
                orders=orders_manify, 
                start_time=start_time, 
                # end_time = end_time,
                start_xy=start_xy, 
                end_nid=buildings[orders_manify[-1]["building_id"]]["gate_id"],
                follow_order_seq=True
            )
            actions_add = actions_add[:-1]
            last_te = actions_add[0]["start_time"]
            for a in actions_add:
                if a["type"] == ACTION_ELEVATOR:
                    a["end_time"] = last_te + a["end_time"] - a["start_time"] + 10
                elif a["type"] == ACTION_DELIVER:
                    a["end_time"] = last_te + a["end_time"] - a["start_time"] + 14
                else:
                    a["end_time"] = last_te + a["end_time"] - a["start_time"]
                a["start_time"] = last_te
                last_te = a["end_time"]
            print(actions_add[0]["start_time"], actions_add[-1]["end_time"], actions_add[-1]["end_time"] - actions_add[0]["start_time"])
            
            path_gps, path_xy, use_time = get_travel_path_t(
                start_nid=buildings[orders_manify[-1]["building_id"]]["gate_id"], 
                end_nid=buildings[end_building]["gate_id"], 
                v=V_WALK)
            time_axis = actions_add[-1]["end_time"]
            actions_add.append(
                {
                    "type": ACTION_WALK,
                    "start_time": time_axis,
                    "end_time": time_axis + use_time,
                    "start_building": orders_manify[-1]["building_id"],
                    "end_building": end_building,
                    "gps": path_gps,
                    "xy": path_xy,
                    "target_orders": actions[12]["target_orders"],
                }
            )
            time_axis += use_time
            print(actions_add[-1]["start_time"], actions_add[-1]["end_time"])

            # actions_add.append({
            #     "type": ACTION_DISCHARGE,
            #     "start_time": time_axis,
            #     "end_time": end_time,
            #     "building": end_building,
            #     "target_orders": actions[12]["target_orders"],
            # })
            actions_add[-1]["end_time"] = end_time

            actions = actions_part1 + actions_add + actions_part2
            actions = postprocess(actions[:-1], traj_points_orig, do_refine_instation=False)
            check_results(actions)
        
        data_new.append((cid, actions))

    pickle.dump(data_new, open(f"data/actions_recover_manify.pkl", "wb"))
    return


if __name__ == "__main__":
    # # 手工编排3D的前几单
    # manify_3D_actions()
    # exit()

    # # 重做post_process
    # data = pickle.load(open(f"data/actions_full.pkl", "rb"))
    # data_new = []
    # for cid, actions in data:
    #     if actions:
    #         traj_points_orig = sum([a["support_points"] for a in actions], [])
    #         traj_points_orig = list(set(traj_points_orig))
    #         traj_points_orig.sort(key=lambda x:x[-1])
    #         actions = postprocess(actions[:-1], traj_points_orig, do_refine_instation=False)
    #         check_results(actions)
    #     data_new.append((int(cid), actions))
    # pickle.dump(data, open(f"data/actions_full.pkl", "wb"))
    # exit()

    # 轨迹和订单数据的时间请使用daytime, 即上午10点为36000.0

    # 输入轨迹 traj_points: [(lon, lat, t), ...]
    traj_points_orig = [
        (116.441672,39.969749,-2000),  # 从快递站出发
        (116.431791,39.969548,-1000),

        # (116.428513,39.971024,-2000),  # 从路区外出发
        # (116.426002,39.971344,-1000),

        (116.428933,39.968672,0),   # 在路区
        (116.428611,39.968898,100),
        (116.428,39.968898, 110),
        (116.428155,39.968898,130),
        (116.42742,39.968803,190),
        (116.42683,39.968795,250),
        (116.426487,39.968766,255),
        (116.42668,39.968799,270),
        (116.426423,39.968766,280),
        (116.427018,39.968396,340),

        (116.4257, 39.9683, 600),
        (116.4264, 39.9681, 700),
        (116.4257, 39.9680, 800),

        (116.425342,39.967147,1000),
        (116.425557,39.966764,1020),
        (116.425873,39.966694,1040),

        (116.421995,39.967854,1400),  # 出路区, 但并不在快递站
        (116.42207,39.96713,1500),
        (116.421985,39.966752,1600),

        (116.425503,39.967965,1700),  # 回路区,
        (116.426539,39.968791,1800),
        (116.426699,39.969038,1900),

        (116.426699,39.969038,2600),  # 单点的cut

        (116.433925,39.969618,3000),  # 去快递站
        (116.438571,39.968236,3200),
        (116.441672,39.969749,3500),
        (116.441672,39.969749,3600),
        (116.441672,39.969749,3700),
    ]
    
    # 输入的轨迹需要是WGS坐标, 不是的话要做此转换
    traj_points_orig = [(*gcj2wgs(lat, lon)[::-1], t) for lon, lat, t in traj_points_orig]

    # 输入订单
    # 注意: 多个小哥的订单id也不能有相同的
    orders = [
        {
            "id": -1,
            "building_id": 752,
            "start_time": -1500,
            "finish_time": -500,
            "ddl_time": 0,
            "floor": 1,
            "unit": 1,
            "type": ORDER_DELIVER,
        },
        {
            "id": -2,
            "building_id": 752,
            "start_time": -1500,
            "finish_time": -500,
            "ddl_time": 0,
            "floor": 2,
            "unit": 2,
            "type": ORDER_DELIVER,
        },
        {
            "id": -3,
            "building_id": 752,
            "start_time": -600,
            "finish_time": -500,
            "ddl_time": 0,
            "floor": 2,
            "unit": 3,
            "type": ORDER_CPICK,
        },
        {
            "id": -4,
            "building_id": 752,
            "start_time": -1500,
            "finish_time": -500,
            "ddl_time": 0,
            "floor": 6,
            "unit": 3,
            "type": ORDER_DELIVER,
        },
        {
            "id": -5,
            "building_id": 753,
            "start_time": -1500,
            "finish_time": -500,
            "ddl_time": 0,
            "floor": 3,
            "unit": 1,
            "type": ORDER_DELIVER,
        },
        {
            "id": 0,
            "building_id": 750,
            "start_time": 100,
            "finish_time": 120,
            "ddl_time": 300,
            "floor": 3,
            "unit": 1,
            "type": ORDER_CPICK,
        },
        {
            "id": 1,
            "building_id": 750,
            "start_time": -1500,
            "finish_time": 125,
            "ddl_time": 1000,
            "floor": 3,
            "unit": 1,
            "type": ORDER_DELIVER,
        },
        {
            "id": 2,
            "building_id": 755,
            "start_time": -1500,
            "finish_time": 260,
            "ddl_time": 1000,
            "floor": 3,
            "unit": 1,
            "type": ORDER_DELIVER,
        },
        {
            "id": 3,
            "building_id": 755,
            "start_time": -1500,
            "finish_time": 265,
            "ddl_time": 1000,
            "floor": 3,
            "unit": 1,
            "type": ORDER_DELIVER,
        },
        {
            "id": 1000,
            "building_id": 745,
            "start_time": -1500,
            "finish_time": 600,
            "ddl_time": 1000,
            "floor": 3,
            "unit": 1,
            "type": ORDER_DELIVER,
        },
        {
            "id": 1001,
            "building_id": 736,
            "start_time": -1500,
            "finish_time": 700,
            "ddl_time": 1000,
            "floor": 3,
            "unit": 1,
            "type": ORDER_DELIVER,
        },
        {
            "id": 1002,
            "building_id": 735,
            "start_time": -1500,
            "finish_time": 800,
            "ddl_time": 1000,
            "floor": 3,
            "unit": 1,
            "type": ORDER_DELIVER,
        },
        { 
            "id": 4,
            "building_id": 721,
            "start_time": 600,
            "finish_time": 700,
            "ddl_time": 1000,
            "floor": 3,
            "unit": 1,
            "type": ORDER_BPICK,
        },
        { 
            "id": 5,
            "building_id": 755,
            "start_time": -1500,
            "finish_time": 1800,
            "ddl_time": 3000,
            "floor": 3,
            "unit": 1,
            "type": ORDER_DELIVER,
        },
        { 
            "id": 6,
            "building_id": 757,
            "start_time": -1500,
            "finish_time": 2600,
            "ddl_time": 3000,
            "floor": 3,
            "unit": 1,
            "type": ORDER_DELIVER,
        },
        {
            "id": 100,
            "building_id": 752,
            "start_time": -1500,
            "finish_time": 2700,
            "ddl_time": 3000,
            "floor": 1,
            "unit": 1,
            "type": ORDER_DELIVER,
        },
    ]
    
    # 输入IOD数据, 可以为None表示不用IOD数据
    iod_data = None

    # # 这段代码只是我调试程序使用, 正式跑时去掉
    # t_offset = 2301 + 3600 * 8
    # traj_points_orig = [(lon, lat, t + t_offset) for lon, lat, t in traj_points_orig]
    # for o in orders:
    #     o["finish_time"] += t_offset
    #     if "start_time" in o:
    #         o["start_time"] += t_offset
    #     if "ddl_time" in o:
    #         o["ddl_time"] += t_offset

    # 输入多个小哥的轨迹和订单数据
    many_courier_traj = [traj_points_orig]
    many_courier_orders = [orders]
    courier_ids = ["test"]

    # 生成action
    many_courier_actions = []
    for traj_points_orig, orders, courier_id in zip(many_courier_traj, many_courier_orders, courier_ids):
        print("courier_id:", courier_id)
        actions = main(traj_points_orig, orders, iod_data)
        many_courier_actions.append((courier_id, actions))

    # pprint_actions(actions)

    # 存结果
    pickle.dump(many_courier_actions, open("data/actions_recover.pkl", "wb"))

    # # 3D, 取部分小哥的action, 计算楼的订单状态
    # many_courier_actions = pickle.load(open("data/actions_recover_manify.pkl", "rb"))
    # target_cids = [22284086, 22346500]  # 11, 107
    # target_data = []
    # for x in many_courier_actions:
    #     if x[0] in target_cids:
    #         actions = x[1]
    #         traj_points_orig = sum([a["support_points"] for a in actions], [])
    #         traj_points_orig = list(set(traj_points_orig))
    #         traj_points_orig.sort(key=lambda x:x[-1])
    #         actions = postprocess(actions[:-1], traj_points_orig, do_refine_instation=False)
    #         target_data.append((x[0], actions))
    #         # target_data.append(x)
    # polys = [Polygon(regions[11]["boundary"]), Polygon(regions[107]["boundary"])]
    # buildings = {k: v for k, v in buildings.items() if Point(v["gate_gps"]).within(polys[0]) or Point(v["gate_gps"]).within(polys[1])}
    # bid2status = get_bd_status([x[1] for x in target_data])
    # json.dump(target_data, open("data/actions_3D_manify.json", "w"), ensure_ascii=False)
    # json.dump(bid2status, open("data/bdstatus_manify.json", "w"), ensure_ascii=False)
