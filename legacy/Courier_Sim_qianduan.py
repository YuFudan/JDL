import json
import copy
from copy import deepcopy
from collections import defaultdict, Counter
import random
import pandas as pd
import pickle
import itertools
import argparse
from utils import get_config, genDistanceMat, LK, draw, cal_solution_cost, generate_demo, cal_route_cost
import numpy as np
from networkx import shortest_path
from scipy.spatial.distance import cdist
from ALNS_CVRP import ALNS
from shapely.geometry import Polygon, Point, LineString
from eviltransform import gcj2wgs
import json
import pickle
import random
import numpy as np
from math import ceil
from pyproj import Proj
from copy import deepcopy
from pprint import pprint
from networkx import shortest_path
from collections import defaultdict, Counter
from eviltransform import gcj2wgs
from shapely.geometry import Polygon, Point, LineString

random.seed(233)

projector = Proj("+proj=utm +zone=50 +ellps=WGS84 +datum=WGS84 +units=m +no_defs")

DEMO = "2D"
# DEMO = "3D"

# 楼
if DEMO == "2D":
    buildings = pickle.load(open("data1/buildings_new.pkl", "rb"))
else:
    buildings = json.load(open("data/buildings.json"))
for bd in buildings:
    bd["poly"] = Polygon([projector(*p) for p in bd["points"]])  # 平面投影坐标
    bd["point"] = Point(bd["gate_xy"])
buildings = {bd["id"]: bd for bd in buildings}
print("buildings:", len(buildings))

# 快递站(营业部)
station_lonlats = [(116.441648, 39.969764)]
station_ids = ["快递站"]  # 路网中该快递站节点的node_id
station_names = ["民旺营业部"]
stations = {}
for (lon, lat), nid, name in zip(station_lonlats, station_ids, station_names):
    gps = gcj2wgs(lat, lon)[::-1]
    xy = projector(*gps)
    stations[nid] = {
        "id": nid,
        "name": name,
        "gps": gps,
        "xy": xy,
        "point": Point(xy)
    }
station_ids = set(station_ids)
print("stations:", len(stations))
DEFAULT_SID = "快递站"

# 路网
if DEMO == "2D":
    G = pickle.load(open("data1/G_new.pkl", "rb"))
else:
    G = pickle.load(open("data/G.pkl", "rb"))
intersections = [
    (node[0], node[1]["xy"])
    for node in G.nodes(data=True)
    if ("building" not in node[1] or node[1]["building"] == -1) and node[0] not in station_ids
]  # 所有非楼门非快递站的路网节点坐标
print("intersections:", len(intersections))

# 路区
if DEMO == "2D":
    regions = pickle.load(open("data/regions_all.pkl", "rb"))
else:
    regions = pickle.load(open("data/regions.pkl", "rb"))
for r in regions:
    r["poly"] = Polygon([projector(*p) for p in r["boundary"]])  # 平面投影坐标
regions = {r["id"]: r for r in regions}
print("regions:", len(regions))

# 订单种类
ORDER_DELIVER = "deliver"
ORDER_CPICK = "cpick"
ORDER_BPICK = "bpick"
# 动作种类
ACTION_ELEVATOR = "坐电梯"
ACTION_UPSTAIR = "爬楼"
ACTION_DOWNSTAIR = "下楼"
ACTION_TODOOR = "上门"
ACTION_DELIVER = "派件"
ACTION_CPICK = "C揽"
ACTION_BPICK = "B揽"
ACTION_WALK = "步行"
ACTION_DRIVE = "开车"  # 专指在路区开车
ACTION_TOCAR = "回车"  # 本质就是步行
ACTION_FETCH = "取货"  # 回车取货
ACTION_FROMSTATION = "去路区"
ACTION_TOSTATION = "回站"
ACTION_MEETING = "开早会"
ACTION_ARRANGE = "理货拣货"
ACTION_REST = "休息"
ACTION_DISCHARGE = "卸货"  # 进楼前在楼下理货卸货等
# 默认动作时间
T_ELEVATOR_WAIT1 = 10.0  # 等电梯时间: 进楼第一次等电梯
T_ELEVATOR_WAIT2 = 3.0  # 等电梯时间: 进楼第二次及以后等电梯, 因为一般来说小哥出电梯送完单后, 电梯还没走, 马上就可以坐了
T_ELEVATOR_FLOOR = 2.0  # 电梯移动一层的时间
T_STAIR_UP = 14.0  # 步行上楼一层时间
T_STAIR_DOWN = 6.0  # 步行下楼一层时间
T_TODOOR = 2.0  # 从楼梯/电梯口到门口时间
T_DELIVER = 6.0  # 派件每单的用时
T_BPICK = 20.0  # B揽每单的用时, 需要称重打包等
T_CPICK = 30.0  # C揽每单的用时, 需要称重扫码打包推销等
T_FETCH = 1.0  # 回车取每个货的用时
V_WALK = 1.5  # 步行移动速度
V_CAR = 8.0  # 快递站和路区间往返速度
V_CAR2 = 3.0  # 在路区里开车的速度
# 订单相关时间
T_DELIVER_ADVANCE = 60.0  # 派件单开始时间比成为target的最早时间的提前量
T_PICK_ADVANCE = 60.0  # 揽件单开始时间比成为target的最早时间的提前量
# 早会起止时间
T_MEETING_START = 7 * 3600
T_MEETING_END = 7.5 * 3600
# 回车取货的距离阈值和连续送货量阈值
TOCAR_DISGATE = 200
TOCAR_ODRGATE = 15
# 处理订单iot时的阈值
T_OVERLAP = 10.0  # 订单iot之间的最小间隔
P_SHORT = 0.6  # 调整iot时最小的压缩比例

ORDER2ACTION = {ORDER_DELIVER: ACTION_DELIVER, ORDER_CPICK: ACTION_CPICK, ORDER_BPICK: ACTION_BPICK}
ORDER2T = {ORDER_DELIVER: T_DELIVER, ORDER_CPICK: T_CPICK, ORDER_BPICK: T_BPICK}
ACTION_MOVE = {ACTION_WALK, ACTION_TOCAR, ACTION_DRIVE, ACTION_FROMSTATION, ACTION_TOSTATION}
ACTION_ORDER = {ACTION_DELIVER, ACTION_CPICK, ACTION_BPICK}
ACTION_FLOOR = {ACTION_ELEVATOR, ACTION_UPSTAIR, ACTION_DOWNSTAIR}

# 维护车所在位置, 连续送货量
global car_pos  # tuple(lon, lat) 或 int/str(node_id)
global num_continue
car_pos = None
num_continue = 0

# 是否开启回车取货的功能
ENABLE_FETCH = False

# 统计订单丢失的原因
throw_odr_match2cut = 0  # 订单时间在原始轨迹时间范围外
throw_odr_overtime = 0  # 无关联轨迹点订单, 按物理规则模拟送单, 模拟用时超过真实约束
throw_odr_iot_overlap = 0  # 有关联轨迹点的订单, 按关联轨迹点推算iot, 但多个单间iot有交叠且无法合理处理
throw_odr_iot_short = 0  # iot时间间隔过短
throw_odr_cannot_insert = 0  # 无关联轨迹点的订单无法插入有关联轨迹点的订单的iot之外的时间

def pprint_actions(actions):
    actions = deepcopy(actions)
    for action in actions:
        for k in ["gps", "xy", "support_points", "target_orders", "status"]:
            # for k in ["gps", "xy", "support_points", "status"]:
            # for k in ["gps", "xy", "support_points"]:
            action.pop(k, None)
    pprint(actions)

def find_nearest_node(xy):
    """
    找给定坐标点的最近路网节点(非楼门非快递站)
    """
    x, y = xy
    tmp = [
        [nid, (nx - x) ** 2 + (ny - y) ** 2]
        for nid, (nx, ny) in intersections
    ]
    nid, dis2 = min(tmp, key=lambda x: x[-1])
    return nid, dis2 ** 0.5

def starttime_regu(x):
    '''
    用于时间矫正
    '''
    if x == '-1' or x==-1:
        return x
    else:
        temp = pd.to_timedelta(pd.to_datetime(x) - pd.to_datetime('2022-05-05 00:00:00'))
        return temp.total_seconds()

def need_tocar(bid):
    """输入接下来要去的楼, 判断是否需要回车"""
    global car_pos, num_continue
    if not ENABLE_FETCH:
        return False
    if num_continue > TOCAR_ODRGATE:
        return True
    x1, y1 = buildings[bid]["gate_xy"]
    if isinstance(car_pos, tuple):
        x2, y2 = car_pos
    else:
        x2, y2 = G.nodes[car_pos]["xy"]
    if (x1 - x2) ** 2 + (y1 - y2) ** 2 > TOCAR_DISGATE ** 2:
        return True
    return False

def get_travel_path_t(start_nid=None, start_xy=None, end_nid=None, end_xy=None, v=V_WALK):
    """
    找od间的最短路径, od可以给node_id, 也可以给坐标
    """
    if start_nid:  # 起点是node
        onid, gps_pre, xy_pre, dis_pre = start_nid, [], [], 0
    else:  # 起点是一个坐标
        onid, dis_pre = find_nearest_node(start_xy)
        gps_pre, xy_pre = [projector(*start_xy, inverse=True)], [start_xy]
    if end_nid:  # 终点是node
        dnid, gps_post, xy_post, dis_post = end_nid, [], [], 0
    else:  # 终点是一个坐标
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
        sid, ts = max(list(sid2ts.items()), key=lambda x: x[1])
        return sid, ts
    else:
        return None, None

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
        floor_odrs = sorted(list(floor2odrs.items()), key=lambda x: x[0])
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
        if end_time > actions[-1]["end_time"]:  # 若模拟时间比约束用时更短, 在前面插入一段休息
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

    # 更新连续送货量
    global num_continue
    num_continue += len([x for x in actions if x["type"] == ACTION_DELIVER])

    return actions

def gen_bds_actions(bid_odrs, start_time):
    """
    [(bid, odrs), ...]代表小哥送单序列, 假设小哥初始已经在第一栋楼门, 按简单物理逻辑生成其行为
    """
    global car_pos, num_continue
    assert car_pos == buildings[bid_odrs[0][0]]["gate_id"]  # 检查车已在第一栋楼
    if ENABLE_FETCH:
        assert num_continue == 0  # 检查到第一栋楼时, 货已补满
    actions = []
    time_axis = start_time
    last_bid = bid_odrs[0][0]

    for i, (bid, odrs) in enumerate(bid_odrs):
        # 在楼间移动
        if i > 0:  # i == 0时, 既不需要在楼间移动, 也不用取货, 因为初始num_continue为0
            if ENABLE_FETCH and i == len(bid_odrs) - 1:  # 要去最后一栋楼时, 强制回车, 目的是把车开到最后一栋楼
                tocar = True
            else:
                tocar = need_tocar(bid)  # 中途的楼, 根据车与楼的距离以及连续送货量判断是否回车
            if tocar:
                actions += gen_tocar_actions(
                    start_time=time_axis,
                    start_bid=last_bid,
                    end_bid=bid,
                    target_odrs=odrs)
                time_axis = actions[-1]["end_time"]
            elif bid != last_bid:
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

    if not ENABLE_FETCH:
        car_pos = buildings[bid_odrs[-1][0]]["gate_id"]

    return actions

def gen_no_odr_actions(start_time, end_time=None, start_nid=None, start_xy=None, end_nid=None, end_xy=None):
    """
    对于没有订单的一段轨迹, 按简单物理逻辑生成其移动行为
    并考虑满足end_time时间约束: 若移动完成时小于end_time, 补一段休息行为, 否则直接认为移动在end_time结束
    """
    global car_pos, num_continue
    if not end_time is None:
        assert start_time < end_time
    actions = []

    action_type, v = ACTION_DRIVE, V_CAR2
    if start_nid:  # 起点是node
        # assert car_pos == start_nid  # 检查车已在起点
        if start_nid in station_ids:
            # assert end_nid not in station_ids
            action_type, v = ACTION_FROMSTATION, V_CAR
    else:  # 起点是一个坐标
        assert car_pos == tuple(start_xy)  # 检查车已在起点
    if end_nid:  # 终点是node
        if end_nid in station_ids:
            action_type, v = ACTION_TOSTATION, V_CAR
    if action_type == ACTION_DRIVE and not ENABLE_FETCH:
        action_type, v = ACTION_WALK, V_WALK

    # 开车从起点直接去终点
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

    # 更新车位置
    if end_nid:
        car_pos = end_nid
    else:
        car_pos = tuple(end_xy)

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

def gen_has_odr_actions(orders, start_time, end_time=None, start_nid=None, start_xy=None, end_nid=None, end_xy=None):
    """
    对于有订单的一段轨迹, 按简单物理逻辑生成其行为
    若发现生成的行为超出end_time时间约束, 则放弃其中订单最少的楼的所有单, 重新尝试, 直至满足时间约束或所有订单均被放弃
    """
    global throw_odr_overtime, car_pos, num_continue
    car_pos_copy = deepcopy(car_pos)
    num_continue_copy = deepcopy(num_continue)
    if not end_time is None:
        assert start_time < end_time
    actions = []

    # 按楼距起点的距离决定小哥去楼的顺序(不考虑订单完成时间, 去每栋楼时送完该楼所有订单)
    # bid2odrs = defaultdict(list)
    # for odr in orders:
    #     bid2odrs[odr["building_id"]].append(odr)
    # nid = start_nid if start_nid else find_nearest_node(start_xy)[0]
    # start_p = Point(G.nodes[nid]["xy"])
    # bid_odrs_diss = [
    #     (bid, odrs, start_p.distance(buildings[bid]["point"]))
    #     for bid, odrs in bid2odrs.items()
    # ]
    # bid_odrs_diss.sort(key=lambda x: x[-1])
    # bid_odrs = [x[:2] for x in bid_odrs_diss]

    # 直接按订单顺序访问楼
    bid_odrs = []
    for odr in orders:
        bid = odr["building_id"]
        if bid_odrs and bid_odrs[-1][0] == bid:
            bid_odrs[-1][1].append(odr)
        else:
            bid_odrs.append([bid, [odr]])

    # 从起点开车到第一栋楼
    first_bid = bid_odrs[0][0]
    if start_nid:  # 起点是node
        # assert car_pos == start_nid  # 检查车已在起点
        if start_nid in station_ids:
            action_type, v = ACTION_FROMSTATION, V_CAR
        elif ENABLE_FETCH:
            action_type, v = ACTION_DRIVE, V_CAR2
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
        assert car_pos == tuple(start_xy)  # 检查车已在起点
        if ENABLE_FETCH:
            action_type, v = ACTION_DRIVE, V_CAR2
        else:
            action_type, v = ACTION_WALK, V_WALK
        path_gps, path_xy, use_time = get_travel_path_t(
            start_xy=start_xy,
            end_nid=buildings[first_bid]["gate_id"],
            v=v)
        actions.append({
            "type": action_type,
            "start_time": start_time,
            "end_time": start_time + use_time,
            "start_xy": start_xy,
            "end_building": first_bid,
            "gps": path_gps,
            "xy": path_xy,
            "target_orders": bid_odrs[0][1],
        })
    time_axis = start_time + use_time

    # 更新车位置
    car_pos = buildings[first_bid]["gate_id"]
    # 到第一栋楼时, 补满手中的货
    if ENABLE_FETCH and num_continue > 0:
        use_time = min(num_continue, TOCAR_ODRGATE) * T_FETCH
        actions.append({
            "type": ACTION_FETCH,
            "start_time": time_axis,
            "end_time": time_axis + use_time,
            "source_building": first_bid,
            "target_building": first_bid,
            "target_orders": bid_odrs[0][1],
        })
        time_axis += use_time
        num_continue = 0

    # 在楼中送单及楼间移动的action
    actions += gen_bds_actions(
        bid_odrs=bid_odrs,
        start_time=time_axis
    )
    
    # 终点已经是最后一栋楼, 不需要移动
    assert buildings[bid_odrs[-1][0]]["gate_id"] == end_nid
    # # 从最后一栋楼开车到终点
    # last_bid = bid_odrs[-1][0]
    # assert car_pos == buildings[last_bid]["gate_id"]  # 检查车已在最后一栋楼
    # last_action_end_time = actions[-1]["end_time"]
    # if end_nid:  # 终点是node
    #     if end_nid in station_ids:
    #         action_type, v = ACTION_TOSTATION, V_CAR
    #     elif ENABLE_FETCH:
    #         action_type, v = ACTION_DRIVE, V_CAR2
    #     else:
    #         action_type, v = ACTION_WALK, V_WALK
    #     path_gps, path_xy, use_time = get_travel_path_t(
    #         start_nid=buildings[last_bid]["gate_id"],
    #         end_nid=end_nid,
    #         v=v)
    #     actions.append({
    #         "type": action_type,
    #         "start_time": last_action_end_time,
    #         "end_time": last_action_end_time + use_time,
    #         "start_building": last_bid,
    #         "end_node": end_nid,
    #         "gps": path_gps,
    #         "xy": path_xy,
    #         "target_orders": [],
    #     })
    #     # 更新车位置
    #     car_pos = end_nid
    # else:  # 终点是一个坐标
    #     if ENABLE_FETCH:
    #         action_type, v = ACTION_DRIVE, V_CAR2
    #     else:
    #         action_type, v = ACTION_WALK, V_WALK
    #     path_gps, path_xy, use_time = get_travel_path_t(
    #         start_nid=buildings[last_bid]["gate_id"],
    #         end_xy=end_xy,
    #         v=v)
    #     actions.append({
    #         "type": action_type,
    #         "start_time": last_action_end_time,
    #         "end_time": last_action_end_time + use_time,
    #         "start_building": last_bid,
    #         "end_xy": end_xy,
    #         "gps": path_gps,
    #         "xy": path_xy,
    #         "target_orders": [],
    #     })
    #     # 更新车位置
    #     car_pos = tuple(end_xy)

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
            # if p > P_SHORT:  # 若超时不多, 按比例缩放每个动作的用时
            if True:
                last_end_time = start_time
                for action in actions:
                    action["end_time"] = last_end_time + (action["end_time"] - action["start_time"]) * p
                    action["start_time"] = last_end_time
                    last_end_time = action["end_time"]

    return actions

def gen_odrs_actions(orders, start_time, end_time, start_nid=None, start_xy=None, end_nid=None, end_xy=None):
    """
    把上面两个函数包装一下
    """
    if orders:
        return gen_has_odr_actions(orders, start_time, end_time, start_nid, start_xy, end_nid, end_xy)
    else:
        return gen_no_odr_actions(start_time, end_time, start_nid, start_xy, end_nid, end_xy)

def infer_actions(cuts, nearstation_gate, sample_gap, outbd_gate, nearbd_gate, nearodr_gate, relateodr_gate):
    """
    主函数
    """
    global car_pos, num_continue
    car_pos = None
    num_continue = 0

    actions = []
    last_cut_end_time = None
    for cut in cuts:
        print("start_time:", cut["start_time"], "end_time:", cut["end_time"])
        
        # 第一波回站 到 第二波去路区 之间 在站里休息
        if last_cut_end_time and cut["start_time"] > last_cut_end_time:
            actions.append({
                "type": ACTION_REST,
                "start_time": last_cut_end_time,
                "end_time": cut["start_time"],
                "station_id": DEFAULT_SID,
                "gps": stations[DEFAULT_SID]["gps"],
                "xy": stations[DEFAULT_SID]["xy"],
                "target_orders": []
            })
        last_cut_end_time = cut["end_time"]

        # session中的行为
        odrs = cut["orders"]
        if len(odrs) == 0:  # session里面没有订单, 直接在站里休息不出去了
            actions.append({
                "type": ACTION_REST,
                "start_time": cut['start_time'],
                "end_time": cut['end_time'],
                "station_id": DEFAULT_SID,
                "gps": stations[DEFAULT_SID]["gps"],
                "xy": stations[DEFAULT_SID]["xy"],
                "target_orders": []
            })
        else:  # 直接调用fudan的gen_odrs_actions生成揽派行为
            # 需要修改gen_odrs_actions里面开头那块代码, 不要按照订单所在楼距离起点的远近决定访问顺序
            # 而是直接按odrs的顺序(已经是锦炜ALNS跑出来的顺序)
            last_bid = odrs[-1]["building_id"]

            # 计算从最后一栋楼回站的用时
            _, _, t_tostation = get_travel_path_t(
                start_nid=buildings[last_bid]["gate_id"], 
                end_nid=DEFAULT_SID, 
                v=V_CAR
            )
            assert cut["end_time"] - cut["start_time"] > t_tostation

            # 从站里出发到按订单顺序访问完所有楼
            actions += gen_has_odr_actions(
                orders=odrs, 
                start_time=cut["start_time"], 
                end_time=cut["end_time"] - t_tostation, 
                start_nid=DEFAULT_SID,
                end_nid=buildings[last_bid]["gate_id"],
            )
            
            # 从最后一栋楼回站
            actions += gen_no_odr_actions(
                start_time=cut["end_time"] - t_tostation, 
                end_time=cut["end_time"], 
                start_nid=buildings[last_bid]["gate_id"],
                end_nid=DEFAULT_SID,
            )

    return actions

def main_generate_action(start_time_morning_for_cut,end_time_morning_for_cut,start_time_norn_for_cut,end_time_norn_for_cut,orders_morning_sequence,orders_norn_sequence, do_preprocess=True):
    """程序入口"""
    global throw_odr_match2cut, throw_odr_overtime, throw_odr_iot_overlap, throw_odr_iot_short, throw_odr_cannot_insert
    throw_odr_match2cut = 0
    throw_odr_overtime = 0
    throw_odr_iot_overlap = 0
    throw_odr_iot_short = 0
    throw_odr_cannot_insert = 0

    def preprocess(orders):
        """预处理输入订单数据, 并做合法性检查"""
        # 预处理订单数据
        oids = []
        for odr in orders:
            # print(odr)
            oids.append(odr["id"])
            assert isinstance(odr["building_id"], int)
            floor = int(odr["floor"])
            # if floor == -1:  # 对于不确定楼层的订单, 根据楼是否为电梯楼随机生成
            #     print(odr["building_id"])
            #     if buildings[odr["building_id"]]["is_elevator"]:
            #
            #         floor = random.choice([6, 7, 8, 9, 10])
            #     else:
            #         floor = random.choice([1, 2, 3, 4, 5])
            # assert 1 <= floor <= 30
            odr["floor"] = floor
            unit = int(odr["unit"])
            assert 1 <= unit <= 10
            odr["unit"] = unit
            if (odr["type"] == ORDER_CPICK) or (odr["type"] == ORDER_BPICK):
                assert odr["finish_time"] > odr["start_time"]
        # assert len(oids) == len(set(oids))
        for odr in orders:  # 转成标准数据格式
            for k, v in odr.items():
                if isinstance(v, np.int64):
                    odr[k] = int(v)
                elif isinstance(v, np.float64):
                    odr[k] = float(v)
                assert isinstance(v, str) or isinstance(v, int) or isinstance(v, float)
        # 找出所有订单所在楼所在路区
        related_rids = []
        for odr in orders:
            # print(buildings)
            # print(odr["building_id"])
            related_rids.append(buildings[odr["building_id"]]["region"])
        related_rids = set(related_rids)

        return orders, related_rids

    orders_morning_sequence, related_rids = preprocess(orders_morning_sequence)
    orders_norn_sequence, related_rids = preprocess(orders_norn_sequence)

    # 将订单分配到轨迹段
    # 按时间间隔阈值将轨迹切段
    # cuts = match_order_to_cut(orders, cuts, tm_gap_gate=60)

    # 处理session起止时间异常
    if orders_morning_sequence:
        a = start_time_morning_for_cut < 12 * 3600
        b = end_time_morning_for_cut - start_time_morning_for_cut > 120 * len(orders_morning_sequence) + 500
        if not (a and b):
            start_time_morning_for_cut = 8.5 * 3600
            end_time_morning_for_cut = start_time_morning_for_cut + 150 * len(orders_morning_sequence) + 500
        if end_time_morning_for_cut - start_time_morning_for_cut > 300 * len(orders_morning_sequence) + 500:
            end_time_morning_for_cut = start_time_morning_for_cut + 300 * len(orders_morning_sequence) + 500
        end_time_morning_for_cut = min(16 * 3600, end_time_morning_for_cut)
    if orders_norn_sequence:
        a = start_time_norn_for_cut > 12 * 3600
        b = end_time_norn_for_cut - start_time_norn_for_cut > 120 * len(orders_norn_sequence) + 500
        if not (a and b):
            start_time_norn_for_cut = 17 * 3600
            end_time_norn_for_cut = start_time_norn_for_cut + 150 * len(orders_norn_sequence) + 500
        if end_time_norn_for_cut - start_time_norn_for_cut > 300 * len(orders_norn_sequence) + 500:
            end_time_norn_for_cut = start_time_norn_for_cut + 300 * len(orders_norn_sequence) + 500
    if orders_morning_sequence and orders_norn_sequence:
        print(start_time_morning_for_cut, end_time_morning_for_cut)
        print(start_time_norn_for_cut, end_time_norn_for_cut)
        assert start_time_norn_for_cut > end_time_morning_for_cut

    cuts = []
    if orders_morning_sequence:
        cuts.append({
            "type": "inter",  # head型的轨迹段, 由若干个(>0)被过滤掉的原始轨迹点, 以及末尾一个没被过滤掉的轨迹点构成
            "orders": orders_morning_sequence,
            'start_time':start_time_morning_for_cut,
            'end_time':end_time_morning_for_cut

        })
    if orders_norn_sequence:
        cuts.append({
            "type": "inter",  # head型的轨迹段, 由若干个(>0)被过滤掉的原始轨迹点, 以及末尾一个没被过滤掉的轨迹点构成
            "orders": orders_norn_sequence,
            'start_time': start_time_norn_for_cut,
            'end_time': end_time_norn_for_cut
        })

    # 推断小哥行为
    actions = infer_actions(
        cuts=cuts,
        nearstation_gate=300,
        sample_gap=10,
        outbd_gate=30,
        nearbd_gate=5,
        nearodr_gate=120,
        relateodr_gate=120
    )
    if not actions:
        return actions
    print("action num:", len(actions))
    print("action time range:", actions[-1]["end_time"] - actions[0]["start_time"])
    print("action start/end time:", actions[0]["start_time"], actions[-1]["end_time"])

    # 报告订单丢失情况
    print("throw_odr_match2cut:", throw_odr_match2cut)
    print("throw_odr_overtime:", throw_odr_overtime)
    print("throw_odr_iot_overlap:", throw_odr_iot_overlap)
    print("throw_odr_cannot_insert:", throw_odr_cannot_insert)
    print("throw_odr_iot_short:", throw_odr_iot_short)
    
    # 后处理
    def postprocess(actions):
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
                        for a in actions[i + 1:j]:
                            if a["type"] in ACTION_ORDER:
                                break
                        else:  # 吸收夹在中间的action
                            actions[i]["end_time"] = actions[j]["end_time"]
                            actions = actions[:i + 1] + actions[j + 1:]
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
            oid_ts_tochange = []  # 记录匹配不到任何营业部时段的收货时间
            for oid, t in oid_ts:
                aidx_td = []
                for i, ts, te in aidx_ts_te:
                    if ts <= t <= te:
                        aidx2ts[i].append(t)
                        break
                    else:
                        aidx_td.append((i, t - ts if t < ts else t - te))
                else:  # 收货时间未落在任何在营业部里的时段内
                    aidx_td.sort(key=lambda x: abs(x[1]))
                    for i, td in aidx_td:
                        if td < 0:
                            if i == 0:
                                assert t > 5 * 3600
                                aidx2td[i].append(td)
                                aidx2ts[i].append(t)
                                break
                            elif actions[i - 1]["type"] == ACTION_REST and actions[i - 1]["start_time"] < t:
                                aidx2td[i].append(td)
                                aidx2ts[i].append(t)
                                break
                        else:
                            if i == len(actions) - 1:
                                aidx2td[i].append(td)
                                aidx2ts[i].append(t)
                                break
                            elif actions[i + 1]["type"] == ACTION_REST and actions[i + 1]["end_time"] > t:
                                aidx2td[i].append(td)
                                aidx2ts[i].append(t)
                                break
                    else:  # 无法通过调整营业部的时段范围来使得能够匹配上的收货时间
                        oid_ts_tochange.append((oid, t))
            # 调整在营业部的时段
            for i, tds in aidx2td.items():
                a = [t for t in tds if t < 0]
                if a:
                    a = min(a)
                    if i == 0:
                        actions[i]["start_time"] += a
                    else:
                        actions[i - 1]["end_time"] += a
                        actions[i]["start_time"] += a
                a = [t for t in tds if t > 0]
                if a:
                    a = max(a)
                    if i == len(actions) - 1:
                        actions[i]["end_time"] += a
                    else:
                        actions[i]["end_time"] += a
                        actions[i + 1]["start_time"] += a
            # 调整收货时间
            oid2ts_new = {}
            aidx_range = [[i, (a["start_time"], a["end_time"])] for i, a in enumerate(actions) if
                          a["type"] == ACTION_ARRANGE]
            aidx2ts_range = {i: (min(ts), max(ts)) for i, ts in aidx2ts.items()}
            for oid, t in oid_ts_tochange:
                for i, (_, (ts, _)) in enumerate(aidx_range):
                    if ts > t:
                        break
                aidx, (ts, te) = aidx_range[i - 1]  # 无法匹配的收货时间, 改到上一次在营业部里的时间里
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
        # actions = refine_instation_actions(actions)

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
        oid2t_target = {o: min(ts) for o, ts in oid2t_target.items()}
        for action in actions:
            for o in action["target_orders"]:
                o["target_time"] = oid2t_target.get(o["id"], None)
                o["serving_time"] = oid2t_serving.get(o["id"], None)
                o["served_time"] = oid2t_served.get(o["id"], None)
                if not o["target_time"] is None:
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
            if atype == ACTION_WALK:  # 路区和站间往返不计入移动距离
                return {"traveled_length": act["length"]}
            elif atype == ACTION_UPSTAIR:  # 坐电梯或下楼不计爬楼层数
                return {"climbed_floors": act["num"]}
            elif atype == ACTION_DELIVER:
                return {
                    "delivered_orders": 1,
                    "delivered_on_time": 1,
                }
            elif atype == ACTION_CPICK:
                on_time = action["end_time"] <= action["target_orders"][0]["ddl_time"]
                return {
                    "cpicked_orders": 1,
                    "cpicked_on_time": 1 if on_time else 0,
                }
            elif atype == ACTION_BPICK:
                on_time = action["end_time"] <= action["target_orders"][0]["ddl_time"]
                return {
                    "bpicked_orders": 1,
                    "bpicked_on_time": 1 if on_time else 0,
                }
            return {}

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
        }
        for action in actions:
            action["status"] = deepcopy(vars_maintain)
            for k, v in calculate_var(action).items():
                vars_maintain[k] += v

        # 增加go_for_picking字段
        cpick_set = set()  # 所有已成为target但未完成的C揽单
        for action in actions:
            for odr in action["target_orders"]:
                if odr["type"] == ORDER_CPICK:
                    cpick_set.add(odr["id"])
            action["go_for_picking"] = True if cpick_set else False
            if action["type"] == ACTION_CPICK:
                cpick_set.remove(action["target_orders"][0]["id"])

        actions.append({
            "type": ACTION_REST,
            "start_time": actions[-1]["end_time"],
            "end_time": actions[-1]["end_time"] + 0.1,
            "status": deepcopy(vars_maintain),
            "support_points": [],
            "target_orders": [],
            "go_for_picking": False
        })

        # # 给action加上support_points字段
        # i, j = 0, 0
        # pts = [(*projector(x, y, inverse=True), t) for x, y, t in traj_points_orig]
        # for a in actions:
        #     ts, te = a["start_time"], a["end_time"]
        #     while i < len(pts):  # i是第一个t>=ts的点, 若最后一点仍<ts, i越界
        #         if pts[i][-1] < ts:
        #             i += 1
        #         else:
        #             break
        #     while j < len(pts):  # j是第一个t>te的点, 若最后一个点仍<=te, j越界
        #         if pts[j][-1] <= te:
        #             j += 1
        #         else:
        #             break
        #     a["support_points"] = pts[i:j]

        return actions

    actions = postprocess(actions)

    # # 检查行为序列合法性
    # check_results(actions)

    # 避免由于浮点数精度问题带来的action时间不首尾相接(check_results中已经检查过事实上是首尾相接)
    last_te = actions[0]["start_time"]
    for action in actions:
        action["start_time"] = last_te
        last_te = action["end_time"]

    assert len([a for a in actions if a["type"] in ACTION_ORDER]) == len(orders_morning_sequence) + len(orders_norn_sequence)

    return actions


class Agent():  # 使用 __init__函数来初始化模型
    '''
    每个快递员Agent的属性与功能
    '''

    # self 作为对象的初始参数
    def __init__(self):

        self.StartTime_Morning=None # 快递员早上开始工作时间(上班打卡)
        self.StartTime_Norn=None # 快递员下午开始工作时间
        self.StartTime_Morning_sorted = None  # 快递员早上理货完成时间
        self.StartTime_Norn_sorted = None # 快递员下午理货完成时间
        self.ArriveRoadareaTime_Morning = None  # 快递员早上到达路区的时间
        self.DepartRoadareaTime_Morning = 50400 - 30 * 60 - 10 * 60  # 快递员早上离开路区的时间
        self.ArriveRoadareaTime_Norn = None  # 快递员早上到达路区的时间
        self.DepartRoadareaTime_Norn = 72000 - 10 * 60  # 快递员早上离开路区的时间
        self.LunchTime=50400 - 30 * 60 # 快递员开始午餐时间
        self.EndTime=None # 快递员当天下班时间（下班打卡）
        self.Physical=1.0 # 快递员体力量化值
        self.Feeling=1.0 # 快递员心情量化值
        self.RoadSpeed=6 # 快递员骑车在大路上跑的速度(m/s)
        self.AreaSpeed=1.5 # 快递员在路区时楼与楼之间拖车穿梭的速度(m/s)
        self.StairSpeed=10 # 快递员上楼下楼的速度(层/s)
        self.ElevatorSpeed=2 # 快递员乘电梯上下楼的速度(层/s)
        self.HandUplimit=20 # 快递员爬楼时最多可以拿多少个快递（电梯可以推车,楼梯楼可以拿袋子）
        self.TrailerUplimit=30 # 快递员在小区内部拖车能够装货物的上限
        self.NapTime = 30  # 快递员抽烟或者休息的时间(500s，其中抽烟预计300s，剩余时间用于休息)
        self.SingleSortTime = 8  # 快递员在营业部或者小区门口整理单件货物的时间(s)
        self.order_ok_num = 0 # 快递员已完成订单数量
        self.stair_sum = 0 # 快递员总的爬楼量
        self.deal_one_time = 5 # 快递员派送和揽收一件订单的时间

    def first_layer(self, order_morning, order_noon):
        '''
        该阶段把快递员一天的时间划分为若干个时间段，包含上午打卡时间、上午理货完成时间、上午到达路区时间、上午离开路区时间、开始午餐时间、下午开始上班时间、下午理货完成时间、下午到达路区时间、下午离开路区时间、下班打卡时间
        :param order_morning:该快递员早上订单
        :param order_noon:该快递员下午订单
        :param distance_to_roadarea:营业部到该快递员所在路区距离
        :return:划分时间段
        '''
        order_num_morning = len(order_morning)
        order_num_noon = len(order_noon)
        self.DepartRoadareaTime_Morning = self.StartTime_Morning + order_num_morning * 150
        self.DepartRoadareaTime_Norn = self.StartTime_Norn + order_num_noon * 150

        return self.StartTime_Morning,self.DepartRoadareaTime_Morning,self.StartTime_Norn,self.DepartRoadareaTime_Norn

    def second_layer_generate(self, orders, station_location, building_dict,time_last_stage):
        '''
        第一阶段：该阶段输入快递员所需要处理的派件订单（以字典形式保存），初步生成快递员访问各个订单的顺序以及访问各个订单的时刻
        :param orders: list，内部元素是字典{'id','type','xiaoqu','building','floor'}，其中building标号继续采用之前整个区域内楼的编号，新增路区编号需要将每栋楼对应到对应的小区里
        :param station_location:营业部的经纬度坐标（WGS）
        :param building_dict：一个字典list，每个字典包含楼宇的building_id以及经纬度坐标
        :return:快递员访问各个订单的顺序以及访问各个订单的时刻
        '''
        total_action_list = [] # total_action_list中包含每栋楼内快递员的动作（到达，揽派件，休息），list[list]
        total_time_list = time_last_stage # total_time_list中包含每个动作的开始时间与结束时间，list
        ##### 调用ALNS算法来求解VRP，规划访问每栋楼的顺序
        # 记录该批订单中出现的不同楼编号
        building_list = [item['building_id'] for item in orders for key in item if key == 'building_id']
        building_list1 = list(set(building_list))
        building_list1.sort(key=building_list.index)
        # 记录该批订单中出现的不同楼的坐标，顺序与building_list1一致
        building_location = []
        for i in range(len(building_list1)):
            for x in building_dict:
                if x['gate_id'] == str(building_list1[i]):
                    building_location.append(x['gate_gps'])
        if len(building_location)>2:
            # 此处添加了营业部坐标（depot），用于VRP求解
            building_location_vrp = building_location
            building_location_vrp.insert(0, station_location)
            demands_vrp = [1. for i in range(len(building_location_vrp))]
            demands_vrp[0] = 0
            demands_vrp = np.array(demands_vrp)
            # 调用的ALNS算法的参数设置
            customers_num = len(building_location_vrp) - 1
            capacity = 100
            EPSILON = 0.00000000001
            destroy_size = 2
            repair_size = 2
            p = 0.7
            building_location_vrp_array = np.array(building_location_vrp)
            distance_matrix = genDistanceMat(building_location_vrp_array[:, 0],
                                             building_location_vrp_array[:, 1])  # 计算距离矩阵
            best_solution, best_cost = ALNS(distance_matrix, building_location_vrp_array, demands_vrp, customers_num,
                                            capacity, EPSILON, destroy_size, repair_size,
                                            p)  # ALNS求解的结果，形式为[[0,...,0],[0,...,0],...]，子list表示一个巡回路径从depot出发到返回depot
            building_list_element = list(itertools.chain.from_iterable(best_solution))  # 转化为一维list
            # 由于我们的情况不需要考虑容量约束，因此得到的解中如果有0则去掉
            temp_j = 0
            temp_return = 0
            for i in range(len(building_list_element)):
                if building_list_element[temp_j] == 0:
                    building_list_element.pop(temp_j)
                    temp_return = temp_return + 1
                    if temp_return > 2:
                        print('ANLS_VRP解中存在折返营业部行为，将其除去')
                else:
                    temp_j += 1
            # 由于best_solution给定的是0:n的顺序，需要对应building_id
            for i in range(len(building_list_element)):
                building_list_element[i] = building_list_element[i] - 1
            building_list1 = np.array(building_list1)
            building_list_element = building_list1[building_list_element]
            building_list_element = building_list_element.tolist()
        else:
            building_list1 = np.array(building_list1)
            temp_list = list(range(len(building_location)))
            random.shuffle(temp_list)
            building_list_element = building_list1[temp_list]
            building_list_element = building_list_element.tolist()

        # 基于ALNS得到的解，按顺序生成每栋楼的动作序列one_building_orders_result，每栋楼的动作序列为[arrive,order1,order2,...depart,Nap]，即第一个动作是arrive，之后动作有派送订单，倒二个动作时depart（起于最后一个订单结束时间，止于下楼的时间），最后一个动作是Nap
        for i in range(len(building_list_element)):
            one_building_orders = [x for x in orders if (x['building_id'] == building_list_element[i])]  # 某栋楼的所有订单
            one_building_orders_1 = sorted(one_building_orders, key=lambda e: e.__getitem__('floor')) # 将订单升序排序
            one_building_orders_result = copy.deepcopy(one_building_orders_1)

            if i == 0:
                # 计算楼与楼、营业部与楼的行驶时间：当i为0时，第i栋楼的第一个到达动作是从营业部出发到达；当i不为0时，第i栋楼的第一个到达动作是从之前的楼出发到这栋楼
                # for building_x in building_dict:
                #     if building_x['gate_id'] == str(building_list_element[i]):
                #         dd = building_x
                #         break
                _,_,path_time = get_travel_path_t(start_nid="快递站",
                                              end_nid=str(building_list_element[i]), v=self.AreaSpeed) # buildings[yingyebu_bid]["gate_id"]
            else:
                _,_,path_time = get_travel_path_t(start_nid=str(building_list_element[i-1]),
                                              end_nid=str(building_list_element[i]), v=self.AreaSpeed)
            # 将到达动作的起止时间添加到total_time_list中
            start_arrive_time = total_time_list[-1] + path_time
            end_arrive_time = total_time_list[-1] + path_time + 120  # 此处简单假设货物均划分好，重新装货时候定为2分钟
            total_time_list.append(start_arrive_time) # 到达楼宇的时刻
            total_time_list.append(end_arrive_time) # 到达楼宇结束拣货的时刻
            # 将到达动作插入到该栋楼动作序列的最前部
            one_building_orders_result.insert(0,
                {
                    "type": 'building_gate',
                    'building_id': building_list_element[i],
                    "start_time": start_arrive_time,
                    "end_time": end_arrive_time,
                    "floor": 0
                })
            one_building_orders_result.insert(1,
                                              {
                                                  "type": 'lihuo',
                                                  'building_id': building_list_element[i],
                                                  "start_time": total_time_list[-1],
                                                  "end_time": total_time_list[-1] + 600,
                                                  "floor": 0
                                              })
            total_time_list.append(total_time_list[-1])  # 到达楼宇的时刻
            total_time_list.append(total_time_list[-1] + 600)  # 到达楼宇结束拣货的时刻
            # 生成每个订单访问的起止时间
            for k in range(len(one_building_orders_result)):
                # 第一个动作是到达
                if k==0 | k==1:
                    pass
                # 第二个动作没有之前的参照订单，但只需考虑该楼层的上楼时间
                elif k==2:
                    dd = None
                    for building_x in building_dict:
                        if building_x['gate_id'] == str(building_list_element[i]):
                            dd = building_x
                    if dd['is_elevator']:
                        time_step = abs(one_building_orders_result[k]['floor'] - 0) * self.ElevatorSpeed
                    else:
                        time_step = abs(one_building_orders_result[k]['floor'] - 0) * self.StairSpeed
                    start_action_time = total_time_list[-1] + time_step
                    end_action_time = total_time_list[-1] + time_step + self.deal_one_time
                    one_building_orders_result[k]['start_time'] = start_action_time
                    one_building_orders_result[k]['end_time'] = end_action_time
                    total_time_list.append(start_action_time)
                    total_time_list.append(end_action_time)
                # 后续订单的起止时间计算
                else:
                    for building_x in building_dict:
                        if building_x['gate_id'] == str(building_list_element[i]):
                            dd = building_x
                            break
                    if dd['is_elevator']:
                        time_step = abs(one_building_orders_result[k]['floor'] - one_building_orders_result[k - 1][
                            'floor']) * self.ElevatorSpeed
                    else:
                        time_step = abs(one_building_orders_result[k]['floor'] - one_building_orders_result[k - 1][
                            'floor']) * self.StairSpeed
                    start_action_time = total_time_list[-1] + time_step
                    end_action_time = total_time_list[-1] + time_step + self.deal_one_time
                    one_building_orders_result[k]['start_time'] = start_action_time
                    one_building_orders_result[k]['end_time'] = end_action_time
                    total_time_list.append(start_action_time)
                    total_time_list.append(end_action_time)

            # 最后一个订单结束后还需一个下楼时间
            start_depart_time = one_building_orders_result[-1]['end_time']
            for building_x in building_dict:
                if building_x['gate_id'] == str(building_list_element[i]):
                    dd = building_x
                    break
            if dd['is_elevator']:
                time_depart_step = abs(one_building_orders_result[-1]['floor'] - 0) * self.ElevatorSpeed
            else:
                time_depart_step = abs(one_building_orders_result[-1]['floor'] - 0) * self.StairSpeed
            end_depart_time = start_depart_time + time_depart_step
            total_time_list.append(start_depart_time)
            total_time_list.append(end_depart_time)
            # 将下楼离开动作插入到当前该栋楼动作序列的最尾部
            one_building_orders_result.append({
                                                  "type": 'building_gate',
                                                  'building_id': building_list_element[i],
                                                  "start_time": start_depart_time,
                                                  "end_time": end_depart_time,
                                                  "floor": 0
                                              })

            self.order_ok_num = self.order_ok_num + len(one_building_orders_result) - 1
            self.stair_sum = self.stair_sum + sum(z['floor'] for z in one_building_orders_result)
            Nap = self.Judge_Nap()
            if Nap:
                start_action_time = total_time_list[-1]
                end_action_time = start_action_time + self.NapTime
                one_building_orders_result.append(
                    {
                        "type": 'nap',
                        "nap": True,
                        'building_id': building_list_element[i],
                        "start_time": start_action_time,
                        "end_time": end_action_time,
                        "floor": 0
                    })
                total_time_list.append(start_action_time)
                total_time_list.append(end_action_time)
                self.order_ok_num = 0
                self.stair_sum = 0
                self.Physical = 1
                self.Feeling = 1
            # 即便没有休息也增设该动作，用于占位，完整序列生成后也可删除
            else:
                start_action_time = total_time_list[-1]
                end_action_time = start_action_time
                one_building_orders_result.append(
                    {
                        "type": 'nap',
                        "nap": False,
                        'building_id': building_list_element[i],
                        "start_time": start_action_time,
                        "end_time": end_action_time,
                        "floor": 0
                    })  # 1000000000表示休息
                total_time_list.append(start_action_time)
                total_time_list.append(end_action_time)
            if len(one_building_orders_result)>3:
                total_action_list.append(one_building_orders_result)

        total_action_list_element = list(itertools.chain.from_iterable(total_action_list)) # 转化为一维list

        # print(total_action_list)

        return total_action_list,total_time_list,total_action_list_element, building_list_element

    def second_layer_adjust(self,total_action_list,total_time_list,pickup_order,building_list_element, building_dict):
        '''
        第二阶段：基于已生成的动作序列以及实时生成的揽收订单，调整动作序列（每次有新的揽收订单生成就立即执行）
        :param total_action_list: 已生成的动作序列
        :param pickup_order: 实时产生的揽收订单
        :return: 调整后的动作序列
        '''
        # 第二阶段：基于实时揽收订单的情况来调整行动序列
        for i in range(len(total_action_list)):
            # 找到揽件订单生成时 快递员在送哪栋楼
            if (pickup_order['start_time'] <= total_action_list[i][-1]['end_time']) & (
                    pickup_order['start_time'] >= total_action_list[i][0]['start_time']):
                # 找到揽件订单不在派件订单的楼宇群中，则新增
                if pickup_order['building_id'] not in building_list_element:
                    # 在第i栋楼list后再新增一个访问楼list
                    # print(i)
                    # print(building_list_element)
                    _,_,path_time = get_travel_path_t(start_nid=str(total_action_list[i][-1]['building_id']),
                                                  end_nid=str(pickup_order['building_id']), v=self.AreaSpeed)
                    start_arrive_time = total_action_list[i][-1]['end_time'] + path_time
                    # print('total_action_list')
                    # print(total_action_list)
                    end_arrive_time = total_action_list[i][-1]['end_time'] + path_time + 120  # 此处简单假设货物均划分好，重新装货时候定为2分钟
                    # 将到达动作插入到新增楼动作序列的开始
                    total_action_list.insert(i + 1, [
                        {
                            "type": 'building_gate',
                            'building_id': pickup_order['building_id'],
                            'floor':0,
                            "start_time": start_arrive_time,
                            "end_time": end_arrive_time,
                        }])
                    for building_x in building_dict:
                        if building_x['gate_id'] == str(pickup_order['building_id']):
                            dd = building_x
                            break
                    if dd['is_elevator']: # 订单中需要有是否电梯房的信息
                        up_stair_time = abs(pickup_order['floor']) * self.ElevatorSpeed
                    else:
                        up_stair_time = abs(pickup_order['floor']) * self.StairSpeed
                    temp_pickup_order = pickup_order
                    start_action_time = total_action_list[i + 1][0]['end_time'] + up_stair_time
                    end_action_time = total_action_list[i + 1][0][
                                          'end_time'] + up_stair_time + self.deal_one_time
                    temp_pickup_order['start_time'] = start_action_time
                    temp_pickup_order['end_time'] = end_action_time
                    total_action_list[i + 1].append(temp_pickup_order)

                    # 最后一个订单结束后还需一个下楼时间
                    start_depart_time = total_action_list[i + 1][-1]['end_time']
                    for building_x in building_dict:
                        if building_x['gate_id'] == str(pickup_order['building_id']):
                            dd = building_x
                            break
                    if dd['is_elevator']: # 订单中需要有是否电梯房的信息
                        time_depart_step = abs(total_action_list[i + 1][-1]['floor'] - 0) * self.ElevatorSpeed
                    else:
                        time_depart_step = abs(total_action_list[i + 1][-1]['floor'] - 0) * self.StairSpeed
                    end_depart_time = start_depart_time + time_depart_step
                    # 将下楼离开动作插入到当前该栋楼动作序列的最尾部
                    total_action_list[i + 1].append({
                        "type": 'building_gate',
                        'building_id': pickup_order['building_id'],
                        'floor':0,
                        "start_time": start_depart_time,
                        "end_time": end_depart_time,
                    })

                    self.order_ok_num = self.order_ok_num + len(total_action_list[i + 1]) - 1
                    self.stair_sum = self.stair_sum + sum(z['floor'] for z in total_action_list[i + 1])
                    Nap = self.Judge_Nap()
                    if Nap:
                        start_action_time = total_action_list[i + 1][-1]['end_time']
                        end_action_time = start_action_time + self.NapTime
                        total_action_list[i + 1].append(
                            {
                                "type": 'nap',
                                "nap": True,
                                'floor':0,
                                'building_id': pickup_order['building_id'],
                                "start_time": start_action_time,
                                "end_time": end_action_time,
                            })
                    # 即便没有休息也增设该动作，用于占位，完整序列生成后也可删除
                    else:
                        start_action_time = total_action_list[i + 1][-1]['end_time']
                        end_action_time = start_action_time
                        total_action_list[i + 1].append(
                            {
                                "type": 'nap',
                                "nap": False,
                                'floor':0,
                                'building_id': pickup_order['building_id'],
                                "start_time": start_action_time,
                                "end_time": end_action_time,
                            })
                    # 对后续动作时间补充
                    if (i+2<len(total_action_list)):
                        _, _, path_time1 = get_travel_path_t(start_nid=str(total_action_list[i][1]['building_id']),
                                                         end_nid=str(total_action_list[i + 2][1]['building_id']),
                                                         v=self.AreaSpeed)
                        _, _, path_time2 = get_travel_path_t(start_nid=str(total_action_list[i + 1][1]['building_id']),
                                                         end_nid=str(total_action_list[i + 2][1]['building_id']),
                                                         v=self.AreaSpeed)
                        time_incre_tail = total_action_list[i + 1][-1]['end_time'] - total_action_list[i][-1][
                        'end_time'] + path_time2 - path_time1  # 暂时先按照这个来算
                        for i1 in range(i + 2, len(total_action_list)):
                            for x in total_action_list[i1]:
                                x['start_time'] = x['start_time'] + time_incre_tail
                                x['end_time'] = x['end_time'] + time_incre_tail
                    else:
                        pass
                # 找到揽件订单生成时快递员在送第i栋楼，若揽件订单与此时快递员同一栋楼则直接插入该楼订单尾部
                elif pickup_order['building_id'] == total_action_list[i][1]['building_id']:
                    for building_x in building_dict:
                        if building_x['gate_id'] == str(pickup_order['building_id']):
                            dd = building_x
                            break
                    if dd['is_elevator']: # 订单中需要有是否电梯房的信息
                        up_stair_time = abs(
                            pickup_order['floor'] - total_action_list[i][-3]['floor']) * self.ElevatorSpeed
                        time_incre_tail = abs(
                            pickup_order['floor'] - total_action_list[i][-3]['floor']) * self.ElevatorSpeed + total_action_list[i][-3]['floor'] * self.ElevatorSpeed + self.deal_one_time
                    else:
                        up_stair_time = abs(
                            pickup_order['floor'] - total_action_list[i][-3]['floor']) * self.StairSpeed
                        time_incre_tail = abs(
                            pickup_order['floor'] - total_action_list[i][-3]['floor']) * self.StairSpeed + total_action_list[i][-3]['floor'] * self.StairSpeed + self.deal_one_time
                    temp_pickup_order = pickup_order
                    start_action_time = total_action_list[i][-3]['end_time'] + up_stair_time
                    end_action_time = total_action_list[i][-3]['end_time'] + up_stair_time + self.deal_one_time
                    # time_incre_tail = up_stair_time + self.deal_one_time
                    temp_pickup_order['start_time'] = start_action_time
                    temp_pickup_order['end_time'] = end_action_time
                    total_action_list[i].insert(-2,temp_pickup_order)
                    # 对后续动作时间补充
                    total_action_list[i][-2]['start_time'] = total_action_list[i][-2]['start_time'] + time_incre_tail
                    total_action_list[i][-2]['end_time'] = total_action_list[i][-2]['end_time'] + time_incre_tail
                    total_action_list[i][-1]['start_time'] = total_action_list[i][-1]['start_time'] + time_incre_tail
                    total_action_list[i][-1]['end_time'] = total_action_list[i][-1]['end_time'] + time_incre_tail
                    for i1 in range(i + 1,len(total_action_list)):
                        for x in total_action_list[i1]:
                            x['start_time'] = x['start_time'] + time_incre_tail
                            x['end_time'] = x['end_time'] + time_incre_tail
                # 找到揽件订单生成时快递员在送第i栋楼且该订单所在楼快递员之前并未访问过，将该楼顺序提前：即送完第i栋楼则送揽件所在楼，且揽件订单插入到该楼揽派件任务尾部
                elif (pickup_order['building_id'] != total_action_list[i][1]['building_id']) & (building_list_element.index(pickup_order['building_id']) > i):
                    for building_x in building_dict:
                        if building_x['gate_id'] == str(pickup_order['building_id']):
                            dd = building_x
                            break
                    if dd['is_elevator']: # 订单中需要有是否电梯房的信息
                    # if total_action_list[building_list_element.index(pickup_order['building_id'])][1]['is_elevator']:
                        up_stair_time = abs(
                            total_action_list[building_list_element.index(pickup_order['building_id'])][-3]['floor'] - pickup_order['floor']) * self.ElevatorSpeed
                        time_incre_tail = abs(
                            pickup_order['floor'] - total_action_list[building_list_element.index(pickup_order['building_id'])][-3]['floor']) * self.ElevatorSpeed + \
                                          pickup_order['floor'] * self.ElevatorSpeed - total_action_list[building_list_element.index(pickup_order['building_id'])][-3]['floor'] * self.ElevatorSpeed
                    else:
                        up_stair_time = abs(
                            total_action_list[building_list_element.index(pickup_order['building_id'])][-3]['floor'] - pickup_order['floor']) * self.StairSpeed
                        time_incre_tail = abs(
                            pickup_order['floor'] - total_action_list[building_list_element.index(pickup_order['building_id'])][-3]['floor']) * self.ElevatorSpeed + \
                                          pickup_order['floor'] * self.ElevatorSpeed - total_action_list[building_list_element.index(pickup_order['building_id'])][-3]['floor'] * self.ElevatorSpeed
                    temp_pickup_order = pickup_order
                    # time_incre_tail = up_stair_time + self.deal_one_time
                    temp_pickup_order['start_time'] = total_action_list[building_list_element.index(pickup_order['building_id'])][-3]['end_time'] + up_stair_time
                    temp_pickup_order['end_time'] = total_action_list[building_list_element.index(pickup_order['building_id'])][-3]['end_time'] + up_stair_time + self.deal_one_time
                    # 在该楼的揽派件订单序列末尾插入揽件订单
                    total_action_list[building_list_element.index(pickup_order['building_id'])].insert(-2, temp_pickup_order)
                    # 对后续动作时间补充
                    total_action_list[building_list_element.index(pickup_order['building_id'])][-2]['start_time'] = total_action_list[building_list_element.index(pickup_order['building_id'])][-2]['start_time'] + time_incre_tail
                    total_action_list[building_list_element.index(pickup_order['building_id'])][-2]['end_time'] = total_action_list[building_list_element.index(pickup_order['building_id'])][-2]['end_time'] + time_incre_tail
                    total_action_list[building_list_element.index(pickup_order['building_id'])][-1]['start_time'] = total_action_list[building_list_element.index(pickup_order['building_id'])][-1]['start_time'] + time_incre_tail
                    total_action_list[building_list_element.index(pickup_order['building_id'])][-1]['end_time'] = total_action_list[building_list_element.index(pickup_order['building_id'])][-1]['end_time'] + time_incre_tail
                    for i1 in range(building_list_element.index(pickup_order['building_id']) + 1, len(total_action_list)):
                        for x in total_action_list[i1]:
                            x['start_time'] = x['start_time'] + time_incre_tail
                            x['end_time'] = x['end_time'] + time_incre_tail
                # 找到揽件订单生成时快递员在送第i栋楼且该订单所在楼快递员之前已访问过，则重新前往该楼访问
                elif (pickup_order['building_id'] != total_action_list[i][1]['building_id']) & (building_list_element.index(pickup_order['building_id']) < i):
                    # 在第i栋楼list后再新增一个访问楼list
                    print(i)
                    print(len(total_action_list[i]))
                    print(str(total_action_list[i][1]['building_id']))
                    print(pickup_order['building_id'])
                    _,_,path_time = get_travel_path_t(start_nid=str(total_action_list[i][1]['building_id']),
                                                  end_nid=str(pickup_order['building_id']), v=self.AreaSpeed)
                    start_arrive_time = total_action_list[i][-1]['end_time'] + path_time
                    end_arrive_time = total_action_list[i][-1]['end_time'] + path_time + 120  # 此处简单假设货物均划分好，重新装货时候定为2分钟
                    # 将到达动作插入到新增楼动作序列的最尾部
                    total_action_list.insert(i+1,[
                        {
                                                          "type": 'building_gate',
                                                          'building_id': total_action_list[i][1]['building_id'],
                                                          'floor':0,
                                                          "start_time": start_arrive_time,
                                                          "end_time": end_arrive_time,
                                                      }])
                    for building_x in building_dict:
                        if building_x['gate_id'] == str(pickup_order['building_id']):
                            dd = building_x
                            break
                    if dd['is_elevator']: # 订单中需要有是否电梯房的信息
                    # if total_action_list[building_list_element.index(pickup_order['building_id'])][1]['is_elevator']:
                        up_stair_time = abs(pickup_order['floor']) * self.ElevatorSpeed
                    else:
                        up_stair_time = abs(pickup_order['floor']) * self.StairSpeed
                    temp_pickup_order = pickup_order
                    start_action_time = total_action_list[i+1][0]['end_time'] + up_stair_time
                    end_action_time = total_action_list[i+1][0]['end_time'] + up_stair_time + self.deal_one_time
                    temp_pickup_order['start_time'] = start_action_time
                    temp_pickup_order['end_time'] = end_action_time
                    total_action_list[i+1].append(temp_pickup_order)

                    # 最后一个订单结束后还需一个下楼时间
                    start_depart_time = total_action_list[i+1][-1]['end_time']
                    for building_x in building_dict:
                        if building_x['gate_id'] == str(pickup_order['building_id']):
                            dd = building_x
                            break
                    if dd['is_elevator']: # 订单中需要有是否电梯房的信息
                    # if total_action_list[i+1][-1]['is_elevator']:
                        time_depart_step = abs(total_action_list[i+1][-1]['floor'] - 0) * self.ElevatorSpeed
                    else:
                        time_depart_step = abs(total_action_list[i+1][-1]['floor'] - 0) * self.StairSpeed
                    end_depart_time = start_depart_time + time_depart_step
                    # 将下楼离开动作插入到当前该栋楼动作序列的最尾部
                    total_action_list[i+1].append({
                        "type": 'building_gate',
                        'building_id': 1,
                        'floor':0,
                        "start_time": start_depart_time,
                        "end_time": end_depart_time,
                    })

                    self.order_ok_num = self.order_ok_num + len(total_action_list[i+1]) - 1
                    self.stair_sum = self.stair_sum + sum(z['floor'] for z in total_action_list[i+1])
                    Nap = self.Judge_Nap()
                    if Nap:
                        start_action_time = total_action_list[i+1][-1]['end_time']
                        end_action_time = start_action_time + self.NapTime
                        total_action_list[i+1].append(
                            {
                                "type": 'nap',
                                "nap": False,
                                'building_id': building_list_element[i],
                                'floor':0,
                                "start_time": start_action_time,
                                "end_time": end_action_time,
                            })
                    # 即便没有休息也增设该动作，用于占位，完整序列生成后也可删除
                    else:
                        start_action_time = total_action_list[i+1][-1]['end_time']
                        end_action_time = start_action_time
                        total_action_list[i+1].append(
                            {
                                "type": 'nap',
                                "nap": False,
                                'building_id': 1,
                                'floor':0,
                                "start_time": start_action_time,
                                "end_time": end_action_time,
                            })
                        # 对后续动作时间补充
                    if (i+2<len(total_action_list)):
                        _, _, path_time1 = get_travel_path_t(start_nid=str(total_action_list[i][1]['building_id']),
                                                             end_nid=str(total_action_list[i + 2][1]['building_id']),
                                                             v=self.AreaSpeed)
                        _, _, path_time2 = get_travel_path_t(start_nid=str(total_action_list[i + 1][1]['building_id']),
                                                             end_nid=str(total_action_list[i + 2][1]['building_id']),
                                                             v=self.AreaSpeed)
                        time_incre_tail = total_action_list[i + 1][-1]['end_time'] - total_action_list[i][-1][
                            'end_time'] + path_time2 - path_time1  # 暂时先按照这个来算，暂不考虑a-c-b对a-b的路程时间影响
                        for i1 in range(i + 2, len(total_action_list)):
                            for x in total_action_list[i1]:
                                x['start_time'] = x['start_time'] + time_incre_tail
                                x['end_time'] = x['end_time'] + time_incre_tail
                    else:
                        pass
                else:
                    pass
            else:
                pass

        return total_action_list

    def Judge_Nap(self):
        '''
        基于完成的订单量来计算当前的快递员心情量化值，基于快递员爬楼（楼梯楼）的总量来计算快递员体力的量化值，两个量化值乘积如果小于某一阈值则休息，否则不休息
        :return:目前是否休息
        '''
        self.Feeling = 1 - 0.5 * self.order_ok_num
        self.Physical = 1 - 0.05 * self.stair_sum
        if self.Feeling * self.Physical <=0.01:
            return True
        else:
            return False

def main_order(orders_used_deliver,orders_used_pickup,start_time_morning,start_time_norn,station_location, buildings_order_use,session_num_ratio):
    '''
    实现：1、基于订单确定大概时间段；2、生成工作序列（包括调整工作序列）
    :param orders:
    :param orders_pickup:
    :param start_time_morning: 上班打卡时间
    :param start_time_norn: 下班打卡时间
    :param station_location: 营业部经纬度坐标
    :param buildings_order_use: 楼宇的信息
    :param distance_to_area: 营业部前往路区的距离
    :return:
    '''
    # 基于订单划分上午订单和下午订单，基于楼宇的顺序（规则即上午访问一批楼，下午访问另一批楼），注意：上午的揽件订单不一定出现在上午访问的楼中，同理下午也是
    orders_morning_session_ratio = session_num_ratio
    print(orders_used_deliver)
    orders_num = len(orders_used_deliver) # 一个快递员一天所有的派送订单
    orders_num_morning = round(orders_num * orders_morning_session_ratio)
    orders_pickup_num = len(orders_used_pickup)  # 一个快递员一天所有的派送订单
    orders_pickup_num_morning = round(orders_pickup_num * orders_morning_session_ratio)
    orders_used_deliver.sort(key=lambda keys: keys.get('building_id')) # 先基于楼宇划分，尽可能早上和下午处理的楼宇不一样，避免重复跑
    orders_sort_building = copy.deepcopy(orders_used_deliver)
    orders_morning = orders_sort_building[:orders_num_morning] # 上午处理的派件订单
    orders_noon = orders_sort_building[orders_num_morning:] # 下午处理的派件订单
    orders_used_pickup.sort(key=lambda keys: keys.get('building_id'))  # 先基于楼宇划分，尽可能早上和下午处理的楼宇不一样，避免重复跑
    orders_pickup_sort_building = copy.deepcopy(orders_used_pickup)
    orders_pickup_morning = orders_pickup_sort_building[:orders_num_morning]  # 上午处理的揽件订单
    orders_pickup_noon = orders_pickup_sort_building[orders_pickup_num_morning:]  # 下午处理的揽件订单

    # 初始化Agent的部分参数
    courier_agent = Agent()
    # courier_agent.StairSpeed = stair_time1
    # courier_agent.ElevatorSpeed = stair_time1
    courier_agent.StartTime_Morning = start_time_morning
    courier_agent.StartTime_Norn = start_time_norn
    courier_agent.StartTime_Morning, courier_agent.DepartRoadareaTime_Morning,courier_agent.ArriveRoadareaTime_Norn, courier_agent.DepartRoadareaTime_Norn = courier_agent.first_layer(orders_morning,orders_noon)

    whole_working_stages_list = [[courier_agent.StartTime_Morning, courier_agent.DepartRoadareaTime_Morning], # 打卡到清点货物
                                 [courier_agent.StartTime_Norn,courier_agent.DepartRoadareaTime_Norn]] # 下午工作

    orders_morning_sequence = []
    orders_norn_sequence = []

    # 局部计算指标
    one_courier_session = []

    # 针对最上层每个部分，形成具体的动作与时间，并调整时间段
    whole_working_stages_action_list = []
    each_courier_time_list = []
    for i in range(len(whole_working_stages_list)):

        if i == 0:
            # 初步生成派送序列
            time_last_stage = [whole_working_stages_list[i][0], whole_working_stages_list[i][0]]
            if len(orders_morning)>0:
                total_action_list,total_time_list,total_action_list_element,building_list_element = courier_agent.second_layer_generate(orders_morning, station_location,buildings_order_use,time_last_stage)
            else:
                total_action_list = []
                total_time_list = []
                total_action_list_element = []
                building_list_element = []
            # 将揽件订单逐步插入序列
            for i11 in range(len(orders_pickup_morning)):
                total_action_list = courier_agent.second_layer_adjust(total_action_list, total_time_list,orders_pickup_morning[i11],building_list_element,buildings_order_use)
            total_action_list_element_morning = list(itertools.chain.from_iterable(total_action_list))  # 转化为一维list
            whole_working_stages_action_list = whole_working_stages_action_list + total_action_list_element_morning
            # 计算工作时间段，并填入session
            if len(whole_working_stages_action_list)>0:
                temp = [whole_working_stages_list[i][0], whole_working_stages_action_list[-1]['end_time']]
            else:
                temp = [whole_working_stages_list[i][0], whole_working_stages_list[i][1]]
            each_courier_time_list.append(temp)
            one_courier_session.append(temp[0])
            one_courier_session.append(temp[1])
            # 计算时间节点 用于满足前端约束
            if total_action_list_element_morning:
                start_time_morning_for_cut = total_action_list_element_morning[0]['start_time']
            else:
                start_time_morning_for_cut = 0
            if total_action_list_element_morning:
                end_time_morning_for_cut = total_action_list_element_morning[-1]['end_time']
            else:
                end_time_morning_for_cut = 0.0000001
            # 生成早上的订单序列（仅订单）以及计算早上处理的订单量
            for j in range(len(total_action_list_element_morning)):
                if 'id' in total_action_list_element_morning[j]:
                    orders_morning_sequence.append(total_action_list_element_morning[j])
            session_express_num_morning = 0
            for x in total_action_list_element_morning:
                if 'id' in x:
                    session_express_num_morning = session_express_num_morning + 1

        elif i == 1:
            # 初步生成派送序列
            time_last_stage = [whole_working_stages_list[i][0], whole_working_stages_list[i][0]]
            if len(orders_noon) > 0:
                total_action_list, total_time_list, total_action_list_element,building_list_element = courier_agent.second_layer_generate(orders_noon,station_location,buildings_order_use,time_last_stage)
            else:
                total_action_list = []
                total_time_list = []
                total_action_list_element = []
                building_list_element = []
            # 将揽件订单逐步插入序列
            for i111 in range(len(orders_pickup_noon)):
                total_action_list = courier_agent.second_layer_adjust(total_action_list, total_time_list, orders_pickup_noon[i111],building_list_element,buildings_order_use)
            total_action_list_element_norn = list(itertools.chain.from_iterable(total_action_list))  # 转化为一维list
            whole_working_stages_action_list = whole_working_stages_action_list + total_action_list_element_norn
            # 计算时间段
            if len(whole_working_stages_action_list)>0:
                temp1 = [whole_working_stages_list[i][0],
                    whole_working_stages_action_list[-1]['end_time']]
            else:
                temp1 = [whole_working_stages_list[i][0], whole_working_stages_list[i][1]]
            each_courier_time_list.append(temp1)
            one_courier_session.append(temp1[0])
            one_courier_session.append(temp1[1])
            # 提取订单顺序
            for j in range(len(total_action_list_element_norn)):
                if 'id' in total_action_list_element_norn[j]:
                    orders_norn_sequence.append(total_action_list_element_norn[j])
            # 计算时间节点 用于满足前端约束
            if total_action_list_element_norn:
                start_time_norn_for_cut = total_action_list_element_norn[0]['start_time']
            else:
                start_time_norn_for_cut = 0
            if total_action_list_element_norn:
                end_time_norn_for_cut = total_action_list_element_norn[-1]['end_time']
            else:
                end_time_norn_for_cut = 0.000000001
            # 计算时间段内订单数量
            session_express_num_noon = 0
            for x in total_action_list_element_norn:
                if 'id' in x:
                    session_express_num_noon = session_express_num_noon + 1
        else:
            pass

    one_courier_session_num = [session_express_num_morning, session_express_num_noon]
    
    # 后处理, 处理orders_morning_sequence和orders_norn_sequence中存在的重复订单
    oid_set = set()
    orders_morning_sequence_new = []
    for o in orders_morning_sequence:
        if o["id"] not in oid_set:
            orders_morning_sequence_new.append(o)
            oid_set.add(o["id"])
    orders_norn_sequence_new = []
    for o in orders_norn_sequence:
        if o["id"] not in oid_set:
            orders_norn_sequence_new.append(o)
            oid_set.add(o["id"])
    orders_morning_sequence = orders_morning_sequence_new
    orders_norn_sequence = orders_norn_sequence_new

    # 后处理, 补上丢失的订单
    oid_set = set(o["id"] for o in orders_morning_sequence + orders_norn_sequence)
    for o in orders_used_deliver + orders_used_pickup:
        if o["id"] not in oid_set:
            bid = o["building_id"]
            if o["finish_time"] < 15 * 3600:
                for i, o2 in enumerate(orders_morning_sequence):  # 插入到上午的单的在相同的楼的单后, 没有相同楼的直接放在最后
                    if o2["building_id"] == bid:
                        break
                orders_morning_sequence = orders_morning_sequence[:i] + [o] + orders_morning_sequence[i:]
            else:
                for i, o2 in enumerate(orders_norn_sequence):  # 插入到下午的单的在相同的楼的单后, 没有相同楼的直接放在最后
                    if o2["building_id"] == bid:
                        break
                orders_norn_sequence = orders_norn_sequence[:i] + [o] + orders_norn_sequence[i:]
    assert len(orders_used_deliver) + len(orders_used_pickup) == len(orders_morning_sequence) + len(orders_norn_sequence)

    return (start_time_morning_for_cut, end_time_morning_for_cut, start_time_norn_for_cut, end_time_norn_for_cut,
            orders_morning_sequence, orders_norn_sequence, one_courier_session, one_courier_session_num)

if __name__ == "__main__":
    # 主程序
    # 民旺所有快递员编号
    operator_0505 = [22346500, 20570125, 22102543, 20470295, 20479259, 292917, 22284086, 22346293, 21387966, 25793,
                     21224130, 21224135, 22346312, 34645, 21357149, 185567, 22346603, 291308, 5101, 21495662, 20039663,
                     6128, 20937077, 20101500]
    operator_name = ['李占友','詹科生','郑路萍','潘小龙','朱宁杰','王鹏','郝兴海','王新新','于龙','张建明',
                     '李仲鑫','王俊瑛','李忠庆','纪革文','刘光军','冯立强','张双双','梁延长','高振省','尹立川','赵磊',
                     '李明新','富海波','王万友']
    
    # stair_time = [107.66765873015873, 103.67795044036424, 129.4920634920635, 181.70634920634922, 122.66865079365078, 73.61583522297808, 52.12993506493507, 225.02555555555554, 69.44478624376579, 237.8124999999998, 73.29537037037038, 39.462344429511944, 177.36907944996182, 301.08051948051946, 39.462344429511944, 39.462344429511944, 234.47329931972791, 66.9257320022026, 195.60833333333335, 115.42857142857143, 59.52750582750583, 78.92542306178669, 118.59368728334246, 121.89009834368531]
    # 从fudan代码中提取出到达路区时间
    with open("data1/actions_test_2D_nofetch.json", 'r', encoding='utf8') as fp:
        file_data_action_2D = json.load(fp)
    type_start_morning_list = []
    type_end_morning_list = []
    for i in range(len(file_data_action_2D)):
        for x in file_data_action_2D[i][1]:
            if x['type'] == '去路区':
                type_start_morning_list.append(x['end_time'])
            if x['type'] == '回站':
                type_end_morning_list.append(x['start_time'])
    many_courier_orders_deliver = []
    many_courier_orders_pickup = []
    # 快递员上下午开始上班时间以及订单量

    session_list_true = open('data/cjw0902_session开始结束时间_only0505.pkl', 'rb')
    session_express_num_true = open('data/cjw0902_5月5日上下午单量.pkl', 'rb')
    session_list_true = pickle.load(session_list_true)
    session_express_num_true = pickle.load(session_express_num_true)
    session_list_true_list = []
    session_express_num_true_list = []
    for k, v in session_list_true.items():
        session_list_true_list.append(v)
    for k, v in session_express_num_true.items():
        session_express_num_true_list.append(v)
    session_list_true_list = np.array(session_list_true_list)
    session_list_true_list = session_list_true_list - 300.
    session_express_num_true_list = np.array(session_express_num_true_list)
    session_express_num_ratio = np.divide(session_express_num_true_list[:, 0], np.sum(session_express_num_true_list, 1))
    session_express_num_ratio = np.nan_to_num(session_express_num_ratio)

    # 计算指标
    session_list = []
    session_express_num = []
    number_in_slice_all = []

    # 订单在快递员请假情况下重分配
    data_orders_origin = open('data/cjw0922_many_courier_orders_manify_replan.pkl', 'rb')
    data_orders_origin = pickle.load(data_orders_origin)
    #data_orders_origin = data_orders_origin[1]
    for courier_all_num in data_orders_origin:
        one_courier_deliver = []
        one_courier_pickup = []
        for courier_package in courier_all_num:
            if courier_package:
                if courier_package['type'] == 'deliver':
                    one_courier_deliver.append(courier_package)
                else:
                    one_courier_pickup.append(courier_package)
        many_courier_orders_deliver.append(one_courier_deliver)
        many_courier_orders_pickup.append(one_courier_pickup)


    # 读取订单数据文件
    # 注意: 多个小哥的订单id也不能有相同的
    data_orders_origin = pd.read_csv('../data_new/所有快递人员_0505_all_final_refine_加入路区划分前后_加入派单开始时间_去除三个异常值_加入预计妥投时间_新的buildingid_加入售后取件.csv')
    # data_orders_origin = pd.read_csv('data/所有快递人员_0505_all_final_refine_加入路区划分前后.csv')
    data_orders_origin.rename(columns={'Unnamed: 0': 'No'}, inplace=True)
    data_orders_origin = data_orders_origin.dropna(subset=['building_id'])
    data_orders_origin = data_orders_origin[data_orders_origin['building_id'] != -1].reset_index(drop=True)
    data_orders_origin = data_orders_origin[data_orders_origin['building_id'] != '-1'].reset_index(drop=True)
    #     print('len orders',len(data_orders_origin))
    data_orders_origin['start_time1'] = data_orders_origin['start_time'].apply(starttime_regu)
    data_orders_origin['end_time1'] = pd.to_datetime(data_orders_origin['end_time']) - pd.to_datetime(
        '2022-05-05 00:00:00')
    data_orders_origin['end_time1'] = pd.to_timedelta(data_orders_origin['end_time1'])
    data_orders_origin['end_time1'] = data_orders_origin['end_time1'].dt.total_seconds()
    data_orders_origin['ddl_time1'] = data_orders_origin['ddl_time'].apply(starttime_regu)
    ## 0905出去yfd start time过小的三个订单
    data_orders_origin=data_orders_origin[data_orders_origin['end_time1']>7.5*3600].reset_index(drop=True)
    # 生成每个快递员的派件订单和揽件订单
    for j in operator_name:
        tmp_ = data_orders_origin[data_orders_origin['路区划分后订单对应快递员'] == j].reset_index(drop=True)
#     for j in operator_0505:
#         tmp_ = data_orders_origin[data_orders_origin['operator_id'] == j].reset_index(drop=True)
        print(j, dict(tmp_['type'].value_counts()))
        orders_1 = []
        orders_pickup_1 = []
        for i in range(tmp_.shape[0]):
            # if int(tmp_['building_id'][i]) == 645 or int(tmp_['building_id'][i]) == 602 or int(tmp_['building_id'][i]) == 445 or int(tmp_['building_id'][i]) == 655:
            #     continue
            if tmp_['type'][i] == 'deliver':
                if int(tmp_['floors'][i]) == -1:  # 对于不确定楼层的订单, 根据楼是否为电梯楼随机生成
                    floor = random.choice([1, 2, 3, 4, 5])
                else:
                    floor = tmp_['floors'][i]
                temp_order = {
                    "id": int(tmp_['No'][i]),
                    "building_id": int(tmp_['building_id'][i]),
                    # "floor": int(tmp_['floors'][i]),
                    "floor": int(floor),
                    "unit": min(int(tmp_['unit'][i]), 10),
                    "type": tmp_['type'][i],
                }
                orders_1.append(temp_order)
            else:
                if int(tmp_['floors'][i]) == -1:  # 对于不确定楼层的订单, 根据楼是否为电梯楼随机生成
                    floor = random.choice([1, 2, 3, 4, 5])
                else:
                    floor = tmp_['floors'][i]
                temp_order = {
                    "id": int(tmp_['No'][i]),
                    "building_id": int(tmp_['building_id'][i]),
                    "start_time": tmp_['start_time1'][i],
                    "floor": int(floor),
                    "unit": min(int(tmp_['unit'][i]), 10),
                    "type": tmp_['type'][i],
                    "ddl_time": (tmp_['ddl_time1'][i]),
                    "from_sale":bool(tmp_['from_sale'][i]),

                }
                orders_pickup_1.append(temp_order)
        many_courier_orders_deliver.append(orders_1)
        many_courier_orders_pickup.append(orders_pickup_1)

    courier_ids = [str(i) for i in operator_0505]  # 例如"haoxinghai", lizhanyou"

    distance_to_roadarea = 1000 # 营业部到各个路区的距离 ####此处还需修改
    # buildings = json.load(open("data/buildings.json"))

    # orders2_pickup = []  # 样例2：快递员2的揽件订单
    station_location = [116.4352141813896, 39.96816730518909] # 营业部的经纬度坐标 GCJ: (116.441433,39.96955）; WGS: (116.4352141813896, 39.96816730518909)
    buildings_order_use = pickle.load(open("data1/buildings_new.pkl", "rb"))  # 楼宇信息
    regions = pickle.load(open("data1/regions_all.pkl", "rb"))
    G1 = pickle.load(open("data1/G_all.pkl", "rb"))
    many_courier_actions = [] # 各个快递员的动作序列
    many_courier_time = [] # 各个快递员的时间段序列
    many_courier_orders = [] # 用于仿真

    # 生成各个快递员的动作序列
    order_seq_i = 0
    assert sum(len(x) for x in many_courier_orders_deliver) + sum(len(x) for x in many_courier_orders_pickup) == 3842
    for orders_deliver,orders_pickup, courier_id in zip(many_courier_orders_deliver, many_courier_orders_pickup, courier_ids):
        (start_time_morning_for_cut, end_time_morning_for_cut, start_time_norn_for_cut, end_time_norn_for_cut,
            orders_morning_sequence, orders_norn_sequence, one_courier_session, one_courier_session_num) = \
            main_order(
                orders_deliver,
                orders_pickup,
                session_list_true_list[order_seq_i,0],
                session_list_true_list[order_seq_i,2],
                station_location,
                buildings_order_use,
                session_express_num_ratio[order_seq_i]  # 此处是所有订单，还需要将订单划分为上午与下午
            ) 
        assert len(orders_morning_sequence) + len(orders_norn_sequence) == len(orders_deliver) + len(orders_pickup)
        actions = main_generate_action(start_time_morning_for_cut,end_time_morning_for_cut,start_time_norn_for_cut,end_time_norn_for_cut,orders_morning_sequence,orders_norn_sequence)

        # many_courier_orders.append((courier_id, order_sequence))
        many_courier_actions.append((courier_id, actions))
        order_seq_i = order_seq_i + 1
        # many_courier_time.append((courier_id,time_list))
        # many_courier_actions.append((courier_id, actions))
        time_stamp = []
        # print(len(actions))
        all_order_sequence = orders_morning_sequence + orders_norn_sequence
        for x in all_order_sequence:
            if 'id' in x:
                time_stamp.append(x['end_time'])
        time_slice = np.linspace(0, 86400, 25).tolist()
        number_in_slice = []
        for i in range(len(time_slice)):
            if i == (len(time_slice) - 1):
                continue
            temp_number = 0
            for y in time_stamp:
                 if (y <= time_slice[i + 1]) & (y > time_slice[i]):
                     temp_number = temp_number + 1
            # temp_number = sum((y <= time_slice[i + 1]) & (y >= time_slice[i]) for y in time_stamp)
            number_in_slice.append(temp_number)
        x_data = range(1, 25)
        number_in_slice_all.append(number_in_slice)
        session_list.append(one_courier_session)
        session_express_num.append(one_courier_session_num)
    # pprint_actions(actions)

    # 存结果
    pickle.dump(many_courier_actions, open(f"../data_cjw/actions_absent_{DEMO}.pkl", "wb"))
    pickle.dump(session_express_num, open(f"../data_cjw/session_express_num.pkl", "wb"))
    pickle.dump(session_list, open(f"../data_cjw/session_list.pkl", "wb"))
    pickle.dump(number_in_slice_all, open(f"../data_cjw/number_in_slice_all.pkl", "wb"))

    # pickle.dump(many_courier_actions, open(f"data/action_by_simulate_minwang.pkl", "wb"))
    # pickle.dump(many_courier_time, open(f"data/time_by_simulate_minwang.pkl", "wb"))

    # for courier_id,orders in many_courier_orders:
    #     actions = main_generate_action(start_time_morning_for_cut,end_time_morning_for_cut,start_time_norn_for_cut,end_time_norn_for_cut,orders_morning_sequence,orders_norn_sequence)

print(type_start_morning_list)
print(type_end_morning_list)
