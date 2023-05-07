import pickle
import random
from collections import Counter, defaultdict
from copy import deepcopy
from pprint import pprint

import networkx as nx
import numpy as np
from hyperopt import STATUS_OK, Trials,fmin, hp, tpe  # TODO: 如果去掉这行, 则读缓存的NN模型时卡住, 不知道为啥
from shapely.geometry import LineString
from tqdm import tqdm

from lkh import LKH

DPT = "hk"  # 指定营业部department
if DPT == "mxl":
    from mxl.constants_all import *
    from mxl.params_eval import *
elif DPT == "hk":
    from hk.constants_all import *
    from hk.params_eval import *

random.seed(233)
MAX_STALL_N = 1  # 将摆摊地点过多的单拆成多个摆摊, 以增加检查是否该跑去揽收的频率


def cat(arr: list) -> list:
    """
    首尾相接
    """
    return sum((i[1:] for i in arr[1:]), arr[0][:-1])


def split(arr, f):
    """
    把满足f和不满足的分开
    """
    a1 = []
    a2 = []
    for i in arr:
        (a1 if f(i) else a2).append(i)
    return a1, a2


def split_at(arr, f):
    """
    在首个不满足f的位置切成两段
    """
    for i, a in enumerate(arr):
        if not f(a):
            break
    else:
        return arr, []
    return arr[:i], arr[i:]


class Simulator:
    def __init__(self, G, buildings, distance=None):
        self.G = G
        self.buildings = buildings
        self.bid2nid = {i["id"]: i["gate_id"] for i in buildings.values()}
        if distance is None:
            self.distance = {
                i: j
                for i, j in nx.all_pairs_bellman_ford_path_length(G, weight="length")
            }
        else:
            self.distance = distance
        self.bid2t_stall = {
            bid: b["stall_time"] for bid, b in buildings.items() if "stall_time" in b
        }

    def set_config(self, courier_id, courier_config):
        self.courier_id = courier_id
        self.config_orig = deepcopy(courier_config)
        self.config = courier_config
        self.order2t = {
            ORDER_DELIVER: courier_config["t_deliver"],
            ORDER_CPICK: courier_config["t_bpick"],
            ORDER_BPICK: courier_config["t_cpick"],
        }
        self.order2reserve = {
            ORDER_CPICK: courier_config["t_cpick_reserve"],
            ORDER_BPICK: courier_config["t_bpick_reserve"],
        }
        if "t_stall_multiplier" in courier_config:
            p = courier_config["t_stall_multiplier"]
            self.bid2t_stall = {
                bid: b["stall_time"] * p
                for bid, b in self.buildings.items()
                if "stall_time" in b
            }

    def generate_action(
        self,
        start_time_morning: float,
        orders_morning: dict,
        start_time_afternoon: float,
        orders_afternoon: dict,
        only_one_wave=False,
        no_start_arrange=False,
    ):
        """
        根据时间和订单信息生成单个快递员行为

        start_time_morning：    上午快递员到达第一栋楼的时间
        orders_morning：        上午的订单
        start_time_afternoon：  下午快递员到达第一栋楼的时间
        orders_afternoon：      下午的订单
        """
        # TODO: 大促期间小哥可能不止2波, 将代码修改为可以有任意波
        # 由于此不灵活性, 暂停止维护此函数, 请使用generate_wave()

        # 上午的行为
        actions_1 = self.generate_wave(start_time_morning, orders_morning)
        # 下的行为
        if not only_one_wave:
            actions_2 = self.generate_wave(start_time_afternoon, orders_afternoon)
        # 总行为
        actions = []
        # 从站点去路区
        node_start_1 = self.bid2nid[actions_1[0]["building"]]
        t_from_station_1 = (
            self.distance[DEFAULT_SID][node_start_1] / self.config["v_car"]
        )
        # t_day_start = T_DAY_START
        t_day_start = max(
            T_DAY_START, start_time_morning - t_from_station_1 - 3600
        )  # 最多在离站前拣货1h
        t_rest_1 = start_time_morning - (t_day_start + t_from_station_1)
        assert t_rest_1 >= 0, f"上午开始时间过早，快递员无法到达"
        from_path_gps_1, from_path_xy_1 = self._get_path(DEFAULT_SID, node_start_1)
        actions += [
            {
                "type": ACTION_ARRANGE,
                "start_time": t_day_start,
                "end_time": start_time_morning - t_from_station_1,
                "station_id": DEFAULT_SID,
                "gps": self.G.nodes[DEFAULT_SID]["gps"],
                "xy": self.G.nodes[DEFAULT_SID]["xy"],
                "target_orders": [],
            },
            {
                "type": ACTION_FROMSTATION,
                "start_time": start_time_morning - t_from_station_1,
                "end_time": start_time_morning,
                "start_node": DEFAULT_SID,
                "end_building": actions_1[0]["building"],  # 第一栋楼的id
                "gps": from_path_gps_1,  # [(lon, lat), ...] 从快递站到第一栋楼的最短路
                "xy": from_path_xy_1,  # [(x, y), ...]
                "target_orders": actions_1[0]["target_orders"],
            },
        ]
        if no_start_arrange:
            actions = actions[1:]
        node_end_1 = self.bid2nid[actions_1[-1]["building"]]
        t_to_station_1 = self.distance[node_end_1][DEFAULT_SID] / self.config["v_car"]

        if only_one_wave:
            to_path_gps_1, to_path_xy_1 = self._get_path(node_end_1, DEFAULT_SID)
            actions += actions_1
            actions.append(
                {
                    "type": ACTION_TOSTATION,
                    "start_time": actions_1[-1]["end_time"],
                    "end_time": actions_1[-1]["end_time"] + t_to_station_1,
                    "start_building": actions_1[-1]["building"],  # 最后一栋楼的id
                    "end_node": DEFAULT_SID,
                    "gps": to_path_gps_1,  # [(lon, lat), ...] 从最后一栋楼到快递站的最短路
                    "xy": to_path_xy_1,  # [(x, y), ...]
                    "target_orders": [],
                }
            )
            return actions

        node_start_2 = self.bid2nid[actions_2[0]["building"]]
        t_from_station_2 = (
            self.distance[DEFAULT_SID][node_start_2] / self.config["v_car"]
        )
        t_rest_2 = start_time_afternoon - (
            actions_1[-1]["end_time"] + t_to_station_1 + t_from_station_2
        )
        if t_rest_2 < 0:
            print(f"警告：下午开始时间过早，快递员会迟到{-t_rest_2/60:.2f}分钟，尝试缩短动作耗时")
            # 可用于收缩的动作类型、可收缩比例；从上到下依次考虑
            to_shrink = [
                [{ACTION_REST}, 0.9],
                [{ACTION_WALK}, 0.2],
                [{ACTION_UPSTAIR, ACTION_DOWNSTAIR, ACTION_TODOOR}, 0.5],
            ]
            for a, b in zip(actions_1, actions_1[1:]):
                assert a["end_time"] == b["start_time"]
            t = t_rest_2
            ts = [0] * len(to_shrink)
            for i, (a, b) in enumerate(to_shrink):
                for act in actions_1:
                    if act["type"] in a:
                        t_ = b * (act["end_time"] - act["start_time"])
                        ts[i] += t_
                        t += t_
                        if t >= 0:
                            break
                if t >= 0:
                    break
            else:
                print(f"错误：即使缩短时间，快递员依然会迟到{-t/60:.2f}分钟")
            ks = [[a, 1 - b] for a, b in to_shrink[: i + 1]]
            t0 = 0
            for act in actions_1:
                if act["type"] in to_shrink[i][0]:
                    t0 += act["end_time"] - act["start_time"]
            ks[i][1] = 1 - (-t_rest_2 - sum(ts[:i]) + 1) / (t0 + 1e-6)
            assert 0 < ks[i][1] < 1
            for a, b in ks:
                for act in actions_1:
                    if act["type"] in a:
                        act["end_time"] = act["start_time"] + b * (
                            act["end_time"] - act["start_time"]
                        )
            t = actions_1[0]["start_time"]
            for a in actions_1:
                a["end_time"] += t - a["start_time"]
                a["start_time"] = t
                t = a["end_time"]
            t_rest_2 = start_time_afternoon - (
                actions_1[-1]["end_time"] + t_to_station_1 + t_from_station_2
            )
            assert t_rest_2 >= 0, t_rest_2
        to_path_gps_1, to_path_xy_1 = self._get_path(node_end_1, DEFAULT_SID)
        from_path_gps_2, from_path_xy_2 = self._get_path(DEFAULT_SID, node_start_2)
        actions += actions_1
        actions += [
            # 回站
            {
                "type": ACTION_TOSTATION,
                "start_time": actions_1[-1]["end_time"],
                "end_time": actions_1[-1]["end_time"] + t_to_station_1,
                "start_building": actions_1[-1]["building"],  # 最后一栋楼的id
                "end_node": DEFAULT_SID,
                "gps": to_path_gps_1,  # [(lon, lat), ...] 从最后一栋楼到快递站的最短路
                "xy": to_path_xy_1,  # [(x, y), ...]
                "target_orders": [],
            },
            # 午休
            {
                "type": ACTION_ARRANGE,
                "start_time": actions_1[-1]["end_time"] + t_to_station_1,
                "end_time": actions_1[-1]["end_time"] + t_to_station_1 + t_rest_2,
                "station_id": DEFAULT_SID,
                "gps": self.G.nodes[DEFAULT_SID]["gps"],
                "xy": self.G.nodes[DEFAULT_SID]["xy"],
                "target_orders": [],
            },
            # 离站
            {
                "type": ACTION_FROMSTATION,
                "start_time": actions_1[-1]["end_time"] + t_to_station_1 + t_rest_2,
                "end_time": start_time_afternoon,
                "start_node": DEFAULT_SID,
                "end_building": actions_2[0]["building"],  # 第一栋楼的id
                "gps": from_path_gps_2,  # [(lon, lat), ...] 从快递站到第一栋楼的最短路
                "xy": from_path_xy_2,  # [(x, y), ...]
                "target_orders": actions_2[0]["target_orders"],
            },
        ]
        actions += actions_2
        node_end_2 = self.bid2nid[actions_2[-1]["building"]]
        t_to_station_2 = self.distance[node_end_2][DEFAULT_SID] / self.config["v_car"]
        to_path_gps_2, to_path_xy_2 = self._get_path(node_end_2, DEFAULT_SID)
        actions.append(
            # 回站
            {
                "type": ACTION_TOSTATION,
                "start_time": actions_2[-1]["end_time"],
                "end_time": actions_2[-1]["end_time"] + t_to_station_2,
                "start_building": actions_2[-1]["building"],  # 最后一栋楼的id
                "end_node": DEFAULT_SID,
                "gps": to_path_gps_2,  # [(lon, lat), ...] 从最后一栋楼到快递站的最短路
                "xy": to_path_xy_2,  # [(x, y), ...]
                "target_orders": [],
            }
        )
        return actions

    def _get_path(self, a, b):
        assert a != b
        path = nx.shortest_path(self.G, a, b, weight="length")
        assert path
        path_gps = cat([self.G.edges[i, j]["gps"] for i, j in zip(path, path[1:])])
        path_xy = cat([self.G.edges[i, j]["xy"] for i, j in zip(path, path[1:])])
        return path_gps, path_xy

    @staticmethod
    def _offset_action_time(actions, offset, inplace=True):
        if not inplace:
            actions = deepcopy(actions)
        for i in actions:
            i["start_time"] += offset
            i["end_time"] += offset
        return actions

    def _generate_one_building(self, building_id, uf):
        """
        生成一栋楼的行为
        """
        if building_id in self.bid2t_stall:
            actions = []
            ct = 0
            t = self.bid2t_stall[building_id]
            for u, fs in sorted(uf.items()):
                for f, os in sorted(fs.items()):
                    for o in os:
                        actions.append(
                            {
                                "type": ORDER2ACTION[o["type"]],
                                "start_time": ct,
                                "end_time": ct + t,
                                "building": building_id,
                                "unit": u,
                                "floor": f,
                                "target_orders": [o],  # 该单
                                "is_stall": True,
                            }
                        )
                        ct += t
            return actions

        if self.buildings[building_id]["is_elevator"]:
            t_wait_1 = self.config["t_elevator_wait1"]
            t_wait_2 = self.config["t_elevator_wait2"]
            t_up = t_down = self.config["t_elevator_floor"]
            a_up = a_down = ACTION_ELEVATOR
        else:
            t_wait_1 = t_wait_2 = 0
            t_up = self.config["t_stair_up"]
            t_down = self.config["t_stair_down"]
            a_up = ACTION_UPSTAIR
            a_down = ACTION_DOWNSTAIR
        # TODO: 设置理货时间、多次往返取货、体力值+休息
        # TODO: 增加等电梯的动作，现在是将等电梯的时间并入了坐电梯的时间
        actions = []
        ct = 0
        first_time = True
        for u, fs in sorted(uf.items()):
            first_time = True
            last_f = 1
            for f, os in sorted(fs.items()):
                if f == 1:
                    assert last_f == 1
                else:
                    if first_time:
                        t1 = ct + t_wait_1 + (f - last_f) * t_up
                        first_time = False
                    else:
                        t1 = ct + t_wait_2 + (f - last_f) * t_up
                    actions.append(
                        {
                            "type": a_up,
                            "start_time": ct,
                            "end_time": t1,
                            "building": building_id,  # 所在building_id
                            "unit": u,  # 所在单元
                            "from": last_f,  # 当前楼层
                            "to": f,  # 目标楼层
                            "num": f - last_f,  # to - from, 上楼为正, 下楼为负
                            "target_orders": os,  # 目标楼层的所有要完成的单, 下楼为[]
                        }
                    )
                    ct = t1
                for o in os:
                    t1 = ct + self.config["t_todoor"]
                    t2 = t1 + self.order2t[o["type"]]
                    actions += [
                        {
                            "type": ACTION_TODOOR,
                            "start_time": ct,
                            "end_time": t1,
                            "building": building_id,
                            "unit": u,
                            "floor": f,  # 所在楼层
                            "target_orders": [o],  # 该单
                        },
                        {
                            "type": ORDER2ACTION[o["type"]],
                            "start_time": t1,
                            "end_time": t2,
                            "building": building_id,
                            "unit": u,
                            "floor": f,
                            "target_orders": [o],  # 该单
                        },
                    ]
                    ct = t2
                last_f = f
            if last_f != 1:
                assert not first_time
                t1 = ct + t_wait_2 + (last_f - 1) * t_down
                actions.append(
                    {
                        "type": a_down,
                        "start_time": ct,
                        "end_time": t1,
                        "building": building_id,  # 所在building_id
                        "unit": u,  # 所在单元
                        "from": last_f,  # 当前楼层
                        "to": 1,  # 目标楼层
                        "num": 1 - last_f,  # to - from, 上楼为正, 下楼为负
                        "target_orders": [],  # 目标楼层的所有要完成的单, 下楼为[]
                    }
                )
                ct = t1
            t1 = (
                ct
                + self.config["t_between_units"]
                + self.config["t_rest_per_package"] * sum(len(i) for i in fs.values())
            )
            actions.append(
                {
                    "type": ACTION_REST,
                    "start_time": ct,
                    "end_time": t1,
                    "building": building_id,
                    "target_orders": [],
                }
            )
            ct = t1
        actions[-1]["end_time"] -= self.config["t_between_units"]
        return actions

    def _get_bs(self, bid_odrs):
        def get_uf(os):
            uf = defaultdict(lambda: defaultdict(list))
            for o in os:
                uf[o["unit"]][o["floor"]].append(o)
            return uf
        bs = [{"id": bid, "uf": get_uf(odrs)} for bid, odrs in bid_odrs]
        for b in bs:
            b["act"] = self._generate_one_building(b["id"], b["uf"])
        for a, b in zip(bs, bs[1:]):
            a["to_next"] = (
                self.distance[self.bid2nid[a["id"]]][
                    self.bid2nid[b["id"]]
                ]
                / self.config["v_walk"]
            )
        bs[-1]["to_next"] = 0
        # 将订单数很多的一次摆摊拆成多次, 以增加检查是否该跑去揽收的频率
        bs_new = []
        for b in bs:
            if b["id"] not in self.bid2t_stall:
                bs_new.append(b)
            else:
                os = self._unpack_uf(b["uf"])
                if len(os) <= MAX_STALL_N:
                    bs_new.append(b)
                else:
                    bid = b["id"]
                    to_next = b["to_next"]
                    for i in range(0, len(os), MAX_STALL_N):
                        uf = defaultdict(lambda: defaultdict(list))
                        for o in os[i : i + MAX_STALL_N]:
                            uf[o["unit"]][o["floor"]].append(o)
                        bs_new.append(
                            {
                                "id": bid,
                                "uf": uf,
                                "act": self._generate_one_building(bid, uf),
                                "to_next": 0,
                            }
                        )
                    bs_new[-1]["to_next"] = to_next
        return bs_new

    def _generate_seq_stat(self, orders, bid2seq):
        assert bid2seq is not None
        assert orders
        bid_odrs = sorted(
            group_by(orders, "building_id").items(), key=lambda x: bid2seq.get(x[0], random.random())
        )
        return self._get_bs(bid_odrs)

    def _generate_seq_nn(self, orders, start_node, start_time, bid2seq, seq_nn):
        assert seq_nn is not None and bid2seq is not None
        P_STAT2LKH = 0.4
        K = 5   # 限定nn的候选结果在统计值的前K个中
        N = 20  # 若统计值最优先的楼连续N次没有被选中则强制选取

        cid = self.courier_id
        t = start_time

        future = {o["id"] for o in orders if o["type"] != ORDER_DELIVER and o["start_time"] > t}
        todo = {o["id"] for o in orders if o["type"] == ORDER_DELIVER or o["start_time"] <= t}
        assert todo
        orders = {o["id"]: o for o in orders}
        
        # 决定初始楼
        bid2odrs = group_by([orders[x] for x in todo], "building_id")
        p = sum(len(odrs) for bid, odrs in bid2odrs.items() if bid in bid2seq) / len(orders)
        use_stat = p > P_STAT2LKH
        if use_stat:  # 若大量单属于bid2seq中的楼, 则按bid2seq取优先级最高的楼
            bid = min([bid for bid in bid2odrs if bid in bid2seq], key=lambda x: bid2seq[x])
        else:        # 否则取离起点最近的楼
            bid = min(bid2odrs, key=lambda x: self.distance[start_node][self.bid2nid[x]])

        bid2n = defaultdict(int)  # 记录bid在stat的topK中待了多少次没被选中
        bs = []
        while True:
            # 送当前楼
            bs_add = self._get_bs([(bid, bid2odrs[bid])])
            for b in bs_add:
                t += b["act"][-1]["end_time"]
                assert b["to_next"] == 0
            # 更新todo和future
            todo -= {o["id"] for o in bid2odrs[bid]}
            happened = {oid for oid in future if orders[oid]["start_time"] <= t}
            future -= happened
            todo |= happened
            if not todo:
                bs += bs_add
                break
            # 决定下一栋楼
            bid2odrs = group_by([orders[x] for x in todo], "building_id")
            p = sum(len(odrs) for bid, odrs in bid2odrs.items() if bid in bid2seq) / len(orders)
            use_stat = p > P_STAT2LKH
            if use_stat:  # 基于历史统计顺序, 对nn的结果做大致的限定
                bids_stat = sorted(bid2odrs, key=lambda x: bid2seq.get(x, random.random()))
                for x in bids_stat[:K]:
                    if bid2n[x] >= N:
                        nbid = x
                        break
                else:
                    nbid = seq_nn.infer(
                        odrs=sum(bid2odrs.values(), []),
                        cid=cid,
                        bid=bid,
                        t=t,
                        candidates=bids_stat[:K])
                for x in bids_stat[:K]:
                    if x != nbid:
                        bid2n[x] += 1
                    else:
                        bid2n[x] = 0
            else:
                nbid = seq_nn.infer(
                    odrs=sum(bid2odrs.values(), []),
                    cid=cid,
                    bid=bid,
                    t=t,
                    candidates=None)
                bid2n[nbid] = 0
            # 更新t_travel
            t_travel = self.distance[self.bid2nid[bid]][self.bid2nid[nbid]] / self.config["v_walk"]
            bs_add[-1]["to_next"] = t_travel
            bs += bs_add
            t += t_travel
            bid = nbid
        
        # 如果送完todo之后还有future, 则无视future还没发生, 直接按一个随便的顺序送完
        # 实际上这部分结果并不会被使用, 见generate_wave中的相关逻辑
        # (只采取了返回的bs中的第一个, 其to_next也将在后续被正确赋值)
        if future:
            bid2odrs = group_by([orders[x] for x in future], "building_id")
            bs += self._get_bs(bid2odrs)

        return bs

    def _generate_seq_lkh(self, orders, start_node, end_node, start_time=-1):
        """
        生成访问顺序
        如果start_time!=0, 会考虑时间
        """
        # TODO: 解决以下assert
        # assert not any(
        #     self.bid2nid[i["building_id"]] in [start_node, end_node]
        #     for i in orders
        # )
        # if start_node == end_node and start_time != -1:
        #     pickle.dump(
        #         (orders, start_node, end_node, start_time), open("_test_data.pkl", "wb")
        #     )

        # LKH求解VRP得到访问每栋楼的顺序
        id2node = [start_node] + sorted(
            {self.bid2nid[i["building_id"]] for i in orders}
        )
        assert len(id2node) == len(set(id2node))
        node2id = {j: i for i, j in enumerate(id2node)}
        assert len(id2node) == len(node2id)
        dist = np.empty((len(node2id),) * 2)
        for a, b in node2id.items():
            for c, d in node2id.items():
                dist[b, d] = self.distance[a][c]

        if start_node == end_node:
            if start_time == -1:
                sol = list(
                    LKH().solve_acvrp(dist, [1] * (len(dist) - 1), 1e5, 1, scale=1e3)
                )[-1].solution
            else:
                inf = 1e15
                ts = {i: inf for i in id2node}
                for i in orders:
                    b = self.bid2nid[i["building_id"]]
                    ts[b] = min(ts[b], i["ddl_time"] - start_time)
                assert sum(1 for i in ts.values() if i > inf / 2) == 1, ts
                demand = np.zeros((len(dist), 5))
                demand[:, 1] = 1
                demand[:, 3] = list(ts.values())
                demand[:, 4] = T_PICK_ESTIMATE
                demand[:, 2] = np.minimum(demand[:, 2], demand[:, 3])
                sol = list(
                    LKH().solve_acvrpspdtw(
                        dist, dist, demand, 1e5, 1, scale=1e3, always=True
                    )
                )[-1].solution
            assert sol.count(0) == 2 and sol[0] == 0 == sol[-1]
            sol = sol[1:-1]
        else:
            n = len(node2id)
            inf = 1e15
            dist2 = np.zeros((n + 1,) * 2)
            dist2[:n, :n] = dist
            dist2[0, -1] = inf
            dist2[1:-1, 0] = inf
            dist2[-1, 1:-1] = inf
            dist2[1:-1, -1] = [self.distance[i][end_node] for i in id2node[1:]]
            if start_time == -1:
                sol = list(
                    LKH().solve_acvrp(dist2, [1] * (len(dist2) - 1), 1e5, 1, scale=1e3)
                )[-1].solution
            else:
                ts = {i: inf for i in id2node}
                for i in orders:
                    b = self.bid2nid[i["building_id"]]
                    ts[b] = min(ts[b], i["ddl_time"] - start_time)
                assert sum(1 for i in ts.values() if i > inf / 2) == 1, ts
                demand = np.zeros((n + 1, 5))
                demand[:, 1] = 1
                demand[:, 3] = [*ts.values(), inf]
                demand[:, 4] = T_PICK_ESTIMATE
                demand[:, 2] = np.minimum(demand[:, 2], demand[:, 3])
                sol = list(
                    LKH().solve_acvrpspdtw(
                        dist2, dist2, demand, 1e5, 1, scale=1e3, always=True
                    )
                )[-1].solution
            assert sol.count(0) == 2 and sol[0] == 0 == sol[-1] and sol[-2] == n
            sol = sol[1:-2]

        sol = {j: i + 1 for i, j in enumerate(sol)}
        bid2odrs = defaultdict(list)
        for o in orders:
            bid2odrs[o["building_id"]].append(o)
        b2id = {i: sol[node2id[self.bid2nid[i]]] for i in bid2odrs}
        assert sorted(b2id.values()) == list(range(1, len(b2id) + 1))
        id2b = {j: i for i, j in b2id.items()}
        bid_odrs = [[id2b[i], bid2odrs[id2b[i]]] for i in range(1, len(b2id) + 1)]
        bs = self._get_bs(bid_odrs)
        return bs

    @staticmethod
    def _unpack_uf(uf):
        return [k for i in uf.values() for j in i.values() for k in j]

    def generate_wave(self, start_time, orders, seq_type="lkh", bid2seq=None, seq_nn=None):
        """生成一波的行为"""
        _n = len(orders)
        assert _n > 0

        # 根据积极性, 调整config
        def k2p(k):
            if k in {
                "motivation_multiplier",
                "t_bpick_reserve",
                "t_cpick_reserve",
                "v_car",
                "motivation",
                "n_avg",
            }:
                return 1
            if k == "v_walk":
                return 1 / wt_p
            # if k in {
            #     "t_between_units", "t_bpick", "t_cpick", "t_deliver", "t_elevator_wait1", "t_stair", "t_stall_multiplier",
            #     "t_elevator_floor", "t_stair_up", "t_stair_down", "t_todoor", "t_rest_per_package", "t_elevator_wait2"}:
            #     return wt_p
            assert "t_" in k
            return wt_p
        n_offset = _n / self.config_orig["n_avg"] - 1  # 归一化单量波动
        wt_p = 1 + n_offset * self.config_orig["motivation"] * self.config_orig["motivation_multiplier"]  # 工作时长放缩比例
        wt_p = max(0.1, wt_p)
        self.config = {k: v * k2p(k) for k, v in self.config_orig.items()}
        if "t_stall_multiplier" in self.config:
            p = self.config["t_stall_multiplier"]
            self.bid2t_stall = {
                bid: b["stall_time"] * p
                for bid, b in self.buildings.items()
                if "stall_time" in b
            }

        # 生成初始派送序列: 按顺序排列的楼及相应的单和动作, bs中的元素为{"id", "uf", "act", "to_next"}
        dodrs = [i for i in orders if i["type"] == "deliver"]
        assert dodrs  # TODO: 处理没有派件的情况
        if seq_type == "lkh":
            bs = self._generate_seq_lkh(dodrs, DEFAULT_SID, DEFAULT_SID)
        elif seq_type == "stat":
            bs = self._generate_seq_stat(dodrs, bid2seq)
        elif seq_type == "nn":
            bs = self._generate_seq_nn(dodrs, DEFAULT_SID, start_time, bid2seq, seq_nn)

        # 向派送序列中, 插入揽收单
        ps = [i for i in orders if i["type"] != "deliver"]
        ps.sort(key=lambda x: x["start_time"])  # 当前还未插入的pick-up单
        ib = 0  # 当前插入过的b_idx
        ct = start_time + bs[0]["act"][-1]["end_time"] # 当前时间
        assert sum(len(self._unpack_uf(i["uf"])) for i in bs) + len(ps) == _n
        assert all(all(j["building"] == i["id"] for j in i["act"]) for i in bs)
        while ps:
            while ct < ps[0]["start_time"]:  # 更新ib: 送完ib后, ps[0]发生
                ib += 1
                if ib == len(bs):
                    break
                ct += bs[ib - 1]["to_next"] + bs[ib]["act"][-1]["end_time"]
            if ib == len(bs):  # 如果送完bs, ps[0]都还没发生, 则等待到ps[0]发生后
                ib -= 1
                bs[ib]["act"][-1]["end_time"] += (
                    ps[0]["start_time"] - ct + 1
                )  # 额外多等1s以防止浮点误差死循环
                bs[ib]["act"][-1]["wait_for_pick"] = True
                ct = ps[0]["start_time"] + 1
                continue
            while True:  # 在完成新插入的单之后, 也可能会接到新的单, 因此循环
                to_insert, ps = split_at(
                    ps, lambda x: x["start_time"] <= ct
                )  # 分为当前需要插入的, 和之后再插入的
                if not to_insert:
                    break
                p1, to_insert = split(  # 分为恰好就在当前楼的, 和不在的
                    to_insert, lambda p: p["building_id"] == bs[ib]["id"]
                )
                if not p1:  # 如果没有在当前楼的, 则这里还暂时不会处理, 时间不更新, 因此也可以退出循环
                    break
                # 处理恰好就在当前楼的
                uf = defaultdict(lambda: defaultdict(list))
                for i in p1:
                    uf[i["unit"]][i["floor"]].append(i)
                act = self._generate_one_building(bs[ib]["id"], uf)
                if bs[ib]["id"] in self.bid2t_stall:  # 摆摊的, 直接合并进bs[ib]
                    bs[ib]["act"] += self._offset_action_time(
                        act, bs[ib]["act"][-1]["end_time"]
                    )
                    for u, fs in uf.items():
                        for f, os in fs.items():
                            for o in os:
                                bs[ib]["uf"][u][f].append(o)
                else:  # 上下楼的, 加一个新的b在后面, 重新上下楼一次(合并进去有点麻烦, 懒得写)
                    assert "unit" in bs[ib]["act"][-2] and "unit" in act[-2]
                    if bs[ib]["act"][-2]["unit"] != act[-2]["unit"]:
                        assert bs[ib]["act"][-1]["type"] == ACTION_REST
                        bs[ib]["act"][-1]["end_time"] += self.config["t_between_units"]
                        ct += self.config["t_between_units"]
                    new_b = {
                        "id": bs[ib]["id"],
                        "uf": uf,
                        "act": act,
                        "to_next": bs[ib]["to_next"],  # 继承ib的移动时间
                    }
                    bs[ib]["to_next"] = 0  # ib的移动时间归零
                    bs.insert(ib + 1, new_b)
                    ib += 1
                ct += act[-1]["end_time"]
                ps = to_insert + ps  # ps还原为尚未处理的揽件单, 在循环开始时再次被分为to_insert, ps
                assert sum(len(self._unpack_uf(i["uf"])) for i in bs) + len(ps) == _n
                assert all(all(j["building"] == i["id"] for j in i["act"]) for i in bs)
            if not to_insert:
                continue
            # 处理不在当前楼的
            if ib + 1 < len(bs):  # 后面还有派送
                p1, to_insert = split(  # 分为恰好就在下一楼的, 和不在的
                    to_insert, lambda p: p["building_id"] == bs[ib + 1]["id"]
                )
                if p1:  # 恰好就在下一楼的: 假如直接合并进去 将得到的act
                    uf = deepcopy(bs[ib + 1]["uf"])
                    for i in p1:
                        uf[i["unit"]][i["floor"]].append(i)
                    act = self._generate_one_building(bs[ib + 1]["id"], uf)
                else:  # 下一楼原本的act
                    act = bs[ib + 1]["act"]
                # 预估送下一栋楼会不会导致, 不在下一楼的揽收单订单超时
                t1 = ct + bs[ib]["to_next"] + act[-1]["end_time"]  # 送完下一楼后的时刻
                p2, to_insert = split(  # 不在下一楼的, 分为会超时的和不会超时的
                    to_insert,
                    lambda p: p["ddl_time"]
                    <= t1 + self.order2reserve[p["type"]],  # 送完下一楼后, 还需要一段时间才能送完揽收单
                )
                if p2:  # 有 不在下一楼的单会因为送下一楼而超时, 则先完成这些单
                    if p1:  # 此时下一楼发生了改变, p1不再特殊, 直接都扔进去分为p2和to_insert
                        p2, to_insert = split(
                            p1 + p2 + to_insert,
                            lambda p: p["ddl_time"]
                            <= t1 + self.order2reserve[p["type"]],
                        )
                    if seq_type == "nn":
                        new_bs = self._generate_seq_nn(p2, self.bid2nid[bs[ib]["id"]], ct, bid2seq, seq_nn)
                    else:
                        new_bs = self._generate_seq_lkh(
                            p2,
                            self.bid2nid[bs[ib]["id"]],
                            self.bid2nid[bs[ib + 1]["id"]],
                            start_time=ct)
                    bs[ib]["to_next"] = (
                        self.distance[self.bid2nid[bs[ib]["id"]]][
                            self.bid2nid[new_bs[0]["id"]]
                        ]
                        / self.config["v_walk"]
                    )
                    new_bs[-1]["to_next"] = (
                        self.distance[self.bid2nid[new_bs[-1]["id"]]][
                            self.bid2nid[bs[ib + 1]["id"]]
                        ]
                        / self.config["v_walk"]
                    )
                    bs[ib + 1 : ib + 1] = new_bs  # 这个语法的意思是在ib+1处插入一段数组
                    ib += 1
                    ct += bs[ib]["to_next"] + bs[ib]["act"][-1]["end_time"]
                else:  # 送下一栋楼不会导致超时, 则继续按原计划送, 顺便完成p1
                    ib += 1
                    if p1:  # 直接合并进ib
                        bs[ib]["uf"] = uf
                        bs[ib]["act"] = act
                    ct = t1
                ps = to_insert + ps  # 没处理的单继续留到后面再看
                assert sum(len(self._unpack_uf(i["uf"])) for i in bs) + len(ps) == _n
                assert all(all(j["building"] == i["id"] for j in i["act"]) for i in bs)
            else:  # 后面没有派送了, 则对所有to_insert生成new_bs, 但实际只采取其中第一个, 后面留到之后再看(可能还没到start_time)
                assert ib == len(bs) - 1
                if seq_type == "nn":
                    new_bs = self._generate_seq_nn(to_insert, self.bid2nid[bs[ib]["id"]], ct, bid2seq, seq_nn)
                else:
                    new_bs = self._generate_seq_lkh(
                        to_insert,
                        self.bid2nid[bs[ib]["id"]],
                        DEFAULT_SID,
                        start_time=ct)
                bs[ib]["to_next"] = (
                    self.distance[self.bid2nid[bs[ib]["id"]]][
                        self.bid2nid[new_bs[0]["id"]]
                    ]
                    / self.config["v_walk"]
                )
                bs.append(new_bs[0])
                ib += 1
                ct += bs[ib - 1]["to_next"] + bs[ib]["act"][-1]["end_time"]
                ps = sum((self._unpack_uf(i["uf"]) for i in new_bs[1:]), ps)
                assert sum(len(self._unpack_uf(i["uf"])) for i in bs) + len(ps) == _n
                assert all(all(j["building"] == i["id"] for j in i["act"]) for i in bs)
                ps.sort(key=lambda x: x["start_time"])
        actions = []
        ct = start_time
        last = None
        for i in bs:
            if last is not None and last["id"] != i["id"]:
                t1 = ct + last["to_next"]
                path_gps, path_xy = self._get_path(
                    self.bid2nid[last["id"]], self.bid2nid[i["id"]]
                )
                actions.append(
                    {
                        "type": ACTION_WALK,
                        "start_time": ct,
                        "end_time": t1,
                        "start_building": last["id"],
                        "end_building": i["id"],
                        "gps": path_gps,
                        "xy": path_xy,
                        "target_orders": self._unpack_uf(
                            i["uf"]
                        ),
                    }
                )
                ct = t1
            actions += self._offset_action_time(i["act"], ct)
            ct = actions[-1]["end_time"]
            last = i
        assert (
            sum(
                1
                for i in actions
                if i["type"] in {ACTION_DELIVER, ACTION_BPICK, ACTION_CPICK}
            )
            + len(ps)
            == _n
        )
        return actions

    def generate_one_courier_one_day(self, st_odr_seq_types, bid2seq=None, seq_nn=None):
        """对每一波generate_wave, 然后填充station相关的行为"""
        st_odr_seq_types.sort(key=lambda x: x[0])
        actions = []
        for i, (st, odrs, seq_type) in enumerate(st_odr_seq_types):
            a = self.generate_wave(st, odrs, seq_type, bid2seq, seq_nn)

            if i == 0:
                # 拣货, 去路区
                ns = self.bid2nid[a[0]["building"]]
                t_from_station = self.distance[DEFAULT_SID][ns] / self.config["v_car"]
                t_day_start = max(T_DAY_START, st - t_from_station - 3600)  # 最多在离站前拣货1h
                assert st > t_day_start + t_from_station, f"上午开始时间过早，快递员无法到达"
                from_path_gps, from_path_xy = self._get_path(DEFAULT_SID, ns)
                actions += [
                    {
                        "type": ACTION_ARRANGE,
                        "start_time": t_day_start,
                        "end_time": st - t_from_station,
                        "station_id": DEFAULT_SID,
                        "gps": self.G.nodes[DEFAULT_SID]["gps"],
                        "xy": self.G.nodes[DEFAULT_SID]["xy"],
                        "target_orders": [],
                    },
                    {
                        "type": ACTION_FROMSTATION,
                        "start_time": st - t_from_station,
                        "end_time": st,
                        "start_node": DEFAULT_SID,
                        "end_building": a[0]["building"],  # 第一栋楼的id
                        "gps": from_path_gps,  # [(lon, lat), ...] 从快递站到第一栋楼的最短路
                        "xy": from_path_xy,    # [(x, y), ...]
                        "target_orders": a[0]["target_orders"],
                    },
                ]
            else:
                # 回路区, 拣货, 去路区
                ne = self.bid2nid[actions[-1]["building"]]
                ns = self.bid2nid[a[0]["building"]]
                t_to_station = self.distance[ne][DEFAULT_SID] / self.config["v_car"]
                t_from_station = self.distance[DEFAULT_SID][ns] / self.config["v_car"]
                t_rest = st - (actions[-1]["end_time"] + t_to_station + t_from_station)
                assert t_rest > 0, f"下午开始时间过早，快递员无法到达"
                to_path_gps, to_path_xy = self._get_path(ne, DEFAULT_SID)
                from_path_gps, from_path_xy = self._get_path(DEFAULT_SID, ns)
                actions += [
                    {
                        "type": ACTION_TOSTATION,
                        "start_time": actions[-1]["end_time"],
                        "end_time": actions[-1]["end_time"] + t_to_station,
                        "start_building": actions[-1]["building"],  # 最后一栋楼的id
                        "end_node": DEFAULT_SID,
                        "gps": to_path_gps,  # [(lon, lat), ...] 从最后一栋楼到快递站的最短路
                        "xy": to_path_xy,  # [(x, y), ...]
                        "target_orders": [],
                    },
                    {
                        "type": ACTION_ARRANGE,
                        "start_time": actions[-1]["end_time"] + t_to_station,
                        "end_time": actions[-1]["end_time"] + t_to_station + t_rest,
                        "station_id": DEFAULT_SID,
                        "gps": self.G.nodes[DEFAULT_SID]["gps"],
                        "xy": self.G.nodes[DEFAULT_SID]["xy"],
                        "target_orders": [],
                    },
                    {
                        "type": ACTION_FROMSTATION,
                        "start_time": actions[-1]["end_time"] + t_to_station + t_rest,
                        "end_time": st,
                        "start_node": DEFAULT_SID,
                        "end_building": a[0]["building"],  # 第一栋楼的id
                        "gps": from_path_gps,  # [(lon, lat), ...] 从快递站到第一栋楼的最短路
                        "xy": from_path_xy,  # [(x, y), ...]
                        "target_orders": a[0]["target_orders"],
                    }
                ]

            actions += a
        
        # 最后回站
        ne = self.bid2nid[actions[-1]["building"]]
        t_to_station = self.distance[ne][DEFAULT_SID] / self.config["v_car"]
        to_path_gps, to_path_xy = self._get_path(ne, DEFAULT_SID)
        actions.append(
            {
                "type": ACTION_TOSTATION,
                "start_time": actions[-1]["end_time"],
                "end_time": actions[-1]["end_time"] + t_to_station,
                "start_building": actions[-1]["building"],  # 最后一栋楼的id
                "end_node": DEFAULT_SID,
                "gps": to_path_gps,  # [(lon, lat), ...] 从最后一栋楼到快递站的最短路
                "xy": to_path_xy,  # [(x, y), ...]
                "target_orders": [],
            }
        )
        
        # 后处理, 检查合法
        actions = post_process(actions)
        check_results(self.G, self.buildings, actions)
        last_te = actions[0]["start_time"]  # 避免由于浮点数精度问题带来的action时间不首尾相接(已经检查过事实上是首尾相接)
        for a in actions:
            a["start_time"] = last_te
            last_te = a["end_time"]

        # 检查输入输出中的订单一致
        counter1 = Counter([o["type"] for x in st_odr_seq_types for o in x[1]])
        counter2 = Counter([a["target_orders"][0]["type"] for a in actions if a["type"] in ACTION_ORDER])
        for otp in [ORDER_DELIVER, ORDER_CPICK, ORDER_BPICK]:
            assert counter1[otp] == counter2[otp], (
                list(counter1.items()),
                list(counter2.items()),
            )

        return actions


def post_process(actions):
    # 计算移动路线长度
    ACTION_MOVE = {ACTION_WALK, ACTION_FROMSTATION, ACTION_TOSTATION}
    for action in actions:
        if action["type"] in ACTION_MOVE:
            if len(action["xy"]) == 1:
                x, y = action["xy"][0]
                action["xy"] = [(x, y), (x + 1, y + 1), (x, y)]
                action["gps"] = [projector(*p, inverse=True) for p in action["xy"]]
            action["length"] = LineString(action["xy"]).length

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
                if o["target_time"] < o["start_time"]:
                    # print("adjust order start_time ahead of target_time,", o["type"])
                    t_advance = (
                        T_DELIVER_ADVANCE
                        if o["type"] == ORDER_DELIVER
                        else T_PICK_ADVANCE
                    )
                    o["start_time"] = min(o["start_time"], o["target_time"] - t_advance)

    # # 调整派件单开始时间到上一次arrange的时段内
    st, et = None, None
    for a in actions:
        if a["type"] == ACTION_ARRANGE:
            st, et = a["start_time"], a["end_time"]
        if st is not None:
            for o in a["target_orders"]:
                if o["type"] == ORDER_DELIVER and o["served_time"] > et:
                    if not st <= o["start_time"] <= et:
                        o["start_time"] = random.uniform(st, et)

    # # 调整派件单开始时间, 使得第二波送的货在第一波回站后才产生
    # for action in actions:
    #     if action["type"] == ACTION_TOSTATION:
    #         t_back1_st = action["start_time"]
    #         t_back1_et = action["end_time"]
    #         break
    # else:
    #     assert False
    # cnt = 0
    # for action in actions:
    #     if action["type"] == ACTION_FROMSTATION:
    #         cnt += 1
    #         if cnt == 2:
    #             t_from2_st = action["start_time"]
    #             break
    # else:
    #     assert False
    # for action in actions:
    #     for o in action["target_orders"]:
    #         if (
    #             o["type"] == ORDER_DELIVER
    #             and o["served_time"] > t_back1_st
    #             and o["start_time"] < t_back1_et
    #         ):
    #             o["start_time"] = (t_back1_et + t_from2_st) / 2

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
        if atype == ACTION_WALK:
            var = {"traveled_length": act["length"]}
        elif atype == ACTION_UPSTAIR:  # 坐电梯或下楼不计爬楼层数
            var = {"climbed_floors": act["num"]}
        elif atype == ACTION_DELIVER:
            on_time = action["end_time"] <= action["target_orders"][0]["ddl_time"]
            return {
                "delivered_orders": 1,
                "delivered_on_time": 1 if on_time else 0,
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
    }
    actions.append(
        {
            "type": ACTION_REST,
            "start_time": actions[-1]["end_time"],
            "end_time": actions[-1]["end_time"] + 0.1,
            "target_orders": [],
        }
    )
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

    return actions


def check_results(G, buildings, actions):
    """
    检查action序列的合法性
    """
    last_te = None
    for i, action in enumerate(actions):
        assert action["end_time"] > action["start_time"], ("时长不为正", action)
        if last_te:
            try:
                assert abs(action["start_time"] - last_te) < 0.1, ("时间不首尾相接", action)
            except:
                print(action["start_time"], last_te)
                pprint_actions(actions[i - 20 : i + 1])
                exit()
        last_te = action["end_time"]
        last_a = action

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
    for i, action in enumerate(actions):
        if action["type"] in ACTION_MOVE:
            start_pos, end_pos = get_start_end_pos(action)
            if last_pos and get_dis(last_pos, start_pos) >= 1:
                print("位置不首尾相接")
                for a in actions[max(0, i - 3) : i + 1]:
                    print(a)
                assert False
            last_pos = end_pos

    ACTION_ORDER = {ACTION_DELIVER, ACTION_CPICK, ACTION_BPICK}
    ACTION_FLOOR = {ACTION_ELEVATOR, ACTION_UPSTAIR, ACTION_DOWNSTAIR}
    for action in actions:
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


def pprint_actions(actions):
    pprint(
        [
            {
                i: j
                for i, j in a.items()
                if i not in {"gps", "xy", "support_points", "target_orders", "status"}
            }
            for a in actions
        ]
    )


if __name__ == "__main__":
    from evaluate import add_stall_to_map
    from seq_model_nn import SeqModelNN

    def stat_bid2seq(train_data):
        """统计历史访问楼的平均顺序"""
        cid2bid2seq = {}
        for cid, train in train_data.items():
            bid2seqs = defaultdict(list)
            for w in train:
                bids = [o["building_id"] for o in w["orders"]]
                n = len(bids)
                bid2seq = defaultdict(list)
                for i, bid in enumerate(bids):
                    bid2seq[bid].append(i / n)
                for bid, ss in bid2seq.items():
                    bid2seqs[bid].append(np.mean(ss))  # 一波中, bid出现次序的平均
            cid2bid2seq[cid] = {bid: np.mean(ss) for bid, ss in bid2seqs.items()}  # 所有波的平均
        return cid2bid2seq

    def params2config(ps, motivation):
        config = deepcopy(ps)
        config["t_elevator_floor"] = 3.0
        config["v_car"] = 8.0
        config["t_stair_up"] = 2 / 3 * ps["t_stair"]
        config["t_stair_down"] = 1 / 3 * ps["t_stair"]
        config["t_todoor"] = 5.0
        config["t_rest_per_package"] = 5.0
        config["t_elevator_wait2"] = config["t_elevator_wait1"] / 2
        config["motivation"], config["n_avg"] = motivation
        return config
    
    path = f"{DPT}/data/eval_datas_{len(TRAIN_DATES)}_{len(TEST_DATES)}.pkl"
    train_data, test_data, cid2stall_info, bellman_ford = pickle.load(open(path, "rb"))

    G, buildings = add_stall_to_map(G, buildings, cid2stall_info)

    if SEQ_TYPE == "nn":
        dist = defaultdict(dict)
        for bid1, b1 in buildings.items():
            d = dist[bid1]
            db = bellman_ford[b1["gate_id"]]
            for bid2, b2 in buildings.items():
                d[bid2] = db[b2["gate_id"]]
        train_cid2wodrs = {
                cid: [w["orders"] for w in waves] 
                for cid, waves in train_data.items()
            }
        seq_nn = SeqModelNN(
            dist=dist, 
            cid2wodrs=train_cid2wodrs,
            cache_dataset=f"{DPT}/data/seq_nn_datasets_{len(TRAIN_DATES)}_{len(TEST_DATES)}.pkl")
        seq_nn.train(cache_model=CACHE_SEQ_NN)
    else:
        seq_nn = None

    simulator = Simulator(
        G=G, 
        buildings=buildings, 
        distance=bellman_ford)

    cid2motivation = pickle.load(open(f"{DPT}/data/cid2motivation_{SEQ_TYPE}.pkl", "rb"))

    cid2ps = pickle.load(open(f"{DPT}/data/bayes_params_{SEQ_TYPE}.pkl", "rb"))
    
    cid2bid2seq = stat_bid2seq(train_data)

    date = "2023-03-09"
    waves = [w for data in (train_data, test_data) for ws in data.values() for w in ws if w["date"] == "2023-03-09"]
    cid2ws = group_by(waves, "cid")

    results = []
    for cid, ws in tqdm(cid2ws.items()):
        simulator.set_config(
            courier_id=cid, 
            courier_config=params2config(cid2ps[cid], cid2motivation[cid]))
        bid2seq = cid2bid2seq[cid]
        st_odr_seq_types = []
        for w in ws:
            seq_type = SEQ_TYPE
            if seq_type == "stat":
                bid2odrs = group_by(w["orders"], "building_id")
                p = sum(len(odrs) for bid, odrs in bid2odrs.items() if bid in bid2seq) / len(w["orders"])
                if p < P_STAT2LKH:
                    print("use LKH seq anyway due to small p:", p)
                    seq_type = "lkh"
            st_odr_seq_types.append([w["wave_traj"][0], w["orders"], seq_type])
        actions = simulator.generate_one_courier_one_day(st_odr_seq_types, bid2seq, seq_nn)
        results.append([cid, actions])
    
    pickle.dump(results, open(f"{DPT}/data/actions_sim_demo_{date}.pkl", "wb"))
    