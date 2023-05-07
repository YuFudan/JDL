"""
自动贝叶斯调参, 调模拟模型参数, 拟合训练集
https://blog.51cto.com/u_15642578/5305814
"""
import os
import random
from collections import defaultdict
from copy import deepcopy
from multiprocessing import Pool
from pprint import pprint

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import yaml
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

from constants_mxl import *
from gen_sim_actions import Simulator
from validate.evaluate import (cal_gt_metric, cal_sim_metric,
                               merge_one_day_metrics, print_table)

DO_FILTER = False  # 训练时设为False, 生成结果时分别False, True


# 训练集
train_data = pickle.load(open("validate/data/train_data_mxl.pkl", "rb"))
train_metrics = sum([[cal_gt_metric(x) for x in v] for v in train_data.values()], [])
train_metrics = {(x["cid"], x["date"], x["wave_idx"]): x for x in train_metrics}
print("train courier num:", len(train_data))
# 测试集
if DO_FILTER:
    test_data = pickle.load(open("validate/data/test_data_mxl.pkl", "rb"))
else:
    test_data = pickle.load(open("validate/data/test_data_nofilter_mxl.pkl", "rb"))

# TODO:
test_data = pickle.load(open("validate/data/test_data_nofilter_mxl_9.pkl", "rb"))

test_metrics = sum([[cal_gt_metric(x) for x in v] for v in test_data.values()], [])
test_metrics = {(x["cid"], x["date"], x["wave_idx"]): x for x in test_metrics}
test_cids = list(test_data.keys())
print("test courier num:", len(test_data))
assert all(k in train_data for k in test_data)
# 摆摊信息统计
cid2stall_info = pickle.load(open("validate/data/cid2stall_info_mxl.pkl", "rb"))
STALL_BID_OFFSET = 1e6  # 从摆摊location_id映射到bid
# 小哥积极性统计
cid2motivation = pickle.load(open("validate/data/cid2motivation.pkl", "rb"))

random.seed(233)

def process_for_stall(G, buildings, cid2stall_info):
    # 添加新bd新node
    nodes_new = []
    buildings_new = {}
    for cid, (bid2p_lid, lid2gps_time) in cid2stall_info.items():
        for lid, (gps, time) in lid2gps_time.items():
            bid = int(STALL_BID_OFFSET + lid)
            xy = projector(*gps)
            x, y = xy
            xys = [(x-1, y-1), (x+1, y-1), (x+1, y+1), (x-1, y+1), (x-1, y-1)]
            gpss = [projector(*p, inverse=True) for p in xys]
            # 找出楼所在的路区
            p = Point(xy)
            dis_ids = []
            for r in regions.values():
                dis = p.distance(r["poly"])
                if dis == 0:
                    rid = r["id"]
                    break
                dis_ids.append((dis, r["id"]))
            else:
                rid = min(dis_ids, key=lambda x:x[0])[1]
            
            buildings_new[bid] = {
                "id": bid,
                "gate_id": str(bid),
                "name": "摆摊点",
                'is_elevator': False,
                "floor": 7,
                "points": gpss,
                "poly": Polygon(xys),
                "gate_gps": gps,
                "gate_xy": xy,
                "point": Point(xy),
                "region": rid,
                "stall_time": time
            }
            nodes_new.append((str(bid), {
                "id": str(bid),
                "gps": gps,
                "xy": xy
            }))

    # 连接到最近的node, 添加新edge
    edges_new = []
    nodes = {nid: n for nid, n in G.nodes(data=True)}
    for nid, node in nodes_new:
        x, y = node["xy"]
        dis_n1s = []
        for nid1, n1 in nodes.items():
            x1, y1 = n1["xy"]
            gps1 = n1["gps"]
            dis2 = (x - x1) ** 2 + (y - y1) ** 2
            if dis2 < 100:
                edges_new.append({
                    "od": (nid, nid1),
                    "points": [node["gps"], gps1] 
                })
                break
            else:
                dis_n1s.append((dis2, n1))
        else:
            dis2, n1 = min(dis_n1s, key=lambda x: x[0])
            # print("nearset dis2:", dis2)
            edges_new.append({
                "od": (nid, n1["id"]),
                "points": [node["gps"], n1["gps"]] 
            })
    edges_new = edges_new + [
        {
            "od": (road["od"][1], road["od"][0]),
            "points": road["points"][::-1]
        } for road in edges_new
    ]
    edges_new = [
        (
            road["od"][0],
            road["od"][1],
            {
                "od": road["od"],
                "gps": road["points"],
                "xy": [projector(*p) for p in road["points"]],
            }
        ) for road in edges_new
    ]
    for e in edges_new:
        e[2]["length"] = LineString(e[2]["xy"]).length
    G.add_nodes_from(nodes_new)
    G.add_edges_from(edges_new)
    buildings.update(buildings_new)
    
    return G, buildings

G, buildings = process_for_stall(G, buildings, cid2stall_info)
# bellman_ford = {i: j for i, j in nx.all_pairs_bellman_ford_path_length(G, weight="length")}
# pickle.dump(bellman_ford, open("data_mxl/G_bellman_ford.pkl", "wb"))
bellman_ford = pickle.load(open("data_mxl/G_bellman_ford.pkl", "rb"))

# 将订单按bid的摆摊比例, 改变bid到摆摊点 TODO:
if not DO_FILTER:  # 确保为False
    oid2bid_stall = {}
    for data in [train_data, test_data]:
        for cid, waves in data.items():
            bid2p_lid, lid2gps_time = cid2stall_info[cid]
            for w in waves:
                for o in w["orders"]:
                    if o["building_id"] in bid2p_lid:
                        p, lid = bid2p_lid[o["building_id"]]
                        if random.random() < p:
                            # o["building_id"] = int(STALL_BID_OFFSET + lid)
                            oid2bid_stall[o["id"]] = int(STALL_BID_OFFSET + lid)
    # pickle.dump(oid2bid_stall, open("validate/data/oid2bid_stall.pkl", "wb"))
# oid2bid_stall = pickle.load(open("validate/data/oid2bid_stall.pkl", "rb"))
print("stall num:", len(oid2bid_stall))
stall_cnt = 0
for data in [train_data, test_data]:
    for cid, waves in data.items():
        bid2p_lid, lid2gps_time = cid2stall_info[cid]
        for w in waves:
            for o in w["orders"]:
                if o["building_id"] in bid2p_lid:
                    if o["id"] in oid2bid_stall:
                        o["building_id"] = oid2bid_stall[o["id"]]
                        stall_cnt += 1
print("true stall num:", stall_cnt)

def get_seq(train_data):
    """
    从历史数据学送单顺序
    """
    cid2bid2seq = {}
    for cid, train in train_data.items():
        bid2seqs = defaultdict(list)
        for w in train:
            bids = [
                o["building_id"] 
                for o in sorted(w["orders"], key=lambda x: x["finish_time"])
                # if o["type"] == ORDER_DELIVER
            ]
            n = len(bids)
            bid2seq = defaultdict(list)
            for i, bid in enumerate(bids):
                bid2seq[bid].append(i / n)
            for bid, ss in bid2seq.items():
                bid2seqs[bid].append(np.mean(ss))  # 一波中, bid出现次序的平均
        cid2bid2seq[cid] = {bid: np.mean(ss) for bid, ss in bid2seqs.items()}  # 所有波的平均
    return cid2bid2seq

cid2bid2seq = get_seq(train_data)

simulator = Simulator(G, buildings, bellman_ford)
global target_cid

# 初始搜索空间分布
space = {
    't_elevator_wait1': hp.uniform('t_elevator_wait1', 10.0, 25.0),
    # 't_elevator_wait2': hp.uniform('t_elevator_wait2', 5.0, 15.0),  # 假设为wait1的一半
    # 't_stair_up': hp.uniform('t_stair_up', 14.0, 28.0),
    # 't_stair_down': hp.uniform('t_stair_down', 7.0, 14.0),
    't_stair': hp.uniform('t_stair', 20.0, 45.0),    # 上楼数总是等于下楼数, 简并成1个参数即可
    # "t_todoor": hp.uniform('t_todoor', 3.0, 8.0),  # 每次送单前总是有1次todoor, 则不用调, 调了送单时间相当于调了这个
    't_deliver': hp.uniform('t_deliver', 20.0, 120.0),
    't_bpick': hp.uniform('t_bpick', 20.0, 100.0),
    't_cpick': hp.uniform('t_cpick', 20.0, 100.0),
    "t_between_units": hp.uniform('t_between_units', 5.0, 20.0),
    # "t_rest_per_package": hp.uniform('t_rest_per_package', 5.0, 30.0),  # 简并到每单的时间中

    "v_walk": hp.uniform('v_walk', 0.9, 3.5),
    # "walk_type": hp.choice("walk_type", [
    #     {"walk_type": "walk", "v_walk": hp.uniform("v_walk_walk", 1.2, 1.4)},
    #     {"walk_type": "drive", "v_walk": hp.uniform("v_walk_drive", 10, 15)},
    #     ]),

    # "t_cpick_reserve": hp.loguniform('t_cpick_reserve', np.log(100), np.log(1800)),
    # "t_bpick_reserve": hp.loguniform('t_bpick_reserve', np.log(100), np.log(1800)),
    "t_cpick_reserve": hp.uniform('t_cpick_reserve', 1800, 7200),
    "t_bpick_reserve": hp.uniform('t_bpick_reserve', 1800, 7200),

    "t_stall_multiplier": hp.uniform('t_stall_multiplier', 0.8, 1.2),  # 放缩摆摊每单用时历史统计值
    "motivation_multiplier": hp.uniform('motivation_multiplier', 0.5, 1.5),  # 放缩小哥积极性参数历史统计值
}


def params2config(ps, cid):
    config = deepcopy(ps)
    config["t_elevator_floor"] = 3.0
    config["v_car"] = 8.0
    config["t_stair_up"] = 2 / 3 * ps["t_stair"]
    config["t_stair_down"] = 1 / 3 * ps["t_stair"]
    config["t_todoor"] = 5.0
    config["t_rest_per_package"] = 5.0
    config["t_elevator_wait2"] = config["t_elevator_wait1"] / 2
    coef, n_avg = cid2motivation[cid]
    config["motivation"] = coef  # 积极性: 任务量对时长的修正系数
    config["n_avg"] = n_avg      # 参考单量
    return config


def pprint_ps(ps):
    pprint({k: round(v, 2) for k, v in ps.items()})


def post_process_for_validate(actions):
    # 计算移动路线长度
    ACTION_MOVE = {ACTION_WALK, ACTION_FROMSTATION, ACTION_TOSTATION}
    for action in actions:
        if action["type"] in ACTION_MOVE:
            if len(action["xy"]) == 1:
                x, y = action["xy"][0]
                action["xy"] = [(x, y), (x+1, y+1), (x, y)]
                action["gps"] = [projector(*p, inverse=True) for p in action["xy"]]
            action["length"] = LineString(action["xy"]).length

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

    return actions


def run_sim(wave):
    global target_cid
    bid2seq = cid2bid2seq[target_cid]
    orders = wave["orders"]
    bid2odrs = defaultdict(list)
    for o in orders:
        bid2odrs[o["building_id"]].append(o)
    p = sum(len(odrs) for bid, odrs in bid2odrs.items() if bid in bid2seq) / len(orders)
    if p < 0.45:
        print("use LKH seq anyway due to small p:", p)
    simulator.set_use_bid2seq(p > 0.45)  # 若过半的单所在bid, 没有统计得到的送单顺序信息, 则不指定送单顺序
    actions = simulator._generate_half_day(
        start_time=wave["wave_traj"][0],
        orders=wave["orders"])
    actions = post_process_for_validate(actions)
    return {
        "cid": wave["cid"],
        "date": wave["date"],
        "wave_idx": wave["wave_idx"],
        "actions": actions
    }


def objective(params, mode="train", return_actions=False, print_loss=False, merge_wave=True):
    """
    输入模拟模型参数
        跑模拟模型
        计算指标差异
    输出loss
    """
    global target_cid
    data = train_data if mode == "train" else test_data
    gt_metrics = train_metrics if mode == "train" else test_metrics
    
    config = params2config(params, target_cid)
    # if mode == "train":
    #     config['v_walk'] = config['walk_type']['v_walk']
    # else:
    #     config['v_walk'] = config['v_walk_walk'] if 'v_walk_walk' in config else config['v_walk_drive']
    bid2seq = cid2bid2seq[target_cid]

    # 跑模拟模型
    simulator.set_config(config, bid2seq)
    waves = data[target_cid]
    sim_actions = process_map(run_sim, waves, chunksize=1, max_workers=min(48, len(waves)), disable=True)
    sim_actions = [x for x in sim_actions if x]
    assert sim_actions
    sim_metrics = cal_sim_metric(sim_actions)
    sim_metrics = {(x["cid"], x["date"], x["wave_idx"]): x for x in sim_metrics}

    if merge_wave:
        gt_metrics, sim_metrics = map(merge_one_day_metrics, [list(gt_metrics.values()), list(sim_metrics.values())])
        gt_metrics = {(x["cid"], x["date"], x["wave_idx"]): x for x in gt_metrics}
        sim_metrics = {(x["cid"], x["date"], x["wave_idx"]): x for x in sim_metrics}
        
    # 计算loss
    e, c, cw, b, bw = [], [], [], [], []
    for (cid, date, wid), msim in sim_metrics.items():
        mgt = gt_metrics[(cid, date, wid)]
        dnum, cnum, bnum = mgt["dcbnum"]
        cotsim, botsim = msim["cbotnum"]
        cotgt, botgt = mgt["cbotnum"]
        dcotr = 100 * (cotsim - cotgt) / (cnum + 1e-12)
        dbotr = 100 * (botsim - botgt) / (bnum + 1e-12)
        n = dnum + cnum + bnum
        eff_gt = 3600 * n / (mgt["wt"] + 1e-12)
        eff_sim = 3600 * n / (msim["wt"] + 1e-12)
        deff_rel = 100 * (eff_sim - eff_gt) / (eff_gt + 1e-12)
        e.append(deff_rel)
        c.append(dcotr)
        cw.append(cnum)
        b.append(dbotr)
        bw.append(bnum)
    e, c, cw, b, bw = np.array(e), np.array(c), np.array(cw), np.array(b), np.array(bw)
    e_mae = np.mean(np.abs(e))
    c_mae = (np.abs(c) @ cw) / (cw.sum() + 1e-12)
    b_mae = (np.abs(b) @ bw) / (bw.sum() + 1e-12)
    loss = e_mae + c_mae + b_mae

    if print_loss:
        e_me = np.mean(e)
        c_me = (c @ cw) / (cw.sum() + 1e-12)
        b_me = (b @ bw) / (bw.sum() + 1e-12)
        columns = [" ", "效率", "C揽及时率", "B揽及时率"]
        lines = [
            ["MAE", e_mae, c_mae, b_mae],
            ["ME", e_me, c_me, b_me],
        ]
        print_table(columns, lines)

    if return_actions:
        return {"loss": loss, "status": STATUS_OK}, sim_actions
    return {"loss": loss, "status": STATUS_OK}


def get_logs(trials):
    logs = []
    for log in trials.trials:
        if log["result"]["status"] != STATUS_OK:
            continue
        loss = log["result"]["loss"]
        vals = {k: v[0] for k, v in log["misc"]["vals"].items()}
        logs.append([loss, vals])
    return logs


def plot_logs(logs, path):
    plt.figure()
    idxs = list(range(len(logs)))
    losses, valss = zip(*logs)
    plt.plot(idxs, losses, label="loss")
    best_loss = min(losses)
    best_idx = losses.index(best_loss)
    plt.axvline(x=best_idx, ymin=0, ymax=best_loss, c="black", ls="--", lw=1)
    plt.axhline(y=best_loss, xmin=0, xmax=best_idx, c="black", ls="--", lw=1)
    
    keys = ["t_elevator_wait1", "t_stair", "t_deliver", "t_bpick", "t_cpick", "t_cpick_reserve", "t_bpick_reserve"]
    for k in keys:
        if k in ["t_cpick_reserve", "t_bpick_reserve"]:
            multiple = 0.1
        else:
            multiple = 1
        plt.plot(idxs, [x[k] * multiple for x in valss], label=k.split("t_")[1])
    plt.legend()
    plt.savefig(path)


def train():
    global target_cid
    cid2best_loss = {cid: 1e10 for cid in test_cids}
    cid2best_params = {}
    for nn in range(10):  # 跑10次, 希望由于不同的随机初值带来更好的结果
        print("random try:", nn)
        random.seed(nn)
        for target_cid in test_cids:
            print("target courier:", target_cid, cid2name[target_cid])
            # trials = Trials()
            trials = None
            params = fmin(
                objective, 
                space, 
                algo=tpe.suggest, 
                max_evals=55, 
                trials=trials)
            loss = objective(params)["loss"]
            if loss < cid2best_loss[target_cid]:
                cid2best_loss[target_cid] = loss
                cid2best_params[target_cid] = params
            # pprint(best_params)
            # plot_logs(get_logs(trials), f"validate/figure/bayes_tune_{target_cid}_{cid2name[target_cid]}.png")
        pickle.dump(cid2best_params, open(f"validate/data/bayes_tune_params_random_best_mxl.pkl", "wb"))
        pickle.dump(cid2best_loss, open(f"validate/data/bayes_tune_loss_random_best_mxl.pkl", "wb"))


def regularize():
    global target_cid
    cid2ps = pickle.load(open("validate/data/bayes_tune_params_random_best_mxl.pkl", "rb"))
    cid2loss = pickle.load(open("validate/data/bayes_tune_loss_random_best_mxl.pkl", "rb"))
    cid2ps_rgl = {}
    # 取效果较好的, 计算平均参数
    avg_ps = defaultdict(list)
    for cid, ps in cid2ps.items():
        loss = cid2loss[cid]
        if loss > 40:
            continue
        for k, v in ps.items():
            avg_ps[k].append(v)
    avg_ps = {k: np.mean(v) for k, v in avg_ps.items()}
    pprint(avg_ps)

    # 比较原始参数 和 用平均参数正则后的参数 的loss, 取最好
    for target_cid, ps in cid2ps.items():
        loss = cid2loss[target_cid]
        if loss < 20:  # 已经足够好, 就不正则
            cid2ps_rgl[target_cid] = ps
        else:
            ps_rgl = {k: 0.6 * v + 0.4 * avg_ps[k] for k, v in ps.items()}  # 64开
            loss_rgl = objective(ps_rgl)['loss']
            if loss_rgl < loss:
                print("regularize:", round(loss_rgl), round(loss))
                cid2ps_rgl[target_cid] = ps_rgl
            else:
                cid2ps_rgl[target_cid] = ps
    pickle.dump(cid2ps_rgl, open("validate/data/bayes_tune_params_rgl_mxl.pkl", "wb"))


def handle():
    global target_cid
    # # 找loss大的小哥手调
    # cid2loss = pickle.load(open("validate/data/bayes_tune_loss_random_best_mxl.pkl", "rb"))
    # for cid, loss in cid2loss.items():
    #     print(cid, loss)
    # exit()

    cid2ps = pickle.load(open("validate/data/bayes_tune_params_rgl_mxl.pkl", "rb"))

    # for cid in [20485832, 227164]:
    #     x = cid2ps[cid]
    #     x = {k: round(v, 2) for k, v in x.items()}
    #     pprint(x)
    # exit()

    # # 生成训练集action (DO_FILTER设为False)
    # cid2as = {}
    # for target_cid in tqdm(test_cids):
    #     cid2as[target_cid] = objective(cid2ps[target_cid], return_actions=True)[1]
    # pickle.dump(cid2as, open("validate/data/bayes_tune_train_actions_hdl_mxl.pkl", "wb"))
    # # 生成测试集action
    # cid2as = {}
    # for target_cid in tqdm(test_cids):
    #     cid2as[target_cid] = objective(cid2ps[target_cid], mode="test", return_actions=True)[1]
    # if DO_FILTER:
    #     pickle.dump(cid2as, open("validate/data/bayes_tune_actions_hdl_mxl.pkl", "wb"))
    # else:
    #     pickle.dump(cid2as, open("validate/data/bayes_tune_actions_hdl_nofilter_mxl.pkl", "wb"))
    # exit()

    # target_cid = 20862519  # 郝刚强
    # # ps = cid2ps[target_cid]
    # # pprint_ps(ps)
    # # objective(ps, print_loss=True)
    # ps = {
    #     't_between_units': 15,
    #     't_bpick': 30,
    #     't_bpick_reserve': 7200,
    #     't_cpick': 60,
    #     't_cpick_reserve': 7200,
    #     't_deliver': 35,
    #     't_elevator_wait1': 20,
    #     't_stair': 40,
    #     't_stall_multiplier': 1.0,
    #     'v_walk': 3.2
    # }
    # # objective(ps, print_loss=True)
    # cid2ps[target_cid] = ps
    
    # target_cid = 154889  # 胡亮亮
    # # ps = cid2ps[target_cid]
    # # pprint_ps(ps)
    # # objective(ps, print_loss=True)
    # ps = {
    #     't_between_units': 10,
    #     't_bpick': 70,
    #     't_bpick_reserve': 4000,
    #     't_cpick': 50,
    #     't_cpick_reserve': 7200,
    #     't_deliver': 22,
    #     't_elevator_wait1': 10,
    #     't_stair': 20,
    #     't_stall_multiplier': 0.8,
    #     'v_walk': 1.37}
    # # objective(ps, print_loss=True)
    # cid2ps[target_cid] = ps

    # target_cid = 22607367  # 谭茂学
    # # ps = cid2ps[target_cid]
    # # pprint_ps(ps)
    # # objective(ps, print_loss=True)
    # ps = {
    #     't_between_units': 20,
    #     't_bpick': 80,
    #     't_bpick_reserve': 5400,
    #     't_cpick': 60,
    #     't_cpick_reserve': 7200,
    #     't_deliver': 135,
    #     't_elevator_wait1': 20,
    #     't_stair': 40,
    #     't_stall_multiplier': 0.8,
    #     'v_walk': 1.0}
    # # objective(ps, print_loss=True)
    # cid2ps[target_cid] = ps

    # target_cid = 22469286  # 余飞龙
    # # ps = cid2ps[target_cid]
    # # pprint_ps(ps)
    # # objective(ps, print_loss=True)
    # ps = {
    #     't_between_units': 20,
    #     't_bpick': 80,
    #     't_bpick_reserve': 3988.92,
    #     't_cpick': 50,
    #     't_cpick_reserve': 2084.44,
    #     't_deliver': 130,
    #     't_elevator_wait1': 30,
    #     't_stair': 45,
    #     't_stall_multiplier': 1.2,
    #     'v_walk': 1.0}
    # # objective(ps, print_loss=True)
    # cid2ps[target_cid] = ps
   
    # target_cid = 20485832  # 赵东震
    # # ps = cid2ps[target_cid]
    # # pprint_ps(ps)
    # # objective(ps, print_loss=True)
    # ps = {
    #     't_between_units': 5.16,
    #     't_bpick': 77.7,
    #     't_bpick_reserve': 6749.82,
    #     't_cpick': 78.76,
    #     't_cpick_reserve': 7200,
    #     't_deliver': 30.02,
    #     't_elevator_wait1': 13.52,
    #     't_stair': 30,
    #     't_stall_multiplier': 0.94,
    #     'v_walk': 3.2
    # }
    # # objective(ps, print_loss=True)
    # cid2ps[target_cid] = ps

    # 生成测试集action
    cid2as = {}
    for target_cid in tqdm(test_cids):
        cid2as[target_cid] = objective(cid2ps[target_cid], mode="test", return_actions=True)[1]
    pickle.dump(cid2as, open("validate/data/bayes_tune_actions_hdl_nofilter_mxl_9.pkl", "wb"))  # TODO:
    exit()
    if DO_FILTER:
        pickle.dump(cid2as, open("validate/data/bayes_tune_actions_hdl_mxl.pkl", "wb"))
        pickle.dump(cid2ps, open("validate/data/bayes_tune_params_hdl_mxl.pkl", "wb"))
    else:
        pickle.dump(cid2as, open("validate/data/bayes_tune_actions_hdl_nofilter_mxl.pkl", "wb"))
        pickle.dump(cid2ps, open("validate/data/bayes_tune_params_hdl_nofilter_mxl.pkl", "wb"))
  

if __name__ == "__main__":
    # train()
    
    # regularize()
    
    handle()
