"""
自动贝叶斯调参, 调模拟模型参数, 拟合训练集
https://blog.51cto.com/u_15642578/5305814
"""
import os
import pickle
import random
from collections import defaultdict
from copy import deepcopy
from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from shapely.geometry import LineString
from sklearn import linear_model
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

from evaluate import (add_stall_to_map, cal_gt_metric, cal_sim_metric,
                      merge_one_day_metrics, merge_post)
from gen_sim_actions import Simulator
from seq_model_nn import SeqModelNN

DPT = "hk"  # 指定营业部department
if DPT == "mxl":
    from mxl.constants_all import *
    from mxl.params_eval import *
elif DPT == "hk":
    from hk.constants_all import *
    from hk.params_eval import *
    
random.seed(233)
WORKERS = 32

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


def update_space(params, p=0.25):
    global space
    low, high = 1- p, 1 + p
    for k, v in params.items():
        space[k] = hp.uniform(k, v * low, v * high)


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
    orders = wave["orders"]
    bid2seq = cid2bid2seq[target_cid]
    seq_type = SEQ_TYPE
    if SEQ_TYPE == "stat":
        bid2odrs = defaultdict(list)
        for o in orders:
            bid2odrs[o["building_id"]].append(o)
        p = sum(len(odrs) for bid, odrs in bid2odrs.items() if bid in bid2seq) / len(orders)
        if p < P_STAT2LKH:
            print("use LKH seq anyway due to small p:", p)
            seq_type = "lkh"
    actions = simulator.generate_wave(
        start_time=wave["wave_traj"][0],
        orders=orders,
        seq_type=seq_type,
        bid2seq=bid2seq if seq_type != "lkh" else None,
        seq_nn=seqmodel if seq_type == "nn" else None)
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

    simulator.set_config(
        courier_id=target_cid, 
        courier_config=params2config(params, cid2motivation[target_cid]))
    waves = data[target_cid]

    # sim_actions = [run_sim(w) for w in waves]
    sim_actions = process_map(
        run_sim,
        waves,
        chunksize=1, 
        max_workers=min(WORKERS, len(waves)), 
        disable=True)
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
        _, cotsim, botsim = msim["otnum"]
        _, cotgt, botgt = mgt["otnum"]
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


def get_motivation(metrics_gt, metrics_sim, merge_wave=True):
    """
    估计任务量对小哥积极性的影响
    线性回归:
        横轴: 单量/平均单量
        纵轴: -真值时长/模拟时长 (取相反数以使用正系数回归)
    所得直线斜率(再取相反数回来) 即为 单量波动量 影响 时长修正比例 的系数
    用该系数 乘以 模拟器每个动作的用时参数 即可将模拟时长修正为真值时长
    """
    if merge_wave:
        metrics_gt, metrics_sim = map(merge_one_day_metrics, [metrics_gt, metrics_sim])
        merge_post(metrics_gt, metrics_sim)

    def cid2date_wid2m(ms):
        cid2date_wid2m = defaultdict(dict)
        for x in ms:
            cid2date_wid2m[x["cid"]][(x["date"], x["wave_idx"])] = x
        return cid2date_wid2m
    metrics_gt, metrics_sim = map(cid2date_wid2m, [metrics_gt, metrics_sim])
    
    plt.figure(figsize=(5, len(metrics_gt)*3))
    idx = 1
    cid2coef_navg = {}
    for cid, date_wid2gt in metrics_gt.items():
        date_wid2sim = metrics_sim[cid]
        gt = list(date_wid2gt.values())
        sim = [date_wid2sim[k] for k in date_wid2gt]
        n_avg = np.mean([sum(x["dcbnum"]) for x in gt])
        n_t = [
            [sum(y["dcbnum"]) / n_avg, -y["wt"] / x["wt"]]
            for x, y in zip(sim, gt)
        ]
        X, Y = [np.array(x).reshape(-1, 1) for x in zip(*n_t)]
        reg = linear_model.LinearRegression(positive=True)  # 确保单量与小哥积极性正相关
        mask = np.ones_like(Y, dtype=bool)
        cnt = 0
        while(True):
            X_train, Y_train = X[mask].reshape(-1, 1), Y[mask].reshape(-1, 1)
            reg.fit(X_train, Y_train)
            params = list(reg.coef_) + [reg.intercept_]

            losses = reg.predict(X) - Y
            last_mask = mask
            loss_gate = 0.1
            while(True):
                mask = np.abs(losses) < loss_gate  # 下次只用loss小的训练
                if mask.sum() / len(X) > 0.6:
                    break
                loss_gate += 0.05
            # inlier_losses = reg.predict(X_train) - Y_train
            # print("inlier loss:", np.mean([abs(x) for x in inlier_losses]), np.mean(inlier_losses))
            if np.all(mask == last_mask):  # mask不变则退出
                break

            cnt += 1
            if cnt > 10:
                break
        cid2coef_navg[cid] = [-params[0].tolist()[0], n_avg]

        plt.subplot(len(metrics_gt), 1, idx)
        plt.scatter(X[mask], Y[mask])
        plt.scatter(X[~mask], Y[~mask])
        xs = np.linspace(min(X), max(X), 50)
        ys = [params[0]*x + params[1] for x in xs]
        plt.plot(xs, ys)
        idx += 1
    plt.savefig(f"{DPT}/figure/n_t_relation.png")
    return cid2coef_navg


def train(n_random, n_search):
    global target_cid
    cid2best_loss = {cid: 1e10 for cid in test_cids}
    cid2best_params = {}
    for nn in range(n_random):  # 希望由于不同的随机初值带来更好的结果
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
                max_evals=n_search, 
                trials=trials)
            loss = objective(params)["loss"]
            if loss < cid2best_loss[target_cid]:
                cid2best_loss[target_cid] = loss
                cid2best_params[target_cid] = params
            # plot_logs(get_logs(trials), f"figure/bayes_{target_cid}_{cid2name[target_cid]}.png")
        pickle.dump(cid2best_params, open(f"{DPT}/data/bayes_params_{SEQ_TYPE}.pkl", "wb"))
    return cid2best_params


def main_float(cid2ps_last):
    global target_cid
    cid2ps = {}
    for target_cid in test_cids:
        print("target courier:", target_cid, cid2name[target_cid])
        update_space(cid2ps_last[target_cid])  # 在上一轮窗口的调参结果附近搜索
        params = fmin(
            objective, 
            space, 
            algo=tpe.suggest, 
            max_evals=30, 
            trials=None)
        cid2ps[target_cid] = params
    cid2as = {}
    for target_cid in test_cids:
        cid2as[target_cid] = objective(cid2ps[target_cid], mode="test", return_actions=True)[1]
    return cid2ps, cid2as
    

if __name__ == "__main__":
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
    exit()

    # cid2ps = pickle.load(open(f"{DPT}/data/bayes_params_{SEQ_TYPE}.pkl", "rb"))
    # for cid, ps in cid2ps.items():
    #     print(cid2name[cid])
    #     pprint_ps(ps)
    # exit()

    # # 滑动窗口训练集
    # cid2motivation = pickle.load(open(f"{DPT}/data/cid2motivation.pkl", "rb"))
    # cid2ps_last = pickle.load(open(f"{DPT}/data/bayes_tune_params_rgl.pkl", "rb"))
    # for n in range(1, 30):
    #     print("float", n)

    #     train_data, test_data, cid2stall_info, bellman_ford = pickle.load(open(f"{DPT}/data/eval_datas_float_{n}.pkl", "rb"))
    #     train_metrics = sum([[cal_gt_metric(x, bellman_ford) for x in v] for v in train_data.values()], [])
    #     train_metrics = {(x["cid"], x["date"], x["wave_idx"]): x for x in train_metrics}
    #     test_metrics = sum([[cal_gt_metric(x, bellman_ford) for x in v] for v in test_data.values()], [])
    #     test_metrics = {(x["cid"], x["date"], x["wave_idx"]): x for x in test_metrics}
    #     test_cids = list(test_data.keys())
    #     print("test courier num:", len(test_data))
    #     assert all(k in train_data for k in test_data)
        
    #     cid2bid2seq = stat_bid2seq(train_data)

    #     buildings = pickle.load(open(f"{DPT}/data/buildings.pkl", "rb"))
    #     for bd in buildings:
    #         bd["poly"] = Polygon([projector(*p) for p in bd["points"]])  # 平面投影坐标
    #         bd["point"] = Point(bd["gate_xy"])
    #     buildings = {bd["id"]: bd for bd in buildings}
    #     G = pickle.load(open(f"{DPT}/data/G.pkl", "rb"))
    #     G, buildings = add_stall_to_map(G, buildings, cid2stall_info)
    #     simulator = Simulator(G, buildings, bellman_ford)

    #     cid2ps, cid2as = main_float(cid2ps_last)
    #     cid2ps_last = cid2ps
    #     pickle.dump(cid2ps, open(f"{DPT}/data/bayes_params_float_{n}.pkl", "wb"))
    #     pickle.dump(cid2as, open(f"{DPT}/data/bayes_actions_float_{n}.pkl", "wb"))

    path = f"{DPT}/data/eval_datas_{len(TRAIN_DATES)}_{len(TEST_DATES)}.pkl"
    train_data, test_data, cid2stall_info, bellman_ford = pickle.load(open(path, "rb"))
    train_metrics = sum([[cal_gt_metric(x, bellman_ford) for x in v] for v in train_data.values()], [])
    train_metrics = {(x["cid"], x["date"], x["wave_idx"]): x for x in train_metrics}
    test_metrics = sum([[cal_gt_metric(x, bellman_ford) for x in v] for v in test_data.values()], [])
    test_metrics = {(x["cid"], x["date"], x["wave_idx"]): x for x in test_metrics}
    test_cids = list(test_data.keys())
    print("test courier num:", len(test_data))
    assert all(k in train_data for k in test_data)

    cid2bid2seq = stat_bid2seq(train_data)

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
        seqmodel = SeqModelNN(
            dist=dist, 
            cid2wodrs=train_cid2wodrs,
            cache_dataset=f"{DPT}/data/seq_nn_datasets_{len(TRAIN_DATES)}_{len(TEST_DATES)}.pkl")
        seqmodel.train(cache_model=CACHE_SEQ_NN)
    else:
        seqmodel = None
    exit()


    simulator = Simulator(
        G=G, 
        buildings=buildings, 
        distance=bellman_ford)

    # 训练: 先不带随单量变化的积极性系数跑, 根据结果估计出积极性系数
    path = f"{DPT}/data/cid2motivation_{SEQ_TYPE}.pkl"
    if os.path.exists(path):
        cid2motivation = pickle.load(open(path, "rb"))
    else:
        cid2motivation = {cid: [0, 100] for cid in test_cids}
        print("Pre Train...")
        cid2ps = train(n_random=1, n_search=35)
        # cid2ps = pickle.load(open(f"{DPT}/data/bayes_params_{SEQ_TYPE}.pkl", "rb"))
        cid2as = {}
        for target_cid in tqdm(test_cids):
            cid2as[target_cid] = objective(cid2ps[target_cid], return_actions=True)[1]
        train_sim_metrics = cal_sim_metric(sum(cid2as.values(), []))
        cid2motivation = get_motivation(list(train_metrics.values()), train_sim_metrics)
        pickle.dump(cid2motivation, open(path, "wb"))

    # 正式训练
    cid2ps = train(n_random=1, n_search=50)
    # cid2ps = pickle.load(open(f"{DPT}/data/bayes_params_{SEQ_TYPE}.pkl", "rb"))

    # # 手调参数
    # target_cid = 20862519
    # ps = cid2ps[target_cid]
    # pprint_ps(ps)
    # objective(ps, print_loss=True)
    # ps = {...}
    # objective(ps, print_loss=True)
    # cid2ps[target_cid] = ps
    
    # 生成测试集actions
    cid2as = {}
    for target_cid in tqdm(test_cids):
        cid2as[target_cid] = objective(cid2ps[target_cid], mode="test", return_actions=True)[1]
    pickle.dump(cid2as, open(f"{DPT}/data/sim_actions_{SEQ_TYPE}.pkl", "wb"))

    # 生成训练集actions
    cid2as = {}
    for target_cid in tqdm(test_cids):
        cid2as[target_cid] = objective(cid2ps[target_cid], return_actions=True)[1]
    pickle.dump(cid2as, open(f"{DPT}/data/train_sim_actions_{SEQ_TYPE}.pkl", "wb"))
