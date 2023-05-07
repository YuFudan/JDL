
import random
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties
from tqdm import tqdm

from auto_bayes_tune import (params2config, post_process_for_validate,
                             stat_bid2seq)
from evaluate import add_stall_to_map
from gen_sim_actions import Simulator
from seq_model_nn import SeqModelNN

DPT = "mxl"  # 指定营业部department
if DPT == "mxl":
    from mxl.constants_all import *
    from mxl.params_eval import *
elif DPT == "hk":
    from hk.constants_all import *
    from hk.params_eval import *

random.seed(233)
np.random.seed(233)


def run_wave(wave):
    orders = wave["orders"]
    cid = wave["cid"]
    simulator.set_config(
        courier_id=cid, 
        courier_config=params2config(cid2ps[cid], cid2motivation[cid]))
    bid2seq = cid2bid2seq[cid]
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
        start_time=wave["start_time"],
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


def get_metrics(actions):
    date2cid2as = defaultdict(lambda: defaultdict(list))
    for x in actions:
        date2cid2as[x["date"]][x["cid"]].append(x)
    
    date2ms = {}
    for date, cid2xs in date2cid2as.items():
        wts = []
        dcb_cnt = [0 for _ in range(6)]
        for xs in cid2xs.values():
            wt = 0
            for x in xs:
                actions = x["actions"]
                work_time = actions[-1]["end_time"] - actions[0]["start_time"]
                for a in actions:
                    if a.get("wait_for_pick", False):
                        work_time -= a["end_time"] - a["start_time"]
                wt += work_time
                s = actions[-1]["status"]
                for i, k in enumerate(["delivered_orders", "delivered_on_time",\
                    "cpicked_orders", "cpicked_on_time", "bpicked_orders", "bpicked_on_time"]):
                    dcb_cnt[i] += s[k]
            wts.append(wt)
        wt_std = np.std(wts)
        d, dot, c, cot, b, bot = dcb_cnt
        dotr, cotr, botr = dot / d, cot / c, bot / b
        date2ms[date] = [wt_std, dotr, cotr, botr]
    return date2ms


def plot(date2ms_mdi):
    font = FontProperties(fname=r"msyh.ttc")
    plt.figure()
    for date2ms, color, name in zip(date2ms_mdi, ["green", "orange", "red"], ["规划", "规划+计划", "规划+计划+执行"]):
        std, d, c, b = zip(*date2ms.values())
        std = [x / 60 for x in std]
        d = [x * 100 for x in d]
        c = [x * 100 for x in c]
        b = [x * 100 for x in b]
        for y, marker, name2 in zip([d, c, b], ["x", "o", "^"], ["_派送", "_C揽", "_B揽"]):
            plt.scatter(x=std, y=y, c=color, marker=marker, label=name + name2)
    plt.xlabel("工作时长标准差 (min)", fontproperties=font)
    plt.ylabel("及时率", fontproperties=font)
    plt.legend(prop=font)
    plt.tight_layout()
    plt.savefig(f"{DPT}/figure/validate_replan_order.png")


if __name__ == "__main__":
    # actions_mdi = pickle.load(open(f"{DPT}/data/actions_mdi.pkl", "rb"))
    # date2ms_mdi = [get_metrics(x) for x in actions_mdi]
    # plot(date2ms_mdi)
    # exit()

    path = f"{DPT}/data/eval_datas_{len(TRAIN_DATES)}_{len(TEST_DATES)}.pkl"
    train_data, test_data, cid2stall_info, bellman_ford = pickle.load(open(path, "rb"))
    cidwaves = defaultdict(list)
    for data in [train_data, test_data]:
        for cid, ws in data.items():
            for w in ws:
                cidwaves[cid].append(w)

    cid2bid2seq = stat_bid2seq(cidwaves)

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

    simulator = Simulator(
        G=G, 
        buildings=buildings, 
        distance=bellman_ford)

    cid2motivation = pickle.load(open(f"{DPT}/data/cid2motivation_{SEQ_TYPE}.pkl", "rb"))

    cid2ps = pickle.load(open(f"{DPT}/data/bayes_params_{SEQ_TYPE}.pkl", "rb"))

    cid2waves_mdi = pickle.load(open(f"{DPT}/data/orders_replan_month_day_intime.pkl", "rb"))

    actions_mdi = []
    for cid2waves in cid2waves_mdi:
        actions_mdi.append([run_wave(w) for ws in tqdm(cid2waves.values()) for w in ws])
    pickle.dump(actions_mdi, open(f"{DPT}/data/actions_mdi.pkl", "wb"))
