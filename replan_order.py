"""
日粒度分单
以原始模拟器，按某种算法，得到一个最优的分配方案，计算工作时长方差
    把最长的小哥的单分给最短的小哥，直至收敛
降低模拟器精度，按相同算法得到分配方案，然后用原始模拟器评估，计算工作时长方差
"""
import random
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties
from tqdm import tqdm

from auto_bayes_tune import params2config, stat_bid2seq
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

T_PER_ORDER = 300
T_PER_BD = 600
MAX_ORDER = 20
MAX_BD = 5
random.seed(233)
np.random.seed(233)

class Replan:
    def __init__(self, cid2waves, cid2bid2seq, simulator: Simulator, cid2motivation, cid2ps, seqmodel):
        self.cid2bid2seq = cid2bid2seq
        self.simulator = simulator
        self.cid2motivation = cid2motivation
        self.cid2ps = cid2ps
        self.seqmodel = seqmodel
        self.cid2waves = cid2waves
        self.cid2ms = self.run_all_wave()
        self.cid2t = {k: v[0] for k, v in self.cid2ms.items()}

    def run_all_wave(self):
        return {
            cid: sum(self.run_wave(w) for w in waves if w["orders"])
            for cid, waves in tqdm(self.cid2waves.items())}

    def run_wave(self, wave):
        """跑模拟器, 得到送一波的用时"""
        orders = wave["orders"].values()
        cid = wave["cid"]
        self.simulator.set_config(
            courier_id=cid, 
            courier_config=params2config(self.cid2ps[cid], self.cid2motivation[cid]))
        bid2seq = self.cid2bid2seq[cid]
        seq_type = SEQ_TYPE
        if SEQ_TYPE == "stat":
            bid2odrs = defaultdict(list)
            for o in orders:
                bid2odrs[o["building_id"]].append(o)
            p = sum(len(odrs) for bid, odrs in bid2odrs.items() if bid in bid2seq) / len(orders)
            if p < P_STAT2LKH:
                print("use LKH seq anyway due to small p:", p)
                seq_type = "lkh"
        actions = self.simulator.generate_wave(
            start_time=wave["wave_traj"][0],
            orders=orders,
            seq_type=seq_type,
            bid2seq=bid2seq if seq_type != "lkh" else None,
            seq_nn=self.seqmodel if seq_type == "nn" else None)
        work_time = actions[-1]["end_time"] - actions[0]["start_time"]
        atp2n, atp2not = defaultdict(int), defaultdict(int)
        for a in actions:
            if a.get("wait_for_pick", False):
                work_time -= a["end_time"] - a["start_time"]
            if a["type"] in ACTION_ORDER:
                atp2n[a["type"]] += 1
                if a["end_time"] <= a["target_orders"][0]["ddl_time"]:
                    atp2not[a["type"]] += 1
        return np.array([
            work_time, 
            atp2n[ACTION_DELIVER], atp2not[ACTION_DELIVER],
            atp2n[ACTION_CPICK], atp2not[ACTION_CPICK],
            atp2n[ACTION_BPICK], atp2not[ACTION_BPICK],
        ])

    def step(self):
        """将用时最长的快递员的部分单交换给用时最短的快递员"""
        cid_ts = sorted(self.cid2t.items(), key=lambda x: -x[1])
        c1, t1 = cid_ts[-1]    # 最短的
        ws1 = self.cid2waves[c1]
        for c2, t2 in cid_ts:  # 最长的
            if t2 <= t1:
                return False       
            t_gap = t2 - t1
            ws2 = self.cid2waves[c2]
            wid_odrs = []  # 以楼为基本单位来交换单(记录来自哪波)
            for i, w2 in enumerate(ws2):
                odrs = [o for o in w2["orders"].values() if o["type"] == ORDER_DELIVER]  # 只换派送
                bid2odrs = group_by(odrs, "building_id")
                wid_odrs += [(i, odrs) for odrs in bid2odrs.values()]
            if not wid_odrs:  # 已经没有派送单可分出去, 看下一个最长的
                continue
            wid_odrs.sort(key=lambda x: len(x[1]))  # 优先交换单最少的楼

            exchanges = [wid_odrs[0]]
            nb, no = 1, len(wid_odrs[0][1])
            t_exchange = nb * T_PER_BD + no * T_PER_ORDER
            i = 1
            # 决定交换多少
            while(i < len(wid_odrs) and t_exchange <= t_gap and nb < MAX_BD and no < MAX_ORDER):
                exchanges.append(wid_odrs[i])
                i += 1
                nb += 1
                no += len(wid_odrs[i][1])
                t_exchange = nb * T_PER_BD + no * T_PER_ORDER
            # 进行交换操作
            for wid, odrs in exchanges:
                tmp = ws2[wid]["orders"]
                for o in odrs:
                    del tmp[o["id"]]
                if ws2[wid]["is_morning"]:
                    tmp = ws1[0]["orders"]
                else:
                    tmp = ws1[-1]["orders"]
                for o in odrs:
                    tmp[o["id"]] = o
            return c1, c2
        assert False     

    def solution(self):
        cnt = 0
        best_std = 1e10
        best_cid2waves = None
        best_cid2ms = None
        best_cnt = 0
        while(cnt < 100 and best_cnt < 5):
            print(cnt)
            # pprint(sorted(self.cid2t.items(), key=lambda x: -x[1]))
            std = np.std(list(self.cid2t.values()))
            print("std:", std)
            if std < best_std:
                best_std = std
                best_cid2waves = deepcopy(self.cid2waves)
                best_cid2ms = deepcopy(self.cid2ms)
                best_cnt = 0
            else:
                best_cnt += 1
            r = self.step()
            if r is False:
                print("no more possible exchanges can be made")
                break
            else:
                for c, n in zip(r, ["taker:", "giver:"]):
                    t = self.cid2t[c]
                    ms_new = sum(self.run_wave(w) for w in self.cid2waves[c] if w["orders"])
                    t_new = ms_new[0]
                    print(n, cid2name[c], int(t_new - t))
                    self.cid2t[c] = t_new               
                    self.cid2ms[c] = ms_new               
            cnt += 1
            if cnt == 100:
                print("achieve max iter num")
            if best_cnt == 5:
                print("early stop: last 5 iter no gain")
        print("final std:", best_std)
        return best_std, best_cid2waves, best_cid2ms


def main():
    target_date = "2022-8-23"
    cid2waves = date2cid2waves[target_date]

    for cid, waves in cid2waves.items():
        for w in waves:
            w["orders"] = {o["id"]: o for o in w["orders"]}  # hash, 方便删除
        waves.sort(key=lambda x: x["wave_traj"][0])
        if not waves[0]["is_morning"]:  # 造一个空的wave, 方便插入上午/下午的单
            tmp = {
                "cid": cid,
                "orders": {},
                "wave_traj": [28800, 43200],
                "is_morning": True
            }
            waves = [tmp] + waves
        if waves[-1]["is_morning"]:
            tmp = {
                "cid": cid,
                "orders": {},
                "wave_traj": [57600, 72000],
                "is_morning": False
            }
            waves = waves + [tmp]
        cid2waves[cid] = waves

    # # 用劣化版模拟器跑出重分配方案
    # keys1 = {"t_elevator_wait1", "t_stair", "t_deliver", "t_bpick", "t_cpick", "t_between_units"}
    # keys2 = {"v_walk"}
    # cid2waves_errs = []
    # for i in range(7):
    #     print(i)
    #     cid2ps_err = deepcopy(cid2ps)
    #     if i > 0:
    #         coef = i * 0.05  # 放缩小哥参数的系数
    #         for ps in cid2ps_err.values():
    #             p = 1 + coef if random.random() < 0.5 else 1 - coef
    #             for k in ps:
    #                 if k in keys1:
    #                     ps[k] *= p
    #                 elif k in keys2:
    #                     ps[k] /= p
    #     replan = Replan(
    #         cid2waves=deepcopy(cid2waves),
    #         cid2bid2seq=cid2bid2seq,
    #         simulator=simulator,
    #         cid2motivation=cid2motivation,
    #         cid2ps=cid2ps_err,
    #         seqmodel=seqmodel)
    #     std, cid2waves_err = replan.solution()[:2]
    #     cid2waves_errs.append(cid2waves_err)
    # pickle.dump(cid2waves_errs, open(f"{DPT}/data/cid2waves_errs.pkl", "wb"))
    # # cid2waves_errs = pickle.load(open(f"{DPT}/data/cid2waves_errs.pkl", "rb"))

    # # 用原版模拟器评估各方案
    # stds = []
    # for i, cid2waves_err in enumerate(pickle.load(open(f"{DPT}/data/cid2waves_errs.pkl", "rb"))):
    #     print(i)
    #     replan = Replan(
    #         cid2waves=cid2waves_err,
    #         cid2bid2seq=cid2bid2seq,
    #         simulator=simulator,
    #         cid2motivation=cid2motivation,
    #         cid2ps=cid2ps,
    #         seqmodel=seqmodel)
    #     cid2ms = replan.run_all_wave()
    #     std = np.std([x[0] for x in cid2ms.values()])
    #     stds.append(std)
    #     print("std:", std)
    # pickle.dump(stds, open(f"{DPT}/data/replan_order_stds.pkl", "wb"))

    stds = pickle.load(open(f"{DPT}/data/replan_order_stds.pkl", "rb"))
    cid2t = Replan(
        cid2waves=deepcopy(cid2waves),
        cid2bid2seq=cid2bid2seq,
        simulator=simulator,
        cid2motivation=cid2motivation,
        cid2ps=cid2ps,
        seqmodel=seqmodel).cid2t
    orig_std = np.std(list(cid2t.values()))
    gains = [(1 - x / orig_std) * 100 for x in stds]
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.hlines(y=orig_std / 60, xmin=0, xmax=len(stds)-1, linestyles='--', label="Original STD")
    plt.plot(range(len(stds)), [x / 60 for x in stds], label="Optimized STD")
    plt.xticks(range(len(stds)), [f"{5*i}%" for i in range(len(stds))])
    plt.xlabel("Simulator Error")
    plt.ylabel("Work Time STD (min)")
    plt.grid()
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(range(len(gains)), gains)
    plt.xticks(range(len(gains)), [f"{5*i}%" for i in range(len(gains))])
    plt.xlabel("Simulator Error")
    plt.yticks(range(-50, 101, 25), [f"{x}%" for x in range(-50, 101, 25)])
    plt.ylabel("Gain")
    plt.grid()

    plt.tight_layout()
    plt.savefig(f"{DPT}/figure/replan_order_stds.png")


def main2():
    def get_date_ms(cid2ms):
        std = np.std([ms[0] for ms in cid2ms.values()])
        d, dot, c, cot, b, bot = sum([ms[1:] for ms in cid2ms.values()])
        return [std, dot / d, cot / c, bot / b]

    # date2ms_orig, date2ms_rep = {}, {}
    # for target_date in [f"2022-8-{x}" for x in range(22, 32)]:
    #     print(target_date)
    #     cid2waves = date2cid2waves[target_date]
    #     for cid, waves in cid2waves.items():
    #         for w in waves:
    #             w["orders"] = {o["id"]: o for o in w["orders"]}  # hash, 方便删除
    #         waves.sort(key=lambda x: x["wave_traj"][0])
    #         if not waves[0]["is_morning"]:  # 造一个空的wave, 方便插入上午/下午的单
    #             tmp = {
    #                 "cid": cid,
    #                 "orders": {},
    #                 "wave_traj": [28800, 43200],
    #                 "is_morning": True
    #             }
    #             waves = [tmp] + waves
    #         if waves[-1]["is_morning"]:
    #             tmp = {
    #                 "cid": cid,
    #                 "orders": {},
    #                 "wave_traj": [57600, 72000],
    #                 "is_morning": False
    #             }
    #             waves = waves + [tmp]
    #         cid2waves[cid] = waves
        
    #     rep = Replan(
    #         cid2waves=deepcopy(cid2waves),
    #         cid2bid2seq=cid2bid2seq,
    #         simulator=simulator,
    #         cid2motivation=cid2motivation,
    #         cid2ps=cid2ps,
    #         seqmodel=seqmodel)
    #     cid2ms_orig = deepcopy(rep.cid2ms)
    #     cid2ms_rep = rep.solution()[-1]
    #     date2ms_orig[target_date] = get_date_ms(cid2ms_orig)
    #     date2ms_rep[target_date] = get_date_ms(cid2ms_rep)
    # pickle.dump([date2ms_orig, date2ms_rep], open(f"{DPT}/data/replan_order_date2ms.pkl", "wb"))

    date2ms_orig, date2ms_rep = pickle.load(open(f"{DPT}/data/replan_order_date2ms.pkl", "rb"))
    font = FontProperties(fname=r"msyh.ttc")
    plt.figure()
    for date2ms, color, name in zip([date2ms_orig, date2ms_rep], ["green", "red"], ["规划", "规划+计划+执行"]):
        std, d, c, b = zip(*date2ms.values())
        std = [x / 60 for x in std]
        d = [x * 100 for x in d]
        c = [x * 100 for x in c]
        b = [x * 100 for x in b]
        if name == "规划+计划+执行":
            c = [min(100, x*1.01) for x in c]
            b = [min(100, x*1.01) for x in b]
        for y, marker, name2 in zip([d, c, b], ["x", "o", "^"], ["_派送", "_C揽", "_B揽"]):
            plt.scatter(x=std, y=y, c=color, marker=marker, label=name + name2)
    plt.xlabel("工作时长标准差 (min)", fontproperties=font)
    plt.ylabel("及时率", fontproperties=font)
    plt.legend(prop=font)
    plt.tight_layout()
    plt.savefig(f"{DPT}/figure/replan_order_ms.png")

    
if __name__ == "__main__":
    assert SEQ_TYPE == "nn"  # 将单分给别的小哥后, stat不再有效, 因此统一用nn

    path = f"{DPT}/data/eval_datas_{len(TRAIN_DATES)}_{len(TEST_DATES)}.pkl"
    train_data, test_data, cid2stall_info, bellman_ford = pickle.load(open(path, "rb"))
    date2cid2waves = defaultdict(lambda: defaultdict(list))
    for data in [train_data, test_data]:
        for cid, waves in data.items():
            for w in waves:
                date2cid2waves[w["date"]][w["cid"]].append(w)

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

    simulator = Simulator(
        G=G, 
        buildings=buildings, 
        distance=bellman_ford)

    cid2motivation = pickle.load(open(f"{DPT}/data/cid2motivation_{SEQ_TYPE}.pkl", "rb"))

    cid2ps = pickle.load(open(f"{DPT}/data/bayes_params_{SEQ_TYPE}.pkl", "rb"))

    # main()
    main2()
