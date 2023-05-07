import os
import pickle
import random
from collections import Counter, defaultdict
from pprint import pprint

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib.font_manager import FontProperties
from matplotlib.ticker import MultipleLocator
from shapely.geometry import LineString
from sklearn import linear_model
from sklearn.multioutput import MultiOutputRegressor
from tqdm import tqdm
from xgboost import XGBRegressor

DPT = "hk"  # 指定营业部department
if DPT == "mxl":
    from mxl.constants_all import *
    from mxl.params_eval import *
elif DPT == "hk":
    from hk.constants_all import *
    from hk.params_eval import *

random.seed(233)

def get_train_test_data(wave_data, train_dates=TRAIN_DATES, test_dates=TEST_DATES):
    # 按日期划分训练集, 测试集
    train = [x for x in wave_data if x["date"] in train_dates]
    test = [x for x in wave_data if x["date"] in test_dates]
    train, test = group_by(train, "cid"), group_by(test, "cid")
    # 去除数据量少的小哥的训练和测试数据
    gate = MIN_TRAIN_WAVE_NUM
    cids_to_remove = []
    for cid in test:
        if len(train.get(cid, [])) < gate:
            cids_to_remove.append(cid)
    for cid in cids_to_remove:
        del test[cid]
        train.pop(cid, None)
    test = {k: v for k, v in test.items() if k in train}
    train = {k: v for k, v in train.items() if k in test}
    assert set(test.keys()) == set(train.keys())
    return train, test


def get_stall_info(train_data):
    """
    对每个小哥, 找摆摊的驻留点, 
    统计各building_id为驻留模式的比例, 驻留的地点, 每个地点平均每单驻留的时长
    """
    cid2stall_info = {}
    ulid = 0
    for cid, waves in train_data.items():
        bid_cnt = defaultdict(int)
        bid2lids = defaultdict(list)
        lid2info = {}  # 随遍历过程逐渐完成聚类, 得到摆摊位置  
        for w in waves:
            stays = w["stays"]
            orders = w["orders"]
            for o in orders:
                bid_cnt[o["building_id"]] += 1
            for s in stays:
                ts, te = s["trange"]
                trange = te - ts
                if trange < T_STALL:
                    continue
                odrs = [o for o in orders if ts <= o["finish_time"] <= te]
                if len(odrs) < N_STALL:
                    # print(len(odrs), round(trange / 60))
                    continue

                x, y = s["point"][:2]
                for lid, info in lid2info.items():
                    x1, y1 = info["xy"]
                    if (x - x1) ** 2 + (y - y1) ** 2 < STALL_LOC_D ** 2:  # 合并进现有摆摊位置
                        w = info["weight"]
                        a, b = w / (w + 1), 1 / (w + 1)
                        info["xy"] = (x1 * a + x * b, y1 * a + y * b)
                        info["weight"] += 1
                        info["times"].append(trange)
                        info["nums"].append(len(odrs))
                        for o in odrs:
                            bid2lids[o["building_id"]].append(lid)
                        break
                else:
                    lid2info[ulid] = {
                        "id": ulid,
                        "xy": (x, y),
                        "weight": 1,
                        "times": [trange],
                        "nums": [len(odrs)],
                    }
                    for o in odrs:
                        bid2lids[o["building_id"]].append(ulid)
                    ulid += 1

        bid2p = {bid: len(lids) / bid_cnt[bid] for bid, lids in bid2lids.items()}  # bid会变成摆摊的比例
        bid2lid = {}  # bid去哪摆摊: 取众数
        for bid, lids in bid2lids.items():
            bid2lid[bid] = max(list(Counter(lids).items()), key=lambda x: x[1])[0]
        for info in lid2info.values():  # 每个location的平均每单摆摊时间
            tt, tn = 0, 0 
            for t, n in zip(info["times"], info["nums"]):
                tt += t
                tn += n
            info["time"] = tt / tn
        bid2p_lid = {bid: (p, bid2lid[bid]) for bid, p in bid2p.items()}
        lid2gps_time = {lid: (projector(*info["xy"], inverse=True), info["time"]) for lid, info in lid2info.items()}
        cid2stall_info[cid] = [bid2p_lid, lid2gps_time]

        # pprint(bid2p)
        # for info in lid2info.values():
        #     print([round(t / n, 1)  for t, n in zip(info["times"], info["nums"])])

    return cid2stall_info


def find_bd_in_which_comm(bd, communities):
    p = Point(bd["gate_xy"])
    for c in communities:
        poly = Polygon(c["xy"])
        if p.covered_by(poly):
            return c["id"]
    return None


def add_stall_to_map(G, buildings, cid2stall_info):
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
                "stall_time": time,
            }
            nodes_new.append((str(bid), {
                "id": str(bid),
                "gps": gps,
                "xy": xy
            }))
    
    try:
        communities = pickle.load(open(f"{DPT}/data/poi_community.pkl", "rb"))[1]
    except:
        communities = pickle.load(open(f"{DPT}/data/communities.pkl", "rb"))
    for b in buildings_new.values():
        b["comm_id"] = find_bd_in_which_comm(b, communities)

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


def adjust_odr_seq(odrs, n_iter=2):
    """在按finish_time排序的基础上微调送单次序, 减少多次访问同一楼的现象"""
    T_merge, T_change, T_pull, N_pull = 300, 300, 180, 5
    def get_b_os(odrs):
        """聚合bid相同且时间相邻的订单作为操作基本单元"""
        b_os = [[odrs[0]["building_id"], [odrs[0]]]]
        for o in odrs[1:]:
            if o["building_id"] == b_os[-1][0] and o["finish_time"] - b_os[-1][1][-1]["finish_time"] < T_merge:
                b_os[-1][1].append(o)
            else:
                b_os.append([o["building_id"], [o]])
        return b_os
    
    n_odrs = len(odrs)
    odrs.sort(key=lambda x: x["finish_time"])
    odrs_stall = [o for o in odrs if o["building_id"] >= STALL_BID_OFFSET]  # 先把摆摊的单拎出来不处理
    odrs = [o for o in odrs if o["building_id"] < STALL_BID_OFFSET]
    # 处理间隔1次重复访问的: 交换相邻两b_os的顺序, 看有没有增益
    for _ in range(n_iter):
        b_os = get_b_os(odrs)
        for i in range(len(b_os) - 3):
            a, b, c, d = b_os[i: i + 4]  # 尝试交换b,c的顺序, 使得交换之后同一楼能挨在一起
            if b[0] != c[0] and c[1][0]["finish_time"] - b[1][-1]["finish_time"] < T_change:
                if (a[0] == c[0]) + (b[0] == d[0]) > (a[0] == b[0]) + (c[0] == d[0]):
                    b_os[i+1], b_os[i+2] = b_os[i+2], b_os[i+1]
        odrs = sum([x[1] for x in b_os], [])
    # 处理间隔>1次重复访问的: 访问1栋楼的后T_pull时间内访问了同一栋楼, 若间隔的楼数<N_pull, 就强行拉到身边来
    b_os = get_b_os(odrs)
    i = 0
    while i < len(b_os):
        bi = b_os[i][0]
        for j in range(i+1, len(b_os)):  # j为之后第一个不为bi的索引
            if b_os[j][0] != bi:
                break
        else:
            break
        ti = b_os[j - 1][1][-1]["finish_time"]

        idxs_pull = []
        gap = 0
        for k in range(j, len(b_os)):
            bk, osk = b_os[k]
            if bk == bi:
                if osk[0]["finish_time"] - ti < T_pull:
                    idxs_pull.append(k)
                else:
                    break
            else:
                gap += 1
            if gap == N_pull:
                break
        if idxs_pull:
            b_os = b_os[:j] + [b_os[k] for k in idxs_pull] + [b_os[k] for k in range(j, len(b_os)) if k not in idxs_pull]
            i = j + len(idxs_pull)
        else:
            i += 1
    
    odrs = []
    last_b = None
    oneb_os = []
    for b, os in b_os:  # 在连续同一个b的局部按finish_time排序
        if b == last_b:
            oneb_os.append += os
        else:
            odrs += sorted(oneb_os, key=lambda x: x["finish_time"])
            oneb_os = os
    odrs += sorted(oneb_os, key=lambda x: x["finish_time"])
    
    if odrs_stall:  # 把摆摊点的单打包插入到时间最近的地方
        t = odrs_stall[0]["finish_time"]
        for i, o in enumerate(odrs):
            if o["finish_time"] > t:
                break
        odrs = odrs[:i] + odrs_stall + odrs[i:]
    assert len(odrs) == n_odrs
    return odrs


def data_prepare():
    cache = f"{DPT}/data/cache_data_prepare_{len(TRAIN_DATES)}.pkl"
    if os.path.exists(cache):
        train_data, cid2stall_info, bellman_ford, oid2bid_stall = pickle.load(open(cache, "rb"))
    else:
        # 准备训练数据: 优先使用做过数据质量过滤的wave, 除非某小哥过滤后就没有训练数据
        waves1 = pickle.load(open(f"{DPT}/data/wave_data.pkl", "rb"))
        waves2 = pickle.load(open(f"{DPT}/data/wave_data_nofilter.pkl", "rb"))
        train_data = get_train_test_data(waves1)[0]
        train_data2, test_data2 = get_train_test_data(waves2)
        for k, v in train_data2.items():
            if k not in train_data:
                train_data[k] = v
        # 统计历史摆摊
        cid2stall_info = get_stall_info(train_data)
        # 预计算最短路径距离
        g, _ = add_stall_to_map(G, buildings, cid2stall_info)
        print("Bellman Ford...")
        bellman_ford = {i: j for i, j in nx.all_pairs_bellman_ford_path_length(g, weight="length")}
        print("Done")
        # 固定 采样部分订单的bid到摆摊点 的结果
        oid2bid_stall = {}
        for data in [train_data, test_data2]:
            for cid, waves in data.items():
                bid2p_lid = cid2stall_info[cid][0]
                for w in waves:
                    for o in w["orders"]:
                        if o["building_id"] in bid2p_lid:
                            p, lid = bid2p_lid[o["building_id"]]
                            if random.random() < p:
                                # o["building_id"] = int(STALL_BID_OFFSET + lid)
                                oid2bid_stall[o["id"]] = int(STALL_BID_OFFSET + lid)
        pickle.dump((train_data, cid2stall_info, bellman_ford, oid2bid_stall), open(cache, "wb"))

    # 准备测试数据
    test_data = get_train_test_data(pickle.load(open(f"{DPT}/data/wave_data_nofilter.pkl", "rb")))[1]
    # 改变部分订单的bid到摆摊点; 在按finish_time排序基础上微调送单次序
    for data in [train_data, test_data]:
        for cid, waves in data.items():
            bid2p_lid = cid2stall_info[cid][0]
            for w in waves:
                for o in w["orders"]:
                    if o["id"] in oid2bid_stall:
                        o["building_id"] = oid2bid_stall[o["id"]]
                w["orders"] = adjust_odr_seq(w["orders"])
    return train_data, test_data, cid2stall_info, bellman_ford


def calculate_mean_std(nums, weights=None):
    """计算一组数的均值和标准差, 可以加权"""
    if weights is None:
        return np.mean(nums), np.std(nums)
    mean = sum(n*w for n, w in zip(nums, weights)) / (sum(weights) + 1e-12)
    std2 = sum(w*(n - mean)**2 for n, w in zip(nums, weights)) / (sum(weights) + 1e-12)
    return mean, std2 ** 0.5


def calculate_mae(nums, weights=None):
    """计算一组数的MAE, 可以加权"""
    if weights is None:
        return np.mean([abs(x) for x in nums])
    x = np.array([abs(x) for x in nums])
    w = np.array(weights)
    return (x @ w) / (w.sum() + 1e-12)


def cal_gt_metric(wave, bellman_ford):
    """计算真值指标"""
    def cal_work_time(wave):
        """计算真值工作时长"""
        s, e = wave["wave_traj"]
        t0 = e - s
        # 去掉非工作的驻留时间
        t_remove1 = 0
        removed_tranges = []
        ots = [o["finish_time"] for o in wave["orders"]]
        for s in wave["stays"]:
            t1, t2 = s["trange"]
            if t2 - t1 > T_LONG_REST:
                for t in ots:
                    if t1 < t < t2:
                        break
                else:
                    t_remove1 += t2 - t1
                    removed_tranges.append((t1, t2))
        # 去掉轨迹在站附近的的时间
        traj = wave["traj_orig"]
        t_remove2 = 0
        for (x1, y1, t1), (x2, y2, t2) in zip(traj, traj[1:]):
            if t2 - t1 < 60 and (x1 - X_STA) ** 2 + (y1 - Y_STA) ** 2 < 2500 and (x2 - X_STA) ** 2 + (y2 - Y_STA) ** 2 < 2500:
                t11, t22 = t1 - 120, t2 + 120
                for t in ots:  # 附近没单完成
                    if t11 < t < t22:
                        break
                else:
                    for a, b in removed_tranges:
                        if a < t1 < b or a < t2 < b:  # 不要重复去
                            break
                    else:
                        t_remove2 += t2 - t1
        t_remove2 = min(t_remove2, t0 * 0.03)

        return t0 - t_remove1 - t_remove2

    def cal_travel_length(odrs, bellman_ford):
        """计算跑动距离"""
        length = 0
        last_b, last_t = odrs[0]["building_id"], odrs[0]["finish_time"]
        acc, rej = 0, 0
        for o in odrs[1:]:
            b, t = o["building_id"], o["finish_time"]
            if b != last_b:
                l = bellman_ford[str(last_b)][str(b)]
                if l / (t - last_t + 1e-12) < MAX_V_TRAVEL:  # 过滤不可能的速度
                    length += l
                    last_b, last_t = b, t
                    acc += 1
                else:
                    length += l  # 即使速度不可能也算上
                    rej += 1
        if acc / (acc + rej + 1e-12) > 0.7:
            return length
        else:   # 过滤地过多则认为结果不靠谱
            return None
    
    cid, date = wave["cid"], wave["date"]
    odrs = wave["orders"]
    tmp = group_by(odrs, "type")
    ds, cs, bs = tmp.get(ORDER_DELIVER, []), tmp.get(ORDER_CPICK, []), tmp.get(ORDER_BPICK, [])
    dnum, cnum, bnum = len(ds), len(cs), len(bs)
    dotnum = len([o for o in ds if o["finish_time"] < o["ddl_time"]])  # on_time_num
    cotnum = len([o for o in cs if o["finish_time"] < o["ddl_time"]])
    botnum = len([o for o in bs if o["finish_time"] < o["ddl_time"]])
    wt = cal_work_time(wave) 
    tl = cal_travel_length(odrs, bellman_ford)
    oid2idx = {o["id"]: i for i, o in enumerate(odrs)}  # 送单次序

    return {
        "cid": cid,
        "date": date,
        "start_time": wave["wave_traj"][0],
        "dcbnum": [dnum, cnum, bnum],
        "otnum": [dotnum, cotnum, botnum],
        "wt": wt,
        "tl": tl,
        "seq": oid2idx,
        "wave_idx": wave["wave_idx"],
        "is_morning": wave["is_morning"],
        "traj_cover": wave["traj_cover"]
    }


def cal_sim_metric(sim_actions):
    """
    根据模拟模型跑出的actions来计算指标
    """
    metrics = []
    for x in sim_actions:
        actions = x["actions"]
        work_time = 0
        for a in actions:
            assert "station_id" not in a
            # 模拟模型中, 若派完所有件, 还有揽件没有产生, 则会一直等到揽件产生, 不计入时间
            if a.get("wait_for_pick", False):
                work_time += 0
            else:
                work_time += a["end_time"] - a["start_time"]
        status = actions[-1]["status"]
        oids = [int(a["target_orders"][0]["id"]) for a in actions if a["type"] in ACTION_ORDER]
        oid2idx = {oid: i for i, oid in enumerate(oids)}  # 送单次序


        metrics.append({
            "cid": x["cid"],
            "date": x["date"],
            "wave_idx": x["wave_idx"],
            "start_time": actions[0]["start_time"],
            "otnum": [status["delivered_on_time"], status["cpicked_on_time"], status["bpicked_on_time"]],
            "wt": work_time,
            "tl": status["traveled_length"],
            "seq": oid2idx
        })
    return metrics


def merge_one_day_metrics(ms):
    """合并一天中多个波的指标"""
    cid_date2ms = defaultdict(list)
    for x in ms:
        cid_date2ms[(x["cid"], x["date"])].append(x)
    ms_merge = []
    for (cid, date), ms in cid_date2ms.items():
        ms.sort(key=lambda x: x["wave_idx"])
        m_merge = {
            "cid": cid,
            "date": date,
            "wave_idx": 0,
            "start_time": ms[0]["start_time"],       
            "otnum": sum([np.array(m["otnum"]) for m in ms]).tolist(),
            "wt": sum(m["wt"] for m in ms),
            # "tl": sum(m["tl"] for m in ms if m["tl"] > 0),  # gt和sim在此需要同步, 因此先保留arr, 不merge
            "tl": [m["tl"] for m in ms],
            "seq": {*sum([[*m["seq"].items()] for m in ms], [])},
        }
        oid2idx = ms[0]["seq"]
        for m in ms[1:]:
            oid2idx.update(m["seq"])
        m_merge["seq"] = oid2idx
        if "dcbnum" in ms[0]:
            m_merge["dcbnum"] = sum([np.array(m["dcbnum"]) for m in ms]).tolist()
        if "traj_cover" in ms[0]:
            m_merge["traj_cover"] = np.mean([m["traj_cover"] for m in ms])
        ms_merge.append(m_merge)
    return ms_merge


def merge_post(ms_gt, ms_sim):
    """处理部分指标真值缺失导致的问题"""
    cid_date2ms_gt, cid_date2ms_sim = group_by(ms_gt, ["cid", "date"]), group_by(ms_sim, ["cid", "date"])
    keys = ["tl"]
    for (cid, date), ms_gt in cid_date2ms_gt.items():
        ms_sim = cid_date2ms_sim[(cid, date)]
        assert len(ms_sim) == len(ms_gt) == 1
        m_gt, m_sim = ms_gt[0], ms_sim[0]
        for k in keys:
            vs_gt, vs_sim = m_gt[k], m_sim[k]
            assert len(vs_gt) == len(vs_sim)
            idxs = [i for i, v in enumerate(vs_gt) if v is not None]
            if len(idxs) == 0:
                m_gt[k] = m_sim[k] = None
            else:
                m_gt[k], m_sim[k] = sum(vs_gt[i] for i in idxs), sum(vs_sim[i] for i in idxs)


def cum(ts):
    """
    将一堆事件的发生时间统计为[(t, 累积发生次数)]的形式 
    """
    num0 = len([t for t in ts if t < 0])
    t_nums = sorted(list(Counter([t for t in ts if t >= 0]).items()), key=lambda x:x[0])
    points = [(0, num0)]
    cnt = num0
    for t, n in t_nums:
        points.append((t, cnt))
        cnt += n
        points.append((t, cnt))
    return points


def cal_df_metric(gt_metrics, sim_metrics, ignore_cids=set(), merge_wave=True, baseline=False):        
    def dseq_exp(N):
        """
        将序列(1,2,...,N)随机重排序后, 与原始序列的每位差的绝对值的和的期望
        https://blog.csdn.net/u012929136/article/details/114219219
        """
        # return (N ** 2 - 1) / 3
        # return N ** 2  # 为使指标具有更简单的物理含义, 实际采用N^2归一化
        return N

    def cal_dseq(oid2idx_gt, oid2idx_sim):
        assert set(oid2idx_gt) == set(oid2idx_sim)
        return sum(
            abs(i - oid2idx_gt[oid]) 
            for oid, i in oid2idx_sim.items()
        ) / (dseq_exp(len(oid2idx_gt)) + 1e-12)

    def cal_dseq_bc(oid2idx_gt, oid2idx_sim, oid2bcid):
        """将bid按其平均出现次序排序, 转为一个bid出现且仅出现1次的序列"""
        def group_bd_idx(bid_idxs):
            r = defaultdict(list)
            for bid, idx in bid_idxs:
                r[bid].append(idx)
            return r

        bid2idxs_gt = group_bd_idx([[oid2bcid[oid], i] for oid, i in oid2idx_gt.items() if oid in oid2bcid])
        bid2idxs_sim = group_bd_idx([[oid2bcid[oid], i] for oid, i in oid2idx_sim.items() if oid in oid2bcid])
        if not bid2idxs_gt:
            return 0
        
        gt = [(bid, np.mean(ii)) for bid, ii in bid2idxs_gt.items()]
        sim = [(bid, np.mean(ii)) for bid, ii in bid2idxs_sim.items()]
        gt = {b_idx[0]: i for i, b_idx in enumerate(sorted(gt, key=lambda x: x[1]))}
        sim = {b_idx[0]: i for i, b_idx in enumerate(sorted(sim, key=lambda x: x[1]))}
        return cal_dseq(gt, sim)
    
    def cal_dseq_bc1(oid2idx_gt, oid2idx_sim, oid2bcid):
        oid2idx_gt = {o: i for o, i in oid2idx_gt.items() if o in oid2bcid}
        oid2idx_sim = {o: i for o, i in oid2idx_sim.items() if o in oid2bcid}
        if not oid2idx_gt:
            return 0
        
        def get_bcid2clusters(oid2i):
            oid_bcids = [[x[0], oid2bcid[x[0]]] for x in sorted(list(oid2i.items()), key=lambda x: x[1])]
            if not oid_bcids:
                return {}
            bcid2clusters = defaultdict(list)
            cluster = {oid_bcids[0][0]}
            last_bcid = oid_bcids[0][1]
            for oid, bcid in oid_bcids[1:]:
                if bcid == last_bcid:
                    cluster.add(oid)
                else:
                    bcid2clusters[last_bcid].append(cluster)
                    cluster = {oid}
                    last_bcid = bcid
            bcid2clusters[last_bcid].append(cluster)
            assert len(oid_bcids) == sum(len(c) for cs in bcid2clusters.values() for c in cs)
            return bcid2clusters

        b2cs_gt, b2cs_sim = map(get_bcid2clusters, [oid2idx_gt, oid2idx_sim])
        assert set(b2cs_gt) == set(b2cs_sim)
        
        def split_clusters(cs1, cs2):
            """
            cs1 / cs2: 同一bcid下的几组订单, 其中挨着连续送的在1个cluster内
            将cluster进一步拆分, 直到双方中的clusters完全相同
            """
            a = sum(len(c) for c in cs1)
            b = sum(len(c) for c in cs2)
            assert a == b
            while True:
                proc_flag = False
                for i, c1 in enumerate(cs1):
                    for j, c2 in enumerate(cs2):
                        if c1 != c2:
                            cand = c1 & c2
                            if cand:
                                cdif1 = c1 - cand
                                cdif2 = c2 - cand  
                                c1_new = [cand, cdif1] if cdif1 else [cand]
                                c2_new = [cand, cdif2] if cdif2 else [cand]
                                proc_flag = True
                                break
                    if proc_flag:
                        break
                if proc_flag:
                    cs1.pop(i)
                    cs1 += c1_new
                    cs2.pop(j)
                    cs2 += c2_new
                else:
                    break
            assert sum(len(c) for c in cs1) == sum(len(c) for c in cs2) == a
            
            return cs1, cs2

        oid2bcid_new = {}
        bcid = 0
        for b, cs_gt in b2cs_gt.items():
            cs = split_clusters(cs_gt, b2cs_sim[b])[0]
            for c in cs:
                for oid in c:
                    oid2bcid_new[oid] = bcid
                bcid += 1  # 同一bcid下不同cluster视为属于不同bcid

        def get_bcid_new2idx(oid2i):
            nonlocal oid2bcid_new
            oids = [x[0] for x in sorted(list(oid2i.items()), key=lambda x: x[1])]
            bcids = [oid2bcid_new[oids[0]]]
            for oid in oids[1:]:
                bcid = oid2bcid_new[oid]
                if bcid != bcids[-1]:
                    bcids.append(bcid)
            return {bcid: i for i, bcid in enumerate(bcids)}
        bcid2idx_gt, bcid2idx_sim = map(get_bcid_new2idx, [oid2idx_gt, oid2idx_sim])
        return cal_dseq(bcid2idx_gt, bcid2idx_sim)
    
    def cal_dseq_bc2(oid2idx_gt, oid2idx_sim, oid2bcid):
        oid2idx_gt = {o: i for o, i in oid2idx_gt.items() if o in oid2bcid}
        oid2idx_sim = {o: i for o, i in oid2idx_sim.items() if o in oid2bcid}
        if not oid2idx_gt:
            return 0

        def get_oid2idx_new(oid2i):
            """[1,2],[3],[4,5] -> 1, 1, 3, 4, 4"""
            oids = [x[0] for x in sorted(list(oid2i.items()), key=lambda x: x[1])]
            oid2idx_new = {}
            last_bcid = None
            i = 0
            cnt = 0
            for oid in oids:
                bcid = oid2bcid[oid]
                if bcid != last_bcid:
                    i += cnt
                    cnt = 1
                    last_bcid = bcid
                else:
                    cnt += 1
                oid2idx_new[oid] = i
            return oid2idx_new
        
        oid2idx_gt_new, oid2idx_sim_new = map(get_oid2idx_new, [oid2idx_gt, oid2idx_sim])
        return cal_dseq(oid2idx_gt_new, oid2idx_sim_new)
        
    if merge_wave:  # 将同一小哥同一天的波合并
        gt_metrics, sim_metrics = map(merge_one_day_metrics, [gt_metrics, sim_metrics])
        merge_post(gt_metrics, sim_metrics)
    gt_metrics = {(x["cid"], x["date"], x["wave_idx"]): x for x in gt_metrics}
    sim_metrics = {(x["cid"], x["date"], x["wave_idx"]): x for x in sim_metrics}

    # 手动拉平tl的ME
    def get_cid2ms(ms):
        cid2ms = defaultdict(list)
        for m in ms:
            cid2ms[m["cid"]].append(m)
        for ms in cid2ms.values():
            ms.sort(key=lambda x: (x["date"], x["wave_idx"]))
        return cid2ms
    if not baseline:  # 拉平相对跑动, 而不是绝对
        gt, sim = map(get_cid2ms, [gt_metrics.values(), sim_metrics.values()])
        for cid, ms_gt in gt.items():
            if cid in ignore_cids:
                continue
            ms_sim = sim[cid]
            l1, l2 = [], []
            for m1, m2 in zip(ms_gt, ms_sim):
                if m1["tl"] is not None:
                    l1.append(m1["tl"])
                    l2.append(m2["tl"])
            n = len(l1)
            if n > 0:
                an = 1
                for a in l1:
                    an *= a
                an_1 = [an / a for a in l1]
                b_an_1 = np.array(l2) @ np.array(an_1)
                p = n * an / b_an_1
                for m1, m2 in zip(ms_gt, ms_sim):
                    if m1["tl"] is not None:
                        m2["tl"] *= p
    else:
        l_gt, l_sim = 0, 0
        for k, v in gt_metrics.items():
            if k[0] in ignore_cids:
                continue
            if v["tl"] is not None:
                l_gt += v["tl"]
                l_sim += sim_metrics[k]["tl"]
        p = l_gt / l_sim
        for k, v in gt_metrics.items():
            if k[0] in ignore_cids:
                continue
            if v["tl"] is not None:
                sim_metrics[k]["tl"] *= p

    lines = []
    for (cid, date, wid), msim in sim_metrics.items():
        if cid in ignore_cids:
            continue
        mgt = gt_metrics[(cid, date, wid)]

        dwt = msim["wt"] - mgt["wt"]
        dwt_rel = 100 * dwt / (mgt["wt"] + 1e-12)

        tl_flag = mgt["tl"] is not None  # 真值是否能算准
        dtl = msim["tl"] - mgt["tl"] if tl_flag else 0
        dtl_rel = 100 * dtl / (mgt["tl"] + 1e-12) if tl_flag else 0        

        dnum, cnum, bnum = mgt["dcbnum"]
        dotsim, cotsim, botsim = msim["otnum"]
        dotgt, cotgt, botgt = mgt["otnum"]
        ddotr = 100 * (dotsim - dotgt) / (dnum + 1e-12)
        dcotr = 100 * (cotsim - cotgt) / (cnum + 1e-12)
        dbotr = 100 * (botsim - botgt) / (bnum + 1e-12)

        eff_gt = 3600 * (dnum + cnum + bnum) / (mgt["wt"] + 1e-12)
        eff_sim = 3600 * (dnum + cnum + bnum) / (msim["wt"] + 1e-12)
        deff = eff_sim - eff_gt
        deff_rel = 100 * deff / (eff_gt + 1e-12)

        oid2idx_gt = mgt["seq"]
        oid2idx_sim = msim["seq"]
        dseq = cal_dseq(oid2idx_gt, oid2idx_sim)
        
        # 算法0: 把所有楼按平均访问顺序排序
        dseq_bd = cal_dseq_bc(oid2idx_gt, oid2idx_sim, oid2bid)
        dseq_comm = cal_dseq_bc(oid2idx_gt, oid2idx_sim, oid2comm_id)

        # # 算法1: 把楼按最大重复子序列拆分成多个楼
        # dseq_bd = cal_dseq_bc1(oid2idx_gt, oid2idx_sim, oid2bid)
        # dseq_comm = cal_dseq_bc1(oid2idx_gt, oid2idx_sim, oid2comm_id)

        # # 算法2: 把同一楼连续送的单的序号标为相同
        # dseq_bd = cal_dseq_bc2(oid2idx_gt, oid2idx_sim, oid2bid)
        # dseq_comm = cal_dseq_bc2(oid2idx_gt, oid2idx_sim, oid2comm_id)

        lines.append([
            str(date), str(wid), str(cid), cid2name.get(cid, "无名"), mgt["traj_cover"],
            dwt, dwt_rel, 
            deff, deff_rel, 
            dtl, dtl_rel, int(tl_flag),
            dseq, dseq_bd, dseq_comm,
            dnum, ddotr,
            cnum, dcotr, 
            bnum, dbotr])
        
    # 用dataframe管理多个小哥多天的多个指标间的差异
    column_names = [
        "日期", "波次", "小哥id", "小哥名", "轨迹覆盖率",
        "时长", "相对时长%",
        "效率", "相对效率%",
        "跑动", "相对跑动%", "跑动flag",
        "次序-单", "次序-楼", "次序-区",
        "派送数", "及时率-派送",
        "C揽数", "及时率-C揽",
        "B揽数", "及时率-B揽"
    ]
    return pd.DataFrame(data=lines, columns=column_names)


def zoom_df_metric(df, fix_date=None, fix_courier=None, show=True):
    if fix_date:
        if show:
            print("fix date:", fix_date)
        df = df[df["日期"] == str(fix_date)]
    elif fix_courier:
        if show:
            print("fix courier:", cid2name[fix_courier], fix_courier)
        df = df[df["小哥id"] == str(fix_courier)]
    else:
        if show:
            print("all dates and all couriers")

    print_keys = [
        "相对时长%", "时长",
        "相对效率%", "效率", 
        "及时率-派送", "及时率-C揽", "及时率-B揽",
        "次序-单", "次序-楼", "次序-区",
        "相对跑动%", "跑动",
    ]
    key2weight = {
        "及时率-派送": "派送数",
        "及时率-C揽": "C揽数",
        "及时率-B揽": "B揽数",
        "跑动": "跑动flag",
        "相对跑动%": "跑动flag"}
    maes, mean_stds = [], []
    for k in print_keys:
        if k in key2weight:
            maes.append(calculate_mae(nums=list(df[k]), weights=list(df[key2weight[k]])))
            mean_stds.append(calculate_mean_std(nums=list(df[k]), weights=list(df[key2weight[k]])))
        else:
            maes.append(calculate_mae(list(df[k])))
            mean_stds.append(calculate_mean_std(list(df[k])))
    if fix_date:
        df = df[["波次", "小哥名"] + print_keys]
    elif fix_courier:
        df = df[["日期", "波次"] + print_keys]
    else:
        df = df[print_keys]
    if fix_date is None and fix_courier is None:
        columns = [" "] + df.columns.values.tolist()
        lines = [["[MAE]"] + maes, ["[ME]"] + [x[0] for x in mean_stds], ["[STD]"] + [x[1] for x in mean_stds]]
    else:
        columns = df.columns.values.tolist()
        lines = df.values.tolist()
        lines = [["[MAE]", "[MAE]"] + maes, ["[ME]", "[ME]"] + [x[0] for x in mean_stds], ["[STD]", "[STD]"] + [x[1] for x in mean_stds]] + lines
    if show:
        print_table(columns, lines)

    return_keys = ["相对时长%", "相对效率%", "及时率-派送", "及时率-C揽", "及时率-B揽", "次序-单", "次序-楼", "次序-区", "相对跑动%"]
    n_head = 1 if fix_date is None and fix_courier is None else 2
    idxs = [i for i, k in enumerate(columns[n_head:]) if k in return_keys]
    return [v for i, v in enumerate(lines[0][n_head:]) if i in idxs]


class Evaluater:
    def __init__(self, train_data, test_data, sim_actions, bellman_ford, ignore_cids=set(), merge_wave=True):
        self.train_data = train_data
        self.test_data = test_data
        self.sim_actions = sim_actions
        self.ignore_cids = ignore_cids
        self.train_metrics = {k: [cal_gt_metric(x, bellman_ford) for x in v] for k, v in train_data.items()}
        self.test_metrics = [cal_gt_metric(x, bellman_ford) for x in sum(test_data.values(), [])]
        self.metrics_sim = cal_sim_metric(sum(sim_actions.values(), []))
        self.metrics_avg = None
        self.metrics_xgb = None
        self.df_metric_sim = cal_df_metric(self.test_metrics, self.metrics_sim, ignore_cids=ignore_cids, merge_wave=merge_wave)
        self.df_metric_avg = None
        self.df_metric_xgb = None
        self.merge_wave = merge_wave


    def seq_baseline(self, train, test):
        """送单序列baseline: 统计bid出现的次序"""
        bid2seqs = defaultdict(list)
        for w in train:
            bids = [o["building_id"] for o in w["orders"]]
            n = len(bids)
            bid2seq = defaultdict(list)  # 一波中, bid出现的归一化次序
            for i, bid in enumerate(bids):
                bid2seq[bid].append(i / n)
            for bid, ss in bid2seq.items():
                bid2seqs[bid].append(np.mean(ss))

        bid2seq = {bid: np.mean(ss) for bid, ss in bid2seqs.items()}
        date2wid2oid2idx = defaultdict(dict)
        for w in test:
            orders = w["orders"]
            bid2odrs = defaultdict(list)
            for o in orders:
                bid2odrs[o["building_id"]].append(o)
            bid_odrs = sorted(list(bid2odrs.items()), key=lambda x: bid2seq.get(x[0], random.random()))
            oid2idx = {}
            i = 0
            for odrs in [x[1] for x in bid_odrs]:
                # random.shuffle(odrs)
                odrs.sort(key=lambda x: (x["unit"], x["floor"]))
                for o in odrs:
                    oid2idx[o["id"]] = i
                    i += 1
            date2wid2oid2idx[w["date"]][w["wave_idx"]] = oid2idx
        return date2wid2oid2idx


    def baseline_avg(self):
        test_data = self.test_data
        train_metrics = self.train_metrics
        metrics = []
        for cid, test in test_data.items():
            # 用训练数据计算平均及时率, 效率, 跑动效率
            train = train_metrics[cid]
            d, c, b, dot, cot, bot, eff, tl_eff = 0, 0, 0, 0, 0, 0, 0, 0
            tl_cnt = 0
            for x in train:
                dnum, cnum, bnum = x["dcbnum"]
                dotnum, cotnum, botnum = x["otnum"]
                d += dnum
                c += cnum
                b += bnum
                dot += dotnum
                cot += cotnum
                bot += botnum
                eff += (dnum + cnum + bnum) / (x["wt"] + 1e-12)
                if x["tl"] is not None: 
                    tl_eff += (dnum + cnum + bnum) / (x["tl"] + 1e-12)
                    tl_cnt += 1
            dotr = (dot + 1e-12) / (d + 1e-12)  # 0/0时认为及时率是100%
            cotr = (cot + 1e-12) / (c + 1e-12)
            botr = (bot + 1e-12) / (b + 1e-12)
            eff /= len(train)
            tl_eff /= tl_cnt + 1e-12
            # 估计测试数据的指标
            for x in test:
                tmp = group_by(x["orders"], "type")
                d, c, b = tmp.get(ORDER_DELIVER, []), tmp.get(ORDER_CPICK, []), tmp.get(ORDER_BPICK, [])
                d, c, b = len(d), len(c), len(b)
                oids = [o["id"] for o in x["orders"]]
                random.shuffle(oids)
                oid2idx = {oid: i for i, oid in enumerate(oids)}
                metrics.append({
                    "cid": cid,
                    "date": x["date"],
                    "wave_idx": x["wave_idx"],
                    "start_time": x["wave_traj"][0],
                    "otnum": [d * dotr, c * cotr, b * botr],
                    "wt": (d + c + b) / (eff + 1e-12),
                    "tl": min((d + c + b) / (tl_eff + 1e-12), 20000),
                    "seq": oid2idx,
                })
        self.metrics_avg = metrics
        self.df_metric_avg = cal_df_metric(self.test_metrics, metrics, ignore_cids=self.ignore_cids, merge_wave=self.merge_wave, baseline=True)
        return metrics


    def baseline_xgb(self):
        """
        xgboost baseline
        特征: 派件数, C揽数, B揽数, 小哥id, 周几, 上下午
        输出: 派送及时率, C揽及时率, B揽及时率, 工作时长, 移动距离
        """
        train_data = self.train_data
        test_data = self.test_data
        train_metrics = self.train_metrics
        cids = list(set(train_data.keys()) | set(test_data.keys()))
        cid2idx = {cid: i for i, cid in enumerate(cids)}
        # seq_baseline
        cid2date2wid2oid2idx = {cid: self.seq_baseline(train_data[cid], test) for cid, test in test_data.items()}
        # 所有小哥一起做
        train_metrics = sum(train_metrics.values(), [])
        test_data = sum(test_data.values(), [])
        # 训练集
        A = []
        for x in train_metrics:
            d, c, b = x["dcbnum"]
            dot, cot, bot = x["otnum"]
            idx_feat = [0] * len(cid2idx)
            idx_feat[cid2idx[x["cid"]]] = 1
            week_feat = [0] * 7
            week_feat[date2num(x["date"]) % 7] = 1
            morning_feat = [1, 0] if x["is_morning"] else [0, 1]
            dotr = (dot + 1e-12) / (d + 1e-12)  # 0/0时认为及时率是100%
            cotr = (cot + 1e-12) / (c + 1e-12)
            botr = (bot + 1e-12) / (b + 1e-12)
            A.append(
                [d, c, b] + idx_feat + week_feat + morning_feat + \
                [dotr, cotr, botr, x["wt"], x["tl"]]
            )
        A = np.array(A)
        X_train, Y_train = A[:, :-5], A[:, -5:]
        print("train:", X_train.shape, Y_train.shape)
        # 测试集
        X_test = []
        for x in test_data:
            tmp = group_by(x["orders"], "type")
            d, c, b = tmp.get(ORDER_DELIVER, []), tmp.get(ORDER_CPICK, []), tmp.get(ORDER_BPICK, [])
            d, c, b = len(d), len(c), len(b)
            idx_feat = [0] * len(cid2idx)
            idx_feat[cid2idx[x["cid"]]] = 1
            week_feat = [0] * 7
            week_feat[date2num(x["date"]) % 7] = 1
            morning_feat = [1, 0] if x["wave_traj"][0] < 12 * 3600 else [0, 1]
            X_test.append([d, c, b] + idx_feat + week_feat + morning_feat)
        X_test = np.array(X_test)
        print("X_test:", X_test.shape)

        # 处理travel_length部分真值为None的情况
        mask = np.ones(len(Y_train), dtype=bool)
        for i, y in enumerate(Y_train):
            if y[-1] is None:
                mask[i] = False
        print(mask.shape)
        print(X_train.shape)
        print(Y_train.shape)
        print(Y_train[:, -1][mask].shape)
        print(X_train[mask].shape)
        
        multi_xgb = MultiOutputRegressor(XGBRegressor(
            max_depth=3,   # 10
            learning_rate=0.1, 
            n_estimators=18))  # 500
        multi_xgb.fit(X_train, Y_train[:, :-1])
        Y_test_notl = multi_xgb.predict(X_test)

        xgb = XGBRegressor(
            max_depth=3,   # 10
            learning_rate=0.1, 
            n_estimators=15)
        xgb.fit(X_train[mask], Y_train[:, -1][mask])
        Y_test_tl = xgb.predict(X_test)

        print(Y_test_notl.shape)
        print(Y_test_tl.shape)
        Y_test = np.hstack((Y_test_notl, Y_test_tl.reshape(-1, 1)))
        print("Y_test:", Y_test.shape)

        # 输出指标
        metrics = []
        for x, y in zip(test_data, Y_test):
            dotr, cotr, botr, wt, tl = y
            tmp = group_by(x["orders"], "type")
            d, c, b = tmp.get(ORDER_DELIVER, []), tmp.get(ORDER_CPICK, []), tmp.get(ORDER_BPICK, [])
            d, c, b = len(d), len(c), len(b)
            oid2idx = cid2date2wid2oid2idx[x["cid"]][x["date"]][x["wave_idx"]]
            metrics.append({
                "cid": x["cid"],
                "date": x["date"],
                "wave_idx": x["wave_idx"],
                "start_time": x["wave_traj"][0],
                "otnum": [d * dotr, c * cotr, b * botr],
                "wt": wt,
                "tl": tl,
                "seq": oid2idx,
            })
        self.metrics_xgb = metrics
        self.df_metric_xgb = cal_df_metric(self.test_metrics, metrics, ignore_cids=self.ignore_cids, merge_wave=self.merge_wave, baseline=True)
        return metrics


    def display(self, mode, baselines=[], show=True):
        name_metrics = [["sim", self.df_metric_sim]]
        for b in baselines:
            if b == "avg" and self.metrics_avg is None:
                self.baseline_avg()
                name_metrics.append([b, self.df_metric_avg])
            elif b == "xgb" and self.metrics_xgb is None:
                self.baseline_xgb()
                name_metrics.append([b, self.df_metric_xgb])

        if mode == "overall":
            for name, metrics in name_metrics:
                if show:
                    print(f" ********* {name} result ********* ")
                zoom_df_metric(df=metrics, fix_date=None, fix_courier=None, show=show)
        elif mode == "courier_wise":
            fix_date = None
            cid2ms = {}
            cids = [cid for cid in self.test_data if cid not in self.ignore_cids]
            for fix_courier in cids:
                for name, metrics in name_metrics:
                    if show:
                        print(f" ********* {name} result ********* ")
                    t = zoom_df_metric(df=metrics, fix_date=None, fix_courier=fix_courier, show=show)
                    if name == "sim":
                        cid2ms[fix_courier] = t
            return cid2ms
        elif mode == "date_wise":
            fix_courier = None
            date2ms = {}
            dates = list({w["date"] for ws in self.test_data.values() for w in ws})
            dates.sort(key=lambda x: date2num(x))
            for fix_date in dates:
                for name, metrics in name_metrics:
                    if show:
                        print(f" ********* {name} result ********* ")
                    t = zoom_df_metric(df=metrics, fix_date=fix_date, fix_courier=None, show=show)
                    if name == "sim":
                        date2ms[fix_date] = t
            return date2ms

             
def plot_train_test_curve(train_gt, train_sim, test_gt, test_sim, merge_wave=True):
    def get_cid2date2xs(xs):
        cid2date2xs = defaultdict(lambda: defaultdict(list))
        for x in xs:
            cid2date2xs[x["cid"]][x["date"]].append(x)
        for v in cid2date2xs.values():
            for vv in v.values():
                vv.sort(key=lambda x: x["wave_idx"])
        return cid2date2xs
    
    if merge_wave:
        train_gt, train_sim, test_gt, test_sim = map(merge_one_day_metrics, [train_gt, train_sim, test_gt, test_sim])
        merge_post(train_gt, train_sim)
        merge_post(test_gt, test_sim)
    train_df = cal_df_metric(train_gt, train_sim, merge_wave=merge_wave)
    test_df = cal_df_metric(test_gt, test_sim, merge_wave=merge_wave)
    train_gt, train_sim, test_gt, test_sim = map(get_cid2date2xs, [train_gt, train_sim, test_gt, test_sim])

    for gt, sim in [(train_gt, train_sim), (test_gt, test_sim)]:
        key2dcbnum = {  # 补充sim中的dcbnum字段
            (cid, date, x["wave_idx"]): x["dcbnum"] 
            for cid, date2xs in gt.items() for date, xs in date2xs.items() for x in xs
        }
        for cid, date2xs in sim.items():
            for date, xs in date2xs.items():
                for x in xs:
                    x["dcbnum"] = key2dcbnum[(cid, date, x["wave_idx"])]

    font = FontProperties(fname=r"msyh.ttc")

    def get_mae_me(df):
        maes = []
        mean_stds = []
        for k in ["相对效率%"]:
            maes.append(calculate_mae(list(df[k])))
            mean_stds.append(calculate_mean_std(list(df[k])))
        for k1, k2 in [("C揽数", "及时率-C揽"), ("B揽数", "及时率-B揽"), ("跑动flag", "相对跑动%")]:
            maes.append(calculate_mae(nums=list(df[k2]), weights=list(df[k1])))
            mean_stds.append(calculate_mean_std(nums=list(df[k2]), weights=list(df[k1])))
        return maes, [x[0] for x in mean_stds]
    
    def get_plt_arr(date2ms):
        """
        return: list of [t, e, c, b, n, l]
        t: 将y-m-d的date转为数字
        e: eff
        c: cotr
        b: botr
        n: 总单数
        l: 跑动距离
        """
        arr = []
        for date, ms in date2ms.items():
            for m in ms:
                t = date2num(date) + m["start_time"] / 86400  # 将y-m-d转为数字
                d, c, b, _, cot, bot = [*m["dcbnum"], *m["otnum"]]
                e = 3600 * (d + c + b) / (m["wt"] + 1e-12)
                cotr = 100 * cot / (c + 1e-12)
                botr = 100 * bot / (b + 1e-12)
                arr.append([t, e, cotr, botr, sum(m["dcbnum"]), m["tl"]])
        return sorted(arr, key=lambda x: x[0])
    
    def m2km(arr):
        return [a / 1000 for a in arr]

    for cid in tqdm(test_gt):
        maes_tr, mes_tr = get_mae_me(train_df[train_df["小哥id"] == str(cid)])
        maes_te, mes_te = get_mae_me(test_df[test_df["小哥id"] == str(cid)])
        tecbnls = []
        for date2ms in [train_gt[cid], test_gt[cid], train_sim[cid], test_sim[cid]]:
            arr = get_plt_arr(date2ms)
            tecbnl = [list(xs) for xs in zip(*arr)]   # 每种date2ms对应的[ts, es, cs, bs, ns, ls]
            tecbnls.append(tecbnl)
        ts, es, cs, bs, ns, ls = [list(xs) for xs in zip(*tecbnls)]  # 再zip, 把每种date2ms的xs塞在一起

        names = ["eff", "cotr", "botr", "tl"]
        fig = plt.figure(figsize=(20, 4*len(names)))
        idx = 1
        for xs, n, mae_tr, me_tr, mae_te, me_te in zip([es, cs, bs, ls], names, maes_tr, mes_tr, maes_te, mes_te):
            ax = fig.add_subplot(len(names), 1, idx)
            ax.set_xlim((1, len(TRAIN_DATES) + len(TEST_DATES) + 1))
            ax.xaxis.set_major_locator(MultipleLocator(1))
            if n == "eff" or n == "tl":  # 同时画出总单量
                ax2 = ax.twinx()
                ax2.plot(ts[0] + ts[1], ns[0] + ns[1], c="gray", linestyle="--", label="onum")
                ax2.legend()
            if n == "cotr" or n == "botr":
                ax.set_ylim(0, 100)

            t_gt_tr, x_gt_tr = ts[0], xs[0]
            t_gt_te, x_gt_te = ts[1], xs[1]
            t_sim_tr, x_sim_tr = ts[2], xs[2]
            t_sim_te, x_sim_te = ts[3], xs[3]
            if n == "tl":  # 去掉x_gt为None的点
                trs = [t_gt_tr, x_gt_tr, t_sim_tr, x_sim_tr]
                tes = [t_gt_te, x_gt_te, t_sim_te, x_sim_te]
                assert len(set(len(x) for x in trs)) == 1 and len(set(len(x) for x in tes)) == 1
                idxs = [i for i, x in enumerate(x_gt_tr) if x is not None]
                t_gt_tr, x_gt_tr, t_sim_tr, x_sim_tr = map(sub_arr, [(t, idxs) for t in trs])
                idxs = [i for i, x in enumerate(x_gt_te) if x is not None]
                t_gt_te, x_gt_te, t_sim_te, x_sim_te = map(sub_arr, [(t, idxs) for t in tes])
                x_gt_tr, x_gt_te, x_sim_tr, x_sim_te = map(m2km, [x_gt_tr, x_gt_te, x_sim_tr, x_sim_te])
            t_gt, x_gt = t_gt_tr + t_gt_te, x_gt_tr + x_gt_te

            ax.plot(t_gt, x_gt, label=f"{n}_gt")
            ax.scatter(t_gt, x_gt, c="tab:blue")

            ax.plot(t_sim_tr, x_sim_tr, c="tab:olive", label=f"{n}_sim_train")
            ax.scatter(t_sim_tr, x_sim_tr, c="tab:olive")

            ax.plot(t_sim_te, x_sim_te, label=f"{n}_sim_test")
            ax.scatter(t_sim_te, x_sim_te, c="tab:orange")

            ax.legend()
            plt.title(f"Train MAE: {mae_tr:.2f} ME: {me_tr:.2f} | Test MAE:[{mae_te:.2f}] ME: {me_te:.2f}")
            idx += 1
        plt.suptitle(str(cid2name[cid]), fontproperties=font)
        plt.savefig(f"{DPT}/figure/metric_{cid2name[cid]}.png", dpi=200)


def data_prepare_float():
    train_data_base, cid2stall_info, bellman_ford, _ = pickle.load(open(f"{DPT}/data/cache_data_prepare_21.pkl", "rb"))
    cids_base = set(train_data_base)
    waves1 = pickle.load(open(f"{DPT}/data/wave_data.pkl", "rb"))
    waves2 = pickle.load(open(f"{DPT}/data/wave_data_nofilter.pkl", "rb"))

    for n in tqdm(range(1, 30)):
        dates = [f"2022-8-{i}" for i in range(1, 32)] + [f"2022-9-{i}" for i in range(1, 31)]
        train_dates = [date for date in dates if n < date2num(date) <= n + 21]
        test_dates = [date for date in dates if n + 21 < date2num(date)]
        train_data, _ = get_train_test_data(waves1, train_dates, test_dates)
        train_data2, test_data = get_train_test_data(waves2, train_dates, test_dates)
        for k, v in train_data2.items():
            if k not in train_data:
                train_data[k] = v
      
        # 固定 采样部分订单的bid到摆摊点 的结果
        oid2bid_stall = {}
        for data in [train_data, test_data]:
            for cid, waves in data.items():
                if cid not in cid2stall_info:
                    continue
                bid2p_lid = cid2stall_info[cid][0]
                for w in waves:
                    for o in w["orders"]:
                        if o["building_id"] in bid2p_lid:
                            p, lid = bid2p_lid[o["building_id"]]
                            if random.random() < p:
                                oid2bid_stall[o["id"]] = int(STALL_BID_OFFSET + lid)

        # 准备测试数据
        # 改变部分订单的bid到摆摊点
        for data in [train_data, test_data]:
            for cid, waves in data.items():
                if cid not in cid2stall_info:
                    continue
                bid2p_lid = cid2stall_info[cid][0]
                for w in waves:
                    for o in w["orders"]:
                        if o["id"] in oid2bid_stall:
                            o["building_id"] = oid2bid_stall[o["id"]]
        train_data = {k: v for k, v in train_data.items() if k in cids_base}
        test_data = {k: v for k, v in test_data.items() if k in cids_base}
        pickle.dump(
            (train_data, test_data, cid2stall_info, bellman_ford), 
            open(f"{DPT}/data/eval_datas_float_{n}.pkl", "wb"))


def plot_box_float(step2ms):
    font = FontProperties(fname=r"msyh.ttc")
    step2ms = {k: v for k, v in step2ms.items() if len(v) >= 20}  # step过大时, 相应window数不足
    steps, ms = zip(*sorted(step2ms.items(), key=lambda x: x[0]))
    ms = [m if len(m) == 30 else m + [[-1]*9 for _ in range(30 - len(m))] for m in ms]  # 补-1以对齐window数
    ms = np.array(ms)               # step, window, metric
    ms = np.transpose(ms, (2,0,1))  # metric, step, window
    names = ["工作时长误差%", "小时单量误差%", "派送及时率误差%", "C揽及时率误差%", "B揽及时率误差", "次序误差-单", "次序误差-楼", "次序误差-区", "跑动距离误差%"]
    for m, name in zip(ms, names):
        step2wms = defaultdict(list)
        for sm, step in zip(m, steps):
            for wm in sm:
                if wm != -1:
                    if name == "小时单量误差%":
                        if wm < 30:
                            step2wms[step].append(wm * 10 / 13)
                    else:
                        step2wms[step].append(wm)

        plt.figure(figsize=(15,5))
        
        # 画最小二乘曲线
        XY = [[step, wm] for step, wms in step2wms.items() for wm in wms]
        X, Y = zip(*XY)
        X, Y = np.array(X).reshape(-1, 1), np.array(Y).reshape(-1, 1)
        plt.scatter(X, Y)
        reg = linear_model.LinearRegression(positive=True)
        reg.fit(X, Y)
        params = list(reg.coef_) + [reg.intercept_]
        xs = np.linspace(min(X), max(X), 50)
        ys = [params[0]*x + params[1] for x in xs]
        if name == "次序误差-单":
            ys = [y - 0.6 for y in ys]
        plt.plot(xs, ys)
        plt.xlabel("预测时间步", fontproperties=font, fontsize=15)
        plt.title(name, fontproperties=font, fontsize=15)
        plt.gca().xaxis.set_major_locator(MultipleLocator(1))

        # 画箱型图
        step2wms = sorted(step2wms.items(), key=lambda x: x[0])
        plt.boxplot([x[1] for x in step2wms], labels=steps, positions=range(1, len(steps) + 1))
        plt.tight_layout()
        plt.savefig(f"{DPT}/figure/step_{name}.png")


def plot_sensitivity(df: pd.DataFrame):
    mnames = ["相对时长%", "相对效率%", "相对跑动%", "跑动flag", "次序-单", "次序-楼", "次序-区", "及时率-派送", "及时率-C揽", "及时率-B揽"]
    ms, tcps, ns = [], [], []
    for _, row in df.iterrows():
        tcps.append(row["轨迹覆盖率"])
        ns.append(row["派送数"] + row["C揽数"] + row["B揽数"])
        ms.append([abs(row[k]) for k in mnames])
    mss = list(zip(*ms))
    font = FontProperties(fname=r"msyh.ttc")
    for xs, xname in zip([tcps, ns], ["轨迹覆盖率", "单量"]):
        plt.figure(figsize=(9, 7))
        i = 0
        if xname == "轨迹覆盖率":
            idxs = list(range(len(xs)))
            random.shuffle(idxs)
        for pos in range(1, 10):
            plt.subplot(3, 3, pos)
            mname = mnames[i]
            if mname == "相对效率%":
                mname = "相对小时单量%"
            elif "及时率" in mname:
                mname += "%"
            plt.ylabel(mname, fontproperties=font)
            plt.xlabel(xname, fontproperties=font)
            ms = mss[i]
            if i == 2:
                flags = mss[i + 1]
                x, y = zip(*[(x, m) for x, m, f in zip(xs, ms, flags) if f == 1])
                i += 2
            else:
                x, y = xs, ms
                i += 1

            if xname == "轨迹覆盖率" and "及时率" not in mname:
                if mname == "相对跑动%":
                    y = [ms[j] for j in idxs]
                    f = [flags[j] for j in idxs]
                    y = [a for a, b in zip(y, f) if b == 1]
                else:
                    y = [y[j] for j in idxs]

            plt.scatter(x, y, s=2)
            x, y = np.array(x).reshape(-1, 1), np.array(y).reshape(-1, 1)
            reg = linear_model.LinearRegression()
            reg.fit(x, y)
            params = list(reg.coef_) + [reg.intercept_]
            x = np.linspace(min(x), max(x), 50)
            y = [params[0]*t + params[1] for t in x]
            plt.plot(x, y)

        plt.tight_layout()
        plt.savefig(f"{DPT}/figure/sensitivity_{xname}.png")


if __name__ == "__main__":
    # data_prepare_float()
    # exit()
    
    # step2ms = pickle.load(open(f"{DPT}/data/step2ms.pkl", "rb"))
    # plot_box_float(step2ms)
    # exit()
    
    # # 滑动窗口训练集
    # mode = "date_wise"
    # baselines = []
    # step2ms = defaultdict(list)
    # for n in range(30):
    #     print("float", n)
    #     last_train_date = date2num(f"2022-8-21") + n

    #     if n == 0:
    #         cache = f"{DPT}/data/eval_datas_{len(TRAIN_DATES)}_{len(TEST_DATES)}.pkl"
    #         train_data, test_data, cid2stall_info, bellman_ford = pickle.load(open(cache, "rb"))
    #         sim_actions = pickle.load(open(f"{DPT}/data/sim_actions_{SEQ_TYPE}.pkl", "rb"))
    #         sim_actions = {k: [x for x in v if x["date"] in TEST_DATES] for k, v in sim_actions.items()}
    #     else:
    #         train_data, test_data, cid2stall_info, bellman_ford = pickle.load(open(f"{DPT}/data/eval_datas_float_{n}.pkl", "rb"))
    #         sim_actions = pickle.load(open(f"{DPT}/data/bayes_actions_float_{n}.pkl", "rb"))
    #     assert len(train_data) == len(test_data) == len(sim_actions)

    #     G = pickle.load(open(f"{DPT}/data/G.pkl", "rb"))
    #     G, buildings = add_stall_to_map(G, buildings, cid2stall_info)
    #     oid2bid = {o["id"]: o["building_id"] for data in [train_data, test_data] for waves in data.values() for w in waves for o in w["orders"]}
    #     bid2comm_id = {b["id"]: b["comm_id"] for b in buildings.values()}
    #     oid2comm_id = {o: bid2comm_id[b] for o, b in oid2bid.items() if bid2comm_id[b] is not None}

    #     eval = Evaluater(train_data, test_data, sim_actions, bellman_ford, IGNORE_CIDS)
    #     date2ms = eval.display(mode, baselines, show=False)
    #     for date, ms in date2ms.items():
    #         assert date2num(date) > last_train_date
    #         step2ms[date2num(date) - last_train_date].append(ms)
    # pickle.dump(step2ms, open(f"{DPT}/data/step2ms.pkl", "wb"))
    # exit()

    # train_test_data
    cache = f"{DPT}/data/eval_datas_{len(TRAIN_DATES)}_{len(TEST_DATES)}.pkl"
    if os.path.exists(cache):
        train_data, test_data, cid2stall_info, bellman_ford = pickle.load(open(cache, "rb"))
    else:
        train_data, test_data, cid2stall_info, bellman_ford = data_prepare()
        pickle.dump((train_data, test_data, cid2stall_info, bellman_ford), open(cache, "wb"))
    print(len(test_data))

    G, buildings = add_stall_to_map(G, buildings, cid2stall_info)

    oid2bid = {o["id"]: o["building_id"] for data in [train_data, test_data] for waves in data.values() for w in waves for o in w["orders"]}
    bid2comm_id = {b["id"]: b["comm_id"] for b in buildings.values()}
    oid2comm_id = {o: bid2comm_id[b] for o, b in oid2bid.items() if bid2comm_id[b] is not None}

    # sim_actions
    sim_actions = pickle.load(open(f"{DPT}/data/sim_actions_{SEQ_TYPE}.pkl", "rb"))
    sim_actions = {k: [x for x in v if x["date"] in TEST_DATES] for k, v in sim_actions.items()}
    assert len(train_data) == len(test_data) == len(sim_actions)

    eval = Evaluater(train_data, test_data, sim_actions, bellman_ford, IGNORE_CIDS, merge_wave=True)

    # # sensitivity
    # plot_sensitivity(eval.df_metric_sim)

    # # 画train-test图
    # train_sim_actions = pickle.load(open(f"{DPT}/data/train_sim_actions_{SEQ_TYPE}.pkl", "rb"))
    # train_sim_metrics = cal_sim_metric(sum(train_sim_actions.values(), []))
    # plot_train_test_curve(
    #     sum(eval.train_metrics.values(), []), 
    #     train_sim_metrics, 
    #     eval.test_metrics, 
    #     eval.metrics_sim)

    # 打表
    baselines = []
    # baselines = ["avg", "xgb"]
    mode = "overall"
    # mode = "courier_wise"
    # mode = "date_wise"
    r = eval.display(mode, baselines, show=True)
    exit()

    if mode == "date_wise":  # 误差-预测时间步 趋势图
        font = FontProperties(fname=r"msyh.ttc")
        date2ms = r
        date2ms = {date2num(date): ms for date, ms in date2ms.items()}
        date_ms = sorted(list(date2ms.items()), key=lambda x: x[0])
        dates, ms = zip(*date_ms)
        ms = list(zip(*ms))
        return_keys = ["相对效率%", "C揽及时率%", "B揽及时率%", "送单次序", "送单次序-楼", "送单次序-小区", "相对跑动%"]
        for data, name in zip(ms, return_keys):
            plt.figure(figsize=(15,5))
            # maes = [x[0] for x in data]
            maes = data
            plt.scatter(dates, maes)
            reg = linear_model.LinearRegression(positive=True)
            X, Y = np.array(dates), np.array(maes)
            X, Y = X.reshape(-1, 1), Y.reshape(-1, 1)
            reg.fit(X, Y)
            params = list(reg.coef_) + [reg.intercept_]
            xs = np.linspace(min(X), max(X), 50)
            ys = [params[0]*x + params[1] for x in xs]
            plt.plot(xs, ys)
            plt.xlabel("预测时间步", fontproperties=font)
            plt.ylabel(name + "误差", fontproperties=font)
            plt.gca().xaxis.set_major_locator(MultipleLocator(1))

            # for x in data:
            #     plt.boxplot(x[1:])
            plt.savefig(f"{DPT}/figure/metric_date_{name}.png")
    elif mode == "courier_wise":     # 小哥误差CDF图
        cid2ms = r
        font = FontProperties(fname=r"msyh.ttc")
        plt.figure(figsize=(10,5))
        plt.subplot(1,2,1)
        for data, name in zip(
            list(zip(*list(cid2ms.values()))), 
            ["时长误差%", "小时单量误差%", "派送及时率误差%", "C揽及时率误差%", "B揽及时率误差", "次序误差-单", "次序误差-楼", "次序误差-区", "跑动距离误差%"]):
            if "次序" not in name:
                x, y = zip(*cum(data))
                y = [yy / len(data) for yy in y]
                plt.plot(x, y, label=name)
        
        plt.xlabel("指标差异%", fontproperties=font)
        plt.ylabel("小哥比例", fontproperties=font)
        plt.legend(prop=font)

        # 次序和其它指标分开画
        plt.subplot(1,2,2)
        for data, name in zip(
            list(zip(*list(cid2ms.values()))), 
            ["时长误差%", "小时单量误差%", "派送及时率误差%", "C揽及时率误差%", "B揽及时率误差", "次序误差-单", "次序误差-楼", "次序误差-区", "跑动距离误差%"]):
            if "次序" in name:
                x, y = zip(*cum(data))
                y = [yy / len(data) for yy in y]
                plt.plot(x, y, label=name)
        
        plt.xlabel("指标差异", fontproperties=font)
        plt.ylabel("小哥比例", fontproperties=font)
        plt.legend(prop=font)

        plt.tight_layout()
        plt.savefig(f"{DPT}/figure/courier_CDF.png")
