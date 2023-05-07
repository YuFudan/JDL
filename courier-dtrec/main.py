import os
import pickle
import random
from collections import Counter, defaultdict
from copy import deepcopy
from itertools import groupby
from math import ceil
from pprint import pprint

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from coord_convert.transform import gcj2wgs
from matplotlib.font_manager import FontProperties
from matplotlib.pyplot import MultipleLocator
from shapely.geometry import Point, Polygon
from sklearn import linear_model
from sklearn.cluster import DBSCAN
from tqdm import tqdm

from constants import *

# 地址校正
DIS_CLUSTER = 5            # 驻留位置聚类半径
LOC_T_PREV = 180           # 统计共现, 订单时间早于驻留起始时间的最大值
LOC_T_POST  = 3600         # 统计共现, 订单时间晚于驻留结束时间的最大值
LOC_NEAR_STAY = 50         # proposal驻留点附近的loc的距离阈值
LOC_NEAR_STAY2 = LOC_NEAR_STAY ** 2
LOC_NEAR_STAYSQRT2 = LOC_NEAR_STAY * 2 ** 0.5

# 驻留点匹配
MAX_T_PREV = 120           # 驻留点匹配, 订单时间早于驻留起始时间的最大值
MAX_T_POST = 1800          # 驻留点匹配, 订单时间晚于驻留结束时间的最大值
MAX_DIS = 100              # 驻留点匹配, 地址最大距离
MIN_MATCH_SCORE = 0.1      # 匹配得分阈值
BIG_MATCH_SCORE = 0.7      # 足够高的匹配得分
SCORE_INSTAY = 1.2         # 订单时间落在驻留时段内的得分

# 物理模型训练
TMIN_BASE = 10   # 保底时间(整理货物, 打电话)
TMAX_BASE = 180
TMIN_CUNIT = 5   # 单元间移动时间
TMAX_CUNIT = 20
TMIN_ORDER = 3   # 直接放门口
TMAX_ORDER = 30  # 等顾客开门
TMIN_STAIR = 12  # 上楼+下楼一层
TMAX_STAIR = 40 
TMIN_ELE_WAIT1 = 3    # 初次等
TMAX_ELE_WAIT1 = 20   
TMIN_ELE_WAIT2 = 3    # 非初次等
TMAX_ELE_WAIT2 = 10
TMIN_ELE = 4     # 上楼+下楼一层
TMAX_ELE = 8

TRAIN_DAYS = list(range(2, 8)) + list(range(801, 832))

random.seed(233)

MODE = "full"

# MODE = "no_stay_ref+mid"   # DTInf
# MODE = "no_stay_ref+unf"
# MODE = "no_stay_ref+smt"   # "smart"模式, 按照原始finish_time放缩

# MODE = "match_t+mid"
# MODE = "match_t+unf"
# MODE = "match_t+smt"

# MODE = "match_s+mid"
# MODE = "match_s+unf"
# MODE = "match_s+smt"

# MODE = "no_stay_ref"    # 不做驻留点提取校正
# MODE = "no_adr_ref"     # 不做地址校正
# MODE = "no_stay_adr_ref"

# MODE = "no_siter"       # 迭代中, 不修改驻留边界
# MODE = "no_miter"       # 迭代中, 不修改订单匹配
# # MODE = "no_piter"     # 迭代中, 不更新物理模型

# # MODE = "match_t"      # 仅据时间匹配 *
# # MODE = "match_s"      # 仅据空间匹配 * 
# # MODE = "tf_mid"       # 细粒度订单完成时间用中间值 *
# # MODE = "tf_unf"       # 细粒度订单完成时间用均匀分布 *
# # MODE = "no_stay_iter" # 迭代中, 不修改驻留边界
# # MODE = "no_match_iter"# 迭代中, 不修改订单匹配

print("MODE:", MODE)

T_MISMATCH = 600
N_ITER_IN = 5
N_ITER_OUT = 2

def get_regions():
    """
    划定1101真值4个小哥当天负责的区域
    """
    mxl = [
        [116.5173, 39.8246],
        [116.5244, 39.8252],
        [116.5252, 39.8225],
        [116.5184, 39.8220],
        [116.5173, 39.8246],
    ]
    xkjy = [
        [116.5136, 39.8211],  # 左上
        [116.5159, 39.8216],  # 凹陷左上
        [116.5165, 39.8191],  # 凹陷左下
        [116.5176, 39.8192],
        [116.5177, 39.8187],
        [116.5200, 39.8187],  # 凹陷右下
        [116.5185, 39.8220],  # 凹陷右上
        [116.5228, 39.8223],  # 右上
        [116.5234, 39.8192],  # 右下
        [116.5203, 39.8179],  # 下
        [116.5150, 39.8168],  # 左下
        [116.5136, 39.8211],  # 左上
    ]
    gyy = [
        [116.5151, 39.8131],
        [116.5326, 39.8204],
        [116.5428, 39.8080],
        [116.5275, 39.8013],
        [116.5151, 39.8131],
    ]

    bounds = [mxl, xkjy, gyy]
    bounds = [[gcj2wgs(*p) for p in ps] for ps in bounds]
    names = ["金色漫香林", "小康家园", "工业园区"]
    cids = [21777999, 22626330, 21173943]
    is_eles = [True,  False,    False]
    regions = [
        {
            "cid": cid,
            "name": name,
            "gps": gps,
            "xy": [projector(*p) for p in gps],
            "poly": Polygon([projector(*p) for p in gps]),
            "is_elevator": is_ele
        } for cid, gps, name, is_ele in zip(cids, bounds, names, is_eles)
    ]

    return regions


def cluster_algo(ps):
    """
    dbscan聚类, 将所有数据中的所有驻留点聚类, 得到loaction池
    """
    db = DBSCAN(eps=DIS_CLUSTER, min_samples=1, n_jobs=-1).fit(ps)
    labels = db.labels_

    # ps_labels = random.sample(list(zip(ps, labels)), 1000)
    # l2ps = defaultdict(list)
    # for p, l in ps_labels:
    #     l2ps[l].append(p)
    # plt.figure()
    # for l, ps in l2ps.items():
    #     c = default_colors[l % len(default_colors)] if l > 0 else l
    #     plt.scatter(*zip(*ps), s=0.1, c=[c]*len(ps), marker=".")
    # plt.savefig("figure/cluster_algo.pdf")

    return labels


def correct_order_address(wave_data, use_cache=True):
    """
    参考ICDE22, 用历史数据中订单地址与驻留位置的共现统计, 校正当天订单地址
    """
    date2waves = defaultdict(list)
    for w in wave_data:
        date2waves[w["date"]].append(w)

    # 对所有驻留位置做聚类, 确保逻辑上是同一位置的驻留点有相同loc
    stays_all = [s for x in wave_data for s in x["stays"]]
    if "stay" in MODE:
        cache_path = "data/locids_no_stay_ref.pkl"
    else:
        cache_path = "data/locids.pkl"
    if use_cache and os.path.exists(cache_path):
        print("use cache", cache_path)
        locids = pickle.load(open(cache_path, "rb"))
    else:
        locs_stay = [s["point"][:2] for s in stays_all]
        locids = cluster_algo(locs_stay)
        pprint(sorted(list(Counter(locids).items()), key=lambda x: -x[1])[:20])
        pickle.dump(locids, open(cache_path, "wb"))
    locid2xy = defaultdict(list)
    for s, locid in zip(stays_all, locids):
        s["locid"] = locid
        assert locid != -1
        locid2xy[locid].append(s["point"][:2])
    locid2xy = {locid: np.mean(np.array(xys), axis=0) for locid, xys in locid2xy.items()}

    def get_wave_occurance(orders, stays):
        """
        订单地址 与 驻留位置 的共现统计
        在一波内, 对每个订单, 记录其ads_id|building_id, 与在订单前一小时内出现的locid
        若订单时间落在驻留时间内, 认为是强共现, 否则是弱共现
        """
        stays = [(s["trange"], s["locid"]) for s in stays]
        adsid2locids = defaultdict(set)
        bid2locids = defaultdict(set)
        adsid2locids_strong = defaultdict(set)
        bid2locids_strong = defaultdict(set)
        all_adsid = set()
        all_bid = set()
        for o in orders:
            t = o["finish_time"]
            all_adsid.add(o["address_id"])
            all_bid.add(o["building_id"])
            for (t1, t2), locid in stays:
                if t1 - LOC_T_PREV <= t and t <= t2 + LOC_T_POST:
                    adsid2locids[o["address_id"]].add(locid)
                    bid2locids[o["building_id"]].add(locid)
                    if t1 <= t <= t2 + LOC_T_PREV:  # 考虑到正常也会稍晚妥投
                        adsid2locids_strong[o["address_id"]].add(locid)
                        bid2locids_strong[o["building_id"]].add(locid)
        bid2locids.pop(-1, None)
        bid2locids_strong.pop(-1, None)
        all_bid.discard(-1)
        return all_adsid, all_bid, adsid2locids, bid2locids, adsid2locids_strong, bid2locids_strong
    
    # 统计每天中的共现
    if "stay" in MODE:
        cache_path = "data/occs_no_stay_ref.pkl"
    else:
        cache_path = "data/occs.pkl"
    if use_cache and os.path.exists(cache_path):
        print("use cache", cache_path)
        date2occs = pickle.load(open(cache_path, "rb"))
    else:
        date2occs = {
            date: [get_wave_occurance(w["orders"], w["stays"]) for w in waves]
            for date, waves in date2waves.items()
        }
        pickle.dump(date2occs, open(cache_path, "wb"))

    def find_near_locs(x, y):
        """
        找一个坐标附近的所有locid
        """
        nonlocal locid2xy
        near_locs = set()
        for locid, (x1, y1) in locid2xy.items():
            a = abs(x - x1)
            if a < LOC_NEAR_STAYSQRT2:
                b = abs(y - y1)
                if a + b < LOC_NEAR_STAYSQRT2:
                    if a ** 2 + b ** 2 < LOC_NEAR_STAY2:
                        near_locs.add(locid)
        return near_locs

    def propose_locs(adsid, bid, locids, occs):
        """
        给定某订单的adsid和bid, 以及时间附近的驻留点附近的所有locid, 
        根据之前统计的共现, 进一步propose出其中最可能的locid
        """
        if not locids:
            return []

        # 所有含该adsid的波中, 与该adsid共现的有locid的波的比例大于阈值
        locid2occads = defaultdict(int)
        locid2occads_strong = defaultdict(int)
        occs_ads = [x for x in occs if adsid in x[0]]
        for locid in locids:
            for occ in occs_ads:
                locid2occads[locid] += locid in occ[2][adsid]  # adsid2locids
                locid2occads_strong[locid] += locid in occ[4][adsid]
        locid2score_ads = {
            i: (locid2occads[i] + locid2occads_strong[i]) / (len(occs_ads) + 1e-12)
            for i in locids
        }

        # 所有含该bid的波中, 与该bid共现的有locid的波的比例大于阈值
        if bid != -1:
            locid2occbd = defaultdict(int)
            locid2occbd_strong = defaultdict(int)
            occs_bd = [x for x in occs if bid in x[1]]
            for locid in locids:
                for occ in occs_bd:
                    locid2occbd[locid] += locid in occ[3][adsid]  # bid2locids
                    locid2occbd_strong[locid] += locid in occ[5][adsid]
            locid2score_bd = {
                i: (locid2occbd[i] + locid2occbd_strong[i]) / (len(occs_bd) + 1e-12)
                for i in locids
            }
        else:
            locid2score_bd = {i: 0 for i in locids}

        locids = {i for i in locids if locid2score_ads[i] > 0.2 or locid2score_bd[i] > 0.2}
        if not locids:
            return []

        # 所有不含bid的波中, 与任意adsid共现的有locid的波的比例小于阈值
        if bid != -1:
            occs_nobd = [x for x in occs if bid not in x[1]]
            locid2occnobd = defaultdict(int)
            for locid in locids:
                for occ in occs_nobd:
                    for v in occ[2].values():  # adsid2locids
                        if locid in v:
                            locid2occnobd[locid] += 1
                            break
            locid2score_nobd = {
                i: locid2occnobd[i]/ (len(occs_nobd) + 1e-12)
                for i in locids
            }
        else:
            locid2score_nobd = {i: 0 for i in locids}
        
        # 返回得分最高的几个
        locid2score = {
            i: 
            locid2score_ads[i] + 0.5 * locid2score_bd[i] - 0.5 * locid2score_nobd[i] 
            for i in locids
        }
        locid_score = [(l, s) for l, s in locid2score.items() if s > 0.5]
        locid_score.sort(key=lambda x: -x[1])
        return locid_score[:3]

    # 缓存: 找xy附近的locs
    if "stay" in MODE:
        cache_path = "data/xy2near_locs_no_stay_ref.pkl"
    else:
        cache_path = "data/xy2near_locs.pkl"
    if use_cache and os.path.exists(cache_path):
        print("use cache", cache_path)
        xy2near_locs = pickle.load(open(cache_path, "rb"))
    else:
        xy2near_locs = {}

    # 对某天每波的每单, 使用其它天的共现, 校准地址
    for date, waves in tqdm(date2waves.items()):
        occs = [occ for d, occs in date2occs.items() for occ in occs if d != date]  # 使用其它天的共现
        for w in waves:
            stays = [(s["trange"], s["point"][:2]) for s in w["stays"]]
            for o in w["orders"]:
                t = o["finish_time"]
                locids = set()
                for (t1, t2), xy in stays:  # 初步筛选出时间吻合的驻留点附近的所有loc
                    if t1 - LOC_T_PREV <= t and t <= t2 + LOC_T_POST:
                        if xy in xy2near_locs:
                            locids |= xy2near_locs[xy]
                        else:
                            near_locs = find_near_locs(*xy)
                            locids |= near_locs
                            xy2near_locs[xy] = near_locs
                locid_scores = propose_locs(o["address_id"], o["building_id"], locids, occs)  # 在这些loc中, 进一步根据共现筛选
                # if locid_scores:
                #     pprint(locid_scores)
                #     print("============================")
                o["loc_scores"] = [(locid2xy[locid], score) for locid, score in locid_scores]

    # 更新缓存
    pickle.dump(xy2near_locs, open(cache_path, "wb"))

    return wave_data
        
        
def get_train_test_data(wave_data):
    # 按日期划分训练集, 测试集
    train_data = [x for x in wave_data if x["date"] in TRAIN_DAYS]
    test_data = [x for x in wave_data if x["date"] not in TRAIN_DAYS]
    # 去除数据量少的小哥的训练和测试数据
    train_data.sort(key=lambda x: x["cid"])
    test_data.sort(key=lambda x: x["cid"])
    train_data = {k: list(v) for k, v in groupby(train_data, key=lambda x: x["cid"])}
    test_data = {k: list(v) for k, v in groupby(test_data, key=lambda x: x["cid"])}
    cids_to_remove = []
    for cid, test in test_data.items():
        train = train_data.get(cid, [])
        if len(train) < 20:
            # print("remove courier:", cid, cid2name.get(cid, "无名"), len(train), len(test))
            cids_to_remove.append(cid)
        else:
            # print("use courier:", cid, cid2name.get(cid, "无名"), len(train), len(test))
            pass
    return train_data, test_data


def cal_score_t(ts1, ts2, to):
    if "match_s" in MODE:
        return 1
    if to < ts1:
        t = ts1 - to
        return 0 if t > MAX_T_PREV else 1 - t / MAX_T_PREV
    elif to > ts2:
        t = to - ts2
        if t > MAX_T_POST:
            return 0
        else:
            return 1 - t / MAX_T_POST
    else:
        return SCORE_INSTAY  # 偏好订单落在驻留时间内
    

def cal_score_s(xs, ys, xo, yo):
    if "match_t" in MODE:
        return 1
    dis = ((xo - xs) ** 2 + (yo - ys) ** 2) ** 0.5
    return 0 if dis > MAX_DIS else 1 - dis / MAX_DIS


def match_s_o(s, o):
    score_t = cal_score_t(*s["trange"], o["finish_time"])
    if score_t < MIN_MATCH_SCORE:
        return 0, None
    # 找所有可能的地址
    if "adr" in MODE:
        o["xys"] = [o["xy"]]
    else:
        if o["building_id"] != -1:
            xys = [o["xy"]] + [buildings[o["building_id"]]["gate_xy"]] + [x[0] for x in o["loc_scores"]]
        else:
            xys = [o["xy"]] + [x[0] for x in o["loc_scores"]]
        if o["address_xy"]:
            xys.append(o["address_xy"])
        o["xys"] = xys
    # 依次尝试所有可能的地址, 找到足够好的结果则不继续找, 否则找完取最大
    matched_results = []
    for xy in o["xys"]:
        score = score_t * cal_score_s(*s["point"][:2], *xy)
        if score > BIG_MATCH_SCORE:
            return score, xy
        elif score >= MIN_MATCH_SCORE:
            matched_results.append([xy, score])
    if matched_results:
        max_score = max(x[-1] for x in matched_results)
        tmp = [x for x in matched_results if x[-1] == max_score]
        xy, score = tmp[0] if len(tmp) == 1 else random.choice(tmp)
        return score, xy
    else:
        return 0, None
        

def match_stay_order(wave):
    stays, orders = wave["stays"], wave["orders"]
    if not stays or not orders:
        return wave
    
    for s in stays:
        s["oids_matched"] = []
        s["match_xys"] = []
        s["match_scores"] = []
    for o in orders:
        oid = o["id"]
        t = o["finish_time"]
        if "match_t" in MODE:  # 模拟小哥延迟妥投的现象
            if random.random() < 0.3:  
                t = o["finish_time"] = t + random.uniform(0, 300)
        scores_t = [cal_score_t(*s["trange"], t) for s in stays]
        # 找所有可能的地址
        if "adr" in MODE:
            o["xys"] = [o["xy"]]
        else:
            if o["building_id"] != -1:
                xys = [o["xy"]] + [buildings[o["building_id"]]["gate_xy"]] + [x[0] for x in o["loc_scores"]]
            else:
                xys = [o["xy"]] + [x[0] for x in o["loc_scores"]]
            if o["address_xy"]:
                xys.append(o["address_xy"])
            o["xys"] = xys
        # 依次尝试所有可能的地址, 找到足够好的结果则不继续找, 否则找完取最大
        matched_results = []
        has_big_score = False
        for xy in o["xys"]:
            scores = []
            for s, score_t in zip(stays, scores_t): 
                if score_t < MIN_MATCH_SCORE:
                    scores.append(0)
                else:
                    score_s = cal_score_s(*s["point"][:2], *xy)
                    scores.append(score_t * score_s)
            max_score = max(scores)
            idxs = [i for i, score in enumerate(scores) if score == max_score]
            idx = idxs[0] if len(idxs) == 1 else random.choice(idxs)
            match_score = max_score
            if max_score > BIG_MATCH_SCORE:
                o["sid_matched"] = idx
                o["match_xy"] = xy
                o["match_score"] = match_score
                stays[idx]["oids_matched"].append(oid)
                stays[idx]["match_xys"].append(xy)
                stays[idx]["match_scores"].append(match_score)
                has_big_score = True
                break
            elif max_score >= MIN_MATCH_SCORE:
                matched_results.append([idx, xy, match_score])
        if not has_big_score and matched_results:
            max_score = max(x[-1] for x in matched_results)
            # if max_score > MIN_MATCH_SCORE:
            if True:
                tmp = [x for x in matched_results if x[-1] == max_score]
                idx, xy, score = tmp[0] if len(tmp) == 1 else random.choice(tmp)
                o["sid_matched"] = idx
                o["match_xy"] = xy
                o["match_score"] = score
                stays[idx]["oids_matched"].append(oid)
                stays[idx]["match_xys"].append(xy)
                stays[idx]["match_scores"].append(score)

    if MODE == "full":
        # 对于未匹配的订单, 认为是由于妥投时间严重滞后或者地址严重错误导致, 尝试根据时/空之一匹配(只匹配到未匹配的驻留点, 避免泛滥)
        orders_unmatched = [o for o in orders if "sid_matched" not in o]
        stays_unmatched = [s for s in stays if len(s["oids_matched"]) == 0]
        if orders_unmatched and stays_unmatched:
            for o in orders_unmatched:
                oid = o["id"]
                xys = o["xys"]
                t = o["finish_time"]
                scores_t = [cal_score_t(*s["trange"], t) for s in stays_unmatched]
                score_t_max = max(scores_t)
                if score_t_max > 0.8:  # 纯看时间匹配
                    idx = scores_t.index(score_t_max)
                    scores_s = [cal_score_s(*stays_unmatched[idx]["point"][:2], *xy) for xy in xys]
                    score_s_max = max(scores_s)
                    xy = xys[scores_s.index(score_s_max)]
                    score = score_t_max * score_s_max
                    o["sid_matched"] = idx
                    o["match_xy"] = xy
                    o["match_score"] = score
                    stays_unmatched[idx]["oids_matched"].append(oid)
                    stays_unmatched[idx]["match_xys"].append(xy)
                    stays_unmatched[idx]["match_scores"].append(score)
                else:  # 纯看空间匹配
                    continue
                    matched_results = []
                    for xy in xys:
                        scores_s = []
                        for s in stays_unmatched:
                            if t < s["trange"][0] - MAX_T_PREV:  # 认为不可能提前妥投
                                scores_s.append(0)
                            else:
                                scores_s.append(cal_score_s(*s["point"][:2], *xy))
                        score_s_max = max(scores_s)
                        if score_s_max > 0.9:
                            idxs = [i for i, score in enumerate(scores_s) if score == score_s_max]
                            idx = idxs[0] if len(idxs) == 1 else random.choice(idxs)
                            score = score_s_max * cal_score_t(*stays_unmatched[idx]["trange"], t)
                            matched_results.append([idx, xy, score, score_s_max])
                    if matched_results:
                        score_s_max = max(x[-1] for x in matched_results)
                        tmp = [x for x in matched_results if x[-1] == score_s_max]
                        idx, xy, score, _ = tmp[0] if len(tmp) == 1 else random.choice(tmp)
                        o["sid_matched"] = idx
                        o["match_xy"] = xy
                        o["match_score"] = score
                        stays_unmatched[idx]["oids_matched"].append(oid)
                        stays_unmatched[idx]["match_xys"].append(xy)
                        stays_unmatched[idx]["match_scores"].append(score)

    return wave


def refine_match_with_physics(wave, params):
    stays, orders = wave["stays"], wave["orders"]
    oid2odr = {o["id"]: o for o in orders}
    is_ele = len(params) == 6

    def pre_t_stay(oids):
        """
        物理模拟预测驻留时长
        """
        unit2floors = defaultdict(list)
        for oid in oids:
            o = oid2odr[oid]
            unit2floors[o["unit"]].append(o["floor"])
        onum = len(oids)
        unit = len(unit2floors)
        # wait1 = unit
        wait1 = sum(max(floors) > 1 for floors in unit2floors.values())
        floor = sum(max(floors) - 1 for floors in unit2floors.values())
        # wait2 = sum(len(set(floors)) - 1 for floors in unit2floors.values())
        wait2 = sum(len(set(floors) - {1}) for floors in unit2floors.values())
        if is_ele:
            return np.array([onum, unit - 1, wait1, wait2, floor, 1]) @ params
        else:
            return np.array([onum, unit - 1, floor, 1]) @ params

    # 筛选是inlier的驻留点, 判断是生产者还是消费者
    stays_inlier = []
    for s in stays:
        s["t"] = t = s["trange"][1] - s["trange"][0]
        if len(s["oids_matched"]) == 0:  # 未匹配到单的也算
            s["is_taker"] = True
            s["t_pre"] = t
            stays_inlier.append(s)
        else:
            s["t_pre"] = t_pre = pre_t_stay(s["oids_matched"])
            # if abs(t_pre - t) < 150:  # inlier
            if t - t_pre < 150:  # inlier
                s["is_taker"] = t_pre < t
                stays_inlier.append(s)

    def exchange_orders(giver, taker):
        """
        将giver的单匀给taker
        """
        t_g = giver["t"]
        loss_g = giver["t_pre"] - t_g
        t_t = taker["t"]
        loss_t = taker["t_pre"] - t_t  # 未匹配到单的, 初始loss为0
        last_loss = abs(loss_g) + abs(loss_t)
        oids_g = [x for x in sorted(list(zip(giver["oids_matched"], giver["match_scores"])), key=lambda x: x[1])]
        has_success = False
        for oid, score_g in oids_g:  # 按giver的匹配得分从低到高的顺序尝试将单交给taker
            # if len(giver["oids_matched"]) == 1:
            #     break  # 不允许把giver变成rest
            o = oid2odr[oid]
            score, xy = match_s_o(taker, o)
            if score < max(MIN_MATCH_SCORE * 1.5, score_g * 0.1):
                continue
            t_pre_g = pre_t_stay([x for x in giver["oids_matched"] if x != oid])
            t_pre_t = pre_t_stay(taker["oids_matched"] + [oid])
            loss_taker = abs(t_pre_t - t_t)
            if not taker["oids_matched"]:  # 若taker初始为休息, last_loss中为0, 计算新loss时打折
                loss_taker *= 0.3
            loss_giver = abs(t_pre_g - t_g)
            if len(giver["oids_matched"]) == 1:  # 若giver给出单后变成rest, loss为整个时长
                loss_giver = min(t_g, t_pre_g) * 2  # 调为2, acc变好0.2%, dtf变差0.7s
            if loss_taker + loss_giver < last_loss:  # 若交换后总loss减小, 则交换
                has_success = True
                for i, x in enumerate(giver["oids_matched"]):
                    if x == oid:
                        break
                for k in ["oids_matched", "match_xys", "match_scores"]:
                    giver[k] = giver[k][:i] + giver[k][i+1:]
                giver["t_pre"] = t_pre_g
                giver["is_taker"] = t_pre_g < t_g if giver["oids_matched"] else True
                taker["oids_matched"].append(oid)
                taker["match_xys"].append(xy)
                taker["match_scores"].append(score)
                taker["t_pre"] = t_pre_t
                taker["is_taker"] = t_pre_t < t_t
                last_loss = abs(t_pre_g - t_g) + abs(t_pre_t - t_t)
                if giver["is_taker"] != False and taker["is_taker"] != True:  # 当角色发生变化则提前终止
                    return has_success
        return has_success

    # 在相邻两驻留点间交换单
    if MODE != "no_miter":
        success_cnt = 0
        # s12s = zip(stays_inlier, stays_inlier[1:])
        s12s = zip(stays_inlier[::-1][1:], stays_inlier[::-1])
        for s1, s2 in s12s:
            if s2["trange"][0] - s1["trange"][1] > 300 or s1["is_taker"] == s2["is_taker"]:
                continue  # 只在时间不远的驻留点间交换单
            if not s1["is_taker"] and s2["is_taker"]:
                success_cnt += exchange_orders(s1, s2)
            elif s1["is_taker"] and not s2["is_taker"]:
                success_cnt += exchange_orders(s2, s1)
        # print("exchange num:", success_cnt)
   
    if MODE == "no_siter":
        return wave

    # 交换后, 认为订单匹配结果正确, 根据预测驻留时长, 调整驻留时长
    to_remove_j = []
    for i, s in enumerate(stays):
        t = s["t"]
        t_pre = s["t_pre"]
        ts, te = s["trange"]
        # if not (s["oids_matched"] and 0 < abs(t_pre - t) < 150):
        if not (s["oids_matched"] and t - t_pre < 150):
            continue
        td = min(10, (t_pre - t) * 0.1)
        # td = 0
        t1 = ts - td
        t2 = te + td
        if td <= 0: 
            s["trange"] = [t1, t2]
            s["t"] = t2 - t1
        else:  # 向两边扩张, 需要考虑两边的驻留点, 不能与之重合
            # 调整向两边扩张的比例, 可能不再是五五开
            tmin = -1e6
            for j in range(i - 1, -1, -1):
                ss = stays[j]
                if ss["oids_matched"]:
                    tmin = ss["trange"][1]
                    break
            tmax = 1e6
            for j in range(i + 1, len(stays)):
                ss = stays[j]
                if ss["oids_matched"]:
                    tmax = ss["trange"][0]
                    break 
            if t1 < tmin and t2 < tmax:  # t1受阻, t2还有空余, 则增大t2
                t2 += tmin - t1
                t2 = min(t2, tmax)
                t1 = tmin
            elif t2 > tmax and t1 > tmin:  # t2受阻, t1还有空余, 则减小t1
                t1 -= t2 - tmax
                t1 = max(tmin, t1)
                t2 = tmax
            # 进行扩张
            for j in range(i - 1, -1, -1):
                ss = stays[j]
                a, b = ss["trange"]
                if t1 >= b:
                    break
                else:
                    if ss["oids_matched"]:
                        t1 = b
                        break
                    else:
                        if t1 > a:
                            ss["trange"] = [a, t1]
                            ss["t"] = t1 - a
                            break
                        else:
                            to_remove_j.append(j)
            for j in range(i + 1, len(stays)):
                ss = stays[j]
                a, b = ss["trange"]
                if t2 <= a:
                    break
                else:
                    if ss["oids_matched"]:
                        t2 = a
                        break
                    else:
                        if t2 < b:
                            ss["trange"] = [t2, b]
                            ss["t"] = b - t2
                            break
                        else:
                            to_remove_j.append(j)
            s["trange"] = [t1, t2]
            s["t"] = t2 - t1
    wave["stays"] = [s for i, s in enumerate(stays) if i not in to_remove_j]

    return wave


def train_physics(waves, region):
    is_ele = region["is_elevator"]
    oid2odr = {o["id"]: o for w in waves for o in w["orders"]}

    stays = []
    for w in waves:
        poly = region["poly"]
        for s in w["stays"]:
            if len(s["oids_matched"]) > 0 and s["trange"][1] - s["trange"][0] < 900:  # 有匹配单且<15min
                p = Point(s["point"][:2])
                if p.within(poly):  # 在小哥1101负责的区域内
                    stays.append(s)

    samples = []
    filter_cnt = 0
    for s in stays:
        # 粗略估计最短和最长用时, 过滤驻留时长异常的
        unit2floors = defaultdict(list)
        oids = s["oids_matched"]
        for oid in oids:
            o = oid2odr[oid]
            unit2floors[o["unit"]].append(o["floor"])
        onum = len(oids)
        unit = len(unit2floors)
        # wait1 = unit
        wait1 = sum(max(floors) > 1 for floors in unit2floors.values())
        floor = sum(max(floors) - 1 for floors in unit2floors.values())
        # wait2 = sum(len(set(floors)) - 1 for floors in unit2floors.values())
        wait2 = sum(len(set(floors) - {1}) for floors in unit2floors.values())
        tmin = TMIN_BASE + TMIN_ORDER * onum + TMIN_CUNIT * (unit - 1)
        tmax = TMAX_BASE + TMAX_ORDER * onum + TMAX_CUNIT * (unit - 1)
        if is_ele:
            tmin += TMIN_ELE_WAIT1 * wait1 + TMIN_ELE_WAIT2 * wait2 + TMIN_ELE * floor
            tmax += TMAX_ELE_WAIT1 * wait1 + TMAX_ELE_WAIT2 * wait2 + TMAX_ELE * floor  
        else:
            tmin += TMIN_STAIR * floor
            tmax += TMAX_STAIR * floor
        t_stay = s["trange"][1] - s["trange"][0]
        if not tmin < t_stay < tmax:
            # print(tmin, tmax, t_stay)
            filter_cnt += 1
            continue
        if is_ele:
            samples.append([onum, unit - 1, wait1, wait2, floor, t_stay])
        else:
            samples.append([onum, unit - 1, floor, t_stay])
    # print("samples:", len(samples), filter_cnt)

    samples = np.array(samples)
    X, Y = samples[:, :-1], samples[:, -1]
    regs = [
        linear_model.LinearRegression(positive=True), 
        linear_model.Ridge(alpha=0.1, positive=True)
    ]
    results = []  # 每种回归器的最佳结果
    for reg in regs:
        mask = np.ones_like(Y, dtype=bool)
        cnt = 0
        while(True):
            X_train, Y_train = X[mask], Y[mask]
            reg.fit(X_train, Y_train)
            params = list(reg.coef_) + [reg.intercept_]

            losses = reg.predict(X) - Y
            last_mask = mask
            mask = np.abs(losses) < 150  # 下次只用loss小的训练

            # print("all    loss:", np.mean([abs(x) for x in losses]), np.mean(losses))
            inlier_losses = reg.predict(X_train) - Y_train
            # print("inlier loss:", np.mean([abs(x) for x in inlier_losses]), np.mean(inlier_losses))

            if np.all(mask == last_mask):  # mask不变则退出
                break
            cnt += 1
            if cnt > 10:
                break
        
        results.append((params, np.mean([abs(x) for x in inlier_losses])))

    params = min(results, key=lambda x: x[-1])[0]
    para_names = ["order", "cunit", "wait1", "wait2", "ele", "base"] \
        if is_ele else ["order", "cunit", "floor", "base"]
    # pprint({n: p for n, p in zip(para_names, params)})

    return params


def evaluate(wave, actions, do_plot=False, params=None):
    stays = wave["stays"]
    stay_ranges = [[s["trange"], bool(s["oids_matched"])] for s in stays]
    st, et = wave["wave"]
    oid2odr = {o["id"]: o for o in wave["orders"]}

    # 截取在波时段内的真值
    actions = deepcopy(actions)
    for i, a in enumerate(actions):
        if a["et"] > st:
            a["st"] = max(a["st"], st)  # 动作时间截取到波时段中
            break
    for j in range(len(actions)-1, -1, -1):
        a = actions[j]
        if a["st"] < et:
            a["et"] = min(a["et"], et)
            break
    actions = actions[i:j+1]

    def cal_recover_action_time(st, et):
        """
        计算在st到et时段内, 还原的work, rest, walk时长
        """
        nonlocal stay_ranges
        work = 0
        rest = 0
        for (t1, t2), is_work in stay_ranges:
            if t2 <= st:
                continue
            if t1 >= et:
                break
            t = min(et, t2) - max(st, t1)
            assert t >= 0
            if is_work:
                work += t
            else:
                rest += t
        walk = et - st - work - rest
        assert walk >= 0
        return work, rest, walk

    # 粗粒度指标: 行为类型acc
    total_t = 0    # 总时长
    correct_t = 0  # 正确时长
    dtt, dct, rtt, rct, wtt, wct = 0, 0, 0, 0, 0, 0  # deliver, rest, walk
    for a in actions:
        if a["act"] == IGNORE:
            continue
        st, et = a["st"], a["et"]
        t = a["et"] - a["st"]
        work, rest, walk = cal_recover_action_time(st, et)
        if a["act"] == WORK:
            total_t += t
            dtt += t
            correct_t += work
            dct += work
        elif a["act"] == OTHER:
            total_t += t
            wtt += t
            correct_t += rest + walk
            wct += rest + walk
        elif a["act"] == REST:
            total_t += t
            rtt += t
            correct_t += rest
            rct += rest

    if not params:
        return {"acc": np.array([correct_t, total_t, dct, dtt, wct, wtt, rct, rtt])}
    
    is_ele = len(params) == 6

    def pre_t_o(st, et, oids):
        """
        物理模拟预测订单完成时间
        """
        if is_ele:
            to, tu, tw1, tw2, te, tb = params
            te /= 2
        else:
            to, tu, tf, tb = params
            tfu, tfd = tf * 2 / 3, tf / 3
        odrs = [oid2odr[oid] for oid in oids]
        unit2odrs = defaultdict(list)
        oid2t = {}
        t = tb / 2  # 时间轴
        t_stones = [[t, ARRANGE]]
        for odr in odrs:
            unit2odrs[odr["unit"]].append(odr)
        for i, odrs in enumerate(unit2odrs.values()):
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
                    if is_ele:
                        t_wait = tw2 if already_wait else tw1
                        use_time = t_wait + te * up_num
                        already_wait = True
                    else:
                        use_time = tfu * up_num
                    t += use_time
                    last_floor = floor
                    t_stones.append([t, UP])
                # 送单
                for odr in odrs:
                    t += to
                    oid2t[odr["id"]] = t
                t_stones.append([t, DELIVER])
            # 下楼
            down_num = last_floor - 1
            if down_num > 0:
                if is_ele:
                    use_time = tw2 + te * down_num
                else:
                    use_time = tfd * down_num
                t += use_time
                last_floor = 1
                t_stones.append([t, DOWN])
            # 单元间移动
            if i < len(unit2odrs) - 1:
                t += tu
                t_stones.append([t, UNIT])
        t += tb / 2
        t_stones.append([t, ARRANGE])
        
        # 放缩
        p = (et - st) / t  
        oid2t = {oid: st + x * p for oid, x in oid2t.items()}
        for x in t_stones:
            x[0] = st + x[0] * p

        return oid2t, t_stones

    def pre_t_o_unif(st, et, oids):
        t = (et - st) / (len(oids) + 1)
        ts = [st + t * i for i in range(1, len(oids) + 1)]
        odrs = [oid2odr[oid] for oid in oids]
        odrs.sort(key=lambda x: (x["unit"], x["floor"]))

        oid2t = {o["id"]: t for o, t in zip(odrs, ts)}
        t = oid2t[odrs[0]["id"]]
        t_stones = [[t/3, ARRANGE], [t*2/3, UP], [t, DELIVER]]
        last_unit = odrs[0]["unit"]
        for odr in odrs[1:]:
            t1 = t_stones[-1][0]
            t2 = oid2t[odr["id"]]
            if odr["unit"] == last_unit:
                t = (t2 - t1) / 2
                t_stones.append([t1 + t, UP])
                t_stones.append([t2, DELIVER])
            else:
                t = (t2 - t1) / 4
                t_stones.append([t1 + t, DOWN])
                t_stones.append([t1 + 2 * t, UNIT])
                t_stones.append([t1 + 3 * t, UP])
                t_stones.append([t2, DELIVER])
                last_unit = odr["unit"]

        last_et = st
        ranges = []
        for t, atp in t_stones:
            ranges.append([[last_et, t], atp])
            last_et = t
        t = (last_et + et) / 2
        ranges.append([[last_et, t], DOWN])
        ranges.append([[t, et], ARRANGE])

        return oid2t, ranges

    def pre_t_o_smt(st, et, oids):
        odrs = [oid2odr[oid] for oid in oids]
        ts = sorted([o["finish_time"] for o in odrs])
        for i in range(len(ts) - 1):
            t1, t2 = ts[i], ts[i+1]
            if t1 >= t2:
                ts[i+1] = t1 + 1e-3
        if len(ts) > 1:
            T = et - st
            sst = st + T / 10
            eet = et - T / 10
            p = (eet - sst) / (ts[-1] - ts[0])
            ofst = sst - ts[0] * p
            ts = [t * p + ofst for t in ts]
        else:
            assert len(ts) == 1
            if not st <= ts[0] <= et:
                ts = [(st + et) / 2]

        odrs.sort(key=lambda x: (x["unit"], x["floor"]))
        oid2t = {o["id"]: t for o, t in zip(odrs, ts)}
        t = oid2t[odrs[0]["id"]]
        t_stones = [[t/3, ARRANGE], [t*2/3, UP], [t, DELIVER]]
        last_unit = odrs[0]["unit"]
        for odr in odrs[1:]:
            t1 = t_stones[-1][0]
            t2 = oid2t[odr["id"]]
            if odr["unit"] == last_unit:
                t = (t2 - t1) / 2
                t_stones.append([t1 + t, UP])
                t_stones.append([t2, DELIVER])
            else:
                t = (t2 - t1) / 4
                t_stones.append([t1 + t, DOWN])
                t_stones.append([t1 + 2 * t, UNIT])
                t_stones.append([t1 + 3 * t, UP])
                t_stones.append([t2, DELIVER])
                last_unit = odr["unit"]

        last_et = st
        ranges = []
        for t, atp in t_stones:
            ranges.append([[last_et, t], atp])
            last_et = t
        t = (last_et + et) / 2
        ranges.append([[last_et, t], DOWN])
        ranges.append([[t, et], ARRANGE])

        return oid2t, ranges

    # 细粒度指标: 订单完成时间差异
    oid2tf_gt = {}  # 订单完成时间真值
    idx_baitan = {33, 55, 100}  # 不比较摆摊时段内的单
    for a in actions:
        if a["oids"]:
            if {x["idx"] for x in a["actions_orig"]} & idx_baitan:
                continue
            _dict, t_stones = pre_t_o(a["st"], a["et"], a["oids"])
            oid2tf_gt.update(_dict)
            a["t_stones"] = t_stones
    oid2tf = {}  # 订单完成时间预测值
    stay_fine_ranges = []  # 细粒度行为区间
    for s in stays:
        t_mid = (s["trange"][0] + s["trange"][1]) / 2
        if s["oids_matched"]:
            if "mid" in MODE:
                oid2tf.update({oid: t_mid for oid in s["oids_matched"]})
                stay_fine_ranges.append([s["trange"], NOT_WORK])
            elif "unf" in MODE or "smt" in MODE:
                _f = pre_t_o_unif if "unf" in MODE else pre_t_o_smt
                _dct, ranges = _f(*s["trange"], s["oids_matched"])
                oid2tf.update(_dct)
                stay_fine_ranges += ranges
                assert abs(s["trange"][0] - ranges[0][0][0]) < 0.1
                assert abs(s["trange"][1] - ranges[-1][0][1]) < 0.1
            else:
                _dct, t_stones = pre_t_o(*s["trange"], s["oids_matched"])
                oid2tf.update(_dct)
                last_et = s["trange"][0]
                ranges = []
                for t, atp in t_stones:
                    ranges.append([[last_et, t], atp])
                    last_et = t
                stay_fine_ranges += ranges
                assert abs(s["trange"][0] - ranges[0][0][0]) < 0.1
                assert abs(s["trange"][1] - ranges[-1][0][1]) < 0.1
        else:
            stay_fine_ranges.append([s["trange"], NOT_WORK])

    def cal_recover_action_fine_time(st, et):
        """
        计算在st到et时段内, 还原的细粒度时长
        """
        nonlocal stay_fine_ranges
        up, down, unit, deliver, arrange = 0, 0, 0, 0, 0
        for (t1, t2), atp in stay_fine_ranges:
            if t2 <= st:
                continue
            if t1 >= et:
                break
            t = min(et, t2) - max(st, t1)
            assert t >= 0
            if atp == UP:
                up += t
            elif atp == DOWN:
                down += t
            elif atp == UNIT:
                unit += t
            elif atp == DELIVER:
                deliver += t
            elif atp == ARRANGE:
                arrange += t
            else:
                assert atp == NOT_WORK
        return up, down, unit, deliver, arrange

    # 细粒度指标: 行为类型acc
    total_fine_t = total_t - dtt   # 在rest, walk部分和粗粒度一样
    correct_fine_t = correct_t - dct
    uptt, upct, dott, doct, utt, uct, fdtt, fdct, att, act = [0 for _ in range(10)]  # up, down, unit, deliver, arrange
    for a in actions:
        if a["oids"]:
            if {x["idx"] for x in a["actions_orig"]} & idx_baitan:  # 不比较摆摊时段内的单
                continue
            st, et = a["st"], a["et"]
            total_fine_t += a["et"] - a["st"]
            t_stones = a["t_stones"]
            last_et = st
            for t, atp in t_stones:
                up, down, unit, deliver, arrange = cal_recover_action_fine_time(last_et, t)
                tt = t - last_et
                last_et = t
                if atp == UP:
                    uptt += tt
                    correct_fine_t += up
                    upct += up
                elif atp == DOWN:
                    dott += tt
                    correct_fine_t += down
                    doct += down
                elif atp == UNIT:
                    utt += tt
                    correct_fine_t += unit
                    uct += unit
                elif atp == DELIVER:
                    fdtt += tt
                    correct_fine_t += deliver
                    fdct += deliver
                elif atp == ARRANGE:
                    att += tt
                    correct_fine_t += arrange
                    act += arrange
                else:
                    assert False

    mis_match = 0
    losses = []
    for oid, t_gt in oid2tf_gt.items():
        if oid not in oid2tf:
            mis_match += 1
        else:
            if abs(oid2tf[oid] - t_gt) > T_MISMATCH:
                # print(oid)
                if oid != 313:
                    mis_match += 1
            else:
                losses.append(oid2tf[oid] - t_gt)
    default_loss = np.mean([abs(x) for x in losses]) * 1.5
    losses += [default_loss for _ in range(mis_match)]  # TODO:

    if do_plot:
        # 画图对比时间轴
        plt.cla()
        plt.figure(figsize=(30, 2))
        traj = wave["traj"]
        tj_st, tj_et = traj[0][-1], traj[-1][-1]
        plt.xlim((int(tj_st / 3600), ceil(tj_et / 3600)))
        plt.ylim((0.3, 2.1))
        plt.gca().xaxis.set_major_locator(MultipleLocator(0.1))
        font = FontProperties(fname=r"msyh.ttc")
        color_map = {
            WORK: "red",
            REST: "green",
            OTHER: "blue",
            IGNORE: "gray"
        }

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
        for group in groups:
            y = 1.3
            for t, odrs in group:
                x, xmin, xmax = t / 3600, (t - 5) / 3600, (t + 5) / 3600
                for o in odrs:
                    plt.hlines(y=y, xmin=xmin, xmax=xmax, color="red")
                    y += 0.02
                    plt.text(x=x, y=y, s=str(o["id"]), color="black", ha="center", size=1)
                    y += 0.05

        for s in stays:
            oids = s["oids_matched"]
            color = "red" if oids else "green"
            t1, t2 = s["trange"]
            plt.hlines(y=1.1, xmin=t1 / 3600, xmax=t2 / 3600, color=color)
            x = (t1 + t2) / 7200
            y = 1.15
            for oid, score in zip(oids, s["match_scores"]):
                plt.text(x=x, y=y, s=str(oid) + '+' + str(round(100*score,)), color="black", ha="center", size=1)
                y += 0.03
            if params and oids:
                unit2floors = defaultdict(list)
                for oid in oids:
                    o = oid2odr[oid]
                    unit2floors[o["unit"]].append(o["floor"])
                onum = len(oids)
                unit = len(unit2floors)
                floor = sum(max(floors) - 1 for floors in unit2floors.values())       # 需要爬楼层数
                wait2 = sum(len(set(floors)) - 1 for floors in unit2floors.values())  # 需要第二次等电梯次数
                if is_ele:
                    t_stay = np.array([onum, unit - 1, unit, wait2, floor, 1]) @ params
                else:
                    t_stay = np.array([onum, unit - 1, floor, 1]) @ params
                t = (t1 + t2) / 2
                t1, t2 = t - t_stay / 2, t + t_stay / 2
                plt.hlines(y=1.7, xmin=t1 / 3600, xmax=t2 / 3600, color="red")
                plt.text(x=t / 3600, y=1.8, s=str(onum) + "_" + str(floor) + "_" + str(unit-1), color="black", ha="center", size=1)
        for s in stays:
            oids = s["oids_matched"]
            for oid in oids:
                plt.scatter(y=1.1, x=oid2tf[oid] / 3600, color="black", s=0.5)

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
                plt.text(x=x, y=y, s=string, color="black", ha="center", size=1, fontproperties=font)
                y -= 0.03
            if a["oids"]:
                for oid in a["oids"]:
                    plt.text(x=x, y=y, s=str(oid), color="black", ha="center", size=1, fontproperties=font)
                    y -= 0.03
        for a in actions:
            if a["oids"]:
                for oid in a["oids"]:
                    if oid in oid2tf_gt:
                        plt.scatter(y=1, x=oid2tf_gt[oid] / 3600, color="black", s=0.5)
        plt.savefig(f"figure/match_gt_{cid2name[wave['cid']]}_{wave['date']}_{wave['wave_idx']}.pdf")

    return {
        "acc": np.array([correct_t, total_t, dct, dtt, wct, wtt, rct, rtt]),
        "mr": np.array([len(oid2tf_gt) - mis_match, len(oid2tf_gt)]),
        "dtf": losses,
        "acc_fine": np.array([correct_fine_t, total_fine_t, upct, uptt, doct, dott, uct, utt, fdct, fdtt, act, att])
    }


def main(train_waves, test_waves, region, actions):
    waves = [b for a in [train_waves, test_waves] for b in a]
    metrics_out = []
    for n_iter_out in tqdm(range(N_ITER_OUT)):  # 外环迭代
        # 物理模型更新
        if not (MODE == "no_piter" and n_iter_out > 0):
            params = train_physics(waves, region)
        # 驻留-订单更新
        metrics_in = []
        for n_iter_in in range(N_ITER_IN):  # 内环迭代
            if n_iter_out == 0 and n_iter_in == 0:  # 初始指标
                metrics = [evaluate(w, actions, params=params, do_plot=False) for w in test_waves]
                metrics_in.append({
                    "acc": sum(m["acc"] for m in metrics),
                    "mr": sum(m["mr"] for m in metrics),
                    "dtf": sum([m["dtf"] for m in metrics], []),
                    "acc_fine": sum(m["acc_fine"] for m in metrics),
                })
            # for w in waves:
            for w in test_waves:  # 只更新当天的驻留边界, 否则会导致物理模型更新过多, 这不好
                refine_match_with_physics(w, params)
            
            if n_iter_out == 0 and n_iter_in == 4:
                metrics = [evaluate(w, actions, params=params, do_plot=False) for w in test_waves]
                # show_case1(test_data[21173943][0], cid2label_actions[21173943])
            else:
                metrics = [evaluate(w, actions, params=params) for w in test_waves]
            metrics_in.append({
                "acc": sum(m["acc"] for m in metrics),
                "mr": sum(m["mr"] for m in metrics),
                "dtf": sum([m["dtf"] for m in metrics], []),
                "acc_fine": sum(m["acc_fine"] for m in metrics),
            })
        metrics_out.append(metrics_in)

    return metrics_out


if __name__ == "__main__":
    # 校正所有波的订单地址
    if MODE == "no_stay_ref":
        wave_data = pickle.load(open("data/wave_data_no_stay_ref.pkl", "rb"))
    else:
        wave_data = pickle.load(open("data/wave_data.pkl", "rb"))
    wave_data = correct_order_address(wave_data, use_cache=True)
    if MODE == "no_stay_ref":
        pickle.dump(wave_data, open("data/wave_data_no_stay_ref_corrected.pkl", "wb"))
    else:
        pickle.dump(wave_data, open("data/wave_data_corrected.pkl", "wb"))

    # 对所有波做驻留点匹配
    if "stay" in MODE:
        wave_data = pickle.load(open("data/wave_data_no_stay_ref_corrected.pkl", "rb"))
    else:
        wave_data = pickle.load(open("data/wave_data_corrected.pkl", "rb"))
    for w in tqdm(wave_data):
        match_stay_order(w)
    for x in ["stay", "adr", "match"]:
        if x in MODE:
            pickle.dump(wave_data, open(f"data/wave_data_{MODE}_matched.pkl", "wb"))
            break
    else:
        pickle.dump(wave_data, open("data/wave_data_matched.pkl", "wb"))

    # 划分训练和测试集
    for x in ["stay", "adr", "match"]:
        if x in MODE:
            try:
                wave_data = pickle.load(open(f"data/wave_data_{MODE}_matched.pkl", "rb"))
            except:
                wave_data = pickle.load(open(f"data/wave_data_{MODE.replace('smt', 'unf')}_matched.pkl", "rb"))
            break
    else:
        wave_data = pickle.load(open("data/wave_data_matched.pkl", "rb"))
    train_data, test_data = get_train_test_data(wave_data)

    cid2label_actions = pickle.load(open("ground_truth/gt_label_actions.pkl", "rb"))
    cid2region = {r["cid"]: r for r in get_regions()}

    # 迭代: 匀单-调驻留边界 + 物理模型重训
    cids = [21777999, 22626330, 21173943]
    cid2metrics = {}
    for cid in cids:
        print(cid2name[cid])
        actions = cid2label_actions[cid]
        region = cid2region[cid]
        metrics = main(train_data[cid], test_data[cid], region, actions)
        cid2metrics[cid] = [y for x in metrics for y in x]
        
        m0 = metrics[0][0]
        m1 = metrics[-1][-1]
        acc0 = m0["acc"][0] / m0["acc"][1]
        mr0 = m0["mr"][0] / m0["mr"][1]
        dtf0 = np.mean([abs(x) for x in m0["dtf"]])
        acc1 = m1["acc"][0] / m1["acc"][1]
        mr1 = m1["mr"][0] / m1["mr"][1]
        dtf1 = np.mean([abs(x) for x in m1["dtf"]])
        a, b, c = acc1 - acc0, mr1 - mr0, dtf1 - dtf0
        print("acc:", f"{a:+.6f}")
        print("mr:", f"{b:+.6f}")
        print("dtf:", f"{c:+.6f}")
        print("===========================================", dtf0)

    cid2accs = {}
    cid2mrs = {}
    cid2dtfs = {}
    cid2accs_fine = {}
    for cid, ms in cid2metrics.items():
        cid2accs[cid] = np.array([m["acc"] for m in ms])
        cid2mrs[cid] = np.array([m["mr"] for m in ms])
        cid2dtfs[cid] = [m["dtf"] for m in ms]
        cid2accs_fine[cid] = np.array([m["acc_fine"] for m in ms])
    cid2accs["full"] = sum(cid2accs.values())
    cid2mrs["full"] = sum(cid2mrs.values())
    cid2dtfs["full"] = [
        sum([cid2dtfs[cid][i] for cid in cids], [])
        for i in range(N_ITER_OUT * N_ITER_IN + 1)
    ]
    cid2accs_fine["full"] = sum(cid2accs_fine.values())
    cid2accs = {k: [
        [
            ct / tt,  
            dct / (dtt + 1e-12),
            wct / (wtt + 1e-12),
            rct / (rtt + 1e-12)
        ]
        for ct, tt, dct, dtt, wct, wtt, rct, rtt in v]
        for k, v in cid2accs.items()
    }
    cid2mrs = {k: [a / b for a, b in v] for k, v in cid2mrs.items()}
    cid2dtfs = {k: [np.mean([abs(x) for x in loss]) for loss in v] for k, v in cid2dtfs.items()}
    # cid2accs_fine = {k: [a / b for a, b in v] for k, v in cid2accs_fine.items()}
    cid2accs_fine = {k: [
        [
            ct / tt,  
            upct / (uptt + 1e-12),
            doct / (dott + 1e-12),
            uct / (utt + 1e-12),
            dct / (dtt + 1e-12),
            act / (att + 1e-12),
        ]
        for ct, tt, upct, uptt, doct, dott, uct, utt, dct, dtt, act, att in v]
        for k, v in cid2accs_fine.items()
    }
    a, b, c, d = \
        cid2accs["full"][-1][0] - cid2accs["full"][0][0], \
        cid2mrs["full"][-1] - cid2mrs["full"][0], \
        cid2dtfs["full"][-1] - cid2dtfs["full"][0], \
        cid2accs_fine["full"][-1][0] - cid2accs_fine["full"][0][0]
    print("final full acc:", f"{a:+.6f}")
    print("final full acc fine:", f"{d:+.6f}")
    print("final full mr:", f"{b:+.6f}")
    print("final full dtf:", f"{c:+.6f}")
    
    a, b, c, d = \
        max([x[0] for x in cid2accs["full"][1:]]) - cid2accs["full"][0][0], \
        max(cid2mrs["full"][1:]) - cid2mrs["full"][0], \
        min(cid2dtfs["full"][1:]) - cid2dtfs["full"][0], \
        max([x[0] for x in cid2accs_fine["full"][1:]]) - cid2accs_fine["full"][0][0]
    print("max full acc:", f"{a:+.6f}")
    print("max full acc fine:", f"{d:+.6f}")
    print("max full mr:", f"{b:+.6f}")
    print("min full dtf:", f"{c:+.6f}")
    
    print("ACC:", f"{max([x[0] for x in cid2accs['full']]):.6f}")
    print("ACC fine:", f"{max([x[0] for x in cid2accs_fine['full']]):.6f}")
    print("MR:", f"{max(cid2mrs['full']):.6f}")
    print("DTF:", f"{min(cid2dtfs['full']):.6f}")

    pickle.dump((cid2accs, cid2mrs, cid2dtfs, cid2accs_fine), open(f"data/metrics_{MODE}", "wb"))

    # font = FontProperties(fname=r"msyh.ttc")
    # plt.cla()
    # plt.figure()
    # for cid, accs in cid2accs.items():
    #     plt.plot(range(N_ITER_OUT * N_ITER_IN + 1), accs, label=f"{cid2name[cid]}" if isinstance(cid, int) else cid)
    # plt.legend(prop=font)
    # plt.savefig("figure/iter_acc.png")

    # plt.cla()
    # plt.figure()
    # for cid, mrs in cid2mrs.items():
    #     plt.plot(range(N_ITER_OUT * N_ITER_IN + 1), mrs, label=f"{cid2name[cid]}" if isinstance(cid, int) else cid)
    # plt.legend(prop=font)
    # plt.savefig("figure/iter_mr.png")

    # plt.cla()
    # plt.figure()
    # for cid, dtfs in cid2dtfs.items():
    #     plt.plot(range(N_ITER_OUT * N_ITER_IN + 1), dtfs, label=f"{cid2name[cid]}" if isinstance(cid, int) else cid)
    # plt.legend(prop=font)
    # plt.savefig("figure/iter_tf.png")

    print("MODE:", MODE)
