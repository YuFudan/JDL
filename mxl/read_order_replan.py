import json
import pickle
import random
import time
import pandas as pd
from collections import defaultdict

import numpy as np
from constants_all import *
from params_eval import *
from coord_convert.transform import gcj2wgs
from shapely.geometry import Point
from tqdm import tqdm

MATCH_TO_NEAR_BD = 50      # 匹配到楼的距离阈值

TP_MAP = {
    "派送": ORDER_DELIVER, 
    "C揽": ORDER_CPICK, 
    "B揽": ORDER_BPICK,
    "售后": ORDER_BPICK,
}


def get_wave_start_time(cid2date2waves):
    cid2wst = {}  # 求波的平均开始时间
    for cid, date2waves in cid2date2waves.items():
        mornings, afternoons = [], []
        for date, ws in date2waves.items():
            ms = [w["wave_traj"][0] for w in ws if w["is_morning"]]
            if ms:
                mornings.append(min(ms))
            ns = [w["wave_traj"][0] for w in ws if not w["is_morning"]]
            if ns:
                afternoons.append(min(ns))
        cid2wst[cid] = [
            np.mean(mornings) if mornings else None,
            np.mean(afternoons) if afternoons else None,
        ]
    dft_morning = np.mean([x[0] for x in cid2wst.values() if x[0]])
    dft_noon = np.mean([x[1] for x in cid2wst.values() if x[1]])
    # print(time_conventer(dft_morning))
    # print(time_conventer(dft_noon))
    for x in cid2wst.values():
        if x[0] is None:
            x[0] = dft_morning
        if x[1] is None:
            x[1] = dft_noon
    return cid2wst


def match_order_to_building(p: Point):
    dis_bids = []
    for bid, b in buildings.items():
        dis = p.distance(b["poly"])
        if dis <= 1:
            return bid
        dis_bids.append([dis, bid])
    dis, bid = min(dis_bids)
    if dis < MATCH_TO_NEAR_BD:
        return bid
    else:
        return None


def read(paths, cid2wst):
    """
    区分两波: 派送以ddl 15:00为界限, 揽收看start_time
    波的开始时间: 训练集平均
    """
    data = []
    for path in paths:
        data += pd.read_csv(
            path, 
            usecols=["operator_id", "kind", "日期", "deadline", "lat", "lng", "coarse_aoi_id", "create_time"], 
            dtype={"operator_id": int, "kind": str, "日期": str, "deadline": str, "lat": float, "lng": float, \
                   "coarse_aoi_id": str, "create_time": str})[[
            "operator_id", "kind", "日期", "deadline", "lat", "lng", "coarse_aoi_id", "create_time"]].values.tolist()      
    uid = 0
    cid_date2orders = defaultdict(list)
    st_err_cnt, nobd_cnt = 0, 0
    for cid, tp, date, ddl, lat, lon, bid, st in tqdm(data):
        if cid not in cid2date2waves:
            continue

        tp = TP_MAP[tp]

        assert "-" in date and len(date) == 10
        t_base = t_str2stp(date + " 00:00:00")
        ddl = t_str2stp(ddl) - t_base
        st = t_str2stp(st) - t_base
        if tp != ORDER_DELIVER:
            if not 0 <= st < 86400:
                st_err_cnt += 1
                continue

        # bid = bid_mapping[bid]  # TODO: 给的aoi_id与之前拿到的aoi文件的id不匹配
        lon, lat = gcj2wgs(lon, lat)
        bid = match_order_to_building(Point(projector(lon, lat)))
        if bid is None:
            nobd_cnt += 1
            continue

        cid_date2orders[cid, date].append({
            "id": uid,
            "type": tp,
            "gps": [lon, lat],
            "start_time": st,
            "ddl_time": ddl,
            "building_id": bid,
            "unit": 1,   # TODO: 无数据
            "floor": 1   # 无数据
        })
        uid += 1
    print("st_err_cnt:", st_err_cnt)
    print("nobd_cnt:", nobd_cnt)

    cid2waves = defaultdict(list)
    for (cid, date), orders in cid_date2orders.items():
        mst, nst = cid2wst[cid]
        odrs_d = [o for o in orders if o["type"] == ORDER_DELIVER]
        odrs_d1 = [o for o in odrs_d if o["ddl_time"] < 15.1 * 3600]
        odrs_d2 = [o for o in odrs_d if o["ddl_time"] >= 15.1 * 3600]
        odrs_nd = [o for o in orders if o["type"] != ORDER_DELIVER]
        odrs_nd1 = [o for o in odrs_nd if o["start_time"] < nst - 3600]
        odrs_nd2 = [o for o in odrs_nd if o["start_time"] >= nst - 3600]
        print(len(odrs_d1), len(odrs_d2), len(odrs_nd1), len(odrs_nd2))
        cnt = 0
        if odrs_d1 or odrs_nd1:
            cid2waves[cid].append({
                "cid": cid,
                "date": date,
                "orders": odrs_d1 + odrs_nd1,
                "start_time": mst,
                "wave_idx": cnt,
                "is_morning": True
            })
            cnt += 1
        if odrs_d2 or odrs_nd2:
            cid2waves[cid].append({
                "cid": cid,
                "date": date,
                "orders": odrs_d2 + odrs_nd2,
                "start_time": nst,
                "wave_idx": cnt,
                "is_morning": False
            })
            cnt += 1
    return cid2waves


if __name__ == "__main__":
    # result = pickle.load(open("data/orders_replan_month_day_intime.pkl", "rb"))
    # date2cid2no_mdi = []
    # for r in result:
    #     ws = sum(r.values(), [])
    #     date2ws = group_by(ws, "date")
    #     date2cid2ws = {date: group_by(ws, "cid") for date, ws in date2ws.items()}
    #     date2cid2no = {}
    #     for date, cid2ws in date2cid2ws.items():
    #         print(date)
    #         cid2no = defaultdict(int)
    #         for cid, ws in cid2ws.items():
    #             for w in ws:
    #                 cid2no[cid] += len(w["orders"])
    #         date2cid2no[date] = cid2no
    #     date2cid2no_mdi.append(date2cid2no)
    # a, b, c = date2cid2no_mdi
    # for date, cid2no_a in a.items():
    #     print(date)
    #     cid2no_b = b[date]
    #     cid2no_c = c[date]
    #     cid2no = {cid: [n, cid2no_b[cid], cid2no_c[cid]] for cid, n in cid2no_a.items()}
    #     pprint(cid2no)
    # exit()

    cache = f"data/eval_datas_{len(TRAIN_DATES)}_{len(TEST_DATES)}.pkl"
    train_data, test_data, cid2stall_info, _ = pickle.load(open(cache, "rb"))
    cid2date2waves = defaultdict(lambda: defaultdict(list))
    for data in [train_data, test_data]:
        for cid, ws in data.items():
            for w in ws:
                cid2date2waves[cid][w["date"]].append(w)
    cid2wst = get_wave_start_time(cid2date2waves)
    # for mst, nst in cid2wst.values():
    #     print(time_conventer(mst), time_conventer(nst))

    # bid_mapping = open("orig_data/manxianglin_idmapping.csv").readlines()
    # bid_mapping = {
    #     line.split(',')[1].strip('\n'):
    #     int(line.split(',')[0])
    #     for line in bid_mapping}

    result = []
    for mode in ["规划", "计划", "执行"]:
        paths = [
            f"orig_data/order_3/{mode}_正常_订单2023-03-{date:02d}.csv" 
            for date in range(3, 10)]
        cid2waves = read(paths, cid2wst)
        for cid, waves in cid2waves.items():
            if cid in cid2stall_info:
                bid2p_lid = cid2stall_info[cid][0]
                for w in waves:
                    for o in w["orders"]:
                        if o["building_id"] in bid2p_lid:
                            p, lid = bid2p_lid[o["building_id"]]
                            if random.random() < p:
                                o["building_id"] = int(STALL_BID_OFFSET + lid)
        result.append(cid2waves)
        print(len(cid2waves))
    pickle.dump(result, open("data/orders_replan_month_day_intime.pkl", "wb"))
