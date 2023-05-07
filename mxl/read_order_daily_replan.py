import json
import pickle
import random
import time
import pandas as pd
from collections import defaultdict, Counter

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


def fillin_nan_attr(orders):
    """处理订单building_id, unit, floor信息"""
    for o in orders:
        if o["start_time"] > o["finish_time"]:
            o["start_time"] = o["finish_time"] - 600
    orders.sort(key=lambda x: x["finish_time"])
    # 处理building_id为None的情况
    # 不直接暴力丢弃; 而是取时间最近的非-1的building_id
    t_bids = [(o["finish_time"], o["building_id"]) for o in orders if o["building_id"] is not None]
    if len(t_bids) == 0:
        assert False
        return []
    for o in orders:
        if o["building_id"] == None:
            t = o["finish_time"]
            if len(t_bids) == 1:
                o["building_id"] = t_bids[0][1]
            else:
                for (t1, b1), (t2, b2) in zip(t_bids, t_bids[1:]):
                    if t2 >= t:
                        break
                o["building_id"] = b1 if abs(t-t1) < abs(t-t2) else b2      
    # 处理unit, floor
    for o in orders:
        floor = int(o["floor"])
        bd = buildings[o["building_id"]]
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
        floor = max(1, min(20, floor))
        o["floor"] = floor
        unit = int(o["unit"])
        if unit == -1:
            unit = 1
        unit = max(1, min(10, unit))
        o["unit"] = unit
    return orders


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


def get_wave_range(orders):
    def cum(ts):
        num0 = len([t for t in ts if t <= 0])
        t_nums = sorted(list(Counter([t for t in ts if t > 0]).items()), key=lambda x:x[0])
        points = []
        cnt = num0
        for t, n in t_nums:
            cnt += n
            points.append((t, cnt))
        return points
    
    finish_times = [o["finish_time"] for o in orders]
    finish_points = cum(finish_times)
    
    def get_st(finish_points):
        for i in range(len(finish_points)-1, -1, -1):
            t1, n1 = finish_points[i]
            if n1 > len(orders) * 0.2:
                continue 
            if i == 0:  
                return t1
            for j in range(i-1, -1, -1):
                t2, n2 = finish_points[j]  # 往前找到订单完成数减少 达到一定阈值的t2(考虑到在站里也会有零星的完成)
                if n1 - n2 >= 3:
                    break
            assert j < i
            if t1 - t2 > 3600:
                return t1 - 600
        return finish_points[0][0]

    def get_et(finish_points):
        for i, (t1, n1) in enumerate(finish_points):
            if n1 < len(orders) * 0.8:
                continue
            if i == len(finish_points) - 1:  
                return t1
            for j in range(i+1, len(finish_points)):
                t2, n2 = finish_points[j]  # 往后找到订单完成数增加 达到一定阈值的t2(考虑到在站里也会有零星的完成)
                if n2 - n1 >= 3:
                    break
            assert j > i
            if t2 - t1 > 3600:
                return t1 + 600
        return finish_points[-1][0]

    st, et = get_st(finish_points), get_et(finish_points)
    assert et > st, (time_conventer(st), time_conventer(et))
    return st, et


def read(path):
    # # 按finish_time分波, 15:30为界, 并推断波开始时间
    # data = pd.read_csv(
    #     path, 
    #     usecols=["operator_id", "kind", "日期", "lat", "lng", "start_time1", "end_time1", "ddl_time1", "units", "floors"], 
    #     dtype={"operator_id": int, "kind": str, "日期": str, "lat": float, "lng": float, \
    #             "start_time1": np.float64, "end_time1": np.float64, "ddl_time1": np.float64})[[
    #     "operator_id", "kind", "日期", "lat", "lng", "start_time1", "end_time1", "ddl_time1", "units", "floors"]].values.tolist()      
    # uid = 0
    # cid_date2orders = defaultdict(list)
    # st_err_cnt, nobd_cnt = 0, 0
    # for cid, tp, date, lat, lon, st, et, ddl, u, f in tqdm(data):
    #     if np.isnan(st) or np.isnan(ddl):
    #         continue

    #     tp = TP_MAP[tp]

    #     assert "-" in date and len(date) == 10
        
    #     lon, lat = gcj2wgs(lon, lat)
    #     bid = match_order_to_building(Point(projector(lon, lat)))
    #     if bid is None:
    #         nobd_cnt += 1
    #         continue
        
    #     st, et, ddl = float(st), float(et), float(ddl)
    #     if tp != ORDER_DELIVER:
    #         if not 0 <= st < 86400:
    #             st_err_cnt += 1
    #             continue
        
    #     if isinstance(u, str) and len(u) == 1 and u.isdigit():
    #         u = int(u)
    #     else:
    #         u = -1
    #     f = int(f)
    #     if f >= 0:
    #         f += 1  # 模拟器中1楼表示底层
    #     else:
    #         assert f == -1
        
    #     cid_date2orders[cid, date].append({
    #         "id": uid,
    #         "type": tp,
    #         "gps": [lon, lat],
    #         "building_id": bid,
    #         "start_time": st,
    #         "finish_time": et,
    #         "ddl_time": ddl,
    #         "unit": u,
    #         "floor": f,
    #     })
    #     uid += 1
    # print("st_err_cnt:", st_err_cnt)
    # print("nobd_cnt:", nobd_cnt)
    # pickle.dump(cid_date2orders, open("data/cid_date2orders_daily_replan.pkl", "wb"))
    cid_date2orders = pickle.load(open("data/cid_date2orders_daily_replan.pkl", "rb"))

    wave_data = []
    for (cid, date), orders in cid_date2orders.items():
        odrs_d = [o for o in orders if o["type"] == ORDER_DELIVER]
        odrs_d1 = [o for o in odrs_d if o["finish_time"] < 15.5 * 3600]
        odrs_d2 = [o for o in odrs_d if o["finish_time"] >= 15.5 * 3600]
        odrs_nd = [o for o in orders if o["type"] != ORDER_DELIVER]
        odrs_nd1 = [o for o in odrs_nd if o["finish_time"] < 15.5 * 3600]
        odrs_nd2 = [o for o in odrs_nd if o["finish_time"] >= 15.5 * 3600]
        odrs1, odrs2 = odrs_d1 + odrs_nd1, odrs_d2 + odrs_nd2
        f1, f2 = len(odrs1) > 10, len(odrs2) > 10
        if f1:
            st1, et1 = get_wave_range(odrs1)
            print(time_conventer(st1), time_conventer(et1), len(odrs1))
        if f2:
            st2, et2 = get_wave_range(odrs2)
            print(time_conventer(st2), time_conventer(et2), len(odrs2))
        if f1 and f2:
            et1 = min(et1, 15.5 * 3600)
            st2 = max(st2, 15.5 * 3600)
        cnt = 0
        if f1:
            wave_data.append({
                "cid": cid,
                "date": date,
                "orders": fillin_nan_attr(odrs1),
                "stays": [],
                "traj": [],
                "traj_orig": [],
                "traj_cover": 0.0,
                "wave_traj": (st1, et1),
                "wave_idx": cnt,
                "is_morning": True
            })
            cnt += 1
        if f2:
            wave_data.append({
                "cid": cid,
                "date": date,
                "orders": fillin_nan_attr(odrs2),
                "stays": [],
                "traj": [],
                "traj_orig": [],
                "traj_cover": 0.0,
                "wave_traj": (st2, et2),
                "wave_idx": cnt,
                "is_morning": False
            })
            cnt += 1
    return wave_data


if __name__ == "__main__":
    wave_data = read("orig_data/order_3/日分单数据地址new.csv")
    print(len(wave_data))
    pickle.dump(wave_data, open("data/wave_data_daily_replan.pkl", "wb"))

    print(len(set(x["cid"] for x in wave_data)))
    print(set(x["date"] for x in wave_data))
