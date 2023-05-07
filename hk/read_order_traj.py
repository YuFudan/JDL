import json
import pickle
import random
import time
from collections import defaultdict

import numpy as np
import pandas as pd
from constants_all import *
from coord_convert.transform import gcj2wgs
from shapely.geometry import Point
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

MIN_ORDER_NUM = 50         # 某小哥某天的总单量阈值
MAX_ORDER_NUM = 500        # 某小哥某天的总单量阈值
DIS_REGION = 100           # 过滤路区外的轨迹点
MATCH_TO_NEAR_BD = 50      # 匹配到楼的距离阈值
MERGE_BPICK_T = 20
MERGE_BPICK_D = 20
WORKERS = 32

TP_MAP = {
    "派送": ORDER_DELIVER, 
    "C揽": ORDER_CPICK, 
    "B揽": ORDER_BPICK,
    "售后": ORDER_BPICK,  # TODO: 售后算成B揽
}


def fillin_nan_attr(orders):
    """处理订单building_id, unit, floor信息"""
    orders = [o for o in orders if o["finish_time"] >= o["start_time"]]
    orders.sort(key=lambda x: x["finish_time"])
    # 处理building_id为None的情况
    # 不直接暴力丢弃; 而是取时间最近的非-1的building_id
    t_bids = [(o["finish_time"], o["building_id"]) for o in orders if o["building_id"] != None]
    if len(t_bids) == 0:
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


def tackle_multiple_coord(traj):
    """处理同时刻多个不同坐标的问题"""
    traj = list(set(traj))
    t2xys = defaultdict(list)
    for x, y, t in traj:
        t2xys[t].append((x, y))
    t_xys = list(t2xys.items())
    t_xys.sort(key=lambda x: x[0])
    traj = []
    multiple_cnt= 0
    for t, xys in t_xys:
        if len(xys) == 1:
            traj.append((*xys[0], float(t)))
        else:
            xs, ys = zip(*xys)
            x, y = np.mean(xs), np.mean(ys)  # 取平均坐标
            dis = np.mean([((x1 - x) ** 2 + (y1 - y) ** 2) ** 0.5 for x1, y1 in xys])
            if dis < 10:
                traj.append((x, y, float(t)))
            else:
                multiple_cnt += 1
    return traj, multiple_cnt


def merge_bpick(orders):
    """若B揽间隔<20s, <20m 且中间没有夹着其它类型的单则合并"""
    def get_merged_order(odrs):
        if len(odrs) < 2:
            return odrs[0]
        t = [(o["gps"], o["start_time"], o["finish_time"], o["ddl_time"], o["floor"]) for o in odrs]
        gps, st, et, ddl, f = zip(*t)
        lon, lat = np.mean(np.array(gps), axis=0)
        return {
            "id": odrs[0]["id"],
            "type": ORDER_BPICK,
            "gps": [lon, lat],
            "start_time": min(st),
            "finish_time": max(et),
            "ddl_time": min(ddl),
            "unit": 1,
            "floor": max(f)}
    
    orders.sort(key=lambda x: x["finish_time"])
    bgroups = []
    bgroup = []
    for o in orders:
        if o["type"] != ORDER_BPICK:
            if bgroup:
                bgroups.append(bgroup)
                bgroup = []
        else:
            bgroup.append(o)
    if bgroup:
        bgroups.append(bgroup)
    r = [o for o in orders if o["type"] != ORDER_BPICK]
    assert len(r) + sum(len(x) for x in bgroups) == len(orders)
    for odrs in bgroups:
        t0, x0, y0 = odrs[0]["finish_time"], *projector(*odrs[0]["gps"])
        merge = [odrs[0]]
        for o in odrs[1:]:
            t, x, y = o["finish_time"], *projector(*o["gps"])
            if t - t0 < MERGE_BPICK_T and ((x - x0)**2 + (y - y0)**2)**0.5 < MERGE_BPICK_D:
                merge.append(o)
            else:
                r.append(get_merged_order(merge))
                merge = [o]
            t0, x0, y0 = t, x, y
        r.append(get_merged_order(merge))
    return r


def read_orders(paths):
    data = []
    for path in paths:
        data += pd.read_csv(path)[[
            "operator_id", "kind", 
            "lng_gcj", "lat_gcj",
            "end_time", "start_time1", "start_time2", "end_time1", "ddl_time1", 
            "units", "floors"]].values.tolist()
    cid_date2orders = defaultdict(list)
    uid = 0
    et_small_than_st_cnt = 0
    for cid, tp, lon, lat, date, st1, st2, et, ddl, u, f in tqdm(data):
        st = st1 if tp == "派送" else st2  # st1为派送收货时间, st2为揽收订单分配时间
        if np.isnan(st):
            continue
        if np.isnan(et) or not 0 <= et < 86400:
            continue
        if np.isnan(ddl):
            continue
        if et < st:
            et_small_than_st_cnt += 1
            # continue
            st = et - 3600
        cid = int(cid)
        tp = TP_MAP[tp]
        lon, lat = gcj2wgs(float(lon), float(lat))

        date = date.split(" ")[0]
        if "-" in date:
            y, m, d = date.split("-")
        elif "/" in date:
            y, m, d = date.split("/")
        else:
            assert False
        date = "-".join([y, f"{int(m):02d}", f"{int(d):02d}"])
        assert len(date) == 10

        if isinstance(u, str) and len(u) == 1 and u.isdigit():
            u = int(u)
        else:
            u = -1
        f = int(f)
        if f >= 0:
            f += 1  # 模拟器中1楼表示底层
        else:
            assert f == -1
        cid_date2orders[cid, date].append({
            "id": uid,
            "type": tp,
            "gps": [lon, lat],
            "start_time": st,
            "finish_time": et,
            "ddl_time": ddl,
            "unit": u,
            "floor": f,
        })
        uid += 1
    print("et_small_than_st_cnt:", et_small_than_st_cnt)
    return cid_date2orders


def match_order_to_building(o):
    p = Point(projector(*o["gps"]))
    if p.distance(region_fence["poly"]) > DIS_REGION:
        o["building_id"] = None
        return o
    dis_bids = []
    for bid, b in buildings.items():
        dis = p.distance(b["poly"])
        if dis <= 1:
            o["building_id"] = bid
            return o
        dis_bids.append([dis, bid])
    dis, bid = min(dis_bids)
    if dis < MATCH_TO_NEAR_BD:
        o["building_id"] = bid
    else:
        o["building_id"] = None
    return o


def read_traj(path, cid_date_set):
    data = pd.read_csv(
        path, 
        usecols=["gps_time", "operatorid", "gps_lng", "gps_lat"], 
        dtype={"gps_time": str, "operatorid": int, "gps_lng": float, "gps_lat": float})
    data = data[["operatorid", "gps_time", "gps_lng", "gps_lat"]].values
    cid_date2traj = defaultdict(list)
    for cid, t, lon, lat in tqdm(data):
        date = t.split(' ')[0]
        if "-" in date:
            y, m, d = date.split("-")
        elif "/" in date:
            y, m, d = date.split("/")
        date = "-".join([y, f"{int(m):02d}", f"{int(d):02d}"])
        assert len(date) == 10
        if (cid, date) not in cid_date_set:
            continue
        t = time.mktime(time.strptime(t, "%Y-%m-%d %H:%M:%S")) - \
            time.mktime(time.strptime(date + " 00:00:00", "%Y-%m-%d %H:%M:%S"))
        assert 0 <= t < 86400
        x, y = projector(*gcj2wgs(lon, lat))
        p = Point((x, y))
        if p.distance(region_fence["poly"]) < DIS_REGION:
            cid_date2traj[cid, date].append((x, y, t))
    multiple_cnt = 0
    traj_len = 0
    for k, traj in cid_date2traj.items():
        traj, cnt = tackle_multiple_coord(traj)
        cid_date2traj[k] = traj
        multiple_cnt += cnt
        traj_len += len(traj)
    print(traj_len, multiple_cnt)
    return cid_date2traj


def post_process(cid_date2orders, cid_date2traj):
    data = []
    for cid_date, orders in tqdm(cid_date2orders.items()):
        if len(orders) < MIN_ORDER_NUM:
            print("few order num1", len(orders), cid_date)
            continue

        traj = cid_date2traj[cid_date]
        if not traj:
            print("no traj", cid_date)
            continue

        orders = merge_bpick(orders)
        if len(orders) < MIN_ORDER_NUM:
            print("few order num2", len(orders), cid_date)
            continue

        orders = process_map(
            match_order_to_building,
            orders,
            chunksize=1, 
            max_workers=min(WORKERS, len(orders)), 
            disable=True)
        orders = fillin_nan_attr(orders)
        if len(orders) < MIN_ORDER_NUM:
            print("few order num3", len(orders), cid_date)
            continue

        data.append({
            "cid": cid_date[0],
            "date": cid_date[1],
            "orders": orders,
            "traj": traj
        })
    return data
            

if __name__ == "__main__":
    # path1 = "orig_data/order_2023-02-01_2023-02-28.csv"
    # path2 = "orig_data/order_2023-03-01_2023-03-14.csv"
    # cid_date2orders = read_orders([path1, path2])
    # pickle.dump(cid_date2orders, open("data/cid_date2orders.pkl", "wb"))
    cid_date2orders = pickle.load(open("data/cid_date2orders.pkl", "rb"))
    print("order num:", sum(len(x) for x in cid_date2orders.values()))
    
    # path = "orig_data/traj.csv"
    # cid_date2traj = read_traj(path, set(cid_date2orders))
    # pickle.dump(cid_date2traj, open("data/cid_date2traj.pkl", "wb"))
    cid_date2traj = pickle.load(open("data/cid_date2traj.pkl", "rb"))

    data = post_process(cid_date2orders, cid_date2traj)
    pickle.dump(data, open("data/order_traj.pkl", "wb"))
    # data = pickle.load(open("data/order_traj.pkl", "rb"))
    print(sum(len(x["orders"]) for x in data))

    cid2data = group_by(data, "cid")
    pprint({cid2name[k]: len(v) for k, v in cid2data.items()})

    date2data = group_by(data, "date")
    pprint({k: len(v) for k, v in date2data.items()})
