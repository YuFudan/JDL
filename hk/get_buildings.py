import json
import os
import pickle
from collections import defaultdict

import folium
import numpy as np
import pandas as pd
import requests
from constants_spatial import *
from coord_convert.transform import gcj2wgs, wgs2bd, wgs2gcj
from shapely.geometry import Point, Polygon
from tqdm import tqdm

"""
输出格式: 
{
    'id': 0,
    'orig_id': '7vWcHxkxOcv+xZ3d+ftDIW',
    'name': '赛莱克斯微系统科技(北京)有限公司',
    'points': [...],
    'region_id': 7,
    'comm_id': 105,
    'is_elevator': False,
    'default_elevator_tag': True,
    'floor': None,
    'gate_id': '0',
    'gate_gps': (116.53881983859398, 39.79677345306521),
    'gate_xy': (-1504.8223119530483, 1159.5798213851324),
}
"""

def plot_buildings(buildings, regions=None, comms=None, path="figure/test.html"):
    G_folium = get_base_map()
    for b in buildings:
        if b["is_elevator"] is None or b.get("default_elevator_tag", False):
            color = "orange"
        else:
            color = "red" if b["is_elevator"] else "blue"
        folium.PolyLine(
            locations=[wgs2gcj(*p)[::-1] for p in b["points"]],
            opacity=0.8,
            weight=2,
            color=color,
            popup=b["name"]
        ).add_to(G_folium)
        folium.CircleMarker(
            location=wgs2gcj(*b["gate_gps"])[::-1],
            radius=2,
            color=color,
            popup=(b["name"], b["is_elevator"], b["floor"])
        ).add_to(G_folium)
        if "points_nogap" in b:
            folium.PolyLine(
                locations=[wgs2gcj(*p)[::-1] for p in b["points_nogap"]],
                opacity=0.3,
                weight=1,
            ).add_to(G_folium) 
    if regions:
        for r in regions:
            folium.PolyLine(
                locations=[wgs2gcj(*p)[::-1] for p in r["gps"]],
                opacity=0.8,
                weight=3,
                color="red"
            ).add_to(G_folium)
    if comms:
        for c in comms:
            folium.PolyLine(
                locations=[wgs2gcj(*p)[::-1] for p in c["gps"]],
                opacity=0.5,
                weight=1,
                color="green"
            ).add_to(G_folium)
    G_folium.save(path)


def read_building_data(path):
    data = pd.read_excel(path)
    buildings = []
    for i, line in enumerate(data.values):
        poly, bid, name, _, is_elevator, floor, point = line
        polys = poly[10:-2].split("),(")  # 有1个楼里面有2个polygon
        points_all = []
        for poly in polys:
            points = [x.split(" ") for x in poly.split(",")]
            if points[-1] != points[0]:
                points.append(points[0])
            points_all.append([gcj2wgs(float(lon), float(lat)) for lon, lat in points])
        if len(points_all) > 1:
            points = max(points_all, key=lambda x: Polygon(x).area)  # 可视化确认是大的包小的, 直接取大的
        else:
            points = points_all[0]
        
        if isinstance(name, str):
            assert name
        else:
            assert np.isnan(name)
            name = None

        if is_elevator == "有电梯":
            is_elevator = True
        elif is_elevator == "无电梯":
            is_elevator = False
        elif is_elevator == "不确定":
            is_elevator = None
        else:
            assert np.isnan(is_elevator)
            is_elevator = None
        
        if isinstance(floor, str):
            if "不确定" in floor:
                floor = None
            else:
                floor = int(floor)
        else:
            assert np.isnan(floor)
            floor = None
        
        point = gcj2wgs(*(float(x) for x in point[7:-1].split(" ")))
        if not Point(point).covered_by(Polygon(points)):
            point = Polygon(points).representative_point().coords[:][0]
        
        buildings.append({
            "id": i,
            "points": points,
            "orig_id": bid,
            "name": name,
            "is_elevator": is_elevator,
            "floor": floor,
            "gate_gps": point,
            "gate_xy": projector(*point),
            "gate_id": str(i)
        })
    return buildings


def read_comm_data(path):
    data = pd.read_csv(path)
    communities = []
    i = 0
    for line in data.values:
        poly, cid, name = line[:3]
        if "EMPTY" in poly:
            continue
        if poly[:7] == "POLYGON":
            points = [x.split(" ") for x in poly[10:-2].split(",")]
        else:
            assert poly[:5] == "MULTI"
            points = [x.split(" ") for x in poly[16:-3].split(",")]
        if points[-1] != points[0]:
            points.append(points[0])
        points = [gcj2wgs(float(lon), float(lat)) for lon, lat in points]
        communities.append({
            "id": i,
            "orig_id": cid,
            "name": name,
            "gps": points,
            "xy": [projector(*p) for p in points]
        })
        i += 1
    return communities


def find_bd_in_which_comm(bd, communities):
    p = Point(bd["gate_xy"])
    for c in communities:
        poly = Polygon(c["xy"])
        if p.covered_by(poly):
            return c["id"]
    return None


def query_Baidu_building_name(gps):
    """
    调用百度周边搜索API查询楼名
    文档: https://lbsyun.baidu.com/index.php?title=webapi/guide/webservice-placeapi 圆形区域检索
    """
    user_key = "9vHb8Vf7LhlWeqh4vuFCafFB9f9dST5t"
    radius = 50  # 搜索半径
    url_base_nearby_search = f"https://api.map.baidu.com/place/v2/search?&query=住宅$公司$大厦$园区$商场&radius={radius}&output=json&ak={user_key}&scope=2&filter=sort_name:distance|sort_rule:1"
    lon, lat = wgs2bd(*gps)
    try:
        t = requests.get(url_base_nearby_search + f"&location={lat},{lon}").json()["results"]
        if t:
            info = t[0]  # results按distance排序, 取最近的作为该aoi的信息
            name = info["name"]
            print(name)
        else:
            name = "未知"
    except:
        name = "未知"
    return name


def add_extra_info(buildings, regions, comms):
    """添加一些字段和缺失信息"""
    regions = [r for r in regions if "总边界" not in r["name"]]
    rcnt, ccnt, query_cnt, not_known_name_cnt = 0, 0, 0, 0
    for b in buildings:
        if b["is_elevator"] is None:
            b["is_elevator"] = False  # 默认为楼梯楼
            b["default_elevator_tag"] = True
        else:
            b["default_elevator_tag"] = False

        b["region_id"] = find_bd_in_which_comm(b, regions)  # 计算所属路区
        if b["region_id"] is None:
            rcnt += 1
        b["comm_id"] = find_bd_in_which_comm(b, comms)  # 计算所属小区
        if b["comm_id"] is None:
            ccnt += 1

        if b["name"] is None:
            query_cnt += 1
            b["name"] = query_Baidu_building_name(b["gate_gps"])
            if b["name"] == "未知":
                not_known_name_cnt += 1
    print("building not in any region:", rcnt, "/", len(buildings))
    print("building not in any comm:", ccnt, "/", len(buildings))
    print("building not_known_name:", not_known_name_cnt, "/", query_cnt, "/", len(buildings))
    return buildings


if __name__ == "__main__":
    regions = pickle.load(open("data/regions.pkl", "rb"))

    buildings = read_building_data("orig_data/building.xlsx")
    print("buildings:", len(buildings))
    # plot_buildings(buildings, regions=regions)
    
    comms = read_comm_data("orig_data/community.csv")
    print("communities:", len(comms))
    pickle.dump(comms, open("data/communities.pkl", "wb"))
    # plot_buildings(buildings, regions=regions, comms=comms)

    buildings = add_extra_info(buildings, regions, comms)
    pickle.dump(buildings, open("data/buildings.pkl", "wb"))
    plot_buildings(buildings, regions=regions, comms=comms, path="figure/buildings.html")
