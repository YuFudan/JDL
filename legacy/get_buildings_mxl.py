import os
import json
import pickle
import folium
import requests
import matplotlib.pyplot as plt
from tqdm import tqdm
from pyproj import Proj
from collections import defaultdict
from shapely.geometry import Point, Polygon
from coord_convert.transform import gcj2wgs, wgs2gcj, wgs2bd


lon_cen, lat_cen = 116.5270, 39.8141
projector = Proj("+proj=utm +zone=50 +ellps=WGS84 +datum=WGS84 +units=m +no_defs")


def plot_buildings(buildings, regions=None, site=None, path="figure_mxl/test.html"):
    G_folium = folium.Map(
        location=[lat_cen, lon_cen],
        control_scale=True,
        tiles='http://webrd02.is.autonavi.com/appmaptile?lang=zh_cn&size=1&scale=1&style=8&x={x}&y={y}&z={z}',
        attr='高德底图',
        zoom_start=20,
    )
    G_folium.add_child(folium.LatLngPopup())
    for b in buildings:
        folium.PolyLine(
            locations=[wgs2gcj(*p)[::-1] for p in b["points"]],
            opacity=0.8,
            weight=2,
        ).add_to(G_folium)
        if "points_nogap" in b:
            folium.PolyLine(
                locations=[wgs2gcj(*p)[::-1] for p in b["points_nogap"]],
                opacity=0.3,
                weight=1,
            ).add_to(G_folium)
        if "gate_gps" in b:
            color = "orange" if b["default_elevator_tag"] else "black"
            if b["is_elevator"]:
                color = "red"
            folium.CircleMarker(
                location=wgs2gcj(*b["gate_gps"])[::-1],
                radius=2,
                color=color,
                popup=(b["name"], b["is_elevator"], b["floor"])
            ).add_to(G_folium)
    if regions:
        for r in regions:
            folium.PolyLine(
                locations=[wgs2gcj(*p)[::-1] for p in r["boundary"]],
                opacity=0.8,
                weight=3,
                color="red"
            ).add_to(G_folium)
    if site:
        folium.PolyLine(
            locations=[wgs2gcj(*p)[::-1] for p in site["points"]],
            opacity=0.8,
            weight=3,
            color="red"
        ).add_to(G_folium)

    G_folium.save(path)


def read_site():
    """漫香林所管辖的路区总片区围栏"""
    ps = json.load(open("orig_data_mxl/siteFence_geojson.json"))["features"][0]["geometry"]["coordinates"][0]
    site = {
        "name": "漫香林营业部",
        "points": [gcj2wgs(*p) for p in ps]
    }
    return site


def read_building_data(orig_data):
    buildings = []
    for d in orig_data:
        bid = d["properties"]["RoadNo"]
        geo = d["geometry"]
        assert geo["type"] == "Polygon"
        assert len(geo["coordinates"]) == 1
        ps = geo["coordinates"][0]
        buildings.append({
            "id": bid,
            "points": [gcj2wgs(*p) for p in ps],
            # "name":
            # "is_elevator":
            # "gate_gps":
            # "gate_xy":
            # "gate_id":
        })
    
    return buildings


def merge_nogap_info(buildings, buildings_nogap, id_mapping):
    """
    buildings是营业部内的每个细粒度aoi(楼)
    buildings_nogap则是每个楼占有一块更大的领地, 使得填充满整个空间(少数情况下一个领地内有多个楼)
    现直接根据id映射关系表, 将buildings_nogap的信息存到buildings里面
    """
    buildings_nogap = {b["id"]: b for b in buildings_nogap}
    id_mapping = {
        int(line.split(',')[0]): 
        line.split(',')[1].strip('\n')  # id中可能出现空格, 直接strip不行
        for line in id_mapping
    }
    for b in buildings:
        b["points_nogap"] = buildings_nogap[id_mapping[b["id"]]]["points"]
    return buildings


def read_poi_community_data(poi_data, community_data):
    """
    读poi和小区数据, 并通过把poi匹配到小区, 把小区数据中的楼层数和电梯楼信息融进poi中
    """
    # 读poi数据
    pois = []
    name_gps_set = set()  # 数据中存在name和gps都一样的poi, 进行去重
    uid = 0               # 数据中存在相同的id(且可能name和gps不一样), 自己重编id
    for poi in poi_data:
        prop = poi["properties"]
        name = prop["name"]
        assert name
        # pid = prop["id"]

        geo = poi["geometry"]
        assert geo["type"] == "MultiPoint"
        assert len(geo["coordinates"]) == 1
        gps = tuple(gcj2wgs(*geo["coordinates"][0]))
        # gps = tuple(geo["coordinates"][0])  # poi数据中的坐标已经是wgs
        name_gps = (name, gps)
        if name_gps in name_gps_set:
            continue
        xy = projector(*gps)

        xy = projector(*gps)
        pois.append({
            "id": uid,
            "name": name,
            "gps": gps,
            "xy": xy,
            "geo": Point(xy),
        })
        uid += 1
        name_gps_set.add(name_gps)
    assert len(pois) == len(set(p["id"] for p in pois))
    print("pois:", len(pois))

    # 读小区数据
    communities = []
    not_known_elevator_cnt = 0
    not_known_floor_cnt = 0
    for c in community_data:
        prop = c["properties"]
        cid = prop["id"]
        name = prop["aoiName"]
        is_elevator = prop["isElevator"]   # 注意小区的is_elevator和floor信息可能缺失
        if is_elevator is None:
            # print("not known is_elevator!", name)
            not_known_elevator_cnt += 1
        else:
            assert is_elevator in [0, 1]
            is_elevator = bool(is_elevator)
        assert is_elevator in [0, 1, None]
        floor = prop["floors"]
        if floor is None:
            not_known_floor_cnt += 1
        else:
            assert floor > 0

        geo = c["geometry"]
        assert geo["type"] == "Polygon"
        if len(geo["coordinates"]) > 1:  # 有多个多边形时, 直接取最大的那个(画图看过, 是大的包小的, 小的特别小, 应该没啥意义)
            polys = []
            for ps in geo["coordinates"]:
                gpss = [gcj2wgs(*p) for p in ps]
                xys = [projector(*p) for p in gpss]
                polys.append(Polygon(xys))
            polys.sort(key=lambda x:-x.area)
            for poly in polys[1:]:
                assert poly.within(polys[0])
            poly:Polygon = polys[0]
            xys = poly.exterior.coords[:]
            gps = [projector(*p, inverse=True) for p in xys]
        else:
            gpss = [gcj2wgs(*p) for p in geo["coordinates"][0]]
            xys = [projector(*p) for p in gpss]
            poly = Polygon(xys)
        
        xys = [projector(*p) for p in gpss]
        communities.append({
            "id": cid,
            "name": name,
            "is_elevator": is_elevator,  # 可能为None
            "floor": floor,  # 可能为None
            "gps": gpss,
            "xy": xys,
            "geo": poly,
        })

    # 处理id相同的问题
    cid2cs = defaultdict(list)
    for c in communities:
        cid2cs[c["id"]].append(c)
    same_cids = set()
    for cid, cs in cid2cs.items():
        if len(cs) > 1:
            assert len(set(c["name"] for c in cs)) == len(cs)  # 其实是不同的小区: 金色漫香林一期 和 金色漫香林二期 的id相同
            same_cids.add(cid)
    uid = max(c["id"] for c in communities) + 1
    for c in communities:
        if c["id"] in same_cids:
            c["id"] = uid
            uid += 1
    assert len(communities) == len(set(c["id"] for c in communities))

    print("communities:", len(communities))
    print("comm not_known_elevator_cnt:", not_known_elevator_cnt)
    print("comm not_known_floor_cnt:", not_known_floor_cnt)

    # 计算poi所属的小区
    matched_pids = set()
    for c in tqdm(communities):
        poly = c["geo"]
        for p in pois:
            if p["id"] not in matched_pids and p["geo"].within(poly):
                p["community_id"] = c["id"]
                p["is_elevator"] = c["is_elevator"]
                p["floor"] = c["floor"]
                matched_pids.add(p["id"])
    print("matched pois:", len(matched_pids))
    not_known_elevator_cnt = 0
    not_known_floor_cnt = 0
    for p in pois:
        if p["id"] not in matched_pids:
            p["community_id"], p["is_elevator"], p["floor"] = None, None, None
            not_known_elevator_cnt += 1
            not_known_floor_cnt += 1
        else:
            if p["is_elevator"] is None:
                not_known_elevator_cnt += 1
            if p["floor"] is None:
                not_known_floor_cnt += 1
    print("poi not_known_elevator_cnt:", not_known_elevator_cnt)
    print("poi not_known_floor_cnt:", not_known_floor_cnt)

    # 画图
    G_folium = folium.Map(
        location=[lat_cen, lon_cen],
        control_scale=True,
        tiles='http://webrd02.is.autonavi.com/appmaptile?lang=zh_cn&size=1&scale=1&style=8&x={x}&y={y}&z={z}',
        attr='高德底图',
        zoom_start=20,
    )
    G_folium.add_child(folium.LatLngPopup())
    for c in communities:
        if c["is_elevator"] is None:
            color = "orange"
        else:
            color = "blue"
        folium.PolyLine(
            locations=[wgs2gcj(*p)[::-1] for p in c["gps"]],
            opacity=0.8,
            weight=3,
            color=color,
            popup=(c["id"], c["name"], c["is_elevator"], c["floor"]),
        ).add_to(G_folium)
    for poi in pois:
        if poi["community_id"] is None:
            color = "red"
        else:
            if poi["is_elevator"] is None:
                color = "orange"
            else:
                color = "cyan"
        folium.CircleMarker(
            location=wgs2gcj(*poi["gps"])[::-1],
            opacity=0.8,
            radius=3,
            color=color,
            popup=(poi["name"], poi["community_id"], poi["is_elevator"], poi["floor"]),
        ).add_to(G_folium)
    G_folium.save("figure_mxl/poi_community.html")

    return pois, communities


def get_building_semantics(buildings, pois, communities):
    """
    楼名, 楼层数, 有无电梯
    """
    def query_Baidu_building_name(gps):
        """调用百度周边搜索API查询楼名"""
        user_key = "9vHb8Vf7LhlWeqh4vuFCafFB9f9dST5t"
        radius = 50  # 搜索半径
        url_base_nearby_search = f"https://api.map.baidu.com/place/v2/search?&query=住宅&radius={radius}&output=json&ak={user_key}&scope=2&filter=sort_name:distance|sort_rule:1"
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
        
    communities = [c for c in communities if c["is_elevator"] is not None]  # 只对有is_elevor信息的comm感兴趣
    not_matched_bids = set()
    for b in tqdm(buildings):
        gpss = b["points"]
        xys = [projector(*p) for p in gpss]
        poly = Polygon(xys)
        b["geo"] = poly
        for poi in pois:
            if poi["geo"].within(poly):
                b["poi_id"] = poi["id"]
                b["name"] = poi["name"]
                b["is_elevator"] = poi["is_elevator"]
                b["floor"] = poi["floor"]
                break
        else:
            not_matched_bids.add(b["id"])

    buildings = {b["id"]: b for b in buildings}
    for bid in tqdm(not_matched_bids):
        b = buildings[bid]
        poly:Polygon = b["geo"]
        for c in communities:
            if poly.intersects(c['geo']):
                b["community_id"] = c["id"]
                b["name"] = None
                b["is_elevator"] = c["is_elevator"]
                b["floor"] = c["floor"]
                break
        else:
            b["name"], b["is_elevator"], b["floor"] = None, None, None

    not_known_name_cnt = 0
    not_known_elevator_cnt = 0
    not_known_floor_cnt = 0
    buildings = list(buildings.values())
    for b in tqdm(buildings):
        if b["name"] is None:
            centroid_xy = b["geo"].centroid.coords[:][0]
            b["name"] = query_Baidu_building_name(projector(*centroid_xy, inverse=True))
            if b["name"] == "未知":
                not_known_name_cnt += 1
            if b["is_elevator"] is None:
                not_known_elevator_cnt += 1
            if b["floor"] is None:
                not_known_floor_cnt += 1  
    print("building not_known_name_cnt:", not_known_name_cnt)
    print("building not_known_elevator_cnt:", not_known_elevator_cnt)
    print("building not_known_floor_cnt:", not_known_floor_cnt)

    return buildings


def add_default_info(buildings, regions):
    """
    用默认值填充缺失的信息, 并补充gate_gps, gate_xy, region字段
    """
    for r in regions:
        r["poly"] = Polygon(r["boundary"])

    for b in buildings:
        if b["is_elevator"] is None:
            b["is_elevator"] = False  # 默认为楼梯楼
            b["default_elevator_tag"] = True
        else:
            b["default_elevator_tag"] = False

        if b["floor"] is None:
            b["floor"] = 7            # 默认7层
            b["default_floor_tag"] = True
        else:
            b["default_floor_tag"] = False
        
        poly: Polygon = b["geo"]
        gate = poly.centroid
        if not gate.within(poly):
            gate = poly.representative_point()
        gate_xy = gate.coords[:][0]
        gate_gps = projector(*gate_xy, inverse=True)
        b["gate_gps"] = gate_gps
        b["gate_xy"] = gate_xy
        b["gate_id"] = str(b["id"])

        # 找出楼所在的路区
        p = Point(gate_gps)  
        dis_ids = []
        for r in regions:
            dis = p.distance(r["poly"])
            if dis == 0:
                b["region"] = r["id"]
                break
            dis_ids.append((dis, r["id"]))
        else:
            b["region"] = min(dis_ids, key=lambda x:x[0])[1]

        del b["geo"]
    
    return buildings
    

if __name__ == "__main__":
    # # 生成 楼aoi 和 楼点位 geojson
    # buildings = pickle.load(open("data_mxl/buildings.pkl", "rb"))
    # bd_geojson = [
    #     {
    #         "type": "Feature",
    #         "properties": {k: bd[k] for k in ["id", "name", "is_elevator", "floor"] if k in bd},
    #         "geometry": {"type": "Polygon", "coordinates": [bd["points"]]},
    #     } for bd in buildings
    # ]
    # json.dump(bd_geojson, open("data_mxl/buildings_geojson.json", "w"))
    # node_geojson = [
    #     {
    #         "type": "Feature",
    #         "properties": {"id": bd["id"], "name": bd["name"]},
    #         "geometry": {"type": "Point", "coordinates": bd["gate_gps"]},
    #     } for bd in buildings
    # ]
    # json.dump(node_geojson, open("data_mxl/nodes_geojson.json", "w"))
    # exit()

    # 读有留白aoi
    orig_data = json.load(open("orig_data_mxl/building_geojson.json"))["features"]
    buildings = read_building_data(orig_data)
    print("buildings:", len(buildings))

    # 读无留白aoi
    orig_data_nogap = json.load(open("orig_data_mxl/noGapBuilding_geojson.json"))["features"]
    buildings_nogap = read_building_data(orig_data_nogap)
    print("buildings_nogap:", len(buildings_nogap))
    
    regions = pickle.load(open("data_mxl/regions.pkl", "rb"))
    site = read_site()

    # plot_buildings(
    #     buildings=buildings,
    #     regions=regions, 
    #     site=site, 
    #     path="figure_mxl/buildings.html"
    # )

    # plot_buildings(
    #     buildings=buildings_nogap,
    #     regions=regions, 
    #     site=site, 
    #     path="figure_mxl/buildings_nogap.html"
    # )

    # 将无留白的信息直接加到有留白的里面
    id_mapping = open("orig_data_mxl/manxianglin_idmapping.csv").readlines()
    assert len(id_mapping) == len(buildings)
    buildings_with_nogapinfo = merge_nogap_info(buildings, buildings_nogap, id_mapping)
    assert len(buildings_with_nogapinfo) == len(set(b["id"] for b in buildings_with_nogapinfo))

    # 读poi和小区, 大部分小区的is_elevator和floor信息缺失, 因此poi的is_elevator和floor信息缺失也很多
    path = "data_mxl/poi_community.pkl"
    if os.path.exists(path):
        pois, communities = pickle.load(open(path, "rb"))
    else:
        poi_data = json.load(open("orig_data_mxl/POI_geojson.json"))["features"]
        community_data = json.load(open("orig_data_mxl/xiaoqu_geojson.json"))["features"]
        pois, communities = read_poi_community_data(poi_data, community_data)
        pickle.dump((pois, communities), open(path, "wb"))

    # 添加楼的语义信息(楼名, 楼层数, 有无电梯)
    buildings_fullinfo = get_building_semantics(buildings_with_nogapinfo, pois, communities)
    
    # 填充默认信息
    buildings_final = add_default_info(buildings_fullinfo, regions)
    pickle.dump(buildings_final, open("data_mxl/buildings.pkl", "wb"))
    json.dump(buildings_final, open("data_mxl/buildings.json", "w"))

    plot_buildings(
        buildings=buildings,
        regions=regions, 
        site=site, 
        path="figure_mxl/buildings_final.html"
    )
