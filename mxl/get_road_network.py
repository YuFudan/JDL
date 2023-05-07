import json
import pickle
from collections import defaultdict
from math import ceil
from pprint import pprint

import folium
import networkx as nx
from coord_convert.transform import gcj2wgs, wgs2gcj
from networkx import DiGraph
from shapely.geometry import GeometryCollection, LineString, MultiPoint, Point
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
from constants_spatial import *

global lines_matcher
workers = 32
SQRT2 = 2 ** 0.5


def plot_road_network(roads, nodes, buildings=None, path="figure/test.html"):
    colors = get_colors()
    G_folium = get_base_map()
    for nid, n in tqdm(nodes.items()):
        folium.CircleMarker(
            location=wgs2gcj(*n["point"])[::-1],
            radius=3,
            color="black",
            popup=nid,
            opacity=0.5,
        ).add_to(G_folium)
    cnt = 0
    for rid, r in tqdm(roads.items()):
        folium.PolyLine(
            locations=[wgs2gcj(*p)[::-1] for p in r["points"]],
            # color=colors[cnt % len(colors)],
            color="black",
            popup=r["od"],
            weight=2,
            opacity=0.5
        ).add_to(G_folium)
        cnt += 1
    if buildings is not None:
        for b in tqdm(buildings):
            color = "orange" if b["default_elevator_tag"] else "blue"
            if b["is_elevator"]:
                color = "red"
            folium.PolyLine(
                locations=[wgs2gcj(*p)[::-1] for p in b["points"]],
                opacity=0.5,
                color=color,
                weight=2,
                popup=(b["name"], b["is_elevator"], b["floor"])
            ).add_to(G_folium)
    G_folium.save(path)


def calc_intersection_unit(arg):
    """
    计算两条linestring间的交点, 若交点不是二者原有的拐点, 则添加
    """
    r1, r2 = arg
    l1, l2 = r1["geo"], r2["geo"]
    x = l1.intersection(l2)
    if isinstance(x, LineString):
        return False
    if isinstance(x, Point):
        ps = [x]
    elif isinstance(x, MultiPoint):
        ps = [*x.geoms]
    elif isinstance(x, GeometryCollection):
        ps = []
        for y in x.geoms:
            if isinstance(y, Point):
                ps.append(y)
            elif isinstance(y, MultiPoint):
                ps += [*y.geoms]
            else:
                assert False, type(y)
    else:
        assert False, type(x)
    sp1, sp2 = [], []
    for p in ps:
        assert isinstance(p, Point), type(p)
        s1 = l1.project(p)
        sp1.append([s1, p.coords[:][0]])
        s2 = l2.project(p)
        sp2.append([s2, p.coords[:][0]])
    return [[r["id"], sp] for sp, r in [(sp1, r1), (sp2, r2)] if sp]


def read_road_network(orig_data):
    """
    经检查发现, 虽然只给了road的geometry, 而没有给出node
    但road的端点, 被其它road经过时, 也严格就是同一位置, 故可以直接将所有端点取出作为node
    问题: 有时两条路相交之处, 并不是任何路的端点(即二者都没有在交点处分段)
    考虑直接将所有路直接打碎成直线, 只要保证所有逻辑上是同一位置的点, 实际上坐标也完全一致, 则可以构建拓扑
    """
    print("input data:", len(orig_data))

    # 计算所有原始Linestring间的交点, 并在交点处增加linestring的拐点, 确保任何交点处都是linestring的拐点
    pss = [[gcj2wgs(*p) for p in ps] for r in orig_data for ps in r["geometry"]["coordinates"]]
    roads = {i: {
        "id": i, 
        "gps": ps, 
        "xy": [projector(*p) for p in ps]
        } for i, ps in enumerate(pss)}
    for r in roads.values():
        r["geo"] = LineString(r["xy"])
        r["ss"] = [r["geo"].project(Point(p)) for p in r["xy"]]
    print(len(roads))
    pairs = []
    for i, r in enumerate(roads.values()):
        for j in range(i+1, len(roads)):
            pairs.append([r, roads[j]])
    
    results = process_map(
        calc_intersection_unit, 
        pairs, 
        chunksize=min(ceil(len(pairs) / workers), 10000), 
        max_workers=workers)
    results = [r for r in results if r]
    rid2s2p = defaultdict(dict)
    for r in results:
        for rid, sps in r:
            for s, p in sps:
                if s in rid2s2p[rid]:
                    assert p == rid2s2p[rid][s]
                rid2s2p[rid][s] = p
    for rid, s2p in tqdm(rid2s2p.items()):
        r = roads[rid]
        sps = list(zip(r["ss"], r["gps"])) + \
            [(s, projector(*p, inverse=True)) for s, p in s2p.items()]
        ps = [x[1] for x in sorted(sps, key=lambda x: x[0])]
        ps_new = []
        last_p = None
        for lon, lat in ps:
            p = (round(lon, 6), round(lat, 6))
            if p != last_p:
                ps_new.append(p)
                last_p = p
        r["gps"] = ps_new

    p2neighbors = defaultdict(set)  # 记录每个点的邻居点
    cnt = 0
    for r in roads.values():
        ps = r["gps"]
        for p1, p2 in zip(ps, ps[1:]):
            p2neighbors[p1].add(p2)
            p2neighbors[p2].add(p1)
            cnt += 1

    # p2neighbors = defaultdict(set)  # 记录每个点的邻居点
    # cnt = 0
    # for r in orig_data:
    #     for ps in r["geometry"]["coordinates"]:
    #         ps = [gcj2wgs(*p) for p in ps]
    #         ps = [(round(lon, 6), round(lat, 6)) for lon, lat in ps]  # 统一保留6位数, 确保坐标一致
    #         for p1, p2 in zip(ps, ps[1:]):
    #             p2neighbors[p1].add(p2)
    #             p2neighbors[p2].add(p1)
    #             cnt += 1

    print("segments:", cnt)
    print("all nodes:", len(p2neighbors))
    ps_end = set(p for p, nps in p2neighbors.items() if len(nps) != 2)  # 邻居点数量不为2的点作为道路端点, 否则为内部点
    print("end nodes:", len(ps_end))

    def iterative_walk(p1, p2):
        """
        沿着从p1到p2的方向, 继续往后走(p2为内部点), 直到p2成为端点
        """
        nonlocal p2neighbors, ps_end
        path = []
        if p2 in ps_end:
            return path
        else:
            for p in p2neighbors[p2]:
                if p != p1:
                    path.append(p)
                    path += iterative_walk(p2, p)
                    break
        return path

    # 从所有端点开始游走, 直到遇到另一个端点, 此时生成一条路
    # 最终结果应为每条路的正反向都会被生成
    rid, nid = 0, 0
    roads = {}
    nodes = {}
    p2nid = {}
    for p1 in tqdm(ps_end):
        nps = p2neighbors[p1]
        for p2 in nps:
            path = [p1, p2] + iterative_walk(p1, p2)
            od = []
            for p in [path[0], path[-1]]:
                if p not in p2nid:
                    nodes[nid] = {
                        "id": nid,
                        "point": p
                    }
                    od.append(nid)
                    p2nid[p] = nid
                    nid += 1
                else:
                    od.append(p2nid[p])
            od = tuple(od)
            roads[rid] = {
                "id": rid,  # 不能直接用od作为id, 同一od间可能有多条路
                "od": od,
                "points": path,
            }
            rid += 1
    ods = set(r["od"] for r in roads.values())
    for o, d in ods:
        assert (d, o) in ods
    print("roads:", len(roads))
    print("nodes:", len(nodes))

    # 双向路中只保留1条
    od2roads = defaultdict(list)
    for r in roads.values():
        od2roads[r["od"]].append(r)
    rids_reserve = []
    processed_ods = set()
    for od, rs in od2roads.items():
        if od not in processed_ods:
            do = (od[1], od[0])
            assert len(od2roads[do]) == len(rs)
            if od[0] != od[1]:
                rids_reserve += [r["id"] for r in rs]
            else:
                assert len(rs) == 2  # od相同(自环)
                rids_reserve.append(rs[0]["id"])
            processed_ods.add(od)
            processed_ods.add(do)
    rids_reserve = set(rids_reserve)
    roads = {rid: r for rid, r in roads.items() if rid in rids_reserve}
    print("single roads:", len(roads))

    return roads, nodes


def match_unit(gate):
    global lines_matcher
    nid = gate["id"]
    geo = gate["geo"]
    x, y = gate["point"]
    dis_gate = 20
    while True:
        candidates = []
        double_dis_gate = dis_gate * 2  # 若粗过滤阈值太小导致没有候选, 则翻倍重试
        for line in lines_matcher:
            x2, y2 = line["point"]
            # if (abs(x - x2) + abs(y - y2)) / SQRT2 < lane['length'] / 2 + dis_gate:
            tmp = SQRT2 * (abs(x - x2) + abs(y - y2)) - line["length"]  # 先根据1范数估计距离的上限
            if tmp < double_dis_gate:
                s = line["geo"].project(geo, normalized=True)
                assert 0 <= s <= 1
                point2 = line["geo"].interpolate(s, normalized=True)
                distance = geo.distance(point2)
                candidates.append([distance, line["id"], s, point2])
        if candidates:
            break
        else:
            dis_gate *= 2
    assert candidates

    rid, s, point2 = min(candidates, key=lambda x:x[0])[1:]
    lon, lat = projector(*point2.coords[:][0], inverse=True)  # projected_gps    
    return (
        nid, 
        (rid, s),  # rid, s
        (round(lon, 6), round(lat, 6))  # projected_gps
    )


def match_building_to_road(roads, nodes, buildings):
    """
    将楼gate匹配到最近的road, 在匹配点处新建node并将road分割开
    """
    global lines_matcher

    # 将gate匹配到road
    lines_matcher = [
        (LineString([projector(*p) for p in r["points"]]), rid)
        for rid, r in roads.items()
    ]
    lines_matcher = [{
        "id": rid,
        "geo": geo,
        "point": geo.interpolate(0.5, normalized=True).coords[:][0],  # 折线的中点
        "length": geo.length,
        } for geo, rid in lines_matcher
    ]
    gates_matcher = [{
        "id": str(b["id"]),  # 使用str(bid)作为gate_id
        "geo": Point(b["gate_xy"]),
        "point": b["gate_xy"]
        } for b in buildings
    ]
    print("gates:", len(gates_matcher))

    results = process_map(
        match_unit, 
        gates_matcher, 
        chunksize=min(ceil(len(gates_matcher) / workers), 100), 
        max_workers=workers)

    # 添加匹配点作为路网节点;
    # 记录每条路上的匹配点;
    # 添加从gate到匹配点的road;
    # 添加gate作为路网节点
    rid2s_nid = defaultdict(list)
    gps2nid = {}
    node_uid = max(nodes.keys()) + 1
    road_uid = max(roads.keys()) + 1
    gnid2gps = {str(b["id"]): b["gate_gps"] for b in buildings}
    match_to_end_cnt = 0
    match_to_inter_cnt = 0
    new_node_cnt = 0
    new_gate_road_cnt = 0
    new_gate_node_cnt = 0
    for gate_nid, (rid, s), gps in tqdm(results):
        gate_gps = gnid2gps[gate_nid]
        if s == 0:                  # 当匹配到路的端点时, 不要新建点
            nid = roads[rid]["od"][0]
            gps = roads[rid]["points"][0]
            match_to_end_cnt += 1
        elif s == 1:
            nid = roads[rid]["od"][1]
            gps = roads[rid]["points"][-1]
            match_to_end_cnt += 1
        else:
            match_to_inter_cnt += 1
            if gps not in gps2nid:  # 添加匹配点作为路网节点: 不同gate可能匹配到相同点, 确保只建立1个新点
                nodes[node_uid] = {
                    "id": node_uid,
                    "point": gps
                }
                gps2nid[gps] = node_uid
                nid = node_uid
                rid2s_nid[rid].append([s, nid])  # 记录每条路上的匹配点
                node_uid += 1
                new_node_cnt += 1
            else:
                nid = gps2nid[gps]

        assert gps == nodes[nid]["point"]

        roads[road_uid] = {  # 添加从gate到匹配点的road
            "id": road_uid,
            "od": (gate_nid, nid),
            "points": [gate_gps, gps],
        }
        road_uid += 1
        new_gate_road_cnt += 1
        nodes[gate_nid] = {  # 添加gate作为路网节点
            "id": gate_nid,
            "point": gate_gps,
            "building": int(gate_nid)
        }
        new_gate_node_cnt += 1
    print("match_to_end_cnt:", match_to_end_cnt)
    print("match_to_inter_cnt:", match_to_inter_cnt)
    print("new_node_cnt:", new_node_cnt)
    print("new_gate_node_cnt:", new_gate_node_cnt)
    print("new_gate_road_cnt:", new_gate_road_cnt)

    # 在匹配点处拆分路
    road_uid = max(roads.keys()) + 1
    new_road_cnt = 0
    for rid, s_nids in tqdm(rid2s_nid.items()):
        # 计算road上原本的中间点的s
        r =  roads[rid]
        gpss = r["points"]
        xys = [projector(*p) for p in gpss]
        line = LineString(xys)
        s_gpss = []
        for gps, xy in zip(gpss[1:-1], xys[1:-1]):
            s_gpss.append((line.project(Point(xy), normalized=True), gps))
        # 和匹配点放在一起, 排序
        s_nids_gpss = s_nids + s_gpss
        s_nids_gpss.sort(key=lambda x:x[0])
        o, path = r["od"][0], [gpss[0]]
        for _, nid_or_gps in s_nids_gpss:
            if isinstance(nid_or_gps, tuple):  # gps
                path.append(nid_or_gps)
            else:
                path.append(nodes[nid_or_gps]["point"])
                roads[road_uid] = {  # 添加拆分的一段路
                    "id": road_uid,
                    "od": (o, nid_or_gps),
                    "points": path
                }
                o = nid_or_gps
                path = [nodes[nid_or_gps]["point"]]
                road_uid += 1
                new_road_cnt += 1
        roads[road_uid] = {
            "id": road_uid,
            "od": (o, r["od"][1]),
            "points": path + [gpss[-1]]
        }
        road_uid += 1
        new_road_cnt += 1
    print("new_road_cnt:", new_road_cnt)

    # 删去拆分前的路
    print("delete road cnt:", len(rid2s_nid))
    for rid in rid2s_nid.keys():
        del roads[rid]
    
    # 删除自环
    rids_ring = [
        rid for rid, r in roads.items()
        if r["od"][0] == r["od"][1]
    ]
    print("delete ring road cnt:", len(rids_ring))
    for rid in rids_ring:
        del roads[rid]
    
    # 多重边取最短
    for r in roads.values():
        r["length"] = LineString([projector(*p) for p in r["points"]]).length
    od2roads = defaultdict(list)
    for rid, r in roads.items():
        od2roads[r["od"]].append(r)
    rids_long = []
    for rs in od2roads.values():
        if len(rs) > 1:
            rs.sort(key=lambda x:x["length"])
            rids_long += [r["id"] for r in rs[1:]]
    print("delete multiple road cnt:", len(rids_long))
    for rid in rids_long:
        del roads[rid]

    # 检查合法性
    od_nids = set()
    for r in roads.values():
        o, d = r["od"]
        od_nids.add(o)
        od_nids.add(d)
        assert nodes[o]["point"] == r["points"][0]
        assert nodes[d]["point"] == r["points"][-1]
    assert len(od_nids) == len(nodes)  

    return roads, nodes


def special_process(roads, nodes, station_gps):
    """
    检查路网连通性, 发现由于道路相交处, 某条道路在此并没有转折点, 因此没连上, 手动将其连上
    并同时将快递站连到路网上
    """
    for r in roads.values():
        del r["id"]
    tmp = len(roads)
    roads = {r["od"]: r for r in roads.values()}
    assert len(roads) == tmp

    if False:
        cuts = [
            [(1655, 581), 244],
            [(1642, 1074), 243],

            [(848, 299), 1444],
            [(1666, 848), 1443],

            [(649, 648), 926],

            [(1934, 1908), 1287],

            [(258, 257), 1551],
            [(1318, 1169), 1550],

            [(2050, 986), 892],
            [(1972, 799), 713],
            [(1889, 988), 1520, 1356, 1529],
            [(2046, 1962), 237],
            [(1986, 1108), 613],

            [(2071, 374), 1483, 1588],

            [(840, 839), 1488],

            [(1805, 572), 211],

            [(2059, 367), 599],
            [(2144, 2128), 600],

            [(660, 225), 830],
            [(676, 662), 829],
            [(674, 672), 634],
        ]

        connects = [
            (1288, 1423),
            (1288, 1546),
            (1794, 316)
        ]
        for n1, n2 in connects:
            roads[(n1, n2)] = {
                "od": (n1, n2),
                "points": [nodes[n1]["point"], nodes[n2]["point"]]
            }
        for cut in cuts:
            od = cut[0]
            if od not in roads:
                od = (od[1], od[0])
            r = roads[od]
            gpss = r["points"]

            # 计算road上原本的中间点的s
            line = LineString(gpss)
            s_gpss = []
            for gps in gpss[1:-1]:
                s_gpss.append((line.project(Point(gps), normalized=True), gps))
            # 计算插入点的s
            s_nids = []
            for nid in cut[1:]:
                gps = nodes[nid]["point"]
                s_gpss.append((line.project(Point(gps), normalized=True), nid))
            # 放在一起, 排序
            s_nids_gpss = s_nids + s_gpss
            s_nids_gpss.sort(key=lambda x:x[0])
            o, path = r["od"][0], [gpss[0]]
            for _, nid_or_gps in s_nids_gpss:
                if isinstance(nid_or_gps, tuple):  # gps
                    path.append(nid_or_gps)
                else:
                    path.append(nodes[nid_or_gps]["point"])
                    roads[(o, nid_or_gps)] = {  # 添加拆分的一段路
                        "od": (o, nid_or_gps),
                        "points": path
                    }
                    o = nid_or_gps
                    path = [nodes[nid_or_gps]["point"]]
            roads[(o, r["od"][1])] = {
                "od": (o, r["od"][1]),
                "points": path + [gpss[-1]]
            }
        for cut in cuts:  # 删除拆分前的路
            od = cut[0]
            if od not in roads:
                od = (od[1], od[0])
            del roads[od]
        
    # 将快递站连到路网
    nodes["快递站"] = {
        "id": "快递站",
        "point": station_gps
    }
    # roads[("快递站", 2019)] = {
    #     "od": ("快递站", 2019),
    #     "points": [station_gps, nodes[2019]["point"]]
    # }
    roads[("快递站", 1865)] = {
        "od": ("快递站", 1865),
        "points": [station_gps, nodes[1865]["point"]]
    }

    # 检查合法性
    od_nids = set()
    for r in roads.values():
        o, d = r["od"]
        od_nids.add(o)
        od_nids.add(d)
        assert nodes[o]["point"] == r["points"][0]
        assert nodes[d]["point"] == r["points"][-1]
    assert len(od_nids) == len(nodes)  

    return roads, nodes


def plot_G(G, path="figure/test.html"):
    colors = get_colors()
    G_folium = get_base_map()
    for u, v, info in G.edges(data=True):
        if info.get("cpnt", 0) == 0:
            color = "black"
        else:
            color = colors[info["cpnt"] % len(colors)]
        folium.PolyLine(
            locations=[wgs2gcj(*p)[::-1] for p in info["gps"]],
            color=color,
            weight=3,
            opacity=0.5,
            popup=info["od"],
        ).add_to(G_folium)
    for n, info in G.nodes(data=True):
        if info.get("cpnt", 0) == 0:
            color = "black"
        else:
            color = colors[info.get("cpnt", 0) % len(colors)]
        folium.CircleMarker(
            location=wgs2gcj(*info["gps"])[::-1],
            color=color,
            radius=3,
            opacity=0.5,
            popup=n,
        ).add_to(G_folium)

    G_folium.save(path)


def construct_graph(roads, nodes):
    """
    添加反向路, 构造拓扑图
    """
    G_nodes = [
        (nid, { 
            "id": nid,
            "gps": info["point"],
            "xy": projector(*info["point"]),
            "building": info.get("building", None),
        }) for nid, info in nodes.items()
    ]
    for n in G_nodes:
        if n[1]["building"] is None:
            del n[1]["building"]
    
    roads = list(roads.values())
    reverse_roads = [
        {
            "od": (road["od"][1], road["od"][0]),  # 由于已经去掉了重边, 直接用od即可作为id
            "points": road["points"][::-1]
        } for road in roads
    ]
    roads = roads + reverse_roads

    G_edges = [
        (
            road["od"][0],
            road["od"][1],
            {
                "od": road["od"],
                "gps": road["points"],
                "xy": [projector(*p) for p in road["points"]],
            }
        ) for road in roads
    ]
    for e in G_edges:
        e[2]["length"] = LineString(e[2]["xy"]).length
    G = DiGraph()
    G.add_nodes_from(G_nodes)
    G.add_edges_from(G_edges)

    # 处理路网不连通的问题
    cpnts = list(nx.connected_components(G.to_undirected()))
    cpnts_throw = []  # 先直接丢弃不含楼的连通分量
    for cpnt in cpnts:
        for nid in cpnt:
            if isinstance(nid, str):
                break
        else:
            cpnts_throw.append(cpnt)
    for cpnt in cpnts_throw:
        for nid in cpnt:
            G.remove_node(nid)

    # 通过以下画图发现, 是由于道路相交处, 某条道路在此并没有转折点
    cpnts = list(nx.connected_components(G.to_undirected()))
    assert len(cpnts) == 1
    if False:
        print([len(x) for x in cpnts])
        for nid, info in G.nodes(data=True):
            for i, cpnt in enumerate(cpnts):
                if nid in cpnt:
                    info["cpnt"] = i
                    break
            else:
                assert False
        for u, v, info in G.edges(data=True):
            assert G.nodes[u]["cpnt"] == G.nodes[v]["cpnt"]
            info["cpnt"] = G.nodes[u]["cpnt"]
        plot_G(G, "figure/G.html")

    return G


if __name__ == "__main__":
    # orig_data = json.load(open("orig_data/road_geojson.json"))["features"]
    # roads, nodes = read_road_network(orig_data)
    # pickle.dump((roads, nodes), open("data/roads_nodes.pkl", "wb"))

    roads, nodes = pickle.load(open("data/roads_nodes.pkl", "rb"))

    buildings = pickle.load(open("data/buildings.pkl", "rb"))

    # plot_road_network(roads, nodes, buildings, "figure/road_network.html")

    roads, nodes = match_building_to_road(roads, nodes, buildings)

    station_gps = [LON_STA, LAT_STA]
    roads, nodes = special_process(roads, nodes, station_gps)
    print("final roads:", len(roads))
    print("final nodes:", len(nodes))

    # plot_road_network(roads, nodes, buildings, "figure/road_network.html")

    G = construct_graph(roads, nodes)
    plot_G(G, "figure/G.html")

    # 检查每个building的gate确实都连到了路网上
    nid_set = set(x for x in G.nodes)
    for bd in buildings:
        assert str(bd["id"]) in nid_set

    pickle.dump(G, open("data/G.pkl", "wb"))
