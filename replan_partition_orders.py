"""
生成路区重划分下的订单分配
    不直接使用京东提供的订单分配, 因为双屏对比时, 想看到相同的小哥, 最好每个小哥还在原来的路区附近
    而京东的分配, 是依据营业部平日常驻的快递员来的, 没有考虑当天外包和缺勤、病假人员, 所以人员对不上
这里, 依据提供的重划分路区边界, 共25个路区, 首先人工将若干单量较少且邻近的路区合并到23个, 以符合当天23人
然后按照尽量保证小哥还在原来路区附近的原则, 将小哥分配到重划分的路区
"""
import pickle
import folium
from eviltransform import wgs2gcj
from pprint import pprint
from constants_mw import *
from collections import Counter, defaultdict


def plot_regions_with_few_odrs(regions, counter):
    """按照路区中订单的多少, 按不同颜色画出路区"""
    G_folium = folium.Map(
        # location=[39.9180, 116.4712],
        location=[39.958759, 116.426223],
        control_scale=True,
        tiles='http://webrd02.is.autonavi.com/appmaptile?lang=zh_cn&size=1&scale=1&style=8&x={x}&y={y}&z={z}',
        attr='高德底图',
        zoom_start=20,
    )
    G_folium.add_child(folium.LatLngPopup())
    for r in regions:
        points = [wgs2gcj(*p[::-1])[::-1] for p in r["boundary"]]
        # centroid = Polygon(points).centroid.coords[:][0]
        centroid = Polygon(points).representative_point().coords[:][0]
        num = counter[r["id"]]
        if num < 100:
            color = "red"
        elif num < 150:
            color = "orange"
        elif num < 200:
            color = "lime"
        else:
            color = "royalblue"
        folium.CircleMarker(
            location=centroid[::-1],
            opacity=0.8,
            radius=3,
            fill=True,
            popup=(r["id"], num),
            color=color
        ).add_to(G_folium)
        folium.PolyLine(
            locations=[p[::-1] for p in points],
            opacity=0.8,
            weight=3,
            popup=(r["id"], num),
            color=color
        ).add_to(G_folium)
    G_folium.save("figure/regions_with_few_odrs.html")


def partition_orders(orders, merge_rids):
    """将订单分成23份"""
    rid2odrs = defaultdict(list)
    for o in orders:
        rid2odrs[buildings[o["building_id"]]["region_partition"]].append(o)
    result = {}
    for r1, r2 in merge_rids:
        result[(r1, r2)] = rid2odrs[r1] + rid2odrs[r2]
    rid_set = set(y for x in merge_rids for y in x)
    for rid, odrs in rid2odrs.items():
        if rid not in rid_set:
            result[rid] = odrs
    # print(len(result))
    # print([len(v) for v in result.values()])

    return result


def odrs2courier(rid2odrs, cid2odrs):
    """
    将23份订单分给23个小哥, 尽量使小哥还在原来路区附近
    rid2odrs: 重划分路区中的订单
    cid2odrs: 原始划分下每个小哥的订单
    """
    match_pairs = []
    # step1 检查路区和小哥订单的交集大小, 若某路区和某小哥, 交集都是彼此的top1, 则匹配
    rid2oids = {rid: set(o["id"] for o in odrs) for rid, odrs in rid2odrs.items()}
    cid2oids = {cid: set(o["id"] for o in odrs) for cid, odrs in cid2odrs.items()}
    rid2top_cids = defaultdict(list)  
    cid2top_rids = defaultdict(list)
    for rid, roids in rid2oids.items():
        for cid, coids in cid2oids.items():
            num = len(roids & coids)
            rid2top_cids[rid].append((cid, num))
            cid2top_rids[cid].append((rid, num))
    for d in [rid2top_cids, cid2top_rids]:
        for v in d.values():
            v.sort(key=lambda x: -x[1])

    while True:
        for rid, top_cids in rid2top_cids.items():
            top_cid = top_cids[0][0]
            if cid2top_rids[top_cid][0][0] == rid:
                match_pairs.append((rid, top_cid))
                break
        else:  # 不存在交集是彼此top1的情况, 退出
            break
        # 已经匹配成功的rid和cid从rid2top_cids和cid2top_rids中删除
        rid2top_cids.pop(rid)
        cid2top_rids.pop(top_cid)
        if not (rid2top_cids and cid2top_rids):
            break
        for k, v in rid2top_cids.items():
            rid2top_cids[k] = [x for x in v if x[0] != top_cid]
        for k, v in cid2top_rids.items():
            cid2top_rids[k] = [x for x in v if x[0] != rid]
    # 发现做完这步, 已经完成所有匹配
    assert len(set(x[0] for x in match_pairs)) == len(set(x[1] for x in match_pairs)) == len(rid2odrs) == len(cid2odrs)

    return [(cid, rid2odrs[rid]) for rid, cid in match_pairs]


if __name__ == "__main__":
    orders_recover = pickle.load(open("data/orders_recover.pkl", "rb"))  # 24 - 1
    orders_recover = [x for x in orders_recover if len(x[1]) > 0]
    print(len(orders_recover))
    print([len(x[1]) for x in orders_recover])
    print(sum(len(x[1]) for x in orders_recover))

    regions_replan = pickle.load(open("data/regions_partition.pkl", "rb"))  # 25

    # # 通过可视化, 决定如何合并路区, 从25个调整到23个, 以符合当天小哥人数
    # odr_partition_rids = [buildings[o["building_id"]]["region_partition"] for o in sum(orders_recover, [])]
    # counter = Counter(odr_partition_rids)
    # plot_regions_with_few_odrs(regions_replan, counter)

    # 按合并后的重划分路区, 将订单分成23份
    rid2odrs = partition_orders(sum([x[1] for x in orders_recover], []), [(15, 17), (4, 14)])
    
    # 将小哥与新路区匹配, 生成订单分配方案
    orders_partition = odrs2courier(rid2odrs, {cid: odrs for cid, odrs in orders_recover if len(odrs) > 0})
    print(len(orders_partition))
    print([len(x[1]) for x in orders_partition])
    print(sum(len(x[1]) for x in orders_partition))

    pickle.dump(orders_partition, open("data/orders_partition.pkl", "wb"))
