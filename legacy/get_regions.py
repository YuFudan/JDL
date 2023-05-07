"""
读取重划分前后的路区边界.xlsx, .csv
"""
import json
import pickle
from pprint import pprint

import folium
import pandas as pd
from eviltransform import gcj2wgs, wgs2gcj
from pyproj import Proj
from shapely.geometry import Polygon, Point


projector = Proj("+proj=utm +zone=50 +ellps=WGS84 +datum=WGS84 +units=m +no_defs")


def read_regions():
    regions = pd.read_excel("orig_data/regions.xlsx", header=None)
    regions = regions.values
    regions_all = []
    for _, rid, poly in regions:
        # if rid == 11 or rid == 107:  # 11 和平里-十二区  107 化研院南小区
        if True:
            poly = poly[9:-2]
            points = [x.split(" ") for x in poly.split(",")]
            points = [(float(lon), float(lat)) for lon, lat in points]
            points = [gcj2wgs(*p[::-1])[::-1] for p in points]
            regions_all.append({
                # "name": "和平里-十二区" if rid == 11 else "化研院南小区",
                "id": rid,
                "boundary": points,
                "area": Polygon([projector(*p) for p in points]).area
            })
    return regions_all


def read_regions_mxl():
    regions = pd.read_csv("orig_data_mxl/luqu.csv")
    regions = regions.values
    regions_all = []
    for _, features, rid, poly in regions:
        if rid == 0:  # 0号路区实际为所有路区的总轮廓
            continue
        name = features.split("areaName': '")[1].split(',')[0][:-1]
        poly = poly[10:-2]
        points = [x.split(" ") for x in poly.split(", ")]
        points = [(float(lon), float(lat)) for lon, lat in points]
        points = [gcj2wgs(*p[::-1])[::-1] for p in points]
        regions_all.append({
            "name": name,
            "id": rid,
            "boundary": points,
            "area": Polygon([projector(*p) for p in points]).area
        })
    return regions_all


def read_regions_partition():
    regions = pd.read_csv("orig_data/regions_partition.csv", encoding="gbk")
    regions = regions.values
    regions_all = []
    for x in regions:
        poly = x[3][10:-2]
        points = [x.strip().split(" ") for x in poly.split(",")]
        points = [(float(lon), float(lat)) for lon, lat in points]
        points = [gcj2wgs(*p[::-1])[::-1] for p in points]
        regions_all.append({
            # "name": "和平里-十二区" if rid == 11 else "化研院南小区",
            "id": int(x[0]),
            "boundary": points,
            "area": Polygon([projector(*p) for p in points]).area
        })
    return regions_all


def plot_regions(regions, path):
    G_folium = folium.Map(
        # location=[39.958759, 116.426223],  # 民旺
        location=[39.8141, 116.5270],  # 漫香林
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
        folium.CircleMarker(
            location=centroid[::-1],
            opacity=0.8,
            radius=3,
            fill=True,
            popup=(r["id"], round(r["area"]))
        ).add_to(G_folium)
        folium.PolyLine(
            locations=[p[::-1] for p in points],
            opacity=0.8,
            weight=3,
            popup=(r["id"], round(r["area"])),
        ).add_to(G_folium)
    G_folium.save(path)


def compare_partition(regions, regions_partition):
    """对比划分前后的路区"""
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
        folium.PolyLine(
            locations=[p[::-1] for p in points],
            opacity=0.8,
            weight=5,
            popup=(r["id"], round(r["area"])),
        ).add_to(G_folium)
    for r in regions_partition:
        points = [wgs2gcj(*p[::-1])[::-1] for p in r["boundary"]]
        # centroid = Polygon(points).centroid.coords[:][0]
        centroid = Polygon(points).representative_point().coords[:][0]
        folium.PolyLine(
            locations=[p[::-1] for p in points],
            opacity=0.8,
            weight=1,
            color="red",
            popup=(r["id"], round(r["area"])),
        ).add_to(G_folium)
    G_folium.save("figure/regions_compare.html")


def get_geojson(regions):
    regions_geojson = []
    for region in regions:
        if "name" in region:
            properties = {"id": region["id"], "area": region["area"], "name": region["name"]}
        else:
            properties = {"id": region["id"], "area": region["area"]}
        regions_geojson.append(
            {
                "type": "Feature",
                "properties": properties,
                "geometry": {"type": "Polygon", "coordinates": [region["boundary"]]},
            }
        )
    return regions_geojson


def get_building_region(buildings, regions, regions_partition):
    """计算楼在哪个路区"""
    for r in regions:
        r["poly"] = Polygon(r["boundary"])
    for r in regions_partition:
        r["poly"] = Polygon(r["boundary"])
    for bd in buildings:
        p = Point(bd["gate_gps"])

        dis_ids = []
        for r in regions:
            dis = p.distance(r["poly"])
            if dis == 0:
                bd["region"] = r["id"]
                break
            dis_ids.append((dis, r["id"]))
        else:
            bd["region"] = min(dis_ids, key=lambda x:x[0])[1]
        
        dis_ids = []
        for r in regions_partition:
            dis = p.distance(r["poly"])
            if dis == 0:
                bd["region_partition"] = r["id"]
                break
            dis_ids.append((dis, r["id"]))
        else:
            bd["region_partition"] = min(dis_ids, key=lambda x:x[0])[1]
    return buildings


if __name__ == "__main__":
    regions = read_regions_mxl()
    # pickle.dump(regions, open("data_mxl/regions.pkl", "wb"))

    plot_regions(regions, "figure_mxl/regions.html")

    # regions_geojson = get_geojson(regions)
    # json.dump(regions_geojson, open("data_mxl/regions_geojson.json", "w"))
    exit()

    # # 读原始路区
    # regions = read_regions()
    # pickle.dump(regions, open("data/regions_all.pkl", "wb"))
    regions = pickle.load(open("data/regions_all.pkl", "rb"))
    assert len(regions) == len(set(r["id"] for r in regions))
    # plot_regions(regions, path="figure/regions.html")

    # # 生成原始路区geojson
    # regions_geojson = get_geojson(regions)
    # json.dump(regions_geojson, open("data/regions_all_geojson.json", "w"))

    # # 读重划路区
    # regions_partition = read_regions_partition()
    # pickle.dump(regions_partition, open("data/regions_partition.pkl", "wb"))
    regions_partition = pickle.load(open("data/regions_partition.pkl", "rb"))
    assert len(regions_partition) == len(set(r["id"] for r in regions_partition))
    # plot_regions(regions_partition, path="figure/regions_partition.html")

    # # 生成重划路区geojson
    # regions_partition_geojson = get_geojson(regions_partition)
    # json.dump(regions_partition_geojson, open("data/regions_partition_geojson.json", "w"))

    # compare_partition(regions, regions_partition)

    # # 记录楼在哪个路区
    # buildings = pickle.load(open("data/buildings_new.pkl", "rb"))
    # buildings = get_building_region(buildings, regions, regions_partition)
    # pickle.dump(buildings, open("data/buildings_new.pkl", "wb"))
