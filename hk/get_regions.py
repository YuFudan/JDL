import json
import pickle
from pprint import pprint

import folium
import pandas as pd
from constants_spatial import *
from coord_convert.transform import gcj2wgs, wgs2gcj
from shapely.affinity import scale
from shapely.geometry import Polygon
from shapely.ops import unary_union


def read_regions(path):
    regions = []
    polys = []
    for rid, (name, poly) in enumerate(pd.read_csv(path).values):
        points = [x.split(" ") for x in poly[10:-2].split(", ")]
        if name == "京东贝":  # 可视化发现这个路区边界有问题, 特殊处理一下
            points = points[1:20]
            points.append(points[0])
        assert points[0] == points[-1]
        gps = [gcj2wgs(float(lon), float(lat)) for lon, lat in points]
        xy = [projector(*p) for p in gps]
        poly = Polygon(xy)
        polys.append(poly)
        regions.append({
            "id": rid,
            "name": name,
            "gps": gps,
            "xy": xy,
            "area": poly.area
        })
    scale_factor = 1.01
    cnt = 0
    while True:
        cnt += 1
        bound = unary_union([scale(p, xfact=scale_factor, yfact=scale_factor, origin='centroid') for p in polys])
        if isinstance(bound, Polygon):
            break  # 所有路区的总边界
        if cnt == 100:
            assert False
        scale_factor += 0.01
    xy = bound.exterior.coords[:]
    gps = [projector(*p, inverse=True) for p in xy]
    regions.append({
        "id": rid + 1,
        "name": "鸿坤总边界",
        "gps": gps,
        "xy": xy,
        "area": bound.area
    })
    return regions


def plot_regions(regions, path):
    m = get_base_map()
    colors = get_colors()
    for i, r in enumerate(regions):
        color = colors[i % len(colors)]
        folium.Polygon(
            locations=[wgs2gcj(*p)[::-1] for p in r["gps"]],
            opacity=0.5,
            weight=3,
            popup=(r["name"], r["id"], r["area"]),
            color=color if "总边界" not in r["name"] else "black",
            fill="总边界" not in r["name"]
            ).add_to(m)
        centroid = Polygon(r["gps"]).representative_point().coords[:][0]
        folium.CircleMarker(
            location=wgs2gcj(*centroid)[::-1],
            opacity=0.8,
            radius=3,
            fill=True,
            popup=(r["name"], r["id"], r["area"]),
            color=color).add_to(m)
    m.save(path)


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
                "geometry": {"type": "Polygon", "coordinates": [region["gps"]]},
            }
        )
    return regions_geojson


if __name__ == "__main__":
    regions = read_regions("orig_data/region.csv")
    pickle.dump(regions, open("data/regions.pkl", "wb"))

    plot_regions(regions, "figure/regions.html")

    # regions_geojson = get_geojson(regions)
    # json.dump(regions_geojson, open("data/regions_geojson.json", "w"))
