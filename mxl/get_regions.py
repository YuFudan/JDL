import json
import pickle
from pprint import pprint

import folium
import pandas as pd
from eviltransform import gcj2wgs, wgs2gcj
from constants_spatial import *
from shapely.geometry import Polygon


def read_regions(path):
    regions = pd.read_csv(path).values
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


def plot_regions(regions, path):
    G_folium = get_base_map()
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


if __name__ == "__main__":
    regions = read_regions("orig_data/luqu.csv")
    pickle.dump(regions, open("data/regions.pkl", "wb"))

    plot_regions(regions, "figure/regions.html")

    regions_geojson = get_geojson(regions)
    json.dump(regions_geojson, open("data/regions_geojson.json", "w"))
