import folium


def plot_satellite_folum():
    """
    使用高德卫星图作为folium的自定义底图瓦片
    注意, 与使用默认的OSM底图时不同, 传入的经纬度坐标需要是GCJ坐标而不是WGS坐标
    """
    G_folium = folium.Map(
        location=[39.967691, 116.426234],
        control_scale=True,
        tiles='https://webst02.is.autonavi.com/appmaptile?style=6&x={x}&y={y}&z={z}',
        attr='高德-卫星影像图',
        zoom_start=20,
    )
    lonlats = [(116.428653,39.968916), (116.427859,39.968887), (116.427607,39.969051), (116.42632,39.969014), (116.426271,39.968525), (116.425783,39.968509)]
    folium.PolyLine(
        locations=[[lat, lon] for lon, lat in lonlats],
        opacity=0.8,
        weight=5,
        popup="this is a polyline",
        color="red",
    ).add_to(G_folium)
    G_folium.save("test.html")


if __name__ == "__main__":
    plot_satellite_folum()
