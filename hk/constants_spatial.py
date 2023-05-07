import folium
from pyproj import Proj
from coord_convert.transform import wgs2gcj

LON_CEN, LAT_CEN = 116.556390, 39.786331
projector = Proj(f"+proj=tmerc +lat_0={LAT_CEN} +lon_0={LON_CEN}")
X_CEN, Y_CEN = projector(LON_CEN, LAT_CEN)

LON_STA, LAT_STA = 116.552786, 39.766541
X_STA, Y_STA = projector(LON_STA, LAT_STA)


def get_base_map(gaode=True):
    if gaode:
        m = folium.Map(
            location=wgs2gcj(LON_CEN, LAT_CEN)[::-1],
            control_scale=True,
            tiles='http://webrd02.is.autonavi.com/appmaptile?lang=zh_cn&size=1&scale=1&style=8&x={x}&y={y}&z={z}',
            attr='高德底图',
            zoom_start=13,
        )
    else:
        m = folium.Map(
            location=[LAT_CEN, LON_CEN],
            control_scale=True,
            zoom_start=13,
        )
    m.add_child(folium.LatLngPopup())
    return m

def get_colors():
    return ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
