"""
坐标系转换示例代码
"""

# 1. 不同经纬度坐标系间的转换
# 京东数据中的坐标系是GCJ坐标系
# 我们整个项目默认使用WGS坐标系(因为前端的mapbox底图来源于OSM, 是WGS坐标系)
# 如果有从高德获取的数据, 是GCJ坐标系
# 如果有从百度获取的数据, 是BD坐标系

from eviltransform import wgs2gcj, gcj2wgs, gcj2bd, bd2gcj, wgs2bd, bd2wgs  # 注意传参顺序是先lat后lon
# from coord_convert.transform import wgs2gcj, gcj2wgs, gcj2bd, bd2gcj, wgs2bd, bd2wgs
# coord_convert.transform也有这一套工具, 传参顺序是先lon后lat

lon, lat = 116.31221, 40.01139
lat, lon = gcj2wgs(lat, lon)
print(lon, lat)

# 2. 经纬度坐标与平面投影坐标的转换
# 经纬度坐标是球面坐标, 直接使用经纬度坐标计算距离等几何关系是不准确的
# 有时需要投影成平面坐标进行各种计算

from pyproj import Proj

projector = Proj("+proj=utm +zone=50 +ellps=WGS84 +datum=WGS84 +units=m +no_defs")  
lon, lat = 116.31221, 40.01139
x, y = projector(lon, lat)  # 输入的经纬度为WGS
print(x, y)
lon, lat = projector(x, y, inverse=True)
print(lon, lat)
