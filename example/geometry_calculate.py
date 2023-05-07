"""
几何关系计算示例代码: shapely的使用
"""
from shapely.geometry import Point, LineString, Polygon
from shapely import ops
import matplotlib.pyplot as plt

point = Point((0.5, 1.5))
line = LineString([(0, 0), (1,0), (1,1), (2,1), (2,0), (3,0)])  # 折线 
poly = Polygon([(2.5, 1), (3, 1.5), (2.5, 2), (2, 1.5), (2.5, 1)])  # 多边形

plt.axis("equal")
plt.scatter(*point.coords[:][0])
plt.plot(*zip(*line.coords[:]))
plt.plot(*zip(*poly.exterior.coords[:]))

# 折线长度
print(line.length)

# 多边形面积
print(poly.area)

# 多边形几何中心
point_centroid = poly.centroid
plt.scatter(*point_centroid.coords[:][0])

# 点到折线的距离
print(line.distance(point))

# 折线到多边形的距离
print(line.distance(poly))

# 点到多边形的距离
print(point.distance(poly))

# 获取任意点在折线上的位置(沿折线起点走到该点的长度)(若点不在折线上, 用投影点的位置计算)
distance_along_line = line.project(point)
print(distance_along_line)

# 获取沿折线起点走任意长度后的点
point_proj = line.interpolate(distance_along_line)   # 获取point在line上的投影点
point_half = line.interpolate(0.5, normalized=True)  # 获取line的中点
plt.scatter(*point_proj.coords[:][0])
plt.scatter(*point_half.coords[:][0])

# 折线与多边形的最近点
point_line, point_poly = ops.nearest_points(line, poly)
plt.scatter(*point_line.coords[:][0])
plt.scatter(*point_poly.coords[:][0])

# 求两个多边形的相交形
poly2 = Polygon([(2, 1.5), (2.5, 1.5), (2.5, 2), (2, 2), (2, 1.5)])  # 多边形
poly_intersection = poly.intersection(poly2)
print(poly_intersection.area)
plt.plot(*zip(*poly2.exterior.coords[:]))
plt.plot(*zip(*poly_intersection.exterior.coords[:]))

plt.savefig("test.png")
