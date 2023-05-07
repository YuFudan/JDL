"""
可视化示例代码
"""
import folium

# folium可视化, 好处是有OSM底图(输入WGS经纬度坐标)

# 1. OSM底图
G_folium = folium.Map(
    location=[39.9180, 116.4712],  # 目标区域的大概位置即可
    control_scale=True             # 显示比例尺
)

# 2. 画点
folium.CircleMarker(  # CircleMarker的大小是像素, 即无论地图怎么缩放, 始终占一定的像素。想要大小是物理大小的, 用folium.Circle()
    location=[lat, lon],
    opacity=0.8,  # 不透明度
    radius=3,
    fill=True,   
    popup="this is a node",   # popup为鼠标点击弹窗显示的信息, 可以传入int, string, tuple, list, dict等基本数据类型
    color="black",
).add_to(G_folium)

# 3. 画线
folium.PolyLine(
    locations=[[lat, lon] for lon, lat in lonlats],
    opacity=0.8,
    weight=3,
    popup="this is a polyline",
    color="orange",
).add_to(G_folium)

# 4. 保存成html, 浏览器打开查看
G_folium.save("test.html")



# matplotlib可视化, 好处是轻便(输入平面投影坐标)

import matplotlib.pyplot as plt

plt.figure(figsize=(12, 12))
plt.axis("equal")

for lon, lat in nodes.values():
    x, y = projector(lon, lat)
    plt.scatter(x, y, c="black", zorder=2)  # zorder大的在图层上方
for od, geo in roads:
    xys = [projector(*p) for p in geo]
    plt.plot(*zip(*xys), c="gray", zorder=1)
for bd in buildings.values():
    plt.plot(*zip(*bd["xy"]), c="orange", zorder=1)

plt.savefig("figure/roadmap.png")
