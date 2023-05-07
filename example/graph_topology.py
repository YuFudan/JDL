"""
路网拓扑图处理示例代码: Networkx的使用
"""
import pickle
import folium
from networkx import DiGraph, shortest_path  # 有向图
from pyproj import Proj
from shapely.geometry import LineString

_, nodes, roads = pickle.load(open("roadmap.pkl", "rb"))
projector = Proj("+proj=utm +zone=50 +ellps=WGS84 +datum=WGS84 +units=m +no_defs")

def construct_graph(nodes, roads):
    """
    构建路网(输出一个DiGraph对象)
    """
    # 图的node是由node_id和一个信息字典组成的tuple(node_id, {}), 信息字典的内容任意
    G_nodes = []
    for nid, (lon, lat) in nodes.items():
        G_nodes.append(
            (
                nid,
                {
                    "id": nid,
                    "gps": (lon, lat),
                },
            )
        )
    # 图的edge是由o_node_id, d_node_id和一个信息字典组成的tuple(o_node_id, d_node_id, {}), 信息字典的内容任意
    G_edges = []
    for (o, d), gps in roads:
        xys = [projector(*p) for p in gps]
        length = LineString(xys).length
        G_edges.append(
            (
                o,
                d,
                {
                    "id": (o, d),
                    "gps": gps,
                },
            )
        )
        G_edges.append(  # 路是双向路, 因此也添加一条从d到o的路
            (
                d,
                o,
                {
                    "id": (d, o),
                    "gps": gps[::-1],
                    "length": length
                },
            )
        )

    G = DiGraph()
    G.add_nodes_from(G_nodes)
    G.add_edges_from(G_edges)

    return G


G = construct_graph(nodes, roads)


def plot_G(G):
    """
    可视化路网
    """
    lon_cen, lat_cen = 116.4712, 39.9180
    G_folium = folium.Map(location=[lat_cen, lon_cen], control_scale=True)
    for o, d, einfo in G.edges(data=True):  # 遍历edge
        gps = einfo["gps"]
        folium.PolyLine(
            locations=[[lat, lon] for lon, lat in gps],
            opacity=0.8,
            weight=3,
            popup=(o, d),
            color="gray",
        ).add_to(G_folium)
    for nid, ninfo in G.nodes(data=True):  # 遍历node
        lon, lat = ninfo["gps"]
        folium.CircleMarker(
            location=[lat, lon],
            opacity=0.8,
            radius=3,
            fill=True,
            popup=nid,
            color="black",
        ).add_to(G_folium)
    G_folium.save("test.html")


# plot_G(G)


def other_operations(G):
    """
    其它操作
    """
    nodes = list(G.nodes(data=True))
    print(nodes[0])

    edges = list(G.edges(data=True))
    print(edges[0])

    print(G.nodes[14])  # 根据node_id找node

    print(G.edges[14, 1])  # 根据o_node_id, d_node_id找edge

    print(G._pred[14])  # node前驱  

    print(G._succ[1])  # node后继

    print(shortest_path(G, 1, 14, "length"))  # 按length字段求1到14的最短路径

other_operations(G)
