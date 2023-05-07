import os
import pickle

from shapely.geometry import Point, Polygon

path = os.getcwd()  # 获取工作路径
is_root = path.split("/")[-1] == "jd_demo1"
if is_root:
    from hk.constants_spatial import *
    prefix = path + "/hk/data/"
else:
    from constants_spatial import *
    prefix = path + "/data/"

#####   地理文件   #####

# 楼
buildings = pickle.load(open(prefix + "buildings.pkl", "rb"))
for bd in buildings:
    bd["poly"] = Polygon([projector(*p) for p in bd["points"]])  # 平面投影坐标
    bd["point"] = Point(bd["gate_xy"])
buildings = {bd["id"]: bd for bd in buildings}

# 快递站(营业部)
station_lonlats = [(LON_STA, LAT_STA)]
station_ids = ["快递站"]  # 路网中该快递站节点的node_id
station_names = ["鸿坤营业部"]
stations = {}
for gps, nid, name in zip(station_lonlats, station_ids, station_names):
    xy = projector(*gps)
    stations[nid] = {
        "id": nid,
        "name": name,
        "gps": gps,
        "xy": xy,
        "point": Point(xy)
    }
station_ids = set(station_ids)

# 路网
G = pickle.load(open(prefix + "G.pkl", "rb"))
intersections = [
    (node[0], node[1]["xy"]) 
    for node in G.nodes(data=True) 
    if ("building" not in node[1] or node[1]["building"] == -1) and node[0] not in station_ids
]  # 所有非楼门非快递站的路网节点坐标

# 路区
regions = pickle.load(open(prefix + "regions.pkl", "rb"))
for r in regions:
    r["poly"] = Polygon(r["xy"])  # 平面投影坐标
region_fence = [r for r in regions if "总边界" in r["name"]][0]
regions = {r["id"]:r for r in regions if "总边界" not in r["name"]}

#####   业务信息   #####

# 小哥姓名(从订单表中取出)
cid2name = {
    89267: '位全收',
    92755: '姜正喜',
    227165: '李海明',
    20106372: '展望',
    20169502: '庞刚锋',
    20448697: '李延明',
    20601816: 'ZBJ043',
    20707123: '许鹏杰',
    20708462: '赵小平',
    20840560: '陈武峰',
    20840584: '张杰',
    20864736: '杨广胜',
    20885592: '李法波',
    21036134: '胡光辉',
    21529043: '陈祯',
    21744833: '刘云鹏',
    21762059: '张晓磊',
    21948367: '李法周',
    21952368: '党朋菲',
    22015532: 'ZBJ019',
    22119540: '卢春民',
    22196342: '杨浩',
    22294120: '尹成龙',
    22295986: '刘金保',
    22347237: '米涛',
    22431541: '尹东超',
    22476612: '武嘉豪',
    22600383: '谢建如',
    22784062: '崔瑞广',
    22946772: '姚慧佳',
    22969590: '张松',
    28603560: '陈向丽'
}

# 投诉数据
complaint_list = []

# 出勤数据
attendance_dict = {
    20840584: "出勤",
    20169502: "出勤",
    89267: "出勤",
    20707123: "出勤",
    22431541: "出勤",
    22784062: "出勤",
    22600383: "出勤",
    21744833: "出勤",
    22476612: "出勤",
    21948367: "出勤",
    21529043: "出勤",
    92755: "出勤",
    20885592: "出勤",
    227165: "出勤",
    20864736: "出勤",
    22347237: "出勤",
    22294120: "出勤",
    20708462: "出勤",
    21952368: "出勤",
    20840560: "出勤",
    22119540: "出勤",
    22969590: "出勤",
    21762059: "缺勤",
    22946772: "缺勤",
    22196342: "缺勤"
}

# 营业部常驻小哥人数(除去外包和仓管)
NUM_COURIER_ALL = 22
