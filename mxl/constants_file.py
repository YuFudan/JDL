import os
import pickle

from shapely.geometry import Point, Polygon

path = os.getcwd()  # 获取工作路径
is_root = path.split("/")[-1] == "jd_demo1"
if is_root:
    from mxl.constants_spatial import *
    prefix = path + "/mxl/data/"
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
station_names = ["漫香林营业部"]
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
    r["poly"] = Polygon([projector(*p) for p in r["boundary"]])  # 平面投影坐标
regions = {r["id"]:r for r in regions}

#####   业务信息   #####

# # 做功
# cid2e_cum = pickle.load(open("/home/yufudan/jd_demo1/mxl/data/cid2cum_energy.pkl", "rb"))
# E_DISCOUNT = 200  # 将做功除以一个倍数以免数值太大
# for cid, e_cum in cid2e_cum.items():
#     cid2e_cum[cid] = [(t, cum / E_DISCOUNT) for t, cum in e_cum]

# 投诉数据
# complaint_list = [
#     {
#         "id": 21173943,  # 贺建业
#         "info": "服务态度差",
#         "time": 57935
#     },
# ]
# complaint_list.sort(key=lambda x:x["time"])
complaint_list = []

# 出勤数据
# 除去仓管员, 营业部有24名常驻员工, 当天21人到岗, 另有2名外包来干活
# attendance_dict = {  # 1101
#     21136050: "出勤",
#     154889: "出勤",
#     20485832: "出勤",
#     22278962: "出勤",
#     20862519: "出勤",
#     21173943: "出勤",
#     227164: "出勤",
#     28686781: "缺勤",
#     21777999: "出勤",
#     16303305: "缺勤",
#     22342982: "出勤",
#     15408023: "缺勤",
#     22606899: "出勤",
#     22184813: "出勤",
#     21495458: "出勤",
#     3828993: "缺勤",
#     28686698: "缺勤",
#     6105736: "缺勤",
#     22626330: "出勤",
#     15407933: "缺勤",
#     21979464: "出勤",
#     22602631: "出勤",
# }

attendance_dict = {  # 0828
    21136050: "缺勤",
    154889: "出勤",
    20485832: "出勤",
    22278962: "出勤",
    20862519: "出勤",
    21173943: "出勤",
    227164: "出勤",
    28686781: "缺勤",
    21777999: "出勤",
    16303305: "缺勤",
    22342982: "出勤",
    15408023: "缺勤",
    22606899: "出勤",
    22184813: "出勤",
    21495458: "出勤",
    3828993: "缺勤",
    28686698: "缺勤",
    6105736: "缺勤",
    22626330: "缺勤",
    15407933: "缺勤",
    21979464: "出勤",
    22602631: "出勤",
    21952290: "外包",
    22607367: "外包"
}

# 营业部常驻小哥人数(除去外包和仓管)
NUM_COURIER_ALL = 22

# 小哥假名, 姓名随机生成器https://www.qqxiuzi.cn/zh/xingming/
cid2name = {
    21136050: "王强",
    154889: "胡亮亮",
    20485832: "赵东震",
    22278962: "丁云龙",
    20862519: "郝刚强",
    21173943: "王云飞",  ##
    227164: "王青柯",
    28686781: "孟光辉",
    21777999: "李文凯",  ##
    16303305: "冯修平",
    22342982: "胡苏亮",
    15408023: "杨文虹",
    22606899: "赵详志",
    22184813: "田超凡",
    21495458: "葛庆煜",
    3828993: "汤勇锐",
    28686698: "孔新荣",
    6105736: "苏明杰",
    22626330: "王茂林",  #
    15407933: "王剑",
    21979464: "王书亚",
    22602631: "肖明江",  # 
    21952290: "郑凯旋",  # 无名 19 14
    22469286: "余飞龙",  # 无名 15 13
    22607367: "谭茂学"   # 无名 4  2
}
