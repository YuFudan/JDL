import pickle
from collections import defaultdict
from pprint import pprint

import folium
from coord_convert.transform import wgs2gcj
from pyproj import Proj
from shapely.geometry import Polygon

projector = Proj("+proj=tmerc +lat_0=39.816344 +lon_0=116.520481")
LON_CEN, LAT_CEN = 116.520481, 39.816344
X_CEN, Y_CEN = projector(LON_CEN, LAT_CEN)

# 订单种类
ORDER_DELIVER = "deliver"
ORDER_CPICK = "cpick"
ORDER_BPICK = "bpick"

# 行为标注
WORK = "work"    
REST = "rest" 
IGNORE = "ignore" # 大段的没标, 或者回站时间, 算指标时, 忽略此时段
OTHER = "other"   # 打电话, 打单子, 零碎的没标(大概率是楼间移动), 算指标时不忽略, 但只要不是工作都算对
UP = 1            # 细粒度行为
DOWN = 2
UNIT = 3
DELIVER = 4
ARRANGE = 5
NOT_WORK = 6

# 楼
buildings = pickle.load(open("orig_data/buildings.pkl", "rb"))
for b in buildings:
    b["gate_xy"] = projector(*b["gate_gps"])
    b["poly"] = Polygon([projector(*p) for p in b["points"]])
buildings = {bd["id"]: bd for bd in buildings}

# 路区
regions = pickle.load(open("orig_data/regions.pkl", "rb"))
for r in regions:
    r["poly"] = Polygon([projector(*p) for p in r["boundary"]])  # 平面投影坐标
regions = {r["id"]: r for r in regions}

# 营业部位置
LON_STA, LAT_STA = 116.516869, 39.808934
X_STA, Y_STA = projector(LON_STA, LAT_STA)

# 小哥姓名
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
}


def time_conventer(t):
    """将daytime转为hh:mm:ss格式"""
    t = round(t)
    assert 0 <= t < 86400
    h = t // 3600
    t -= h * 3600
    m = t // 60
    s = t - m * 60
    h = str(h) if h > 9 else f"0{h}"
    m = str(m) if m > 9 else f"0{m}"
    s = str(s) if s > 9 else f"0{s}"
    return ":".join([h, m, s])


def get_base_map():
    m = folium.Map(
        location=[LAT_CEN, LON_CEN],
        control_scale=True,
        tiles='http://webrd02.is.autonavi.com/appmaptile?lang=zh_cn&size=1&scale=1&style=8&x={x}&y={y}&z={z}',
        attr='高德底图',
        zoom_start=20,
    )
    # folium.LatLngPopup().add_to(m)
    for b in buildings.values():
        color = "orange" if b["is_elevator"] else "black"
        folium.PolyLine(
            locations=[wgs2gcj(*p)[::-1] for p in b["points"]],
            opacity=0.8,
            weight=0.5,
            color=color,
        ).add_to(m)
    return m


def xy2loc(xy):
    return wgs2gcj(*projector(*xy, inverse=True))[::-1]


default_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']


def print_table(columns, lines):
    """将差异打印成表"""
    def mylen(s):
        return sum(2 if '\u4e00' <= c <= '\u9fff' else 1 for c in s)
    lens = [max(15, mylen(k) + 3) for k in columns]
    head = "".join(k + " " * (l-mylen(k)) for k, l in zip(columns, lens))
    print(head)
    print("=" * (mylen(head) - 3))
    for line in lines:
        line = [f"{x:.4f}" if not isinstance(x, str) else x for x in line]
        print("".join(x + " "*(l - mylen(x)) for x, l in zip(line, lens)))

def group_by(arr, key):
    """arr元素为字典"""
    if len(arr) == 0:
        return {}
    assert isinstance(arr[0], dict)
    r = defaultdict(list)
    if isinstance(key, str):
        for a in arr:
            r[a[key]].append(a)
    else:
        assert isinstance(key, list)
        for a in arr:
            r[tuple(a[k] for k in key)].append(a)
    return r
