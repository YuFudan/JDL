"""
根据订单地址"北京市朝阳区三环以内和平街12区13号楼1单元8号"
匹配到路网上的建筑
注意调用的是高德地理编码接口: https://lbs.amap.com/api/webservice/guide/api/georegeo
需要输入的是一个结构化的地址, 规则遵循：国家、省份、城市、区县、城镇、乡村、街道、门牌号码、屋邨、大厦，如：北京市朝阳区阜通东大街6号
如果订单地址没有诸如"北京市朝阳区"的前缀, 需要补充上
"""
import pickle
import random
import requests
from pyproj import Proj
from eviltransform import gcj2wgs, wgs2gcj, bd2gcj
from shapely.geometry import Polygon, Point


projector = Proj("+proj=utm +zone=50 +ellps=WGS84 +datum=WGS84 +units=m +no_defs")
buildings = pickle.load(open("data/buildings_all_refined.pkl", "rb"))
for bd in buildings:
    bd["poly"] = Polygon([projector(*p) for p in bd["points"]])  # 平面投影坐标


def get_random_user_agent():
    user_agents = [
        "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36 OPR/26.0.1656.60",
        "Opera/8.0 (Windows NT 5.1; U; en)",
        "Mozilla/5.0 (Windows NT 5.1; U; en; rv:1.8.1) Gecko/20061208 Firefox/2.0.0 Opera 9.50",
        "Mozilla/4.0 (compatible; MSIE 6.0; Windows NT 5.1; en) Opera 9.50",
        "Mozilla/5.0 (Windows NT 6.1; WOW64; rv:34.0) Gecko/20100101 Firefox/34.0",
        "Mozilla/5.0 (X11; U; Linux x86_64; zh-CN; rv:1.9.2.10) Gecko/20100922 Ubuntu/10.10 (maverick) Firefox/3.6.10",
        "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/534.57.2 (KHTML, like Gecko) Version/5.1.7 Safari/534.57.2",
        "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.71 Safari/537.36",
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.11 (KHTML, like Gecko) Chrome/23.0.1271.64 Safari/537.11",
        "Mozilla/5.0 (Windows; U; Windows NT 6.1; en-US) AppleWebKit/534.16 (KHTML, like Gecko) Chrome/10.0.648.133 ",
        "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/30.0.1599.101 Safari/537.36",
        "Mozilla/5.0 (Windows NT 6.3; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_2) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/35.0.1916.153 Safari/537.36",
        "Mozilla/5.0 (Windows NT 6.1; WOW64; rv:30.0) Gecko/20100101 Firefox/30.0",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_2) AppleWebKit/537.75.14 (KHTML, like Gecko) Version/7.0.3 Safari/537.75.14",
        "Mozilla/5.0 (compatible; MSIE 10.0; Windows NT 6.2; Win64; x64; Trident/6.0)",
        "Mozilla/5.0 (Windows; U; Windows NT 5.1; it; rv:1.8.1.11) Gecko/20071127 Firefox/2.0.0.11",
        "Opera/9.25 (Windows NT 5.1; U; en)",
        "Mozilla/4.0 (compatible; MSIE 6.0; Windows NT 5.1; SV1; .NET CLR 1.1.4322; .NET CLR 2.0.50727)",
        "Mozilla/5.0 (compatible; Konqueror/3.5; Linux) KHTML/3.5.5 (like Gecko) (Kubuntu)",
        "Mozilla/5.0 (X11; U; Linux i686; en-US; rv:1.8.0.12) Gecko/20070731 Ubuntu/dapper-security Firefox/1.5.0.12",
        "Lynx/2.8.5rel.1 libwww-FM/2.14 SSL-MM/1.4.1 GNUTLS/1.2.9",
        "Mozilla/5.0 (X11; Linux i686) AppleWebKit/535.7 (KHTML, like Gecko) Ubuntu/11.04 Chromium/16.0.912.77 Chrome/16.0.912.77 Safari/535.7",
        "Mozilla/5.0 (X11; Ubuntu; Linux i686; rv:10.0) Gecko/20100101 Firefox/10.0 ",
    ]
    return random.choice(user_agents)


def get_random_key():
    keys = [
        "7b111df4c999cad85c134530eea33e0e", 
        "710aa01769ade93cf70f2f71106e1cef", 
        "2d2de838f89274477b723b6eaddd079a", 
        "6b419f8bfa982f447be3281641510393",
        "3bf9a79926f94c68836a08fce736b102"
    ]
    return random.choice(keys)


def query_API(keywords):
    user_key = get_random_key()
    user_agent = get_random_user_agent()
    type = "190000"  # 高德POI分类码, 190000为地名地址信息, 即xx小区xx号楼这种
    citycode = "110000"  # 北京市
    url_geo_code = f"https://restapi.amap.com/v3/geocode/geo?key={user_key}&address={keywords}&types={type}&region={citycode}"
    try:
        return requests.get(url_geo_code, headers={"User-Agent": user_agent}, timeout=30).json()
    except:
        return url_geo_code


def find_order_in_which_building(order_address, buildings):
    result = query_API(order_address)
    if isinstance(result, str):
        return (-1, f"高德API访问失败: 网络原因或API达到使用限额或被反爬, 请尝试浏览器访问:{result}")
    try:
        formatted_address = result["geocodes"][0]["formatted_address"]  # 高德解析后的结构化地址
        lon, lat = result["geocodes"][0]["location"].split(",")  # 高德解析后的GCJ坐标
        lon, lat = float(lon), float(lat)
    except:
        return (-1, "高德API返回为空", result)

    lat, lon = gcj2wgs(lat, lon)
    x, y = projector(lon, lat)
    point = Point((x, y))
    tmp = []
    for bd in buildings:
        dis = point.distance(bd["poly"])
        if dis == 0:
            return (
                bd["id"],   
                bd["name"],
                formatted_address,
                (lon, lat)
            )
        tmp.append([dis, bd["id"], bd["name"]])
    t = min(tmp, key=lambda x:x[0])
    if t[0] < 30:
        return (
            t[1],
            t[2],
            formatted_address,
            (lon, lat)
        )
            
    return (-1, "高德解析出的地址无法匹配到较近的楼, 解析结果为:", formatted_address, wgs2gcj(lat, lon)[::-1])


if __name__ == "__main__":
    addresses = [
        # "北京朝阳区三环以内北京和平里十二区17号楼1416",   # 原始订单地址
        # "北京市朝阳区三环以内和平街12区13号楼1单元8号", 
        # "北京朝阳区和平街街道和平街十二区20号楼西单元1层",
        # "北京朝阳区三环以内和平街十二区20号楼西单元1层",
        # "北京朝阳区和平街街道和平街12区19号楼西侧公交总站院内",
        # "北京朝阳区三环以内北京和平里十二区17号楼",   # 去掉"楼"以后的字
        # "北京市朝阳区三环以内和平街12区13号楼", 
        # "北京朝阳区和平街街道和平街十二区20号楼",
        # "北京朝阳区三环以内和平街十二区20号楼",
        # "北京朝阳区和平街街道和平街12区19号楼",
        # "北京朝阳区和平街十四区16号楼",

        # "北京海淀区展春园小区9号楼二单元201",
        # "小区9号楼",

        "北京东城区内环到三环里北京市东城区和平里东街18号国家林业局"
    ]

    for order_address in addresses:
        t = find_order_in_which_building(order_address, buildings)
        print(t)
    