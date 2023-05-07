import json
import time
from collections import defaultdict
from pprint import pprint

import numpy as np

#####   通用常量   #####

DEFAULT_SID = "快递站"

# 订单种类
ORDER_DELIVER = "deliver"
ORDER_CPICK = "cpick"
ORDER_BPICK = "bpick"
# 动作种类
ACTION_ELEVATOR = "坐电梯"
ACTION_UPSTAIR = "爬楼"
ACTION_DOWNSTAIR = "下楼"
ACTION_TODOOR = "上门"
ACTION_DELIVER = "派件"
ACTION_CPICK = "C揽"
ACTION_BPICK = "B揽"
ACTION_WALK = "步行"
ACTION_FROMSTATION = "去路区"
ACTION_TOSTATION = "回站"
ACTION_MEETING = "开早会"
ACTION_ARRANGE = "理货拣货"
ACTION_REST = "休息"
ACTION_DISCHARGE = "卸货"# 进楼前在楼下理货卸货等
# 默认动作时间(为gen_actions.py里用到的, gen_sim_actions.py里则不同)
T_ELEVATOR_WAIT1 = 10.0  # 等电梯时间: 进楼第一次等电梯
T_ELEVATOR_WAIT2 = 3.0   # 等电梯时间: 进楼第二次及以后等电梯, 因为一般来说小哥出电梯送完单后, 电梯还没走, 马上就可以坐了
T_ELEVATOR_FLOOR = 2.0   # 电梯移动一层的时间
T_STAIR_UP = 14.0        # 步行上楼一层时间
T_STAIR_DOWN = 6.0       # 步行下楼一层时间
T_TODOOR = 2.0           # 从楼梯/电梯口到门口时间
T_DELIVER = 6.0          # 派件每单的用时
T_BPICK = 20.0           # B揽每单的用时, 需要称重打包等
T_CPICK = 30.0           # C揽每单的用时, 需要称重扫码打包推销等
T_FETCH = 1.0            # 回车取每个货的用时
V_WALK = 1.5             # 步行移动速度
V_CAR = 8.0              # 快递站和路区间往返速度

# 移动的action
ACTION_MOVE = {ACTION_WALK, ACTION_FROMSTATION, ACTION_TOSTATION}
# 揽派的action
ACTION_ORDER = {ACTION_DELIVER, ACTION_CPICK, ACTION_BPICK}
# 上下楼的action
ACTION_FLOOR = {ACTION_ELEVATOR, ACTION_UPSTAIR, ACTION_DOWNSTAIR}
# 计入work_time的action (walk, tocar, drive依据是否有target_orders判断是否计入work_time) 在站里的时间不算
ACTION_WORK = ACTION_ORDER | {ACTION_ELEVATOR, ACTION_UPSTAIR, ACTION_DOWNSTAIR, ACTION_TODOOR, ACTION_DISCHARGE}
# order映射到action
ORDER2ACTION = {ORDER_DELIVER:ACTION_DELIVER, ORDER_CPICK:ACTION_CPICK, ORDER_BPICK:ACTION_BPICK}
# order映射到t
ORDER2T = {ORDER_DELIVER:T_DELIVER, ORDER_CPICK:T_CPICK, ORDER_BPICK:T_BPICK}

######   gen_actions.py   #####

# 订单start_time比target_time的最小提前量
T_DELIVER_ADVANCE = 60.0
T_PICK_ADVANCE = 60.0
# 早会起止时间
T_MEETING_START = 7 * 3600
T_MEETING_END = 7.25 * 3600
# 处理订单iot时的阈值
T_OVERLAP = 10.0         # 订单iot之间的最小间隔
P_SHORT = 0.6            # 调整iot时最小的压缩比例

######   action2status.py   #####

# 订单状态
ORDER_UNSTART = "unstart"
ORDER_WAITING = "waiting"
ORDER_TARGETING = "targeting"
ORDER_SERVING = "serving"
ORDER_SERVED = "served"
# 未完成的订单状态
ORDER_UNFINISHED = {ORDER_SERVING, ORDER_TARGETING, ORDER_WAITING}
# 楼状态
BD_WAITING = "waiting"
BD_SERVING = "serving"
BD_SERVED = "served"
# 每个step的时长
STEP_LEN = 1
# 历史轨迹点个数
N_HISTORY = 150  # 只保留偶数点
# 历史/预测轨迹最大长度
MAX_TRAJ_LEN = 600
# 货车到站时间
TRUCK_ARRIVE_TIME = [22598.0, 58120.0]
# 小哥异常
WARNING_STILL = "长时间驻留"
WARNING_FEW_ORDERS = "单量过少"
T_WARNING_STILL = 40 * 60     # 驻留时间异常阈值
NUM_WARNING_FEW_ORDERS = 5    # 1小时内送单量异常阈值
# aoi内移动距离
L_TODOOR_STAIR = 4
L_TODOOR_ELEVATOR = 10

#####   gen_sim_actions.py   #####

# 早上开始时间
T_DAY_START = 6 * 3600
T_PICK_ESTIMATE = 5 * 60  # 处理一个揽收单的估计耗时
T_PICK_RESERVE = 5 * 60  # 为了避免揽收超时而预留的时间

#####   通用工具函数   #####

def sub_arr(args):
    arr, idxs = args
    return [a for i, a in enumerate(arr) if i in idxs]

def print_table(columns, lines):
    def mylen(s):
        return sum(2 if '\u4e00' <= c <= '\u9fff' else 1 for c in s)
    lens = [max(12, mylen(k) + 3) for k in columns]
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

def get_colors():
    return ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

def pprint_actions(actions):
    ks_not_print = {"gps", "xy", "support_points", "target_orders", "status"}
    pprint([{k: v for k, v in a.items() if k not in ks_not_print} for a in actions])

def t_stp2str(t):
    return time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(t))

def t_str2stp(t):
    try:
        return time.mktime(time.strptime(t, "%Y-%m-%d %H:%M:%S"))
    except:
        return time.mktime(time.strptime(t, "%Y/%m/%d %H:%M:%S"))

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super().default(obj)

def json_dump(obj, file, **args):
    json.dump(obj, file, cls=NumpyEncoder, **args)

def json_dumps(obj, **args):
    return json.dumps(obj, cls=NumpyEncoder, **args)
