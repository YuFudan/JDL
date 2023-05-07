import pickle
import time
from collections import Counter, defaultdict
from copy import deepcopy
from math import atan2, ceil

from networkx import shortest_path
from shapely.geometry import LineString, Point
from tqdm import tqdm

from constants_mxl import *
from utils import pprint_actions, time_conventer


# MODE = "recover"
MODE = "sim"


def gen_general_info(many_courier_actions, metric_status, many_courier_id):
    """场景基本信息(不随时间变化)"""
    print("gen general info")
    num_deliver = sum(actions[-1]["status"]["delivered_orders"] for actions in many_courier_actions)
    num_cpick = sum(actions[-1]["status"]["cpicked_orders"] for actions in many_courier_actions)
    num_bpick = sum(actions[-1]["status"]["bpicked_orders"] for actions in many_courier_actions)
    # 在站人数
    in_station_num = [
        {
            "step": x["step"], 
            "num": x["courier"]["in_station_num"]
        } for x in metric_status
    ]
    in_station_num.sort(key=lambda x:x["step"])
    
    # 统计历史订单配送情况
    deliver_log = []
    for actions, cid in zip(many_courier_actions, many_courier_id):
        hs = [int(o["served_time"] / 3600) for a in actions if a["type"] == ACTION_DELIVER for o in a["target_orders"]]
        hs_counter = Counter(hs)
        deliver_log.append({
            "id": cid,
            "name": cid2name[cid],
            "log": [
                {
                    "time": h,
                    "num": hs_counter[h]
                } for h in range(24)
            ]
        })
    
    # 跳转特殊时间点
    special_times = []
    # 第一个人出发去路区, 所有人到达路区
    set_out_times, arrive_times = [[], []], [[], []]
    back_set_out_times, back_arrive_times = [[], []], [[], []]
    for actions in many_courier_actions:
        actions_fs = [a for a in actions if a["type"] == ACTION_FROMSTATION][:2]  # 最多取前2个
        actions_ts = [a for a in actions if a["type"] == ACTION_TOSTATION][:2]
        for i, a in enumerate(actions_fs):
            set_out_times[i].append(a["start_time"])
            arrive_times[i].append(a["end_time"])
        for i, a in enumerate(actions_ts):
            back_set_out_times[i].append(a["start_time"])
            back_arrive_times[i].append(a["end_time"])

    if set_out_times[0]:
        special_times.append({
            "time": min(set_out_times[0]),
            "info": "第一波首个去路区"
        })
    if set_out_times[1]:
        special_times.append({
            "time": min(set_out_times[1]),
            "info": "第二波首个去路区"
        })
    if arrive_times[0]:
        special_times.append({
            "time": max(arrive_times[0]),
            "info": "第一波全体到路区"
        })
    if arrive_times[1]:
        special_times.append({
            "time": max(arrive_times[1]),
            "info": "第二波全体到路区"
        })
    if back_set_out_times[0]:
        special_times.append({
            "time": min(back_set_out_times[0]),
            "info": "第一波首个回站"
        })
    if back_set_out_times[1]:
        special_times.append({
            "time": min(back_set_out_times[1]),
            "info": "第二波首个回站"
        })
    if back_arrive_times[0]:
        special_times.append({
            "time": max(back_arrive_times[0]),
            "info": "第一波全体回站"
        })
    if back_arrive_times[1]:
        special_times.append({
            "time": max(back_arrive_times[1]),
            "info": "第二波全体回站"
        })
    special_times.sort(key=lambda x:x["time"])

    return {
        # 营业部概况
        "station": {
            "name": station_names[0],
            "position": list(stations.values())[0]["gps"],
            "area": sum(r["area"] for r in regions.values()),
            "num_zone": len(regions),
            "num_courier": len(many_courier_actions),
            "num_courier_all": NUM_COURIER_ALL
        },
        # 总任务量
        "task": {
            "delivery": num_deliver,
            "cpick": num_cpick,
            "bpick": num_bpick,
        },
        # 在快递站人数
        "in_station_num": in_station_num,
        # 投诉情况展示
        "complaint":[
            {
                "id": c["id"],
                "name": cid2name[c["id"]],
                "info": c["info"],
                "time": c["time"]
            } 
            for c in complaint_list
            if attendance_dict[c["id"]] in ["出勤", "外包"]
        ],
        # 出勤情况展示
        "attendance":{
            "normal": [cid2name[cid] for cid, s in attendance_dict.items() if s == "出勤"],
            "absent": [
                {   
                    "name": cid2name[cid],
                    "reason": s
                } for cid, s in attendance_dict.items() if s != "出勤"
            ]
        },
        # 历史订单配送情况
        "deliver_log": deliver_log,
        # 跳转特殊时间点
        "special_times": special_times,
    }


def gen_courier_status(actions, courier_id):
    """从actions的事件, 转为每步的状态"""
    print("gen courier status")
    courier_name = cid2name[courier_id]
    # energy_cum = cid2e_cum[courier_id]
    # 找出小哥一天完成的所有order
    orders = [o for a in actions if a["type"] in ACTION_ORDER for o in a["target_orders"]]
    oid2odr = {o["id"]: o for o in orders}

    # 计算初始位置
    action = actions[0]
    if action.get("gps", None):
        if "__len__" in dir(action["gps"][0]):
            position = action["gps"][0]
        else:
            position = action["gps"]
    else:
        position = buildings[orders[0]["building_id"]]["gate_gps"]
    assert len(position) == 2

    # 随着step更新, 要维护的变量
    vars_maintain = {
        "position": position,
        "traveled_length": 0.0,
        "climbed_floors": 0,
        "delivered_orders": 0,
        "delivered_on_time": 0,
        "cpicked_orders": 0,
        "cpicked_on_time": 0,
        "bpicked_orders": 0,
        "bpicked_on_time": 0,
        "work_time": 0.0,       # 计入理货拣货时间的工作时长
        "true_work_time": 0.0,  # 不计理货拣货时间的工作时长
        "between_building_length": 0.0, 
        "in_building_length": 0.0,
        "in_station_time": 0.0, 
        "out_station_time": 0.0, 
        "in_region_time": 0.0,
    }
    VARS_STA = {"position"}  # 状态量
    VARS_CUM = {             # 广延量
        "traveled_length", "climbed_floors", 
        "delivered_orders", "delivered_on_time", 
        "cpicked_orders", "cpicked_on_time",
        "bpicked_orders", "bpicked_on_time",
        "work_time", "true_work_time",
        "between_building_length", "in_building_length",
        "in_station_time", "out_station_time", "in_region_time",
    }  

    def calculate_var(act, t=None):
        """
        计算某个action从其start_time时刻到t时刻, 对状态变量产生的影响
        若t=None, 计算完整action的影响
        """
        ts, te = act["start_time"], act["end_time"]
        if t is None:
            p = 1
        else:
            assert ts <= t <= te
            p = (t - ts) / (te - ts)
        atype = act["type"]
        var = {}
        if atype in ACTION_MOVE:
            assert act["xy"], "empty move path"
            length = act["length"]
            # position
            if p == 1:
                position = act["gps"][-1]
            else:
                line = LineString(act["xy"])
                x, y = line.interpolate(p, normalized=True).coords[:][0]
                position = projector(x, y, inverse=True)
            # traveled_length
            if atype == ACTION_WALK:
                var = {"position": position, "traveled_length": length * p, "between_building_length": length * p}
            else:  # 往返路区不计移动距离
                var = {"position": position}
        elif atype == ACTION_UPSTAIR:  # 坐电梯或下楼不计爬楼层数
            var = {"climbed_floors": int(act["num"] * p)}
        elif atype == ACTION_DELIVER and p == 1:
            on_time = act["end_time"] <= act["target_orders"][0]["ddl_time"]
            var = {"delivered_orders": 1, "delivered_on_time": 1 if on_time else 0}
        elif atype == ACTION_CPICK and p == 1:
            on_time = act["end_time"] <= act["target_orders"][0]["ddl_time"]
            var = {"cpicked_orders": 1, "cpicked_on_time": 1 if on_time else 0}
        elif atype == ACTION_BPICK and p == 1:
            on_time = act["end_time"] <= act["target_orders"][0]["ddl_time"]
            var = {"bpicked_orders": 1, "bpicked_on_time": 1 if on_time else 0}
        elif atype == ACTION_TODOOR:
            length = L_TODOOR_ELEVATOR if buildings[act["building"]]["is_elevator"] else L_TODOOR_STAIR
            var = {"in_building_length": length * p}  # 建筑内移动

        # true_work_time, 只要不在站里就算工作时间
        # work_time, 理货拣货也算
        if MODE == "recover":
            if "station_id" not in act:
                # 还原的行为中, 有长时间的驻留, 可能对应小哥在休息吃饭等, 计入工作时间时*0.2
                if atype in [ACTION_REST, ACTION_DISCHARGE] and act["end_time"] - act["start_time"] > 10 * 60:
                    var["true_work_time"] = (act["end_time"] - act["start_time"]) * p * 0.2
                    var["work_time"] = (act["end_time"] - act["start_time"]) * p * 0.5  # 行为还原中, 长时间的驻留更多, 这里把比例调大, 以使工作时间相近
                else:
                    var["true_work_time"] = (act["end_time"] - act["start_time"]) * p
                    var["work_time"] = (act["end_time"] - act["start_time"]) * p
                # var["work_time"] = var["true_work_time"]
                var["out_station_time"] = var["work_time"]    # 站外工作时间
                if atype not in [ACTION_FROMSTATION, ACTION_TOSTATION]:
                    var["in_region_time"] = var["work_time"]  # 路区工作时间
            elif atype == ACTION_ARRANGE:
                var["work_time"] = (act["end_time"] - act["start_time"]) * p
                var["in_station_time"] = var["work_time"]     # 站内工作时间
        else:
            if "station_id" not in act:
                # 模拟模型中, 若派完所有件, 还有揽件没有产生, 则会一直等到揽件产生, 计入工作时间时*0.2
                if atype == ACTION_REST and act["end_time"] - act["start_time"] > 10 * 60:
                    var["true_work_time"] =(act["end_time"] - act["start_time"]) * p * 0.2
                else:
                    var["true_work_time"] = (act["end_time"] - act["start_time"]) * p
                var["work_time"] = var["true_work_time"]
                var["out_station_time"] = var["work_time"]    # 站外工作时间
                if atype not in [ACTION_FROMSTATION, ACTION_TOSTATION]:
                    var["in_region_time"] = var["work_time"]  # 路区工作时间
            elif atype == ACTION_ARRANGE:  # 模拟模型中, 小哥在站里时间全用arrange填充, 乘以一个系数以估计真的在arrange的时间
                var["work_time"] = (act["end_time"] - act["start_time"]) * p * 1 / 2  # TODO: 这个数准吗
                var["in_station_time"] = var["work_time"]     # 站内工作时间

        # # 模拟模型放大移动距离
        # if MODE != "recover":
        #     for k in ["traveled_length", "between_building_length"]:
        #         if k in var:
        #             var[k] *= 1.5

        return var

    def calculate_delta_var(var, var_last):
        """
        对于同一个action进行到不同的t产生的不同影响, 求二者的差
        """
        nonlocal VARS_STA
        return {
            k: v if k in VARS_STA else v - var_last.get(k, 0)  # 状态量取最新, 广延量作差
            for k, v in var.items()
        }
    
    def calculate_var_merge(vars):
        """
        对于多个action产生的影响, 求总影响
        其中每个action若在时间步中只执行了一部分, 其影响需要已经是求过差值后的结果
        vars需要按照对应action的时间顺序排列
        """
        nonlocal VARS_CUM
        var_merge = {}

        for k in VARS_CUM:  # 广延量直接相加
            t = sum(var.get(k, 0) for var in vars)
            if t > 0:
                var_merge[k] = t

        for var in vars[::-1]:  # 位置取最新的
            pos = var.get("position", None)
            if not pos is None:
                var_merge["position"] = pos
                break

        return var_merge

    def calculate_order_status(odr, t):
        """计算订单在t时刻的完成状态"""
        if t < odr["start_time"]:
            return ORDER_UNSTART
        elif t < odr["target_time"]:
            return ORDER_WAITING
        elif t < odr["serving_time"]:
            return ORDER_TARGETING
        elif t < odr["served_time"]:
            return ORDER_SERVING
        else:
            return ORDER_SERVED

    def predict_path(action, oid2status, oid2odr, t):
        """
        输入当前所有订单状态oid2status, 当前action, 当前时间t, 预测未来路径
        """
        atp = action["type"]

        # 移动时, 显示该段移动路径尚未完成的部分
        if atp in ACTION_MOVE:
            s_t = (t - action["start_time"]) / (action["end_time"] - action["start_time"])  # 动作完成的比例
            path_xy = action["xy"]
            line = LineString(path_xy)
            p_t = line.interpolate(s_t, normalized=True).coords[:][0]
            for i, p in enumerate(path_xy):
                s = line.project(Point(p), normalized=True)
                if s > s_t:
                    break
            return [projector(*p_t, inverse=True)] + [projector(*p, inverse=True) for p in path_xy[i:]]
        # 在楼内
        elif atp in {ACTION_ELEVATOR, ACTION_UPSTAIR, ACTION_DOWNSTAIR, ACTION_TODOOR, ACTION_DELIVER, ACTION_CPICK, ACTION_BPICK, ACTION_DISCHARGE}:
            # 起始bid
            obid = action["building"]
            # 找到最早的 处于waiting状态下的派件单 或 处于waiting状态且即将变成targeting状态的C/B揽单 所在的楼, 且不与起始楼相同的, 作为终止bid
            t_dbids = []
            for oid, status in oid2status.items():
                if (
                    status == ORDER_WAITING and (
                        oid2odr[oid]["type"] == ORDER_DELIVER or 
                        oid2odr[oid]["type"] in {ORDER_BPICK, ORDER_CPICK} and oid2odr[oid]["target_time"] - t < 10 
                    )
                ):
                    odr = oid2odr[oid]
                    dbid = odr["building_id"]
                    if dbid != obid:
                        t_dbids.append((odr["target_time"], dbid))
            if len(t_dbids) == 0:
                return []
            dbid = min(t_dbids, key=lambda x:x[0])[1]
            path_nodes = shortest_path(G, buildings[obid]["gate_id"], buildings[dbid]["gate_id"], "length")
            path_edges = [G.edges[u, v] for u, v in zip(path_nodes, path_nodes[1:])]
            path_gps = path_edges[0]["gps"] + sum(
                [x["gps"][1:] for x in path_edges[1:]], []
            )
            return path_gps
        else:
            return []

    # geojson_LineString模板
    geojson_template = { 
        "type": "Feature",
        "geometry": {
            "type": "LineString",
            "coordinates": [],
        }
    }

    # 记录所有到达路区的时间, 用于判断小哥异常
    arrive_road_ts = [a["end_time"] for a in actions if a["type"] == ACTION_FROMSTATION]
    arrive_road_ts.sort()

    # 生成每个时间步的状态
    aid = 0
    start_time = actions[0]["start_time"]
    final_time = actions[-1]["end_time"]
    courier_status = []       # 小哥状态
    courier_bd_status = []    # 楼订单状态(仅对该小哥的订单而言)
    historical_points = []    # 小哥历史轨迹
    last_x, last_y = projector(*position)
    last_direction = 0.0
    last_move_t = 1e6         # 上次发生移动的时间, 检测长时间驻留的异常小哥
    last1h_odr_ts = []        # 维护过去1h内完成的订单, 检测1h内订单完成量过少的异常小哥
    warning_cnt = 0
    for t in tqdm(range(ceil(start_time), ceil(final_time) + 1, STEP_LEN)):
        # 更新aid
        last_aid = aid
        while aid < len(actions):
            if actions[aid]["end_time"] <= t:
                aid += 1  # aid始终指向t时刻正在进行的action, 若两action恰在此相接, 指向后者
            else:
                break
        # 当前action
        current_action = actions[min(aid, len(actions)-1)] 

        # 记录t-STEP_LEN到t时间完成的action
        acts_full, act_head, act_tail, act_inter = [], None, None, None
        if ceil(start_time) < t < ceil(final_time):  
            acts_full = actions[last_aid + 1:aid]  # t-STEP_LEN未开始, t已完成
            if aid > last_aid:   
                act_tail = actions[last_aid]       # t-STEP_LEN未完成, t已完成
                act_head = actions[aid]            # t-STEP_LEN未开始, t未完成
            else:
                act_inter = actions[aid]           # t-STEP_LEN未完成, t仍未完成(act_inter或act_tail至多存在一个)
        elif t == ceil(start_time):  # 特殊处理第一个t
            acts_full = actions[:aid]              # 只要aid>0, 第一个action被完整完成
            act_head = actions[aid]                # 否则第一个action被开了个头
        elif t == ceil(final_time):  # 特殊处理最后一个t: 此时aid已经越界
            acts_full = actions[last_aid + 1:aid]
            act_tail = actions[last_aid]

        # 更新订单完成状态
        oid2status = {oid: calculate_order_status(odr, t) for oid, odr in oid2odr.items()}
        # 生成楼状态
        if t == ceil(final_time):
            courier_bd_status.append((t, {}))  # 最后一步强制所有楼都为served
        else:
            bid2ss = defaultdict(list)
            for oid, s in oid2status.items():
                bid2ss[oid2odr[oid]["building_id"]].append((s, oid2odr[oid]["type"]))  # 订单状态, 订单类型
            bid2status = {}
            for bid, ss in bid2ss.items():  # 楼中所有订单的状态
                s_set = set(x[0] for x in ss)
                if ORDER_SERVING in s_set:
                    s = BD_SERVING
                elif ORDER_WAITING in s_set or ORDER_TARGETING in s_set:
                    s = BD_WAITING
                else:
                    s = BD_SERVED
                bid2status[bid] = (s, ss)  # 楼的综合状态s, 订单的细节状态ss: [(odr_s, odr_tp)]
            courier_bd_status.append((t, bid2status))

        # 是否有C揽单处于ORDER_TARGETING或ORDER_SERVING状态
        go_for_picking = False
        for oid, s in oid2status.items():
            if oid2odr[oid]["type"] == ORDER_CPICK:
                if s == ORDER_TARGETING or s == ORDER_SERVING:
                    go_for_picking = True
                    break

        # 计算每个action对要维护的变量带来的影响
        vars = []  # 注意这个列表应按对应action的时间顺序
        if act_tail:
            vars.append(
                calculate_delta_var(
                    var=calculate_var(act_tail), 
                    var_last=calculate_var(act_tail, t - STEP_LEN)
                )
            )
        if act_inter:
            vars.append(
                calculate_delta_var(
                    var=calculate_var(act_inter, t), 
                    var_last=calculate_var(act_inter, t - STEP_LEN)
                )
            )
        if acts_full:
            vars += [calculate_var(act) for act in acts_full]
        if act_head:
            vars.append(calculate_var(act_head, t))

        # 计算action影响的总和
        var_merge = calculate_var_merge(vars)
        
        # 更新要维护的变量
        for k, v in var_merge.items():
            if k in VARS_CUM:  # 广延量直接累加
                vars_maintain[k] += v
            elif k == "position":  # 位置直接覆盖
                vars_maintain[k] = v

        # 历史轨迹
        historical_geojson = deepcopy(geojson_template)
        if t % 2 == 0:  # 降采样
            historical_points.append(vars_maintain["position"])
            historical_points = historical_points[-N_HISTORY:]
            historical_geojson["geometry"]["coordinates"] = historical_points
        else:
            historical_geojson["geometry"]["coordinates"] = historical_points + [vars_maintain["position"]]
        # TODO: 历史轨迹设置长度上限
        # xys = [projector(*p) for p in historical_points]
        # line = LineString(xys)
        # length = line.length
        # if length > MAX_TRAJ_LEN:
        #     p = line.interpolate(length - MAX_TRAJ_LEN).coords[:][0]

        # 预测轨迹
        future_points = predict_path(current_action, oid2status, oid2odr, t)
        future_geojson = deepcopy(geojson_template)
        if len(future_points) > 0:
            future_geojson["geometry"]["coordinates"] = future_points
        
        # 位置更新时, 更新方向
        if "position" in var_merge:
            x, y = vars_maintain["position"]
            direction = atan2(y - last_y, x - last_x)
            last_x, last_y = x, y
            last_direction = direction
        elif len(future_points) > 1:  # 静止, 但有预测路径时, 朝预测路径方向
            x, y = vars_maintain["position"]
            x1, y1 = future_points[1]
            direction = atan2(y1 - y, x1 - x)
            last_direction = direction
        else:
            direction = last_direction

        # 已产生的订单总量
        started = [oid2odr[oid]["type"] for oid, s in oid2status.items() if s != ORDER_UNSTART]
        counter = Counter(started)
        deliver_all = counter[ORDER_DELIVER]
        cpick_all = counter[ORDER_CPICK]
        bpick_all = counter[ORDER_BPICK]

        # 人效
        efficiency = (vars_maintain["delivered_orders"] + vars_maintain["cpicked_orders"] + vars_maintain["bpicked_orders"]) \
            / (vars_maintain["true_work_time"] + 1e-12) * 3600
        # if MODE != "recover":  # TODO: 处理人效过大的问题
        #     efficiency *= 0.7
        # else:
        #     efficiency *= 0.8
        efficiency = max(0, min(efficiency, 29.8))  # 切顶

        # 异常检测
        warnings = []
        # if MODE == "recover":
        if False:
            if "station_id" in current_action or current_action["type"] in {ACTION_FROMSTATION, ACTION_TOSTATION} or t > 75600:
                pass  # 小哥在站里,或往返路区时,或晚上9点后,不检测异常
            elif t > arrive_road_ts[0] + 3600 * 1.5:  # 初次到达路区1.2h后才检测异常
                has_unfinished_odr = False  # 小哥是否存在未完成订单
                cnt = 0
                for oid, s in oid2status.items():
                    if s in ORDER_UNFINISHED: # and oid2odr[oid]["type"] == ORDER_DELIVER:
                        cnt += 1
                        if cnt >= 5:
                            has_unfinished_odr = True
                            break
                if has_unfinished_odr:  # 存在未完成订单才检测异常
                    # 驻留异常: 长时间未移动
                    if t - last_move_t > T_WARNING_STILL:  
                        warnings.append(WARNING_STILL)
                    # 单量少异常: 到达路区满1.5h, 过去1h单量少
                    is_arrive_lately = False  # 是否刚到达路区1h内
                    for t_arrive in arrive_road_ts:
                        if 0 < t - t_arrive < 3600 * 1.5:
                            is_arrive_lately = True
                            break
                    if not is_arrive_lately and len(last1h_odr_ts) < NUM_WARNING_FEW_ORDERS:
                        warnings.append(WARNING_FEW_ORDERS)
            # 更新异常检测量
            acts_done = [act_tail] + acts_full if act_tail else acts_full  # last1h_odr_ts将是按时间顺序的
            # 上次移动时间
            if "position" in var_merge:
                last_move_t = t
            else:  # 若一直待在一栋楼里没动, 也不能算是驻留
                for a in acts_done[::-1]:
                    if a["type"] not in [ACTION_REST, ACTION_DISCHARGE]:
                        last_move_t = a["end_time"]
                        break
            # 过去1h完成单的时间
            if last1h_odr_ts:
                for i, t2 in enumerate(last1h_odr_ts):  # last1h_odr_ts将是按时间顺序的
                    if t - t2 < 3600:
                        break
                else:
                    i += 1
                last1h_odr_ts = last1h_odr_ts[i:]
            last1h_odr_ts += [a["end_time"] for a in acts_done if a["type"] in ACTION_ORDER]
            if warnings:
                # print(t, warnings, courier_name)
                warning_cnt += 1
        
        # # 做功
        # last_e = 0
        # for tt, e in energy_cum:
        #     if tt > t:
        #         break
        #     last_e = e
        # energy = last_e
        energy = 0

        # 输出结果
        courier_status.append(
            {
                "step": t,
                "id": courier_id,
                "name": courier_name,
                "position": vars_maintain["position"],
                "direction": direction,
                "action": current_action["type"],
                
                "climbed_floors": vars_maintain["climbed_floors"],
                "traveled_length": vars_maintain["traveled_length"],
                "between_building_length": vars_maintain["between_building_length"],
                "in_building_length": vars_maintain["in_building_length"],
                "deliver_all": deliver_all,
                "delivered_orders": vars_maintain["delivered_orders"],
                "delivered_on_time": vars_maintain["delivered_on_time"],
                "cpick_all": cpick_all,
                "cpicked_orders": vars_maintain["cpicked_orders"],
                "cpicked_on_time": vars_maintain["cpicked_on_time"],
                "bpick_all": bpick_all,
                "bpicked_orders": vars_maintain["bpicked_orders"],
                "bpicked_on_time": vars_maintain["bpicked_on_time"],
                "work_time": vars_maintain["work_time"],
                "true_work_time": vars_maintain["true_work_time"],
                "in_station_time": vars_maintain["in_station_time"],
                "out_station_time": vars_maintain["out_station_time"],
                "in_region_time": vars_maintain["in_region_time"],
                "energy": energy,
                "efficiency": efficiency,
                "in_station": True if "station_id" in current_action else False,

                "go_for_picking": go_for_picking,
                "historical_points": historical_geojson,
                "future_points": future_geojson,
                "warnings": warnings
            }
        )

    print("warning time:", round(warning_cnt / 60, 2), "min")

    return courier_status, courier_bd_status


def gen_metric_status(many_courier_status):
    """左侧实时指标"""
    print("gen metric status")
    # 生成指标图表状态
    t2courier_status = defaultdict(list)
    for courier_status in many_courier_status:
        for s in courier_status:
            t2courier_status[s["step"]].append(s)
    t_min, t_max = min(t2courier_status.keys()), max(t2courier_status.keys())
    for courier_status in many_courier_status:  # 小哥状态定格在最后一步
        s_copy = deepcopy(courier_status[-1])
        s_copy["in_station"] = False
        for t in range(s_copy["step"] + 1, t_max + 1):
            t2courier_status[t].append(s_copy)
    metric_status = []
    for t in tqdm(range(t_min, t_max + 1)):
        ss = t2courier_status[t]  # 在t时刻有状态的所有小哥状态, 若t时刻已经超出小哥最后一步, 取其最后一步的状态
        num_deliver = sum(s["delivered_orders"] for s in ss)
        num_cpick = sum(s["cpicked_orders"] for s in ss)
        num_bpick = sum(s["bpicked_orders"] for s in ss)
        num_total = num_deliver + num_cpick + num_bpick
        true_work_time_total = sum(s["true_work_time"] for s in ss)
        efficiency = num_total / (true_work_time_total + 1e-12) * 3600  # 单/h
        # if MODE != "recover":  # TODO: 处理人效过大的问题
        #     efficiency *= 0.7
        # else:
        #     efficiency *= 0.8
        efficiency = max(0, min(efficiency, 29.8))  # 切顶
        metric_status.append({
            "step": t,
            "task": {
                "delivery":{
                    "finished": num_deliver,
                    "on_time": sum(s["delivered_on_time"] for s in ss)
                },
                "cpick":{
                    "finished": num_cpick,
                    "on_time": sum(s["cpicked_on_time"] for s in ss)
                },
                "bpick": {
                    "finished": num_bpick,
                    "on_time": sum(s["bpicked_on_time"] for s in ss)
                }
            },
            "courier":{
                "average_traveled_length": sum(s["traveled_length"] for s in ss) / len(ss),
                "average_between_building_length": sum(s["between_building_length"] for s in ss) / len(ss),
                "average_in_building_length": sum(s["in_building_length"] for s in ss) / len(ss),
                "average_work_time": sum(s["work_time"] for s in ss) / len(ss),
                "average_in_station_time": sum(s["in_station_time"] for s in ss) / len(ss),
                "average_out_station_time": sum(s["out_station_time"] for s in ss) / len(ss),
                "average_in_region_time": sum(s["in_region_time"] for s in ss) / len(ss),
                "average_energy": sum(s["energy"] for s in ss) / len(ss),
                "average_climbed_floor": sum(s["climbed_floors"] for s in ss) / len(ss),
                "average_efficiency": efficiency,
                "in_station_num": sum(s["in_station"] for s in ss),
                "in_station_name": [s["name"] for s in ss if s["in_station"]],
                "out_station_name": [s["name"] for s in ss if not s["in_station"]],
            }
        })
    return metric_status


def gen_building_status(many_courier_bd_status, many_courier_id):
    """楼的订单完成状态"""
    print("gen building status")
    
    bd_status = []
    for courier_bd_status, cid in zip(many_courier_bd_status, many_courier_id):
        for t, bid2status in courier_bd_status:
            serving_bids = []
            deliver_bid_num = []
            cpick_bid_num = []
            bpick_bid_num = []
            for bid, (s, ss) in bid2status.items():  # 楼的综合状态s, 订单的细节状态ss: [(odr_s, odr_tp)]
                if s == BD_SERVED:
                    continue
                elif s == BD_SERVING:
                    serving_bids.append(bid)
                otp2cnt = defaultdict(int)  # 按订单类型统计该楼中该小哥未完成的单的数量
                for os, otp in ss:
                    if os in ORDER_UNFINISHED:
                        otp2cnt[otp] += 1
                for k, v in otp2cnt.items():
                    if k == ORDER_DELIVER:
                        deliver_bid_num.append({"building_id": bid, "num": v})
                    elif k == ORDER_CPICK:
                        cpick_bid_num.append({"building_id": bid, "num": v})
                    elif k == ORDER_BPICK:
                        bpick_bid_num.append({"building_id": bid, "num": v})   
            assert len(serving_bids) < 2
            bd_status.append({
                "step": t,
                "courier_id": cid,
                "serving_building_id": serving_bids[0] if len(serving_bids) > 0 else -1,
                "deliver": deliver_bid_num,
                "cpick": cpick_bid_num,
                "bpick": bpick_bid_num
            })

    return bd_status


def gen_true_data(actions, courier_status, courier_id):
    """真实订单和真实轨迹/模拟轨迹数据"""
    print("gen true data")
    orders = []
    true_points = []
    otp2chinese = {ORDER_DELIVER:"派件", ORDER_CPICK:"C揽", ORDER_BPICK:"B揽"}
    for a in actions:
        true_points += a["support_points"]
        if a["type"] in ACTION_ORDER:
            orders += a["target_orders"]
    orders = [{
        "time": o["served_time"],
        "type": otp2chinese[o["type"]],
        "address": buildings[o["building_id"]]["name"],
        "unit": o["unit"],
        "floor": o["floor"]
        } for o in orders
    ]
    orders.sort(key=lambda x:x["time"])

    true_points = [tuple(p) for p in true_points]
    true_points = sorted(list(set(true_points)), key=lambda x:x[-1])
    true_points = [{  # 真实轨迹数据
        "time": p[-1],
        "position": p[:2],
        "is_true": True,
        } for p in true_points
    ]
    sim_points = [{  # 模拟轨迹数据
        "time": s["step"],
        "position": s["position"],
        "is_true": False
        } for i, s in enumerate(courier_status)
        if i % 4 == 0  # 降采样一下
    ]
    points = true_points + sim_points
    points.sort(key=lambda x:x["time"])

    return {"id": courier_id, "trajectory": points, "order": orders}


def gen_message(many_courier_actions, many_courier_id):
    """
    各种消息, 包括:
    开早会
    投诉,
    货车到站, 小哥打卡上班
    C揽产生
    订单及时/超时完成
    """
    print("gen message")
    messages = []
    deliver_start_times = []  # 记录所有订单的收货时间, 以计算两波货车的到货量
    for actions, cid in zip(tqdm(many_courier_actions), many_courier_id):
        name = cid2name[cid]
        # 打卡上班
        for a in actions:
            if a["type"] != ACTION_REST:
                messages.append({
                    "type": "normal",
                    "id": cid,
                    "name": name,
                    "message": f"{name} 打卡上班 {time_conventer(a['start_time'])}",
                    "time": a["start_time"]
                })
                break
        # 订单产生/完成
        orders = [o for a in actions if a["type"] in ACTION_ORDER for o in a["target_orders"]]
        for o in orders:
            address = buildings[o['building_id']]['name']
            if o["type"] == ORDER_CPICK:
                msg = f"C揽产生(售后) {name} {address}" if o.get("from_sale", None) else f"C揽产生 {name} {address}"
                messages.append({
                    "type": "cpick_start",
                    "id": cid,
                    "name": name,
                    "message": msg,
                    "time": o["start_time"],
                })
                on_time = o["served_time"] <= o["ddl_time"]
                tp_en = "finish_ontime" if on_time else "finish_overtime"
                tp_chn = "C揽及时" if on_time else "C揽超时"
                msg = f"{tp_chn}(售后) {name} {address}" if o.get("from_sale", None) else f"{tp_chn} {name} {address}"
                messages.append({
                    "type": tp_en,
                    "id": cid,
                    "name": name,
                    "message": msg,
                    "time": o["served_time"]
                })
            elif o["type"] == ORDER_BPICK:
                on_time = o["served_time"] <= o["ddl_time"]
                tp_en = "finish_ontime" if on_time else "finish_overtime"
                tp_chn = "B揽及时" if on_time else "B揽超时"
                msg = f"{tp_chn}(售后) {name} {address}" if o.get("from_sale", None) else f"{tp_chn} {name} {address}"
                messages.append({
                    "type": tp_en,
                    "id": cid,
                    "name": name,
                    "message": msg,
                    "time": o["served_time"]
                })
            else:
                on_time = o["served_time"] <= o["ddl_time"]
                tp_en = "finish_ontime" if on_time else "finish_overtime"
                tp_chn = "妥投及时" if on_time else "妥投超时"
                messages.append({
                    "type": tp_en,
                    "id": cid,
                    "name": name,
                    "message": f"{tp_chn} {name} {address}",
                    "time": o["served_time"]
                })
                deliver_start_times.append(o["start_time"])
    # 货车到站
    wave2num = {i: 0 for i in range(1, len(TRUCK_ARRIVE_TIME) + 1)}
    for t in deliver_start_times:  # 计算每波的货量
        for i, t2 in enumerate(TRUCK_ARRIVE_TIME[1:]):
            if t < t2:
                wave2num[i+1] += 1
                break
        else:
            wave2num[len(TRUCK_ARRIVE_TIME)] += 1
    wave = 0
    for t in TRUCK_ARRIVE_TIME:
        wave += 1
        num = wave2num[wave]
        t_conv = time_conventer(t)
        for pre in [30, 10, 5, 3, 1]:
            messages.append({
                "type": "normal",
                "id": -1,
                "name": "非人",
                "message": f"第{wave}波货车预计还有{pre}分钟到站, 预计到站时间{t_conv}, 到站货量{num}",
                "time": t - pre * 60
            })
        messages.append({
            "type": "normal",
            "id": -1,
            "name": "非人",
            "message": f"第{wave}波货车到站, 到站时间{t_conv}, 到站货量{num}",
            "time": t
        })
    # 开早会
    messages.append({
        "type": "normal",
        "id": -1,
        "name": "非人",
        "message": "早会开始",
        "time": T_MEETING_START
    })
    messages.append({
        "type": "normal",
        "id": -1,
        "name": "非人",
        "message": "早会结束",
        "time": T_MEETING_END
    })
    # 投诉
    for c in complaint_list:
        messages.append({
            "type": "complaint",
            "id": c["id"],
            "name": cid2name[c["id"]],
            "message": f"投诉 {cid2name[c['id']]} {c['info']}",
            "time": c["time"]
        })
    messages.sort(key=lambda x:x["time"])
    return messages


def gen_work_detail(many_courier_status):
    """
    小哥个人的工作指标
    """
    print("gen courier work detail")
    t2courier_status = defaultdict(list)
    for courier_status in many_courier_status:
        for s in courier_status:
            t2courier_status[s["step"]].append(s)
    t_min, t_max = min(t2courier_status.keys()), max(t2courier_status.keys())
    for courier_status in many_courier_status:
        last_s = courier_status[-1]
        for t in range(last_s["step"] + 1, t_max + 1):  # 填充之后的状态: 定格在最后一步
            t2courier_status[t].append(last_s)
        first_s = deepcopy(courier_status[0])
        for k in [
            "deliver_all", "delivered_orders", "delivered_on_time",
            "cpick_all", "cpicked_orders", "cpicked_on_time",
            "bpick_all", "bpicked_orders", "bpicked_on_time",
            "climbed_floors", "traveled_length", "efficiency",
            "between_building_length", "in_building_length",
            "in_station_time", "out_station_time", "in_region_time", "energy"]:
            first_s[k] = 0
        for t in range(t_min, first_s["step"]):  # 填充之前的状态: 0
            t2courier_status[t].append(first_s)

    work_details = []
    absent_cids =  [k for k, v in attendance_dict.items() if v == "病假" or v == "缺勤"]

    for t in tqdm(range(t_min, t_max + 1)):
        ss = t2courier_status[t]
        assert len(ss) == len(many_courier_status)
        
        order_stat = [{
            "id": s["id"],
            "name": s["name"], 

            "deliver_ontime": s["delivered_on_time"],
            "deliver_overtime": s["delivered_orders"] - s["delivered_on_time"],
            "deliver_all": s["deliver_all"],

            "bpick_ontime": s["bpicked_on_time"],
            "bpick_overtime": s["bpicked_orders"] - s["bpicked_on_time"],
            "bpick_all": s["bpick_all"],

            "cpick_ontime": s["cpicked_on_time"],
            "cpick_overtime": s["cpicked_orders"] - s["cpicked_on_time"],
            "cpick_all": s["cpick_all"],
        } for s in ss]
        order_stat += [{
            "id": cid,
            "name": cid2name[cid], 

            "deliver_ontime": 0,
            "deliver_overtime": 0,
            "deliver_all": 0,

            "bpick_ontime": 0,
            "bpick_overtime": 0,
            "bpick_all": 0,

            "cpick_ontime": 0,
            "cpick_overtime": 0,
            "cpick_all": 0,
        } for cid in absent_cids]
        order_stat.sort(key=lambda x:x["deliver_ontime"], reverse=True)

        climb_stat = [{
            "id": s["id"],
            "name": s["name"],
            "num": s["climbed_floors"],
        } for s in ss]
        climb_stat += [{
            "id": cid,
            "name": cid2name[cid],
            "num": 0,
        } for cid in absent_cids]
        climb_stat.sort(key=lambda x:x["num"], reverse=True)

        travel_stat = [{
            "id": s["id"],
            "name": s["name"],
            "length": s["traveled_length"],
        } for s in ss]
        travel_stat += [{
            "id": cid,
            "name": cid2name[cid],
            "length": 0,
        } for cid in absent_cids]
        travel_stat.sort(key=lambda x:x["length"], reverse=True)

        work_stat = [{
            "id": s["id"],
            "name": s["name"],
            "efficiency": s["efficiency"],
            "attendance": attendance_dict[s["id"]],
            "complaint": [c["info"] for c in complaint_list if c["id"] == s["id"]],
            "between_building_length": s["between_building_length"],
            "in_building_length": s["in_building_length"],
            "in_station_time": s["in_station_time"],
            "out_station_time": s["out_station_time"],
            "in_region_time": s["in_region_time"],
            "energy": s["energy"],
        } for s in ss]
        work_stat += [{
            "id": cid,
            "name": cid2name[cid],
            "efficiency": 0,
            "attendance": attendance_dict[cid],
            "complaint": [],
            "between_building_length": 0,
            "in_building_length": 0,
            "in_station_time": 0,
            "out_station_time": 0,
            "in_region_time": 0,
            "energy": 0,
        } for cid in absent_cids]
        work_stat.sort(key=lambda x:x["efficiency"], reverse=True)

        work_details.append({
            "step": t,
            "order": order_stat,
            "climb": climb_stat,
            "travel": travel_stat,
            "work_stat": work_stat,
        })
    
    return work_details


def write_db(col_prefix, general_info, courier_status, bd_status, metric_status, true_data, message, work_detail):
    """写入MongoDB"""
    from pymongo import MongoClient
    client = MongoClient("")
    db = client["demo_jd"]

    # info
    print("ouput general info")
    col = db[f"{col_prefix}_info"]
    col.drop()
    col.insert_one(general_info)
    
    # courier
    print("output courier status:", len(courier_status))
    col = db[f"{col_prefix}_courier"]
    col.drop()
    col.create_index([("step", 1), ("id", 1)])  # 建立索引
    keys_todel = [
        "deliver_all", "delivered_on_time", 
        "cpick_all", "cpicked_on_time", 
        "bpick_all", "bpicked_on_time", 
        "work_time", "true_work_time", "efficiency",
        "between_building_length", "in_building_length",
        "in_station_time", "out_station_time", "in_region_time", "energy"
    ]
    for x in tqdm(courier_status):
        for k in keys_todel:
            del x[k]
        if x["action"] == ACTION_REST or x["action"] == ACTION_DISCHARGE:  # 休息和卸货显示为驻留
            x["action"] = "驻留"
    col.insert_many(courier_status, ordered=False)
    
    # node_status
    print("output bd status:", len(bd_status))
    col = db[f"{col_prefix}_node_status"]
    col.drop()
    col.create_index([("step", 1), ("courier_id", 1)])  # 建立索引
    col.insert_many(bd_status, ordered=False)
    
    # node
    bd_geojson = [
        {
            "type": "Feature",
            "properties": {"id": bid, "name": bd["name"]},
            "geometry": {"type": "Point", "coordinates": bd["gate_gps"]},
        } for bid, bd in buildings.items()
    ]
    print("output bd node geojson:", len(bd_geojson))
    col = db[f"{col_prefix}_node"]
    col.drop()
    col.insert_many(bd_geojson, ordered=False)
    
    # building
    bd_geojson = [
        {
            "type": "Feature",
            "properties": {"id": bid, "name": bd["name"]},
            "geometry": {"type": "Polygon", "coordinates": [bd["points"]]},
        } for bid, bd in buildings.items()
    ]
    print("output bd geojson:", len(bd_geojson))
    col = db[f"{col_prefix}_building"]
    col.drop()
    col.insert_many(bd_geojson, ordered=False)
    
    # stat
    print("output metric status:", len(metric_status))
    col = db[f"{col_prefix}_stat"]
    col.drop()
    col.create_index([("step", 1)])  # 建立索引
    col.insert_many(metric_status, ordered=False)

    # # event(deprecated)
    # col = db[f"{col_prefix}_event"]
    # col.drop()
    # # col.insert_many(events, ordered=False)
    # # print("output events:", len(events))

    # true_data
    if true_data:
        print("output true data:", len(true_data))
        col = db[f"{col_prefix}_true_data"]
        col.drop()
        col.insert_many(true_data, ordered=False)
        
    # message
    print("output message:", len(message))
    col = db[f"{col_prefix}_message"]
    col.drop()
    col.insert_many(message, ordered=False)
    
    # courier_work_detail
    print("output work detail:", len(work_detail))
    col = db[f"{col_prefix}_courier_work_detail"]
    col.drop()
    col.create_index([("step", 1)])  # 建立索引
    col.insert_many(work_detail, ordered=False)
    
    # metadata
    col = db["metadata"]
    for x in col.find():
        if x["name"] == col_prefix:
            col.update_one(
                {"name": col_prefix}, {"$set": {"start": metric_status[0]["step"], "steps": len(metric_status)}}
            )
            break
    else:
        col.insert_one(
            {"name": col_prefix, "start": metric_status[0]["step"], "steps": len(metric_status), "ready": True}
        )

    return 


def main(many_courier_actions, many_courier_id, col_prefix, has_true_data=True):
    many_courier_status = []
    many_courier_bd_status = []
    many_courier_true_data = []
    for actions, courier_id in zip(many_courier_actions, many_courier_id):
        print(courier_id, cid2name[courier_id])
        courier_status, courier_bd_status = gen_courier_status(actions, courier_id)
        many_courier_status.append(courier_status)
        many_courier_bd_status.append(courier_bd_status)
        if has_true_data:
            true_data = gen_true_data(actions, courier_status, courier_id)
            many_courier_true_data.append(true_data)

    bd_status = gen_building_status(many_courier_bd_status, many_courier_id)

    metric_status = gen_metric_status(many_courier_status)

    general_info = gen_general_info(many_courier_actions, metric_status, many_courier_id)

    message = gen_message(many_courier_actions, many_courier_id)

    work_detail = gen_work_detail(many_courier_status)

    start_time = time.time()
    write_db(
        col_prefix=col_prefix,
        general_info=general_info,
        courier_status=sum(many_courier_status, []), 
        bd_status=bd_status, 
        metric_status=metric_status, 
        true_data=many_courier_true_data,
        message=message,
        work_detail=work_detail,
    )
    print("write DB time:", round((time.time() - start_time) / 60, 1), "min")

    return


def process_for_stall(G, buildings, cid2stall_info):
    STALL_BID_OFFSET = 1e6

    # 添加新bd新node
    nodes_new = []
    buildings_new = {}
    for cid, (bid2p_lid, lid2gps_time) in cid2stall_info.items():
        for lid, (gps, time) in lid2gps_time.items():
            bid = int(STALL_BID_OFFSET + lid)
            xy = projector(*gps)
            x, y = xy
            xys = [(x-1, y-1), (x+1, y-1), (x+1, y+1), (x-1, y+1), (x-1, y-1)]
            gpss = [projector(*p, inverse=True) for p in xys]
            # 找出楼所在的路区
            p = Point(xy)
            dis_ids = []
            for r in regions.values():
                dis = p.distance(r["poly"])
                if dis == 0:
                    rid = r["id"]
                    break
                dis_ids.append((dis, r["id"]))
            else:
                rid = min(dis_ids, key=lambda x:x[0])[1]
            
            buildings_new[bid] = {
                "id": bid,
                "gate_id": str(bid),
                "name": "摆摊点",
                'is_elevator': False,
                "floor": 7,
                "points": gpss,
                "poly": Polygon(xys),
                "gate_gps": gps,
                "gate_xy": xy,
                "point": Point(xy),
                "region": rid,
                "stall_time": time
            }
            nodes_new.append((str(bid), {
                "id": str(bid),
                "gps": gps,
                "xy": xy
            }))

    # 连接到最近的node, 添加新edge
    edges_new = []
    nodes = {nid: n for nid, n in G.nodes(data=True)}
    for nid, node in nodes_new:
        x, y = node["xy"]
        dis_n1s = []
        for nid1, n1 in nodes.items():
            x1, y1 = n1["xy"]
            gps1 = n1["gps"]
            dis2 = (x - x1) ** 2 + (y - y1) ** 2
            if dis2 < 100:
                edges_new.append({
                    "od": (nid, nid1),
                    "points": [node["gps"], gps1] 
                })
                break
            else:
                dis_n1s.append((dis2, n1))
        else:
            dis2, n1 = min(dis_n1s, key=lambda x: x[0])
            # print("nearset dis2:", dis2)
            edges_new.append({
                "od": (nid, n1["id"]),
                "points": [node["gps"], n1["gps"]] 
            })
    edges_new = edges_new + [
        {
            "od": (road["od"][1], road["od"][0]),
            "points": road["points"][::-1]
        } for road in edges_new
    ]
    edges_new = [
        (
            road["od"][0],
            road["od"][1],
            {
                "od": road["od"],
                "gps": road["points"],
                "xy": [projector(*p) for p in road["points"]],
            }
        ) for road in edges_new
    ]
    for e in edges_new:
        e[2]["length"] = LineString(e[2]["xy"]).length
    G.add_nodes_from(nodes_new)
    G.add_edges_from(edges_new)
    buildings.update(buildings_new)
    
    return G, buildings


if __name__ == "__main__":
    # data = pickle.load(open(f"data_mxl/actions_recover.pkl", "rb"))
    # col_prefix = "demo5"  # 漫香林1101行为还原
    # has_true_data = True

    # data = pickle.load(open(f"data_mxl/actions_sim_bayes_828.pkl", "rb"))
    # col_prefix = "demo6"  # 漫香林0828行为模拟
    # has_true_data = False

    MODE = "sim"
    data = pickle.load(open(f"data_mxl/actions_valid_gt.pkl", "rb"))
    col_prefix = "demo7"  # 漫香林 某小哥某天 行为模拟真值(指定真实送单顺序)
    has_true_data = False

    # MODE = "sim"
    # data = pickle.load(open(f"data_mxl/actions_valid_sim.pkl", "rb"))
    # col_prefix = "demo8"  # 漫香林 某小哥某天 行为模拟
    # has_true_data = False

    cid2stall_info = pickle.load(open("validate/data/cid2stall_info_mxl.pkl", "rb"))
    G, buildings = process_for_stall(G, buildings, cid2stall_info)

    # # 生成 楼aoi 和 楼点位 geojson
    # import json
    # buildings = pickle.load(open("data_mxl/buildings.pkl", "rb"))
    # bd_geojson = [
    #     {
    #         "type": "Feature",
    #         "properties": {k: bd[k] for k in ["id", "name", "is_elevator", "floor"] if k in bd},
    #         "geometry": {"type": "Polygon", "coordinates": [bd["points"]]},
    #     } for bd in buildings
    # ]
    # json.dump(bd_geojson, open("data_mxl/buildings_geojson.json", "w"))
    # node_geojson = [
    #     {
    #         "type": "Feature",
    #         "properties": {"id": bd["id"], "name": bd["name"]},
    #         "geometry": {"type": "Point", "coordinates": bd["gate_gps"]},
    #     } for bd in buildings
    # ]
    # json.dump(node_geojson, open("data_mxl/nodes_geojson.json", "w"))
    # exit()
    
    many_courier_actions = [x[1] for x in data]
    many_courier_id = [int(x[0]) for x in data]
    print("actions:")
    print(len(data))
    print([len(x) for x in many_courier_actions])

    many_courier_actions_filtered = []
    many_courier_id_filtered = []
    for actions, cid in zip(many_courier_actions, many_courier_id):
        if len(actions) > 0:
            for a in actions:
                if a["type"] in ACTION_ORDER:
                    many_courier_actions_filtered.append(actions)
                    many_courier_id_filtered.append(cid)
                    break
    print("actions_filtered:")
    print(len(many_courier_actions_filtered))
    print([len(x) for x in many_courier_actions_filtered])
    print([(cid, attendance_dict[cid]) for cid in many_courier_id_filtered])

    # # test
    # many_courier_actions_filtered = [many_courier_actions_filtered[0]]
    # many_courier_id_filtered = [many_courier_id_filtered[0]]

    main(
        many_courier_actions=many_courier_actions_filtered, 
        many_courier_id=many_courier_id_filtered, 
        col_prefix=col_prefix, 
        has_true_data=has_true_data,
    )
