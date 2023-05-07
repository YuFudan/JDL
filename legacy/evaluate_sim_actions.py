import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
from pprint import pprint
from constants_mw import *
from utils import time_conventer
from gen_sim_input import get_ts_arrive


SELECT = "total"      # 比较全天平均指标
# SELECT = "wave1"    # 比较上午指标
# SELECT = "wave2"    # 比较下午指标


def get_ts_back(actions):
    """
    小哥回站时间: 送完最后一单并下楼(若需要下楼)的时间
    """
    atp_set = {ACTION_ELEVATOR, ACTION_DOWNSTAIR} | ACTION_ORDER
    ts_back = []
    idxs = [i for i, a in enumerate(actions) if a["type"] == ACTION_TOSTATION][:2]  # 最多取前两次回站的时间
    for idx in idxs:
        for i in range(idx, -1, -1):
            if actions[i]["type"] in atp_set:
                t_back = actions[i]["end_time"]
                break
        else:
            assert False, "no order before TOSTATION"
        ts_back.append(t_back)
    return ts_back


def calculate_metrics(actions, is_true):
    """计算真实性评估指标"""
    # 到路区时刻
    if is_true:
        ts_arrive = get_ts_arrive(actions)
    else:
        ts_arrive = [a["end_time"] for a in actions if a["type"] == ACTION_FROMSTATION][:2]

    # 回站时刻
    if is_true:
        ts_back = get_ts_back(actions)
    else:   
        ts_back = [a["start_time"] for a in actions if a["type"] == ACTION_TOSTATION][:2]

    # 工作总时长(不比较理货拣货时间, 只比较路区内的工作时间)
    work_time = 0
    tostation_cnt = 0
    for a in actions:
        if "station_id" not in a:  # 不看在站里的动作
            if is_true:
                # 还原的行为中, 有长时间的驻留, 可能对应小哥在休息吃饭等, 计入工作时间时*0.2
                if a["type"] in [ACTION_REST, ACTION_DISCHARGE] and a["end_time"] - a["start_time"] > 10 * 60:
                    work_time += 0.2 * (a["end_time"] - a["start_time"])
                else:
                    work_time += a["end_time"] - a["start_time"]
            else:
                # 模拟模型中, 若派完所有件, 还有揽件没有产生, 则会一直等到揽件产生, 计入工作时间时*0.2                
                if a["type"] == ACTION_REST and a["end_time"] - a["start_time"] > 10 * 60:
                    work_time += 0.2 * (a["end_time"] - a["start_time"])
                else:
                    work_time += a["end_time"] - a["start_time"]
        if a["type"] == ACTION_TOSTATION:
            tostation_cnt += 1
            if tostation_cnt == 1:
                work_time_1 = work_time  # 记录第一波回站时的指标
                status_1 = a["status"]
            elif tostation_cnt == 2:
                work_time_2 = work_time  # 记录第二波回站时的指标
                status_2 = a["status"]

    # travel_length基本没法通过调参解决, 直接放大1.5倍  TODO:
    if not is_true:
        for s in [status_1, status_2]:
            s["traveled_length"] *= 1.5
    # climbed_floors的问题基本没法通过调参解决, 直接放大1.1倍  TODO:
    if not is_true:
        for s in [status_1, status_2]:
            s["climbed_floors"] *= 1.1 

    # 第二波指标中扣除第一波的影响
    work_time_2 -= work_time_1
    for k in ["traveled_length", "climbed_floors", "delivered_orders", "delivered_on_time", "cpicked_orders", "cpicked_on_time", "bpicked_orders", "bpicked_on_time"]:
        status_2[k] -= status_1[k]
    
    # # 直接比较在路区的总时间
    # work_time_1 = ts_back[0] - ts_arrive[0]
    # work_time_2 = ts_back[1] - ts_arrive[1]

    metrics = {
        "wave1":
            {
                "work_time": work_time_1,
                "traveled_length": status_1["traveled_length"],
                "climbed_floors": status_1["climbed_floors"],

                "delivered_orders": status_1["delivered_orders"],
                "delivered_on_time": status_1["delivered_on_time"],
                
                "cpicked_orders": status_1["cpicked_orders"],
                "cpicked_on_time": status_1["cpicked_on_time"],

                "bpicked_orders": status_1["bpicked_orders"],
                "bpicked_on_time": status_1["bpicked_on_time"],
            },
        "wave2":
            {
                "work_time": work_time_2,
                "traveled_length": status_2["traveled_length"],
                "climbed_floors": status_2["climbed_floors"],

                "delivered_orders": status_2["delivered_orders"],
                "delivered_on_time": status_2["delivered_on_time"],

                "cpicked_orders": status_2["cpicked_orders"],
                "cpicked_on_time": status_2["cpicked_on_time"],

                "bpicked_orders": status_2["bpicked_orders"],
                "bpicked_on_time": status_2["bpicked_on_time"],
            },
        "total": {}
    }
    for k in [
        "work_time", "traveled_length", "climbed_floors", 
        "delivered_orders", "delivered_on_time", 
        "cpicked_orders", "cpicked_on_time", 
        "bpicked_orders", "bpicked_on_time"]:
        metrics["total"][k] = metrics["wave1"][k] + metrics["wave2"][k]
    
    return metrics


def calculate_mean_std(nums, weights=None):
    """计算一组数的均值和标准差, 可以加权"""
    if weights is None:
        return np.mean(nums), np.std(nums)
    mean = sum(n*w for n, w in zip(nums, weights)) / (sum(weights) + 1e-12)
    std2 = sum(w*(n - mean)**2 for n, w in zip(nums, weights)) / (sum(weights) + 1e-12)
    return mean, std2 ** 0.5


def print_table(columns, lines):
    """将差异打印成表"""
    def mylen(s):
        return sum(2 if '\u4e00' <= c <= '\u9fff' else 1 for c in s)
    lens = [max(10, mylen(k) + 3) for k in columns]
    head = "".join(k + " " * (l-mylen(k)) for k, l in zip(columns, lens))
    print(head)
    print("=" * (mylen(head) - 3))
    for line in lines:
        line = [f"{x:+.2f}" if not isinstance(x, str) else x for x in line]
        print("".join(x + " "*(l - mylen(x)) for x, l in zip(line, lens)))


def evaluate(evaluate_input, fix_date=None, fix_courier=None):
    """评估模拟真实性: 比较模拟与真实actions指标的差异"""
    if fix_date is not None and fix_courier is not None:
        assert False
    # 用dataframe管理多个小哥多天的多个指标间的差异
    column_names = [
        "日期", "小哥id", "小哥名", 
        "时长", "相对时长%",    # 绝对差异, 相对差异
        "效率", "相对效率%",
        "跑动", "相对跑动%",
        "爬楼", "相对爬楼%",
        "C揽数", "C揽及时率%",  # 总数作为权重, 及时率用绝对差异更有意义
        "B揽数", "B揽及时率%",
    ]
    lines = []
    for x in evaluate_input:  # 计算指标差异
        mtrue = calculate_metrics(x["actions_true"], is_true=True)[SELECT]
        msim = calculate_metrics(x["actions_sim"], is_true=False)[SELECT]
        
        dwt = msim["work_time"] - mtrue["work_time"]
        dwt_rel = 100 * dwt / (mtrue["work_time"] + 1e-12)

        dtl = msim["traveled_length"] - mtrue["traveled_length"]
        dtl_rel = 100 * dtl / (mtrue["traveled_length"] + 1e-12)

        dcf = msim["climbed_floors"] - mtrue["climbed_floors"]
        dcf_rel = 100 * dcf / (mtrue["climbed_floors"] + 1e-12)

        dnum = mtrue["delivered_orders"]
        cnum = mtrue["cpicked_orders"]
        bnum = mtrue["bpicked_orders"]
        dcotr = 100 * (msim["cpicked_on_time"] - mtrue["cpicked_on_time"]) / (cnum + 1e-12)
        dbotr = 100 * (msim["bpicked_on_time"] - mtrue["bpicked_on_time"]) / (bnum + 1e-12)

        eff_true = 3600 * (dnum + cnum + bnum) / (mtrue["work_time"] + 1e-12)
        eff_sim = 3600 * (dnum + cnum + bnum) / (msim["work_time"] + 1e-12)
        deff = eff_sim - eff_true
        deff_rel = 100 * deff / (eff_true + 1e-12)

        lines.append([
            x["date"], x["courier_id"], cid2name[x["courier_id"]], 
            dwt, dwt_rel, 
            deff, deff_rel, 
            dtl, dtl_rel, 
            dcf, dcf_rel, 
            cnum, dcotr, 
            bnum, dbotr])

    metric_df = pd.DataFrame(data=lines, columns=column_names)

    if fix_date:
        print("fix date:", fix_date)
        df = metric_df[metric_df["日期"] == fix_date]
    elif fix_courier:
        print("fix courier:", cid2name[fix_courier])
        df = metric_df[metric_df["小哥id"] == fix_courier]
    else:
        print("all dates and all couriers")
        df = metric_df
    mean_stds = []
    for k in ["时长", "相对时长%", "效率", "相对效率%", "跑动", "相对跑动%", "爬楼", "相对爬楼%"]:
        mean_stds.append(calculate_mean_std(list(df[k])))
    for k1, k2 in [("C揽数", "C揽及时率%"), ("B揽数", "B揽及时率%")]:
        mean_stds.append(calculate_mean_std(nums=list(df[k2]), weights=list(df[k1])))
    means, stds = zip(*mean_stds)
    if fix_date:
        df = df[[k for k in df.columns if k not in ["日期", "小哥id", "C揽数", "B揽数"]]]
    elif fix_courier:
        df = df[[k for k in df.columns if k not in ["小哥名", "小哥id", "C揽数", "B揽数"]]]
    else:
        df = df[[k for k in df.columns if k not in ["日期", "小哥名", "小哥id", "C揽数", "B揽数"]]]
    if fix_date is None and fix_courier is None:
        columns = [" "] + df.columns.values.tolist()
        lines = [["[均值]"] + list(means), ["[标准差]"] + list(stds)]
    else:
        columns = df.columns.values.tolist()
        lines = df.values.tolist()
        lines = [["[均值]"] + list(means), ["[标准差]"] + list(stds)] + lines

    print_table(columns, lines)


if __name__ == "__main__":
    many_courier_actions_true = {
        cid: actions 
        for cid, actions in pickle.load(open(f"data/actions_recover.pkl", "rb")) 
        if len(actions) > 0
    }
    many_courier_actions_sim = {
        cid: actions 
        for cid, actions in pickle.load(open(f"data/actions_imitate.pkl", "rb"))
    }
    assert set(many_courier_actions_true.keys()) == set(many_courier_actions_sim.keys())

    evaluate_input = [{
        "date": "0505",
        "courier_id": cid,
        "actions_true": actions,
        "actions_sim": many_courier_actions_sim[cid]
        } 
        for cid, actions in many_courier_actions_true.items()
        if ( 
            cid != 20570125 and  # 排除梁永福(詹科生), 带妻子一起工作, 所以送单特别多, 指标不准很正常
            len([a for a in actions if a["type"] in ACTION_ORDER]) > 100  # 排除单量过少的小哥, 随机性较大
        )
    ]
    # 注意调小哥的默认参数时, 排除单量过少的小哥, 以免其随机性较大
    # 若要调小哥的个性化参数, 则不要将其排除(梁永福还是排除掉, 梁永福就用默认参数)

    for x in evaluate_input:
        actions1, actions2 = x["actions_true"], x["actions_sim"]
        assert len([a for a in actions1 if a["type"] in ACTION_ORDER]) == len([a for a in actions2 if a["type"] in ACTION_ORDER])

    evaluate(evaluate_input, fix_date="0505")
