import pickle
from constants_mw import *


DO_SET_OUT_OFFSET = False
PARTITION_PRE_TIME = 3600
# TODO: 目前假设小哥均每天有两波工作, 其实大促期间可能不止2波, 有的小哥可能只有1波


def gen_odrs_recover(orders, actions_recover):
    """筛选出行为恢复过程中没有丢弃的订单, 并保证对floor为-1的订单的楼层数的随机生成一致"""
    print("orig order num:", [len(x) for x in orders.values()])
    orders_recover = []
    oid2odr = {o["id"]: o for x in orders.values() for o in x}
    for cid, actions in actions_recover.items():
        odrs = []
        for a in actions:
            if a["type"] in ACTION_ORDER:
                o = a["target_orders"][0]
                o_orig = oid2odr[o["id"]]
                o_orig["floor"] = o["floor"]  # 处理floor为-1的情况
                odrs.append(o_orig)
        orders_recover.append((cid, odrs))
    print("recover order num:", [len(x[1]) for x in orders_recover])
    for _, odrs in orders_recover:
        for o in odrs:
            assert o["floor"] != -1
    return orders_recover


def get_ts_arrive(actions):
    """
    小哥到路区时间: 进入第一栋楼的时间
    """
    atp_set = {ACTION_ELEVATOR, ACTION_UPSTAIR, ACTION_TODOOR} | ACTION_ORDER
    ts_arrive = []
    idxs = [i for i, a in enumerate(actions) if a["type"] == ACTION_FROMSTATION][:2]  # 最多取前两次去路区的时间
    for idx in idxs:
        for i in range(idx, len(actions)):
            if actions[i]["type"] in atp_set:
                t_arrive = actions[i]["start_time"]
                break
        else:
            assert False, "no order after FROMSTATION"
        ts_arrive.append(t_arrive)
    assert len(ts_arrive) == 2
    return ts_arrive


def gen_sim_input(orders, actions_recover):
    """根据actions_recover的到路区时间, 划分上下午订单, 生成快递员模拟模型的输入"""
    sim_model_input = []
    for cid, odrs in orders.items():
        if len(odrs) <= 0:
            continue
        actions = actions_recover[cid]
        t1, t2 = get_ts_arrive(actions)
        t_gate = t2 - PARTITION_PRE_TIME  # 以小哥第二波到路区时间前1h为界, 划分上下午两波的订单
        odrs1 = [o for o in odrs if o["finish_time"] < t_gate]  
        odrs2 = [o for o in odrs if o["finish_time"] >= t_gate]
        assert len(odrs1) > 0 and len(odrs2) > 0
        # print("morning/afternoon odrs num:", len(odrs1), len(odrs2))
        sim_model_input.append({
            "courier_id": cid,
            "start_time_morning": t1,
            "orders_morning": odrs1,
            "start_time_afternoon": t2,
            "orders_afternoon": odrs2
        })
    return sim_model_input


if __name__ == "__main__":
    # # 生成一个月数据的sim_input
    # for i in range(1, 32):
    #     actions_recover = {x[0]: x[1] for x in pickle.load(open(f"data2/actions_recover_05{i:02d}.pkl", "rb"))}
    #     orders = {x[0]: x[1] for x in pickle.load(open(f"data2/orders_05{i:02d}.pkl", "rb"))}
    #     orders_recover = {x[0]: x[1] for x in gen_odrs_recover(orders, actions_recover)}
    #     sim_imitate_input = gen_sim_input(orders_recover, actions_recover)
    #     pickle.dump(sim_imitate_input, open(f"data2/sim_imitate_input_05{i:02d}.pkl", "wb"))
    # exit()

    # 筛选出行为恢复过程中没有丢弃的订单, 保证订单对齐
    actions_recover = {x[0]: x[1] for x in pickle.load(open(f"data/actions_recover.pkl", "rb"))}
    # orders = {x[0]: x[1] for x in pickle.load(open("data/orders.pkl", "rb"))}
    # orders_recover = gen_odrs_recover(orders, actions_recover)
    # print(len(orders_recover))
    # pickle.dump(orders_recover, open("data/orders_recover.pkl", "wb"))
    orders_recover = {x[0]: x[1] for x in pickle.load(open("data/orders_recover.pkl", "rb"))}

    # 生成sim_imitate_input
    sim_imitate_input = gen_sim_input(orders_recover, actions_recover)
    pickle.dump(sim_imitate_input, open("data/sim_imitate_input.pkl", "wb"))

    # 生成sim_partition_input
    orders_partition = {x[0]: x[1] for x in pickle.load(open("data/orders_partition.pkl", "rb"))}
    sim_partition_input = gen_sim_input(orders_partition, actions_recover)
    pickle.dump(sim_partition_input, open("data/sim_partition_input.pkl", "wb"))

    # 生成sim_absent_input
    orders_absent = {x[0]: x[1] for x in pickle.load(open("data/orders_absent.pkl", "rb"))}
    sim_absent_input = gen_sim_input(orders_absent, actions_recover)
    pickle.dump(sim_absent_input, open("data/sim_absent_input.pkl", "wb"))
