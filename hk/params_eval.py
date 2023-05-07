"""evaluate.py和auto_bayes_tune.py的参数"""
from hyperopt import hp

IGNORE_CIDS = {}
# IGNORE_CIDS = {
#     22969590,  # 张松
#     22196342,  # 杨浩
#     22600383,  # 谢建如
#     22119540,  # 卢春明
#     22784062,  # 崔瑞广
#     21952368,  # 党朋菲
#     20840560,  # 陈武峰
#     20707123,  # 许鹏杰
#     21744833,  # 刘云鹏
#     20864736,  # 杨广胜
#     22476612,  # 武嘉豪
#     21529043,  # 陈桢
#     22946772,  # 姚慧佳
# }

MIN_TRAIN_WAVE_NUM = 5      # 过滤训练数据波数过少的小哥
T_LONG_REST = 600           # 去除非工作的驻留时长
T_STALL = 1800              # 摆摊时长阈值
N_STALL = 6                 # 摆摊单数阈值
STALL_LOC_D = 50            # 摆摊位置合并阈值
STALL_BID_OFFSET = 1e6      # 从摆摊location_id映射到bid
MAX_V_TRAVEL = 4            # 计算真值移动距离时, 最大移动速度

TRAIN_DATES = [f"2023-02-{i:02d}" for i in range(1, 29)]
TEST_DATES = [f"2023-03-{i:02d}" for i in range(1, 15)]

# SEQ_TYPE = "lkh"
# SEQ_TYPE = "stat"
SEQ_TYPE = "nn"

P_STAT2LKH = 0.4

CACHE_SEQ_NN = "log/SeqModelNN_230420_211216/pt/979.pt"

# bayes调参搜索空间
space = {
    't_elevator_wait1': hp.uniform('t_elevator_wait1', 10.0, 25.0),
    # 't_elevator_wait2': hp.uniform('t_elevator_wait2', 5.0, 15.0),  # 假设为wait1的一半

    # 't_stair_up': hp.uniform('t_stair_up', 14.0, 28.0),
    # 't_stair_down': hp.uniform('t_stair_down', 7.0, 14.0),
    't_stair': hp.uniform('t_stair', 20.0, 45.0),    # 上楼数总是等于下楼数, 简并成1个参数即可

    # "t_todoor": hp.uniform('t_todoor', 3.0, 8.0),  # 每次送单前总是有1次todoor, 则不用调, 调了送单时间相当于调了这个
    't_deliver': hp.uniform('t_deliver', 20.0, 120.0),
    't_bpick': hp.uniform('t_bpick', 20.0, 120.0),
    't_cpick': hp.uniform('t_cpick', 20.0, 100.0),
    "t_between_units": hp.uniform('t_between_units', 5.0, 20.0),
    # "t_rest_per_package": hp.uniform('t_rest_per_package', 5.0, 30.0),  # 简并到每单的时间中

    "v_walk": hp.uniform('v_walk', 0.9, 5.0),
    # "walk_type": hp.choice("walk_type", [
    #     {"walk_type": "walk", "v_walk": hp.uniform("v_walk_walk", 1.2, 1.4)},
    #     {"walk_type": "drive", "v_walk": hp.uniform("v_walk_drive", 10, 15)},
    #     ]),

    # "t_cpick_reserve": hp.loguniform('t_cpick_reserve', np.log(100), np.log(1800)),
    # "t_bpick_reserve": hp.loguniform('t_bpick_reserve', np.log(100), np.log(1800)),
    "t_cpick_reserve": hp.uniform('t_cpick_reserve', 1800, 7200),
    "t_bpick_reserve": hp.uniform('t_bpick_reserve', 1800, 7200),

    "t_stall_multiplier": hp.uniform('t_stall_multiplier', 0.8, 1.2),  # 放缩摆摊每单用时历史统计值
    "motivation_multiplier": hp.uniform('motivation_multiplier', 0.5, 1.5),  # 放缩小哥积极性参数历史统计值
}
