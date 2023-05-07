import subprocess
import numpy as np
import re
from collections import namedtuple
import os

Log = namedtuple("Log", ["trial", "penalty", "cost", "time", "solution"])


class LKH:
    def __init__(self, *, trial=1000, seed=43, lkh_path="./LKH"):
        self.trial = trial
        self.seed = seed
        self.lkh_path = lkh_path

    def solve_acvrpspdtw(
        self,
        dist_mat,
        time_mat,
        demand,
        capacity,
        n_vehicle,
        *,
        l_opt_limit=6,
        scale=1,
        trial=-1,
        seed=-1,
        log_only=False,
        always=False,
        single_courier=False,
        debug=False,
    ):
        """
        求解ACVRPSPDTW问题
        ---
        #### 注意：LKH只接受整数(int64)数据输入，本函数会对输入乘scale后转为整数

        dist_mat: 距离矩阵，形状为`NxN`，0为depot

        time_mat: 时间矩阵

        demand: 需求矩阵，形状为`Nx5`，`[pickup, delivery, start, end, service]`

        n_vehicle: 车辆数上限

        l_opt_limit: 做λ-opt的λ上限，取值[2,3,4,5,6]

        scale: 缩放系数

        trial: 尝试次数，为正则覆盖默认值

        seed: 种子，0表示随机

        log_only: 仅记录求解过程中的代价和时间，不记录解

        always: 总是输出解，包括不可行解(penalty!=0)

        single_courier: 指明实际上由单个快递员完成，回到仓库时间不会重置
        """
        if trial <= 0:
            trial = self.trial
        if seed < 0:
            seed = self.seed
        dist_mat = (np.array(dist_mat) * scale).astype(int)
        time_mat = (np.array(time_mat) * scale).astype(int)
        demand = (np.array(demand) * scale).astype(int)
        assert demand.shape[0] == dist_mat.shape[0] == time_mat.shape[0]
        n_vehicle = max(1, min(demand.shape[0] - 1, n_vehicle))
        capacity = int(capacity * scale)
        always = always and not log_only
        if type(l_opt_limit) is not int or not 2 <= l_opt_limit <= 6:
            raise ValueError()
        f_par = f"""SPECIAL
PROBLEM_FILE = -
MAX_TRIALS = {trial}
{'ALWAYS_WRITE_OUTPUT' if always else ''}
RUNS = 1
POPULATION_SIZE = 10
TRACE_LEVEL = 1
{'WRITE_SOLUTION_TO_LOG' if not log_only else ''}
{'SINGLE_COURIER' if single_courier else ''}
L_OPT_LIMIT = {l_opt_limit}
SEED = {seed}
$$$
"""
        f_vrp = f"""NAME : xxx
TYPE : VRPSPDTW
DIMENSION : {dist_mat.shape[0]}
EDGE_WEIGHT_TYPE : EXPLICIT
EDGE_WEIGHT_FORMAT : FULL_MATRIX
CAPACITY : {capacity}
VEHICLES : {n_vehicle}
EDGE_WEIGHT_SECTION
"""
        f_vrp += "\n".join(" ".join(str(j) for j in i) for i in dist_mat) + "\n"
        f_vrp += "EDGE_TIME_SECTION\n"
        f_vrp += "\n".join(" ".join(str(j) for j in i) for i in time_mat) + "\n"
        f_vrp += "PICKUP_AND_DELIVERY_SECTION\n"
        for i, (p, d, a, b, c) in enumerate(demand):
            f_vrp += f"{i+1} 0 {a} {b} {c} {p} {d}\n"
        f_vrp += "DEPOT_SECTION\n1\nEOF\n$$$\n"
        last = None
        proc = subprocess.Popen(
            f"stdbuf -oL {self.lkh_path} -",
            shell=True,
            encoding="utf8",
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
        )
        proc.stdin.write(f_par)
        proc.stdin.write(f_vrp)
        proc.stdin.flush()
        try:
            for l in proc.stdout:
                if debug:
                    print(l.rstrip())
                if log_only:
                    ret = re.findall(
                        r"^\* (.+?): Cost = (.+?)_(.+?), Time = (.+?) sec. ", l
                    )
                    if ret:
                        i, p, c, t = ret[0]
                        yield Log(
                            int(i), float(p) / scale, float(c) / scale, float(t), None
                        )
                else:
                    if l.startswith("Solution:"):
                        if last[0] == "*":
                            i, p, c, t = re.findall(
                                r"\* (.+?): Cost = (.+?)_(.+?), Time = (.+?) sec. ",
                                last,
                            )[0]
                            sol = [int(k) for k in l.split(" ", 1)[1].split(",")]
                            yield Log(
                                int(i),
                                float(p) / scale,
                                float(c) / scale,
                                float(t),
                                sol,
                            )
                    last = l
        finally:
            proc.kill()
            proc.wait()

    def solve_acvrp(
        self,
        dist_mat,
        demand,
        capacity,
        n_vehicle,
        *,
        l_opt_limit=6,
        scale=1,
        trial=-1,
        seed=-1,
        log_only=False,
        always=False,
    ):
        """
        求解ACVRP问题
        ---
        #### 注意：LKH只接受整数(int64)数据输入，本函数会对输入乘scale后转为整数

        dist_mat: 距离矩阵，形状为`NxN`，0为depot

        demand: 需求矩阵，形状为`N-1`，即默认depot为0

        n_vehicle: 车辆数上限

        l_opt_limit: 做λ-opt的λ上限，取值[2,3,4,5,6]

        scale: 缩放系数

        trial: 尝试次数，为正则覆盖默认值

        seed: 种子，0表示随机

        log_only: 仅记录求解过程中的代价和时间，不记录解

        always: 总是输出解，包括不可行解(penalty!=0)
        """
        if trial <= 0:
            trial = self.trial
        if seed < 0:
            seed = self.seed
        dist_mat = (np.array(dist_mat) * scale).astype(int)
        demand = (np.array(demand) * scale).astype(int).reshape(-1)
        assert demand.shape[0] + 1 == dist_mat.shape[0]
        n_vehicle = max(1, min(demand.shape[0], n_vehicle))
        capacity = int(capacity * scale)
        always = always and not log_only
        if type(l_opt_limit) is not int or not 2 <= l_opt_limit <= 6:
            raise ValueError()
        f_par = f"""SPECIAL
PROBLEM_FILE = -
MAX_TRIALS = {trial}
{'ALWAYS_WRITE_OUTPUT' if always else ''}
RUNS = 1
POPULATION_SIZE = 10
TRACE_LEVEL = 1
{'WRITE_SOLUTION_TO_LOG' if not log_only else ''}
L_OPT_LIMIT = {l_opt_limit}
SEED = {seed}
$$$
"""
        f_vrp = f"""NAME : xxx
TYPE : ACVRP
DIMENSION : {dist_mat.shape[0]}
EDGE_WEIGHT_TYPE : EXPLICIT
EDGE_WEIGHT_FORMAT : FULL_MATRIX
CAPACITY : {capacity}
VEHICLES : {n_vehicle}
EDGE_WEIGHT_SECTION
"""
        f_vrp += "\n".join(" ".join(str(j) for j in i) for i in dist_mat) + "\n"
        f_vrp += "DEMAND_SECTION\n"
        for i, d in enumerate([0] + demand.tolist()):
            f_vrp += f"{i+1} {d}\n"
        f_vrp += "DEPOT_SECTION\n1\nEOF\n$$$\n"
        last = None
        proc = subprocess.Popen(
            f"stdbuf -oL {self.lkh_path} -",
            shell=True,
            encoding="utf8",
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
        )
        proc.stdin.write(f_par)
        proc.stdin.write(f_vrp)
        proc.stdin.flush()
        try:
            for l in proc.stdout:
                if log_only:
                    ret = re.findall(
                        r"^\* (.+?): Cost = (.+?)_(.+?), Time = (.+?) sec. ", l
                    )
                    if ret:
                        i, p, c, t = ret[0]
                        yield Log(
                            int(i), float(p) / scale, float(c) / scale, float(t), None
                        )
                else:
                    if l.startswith("Solution:"):
                        if last[0] == "*":
                            i, p, c, t = re.findall(
                                r"\* (.+?): Cost = (.+?)_(.+?), Time = (.+?) sec. ",
                                last,
                            )[0]
                            sol = [int(k) for k in l.split(" ", 1)[1].split(",")]
                            yield Log(
                                int(i),
                                float(p) / scale,
                                float(c) / scale,
                                float(t),
                                sol,
                            )
                    last = l
        finally:
            proc.kill()
            proc.wait()

    def solve_pdptw(
        self,
        dist_mat,
        demand,
        capacity,
        n_vehicle,
        *,
        l_opt_limit=6,
        scale=1,
        trial=-1,
        seed=-1,
        log_only=False,
        always=False,
    ):
        """
        求解PDPTW问题
        ---
        #### 注意：LKH只接受整数(int64)数据输入，本函数会对输入乘scale后转为整数

        dist_mat: 距离矩阵，形状为`NxN`，0为depot

        demand:
         - 需求矩阵，形状为`Nx6`
         - `[±demand, from_id, to_id, start, end, service]`
         - depot只有end不为0
         - pickup点demand为正，from_id填0，to_id填要送往的delivery点
         - delivery点demand为负，from_id填来自的pickup点，to_id填0

        n_vehicle: 车辆数上限

        l_opt_limit: 做λ-opt的λ上限，取值[2,3,4,5,6]

        scale: 缩放系数

        trial: 尝试次数，为正则覆盖默认值

        seed: 种子，0表示随机

        log_only: 仅记录求解过程中的代价和时间，不记录解

        always: 总是输出解，包括不可行解(penalty!=0)
        """
        if trial <= 0:
            trial = self.trial
        if seed < 0:
            seed = self.seed
        dist_mat = (np.array(dist_mat) * scale).astype(int)
        demand = np.array(demand)
        demand[:, [0, 3, 4, 5]] *= scale
        demand = demand.astype(int)
        assert demand.shape[0] == dist_mat.shape[0]
        n_vehicle = max(1, min(demand.shape[0] - 1, n_vehicle))
        capacity = int(capacity * scale)
        always = always and not log_only
        if type(l_opt_limit) is not int or not 2 <= l_opt_limit <= 6:
            raise ValueError()
        f_par = f"""SPECIAL
PROBLEM_FILE = -
MAX_TRIALS = {trial}
{'ALWAYS_WRITE_OUTPUT' if always else ''}
RUNS = 1
POPULATION_SIZE = 10
TRACE_LEVEL = 1
{'WRITE_SOLUTION_TO_LOG' if not log_only else ''}
L_OPT_LIMIT = {l_opt_limit}
SEED = {seed}
$$$
"""
        f_vrp = f"""NAME : xxx
TYPE : PDPTW
DIMENSION : {dist_mat.shape[0]}
EDGE_WEIGHT_TYPE : EXPLICIT
EDGE_WEIGHT_FORMAT : FULL_MATRIX
CAPACITY : {capacity}
VEHICLES : {n_vehicle}
EDGE_WEIGHT_SECTION
"""
        f_vrp += "\n".join(" ".join(str(j) for j in i) for i in dist_mat) + "\n"
        f_vrp += "PICKUP_AND_DELIVERY_SECTION\n"
        for i, (d, f, t, a, b, c) in enumerate(demand):
            if i == 0:
                assert d == f == t == a == c == 0
            assert (f == 0) if d > 0 else (t == 0)
            f_vrp += (
                f"{i+1} {d} {a} {b} {c} {0 if f==0 else f+1} {0 if t==0 else t+1}\n"
            )
        f_vrp += "DEPOT_SECTION\n1\nEOF\n$$$\n"
        last = None
        proc = subprocess.Popen(
            f"stdbuf -oL {self.lkh_path} -",
            shell=True,
            encoding="utf8",
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
        )
        proc.stdin.write(f_par)
        proc.stdin.write(f_vrp)
        proc.stdin.flush()
        try:
            for l in proc.stdout:
                if log_only:
                    ret = re.findall(
                        r"^\* (.+?): Cost = (.+?)_(.+?), Time = (.+?) sec. ", l
                    )
                    if ret:
                        i, p, c, t = ret[0]
                        yield Log(
                            int(i), float(p) / scale, float(c) / scale, float(t), None
                        )
                else:
                    if l.startswith("Solution:"):
                        if last[0] == "*":
                            i, p, c, t = re.findall(
                                r"\* (.+?): Cost = (.+?)_(.+?), Time = (.+?) sec. ",
                                last,
                            )[0]
                            sol = [int(k) for k in l.split(" ", 1)[1].split(",")]
                            yield Log(
                                int(i),
                                float(p) / scale,
                                float(c) / scale,
                                float(t),
                                sol,
                            )
                    last = l
        finally:
            proc.kill()
            proc.wait()


def test_acvrpspdtw():
    if os.path.exists("test/200_0.vrp"):
        ls = [i.strip() for i in open("test/200_0.vrp")]
        dim = int(ls[2].rsplit(" ", 1)[1])
        cap = int(ls[5].rsplit(" ", 1)[1])
        n_v = int(ls[6].rsplit(" ", 1)[1])
        dist_mat = np.array([[int(j) for j in i.split()] for i in ls[8 : 8 + dim]])
        time_mat = np.array(
            [[int(j) for j in i.split()] for i in ls[9 + dim : 9 + 2 * dim]]
        )
        demand = np.array(
            [[int(j) for j in i.split()] for i in ls[10 + 2 * dim : 10 + 3 * dim]]
        )
        demand = demand[:, [5, 6, 2, 3, 4]]
    else:
        dim = 21
        cap = 5
        n_v = 5
        dist_mat = np.random.rand(dim, dim) * 10000
        time_mat = np.random.rand(dim, dim) * 10000
        demand = np.array([[0, 0, 0, 1000, 0]] + [[1, 0, 0, 1000, 1]] * (dim - 1))
    lkh = LKH()
    for i in lkh.solve_acvrpspdtw(dist_mat, time_mat, demand, cap, n_v, always=True):
        print(i)


def test_acvrp():
    if os.path.exists("test/200_0.vrp"):
        ls = [i.strip() for i in open("test/200_0.vrp")]
        dim = int(ls[2].rsplit(" ", 1)[1])
        cap = int(ls[5].rsplit(" ", 1)[1])
        n_v = int(ls[6].rsplit(" ", 1)[1])
        dist_mat = np.array([[int(j) for j in i.split()] for i in ls[8 : 8 + dim]])
        demand = np.array(
            [[int(j) for j in i.split()] for i in ls[10 + 2 * dim : 10 + 3 * dim]]
        )
        demand = demand[:, [5, 6, 2, 3, 4]]
    else:
        dim = 21
        cap = 5
        n_v = 5
        dist_mat = np.random.rand(dim, dim) * 10000
        demand = np.array([[0, 0, 0, 1000, 0]] + [[1, 0, 0, 1000, 1]] * (dim - 1))
    lkh = LKH()
    for i in lkh.solve_acvrp(dist_mat, demand[1:, 0], cap, n_v, always=True):
        print(i)


def test_pdptw(n=20):
    dim = 2 * n + 1
    n_v = 5
    cap = n // n_v + 1
    dist_mat = np.random.rand(dim, dim) * 1000
    demand = [[0, 0, 0, 0, 1000, 0]] + sum(
        (
            [[1, 0, 2 * i + 2, 0, 1000, 0], [-1, 2 * i + 1, 0, 0, 1000, 0]]
            for i in range(n)
        ),
        [],
    )
    lkh = LKH()
    for i in lkh.solve_pdptw(dist_mat, demand, cap, n_v, always=True):
        print(i)


if __name__ == "__main__":
    test_acvrp()
    test_acvrpspdtw()
    test_pdptw()
