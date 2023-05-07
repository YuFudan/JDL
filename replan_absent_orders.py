import pickle
import random
from collections import defaultdict
from copy import deepcopy

import networkx as nx
import numpy as np
from tqdm import tqdm

from lkh import LKH


def flood(arr, x):
    """
    将x的水量填入水位为arr的水桶中，返回每个桶应该填多少
    """
    ret = [0] * len(arr)
    l = defaultdict(list)
    for i, j in enumerate(arr):
        l[j].append(i)
    while True:
        if len(l) == 1:
            ind = list(l.values())[0]
            p, q = divmod(x, len(ind))
            for i in ind:
                ret[i] += p
            for i in ind[:q]:
                ret[i] += 1
            return ret
        m1 = min(l)
        m2 = min(set(l) - {m1})
        dm = m2 - m1
        ind = l[m1]
        if x <= dm * len(ind):
            p, q = divmod(x, len(ind))
            for i in ind:
                ret[i] += p
            for i in ind[:q]:
                ret[i] += 1
            return ret
        for i in ind:
            ret[i] += dm
        x -= dm * len(ind)
        l[m2] += l[m1]
        l.pop(m1)


def eval_vrp(dist, vrp):
    if vrp:
        vrp = [0] + vrp
        solver = LKH()
        return list(
            solver.solve_acvrp(
                dist[np.ix_(vrp, vrp)], [1] * (len(vrp) - 1), 1e5, 1, scale=1e3
            )
        )[-1].cost
    else:
        return 0


class Replan:
    def __init__(self, G, buildings):
        self.building2node = {i["id"]: i["gate_id"] for i in buildings}
        self.distance = {
            i: j for i, j in nx.all_pairs_bellman_ford_path_length(G, weight="length")
        }

    def replan(
        self, couriers, remove_index, num_trials=1, use_tqdm=True, tqdm_leave=True
    ):
        """
        规划移除某个快递员移除后的分配方案

        couriers:       list of lists，订单列表
        remove_index:   待移除的快递员序号
        """
        for i in couriers:
            i.sort(key=lambda x: (x["finish_time"], x["building_id"], x["type"]))

        # 把要移除的放在最后
        couriers[remove_index], couriers[-1] = couriers[-1], couriers[remove_index]

        depot = "快递站"
        id2node = [depot] + sorted(
            {self.building2node[j["building_id"]] for i in couriers for j in i}
        )
        node2id = {j: i for i, j in enumerate(id2node)}
        vrps = [
            [node2id[k] for k in {self.building2node[j["building_id"]] for j in i}]
            for i in couriers
        ]
        tasks = []
        for i in couriers:
            task = defaultdict(list)
            for j in i:
                task[node2id[self.building2node[j["building_id"]]]].append(j)
            tasks.append(task)
        burden = [sum(len(i) for i in i.values()) for i in tasks]
        dist = np.empty((len(node2id),) * 2)
        for a, b in node2id.items():
            for c, d in node2id.items():
                dist[b, d] = self.distance[a][c]

        # 第一步：如果快递员将要前往的地点也被其他快递员经过，则公平分配这些订单
        remain = []
        vrp = vrps[-1]
        vrps = vrps[:-1]
        for i in vrp:
            others = []
            for j, k in enumerate(vrps):
                if i in k:
                    others.append(j)
            if others:
                task = tasks[-1][i]
                plan = flood([burden[j] for j in others], len(task))
                p = 0
                for a, b in zip(others, plan):
                    if b:
                        tasks[a][i] += task[p : p + b]
                        p += b
            else:
                remain.append(i)
        tasks[-1] = {i: tasks[-1][i] for i in remain}

        # 第二步：将剩余订单插入给其他快递员
        if remain:
            costs = [eval_vrp(dist, i) for i in vrps]
            backup = (vrps, costs)
            best_cost = 1e999
            best_vrp = None
            with tqdm(
                total=len(remain) * num_trials,
                dynamic_ncols=True,
                disable=not use_tqdm,
                leave=tqdm_leave,
            ) as bar:
                for _ in range(num_trials):
                    vrps, costs = deepcopy(backup)
                    random.shuffle(remain)
                    for i in remain:
                        best = 1e999
                        best_j = -1
                        for j, (r, c) in enumerate(zip(vrps, costs)):
                            c_ = eval_vrp(dist, r + [i]) - c
                            if c_ < best:
                                best = c_
                                best_j = j
                        vrps[best_j].append(i)
                        costs[best_j] += best
                        bar.update(1)
                    if sum(costs) < best_cost:
                        best_cost = sum(costs)
                        best_vrp = vrps
            vrps = best_vrp

        task = tasks[-1]
        output = []
        for i, j in zip(vrps, tasks):
            o = []
            for k in i:
                o += j.get(k, task.get(k))
            output.append(o)
        assert sum(len(i) for i in output) == sum(len(i) for i in couriers)
        return (
            output[:remove_index]
            + [[]]
            + output[remove_index + 1 :]
            + output[remove_index : remove_index + 1]
        )


if __name__ == "__main__":
    G = pickle.load(open("data/G_new.pkl", "rb"))
    buildings = pickle.load(open("data/buildings_new.pkl", "rb"))
    orders_recover = [x for x in pickle.load(open("data/orders_recover.pkl", "rb")) if len(x[1]) > 0]
    cids = [x[0] for x in orders_recover]
    print(len(orders_recover))
    print([len(x[1]) for x in orders_recover])
    print(sum(len(x[1]) for x in orders_recover))

    replan = Replan(G, buildings)
    result = replan.replan([x[1] for x in orders_recover], 1)  # 1为梁永福
    assert len(result[1]) == 0
    print(len(result))
    print([len(x) for x in result])
    print(sum(len(x) for x in result))

    pickle.dump(
        [(cid, odrs) for cid, odrs in zip(cids, result) if len(odrs) > 0], open("data/orders_absent.pkl", "wb")
    )
