"""调试时, 使用python seq_model.py --cuda 2 --train-ratio 0.8 --debug"""
import argparse
import os
import pickle
import random
import time
from copy import deepcopy

import numpy as np
import torch
from setproctitle import setproctitle
from torch import nn
# from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from constants_top import *

F_DIST, F_ND, F_NB, F_TB, F_NC, F_TC = range(6)  # 特征顺序

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=233, help="seed of the experiment")
    parser.add_argument("--cuda", type=int, default=1)
    parser.add_argument("--infer-cuda", type=int, default=-1)
    parser.add_argument("--epoch", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument("--train-ratio", type=float, default=1)
    parser.add_argument("--dim-courier", type=int, default=8, help="dimension of courier embedding")
    parser.add_argument("--dim-building", type=int, default=16, help="dimension of building embedding")
    parser.add_argument("--mlp", type=str, default="64,64", help="MLP structure")
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


class DefaultArgs:
    def __init__(self):
        self.seed = 233
        self.cuda = 1
        self.infer_cuda = -1
        self.epoch = 1000
        self.batch_size = 1000
        self.train_ratio = 1
        self.dim_courier = 8
        self.dim_building = 16
        self.mlp = "64,64"
        self.debug = False


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def make_mlp(*shape, dropout=0.1, act=nn.Tanh, sigma=0):
    ls = [nn.Linear(i, j) for i, j in zip(shape, shape[1:])]
    if sigma > 0:
        for l in ls:
            nn.init.orthogonal_(l.weight, 2**0.5)
            nn.init.constant_(l.bias, 0)
        nn.init.orthogonal_(ls[-1].weight, sigma)
    return nn.Sequential(
        *sum(
            (
                [
                    l,
                    act(),
                    nn.Dropout(dropout),
                ]
                for l in ls[:-1]
            ),
            [],
        ),
        ls[-1],
    )


class Model(nn.Module):
    def __init__(self, num_courier, dim_courier, num_building, dim_building, dim_feature, dim_mlp):
        super().__init__()
        self.ff = make_mlp(
            dim_courier + dim_building * 2 + dim_feature,  # 输入维度
            *dim_mlp,                                      # 中间维度
            1,                                             # 输出维度
            act=nn.ReLU,
        )
        self.emb_courier = nn.Embedding(num_courier, dim_courier)
        self.emb_building = nn.Embedding(num_building, dim_building)
        self.num_building = num_building

    def forward(self, xs):
        cid, bid, ft = zip(*xs)  # 展开batch
        bid = torch.stack(bid)
        x = torch.hstack(
            [
                self.emb_courier(torch.stack(cid)).repeat_interleave(
                    self.num_building, 0
                ),  # repeat用于给每个楼复制相同的cid和bid特征
                self.emb_building(bid).repeat_interleave(self.num_building, 0),
                self.emb_building.weight.repeat(len(xs), 1),
                torch.vstack(ft),
            ]
        )  # (building_num*batch), (D_emb+D_feature)
        y = self.ff(x).view(len(xs), -1)
        with torch.no_grad():
            y[torch.arange(len(y), device=bid.device), bid] = -torch.inf  # 避免预测结果仍为当前楼
        return y


class SeqModelNN:
    def __init__(self, dist, cid2wodrs, args=DefaultArgs(), cache_dataset=None):
        """
        dist: 楼间的路网距离 {bid1: {bid2: 100m}}
        cid2wodrs: 每个小哥多天多波的订单 {cid: [[o1, o2, ...], [o1, o2, ...], ...]}
        """
        self.args = args
        set_seed(args.seed)
        device = torch.device(f"cuda:{args.cuda}" if args.cuda >= 0 else "cpu")
        self.device = device
        self.infer_device = torch.device(f"cuda:{args.infer_cuda}" if args.infer_cuda >= 0 else "cpu")

        cids = sorted(cid2wodrs)
        cid_map = {j: i for i, j in enumerate(cids)}
        self.cid_map = cid_map
        bids = sorted(dist)
        bid_map = {j: i for i, j in enumerate(bids)}
        self.bid_map = bid_map
        self.bid_map_rev = {v: k for k, v in bid_map.items()}

        dist_mapped = np.zeros((len(bids), len(bids)))
        for i, a in enumerate(bids):
            d = dist[a]
            for j, b in enumerate(bids):
                dist_mapped[i,j] = d[b]
        dist = dist_mapped / 200  # 减小数值scale
        del dist_mapped
        self.dist = dist
        
        # 每栋候选楼的状态特征模板
        xs = [[0]*6 for _ in range(len(bids))]      
        for i in range(len(bids)):
            xs[i][F_TB] = xs[i][F_TC] = 1440   # 其中最早的B/C揽还有多久超时(默认值填86400/60)
        self.xs_template = xs

        def make_dataset(odrs):
            """输入一波订单, 输出小哥每次做去下一栋楼的决策时 对应的楼状态特征"""
            seq = [[None, None, None]]
            for i, o in enumerate(odrs):
                bid = bid_map[o['building_id']]
                if bid != seq[-1][0]:
                    seq.append([bid, o["finish_time"], i])
                else:
                    seq[-1][1:] = o["finish_time"], i  # 小哥做出去下一栋楼的决定的时间, 为楼中最后一单的完成时间
            seq = seq[1:]

            dataset = []
            for (bid, t, idx), (nbid, *_) in zip(seq, seq[1:]):
                assert bid != nbid  # next_bid
                xs = deepcopy(self.xs_template)
                for i in range(len(bids)):
                    xs[i][F_DIST] = dist[bid, i]      # 与当前楼的距离
                for o in odrs[idx+1:]:
                    x = xs[bid_map[o['building_id']]]
                    if o['type'] == ORDER_DELIVER:
                        x[F_ND] += 1                  # 尚待完成的派送单数
                    elif o['type'] == ORDER_BPICK and o['start_time'] <= t:
                        x[F_NB] += 1                  # 尚待完成的已产生B揽单数
                        x[F_TB] = min((o['ddl_time'] - t) / 60, x[F_TB])  # B揽超时还有多久
                    elif o['type'] == ORDER_CPICK and o['start_time'] <= t:  
                        x[F_NC] += 1
                        x[F_TC] = min((o['ddl_time'] - t) / 60, x[F_TC])
                dataset.append([bid, nbid, np.array(xs, dtype=np.float32)])
            return dataset
        
        if cache_dataset and os.path.exists(cache_dataset):
            print("Use cached dataset:", cache_dataset)
            dataset = pickle.load(open(cache_dataset, "rb"))
        else:
            print("Making dataset...")
            dataset = [
                [cid_map[cid], make_dataset(odrs)] 
                for cid, wodrs in tqdm(cid2wodrs.items())
                for odrs in wodrs]
            dataset = [[cid, *data] for cid, datas in dataset for data in datas]
            # 每个data为: cid, bid, nbid_bid(记录真值,不用于预测输入), xs(所有楼的状态)
            if cache_dataset:
                pickle.dump(dataset, open(cache_dataset, "wb"))
        print("dataset:", len(dataset))
        self.dataset = dataset

    def train(self, cache_model=None):
        if cache_model is not None:
            print("Use cached seq nn:", cache_model)
            model = Model(
                num_courier=len({i[0] for i in self.dataset}),
                dim_courier=self.args.dim_courier,
                num_building=self.dataset[0][3].shape[0],
                dim_building=self.args.dim_building,
                dim_feature=self.dataset[0][3].shape[1],
                dim_mlp=[int(i) for i in self.args.mlp.split(",")],
            ).to(self.infer_device)
            model.load_state_dict(torch.load(cache_model))
            self.model_best = model
            return
            
        EARLY_STOP = 100
        run_name = time.strftime("SeqModelNN_%y%m%d_%H%M%S")
        setproctitle(f"{run_name}@yufudan")
        print(f"Run: {run_name}")
        
        if self.args.debug:
            # print(f"tensorboard --port 8888 --logdir log/{run_name}")
            # writer = SummaryWriter(f"log/{run_name}")
            # writer.add_text(
            #     "hyperparameters",
            #     "|param|value|\n|-|-|\n%s"
            #     % ("\n".join([f"|{key}|{value}|" for key, value in vars(self.args).items()])),
            # )
            pass
        else:
            os.makedirs(f"log/{run_name}/pt")
        
        model = Model(
            num_courier=len({i[0] for i in self.dataset}),
            dim_courier=self.args.dim_courier,
            num_building=self.dataset[0][3].shape[0],
            dim_building=self.args.dim_building,
            dim_feature=self.dataset[0][3].shape[1],
            dim_mlp=[int(i) for i in self.args.mlp.split(",")],
        ).to(self.device)

        dataset = [
            [
                [
                    torch.tensor(c, dtype=torch.long, device=self.device),
                    torch.tensor(b, dtype=torch.long, device=self.device),
                    torch.tensor(x, dtype=torch.float, device=self.device),
                ],
                torch.tensor(n, dtype=torch.long, device=self.device),
            ]
            for c, b, n, x in self.dataset
        ]
        random.shuffle(dataset)
        t = int(len(dataset) * self.args.train_ratio)
        train_set = dataset[:t]
        test_set = dataset[t:]
        batch_size = min(len(train_set), self.args.batch_size)

        opt = torch.optim.Adam(model.parameters())

        criterion = nn.CrossEntropyLoss()

        test_loss = test_acc = 0
        best_acc = -1
        best_epoch  = -1
        best_acc_saved = -1
        best_epoch_saved  = -1
        with tqdm(range(self.args.epoch), dynamic_ncols=True) as bar: 
            for epoch in bar:
                random.shuffle(train_set)
                train_loss = train_acc = 0
                n = 0
                for i in range(0, len(train_set) // batch_size * batch_size, batch_size):
                    x, y = zip(*train_set[i : i + batch_size])
                    y = torch.stack(y)
                    y_pred = model(x)
                    loss = criterion(y_pred, y)
                    opt.zero_grad()
                    loss.backward()
                    opt.step()

                    train_loss += loss.detach().mean().cpu().item()
                    train_acc += (y_pred.argmax(1).detach() == y).float().mean()
                    n += 1
                    bar.set_description(
                        f"{train_loss/n:.4f} {train_acc/n*100:.2f}% {test_loss:.4f} {test_acc*100:.2f}%"
                    )
                if test_set:
                    with torch.no_grad():
                        x, y = zip(*test_set)
                        y = torch.stack(y)
                        y_pred = model(x)
                        test_loss = criterion(y_pred, y)
                        test_acc = (y_pred.argmax(1).detach() == y).float().mean()
                else:
                    test_loss = train_loss / n
                    test_acc = train_acc / n
                
                if self.args.debug:
                    # writer.add_scalar("fig/train_loss", train_loss / n, epoch)
                    # writer.add_scalar("fig/test_loss", test_loss, epoch)
                    # writer.add_scalars(
                    #         "fig/acc",
                    #         {
                    #             "train": train_acc / n,
                    #             "test": test_acc,
                    #         },
                    #         epoch,
                    # )
                    pass
                elif epoch % 10 == 9:  
                    if test_acc > best_acc_saved:
                        best_acc_saved = test_acc
                        best_epoch_saved = epoch
                    torch.save(
                        model.state_dict(),
                        f"log/{run_name}/pt/{epoch}.pt",
                    )
                
                # early stop
                if test_acc > best_acc:
                    best_acc = test_acc
                    best_epoch = epoch
                if epoch - best_epoch > EARLY_STOP:
                    print(f"Stop since last {EARLY_STOP} epochs no gain")
                    print("best test acc:", best_acc, "best epoch:", best_epoch)
                    break

        if not self.args.debug:
            print("best test acc saved:", best_acc_saved, "best epoch saved:", best_epoch_saved)
            model = Model(
                num_courier=len({i[0] for i in self.dataset}),
                dim_courier=self.args.dim_courier,
                num_building=self.dataset[0][3].shape[0],
                dim_building=self.args.dim_building,
                dim_feature=self.dataset[0][3].shape[1],
                dim_mlp=[int(i) for i in self.args.mlp.split(",")],
            ).to(self.infer_device)
            model.load_state_dict(torch.load(f"log/{run_name}/pt/{best_epoch_saved}.pt"))
            self.model_best = model

    def infer(self, odrs, cid, bid, t, candidates=None, top5=False):
        """
        t时刻, 小哥cid在bid处, 其尚未完成的订单为odrs, 决定下一栋楼去哪
        candidates: 若给定, 则在其中选得分最高的, 否则在所有楼中选得分最高的
        """
        bid_map = self.bid_map
        cid = self.cid_map[cid]
        bid = bid_map[bid]
        dist = self.dist
        model = self.model_best
        
        xs = deepcopy(self.xs_template)
        for i in range(len(bid_map)):
            xs[i][F_DIST] = dist[bid, i]      # 与当前楼的距离
        for o in odrs:
            x = xs[bid_map[o['building_id']]]
            if o['type'] == ORDER_DELIVER:
                x[F_ND] += 1
            elif o['type'] == ORDER_BPICK and o['start_time'] <= t:
                x[F_NB] += 1
                x[F_TB] = min((o['ddl_time'] - t) / 60, x[F_TB])
            elif o['type'] == ORDER_CPICK and o['start_time'] <= t:  
                x[F_NC] += 1
                x[F_TC] = min((o['ddl_time'] - t) / 60, x[F_TC])

        with torch.no_grad():
            ys = model([[
                torch.tensor(cid, dtype=torch.long, device=self.infer_device),
                torch.tensor(bid, dtype=torch.long, device=self.infer_device),
                torch.tensor(xs, dtype=torch.float, device=self.infer_device),
            ]]).squeeze()
            if top5:  # 评估性能用
                top5 = []
                for i, _ in sorted(enumerate(ys.tolist()), key=lambda a: -a[1]):
                    x = xs[i]
                    if x[F_ND] > 0 or x[F_NB] > 0 or x[F_NC] > 0:
                        top5.append(i)
                        if len(top5) == 5:
                            break
                return [self.bid_map_rev[i] for i in top5]
            if candidates is not None:
                i = max([bid_map[cbid] for cbid in candidates], key=lambda a: ys[a])
                x = xs[i]
                assert x[F_ND] > 0 or x[F_NB] > 0 or x[F_NC] > 0
            else:
                # i = ys.argmax().int().item()
                for i, _ in sorted(enumerate(ys.tolist()), key=lambda a: -a[1]):
                    x = xs[i]
                    if x[F_ND] > 0 or x[F_NB] > 0 or x[F_NC] > 0:
                        break
                else:
                    assert False
        return self.bid_map_rev[i]


if __name__ == "__main__":
    from collections import defaultdict

    from evaluate import add_stall_to_map

    DPT = "hk"  # 指定营业部department
    if DPT == "mxl":
        from mxl.constants_all import *
        from mxl.params_eval import *
    elif DPT == "hk":
        from hk.constants_all import *
        from hk.params_eval import *

    cache = f"{DPT}/data/eval_datas_{len(TRAIN_DATES)}_{len(TEST_DATES)}.pkl"
    train_data, test_data, cid2stall_info, bellman_ford = pickle.load(open(cache, "rb"))
    _, buildings = add_stall_to_map(G, buildings, cid2stall_info)

    train_cid2wodrs, test_cid2wodrs = [
        {
            cid: [w["orders"] for w in waves] 
            for cid, waves in data.items()
        } for data in (train_data, test_data)
    ]
    
    dist = defaultdict(dict)
    for bid1, b1 in buildings.items():
        d = dist[bid1]
        db = bellman_ford[b1["gate_id"]]
        for bid2, b2 in buildings.items():
            d[bid2] = db[b2["gate_id"]]

    model = SeqModelNN(
        args=get_args(),
        dist=dist, 
        cid2wodrs=train_cid2wodrs,
        cache_dataset=f"{DPT}/data/seq_nn_datasets_{len(TRAIN_DATES)}_{len(TEST_DATES)}.pkl")
    model.train()
    # model.train(cache_model=CACHE_SEQ_NN)
    print("device:", model.device)
    print("infer device:", model.infer_device)

    acc_cnt_top5 = [0, 0, 0, 0, 0]
    cnt = 0
    for cid, wodrs in tqdm(test_cid2wodrs.items()):
        for odrs in wodrs:
            seq = [[None, None, None]]
            for i, o in enumerate(odrs):
                bid = o['building_id']
                if bid != seq[-1][0]:
                    seq.append([bid, o["finish_time"], i])
                else:
                    seq[-1][1:] = o["finish_time"], i  # 小哥做出去下一栋楼的决定的时间, 为楼中最后一单的完成时间
            seq = seq[1:]
            for (bid, t, i), (nbid, *_) in zip(seq, seq[1:]):
                cnt += 1
                nbid_top5 = model.infer(
                    odrs=odrs[i+1:],
                    cid=cid, 
                    bid=bid, 
                    t=odrs[i]["finish_time"],
                    top5=True)
                for i, nbid_pre in enumerate(nbid_top5):
                    if nbid_pre == nbid:
                        for j in range(i, 5):
                            acc_cnt_top5[j] += 1
    for i, a in enumerate(acc_cnt_top5):
        print(f"top{i+1} acc:", round(a / cnt * 100, 2), "%")
    # top1 acc: 33.66 %
    # top2 acc: 52.08 %
    # top3 acc: 63.94 %
    # top4 acc: 72.66 %
    # top5 acc: 78.79 %
