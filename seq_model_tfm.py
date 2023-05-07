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
from tqdm import tqdm
from tfm_model import TfmModel

from constants_top import *


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
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


class DefaultArgs:
    def __init__(self):
        self.seed = 233
        self.cuda = 1
        self.infer_cuda = -1
        self.debug = False

        self.max_sql = 30  # max_seq_len
        self.epoch = 1000
        self.batch_size = 100
        self.train_ratio = 0.8

        self.dim_courier = 8
        self.dim_building = 16
        self.nemb = 64    # word embedding
        self.nlayers = 3  # encoder/decoder laryer num
        self.nheads = 2
        

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def xavier_init(m):
    if hasattr(m, "weight") and m.weight.dim() > 1:
        nn.init.xavier_uniform_(m.weight.data)


class DataLoader(object):
    def __init__(self, train_sentences, test_sentences, batch_size, id_map, max_sql):
        self.dset_flag = "train"
        self.id_map = id_map
        self.max_sql = max_sql

        print("train sentences:", len(train_sentences), "test sentences:", len(test_sentences))
        self.train = self.tokenize(train_sentences)
        self.test = self.tokenize(test_sentences)
        print("train words:", self.train.size(), "test words:", self.test.size())

        self.train = self.train\
            .narrow(0, 0, batch_size * (self.train.size(0) // batch_size))\
            .view(batch_size, -1).t().contiguous()
        self.test = self.test\
            .narrow(0, 0, batch_size * (self.test.size(0) // batch_size))\
            .view(batch_size, -1).t().contiguous()
        print(self.train.size(), self.test.size())

    def set_train(self):
        self.dset_flag = "train"
        self.train_si = 0

    def set_test(self):
        self.dset_flag = "test"
        self.test_si = 0

    def tokenize(self, sentences):
        tokens = torch.LongTensor(sum(len(s) for s in sentences) + len(sentences))
        i = 0
        for words in sentences:
            for word in words + ["<eos>"]:
                tokens[i] = self.id_map[word]
                i += 1
        return tokens

    def get_batch(self):
        ## train_si and test_si indicates the index of the start point of the current mini-batch
        if self.dset_flag == "train":
            start_index = self.train_si
            seq_len = min(self.max_sql, self.train.size(0)-self.train_si-1)
            dataset = self.train
            self.train_si = self.train_si + seq_len
        else:
            start_index = self.test_si
            seq_len = min(self.max_sql, self.test.size(0)-self.test_si-1)
            dataset = self.test
            self.test_si = self.test_si + seq_len
        data = dataset[start_index:start_index+seq_len, :].transpose(0, 1)  # batch_size, seq_len
        target = dataset[start_index+1:start_index+seq_len+1, :].transpose(0, 1)

        ## end_flag indicates whether a epoch (train or test epoch) has been ended
        if self.dset_flag == "train" and self.train_si+1 == self.train.size(0):
            end_flag = True
            self.train_si = 0
        elif self.dset_flag == "test" and self.test_si+1 == self.test.size(0):
            end_flag = True
            self.test_si = 0
        else:
            end_flag = False
        return data, target, end_flag


class SeqModelTFM:
    def __init__(self, bids, cid2wodrs, args=DefaultArgs(), cache_dataset=None):
        """
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

        bids = sorted(bids)
        bid_map = {j: i for i, j in enumerate(bids)}
        bid_map["<eos>"] = len(bids)  # end of sentence
        self.bid_map = bid_map
        self.bid_map_rev = {v: k for k, v in bid_map.items()}
        
        # TODO: 暂不管cid, 不管时空特征, 只用到序列信息
        if cache_dataset and os.path.exists(cache_dataset):
            print("Use cached dataset:", cache_dataset)
            self.dataloader: DataLoader = pickle.load(open(cache_dataset, "rb"))
        else:
            print("Making dataset...")
            sentences = []
            for wodrs in cid2wodrs.values():
                for odrs in wodrs:
                    sentence = [odrs[0]["building_id"]]
                    for o in odrs[1:]:
                        if o["building_id"] != sentence[-1]:
                            sentence.append(o["building_id"])
                    if len(sentence) > 1:
                        sentences.append(sentence)
            random.shuffle(sentences)
            t = int(len(sentences) * self.args.train_ratio)
            self.dataloader = DataLoader(
                train_sentences=sentences[:t],
                test_sentences=sentences[t:],
                batch_size=self.args.batch_size,
                id_map=bid_map,
                max_sql=self.args.max_sql)
            if cache_dataset:
                pickle.dump(self.dataloader, open(cache_dataset, "wb"))

    def train(self, cache_model=None):
        if cache_model is not None:
            print("Use cached seq nn:", cache_model)
            model = TfmModel(
                nvoc=len(self.bid_map),  # number of vocabulary
                nword=args.nemb,
                nlayers=args.nlayers,
                nheads=args.nheads,
                max_sql=args.max_sql,
                device=self.infer_device,
            ).to(self.infer_device)
            model.load_state_dict(torch.load(cache_model))
            self.model_best = model
            return
                    
        EARLY_STOP = 100
        run_name = time.strftime("SeqModeTFM_%y%m%d_%H%M%S")
        setproctitle(f"{run_name}@yufudan")
        print(f"Run: {run_name}")
        if not self.args.debug:
            os.makedirs(f"log/{run_name}/pt")

        device = self.device
        model = TfmModel(
            nvoc=len(self.bid_map),  # number of vocabulary
            nword=args.nemb,
            nlayers=args.nlayers,
            nheads=args.nheads,
            max_sql=args.max_sql,
            device=device,
        ).to(device)
        model.apply(xavier_init)

        criterion = nn.CrossEntropyLoss()

        optimizer = torch.optim.Adam(model.parameters())

        dataloader = self.dataloader

        def train_epoch():
            model.train(True)
            dataloader.set_train()
            end_flag = False
            total_loss, total_acc = 0, 0
            cnt1, cnt2 = 0, 0
            while not end_flag:
                optimizer.zero_grad()

                x, y, end_flag = dataloader.get_batch()
                x, y = x.to(device).contiguous(), y.to(device).contiguous()  # batch_size, seq_len
                y_pred, _ = model(x, y[:, :-1])     # batch_size, seq_len-1, nvoc
                y_pred = y_pred.contiguous().view(-1, y_pred.shape[-1])  # batch_size*seq_len-1, nvoc
                y = y[:, 1:].contiguous().view(-1)   # batch_size*seq_len-1

                # 算loss时, 每个batch用到了seq_len-1位
                loss = criterion(y_pred, y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * y.size(0)
                cnt1 += y.size(0)

                # 算acc时, 只看最后1位
                y_pred = y_pred.argmax(1).view(self.args.batch_size, -1)[:, -1].view(-1)
                y = y.view(self.args.batch_size, -1)[:, -1].view(-1)
                total_acc += (y_pred == y).float().mean() * y.size(0)
                cnt2 += y.size(0)
                
            return total_loss / cnt1, total_acc / cnt2

        def test_epoch():
            model.train(False)
            dataloader.set_test()
            end_flag = False
            total_loss, total_acc = 0, 0
            cnt1, cnt2 = 0, 0
            while not end_flag:
                x, y, end_flag = dataloader.get_batch()
                x, y = x.to(device).contiguous(), y.to(device).contiguous()  # batch_size, seq_len
                y_pred, _ = model(x, y[:, :-1])     # batch_size, seq_len-1, nvoc
                y_pred = y_pred.contiguous().view(-1, y_pred.shape[-1])  # batch_size*seq_len-1, nvoc
                y = y[:, 1:].contiguous().view(-1)   # batch_size*seq_len-1

                # 算loss时, 每个batch用到了seq_len-1位
                loss = criterion(y_pred, y)
                total_loss += loss.item() * y.size(0)
                cnt1 += y.size(0)

                # 算acc时, 只看最后1位
                y_pred = y_pred.argmax(1).view(self.args.batch_size, -1)[:, -1].view(-1)
                y = y.view(self.args.batch_size, -1)[:, -1].view(-1)
                total_acc += (y_pred == y).float().mean() * y.size(0)
                cnt2 += y.size(0)

            return total_loss / cnt1, total_acc / cnt2
        
        train_losses = []
        test_losses = []
        best_epoch = -1
        best_loss = 1e12
        with tqdm(range(self.args.epoch), dynamic_ncols=True) as bar:
            for epoch in bar:
                train_loss, train_acc = train_epoch()
                test_loss, test_acc = test_epoch() if args.train_ratio < 1 else [train_loss, train_acc]
                train_losses.append(train_loss)
                test_losses.append(test_loss)
                bar.set_description(
                    f"{train_loss:.4f} {train_acc*100:.2f}% {test_loss:.4f} {test_acc*100:.2f}%"
                )

        # test_loss = test_acc = 0
        # best_acc = -1
        # best_epoch  = -1
        # best_acc_saved = -1
        # best_epoch_saved  = -1
        # with tqdm(range(self.args.epoch), dynamic_ncols=True) as bar: 
        #     for epoch in bar:
        #         random.shuffle(train_set)
        #         train_loss = train_acc = 0
        #         n = 0
        #         for i in range(0, len(train_set) // batch_size * batch_size, batch_size):
        #             x, y = zip(*train_set[i : i + batch_size])
        #             y = torch.stack(y)
        #             y_pred = model(x)
        #             loss = criterion(y_pred, y)
        #             opt.zero_grad()
        #             loss.backward()
        #             opt.step()

        #             train_loss += loss.detach().mean().cpu().item()
        #             train_acc += (y_pred.argmax(1).detach() == y).float().mean()
        #             n += 1
        #             bar.set_description(
        #                 f"{train_loss/n:.4f} {train_acc/n*100:.2f}% {test_loss:.4f} {test_acc*100:.2f}%"
        #             )
        #         if test_set:
        #             with torch.no_grad():
        #                 x, y = zip(*test_set)
        #                 y = torch.stack(y)
        #                 y_pred = model(x)
        #                 test_loss = criterion(y_pred, y)
        #                 test_acc = (y_pred.argmax(1).detach() == y).float().mean()
        #         else:
        #             test_loss = train_loss / n
        #             test_acc = train_acc / n
                
        #         if self.args.debug:
        #             # writer.add_scalar("fig/train_loss", train_loss / n, epoch)
        #             # writer.add_scalar("fig/test_loss", test_loss, epoch)
        #             # writer.add_scalars(
        #             #         "fig/acc",
        #             #         {
        #             #             "train": train_acc / n,
        #             #             "test": test_acc,
        #             #         },
        #             #         epoch,
        #             # )
        #             pass
        #         elif epoch % 10 == 9:  
        #             if test_acc > best_acc_saved:
        #                 best_acc_saved = test_acc
        #                 best_epoch_saved = epoch
        #             torch.save(
        #                 model.state_dict(),
        #                 f"log/{run_name}/pt/{epoch}.pt",
        #             )
                
        #         # early stop
        #         if test_acc > best_acc:
        #             best_acc = test_acc
        #             best_epoch = epoch
        #         if epoch - best_epoch > EARLY_STOP:
        #             print(f"Stop since last {EARLY_STOP} epochs no gain")
        #             print("best test acc:", best_acc, "best epoch:", best_epoch)
        #             break

        # if not self.args.debug:
        #     print("best test acc saved:", best_acc_saved, "best epoch saved:", best_epoch_saved)
        #     model = Model(
        #         num_courier=len({i[0] for i in self.dataset}),
        #         dim_courier=self.args.dim_courier,
        #         num_building=self.dataset[0][3].shape[0],
        #         dim_building=self.args.dim_building,
        #         dim_feature=self.dataset[0][3].shape[1],
        #         dim_mlp=[int(i) for i in self.args.mlp.split(",")],
        #     ).to(self.infer_device)
        #     model.load_state_dict(torch.load(f"log/{run_name}/pt/{best_epoch_saved}.pt"))
        #     self.model_best = model

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
    
    args = DefaultArgs()
    model = SeqModelTFM(
        bids=list(buildings.keys()),
        cid2wodrs=train_cid2wodrs,
        args=args,
        cache_dataset=f"{DPT}/data/seq_tfm_datasets_{len(TRAIN_DATES)}_{len(TEST_DATES)}_{int(10*args.train_ratio)}.pkl")
    print("device:", model.device)
    print("infer device:", model.infer_device)
    model.train()
    