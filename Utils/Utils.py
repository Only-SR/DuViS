import torch as t
import torch.nn.functional as F
import numpy as np


class Metric:
    @staticmethod
    def recall(hits, origin):
        recall_list = [hits[user] / len(origin[user]) for user in hits]
        recall = round(sum(recall_list) / len(recall_list), 5)
        return recall

    @staticmethod
    def ndcg(hits, origin, predicted, topk):
        ndcg_list = []
        for user in hits:
            temTopLocs = list(predicted[user])  # Ensure temTopLocs is a list
            dcg = sum(
                [np.reciprocal(np.log2(temTopLocs.index(item) + 2)) for item in origin[user] if item in temTopLocs])
            idcg = sum([np.reciprocal(np.log2(i + 2)) for i in range(min(len(origin[user]), topk))])
            ndcg = dcg / idcg if idcg > 0 else 0
            ndcg_list.append(ndcg)
        return round(sum(ndcg_list) / len(ndcg_list), 5)


def calcRes(self, topLocs, tstLocs, batIds, topk):
    assert topLocs.shape[0] == len(batIds)
    hits = {}
    for i in range(len(batIds)):
        temTopLocs = list(topLocs[i])
        temTstLocs = tstLocs[batIds[i]]
        hits[batIds[i]] = len(set(temTstLocs) & set(temTopLocs))

    recall = Metric.recall(hits, tstLocs)
    ndcg = Metric.ndcg(hits, tstLocs, topLocs, topk)

    return recall, ndcg


def innerProduct(usrEmbeds, itmEmbeds):
    return t.sum(usrEmbeds * itmEmbeds, dim=-1)


def pairPredict(ancEmbeds, posEmbeds, negEmbeds):
    return innerProduct(ancEmbeds, posEmbeds) - innerProduct(ancEmbeds, negEmbeds)

def InfoNCE(view1, view2, temperature: float, b_cos: bool = True):
    if b_cos:
        view1, view2 = F.normalize(view1, dim=1), F.normalize(view2, dim=1)

    pos_score = (view1 @ view2.T) / temperature
    score = t.diag(F.log_softmax(pos_score, dim=1))
    return -score.mean()

def calcRegLoss(model,normalize=False):
    # ret = 0
    # for W in model.parameters():
    #     ret += W.norm(2).square()
    # # ret += (W.norm(2) ** 2) / W.numel()  # 归一化处理
    # return ret
    ret = 0
    for W in model.parameters():
        if normalize:
            ret += (W.norm(2).square() / W.numel())
        else:
            ret += W.norm(2).square()
    return ret


def l2_reg_loss(*args):
    emb_loss = 0
    for emb in args:
        emb_loss += t.norm(emb, p=2) / emb.shape[0]
    return emb_loss


def contrastLoss(embeds1, embeds2, nodes, temp):
    embeds1 = F.normalize(embeds1, p=2)
    embeds2 = F.normalize(embeds2, p=2)
    pckEmbeds1 = embeds1[nodes]
    pckEmbeds2 = embeds2[nodes]
    nume = t.exp(t.sum(pckEmbeds1 * pckEmbeds2, dim=-1) / temp)
    deno = t.exp(pckEmbeds1 @ embeds2.T / temp).sum(-1)
    return -t.log(nume / deno)
