"""Loss function"""

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F


def kl_loss(pred1, pred2, eps=1e-8):
    single_loss = torch.sum(pred2 * torch.log(eps + pred2 / (pred1 + eps)), 1)
    return single_loss


class BirankLoss(nn.Module):
    """
    Compute contrastive loss
    """
    def __init__(self, margin=0, max_violation=False):
        super(BirankLoss, self).__init__()
        self.margin = margin
        self.max_violation = max_violation

    def forward(self, scores):
        # compute image-sentence score matrix
        diagonal = scores.diag().view(scores.size(0), 1)
        d1 = diagonal.expand_as(scores)
        d2 = diagonal.t().expand_as(scores)

        # compare every diagonal score to scores in its column
        # caption retrieval
        cost_s = (self.margin + scores - d1).clamp(min=0)
        # compare every diagonal score to scores in its row
        # image retrieval
        cost_im = (self.margin + scores - d2).clamp(min=0)

        # clear diagonals
        mask = torch.eye(scores.size(0)) > .5
        if torch.cuda.is_available():
            I = mask.cuda()
        cost_s = cost_s.masked_fill_(I, 0)
        cost_im = cost_im.masked_fill_(I, 0)

        # keep the maximum violating negative for each query
        if self.max_violation:
            cost_s = cost_s.max(1)[0]
            cost_im = cost_im.max(0)[0]
        return cost_s.sum() + cost_im.sum()


class CMPMLoss(nn.Module):
    def __init__(self, smooth=10.0, eps=1e-8):
        super(CMPMLoss, self).__init__()
        self.smooth = smooth
        self.eps = eps
        self.softmax = nn.Softmax(dim=-1)
        self.relu = nn.ReLU()

    def forward(self, sims, labels):
        labels = torch.FloatTensor(labels)
        if torch.cuda.is_available():
            torch.FloatTensor(labels).cuda()

        label_mask = labels.view(-1, 1) - labels.view(1, -1)
        label_mask = (torch.abs(label_mask) < 0.5).float()
        label_mask_norm = F.normalize(label_mask, p=1, dim=-1)

        v2t_pred = self.softmax(self.smooth * sims)
        t2v_pred = self.softmax(self.smooth * sims.t())

        v2t_loss = kl_loss(label_mask_norm, v2t_pred, self.eps)
        t2v_loss = kl_loss(label_mask_norm, t2v_pred, self.eps)

        loss = v2t_loss.mean() + t2v_loss.mean()

        return loss


class PolyLoss(nn.Module):
    """
    Compute contrastive loss
    """

    def __init__(self, margin=0, max_violation=False):
        super(PolyLoss, self).__init__()
        self.margin = margin
        self.max_violation = max_violation

    def forward(self, scores):
        epsilon = 1e-5
        size = scores.size(0)
        hh = scores.t()
        label = torch.Tensor([i for i in range(size)])

        loss = list()
        for i in range(size):
            pos_pair_ = scores[i][i]
            pos_pair_ = pos_pair_[pos_pair_ < 1 - epsilon]
            neg_pair_ = scores[i][label != label[i]]

            neg_pair = neg_pair_[neg_pair_ + self.margin > min(pos_pair_)]

            pos_pair = pos_pair_
            if len(neg_pair) < 1 or len(pos_pair) < 1:
                continue

            pos_loss = torch.clamp(0.2 * torch.pow(pos_pair, 2) - 0.7 * pos_pair + 0.5, min=0)
            neg_pair = max(neg_pair)
            neg_loss = torch.clamp(0.9 * torch.pow(neg_pair, 2) - 0.4 * neg_pair + 0.03, min=0)

            loss.append(pos_loss + neg_loss)
        for i in range(size):
            pos_pair_ = hh[i][i]
            pos_pair_ = pos_pair_[pos_pair_ < 1 - epsilon]
            neg_pair_ = hh[i][label != label[i]]

            neg_pair = neg_pair_[neg_pair_ + self.margin > min(pos_pair_)]

            pos_pair = pos_pair_
            if len(neg_pair) < 1 or len(pos_pair) < 1:
                continue
            pos_loss = torch.clamp(0.2 * torch.pow(pos_pair, 2) - 0.7 * pos_pair + 0.5, min=0)

            neg_pair = max(neg_pair)
            neg_loss = torch.clamp(0.9 * torch.pow(neg_pair, 2) - 0.4 * neg_pair + 0.03, min=0)
            loss.append(pos_loss + neg_loss)

        if len(loss) == 0:
            return torch.zeros([], requires_grad=True)

        loss = sum(loss) / size
        return loss


class LadderLoss(nn.Module):
    '''
    Ladder Loss Function for Visual Semantic Embedding Learning
    The distance metric is cosine distance.

    reldeg: relevance degree metric, it must have a __getitem__ method.
            when reldeg is not provided, ladder loss falls back to pwl.
            possible RDs: BowSim, EcoSTSMatrix, CoMatrix (IoU mode)
    '''

    def __init__(self, margins=[0.2], thresholds=[], betas=[], reldeg=None, max_violation=False):
        '''
        Instantiate LadderLoss function.
        Note that you can adjust self.hard_negative to enable or disable
        hard negative sampling on the first ladder component (i.e. pairwise
        ranking loss function). However hard negative sampling, or say
        hard contrastive sampling is mandatory in the rest ladder components.
        '''
        super(LadderLoss, self).__init__()
        self.rd = reldeg
        self.hard_negative = True
        self.margins = margins
        self.max_violation = max_violation
        if len(self.margins) > 1 and (reldeg is None):
            raise ValueError("Missing RelDeg.")
        self.thresholds = thresholds
        if len(self.margins) - 1 != len(self.thresholds):
            raise ValueError("numbers of margin/threshold don't match.")
        elif len(self.thresholds) > 0 and (reldeg is None):
            raise ValueError("where is the reldeg?")
        self.betas = betas
        if len(self.margins) - 1 != len(self.betas):
            raise ValueError("numbers of margin/beta don't match.")
        if len(thresholds) > 0 and (reldeg is None):
            raise ValueError("RelDeg metric is required if set any threshold")

    def forward(self, scores, sids):
        '''
        Forward pass.

        forward pass derived using the cumulative form of inequality
        s(q, q) - s(q, i) > a_1
        s(q, q) - s(q, j) > a_2 + a_1
        s(q, q) - s(q, k) > a+3 + a_2 + a_1
        ...
        '''
        losses = []

        # [ First Ladder ]
        diagonal = scores.diag().view(scores.size(0), 1)

        # [ l-Ladder (l > 0) and so on ]: Mandatory hard-negatives
        rdmat = torch.tensor(self.rd(sids)).float()
        if torch.cuda.is_available():
            rdmat = rdmat.cuda()

        for l, thre in enumerate(self.thresholds):

            if self.max_violation:
                simmask = (rdmat >= thre).float()
                dismask = (rdmat < thre).float()
                gt_sim = scores * simmask + 1.0 * dismask
                gt_dis = scores * dismask
                xvld = self.margins[1+l] - gt_sim.min(dim=1)[0] + gt_dis.max(dim=1)[0]
                xvld = xvld.clamp(min=0)
                vxld = self.margins[1+l] - gt_sim.min(dim=0)[0] + gt_dis.max(dim=0)[0]
                vxld = vxld.clamp(min=0)

                losses.append(self.betas[l] * (xvld.sum() + vxld.sum()))
            else:
                # cumulative
                dismask = (rdmat < thre).float()
                gt_dis = scores * dismask
                xvld = np.sum(self.margins[:l+2]) - diagonal.view(-1) + gt_dis.max(dim=1)[0]
                xvld = xvld.clamp(min=0)
                vxld = np.sum(self.margins[:l+2]) - diagonal.view(-1) + gt_dis.max(dim=0)[0]
                vxld = vxld.clamp(min=0)

                losses.append(self.betas[l] * (xvld.sum() + vxld.sum()))

        return sum(losses)


class BoostabsLoss(nn.Module):
    """
    Compute contrastive loss
    """
    def __init__(self, margin=0, beta=1.0, gamma=0.5):
        super(BoostabsLoss, self).__init__()
        self.margin = margin * beta
        self.margin_pos = self.margin * gamma
        self.margin_neg = self.margin * (1.0 - gamma)

    def forward(self, scores, scores_anchor):
        # compute image-sentence score matrix
        pos_sims = scores.diag().view(scores.size(0), 1)
        pos_sims_anchor = scores_anchor.diag().view(scores_anchor.size(0), 1)

        # compare every diagonal score to scores in its column
        cost_pos = (self.margin_pos + pos_sims_anchor - pos_sims).clamp(min=0).sum()
        cost_neg = (self.margin_neg + scores - scores_anchor).clamp(min=0)

        # clear diagonals
        mask = torch.eye(scores.size(0)) > .5
        if torch.cuda.is_available():
            I = mask.cuda()
        cost_neg = cost_neg.masked_fill_(I, 0)
        cost_neg = cost_neg.max(1)[0].sum() + cost_neg.max(0)[0].sum()
        return cost_pos * 2 + cost_neg


class BoostrelLoss(nn.Module):
    """
    Compute contrastive loss
    """
    def __init__(self, margin=0, beta=1.0):
        super(BoostrelLoss, self).__init__()
        self.margin = margin * beta

    def forward(self, scores, scores_anchor):
        # compute image-sentence score matrix
        diagonal = scores.diag().view(scores.size(0), 1)
        d1 = diagonal.expand_as(scores)
        d2 = diagonal.t().expand_as(scores)

        diagonal_anchor = scores_anchor.diag().view(scores_anchor.size(0), 1)
        d1_anchor = diagonal_anchor.expand_as(scores_anchor)
        d2_anchor = diagonal_anchor.t().expand_as(scores_anchor)

        # compare every diagonal score to scores in its column
        cost_s = (self.margin + (d1_anchor-scores_anchor) - (d1-scores)).clamp(min=0)
        cost_im = (self.margin + (d2_anchor-scores_anchor) - (d2-scores)).clamp(min=0)

        # clear diagonals
        mask = torch.eye(scores.size(0)) > .5
        if torch.cuda.is_available():
            I = mask.cuda()
        cost_s = cost_s.masked_fill_(I, 0)
        cost_im = cost_im.masked_fill_(I, 0)

        cost_s = cost_s.max(1)[0]
        cost_im = cost_im.max(0)[0]
        return cost_s.sum() + cost_im.sum()


class DarkRankLoss(nn.Module):
    """DarkRank: Accelerating Deep Metric Learning via Cross Sample Similarities Transfer, AAAI2017"""
    def __init__(self, alpha=3, beta=3, permute_len=4):
        super(DarkRankLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.permute_len = permute_len

    def forward(self, scores, scores_anchor):
        scoresT, scores_anchorT = scores.t(), scores_anchor.t()

        scores = -1 * self.alpha * scores.pow(self.beta)
        scores_anchor = -1 * self.alpha * scores_anchor.pow(self.beta)
        permute_idx = scores_anchor.sort(dim=1, descending=True)[1][:, 1:(self.permute_len+1)]
        ordered_scores = torch.gather(scores, 1, permute_idx)
        log_prob = (ordered_scores - torch.stack([torch.logsumexp(ordered_scores[:, i:], dim=1)
                                                  for i in range(permute_idx.size(1))], dim=1)).sum(dim=1)
        loss = (-1 * log_prob).mean()

        scoresT = -1 * self.alpha * scoresT.pow(self.beta)
        scores_anchorT = -1 * self.alpha * scores_anchorT.pow(self.beta)
        permute_idx = scores_anchorT.sort(dim=1, descending=True)[1][:, 1:(self.permute_len+1)]
        ordered_scores = torch.gather(scoresT, 1, permute_idx)
        log_prob = (ordered_scores - torch.stack([torch.logsumexp(ordered_scores[:, i:], dim=1)
                                                  for i in range(permute_idx.size(1))], dim=1)).sum(dim=1)
        lossT = (-1 * log_prob).mean()

        return loss + lossT


class RKDLoss(nn.Module):
    """Relational Knowledge Disitllation, CVPR2019"""
    def __init__(self, weight=10.0):
        super(RKDLoss, self).__init__()
        self.weight = weight

    def forward(self, scores, scores_anchor):
        # RKD distance loss
        loss = F.smooth_l1_loss(scores, scores_anchor)
        return loss * self.weight


class SPLoss(nn.Module):
    """Similarity-Preserving Knowledge Distillation, ICCV2019, verified by original author"""

    def __init__(self, weight=10.0):
        super(SPLoss, self).__init__()
        self.weight = weight

    def forward(self, scores, scores_anchor):
        scores = F.normalize(scores)
        scores_anchor = F.normalize(scores_anchor)
        scores_diff = scores_anchor - scores
        loss = (scores_diff * scores_diff).sum(1).mean()

        scoresT = F.normalize(scores.t())
        scores_anchorT = F.normalize(scores_anchor.t())
        scores_diffT = scores_anchorT - scoresT
        lossT = (scores_diffT * scores_diffT).sum(1).mean()

        return (loss + lossT) * self.weight

_loss = {
    'birank': BirankLoss,
    'cmpm': CMPMLoss,
    'poly': PolyLoss,
    'ladder': LadderLoss,
    'boostabs': BoostabsLoss,
    'boostrel': BoostrelLoss,
    'darkrank': DarkRankLoss,
    'sp': SPLoss,
    'rkd': RKDLoss
}


def init_loss(name, **kwargs):
    """Initializes an dataset."""
    avai_losses = list(_loss.keys())
    if name not in avai_losses:
        raise ValueError('Invalid loss function. Received "{}"'.format(name))
    return _loss[name](**kwargs)
