"""Loss function"""

import torch
import torch.nn as nn
from torch.nn import functional as F


def kl_loss(pred1, pred2, eps=1e-8):
    single_loss = torch.sum(pred2 * torch.log(eps + pred2 / (pred1 + eps)), 1)
    return single_loss


def ce_loss(logit, pred):
    single_loss = torch.sum(-pred * F.log_softmax(logit, dim=-1), dim=-1)
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
        label_mask = labels.view(-1, 1) - labels.view(1, -1)
        label_mask = (torch.abs(label_mask) < 0.5).float()
        label_mask_norm = F.normalize(label_mask, p=1, dim=-1)

        v2t_pred = self.softmax(self.smooth * sims)
        t2v_pred = self.softmax(self.smooth * sims.t())

        v2t_loss = kl_loss(label_mask_norm, v2t_pred, self.eps)
        t2v_loss = kl_loss(label_mask_norm, t2v_pred, self.eps)

        loss = v2t_loss.mean() + t2v_loss.mean()

        return loss


# def supremum_margin(reference, margin, edge):
#     margin_fixed = torch.ones_like(reference) * margin
#     margin_adapt = edge * 1.0 - reference
#     new_margin = torch.where(reference < (edge-margin), margin_fixed, margin_adapt)
#     return new_margin.clone().detach()
#
#
# def infimum_margin(reference, margin, edge):
#     margin_fixed = torch.ones_like(reference) * margin
#     margin_adapt = edge * 1.0 + reference
#     new_margin = torch.where(reference > (margin-edge), margin_fixed, margin_adapt)
#     return new_margin.clone().detach()
#
#
# def supremum_margin(reference, margin, edge):
#     scale = margin * 2.0
#     temp = 4.0 / scale
#     new_margin = scale / (1.0 + torch.exp(temp * (reference - edge))) - margin
#     return new_margin.clone().detach()
#
#
# def infimum_margin(reference, margin, edge):
#     scale = margin * 2.0
#     temp = 4.0 / scale
#     new_margin = scale / (1.0 + torch.exp(-temp * (reference + edge))) - margin
#     return new_margin.clone().detach()
#
#
# pos_margin = supremum_margin(pos_sims_t, self.margin * 0.5, 1.0)
# neg_margin = infimum_margin(scores_t, self.margin * 0.5, 1.0)
# margin_s = supremum_margin(d1_t-scores_t, self.margin, 2.0)
# margin_im = supremum_margin(d2_t-scores_t, self.margin, 2.0)


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


class DMLLoss(nn.Module):
    def __init__(self, smooth=10.0):
        super(DMLLoss, self).__init__()
        self.smooth = smooth
        self.loss_kl = nn.KLDivLoss(reduction='batchmean')

    def forward(self, scores, scores_t):

        v2t_loss = self.loss_kl(F.log_softmax(self.smooth * scores, dim=-1),
                                F.softmax(self.smooth * scores_t, dim=-1))
        t2v_loss = self.loss_kl(F.log_softmax(self.smooth * scores.t(), dim=-1),
                                F.softmax(self.smooth * scores_t.t(), dim=-1))

        loss = v2t_loss + t2v_loss

        return loss


class BoostproLoss(nn.Module):
    def __init__(self, smooth=10.0):
        super(BoostproLoss, self).__init__()
        self.smooth = smooth
        self.loss_kl = nn.KLDivLoss(reduction='batchmean')

    def forward(self, scores, scores_t):
        scores_sub = scores - scores_t
        labels = torch.eye(scores.size(0)).to(scores.device)

        v2t_loss = \
            self.loss_kl(F.log_softmax(self.smooth * scores_sub, dim=-1), labels)
        t2v_loss = \
            self.loss_kl(F.log_softmax(self.smooth * scores_sub.t(), dim=-1), labels)

        loss = v2t_loss + t2v_loss

        return loss


_loss = {
    'birank': BirankLoss,
    'cmpm': CMPMLoss,
    'boostabs': BoostabsLoss,
    'boostrel': BoostrelLoss,
    'dml': DMLLoss,
    'boostpro': BoostproLoss,
}


def init_loss(name, **kwargs):
    """Initializes an dataset."""
    avai_losses = list(_loss.keys())
    if name not in avai_losses:
        raise ValueError('Invalid loss function. Received "{}"'.format(name))
    return _loss[name](**kwargs)
