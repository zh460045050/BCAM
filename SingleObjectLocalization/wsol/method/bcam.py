
import torch
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn as nn

__all__ = ['bcam']

def make_variable(tensor, volatile=False):
    """Convert Tensor to Variable."""
    if torch.cuda.is_available():
        tensor = tensor.cuda()
    return Variable(tensor, volatile=volatile)


def get_lossV1(output_dict, target, rate_ff=0, rate_fb=0, rate_bf=0, rate_bb=0):

    ####Load four background scores
    label_fore = output_dict['logits']
    label_back = output_dict['logits_back']
    label_fore_rev = output_dict['logits_rev']
    label_back_rev = output_dict['logits_back_rev']

    ####Generating ground truthes
    label = make_variable(torch.zeros(label_fore.shape[0], label_fore.shape[1])).scatter_(1, target.unsqueeze(-1), 1)
    gt_back_rev = make_variable(torch.zeros(label.shape))
    gt_fore_rev = make_variable(1 - label.clone())
    gt_back = make_variable(torch.ones(label.shape))

    ####Calculate Loss####
    loss_cls1 = nn.CrossEntropyLoss().cuda()(label_fore, target)
    loss_cls2 = F.multilabel_soft_margin_loss(label_fore_rev, gt_fore_rev)
    loss_cls3 = F.multilabel_soft_margin_loss(label_back, gt_back)
    loss_cls4 = F.multilabel_soft_margin_loss(label_back_rev, gt_back_rev)

    loss = rate_ff * loss_cls1 + rate_fb * loss_cls2 + rate_bf * loss_cls3 + rate_bb * loss_cls4

    return loss


def get_loss(output_dict, target, rate_ff=0, rate_fb=0, rate_bf=0, rate_bb=0):

    ####Load four background scores
    label_fore = output_dict['logits']
    label_back = output_dict['logits_back']
    label_fore_rev = output_dict['logits_rev']
    label_back_rev = output_dict['logits_back_rev']
    #l1_regularization = torch.abs(output_dict['fore_weight'] + output_dict['back_weight']).mean(dim=0).sum(dim=0).squeeze()
    #print(l1_regularization)
    ####Generating ground truthes
    label = make_variable(torch.zeros(label_fore.shape[0], label_fore.shape[1])).scatter_(1, target.unsqueeze(-1), 1)
    gt_back_rev = make_variable(torch.zeros(label.shape))
    gt_fore_rev = make_variable(1 - label.clone())
    gt_back = make_variable(torch.ones(label.shape))

    ####Calculate Loss####
    loss_cls1 = nn.CrossEntropyLoss().cuda()(label_fore, target)
    loss_cls2 = F.multilabel_soft_margin_loss(label_fore_rev, gt_fore_rev)
    loss_cls3 = F.multilabel_soft_margin_loss(label_back, gt_back)
    loss_cls4 = F.multilabel_soft_margin_loss(label_back_rev, gt_back_rev)

    loss = rate_ff * loss_cls1 + rate_fb * loss_cls2 + rate_bf * loss_cls3 + rate_bb * loss_cls4 #+ l1_regularization

    return loss