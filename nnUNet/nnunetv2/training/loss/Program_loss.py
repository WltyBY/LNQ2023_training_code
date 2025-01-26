import torch
from torch import nn
import torch.nn.functional as F
from nnunetv2.training.loss.dice import TverskyLoss
import numpy as np
import random


class total_loss(nn.Module):
    def __init__(self):
        super(total_loss, self).__init__()
        # self.ce = RobustCrossEntropyLoss(weight=torch.Tensor([1., 5.]).cuda(), **ce_kwargs)
        self.KL_CELoss = KL_CE_loss()
        # self.SCELoss = SCE_Loss()
        self.tverskyLoss = TverskyLoss(alpha=0.1, beta=0.9, do_bg=False)

    def forward(self, net_main: torch.Tensor, net_aux: torch.Tensor, target: torch.Tensor):
        """
        target must be b, c, x, y(, z) with c=1
        :param net_main:
        :param net_aux:
        :param target:
        :return:
        """

        main_sce_loss = self.tverskyLoss(net_main, target)
        aux_sce_loss = self.tverskyLoss(net_aux, target)
        loss_by_label = 0.5 * (main_sce_loss + aux_sce_loss)

        main_soft = torch.softmax(net_main, dim=1)
        aux_soft = torch.softmax(net_aux, dim=1)
        beta = random.random() + 1e-10
        # pseudo_label_hard = torch.argmax((beta * main_soft.detach() + (1.0 - beta) * aux_soft.detach()), dim=1, keepdim=False)
        pseudo_label_soft = torch.softmax((beta * main_soft.detach() + (1.0 - beta) * aux_soft.detach()), dim=1)
        # pseudo_label_soft_max, _ = torch.max(pseudo_label_soft, dim=1)
        loss_pseudo_label = 0.5 * (self.KL_CELoss(net_main, net_aux, pseudo_label_soft) +
                                   self.KL_CELoss(net_aux, net_main, pseudo_label_soft))

        loss = loss_by_label + loss_pseudo_label

        print("Proposedv2 Loss:", loss.item(), ", loss_by_label:", loss_by_label.item(), ", loss_pseudo_label:",
              loss_pseudo_label.item())
        print("pred in label: ", (torch.sum(torch.argmax(main_soft, dim=1, keepdim=False) * target[:, 0]) /
                                  torch.sum(target[:, 0])).item())
        print("pred in whole: ", (torch.sum(torch.argmax(main_soft, dim=1, keepdim=False)) /
                                  np.prod(target.size())).item())

        return loss


class KL_CE_loss(nn.Module):
    def __init__(self, apply_nonlin=True):
        super(KL_CE_loss, self).__init__()
        self.KL_loss = nn.KLDivLoss(reduction='none')
        self.ce = nn.CrossEntropyLoss(reduction='none')
        # self.ce = nn.CrossEntropyLoss(reduction='none')
        self.apply_nonlin = apply_nonlin
        if self.apply_nonlin:
            self.sm = torch.nn.Softmax(dim=1)
            self.log_sm = torch.nn.LogSoftmax(dim=1)

    def forward(self, pred_main, pred_aux, label):
        """
        the three input are the net output without sigmoid or softmax
         pred_main, pred_aux: [batchsize, channel, (z,) x, y]
         target must be b, c, x, y(, z) with c=1
        """
        ce_loss = self.ce(pred_main, label[:, 0].long())

        if self.apply_nonlin:
            variance = torch.sum(self.KL_loss(self.log_sm(pred_main), self.sm(pred_aux)), dim=1)
        else:
            variance = torch.sum(self.KL_loss(torch.log(pred_main), pred_aux), dim=1)
        exp_variance = torch.exp(-variance)

        loss = (ce_loss * exp_variance).sum() / exp_variance.sum() + variance.mean()

        return loss


class KL_SCE_loss(nn.Module):
    def __init__(self, alpha=0.1, beta=1):
        super(KL_SCE_loss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.KL_loss = nn.KLDivLoss(reduction='none')
        # self.ce = nn.CrossEntropyLoss(weight=torch.Tensor([1, 10.]).to('cuda'), reduction='none')
        self.ce = nn.CrossEntropyLoss(reduction='none')
        self.sm = torch.nn.Softmax(dim=1)
        self.log_sm = torch.nn.LogSoftmax(dim=1)

    def forward(self, pred_main, pred_aux, label):
        """
        the three input are the net output without sigmoid or softmax
         pred_main, pred_aux: [batchsize, channel, (z,) x, y]
         target must be b, c, (z,) x, y with c=1
        """
        shp_x, shp_y = pred_main.shape, label.shape
        # CE
        ce_loss = self.ce(pred_main, label[:, 0].long())

        # RCE
        pred = F.softmax(pred_main, dim=1)
        pred = torch.clamp(pred, min=1e-7, max=1.0)

        if len(shp_x) != len(shp_y):
            label = label.view((shp_y[0], 1, *shp_y[1:]))
        gt = label.long()
        label_one_hot = torch.zeros(shp_x, device=pred.device)
        label_one_hot.scatter_(1, gt, 1)

        label_one_hot = torch.clamp(label_one_hot, min=1e-4, max=1.0)
        rce = -1 * torch.mean((pred * torch.log(label_one_hot)), dim=1)

        sce_loss = self.alpha * ce_loss + self.beta * rce

        # KL
        variance = torch.sum(self.KL_loss(self.log_sm(pred_main), self.sm(pred_aux)), dim=1)
        exp_variance = torch.exp(-variance)

        loss = torch.mean(sce_loss * exp_variance + variance)

        return loss


class SCE_Loss(nn.Module):
    def __init__(self, alpha=1, beta=0.5, eps=1e-7):
        super(SCE_Loss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.eps = eps
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, pred, labels):
        shp_x, shp_y = pred.shape, labels.shape
        # CE 部分，正常的交叉熵损失
        ce = self.cross_entropy(pred, labels[:, 0].long())

        # RCE
        pred = F.softmax(pred, dim=1)
        pred = torch.clamp(pred, min=self.eps, max=1.0)

        if len(shp_x) != len(shp_y):
            labels = labels.view((shp_y[0], 1, *shp_y[1:]))
        gt = labels.long()
        label_one_hot = torch.zeros(shp_x, device=pred.device)
        label_one_hot.scatter_(1, gt, 1)

        # weight = torch.zeros(shp_x, device=pred.device)
        # weight[:, 0] = 0.1
        # weight[:, 1] = 30.
        label_one_hot = torch.clamp(label_one_hot, min=1e-4, max=1.0)
        rce = -1 * torch.mean((pred * torch.log(label_one_hot)), dim=1)

        loss = self.alpha * ce + self.beta * rce.mean()
        return loss


if __name__ == "__main__":
    import time

    KL_loss = KL_CE_loss()
    input1 = torch.rand((2, 2, 64, 128, 224))
    input2 = torch.rand((2, 2, 64, 128, 224))
    label = torch.randint(0, 2, (2, 1, 64, 128, 224))
    start = time.time()
    print(KL_loss(input1, input2, label))
    print(time.time() - start)

    SCELoss = SCE_Loss()
    input1 = torch.rand((2, 2, 64, 128, 224))
    label = torch.randint(0, 2, (2, 1, 64, 128, 224))
    start = time.time()
    print(SCELoss(input1, label))
    print(time.time() - start)

    KL_SCE_loss = KL_SCE_loss()
    input1 = torch.rand((2, 2, 64, 128, 224))
    input2 = torch.rand((2, 2, 64, 128, 224))
    label = torch.randint(0, 2, (2, 1, 64, 128, 224))
    start = time.time()
    print(KL_SCE_loss(input1, input2, label))
    print(time.time() - start)
