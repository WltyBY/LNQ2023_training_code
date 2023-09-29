from typing import Callable

import torch
from nnunetv2.utilities.ddp_allgather import AllGatherGrad
from nnunetv2.utilities.tensor_utilities import sum_tensor
from torch import nn
import numpy as np


class TverskyDiceLoss(nn.Module):
    def __init__(self, apply_nonlin: Callable = None, batch_dice: bool = False, do_bg: bool = True, smooth: float = 1.,
                 ddp: bool = True):
        """
        ！！！使用TverskyDiceLoss来平衡FN和FP
        """
        super(TverskyDiceLoss, self).__init__()

        self.do_bg = do_bg
        self.batch_dice = batch_dice
        self.apply_nonlin = apply_nonlin
        self.smooth = 1e-8
        self.ddp = ddp

        self.pool=nn.AdaptiveAvgPool3d(output_size=(8,8,8))
        
   
    def forward(self, x, y, loss_mask=None):
        shp_x, shp_y = x.shape, y.shape

        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)

        if not self.do_bg:
            x = x[:, 1:]

        # make everything shape (b, c)
        axes = list(range(2, len(shp_x)))

        with torch.no_grad():
            if len(shp_x) != len(shp_y):
                y = y.view((shp_y[0], 1, *shp_y[1:]))

            if all([i == j for i, j in zip(shp_x, shp_y)]):
                # if this is the case then gt is probably already a one hot encoding
                y_onehot = y
            else:
                gt = y.long()
                y_onehot = torch.zeros(shp_x, device=x.device, dtype=torch.bool)
                y_onehot.scatter_(1, gt, 1)

            if not self.do_bg:
                y_onehot = y_onehot[:, 1:]
            sum_gt = y_onehot.sum(axes) if loss_mask is None else (y_onehot * loss_mask)

        intersect = (x * y_onehot) if loss_mask is None else (x * y_onehot * loss_mask)
        
        A=0.3
        B=0.4

        FP_intersect = (x * (1-y_onehot)) if loss_mask is None else (x * y_onehot * loss_mask)
        FN_intersect = ((1-x) * y_onehot) if loss_mask is None else (x * y_onehot * loss_mask)

        if(shp_x[-1]>=32):
            region_TP=self.pool(intersect)
            region_FP=self.pool(FP_intersect)
            region_FN=self.pool(FN_intersect)

            if self.batch_dice:
                region_TP = region_TP.sum(0)
                region_FP = region_FP.sum(0)
                region_FN = region_FN.sum(0)

            adaptive_a=A+B*((region_FP+self.smooth)/(region_FP+region_FN+self.smooth))
            adaptive_b=A+B*((region_FN+self.smooth)/(region_FP+region_FN+self.smooth))

            # if self.ddp and self.batch_dice:
            #     intersect = AllGatherGrad.apply(intersect).sum(0)
            #     sum_pred = AllGatherGrad.apply(sum_pred).sum(0)
            #     sum_gt = AllGatherGrad.apply(sum_gt).sum(0)

            

            region_tversky=(region_TP+self.smooth)/(region_TP+adaptive_a*region_FP+adaptive_b*region_FN+self.smooth)
            region_tversky=region_tversky.sum(list(range(1,len(region_tversky.shape))))

            region_tversky=region_tversky.mean()
            dc=region_tversky
            dc=dc/512
        
        else:
            intersect = intersect.sum((2,3,4))/pow(shp_x[-1],3)
            FP_intersect = FP_intersect.sum((2,3,4))/pow(shp_x[-1],3)
            FN_intersect = FN_intersect.sum((2,3,4))/pow(shp_x[-1],3)
            if self.batch_dice:
                intersect = intersect.sum(0)
                FP_intersect = FP_intersect.sum(0)
                FN_intersect = FN_intersect.sum(0)
            

            adaptive_a=A+B*((FP_intersect+self.smooth)/(FP_intersect+FN_intersect+self.smooth))
            adaptive_b=A+B*((FN_intersect+self.smooth)/(FP_intersect+FN_intersect+self.smooth))

            dc = (intersect + self.smooth) / (intersect + adaptive_a*FP_intersect + adaptive_b*FN_intersect + self.smooth)
        
            dc=dc.mean()

        return -dc
