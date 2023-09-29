import torch
from nnunetv2.training.loss.dice import SoftDiceLoss, MemoryEfficientSoftDiceLoss
from nnunetv2.training.loss.robust_ce_loss import RobustCrossEntropyLoss, TopKLoss
from nnunetv2.utilities.helpers import softmax_helper_dim1
from torch import nn

from nnunetv2.training.loss.dice_PED import TverskyDiceLoss

class Pretrain_MSE_loss(nn.Module):
    def __init__(self, mse_kwargs, weight_ce=1, weight_dice=1, use_ignore_label: bool = False):
        """
        DO NOT APPLY NONLINEARITY IN YOUR NETWORK!

        target mut be one hot encoded
        IMPORTANT: We assume use_ignore_label is located in target[:, -1]!!!

        :param soft_dice_kwargs:
        :param bce_kwargs:
        :param aggregate:
        """
        super(Pretrain_MSE_loss, self).__init__()
        
        self.use_ignore_label = use_ignore_label

        self.mse = nn.MSELoss(**mse_kwargs)

    def forward(self, net_output: torch.Tensor, target: torch.Tensor):
        if self.use_ignore_label:
            # target is one hot encoded here. invert it so that it is True wherever we can compute the loss
            mask = (1 - target[:, -1:]).bool()
            # remove ignore channel now that we have the mask
            target_regions = torch.clone(target[:, :-1])
        else:
            target_regions = target
            mask = None

        
        if mask is not None:
            mse_loss = (self.mse(net_output, target_regions) * mask).sum() / torch.clip(mask.sum(), min=1e-8)
        else:
            mse_loss = self.mse(net_output, target_regions)

        result = mse_loss
        return result
