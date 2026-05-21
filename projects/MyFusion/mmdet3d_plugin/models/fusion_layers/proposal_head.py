import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import BaseModule
from mmdet.models import weight_reduce_loss
from mmdet3d.models.builder import FUSION_LAYERS


def dice_loss(pred, target, weight=None, eps=1e-3, reduction="mean", avg_factor=None):
    pred = pred.reshape(pred.size(0), -1)
    target = target.reshape(target.size(0), -1).float()
    inter = torch.sum(pred * target, 1)
    pred_sq = torch.sum(pred * pred, 1) + eps
    target_sq = torch.sum(target * target, 1) + eps
    loss = 1 - (2 * inter) / (pred_sq + target_sq)
    return weight_reduce_loss(loss, weight, reduction, avg_factor)


@FUSION_LAYERS.register_module()
class CustomDiceLoss(nn.Module):
    def __init__(self, use_sigmoid=True, activate=True, reduction="mean", loss_weight=1.0, eps=1e-3):
        super().__init__()
        self.use_sigmoid = use_sigmoid
        self.activate = activate
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.eps = eps

    def forward(self, pred, target, weight=None, reduction_override=None, avg_factor=None):
        reduction = reduction_override if reduction_override else self.reduction
        if self.activate:
            if self.use_sigmoid:
                pred = pred.sigmoid()
            else:
                raise NotImplementedError
        return self.loss_weight * dice_loss(
            pred, target, weight, eps=self.eps, reduction=reduction, avg_factor=avg_factor
        )


@FUSION_LAYERS.register_module()
class FRPN(BaseModule):
    def __init__(self, in_channels=256, scale_factor=1, mask_thre=0.4, topk_rate_test=0.01, loss_weight=1.0):
        super().__init__()
        self.mask_net = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, 1, kernel_size=3, padding=1, stride=1),
        )
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode="bilinear", align_corners=True)
        self.dice_loss = FUSION_LAYERS.build(
            dict(type="CustomDiceLoss", use_sigmoid=True, loss_weight=1.0)
        )
        self.register_buffer("ce_pos_weight", torch.tensor([2.13], dtype=torch.float32))
        self.mask_thre = mask_thre
        self.topk_rate_test = topk_rate_test
        self.loss_weight = loss_weight

    def forward(self, x):
        return self.upsample(self.mask_net(x))

    def get_bev_mask_loss(self, gt_bev_mask, pred_bev_mask):
        bs, _, bev_h, bev_w = gt_bev_mask.shape
        target = gt_bev_mask.reshape(bs, bev_w * bev_h).permute(1, 0).to(torch.float)
        pred = pred_bev_mask.reshape(bs, bev_w * bev_h).permute(1, 0)
        mask_ce_loss = (
            F.binary_cross_entropy_with_logits(pred, target, pos_weight=self.ce_pos_weight)
            * self.loss_weight
        )
        mask_dc_loss = (
            self.dice_loss(pred_bev_mask.reshape(bs, -1), gt_bev_mask.reshape(bs, -1))
            * self.loss_weight
        )
        return dict(bev_mask_ce_loss=mask_ce_loss, bev_mask_dc_loss=mask_dc_loss)
