import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmdet3d.models.builder import FUSION_LAYERS


@FUSION_LAYERS.register_module()
class Cross_Modal_Fusion(nn.Module):
    def __init__(self, kernel_size=3, img_channels=256, rad_channels=384, out_channels=256):
        super().__init__()
        assert kernel_size in (3, 7), "kernel_size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1
        self.att_img = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False),
            nn.Sigmoid(),
        )
        self.att_radar = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False),
            nn.Sigmoid(),
        )
        self.reduce_mix_bev = ConvModule(
            img_channels + rad_channels,
            out_channels,
            3,
            padding=1,
            conv_cfg=None,
            norm_cfg=dict(type="BN", eps=1e-3, momentum=0.01),
            act_cfg=dict(type="ReLU"),
            inplace=False,
        )

    def forward(self, img_bev, radar_bev):
        img_avg = torch.mean(img_bev, dim=1, keepdim=True)
        img_max, _ = torch.max(img_bev, dim=1, keepdim=True)
        img_att = self.att_img(torch.cat([img_avg, img_max], dim=1))

        radar_avg = torch.mean(radar_bev, dim=1, keepdim=True)
        radar_max, _ = torch.max(radar_bev, dim=1, keepdim=True)
        radar_att = self.att_radar(torch.cat([radar_avg, radar_max], dim=1))

        img_bev = img_bev * radar_att
        radar_bev = radar_bev * img_att
        fused = torch.cat([img_bev, radar_bev], dim=1)
        return self.reduce_mix_bev(fused)
