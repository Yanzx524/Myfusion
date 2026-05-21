import torch
from mmcv.runner import force_fp32
from torch import nn

from mmdet3d.models.builder import VOXEL_ENCODERS
from mmdet3d.models.voxel_encoders.utils import PFNLayer_Radar, get_paddings_indicator


@VOXEL_ENCODERS.register_module()
class RadarPillarFeatureNet(nn.Module):
    def __init__(
        self,
        in_channels=4,
        feat_channels=(64,),
        with_distance=False,
        with_cluster_center=True,
        with_voxel_center=True,
        voxel_size=(0.2, 0.2, 4),
        point_cloud_range=(0, -40, -3, 70.4, 40, 1),
        norm_cfg=dict(type="BN1d", eps=1e-3, momentum=0.01),
        mode="max",
        legacy=True,
        with_velocity_snr_center=True,
    ):
        super().__init__()
        assert len(feat_channels) > 0
        self.legacy = legacy
        if with_cluster_center:
            in_channels += 3
        if with_voxel_center:
            in_channels += 2
        if with_distance:
            in_channels += 1
        if with_velocity_snr_center:
            in_channels += 2

        self._with_distance = with_distance
        self._with_cluster_center = with_cluster_center
        self._with_voxel_center = with_voxel_center
        self._with_velocity_snr_center = with_velocity_snr_center
        self.fp16_enabled = False
        self.in_channels = in_channels

        feat_channels = [in_channels] + list(feat_channels)
        pfn_layers = []
        for i in range(len(feat_channels) - 1):
            pfn_layers.append(
                PFNLayer_Radar(
                    feat_channels[i],
                    feat_channels[i + 1],
                    norm_cfg=norm_cfg,
                    last_layer=i == len(feat_channels) - 2,
                    mode=mode,
                )
            )
        self.pfn_layers = nn.ModuleList(pfn_layers)

        self.vx = voxel_size[0]
        self.vy = voxel_size[1]
        self.x_offset = self.vx / 2 + point_cloud_range[0]
        self.y_offset = self.vy / 2 + point_cloud_range[1]
        self.point_cloud_range = point_cloud_range

    @force_fp32(out_fp16=True)
    def forward(self, features, num_points, coors):
        features_ls = [features]

        if self._with_cluster_center:
            points_mean = features[:, :, :3].sum(dim=1, keepdim=True) / num_points.type_as(
                features
            ).view(-1, 1, 1)
            f_cluster = features[:, :, :3] - points_mean
            features_ls.append(f_cluster)

        dtype = features.dtype
        if self._with_voxel_center:
            if not self.legacy:
                f_center = torch.zeros_like(features[:, :, :2])
                f_center[:, :, 0] = features[:, :, 0] - (
                    coors[:, 3].to(dtype).unsqueeze(1) * self.vx + self.x_offset
                )
                f_center[:, :, 1] = features[:, :, 1] - (
                    coors[:, 2].to(dtype).unsqueeze(1) * self.vy + self.y_offset
                )
            else:
                f_center = features[:, :, :2]
                f_center[:, :, 0] = f_center[:, :, 0] - (
                    coors[:, 3].type_as(features).unsqueeze(1) * self.vx + self.x_offset
                )
                f_center[:, :, 1] = f_center[:, :, 1] - (
                    coors[:, 2].type_as(features).unsqueeze(1) * self.vy + self.y_offset
                )
            features_ls.append(f_center)

        if self._with_distance:
            points_dist = torch.norm(features[:, :, :3], 2, 2, keepdim=True)
            features_ls.append(points_dist)

        if self._with_velocity_snr_center:
            velocity_snr_mean = features[:, :, 3:5].sum(dim=1, keepdim=True) / num_points.type_as(
                features
            ).view(-1, 1, 1)
            velocity_snr_center = features[:, :, 3:5] - velocity_snr_mean
            features_ls.append(velocity_snr_center)

        features = torch.cat(features_ls, dim=-1)
        voxel_count = features.shape[1]
        mask = get_paddings_indicator(num_points, voxel_count, axis=0)
        mask = torch.unsqueeze(mask, -1).type_as(features)
        features *= mask

        for pfn in self.pfn_layers:
            features = pfn(features, num_points)

        return features.squeeze()
