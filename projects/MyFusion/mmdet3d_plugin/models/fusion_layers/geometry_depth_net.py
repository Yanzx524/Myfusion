import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import build_conv_layer
from mmcv.runner import BaseModule
from mmdet.models.backbones.resnet import BasicBlock
from mmdet3d.models.builder import FUSION_LAYERS


def get_downsample_depths_torch(depth, down, processing="min"):
    b, c, h, w = depth.shape
    depth = depth.view(b, h // down, down, w // down, down, 1)
    depth = depth.permute(0, 1, 3, 5, 2, 4).contiguous()
    depth = depth.view(-1, down * down)
    depth_tmp = torch.where(depth == 0.0, 1e5 * torch.ones_like(depth), depth)
    if processing == "min":
        depth = torch.min(depth_tmp, dim=-1).values
    elif processing == "max":
        depth = torch.max(depth_tmp, dim=-1).values
    else:
        depth = torch.mean(depth_tmp, dim=-1)
    return depth.view(b, c, h // down, w // down)


def generate_gaussian_depth_target(depth, stride, cam_depth_range, constant_std=0.5):
    depth = depth.flatten(0, 1)
    b, th, tw = depth.shape
    h = th // stride
    w = tw // stride

    unfold_depth = F.unfold(depth.unsqueeze(1), stride, dilation=1, padding=0, stride=stride)
    unfold_depth = unfold_depth.view(b, -1, h, w).permute(0, 2, 3, 1).contiguous()
    valid_mask = unfold_depth != 0

    std_var = torch.ones((b, h, w), device=depth.device, dtype=depth.dtype) * constant_std
    unfold_depth = torch.where(valid_mask, unfold_depth, torch.full_like(unfold_depth, 1e10))
    min_depth = torch.min(unfold_depth, dim=-1)[0]
    min_depth = torch.where(min_depth == 1e10, torch.zeros_like(min_depth), min_depth)

    x = torch.arange(
        cam_depth_range[0] - cam_depth_range[2] / 2,
        cam_depth_range[1],
        cam_depth_range[2],
        device=depth.device,
        dtype=depth.dtype,
    )
    depth_center = min_depth / cam_depth_range[2]
    std_center = std_var / cam_depth_range[2]
    x = x.view(*([1] * depth_center.dim()), -1)
    z = (x - depth_center.unsqueeze(-1)) / (std_center.unsqueeze(-1) * (2 ** 0.5) + 1e-6)
    cdfs = 0.5 * (1 + torch.erf(z))
    depth_dist = cdfs[..., 1:] - cdfs[..., :-1]
    return depth_dist, min_depth


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.ReLU, drop=0.0):
        super().__init__()
        hidden_features = hidden_features or in_features
        out_features = out_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class SELayer(nn.Module):
    def __init__(self, channels, act_layer=nn.ReLU, gate_layer=nn.Sigmoid):
        super().__init__()
        self.conv_reduce = nn.Conv2d(channels, channels, 1, bias=True)
        self.act1 = act_layer()
        self.conv_expand = nn.Conv2d(channels, channels, 1, bias=True)
        self.gate = gate_layer()

    def forward(self, x, x_se):
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)
        return x * self.gate(x_se)


class _ASPPModule(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, padding, dilation):
        super().__init__()
        self.atrous_conv = nn.Conv2d(
            inplanes,
            planes,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            dilation=dilation,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.atrous_conv(x)))


class ASPP(nn.Module):
    def __init__(self, inplanes, mid_channels=256):
        super().__init__()
        dilations = [1, 6, 12, 18]
        self.aspp1 = _ASPPModule(inplanes, mid_channels, 1, 0, dilations[0])
        self.aspp2 = _ASPPModule(inplanes, mid_channels, 3, dilations[1], dilations[1])
        self.aspp3 = _ASPPModule(inplanes, mid_channels, 3, dilations[2], dilations[2])
        self.aspp4 = _ASPPModule(inplanes, mid_channels, 3, dilations[3], dilations[3])
        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(inplanes, mid_channels, 1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )
        self.conv1 = nn.Conv2d(mid_channels * 5, mid_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x4.size()[2:], mode="bilinear", align_corners=True)
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)
        x = self.relu(self.bn1(self.conv1(x)))
        return self.dropout(x)


def _build_depth_branch_conv(mid_channels, depth_channels):
    if mid_channels != depth_channels:
        downsample = nn.Conv2d(mid_channels, depth_channels, kernel_size=3, stride=1, padding=1)
    else:
        downsample = None

    try:
        dcn = build_conv_layer(
            cfg=dict(
                type="DCN",
                in_channels=depth_channels,
                out_channels=depth_channels,
                kernel_size=3,
                padding=1,
                groups=4,
                im2col_step=128,
            )
        )
    except Exception:
        dcn = nn.Conv2d(depth_channels, depth_channels, kernel_size=3, padding=1, bias=False)

    return nn.Sequential(
        BasicBlock(mid_channels, depth_channels, downsample=downsample),
        BasicBlock(depth_channels, depth_channels),
        BasicBlock(depth_channels, depth_channels),
        ASPP(depth_channels, depth_channels),
        dcn,
        nn.Conv2d(depth_channels, depth_channels, kernel_size=1, stride=1, padding=0),
    )


class DepthNet2(nn.Module):
    def __init__(self, in_channels, mid_channels, context_channels, depth_channels, cam_channels=27):
        super().__init__()
        self.reduce_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )
        self.context_conv = nn.Conv2d(mid_channels, context_channels, kernel_size=1, stride=1, padding=0)
        self.bn = nn.BatchNorm1d(cam_channels)
        self.depth_mlp = Mlp(cam_channels, mid_channels, mid_channels)
        self.depth_se = SELayer(mid_channels)
        self.context_mlp = Mlp(cam_channels, mid_channels, mid_channels)
        self.context_se = SELayer(mid_channels)
        self.depth_conv = _build_depth_branch_conv(mid_channels, depth_channels)

    def forward(self, x, mlp_input):
        mlp_input = mlp_input.reshape(-1, mlp_input.shape[-1])
        if mlp_input.shape[0] > 1:
            mlp_input = self.bn(mlp_input)
        else:
            mlp_input = F.batch_norm(
                mlp_input,
                self.bn.running_mean,
                self.bn.running_var,
                self.bn.weight,
                self.bn.bias,
                training=False,
                momentum=0.0,
                eps=self.bn.eps,
            )
        x = self.reduce_conv(x)
        context = self.context_se(x, self.context_mlp(mlp_input)[..., None, None])
        context = self.context_conv(context)
        depth = self.depth_se(x, self.depth_mlp(mlp_input)[..., None, None])
        depth = self.depth_conv(depth)
        return torch.cat([depth, context], dim=1)


@FUSION_LAYERS.register_module()
class GeometryDepth_Net(BaseModule):
    def __init__(
        self,
        use_extra_depth=False,
        use_radar_depth=False,
        data_config=None,
        downsample=8,
        input_channels=1280,
        numC_input=256,
        numC_Trans=256,
        cam_channels=33,
        grid_config=None,
        loss_prob_weight=1.0,
        loss_abs_weight=0.001,
        foreground_loss_alpha=5.0,
        loss_depth_type="kld",
        **kwargs
    ):
        super().__init__()
        self.use_extra_depth = use_extra_depth
        self.use_radar_depth = use_radar_depth
        self.downsample = downsample
        self.input_channels = input_channels
        self.numC_input = numC_input
        self.numC_Trans = numC_Trans
        self.cam_channels = cam_channels
        self.grid_config = grid_config
        self.data_config = data_config
        self.alpha = foreground_loss_alpha if use_radar_depth else 0.0
        self.loss_prob_weight = loss_prob_weight
        self.loss_abs_weight = loss_abs_weight
        self.loss_depth_type = loss_depth_type
        self.constant_std = 0.5

        ds = torch.arange(*self.grid_config["dbound"], dtype=torch.float).view(-1, 1, 1)
        self.D = ds.shape[0]
        self.cam_depth_range = self.grid_config["dbound"]
        self.depth_net = DepthNet2(
            self.input_channels,
            self.numC_input,
            self.numC_Trans,
            self.D,
            cam_channels=self.cam_channels,
        )

    def get_downsampled_gt_depth(self, gt_depths):
        b, n, h, w = gt_depths.shape
        final_dim = self.data_config["final_dim"] if self.training else self.data_config["final_dim_test"]
        target_h = final_dim[0] // self.downsample
        target_w = final_dim[1] // self.downsample
        down_here = h // target_h
        gt_depths = gt_depths.view(b * n, h // down_here, down_here, w // down_here, down_here, 1)
        gt_depths = gt_depths.permute(0, 1, 3, 5, 2, 4).contiguous()
        gt_depths = gt_depths.view(-1, down_here * down_here)
        gt_depths_tmp = torch.where(gt_depths == 0.0, 1e5 * torch.ones_like(gt_depths), gt_depths)
        gt_depths = torch.min(gt_depths_tmp, dim=-1).values
        gt_depths = gt_depths.view(b * n, h // down_here, w // down_here)
        gt_depths = (
            gt_depths
            - (self.grid_config["dbound"][0] - self.grid_config["dbound"][2] / 2)
        ) / self.grid_config["dbound"][2]
        gt_depths_vals = gt_depths.clone()
        gt_depths = torch.where(
            (gt_depths < self.D + 1) & (gt_depths >= 0.0), gt_depths, torch.zeros_like(gt_depths)
        )
        gt_depths = F.one_hot(gt_depths.long(), num_classes=self.D + 1).view(-1, self.D + 1)[:, 1:]
        return gt_depths_vals, gt_depths.float()

    def get_depth_dist(self, x):
        return x.softmax(dim=1)

    def get_mlp_input(self, rot, tran, intrin, post_rot, post_tran, bda=None):
        b = rot.shape[0]
        if intrin.shape[-2:] == (3, 3):
            intrin_full = torch.zeros((b, 3, 4), device=intrin.device, dtype=intrin.dtype)
            intrin_full[:, :, :3] = intrin
            intrin = intrin_full
        if bda.shape[-2:] == (3, 3):
            bda_full = torch.eye(4, device=bda.device, dtype=bda.dtype).unsqueeze(0).repeat(b, 1, 1)
            bda_full[:, :3, :3] = bda
            bda = bda_full
        mlp_input = torch.stack(
            [
                intrin[:, 0, 0],
                intrin[:, 1, 1],
                intrin[:, 0, 2],
                intrin[:, 1, 2],
                intrin[:, 0, 3],
                intrin[:, 1, 3],
                intrin[:, 2, 3],
                post_rot[:, 0, 0],
                post_rot[:, 0, 1],
                post_tran[:, 0],
                post_rot[:, 1, 0],
                post_rot[:, 1, 1],
                post_tran[:, 1],
                bda[:, 0, 0],
                bda[:, 0, 1],
                bda[:, 1, 0],
                bda[:, 1, 1],
                bda[:, 2, 2],
                bda[:, 0, 3],
                bda[:, 1, 3],
                bda[:, 2, 3],
            ],
            dim=-1,
        )
        sensor2ego = torch.cat([rot, tran.reshape(b, 3, 1)], dim=-1).reshape(b, -1)
        return torch.cat([mlp_input, sensor2ego], dim=-1).to(torch.float32)

    def get_bce_depth_loss(self, depth_labels, depth_preds):
        _, depth_labels = self.get_downsampled_gt_depth(depth_labels)
        depth_preds = depth_preds.permute(0, 2, 3, 1).contiguous().view(-1, self.D)
        fg_mask = torch.max(depth_labels, dim=1).values > 0.0
        if fg_mask.sum() == 0:
            return depth_preds.sum() * 0.0
        depth_labels = depth_labels[fg_mask]
        depth_preds = depth_preds[fg_mask]
        return F.binary_cross_entropy(depth_preds, depth_labels, reduction="none").sum() / fg_mask.sum().clamp(min=1.0)

    def get_kld_depth_loss(self, depth_labels, depth_preds):
        b, _, h, w = depth_preds.shape
        depth_gaussian_labels, depth_values = generate_gaussian_depth_target(
            depth_labels,
            self.downsample,
            self.cam_depth_range,
            constant_std=self.constant_std,
        )
        depth_values = depth_values.view(b, h, w)
        fg_mask = (depth_values >= self.cam_depth_range[0]) & (
            depth_values <= (self.cam_depth_range[1] - self.cam_depth_range[2])
        )
        if fg_mask.sum() == 0:
            return depth_preds.sum() * 0.0
        depth_gaussian_labels = depth_gaussian_labels.view(b, h, w, self.D)
        depth_preds = depth_preds.permute(0, 2, 3, 1).contiguous().view(b, h, w, self.D)
        depth_gaussian_labels = depth_gaussian_labels[fg_mask]
        depth_preds = depth_preds[fg_mask]
        return F.kl_div(torch.log(depth_preds + 1e-4), depth_gaussian_labels, reduction="batchmean", log_target=False)

    def get_depth_loss(self, depth_labels, depth_preds, precise_depth, rangeview_logit=None):
        if depth_labels is None:
            zero = depth_preds.sum() * 0.0
            return dict(depth_loss_prob=zero, depth_loss_abs=zero)
        if self.loss_depth_type == "bce":
            depth_loss_prob = self.get_bce_depth_loss(depth_labels, depth_preds)
        else:
            depth_loss_prob = self.get_kld_depth_loss(depth_labels, depth_preds)
        gt_depths_down = get_downsample_depths_torch(depth_labels, down=self.downsample, processing="min")
        mask = (gt_depths_down > self.cam_depth_range[0]) & (gt_depths_down < self.cam_depth_range[1])
        if mask.sum() == 0:
            depth_loss_abs = precise_depth.sum() * 0.0
        else:
            depth_loss_abs = F.smooth_l1_loss(precise_depth[mask], gt_depths_down[mask])
        return dict(
            depth_loss_prob=self.loss_prob_weight * depth_loss_prob,
            depth_loss_abs=self.loss_abs_weight * depth_loss_abs,
        )

    def forward(self, inputs, radar_depth=None, extra_depth=None, img_metas=None):
        x, rots, trans, intrins, post_rots, post_trans, bda, mlp_input = inputs
        bxn, _, h, w = x.shape
        out = self.depth_net(x, mlp_input)
        mono_digit = out[:, : self.D, ...]
        mono_volume = self.get_depth_dist(mono_digit)
        img_feat = out[:, self.D : self.D + self.numC_Trans, ...]
        batch_size = rots.shape[0]
        num_cams = rots.shape[1]
        return img_feat.view(batch_size, num_cams, -1, h, w), mono_volume
