import torch
import torch.nn as nn
from mmcv.runner import BaseModule
from mmdet3d.models.builder import FUSION_LAYERS

try:
    from packages.Voxelization.bev_pool import bev_pool
except ImportError:
    bev_pool = None


def gen_dx_bx(xbound, ybound, zbound):
    dx = torch.tensor([xbound[2], ybound[2], zbound[2]])
    bx = torch.tensor(
        [xbound[0] + xbound[2] / 2.0, ybound[0] + ybound[2] / 2.0, zbound[0] + zbound[2] / 2.0]
    )
    nx = torch.tensor(
        [
            (xbound[1] - xbound[0]) / xbound[2],
            (ybound[1] - ybound[0]) / ybound[2],
            (zbound[1] - zbound[0]) / zbound[2],
        ]
    )
    return dx, bx, nx


@FUSION_LAYERS.register_module()
class ViewTransformerLSS(BaseModule):
    def __init__(self, grid_config=None, data_config=None, downsample=8, num_in_height=8):
        super().__init__()
        self.grid_config = grid_config
        self.data_config = data_config
        self.downsample = downsample
        self.num_in_height = num_in_height
        self.dx, self.bx, self.nx = gen_dx_bx(
            self.grid_config["xbound"],
            self.grid_config["ybound"],
            self.grid_config["zbound"],
        )
        self.D = torch.arange(*self.grid_config["dbound"], dtype=torch.float).view(-1, 1, 1).shape[0]
        self.frustum = None

    def create_frustum(self, device):
        if self.training:
            ogf_h, ogf_w = self.data_config["final_dim"]
        else:
            ogf_h, ogf_w = self.data_config["final_dim_test"]
        f_h, f_w = ogf_h // self.downsample, ogf_w // self.downsample
        ds = torch.arange(*self.grid_config["dbound"], dtype=torch.float, device=device).view(-1, 1, 1).expand(-1, f_h, f_w)
        xs = torch.linspace(0, ogf_w - 1, f_w, dtype=torch.float, device=device).view(1, 1, f_w).expand(self.D, f_h, f_w)
        ys = torch.linspace(0, ogf_h - 1, f_h, dtype=torch.float, device=device).view(1, f_h, 1).expand(self.D, f_h, f_w)
        return torch.stack((xs, ys, ds), -1)

    def get_geometry(self, rots, trans, intrins, post_rots, post_trans, bda):
        b, n, _ = trans.shape
        device = trans.device
        if self.frustum is None or self.frustum.device != device:
            self.frustum = self.create_frustum(device)

        points = self.frustum - post_trans.view(b, n, 1, 1, 1, 3)
        points = torch.inverse(post_rots.cpu()).to(device).view(b, n, 1, 1, 1, 3, 3).matmul(points.unsqueeze(-1))
        points = torch.cat(
            [points[..., :2, :] * points[..., 2:3, :], points[..., 2:3, :]],
            dim=5,
        )
        if intrins.shape[-1] == 4:
            shift = intrins[:, :, :3, 3]
            points = points - shift.view(b, n, 1, 1, 1, 3, 1)
            intrins = intrins[:, :, :3, :3]
        combine = rots.matmul(torch.inverse(intrins.cpu()).to(device))
        points = combine.view(b, n, 1, 1, 1, 3, 3).matmul(points).squeeze(-1)
        points += trans.view(b, n, 1, 1, 1, 3)

        if bda.shape[-1] == 4:
            points = torch.cat(
                [points, torch.ones(*points.shape[:-1], 1, device=device, dtype=points.dtype)],
                dim=-1,
            )
            points = bda.view(b, 1, 1, 1, 1, 4, 4).matmul(points.unsqueeze(-1)).squeeze(-1)
            points = points[..., :3]
        else:
            points = bda.view(b, 1, 1, 1, 1, 3, 3).matmul(points.unsqueeze(-1)).squeeze(-1)
        return points

    def voxel_pooling(self, geom_feats, x):
        b, n, d, h, w, c = x.shape
        nprime = b * n * d * h * w
        x = x.reshape(nprime, c)
        dx = self.dx.to(x.device)
        bx = self.bx.to(x.device)
        nx = self.nx.to(x.device)
        geom_feats = ((geom_feats - (bx - dx / 2.0)) / dx).long().view(nprime, 3)
        batch_ix = torch.cat(
            [torch.full((nprime // b, 1), i, device=x.device, dtype=torch.long) for i in range(b)]
        )
        geom_feats = torch.cat((geom_feats, batch_ix), 1)
        kept = (
            (geom_feats[:, 0] >= 0)
            & (geom_feats[:, 0] < nx[0])
            & (geom_feats[:, 1] >= 0)
            & (geom_feats[:, 1] < nx[1])
            & (geom_feats[:, 2] >= 0)
            & (geom_feats[:, 2] < nx[2])
        )
        x = x[kept]
        geom_feats = geom_feats[kept]
        if bev_pool is not None and x.is_cuda:
            final = bev_pool(x, geom_feats, b, int(nx[2]), int(nx[0]), int(nx[1]))
        else:
            final = self.voxel_pooling_fallback(x, geom_feats, b, int(nx[2]), int(nx[0]), int(nx[1]))
        return final.permute(0, 1, 3, 4, 2).contiguous()

    def voxel_pooling_fallback(self, feats, coords, batch_size, nz, nx, ny):
        # CPU/debug fallback for environments where the custom CUDA bev_pool op is unavailable.
        out = feats.new_zeros((batch_size * nz * nx * ny, feats.shape[1]))
        linear_idx = ((coords[:, 3] * nz + coords[:, 2]) * nx + coords[:, 0]) * ny + coords[:, 1]
        out.index_add_(0, linear_idx, feats)
        return out.view(batch_size, nz, nx, ny, feats.shape[1]).permute(0, 4, 1, 2, 3).contiguous()

    def forward(self, feat, depth_prob, cam_params):
        b, n, c, h, w = feat.shape
        rots, trans, intrins, post_rots, post_trans, bda = cam_params
        geom = self.get_geometry(rots, trans, intrins, post_rots, post_trans, bda)
        if depth_prob.dim() == 4:
            depth_prob = depth_prob.view(b, n, self.D, h, w)
        x = depth_prob.unsqueeze(2) * feat.unsqueeze(3)
        x = x.permute(0, 1, 3, 4, 5, 2).contiguous()
        return self.voxel_pooling(geom, x)
