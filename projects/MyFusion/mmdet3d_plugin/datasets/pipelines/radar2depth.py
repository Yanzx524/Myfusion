import numpy as np
from copy import deepcopy

from mmdet.datasets.builder import PIPELINES
import torch


@PIPELINES.register_module()
class CreateDepthFromRaDAR:
    def __init__(self, filter_min, filter_max):
        self.filter_min = filter_min
        self.filter_max = filter_max

    def __call__(self, results):
        h, w = results["img_shape"]
        img_metas = deepcopy(results)
        _, _, _, post_rots, post_trans = results["cam_aware"][:5]
        cam2img = torch.as_tensor(img_metas["cam2img"], dtype=torch.float32)
        lidar2cam = torch.as_tensor(img_metas["lidar2cam"], dtype=torch.float32)
        lidar_aug = torch.as_tensor(img_metas["lidar_aug_matrix"], dtype=torch.float32)
        cam2img[:2, :3] = post_rots[:2, :2] @ cam2img[:2, :3]
        cam2img[:2, 2] = post_trans[:2] + cam2img[:2, 2]
        lidar2img = cam2img @ lidar2cam
        lidar2img = lidar2img @ torch.inverse(lidar_aug)

        radar_depth = np.zeros((h, w), dtype=np.float32)
        radar_points = results["points"].tensor.numpy()[:, :3]
        pts_hom = np.concatenate((radar_points, np.ones((radar_points.shape[0], 1))), axis=1)
        img_pts = (lidar2img.cpu().numpy() @ pts_hom.T).T
        img_pts[:, :2] = img_pts[:, :2] / img_pts[:, 2:3]
        valid = (
            (img_pts[:, 0] >= 0)
            & (img_pts[:, 0] < w)
            & (img_pts[:, 1] >= 0)
            & (img_pts[:, 1] < h)
            & (img_pts[:, 2] > self.filter_min)
            & (img_pts[:, 2] < self.filter_max)
        )
        img_pts = img_pts[valid]
        if len(img_pts) > 0:
            x_idx = np.floor(img_pts[:, 0]).astype(int)
            y_idx = np.floor(img_pts[:, 1]).astype(int)
            radar_depth[y_idx, x_idx] = img_pts[:, 2]

        results["radar_depth"] = radar_depth
        return results
