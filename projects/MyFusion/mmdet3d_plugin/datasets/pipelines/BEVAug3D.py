import numpy as np
import torch
from mmdet.datasets.builder import PIPELINES


@PIPELINES.register_module()
class GlobalRotScaleTransFlipAll:
    def __init__(self, bda_aug_conf, is_train=True):
        if not is_train:
            bda_aug_conf = dict(
                rot_range=(0.0, 0.0),
                scale_ratio_range=(1.0, 1.0),
                translation_std=(0.0, 0.0, 0.0),
                flip_dx_ratio=0.0,
                flip_dy_ratio=0.0,
            )
        self.rot_range = bda_aug_conf["rot_range"]
        self.scale_ratio_range = bda_aug_conf["scale_ratio_range"]
        self.translation_std = bda_aug_conf["translation_std"]
        self.flip_dx_ratio = bda_aug_conf["flip_dx_ratio"]
        self.flip_dy_ratio = bda_aug_conf["flip_dy_ratio"]

    def __call__(self, results):
        if "transformation_3d_flow" not in results:
            results["transformation_3d_flow"] = []

        self.rotate_bev_along_z(results)
        self.translate(results)
        self.scale(results)
        self.flip(results)

        results["transformation_3d_flow"].extend(["R", "T", "S", "F"])

        lidar_aug = np.eye(4, dtype=np.float32)
        lidar_aug[:3, :3] = (
            results["pcd_flip_xy"][:3, :3]
            @ (results["pcd_rotation"][:3, :3] * results["pcd_scale_factor"])
        ).numpy()
        lidar_aug[:3, 3] = (
            results["pcd_flip_xy"][:3, :3]
            @ (results["pcd_trans"] * results["pcd_scale_factor"])
        ).numpy()
        results["lidar_aug_matrix"] = lidar_aug
        results["bda_rot"] = lidar_aug
        return results

    def rotate_bev_along_z(self, results):
        angle = np.random.uniform(*self.rot_range)
        if "points" in results:
            results["points"].rotate(np.array(-angle))
        if "gt_bboxes_3d" in results:
            results["gt_bboxes_3d"].rotate(np.array(angle))

        rot_cos = torch.cos(torch.tensor(angle))
        rot_sin = torch.sin(torch.tensor(angle))
        results["pcd_rotation"] = torch.tensor(
            [
                [rot_cos, rot_sin, 0.0, 0.0],
                [-rot_sin, rot_cos, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=torch.float32,
        )

    def translate(self, results):
        trans = np.random.normal(scale=np.array(self.translation_std), size=3).T
        if "points" in results:
            results["points"].translate(trans)
        if "gt_bboxes_3d" in results:
            results["gt_bboxes_3d"].translate(trans)
        results["pcd_trans"] = torch.tensor(trans, dtype=torch.float32)

    def scale(self, results):
        scale_ratio = np.random.uniform(*self.scale_ratio_range)
        if "points" in results:
            results["points"].scale(scale_ratio)
        if "gt_bboxes_3d" in results:
            results["gt_bboxes_3d"].scale(scale_ratio)
        results["pcd_scale_factor"] = scale_ratio

    def flip(self, results):
        mat = torch.eye(4, dtype=torch.float32)
        if np.random.rand() < self.flip_dx_ratio:
            mat[0, 0] = -1
            if "points" in results:
                results["points"].flip(bev_direction="vertical")
            if "gt_bboxes_3d" in results:
                results["gt_bboxes_3d"].flip(bev_direction="vertical")
            results["pcd_vertical_flip"] = True
        if np.random.rand() < self.flip_dy_ratio:
            mat[1, 1] = -1
            if "points" in results:
                results["points"].flip(bev_direction="horizontal")
            if "gt_bboxes_3d" in results:
                results["gt_bboxes_3d"].flip(bev_direction="horizontal")
            results["pcd_horizontal_flip"] = True
        results["pcd_flip_xy"] = mat
