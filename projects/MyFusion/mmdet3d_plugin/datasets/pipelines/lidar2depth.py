import os

import numpy as np
import torch
from mmdet.datasets.builder import PIPELINES


@PIPELINES.register_module()
class CreateDepthFromLiDAR:
    def __init__(self, data_root=None, dataset="kitti"):
        self.data_root = data_root
        self.dataset = dataset
        assert self.dataset in ["kitti", "VoD", "TJ4D"]

    def __call__(self, results):
        img_h, img_w = results["img"].shape[:2]

        if self.dataset == "TJ4D":
            pts_rel = os.path.relpath(results["pts_filename"], self.data_root)
            split = pts_rel.split(os.sep)[0]
            index_num = os.path.splitext(os.path.basename(results["pts_filename"]))[0]
            calib_path = os.path.join(self.data_root, split, "calib", index_num + ".txt")
            lidar_path = calib_path.replace("calib", "velodyne").replace("txt", "bin")
        elif self.dataset == "VoD":
            vod_root = os.path.dirname(os.path.normpath(self.data_root))
            lidar_root = os.path.join(vod_root, "lidar")
            pts_rel = os.path.relpath(results["pts_filename"], self.data_root)
            split = pts_rel.split(os.sep)[0]
            index_num = os.path.splitext(os.path.basename(results["pts_filename"]))[0]
            calib_path = os.path.join(lidar_root, split, "calib", index_num + ".txt")
            lidar_path = calib_path.replace("calib", "velodyne").replace("txt", "bin")
        else:
            return results

        with open(calib_path, "r") as f:
            lines = f.readlines()

        p2 = np.array([float(x) for x in lines[2].split(" ")[1:13]]).reshape(3, 4)
        rect = np.array([float(x) for x in lines[4].split(" ")[1:10]]).reshape(3, 3)
        rect4 = np.zeros((4, 4), dtype=rect.dtype)
        rect4[3, 3] = 1.0
        rect4[:3, :3] = rect
        trv2c = np.array([float(x) for x in lines[5].split(" ")[1:13]]).reshape(3, 4)
        trv2c = np.concatenate([trv2c, np.array([[0.0, 0.0, 0.0, 1.0]])], axis=0)

        lidar2cam = torch.tensor(rect4 @ trv2c, dtype=torch.float32)
        cam2lidar = torch.inverse(lidar2cam)
        rots = cam2lidar[:3, :3]
        trans = cam2lidar[:3, 3]
        intrins, post_rots, post_trans = results["cam_aware"][2:5]

        lidar_points = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 4)
        lidar_points = torch.from_numpy(lidar_points[:, :3]).float()

        projected = self.project_points(
            lidar_points, rots, trans, intrins, post_rots, post_trans
        )
        valid = (
            (projected[..., 0] >= 0)
            & (projected[..., 1] >= 0)
            & (projected[..., 0] <= img_w - 1)
            & (projected[..., 1] <= img_h - 1)
            & (projected[..., 2] > 0)
            & (projected[..., 2] <= 80)
        )

        gt_depths = torch.zeros((img_h, img_w))
        valid_points = projected[valid]
        if len(valid_points) > 0:
            depth_order = torch.argsort(valid_points[:, 2], descending=True)
            valid_points = valid_points[depth_order]
            gt_depths[
                valid_points[:, 1].round().long(),
                valid_points[:, 0].round().long(),
            ] = valid_points[:, 2]

        results["gt_depths"] = gt_depths
        return results

    def project_points(self, points, rots, trans, intrins, post_rots, post_trans):
        points = points.view(-1, 1, 3)
        points = points - trans.view(1, -1, 3)
        inv_rots = torch.inverse(rots).view(-1, 3, 3).unsqueeze(0)
        points = inv_rots @ points.unsqueeze(-1)
        points = torch.cat(
            [points, torch.ones((points.shape[0], points.shape[1], 1, 1))], dim=2
        )
        points = (intrins.view(-1, 4, 4).unsqueeze(0) @ points).squeeze(-1)
        points_d = points[..., 2:3]
        points_uv = points[..., :2] / points_d
        points_uv = (
            post_rots[:2, :2].view(-1, 2, 2).unsqueeze(0) @ points_uv.unsqueeze(-1)
        )
        points_uv = points_uv.squeeze(-1) + post_trans.view(-1, 3)[:, :2].unsqueeze(0)
        return torch.cat((points_uv, points_d), dim=2)
