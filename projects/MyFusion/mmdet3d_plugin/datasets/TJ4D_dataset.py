import os
from os import path as osp

import numpy as np
from mmdet.datasets import DATASETS
from mmdet3d.datasets import KittiDataset


@DATASETS.register_module()
class TJ4DDataset(KittiDataset):
    CLASSES = ("Pedestrian", "Cyclist", "Car", "Truck")

    def _get_pts_filename(self, idx):
        return osp.join(self.root_split, self.pts_prefix, f"{idx:06d}.bin")

    def dynamic_baseline(self, info):
        p2 = np.array(info["calib"]["P2"].astype(np.float32))
        return -p2[0, 3] / (-p2[0, 0])

    def get_data_info(self, index):
        info = self.data_infos[index]
        sample_idx = info["image"]["image_idx"]
        img_filename = os.path.join(self.data_root, info["image"]["image_path"])

        rect = info["calib"]["R0_rect"].astype(np.float32)
        trv2c = info["calib"]["Tr_velo_to_cam"].astype(np.float32)
        p2 = info["calib"]["P2"].astype(np.float32)

        lidar2img = p2 @ rect @ trv2c
        lidar2cam = rect @ trv2c
        cam2img = p2

        input_dict = dict(
            sample_idx=sample_idx,
            pts_filename=self._get_pts_filename(sample_idx),
            img_prefix=None,
            img_info=dict(filename=img_filename),
            lidar2img=lidar2img,
            lidar2cam=lidar2cam,
            cam2img=cam2img,
            focal_length=cam2img[0][0],
            baseline=self.dynamic_baseline(info),
            calib=info["calib"],
        )

        if not self.test_mode:
            input_dict["ann_info"] = self.get_ann_info(index)

        return input_dict
