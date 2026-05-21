from mmcv.parallel import DataContainer as DC
from mmdet.datasets.builder import PIPELINES


@PIPELINES.register_module()
class CustomCollect3D:
    def __init__(
        self,
        keys,
        meta_keys=(
            "filename",
            "ori_shape",
            "img_shape",
            "lidar2img",
            "cam2img",
            "pad_shape",
            "scale_factor",
            "flip",
            "pcd_horizontal_flip",
            "pcd_vertical_flip",
            "box_mode_3d",
            "box_type_3d",
            "img_norm_cfg",
            "pcd_trans",
            "sample_idx",
            "pcd_scale_factor",
            "pcd_rotation",
            "pts_filename",
            "transformation_3d_flow",
            "img_aug_matrix",
            "lidar_aug_matrix",
            "lidar2cam",
            "gt_depths",
            "cam_aware",
            "bda_rot",
            "segmentation",
            "bbox_Mask",
            "radar_depth",
        ),
    ):
        self.keys = keys
        self.meta_keys = meta_keys

    def __call__(self, results):
        data = {}
        img_metas = {}
        for key in self.meta_keys:
            if key in results:
                img_metas[key] = results[key]

        data["img_metas"] = DC(img_metas, cpu_only=True)
        for key in self.keys:
            data[key] = results[key]
        return data
