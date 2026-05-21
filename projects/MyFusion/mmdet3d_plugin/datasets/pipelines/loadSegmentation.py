import os

import numpy as np
from mmdet.datasets.builder import PIPELINES


@PIPELINES.register_module()
class loadSegmentation:
    def __init__(self, data_root=None, dataset="kitti", seg_type="detectron2"):
        self.data_root = data_root
        self.dataset = dataset
        self.seg_type = seg_type
        assert self.dataset in ["kitti", "VoD", "TJ4D"]

    def __call__(self, results):
        index_num = results["pts_filename"].split("/")[-1].split(".")[0]
        if self.dataset == "VoD":
            seg_root = os.path.join(self.data_root, "..", "segmentation")
        elif self.dataset == "TJ4D":
            seg_root = os.path.join(self.data_root, "segmentation")
        else:
            return results

        seg_path = os.path.join(seg_root, index_num + ".npy")
        seg = np.load(seg_path)
        if self.dataset == "TJ4D":
            seg[(seg.shape[0] - 160) :, :] = 0
        results["segmentation"] = seg
        return results
