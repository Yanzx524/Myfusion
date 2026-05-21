import numpy as np
from mmdet.datasets.builder import PIPELINES


@PIPELINES.register_module()
class gen2DMask:
    def __init__(self, use_seg=False, use_softlabel=False, is_train=True):
        self.is_train = is_train
        self.use_seg = use_seg
        self.use_softlabel = use_softlabel

    def __call__(self, results):
        h, w = results["img_shape"]
        if not self.is_train:
            results["bbox_Mask"] = np.zeros((h, w), dtype=np.float32)
            return results

        bbox_mask = np.zeros((h, w), dtype=np.bool_)
        gt_bboxes = results["gt_bboxes"][results["gt_labels"] != -1]
        gt_bboxes = [gt_bboxes[i] for i in range(len(gt_bboxes))]

        y_coords, x_coords = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
        for bbox in gt_bboxes:
            x1, y1, x2, y2 = np.array(bbox).astype(np.int32)
            if not self.use_softlabel:
                bbox_mask[y1:y2, x1:x2] = True
            else:
                y_center = (y1 + y2) / 2
                x_center = (x1 + x2) / 2
                sigma = min(y2 - y1 + 1, x2 - x1 + 1) / 6
                gaussian = np.exp(
                    -((x_coords - x_center) ** 2 + (y_coords - y_center) ** 2)
                    / (2 * sigma**2)
                )
                bbox_mask = np.maximum(bbox_mask, gaussian)

        results["bbox_Mask"] = bbox_mask.astype(np.float32)
        return results
