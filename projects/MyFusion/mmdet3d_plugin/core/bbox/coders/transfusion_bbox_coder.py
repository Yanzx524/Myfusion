import torch
from mmdet.core.bbox import BaseBBoxCoder
from mmdet.core.bbox.builder import BBOX_CODERS


@BBOX_CODERS.register_module()
class TransFusionBBoxCoder(BaseBBoxCoder):
    def __init__(
        self,
        pc_range,
        out_size_factor,
        voxel_size,
        post_center_range=None,
        score_threshold=None,
        code_size=8,
    ):
        self.pc_range = pc_range
        self.out_size_factor = out_size_factor
        self.voxel_size = voxel_size
        self.post_center_range = post_center_range
        self.score_threshold = score_threshold
        self.code_size = code_size

    def encode(self, dst_boxes):
        targets = torch.zeros([dst_boxes.shape[0], self.code_size], device=dst_boxes.device)
        targets[:, 0] = (dst_boxes[:, 0] - self.pc_range[0]) / (
            self.out_size_factor * self.voxel_size[0]
        )
        targets[:, 1] = (dst_boxes[:, 1] - self.pc_range[1]) / (
            self.out_size_factor * self.voxel_size[1]
        )
        targets[:, 3] = dst_boxes[:, 3].log()
        targets[:, 4] = dst_boxes[:, 4].log()
        targets[:, 5] = dst_boxes[:, 5].log()
        targets[:, 2] = dst_boxes[:, 2] + dst_boxes[:, 5] * 0.5
        targets[:, 6] = torch.sin(dst_boxes[:, 6])
        targets[:, 7] = torch.cos(dst_boxes[:, 6])
        if self.code_size == 10:
            targets[:, 8:10] = dst_boxes[:, 7:]
        return targets

    def decode(self, heatmap, rot, dim, center, height, vel, filter=False):
        final_preds = heatmap.max(1, keepdims=False).indices
        final_scores = heatmap.max(1, keepdims=False).values

        center[:, 0, :] = (
            center[:, 0, :] * self.out_size_factor * self.voxel_size[0] + self.pc_range[0]
        )
        center[:, 1, :] = (
            center[:, 1, :] * self.out_size_factor * self.voxel_size[1] + self.pc_range[1]
        )
        dim[:, 0, :] = dim[:, 0, :].exp()
        dim[:, 1, :] = dim[:, 1, :].exp()
        dim[:, 2, :] = dim[:, 2, :].exp()
        height = height - dim[:, 2:3, :] * 0.5
        rots, rotc = rot[:, 0:1, :], rot[:, 1:2, :]
        rot = torch.atan2(rots, rotc)

        if vel is None:
            final_box_preds = torch.cat([center, height, dim, rot], dim=1).permute(0, 2, 1)
        else:
            final_box_preds = torch.cat([center, height, dim, rot, vel], dim=1).permute(0, 2, 1)

        predictions_dicts = []
        for i in range(heatmap.shape[0]):
            predictions_dicts.append(
                dict(
                    bboxes=final_box_preds[i],
                    scores=final_scores[i],
                    labels=final_preds[i],
                )
            )

        if filter is False:
            return predictions_dicts

        if self.score_threshold is not None:
            thresh_mask = final_scores > self.score_threshold

        if self.post_center_range is None:
            raise NotImplementedError(
                "Need post_center_range for filtered batch decoding."
            )

        self.post_center_range = torch.tensor(self.post_center_range, device=heatmap.device)
        mask = (final_box_preds[..., :3] >= self.post_center_range[:3]).all(2)
        mask &= (final_box_preds[..., :3] <= self.post_center_range[3:]).all(2)

        filtered = []
        for i in range(heatmap.shape[0]):
            cmask = mask[i, :]
            if self.score_threshold is not None:
                cmask &= thresh_mask[i]
            filtered.append(
                dict(
                    bboxes=final_box_preds[i, cmask],
                    scores=final_scores[i, cmask],
                    labels=final_preds[i, cmask],
                )
            )
        return filtered
