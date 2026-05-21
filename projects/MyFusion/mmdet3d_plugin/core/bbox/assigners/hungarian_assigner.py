import torch
from mmdet.core.bbox.assigners import AssignResult, BaseAssigner
from mmdet.core.bbox.builder import BBOX_ASSIGNERS
from mmdet.core.bbox.iou_calculators import build_iou_calculator
from mmdet.core.bbox.match_costs import build_match_cost
from mmdet.core.bbox.match_costs.builder import MATCH_COST

try:
    from scipy.optimize import linear_sum_assignment
except ImportError:
    linear_sum_assignment = None


@MATCH_COST.register_module()
class BBoxBEVL1Cost(object):
    def __init__(self, weight):
        self.weight = weight

    def __call__(self, bboxes, gt_bboxes, train_cfg):
        pc_start = bboxes.new(train_cfg["point_cloud_range"][0:2])
        pc_range = bboxes.new(train_cfg["point_cloud_range"][3:5]) - pc_start
        normalized_bboxes_xy = (bboxes[:, :2] - pc_start) / pc_range
        normalized_gt_bboxes_xy = (gt_bboxes[:, :2] - pc_start) / pc_range
        reg_cost = torch.cdist(normalized_bboxes_xy, normalized_gt_bboxes_xy, p=1)
        return reg_cost * self.weight


@MATCH_COST.register_module()
class IoU3DCost(object):
    def __init__(self, weight):
        self.weight = weight

    def __call__(self, iou):
        return -iou * self.weight


@BBOX_ASSIGNERS.register_module()
class HeuristicAssigner3D(BaseAssigner):
    def __init__(
        self,
        dist_thre=100,
        iou_calculator=dict(type="BboxOverlaps3D"),
    ):
        self.dist_thre = dist_thre
        self.iou_calculator = build_iou_calculator(iou_calculator)

    def assign(
        self,
        bboxes,
        gt_bboxes,
        gt_bboxes_ignore=None,
        gt_labels=None,
        query_labels=None,
    ):
        dist_thre = self.dist_thre
        num_gts, num_bboxes = len(gt_bboxes), len(bboxes)

        bev_dist = torch.norm(
            bboxes[:, 0:2][None, :, :] - gt_bboxes[:, 0:2][:, None, :], dim=-1
        )
        if query_labels is not None:
            bev_dist = bev_dist + (query_labels[None] != gt_labels[:, None]) * dist_thre

        nearest_values, nearest_indices = bev_dist.min(1)
        assigned_gt_inds = torch.zeros([num_bboxes], device=bboxes.device, dtype=bboxes.dtype)
        assigned_gt_vals = torch.ones([num_bboxes], device=bboxes.device, dtype=bboxes.dtype) * 10000
        assigned_gt_labels = torch.ones([num_bboxes], device=bboxes.device, dtype=bboxes.dtype) * -1
        for idx_gts in range(num_gts):
            idx_pred = nearest_indices[idx_gts]
            if bev_dist[idx_gts, idx_pred] <= dist_thre:
                if bev_dist[idx_gts, idx_pred] < assigned_gt_vals[idx_pred]:
                    assigned_gt_vals[idx_pred] = bev_dist[idx_gts, idx_pred]
                    assigned_gt_inds[idx_pred] = idx_gts + 1
                    assigned_gt_labels[idx_pred] = gt_labels[idx_gts]

        max_overlaps = torch.zeros([num_bboxes], device=bboxes.device, dtype=bboxes.dtype)
        matched_indices = torch.where(assigned_gt_inds > 0)
        if matched_indices[0].numel() > 0:
            matched_iou = self.iou_calculator(
                gt_bboxes[assigned_gt_inds[matched_indices].long() - 1],
                bboxes[matched_indices],
            ).diag()
            max_overlaps[matched_indices] = matched_iou

        return AssignResult(
            num_gts,
            assigned_gt_inds.long(),
            max_overlaps,
            labels=assigned_gt_labels.long(),
        )


@BBOX_ASSIGNERS.register_module()
class HungarianAssigner3D(BaseAssigner):
    def __init__(
        self,
        cls_cost=dict(type="ClassificationCost", weight=1.0),
        reg_cost=dict(type="BBoxBEVL1Cost", weight=1.0),
        iou_cost=dict(type="IoU3DCost", weight=1.0),
        iou_calculator=dict(type="BboxOverlaps3D"),
    ):
        self.cls_cost = build_match_cost(cls_cost)
        self.reg_cost = build_match_cost(reg_cost)
        self.iou_cost = build_match_cost(iou_cost)
        self.iou_calculator = build_iou_calculator(iou_calculator)

    def assign(self, bboxes, gt_bboxes, gt_labels, cls_pred, train_cfg):
        num_gts, num_bboxes = gt_bboxes.size(0), bboxes.size(0)
        assigned_gt_inds = bboxes.new_full((num_bboxes,), -1, dtype=torch.long)
        assigned_labels = bboxes.new_full((num_bboxes,), -1, dtype=torch.long)

        if num_gts == 0 or num_bboxes == 0:
            if num_gts == 0:
                assigned_gt_inds[:] = 0
            return AssignResult(num_gts, assigned_gt_inds, None, labels=assigned_labels)

        cls_cost = self.cls_cost(cls_pred[0].T, gt_labels)
        reg_cost = self.reg_cost(bboxes, gt_bboxes, train_cfg)
        iou = self.iou_calculator(bboxes, gt_bboxes)
        iou_cost = self.iou_cost(iou)
        cost = cls_cost + reg_cost + iou_cost

        cost = cost.detach().cpu()
        if linear_sum_assignment is None:
            raise ImportError('Please run "pip install scipy" to install scipy first.')
        matched_row_inds, matched_col_inds = linear_sum_assignment(cost)
        matched_row_inds = torch.from_numpy(matched_row_inds).to(bboxes.device)
        matched_col_inds = torch.from_numpy(matched_col_inds).to(bboxes.device)

        assigned_gt_inds[:] = 0
        assigned_gt_inds[matched_row_inds] = matched_col_inds + 1
        assigned_labels[matched_row_inds] = gt_labels[matched_col_inds]

        max_overlaps = torch.zeros_like(iou.max(1).values)
        max_overlaps[matched_row_inds] = iou[matched_row_inds, matched_col_inds]
        return AssignResult(num_gts, assigned_gt_inds, max_overlaps, labels=assigned_labels)
