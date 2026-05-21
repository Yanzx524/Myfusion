from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models import DETECTORS
from mmdet3d.models import builder
from mmdet3d.models.builder import FUSION_LAYERS
from mmdet3d.models.detectors import MVXFasterRCNN
from shapely.geometry import Polygon, box


@DETECTORS.register_module()
class LiteSGDet3D(MVXFasterRCNN):
    def __init__(
        self,
        bev_h_=160,
        bev_w_=160,
        img_channels=256,
        rad_channels=384,
        num_classes=3,
        num_in_height=8,
        use_depth_supervision=True,
        use_props_supervision=True,
        use_box3d_supervision=True,
        use_sa_radarnet=False,
        use_msk2d_supervision=False,
        use_grid_mask=False,
        freeze_images=True,
        freeze_depths=False,
        freeze_radars=False,
        camera_stream="LSS",
        point_cloud_range=None,
        grid_config=None,
        img_norm_cfg=None,
        depth_net=None,
        rangeview_foreground=None,
        img_view_transformer=None,
        RCFusion=None,
        proposal_layer=None,
        **kwargs
    ):
        self.pts_bbox_head_cfg = kwargs.pop("pts_bbox_head", None)
        super().__init__(**kwargs)

        self.bev_h_ = bev_h_
        self.bev_w_ = bev_w_
        self.img_channels = img_channels
        self.rad_channels = rad_channels
        self.num_classes = num_classes
        self.num_in_height = num_in_height
        self.use_depth_supervision = use_depth_supervision
        self.use_props_supervision = use_props_supervision
        self.use_box3d_supervision = use_box3d_supervision
        self.use_sa_radarnet = use_sa_radarnet
        self.use_msk2d_supervision = use_msk2d_supervision
        self.use_grid_mask = use_grid_mask
        self.freeze_images = freeze_images
        self.freeze_depths = freeze_depths
        self.freeze_radars = freeze_radars
        self.lift_method = camera_stream
        self.point_cloud_range = point_cloud_range
        self.grid_config = grid_config
        self.img_norm_cfg = img_norm_cfg
        self.depth_net = FUSION_LAYERS.build(depth_net) if depth_net is not None else None
        self.rangeview_foreground = (
            FUSION_LAYERS.build(rangeview_foreground)
            if rangeview_foreground is not None and use_msk2d_supervision
            else None
        )
        if img_view_transformer is not None:
            img_view_transformer = dict(img_view_transformer)
            img_view_transformer.setdefault("num_in_height", self.num_in_height)
            self.img_view_transformer = FUSION_LAYERS.build(img_view_transformer)
        else:
            self.img_view_transformer = None
        self.cross_attention = FUSION_LAYERS.build(RCFusion) if RCFusion is not None else None
        self.proposal_layer_former = (
            FUSION_LAYERS.build(proposal_layer) if proposal_layer is not None and use_props_supervision else None
        )

        self.xbound = self.grid_config["xbound"]
        self.ybound = self.grid_config["ybound"]
        self.bev_grid_shape = [bev_h_, bev_w_]
        self.bev_cell_size = [
            (self.xbound[1] - self.xbound[0]) / bev_h_,
            (self.ybound[1] - self.ybound[0]) / bev_w_,
        ]
        self.downsample = self.depth_net.downsample if self.depth_net is not None else 8

        if self.pts_bbox_head_cfg is not None and self.use_box3d_supervision:
            pts_train_cfg = self.train_cfg.pts if self.train_cfg else None
            pts_test_cfg = self.test_cfg.pts if self.test_cfg else None
            head_cfg = dict(self.pts_bbox_head_cfg)
            head_cfg.update(train_cfg=pts_train_cfg, test_cfg=pts_test_cfg)
            self.pts_bbox_head = builder.build_head(head_cfg)
        else:
            self.pts_bbox_head = None

        self.init_weights()
        if self.freeze_images:
            self.freeze_img_model()
        if self.freeze_radars:
            self.freeze_pts_model()

    def freeze_img_model(self):
        if self.with_img_backbone:
            for param in self.img_backbone.parameters():
                param.requires_grad = False
        if self.with_img_neck:
            for param in self.img_neck.parameters():
                param.requires_grad = False
        if self.depth_net is not None and self.freeze_depths:
            for param in self.depth_net.parameters():
                param.requires_grad = False

    def freeze_pts_model(self):
        if self.pts_voxel_encoder is not None:
            for param in self.pts_voxel_encoder.parameters():
                param.requires_grad = False
        if self.pts_middle_encoder is not None:
            for param in self.pts_middle_encoder.parameters():
                param.requires_grad = False
        if self.pts_backbone is not None:
            for param in self.pts_backbone.parameters():
                param.requires_grad = False
        if self.pts_neck is not None:
            for param in self.pts_neck.parameters():
                param.requires_grad = False

    def _normalize_img_metas(self, img_metas):
        if isinstance(img_metas, dict):
            return [img_metas]
        if isinstance(img_metas, tuple):
            img_metas = list(img_metas)
        while isinstance(img_metas, list) and len(img_metas) == 1 and isinstance(img_metas[0], (list, tuple)):
            img_metas = list(img_metas[0])
        return img_metas

    def _normalize_points(self, points):
        if isinstance(points, torch.Tensor):
            return [points]
        if isinstance(points, tuple):
            return list(points)
        return points

    def _normalize_img(self, img):
        if img is None:
            return None
        if img.dim() == 3:
            return img.unsqueeze(0)
        return img

    def _normalize_gt(self, value):
        if value is None:
            return None
        if isinstance(value, tuple):
            return list(value)
        if isinstance(value, list):
            return value
        return [value]

    def _stack_camera_params(self, img_metas, device):
        rots, trans, intrins, post_rots, post_trans, bda = [], [], [], [], [], []
        for meta in img_metas:
            cam_aware = meta["cam_aware"]
            rots.append(torch.as_tensor(cam_aware[0], dtype=torch.float32, device=device))
            trans.append(torch.as_tensor(cam_aware[1], dtype=torch.float32, device=device))
            intrins.append(torch.as_tensor(cam_aware[2], dtype=torch.float32, device=device))
            post_rots.append(torch.as_tensor(cam_aware[3], dtype=torch.float32, device=device))
            post_trans.append(torch.as_tensor(cam_aware[4], dtype=torch.float32, device=device))
            bda_item = meta.get("bda_rot", np.eye(4, dtype=np.float32))
            bda.append(torch.as_tensor(bda_item, dtype=torch.float32, device=device))
        return (
            torch.stack(rots, dim=0).unsqueeze(1),
            torch.stack(trans, dim=0).unsqueeze(1),
            torch.stack(intrins, dim=0).unsqueeze(1),
            torch.stack(post_rots, dim=0).unsqueeze(1),
            torch.stack(post_trans, dim=0).unsqueeze(1),
            torch.stack(bda, dim=0),
        )

    def _prepare_aux_tensors(self, img_metas, device, image_hw, feat_hw):
        gt_depths = None
        if all("gt_depths" in meta for meta in img_metas):
            gt_depths = torch.stack(
                [torch.as_tensor(meta["gt_depths"], dtype=torch.float32, device=device) for meta in img_metas],
                dim=0,
            ).unsqueeze(1)
        if all("radar_depth" in meta for meta in img_metas):
            radar_depth = torch.stack(
                [torch.as_tensor(meta["radar_depth"], dtype=torch.float32, device=device) for meta in img_metas],
                dim=0,
            ).unsqueeze(1)
        else:
            radar_depth = torch.zeros((len(img_metas), 1, image_hw[0], image_hw[1]), device=device)

        if all("bbox_Mask" in meta for meta in img_metas):
            bbox_mask = torch.stack(
                [torch.as_tensor(meta["bbox_Mask"], dtype=torch.float32, device=device) for meta in img_metas],
                dim=0,
            ).unsqueeze(1)
            bbox_mask = F.interpolate(bbox_mask, feat_hw, mode="bilinear", align_corners=True)
        else:
            bbox_mask = torch.zeros((len(img_metas), 1, feat_hw[0], feat_hw[1]), device=device)

        if all("segmentation" in meta for meta in img_metas):
            segmentation = torch.stack(
                [torch.as_tensor(meta["segmentation"], dtype=torch.float32, device=device) for meta in img_metas],
                dim=0,
            ).unsqueeze(1)
            segmentation = F.interpolate(segmentation, feat_hw, mode="bilinear", align_corners=True)
        else:
            segmentation = torch.zeros((len(img_metas), 1, feat_hw[0], feat_hw[1]), device=device)

        return gt_depths, radar_depth, bbox_mask, segmentation

    def extract_pts_feat(self, pts, img_metas):
        if not self.with_pts_backbone:
            return None
        voxels, num_points, coors = self.voxelize(pts)
        voxel_features = self.pts_voxel_encoder(voxels, num_points, coors)
        batch_size = coors[-1, 0].item() + 1
        x = self.pts_middle_encoder(voxel_features, coors, batch_size)
        x = self.pts_backbone(x)
        if self.with_pts_neck:
            x = self.pts_neck(x)
        return x

    def extract_img_feat(self, img, img_metas):
        if not self.with_img_backbone or img is None:
            return None
        input_shape = img.shape[-2:]
        for img_meta in img_metas:
            img_meta.update(input_shape=input_shape)
        if img.dim() == 4:
            batch_size, num_cams = img.shape[0], 1
            img = img.unsqueeze(1)
        else:
            batch_size, num_cams = img.shape[:2]
        img = img.view(batch_size * num_cams, *img.shape[-3:])
        img_feats = self.img_backbone(img)
        if self.with_img_neck:
            img_feats = self.img_neck(img_feats)
        if isinstance(img_feats, torch.Tensor):
            img_feats = [img_feats]
        outputs = []
        for feat in img_feats:
            _, c, h, w = feat.shape
            outputs.append(feat.view(batch_size, num_cams, c, h, w))
        return outputs

    def extract_feat(self, points, img, img_metas):
        points = self._normalize_points(points)
        img_metas = self._normalize_img_metas(img_metas)
        img = self._normalize_img(img)
        img_feats = self.extract_img_feat(img, img_metas)
        if img_feats is None:
            raise ValueError("LiteSGDet3D requires image features.")

        batch_size = img_feats[0].shape[0]
        num_cams = img_feats[0].shape[1]
        h = img.shape[-2] // self.downsample
        w = img.shape[-1] // self.downsample
        align_feats = []
        for feat in img_feats:
            b, n, c, fh, fw = feat.shape
            feat = feat.view(b * n, c, fh, fw)
            feat = F.interpolate(feat, (h, w), mode="bilinear", align_corners=True)
            align_feats.append(feat.view(b, n, c, h, w))
        align_feats = torch.cat(align_feats, dim=2)

        cam_params = self._stack_camera_params(img_metas, img.device)
        rots, trans, intrins, post_rots, post_trans, bda = cam_params
        gt_depths, radar_depth, bbox_mask, segmentation = self._prepare_aux_tensors(
            img_metas, img.device, img.shape[-2:], (h, w)
        )
        mlp_input = self.depth_net.get_mlp_input(
            rots[:, 0], trans[:, 0], intrins[:, 0], post_rots[:, 0], post_trans[:, 0], bda
        )
        context, depth = self.depth_net(
            [align_feats.view(batch_size * num_cams, -1, h, w), rots, trans, intrins, post_rots, post_trans, bda, mlp_input],
            radar_depth,
            None,
            img_metas,
        )
        img_bev_feats = self.img_view_transformer(context, depth, cam_params)
        img_bev_feats = img_bev_feats.mean(-1).permute(0, 1, 3, 2).contiguous()

        raw_depth = torch.arange(
            self.grid_config["dbound"][0],
            self.grid_config["dbound"][1],
            self.grid_config["dbound"][2],
            dtype=depth.dtype,
            device=depth.device,
        )
        precise_depth = torch.sum(raw_depth.view(1, -1, 1, 1) * depth, dim=1).unsqueeze(1)

        rangeview_logit = None
        if self.rangeview_foreground is not None and self.use_msk2d_supervision:
            rangeview_logit = self.rangeview_foreground(context[:, 0])

        pts_bev_feats = self.extract_pts_feat(points, img_metas)
        if isinstance(pts_bev_feats, (list, tuple)):
            pts_bev_feats = pts_bev_feats[0]
        bev_feats = self.cross_attention(img_bev_feats, pts_bev_feats)
        bev_feats = bev_feats.permute(0, 1, 3, 2).contiguous()

        bev_mask_logit_former = None
        if self.proposal_layer_former is not None and self.use_props_supervision:
            bev_mask_logit_former = self.proposal_layer_former(bev_feats)

        bev_feats_refined = bev_feats
        bev_mask_logit_latter = None
        bev_feats_out = bev_feats_refined.permute(0, 1, 3, 2).contiguous()
        return dict(
            img_feats=img_feats,
            pts_feats=[bev_feats_out],
            pd_depths=depth,
            gt_depths=gt_depths,
            precise_depth=precise_depth,
            bbox_Mask=bbox_mask,
            segmentation=segmentation,
            rangeview_logit=rangeview_logit,
            bev_mask_logit={"former": bev_mask_logit_former, "latter": bev_mask_logit_latter},
        )

    def forward_pts_train(self, pts_feats, gt_bboxes_3d, gt_labels_3d, img_metas, gt_bboxes_ignore=None):
        outs = self.pts_bbox_head(pts_feats)
        loss_inputs = outs + (gt_bboxes_3d, gt_labels_3d, img_metas)
        losses = self.pts_bbox_head.loss(*loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        return losses, outs

    def generate_bev_mask(self, gt_bboxes_3d, batch_size, device, occ_threshold=0.3):
        gt_bev_mask = []
        if len(gt_bboxes_3d) == 0:
            return torch.zeros((batch_size, 1, self.bev_h_, self.bev_w_), dtype=torch.bool, device=device)
        bev_cell_size = torch.tensor(self.bev_cell_size, device=device)
        for bsid in range(batch_size):
            bev_mask = torch.zeros(self.bev_grid_shape, dtype=torch.bool, device=device)
            valid_boxes = gt_bboxes_3d[bsid]
            if hasattr(valid_boxes, "tensor") and len(valid_boxes.tensor) > 0:
                corners = valid_boxes.corners[:, [0, 2, 4, 6], :2].to(device)
                corners[:, :, 0] = (corners[:, :, 0] - self.xbound[0]) / bev_cell_size[0]
                corners[:, :, 1] = (corners[:, :, 1] - self.ybound[0]) / bev_cell_size[1]
                grid_min = torch.clip(
                    torch.floor(torch.min(corners, axis=1).values).to(torch.int64),
                    0,
                    self.bev_grid_shape[0] - 1,
                )
                grid_max = torch.clip(
                    torch.ceil(torch.max(corners, axis=1).values).to(torch.int64),
                    0,
                    self.bev_grid_shape[1] - 1,
                )
                for n in range(corners.shape[0]):
                    poly = Polygon(corners[n].detach().cpu().numpy()[(0, 1, 3, 2), :])
                    h_list = np.arange(grid_min[n, 0].item() - 1, grid_max[n, 0].item() + 1, 1)
                    w_list = np.arange(grid_min[n, 1].item() - 1, grid_max[n, 1].item() + 1, 1)
                    h_list = np.clip(h_list, 0, self.bev_grid_shape[0] - 1)
                    w_list = np.clip(w_list, 0, self.bev_grid_shape[1] - 1)
                    for h_idx in h_list:
                        for w_idx in w_list:
                            cell = box(h_idx, w_idx, h_idx + 1, w_idx + 1)
                            if poly.intersection(cell).area > occ_threshold:
                                bev_mask[h_idx, w_idx] = True
            gt_bev_mask.append(bev_mask.unsqueeze(0))
        return torch.stack(gt_bev_mask, dim=0)

    def forward_train(
        self,
        points=None,
        img_metas=None,
        gt_bboxes_3d=None,
        gt_labels_3d=None,
        gt_labels=None,
        gt_bboxes=None,
        img=None,
        proposals=None,
        gt_bboxes_ignore=None,
        **kwargs
    ):
        img_metas = self._normalize_img_metas(img_metas)
        gt_bboxes_3d = self._normalize_gt(gt_bboxes_3d)
        gt_labels_3d = self._normalize_gt(gt_labels_3d)
        gt_bboxes = self._normalize_gt(gt_bboxes)
        gt_labels = self._normalize_gt(gt_labels)
        feature_dict = self.extract_feat(points, img=img, img_metas=img_metas)
        losses = {}
        if self.use_box3d_supervision and self.pts_bbox_head is not None and gt_bboxes_3d is not None:
            losses_pts, _ = self.forward_pts_train(
                feature_dict["pts_feats"], gt_bboxes_3d, gt_labels_3d, img_metas, gt_bboxes_ignore
            )
            losses.update(losses_pts)
        if self.use_depth_supervision and self.depth_net is not None:
            losses.update(
                self.depth_net.get_depth_loss(
                    feature_dict["gt_depths"],
                    feature_dict["pd_depths"],
                    feature_dict["precise_depth"],
                )
            )
        if self.use_props_supervision and self.proposal_layer_former is not None and gt_bboxes_3d is not None:
            gt_bev_mask = self.generate_bev_mask(gt_bboxes_3d, len(img_metas), feature_dict["pts_feats"][0].device)
            if feature_dict["bev_mask_logit"]["former"] is not None:
                losses_prop = self.proposal_layer_former.get_bev_mask_loss(
                    gt_bev_mask, feature_dict["bev_mask_logit"]["former"]
                )
                losses.update({f"{k}_former": v for k, v in losses_prop.items()})
        if (
            self.use_msk2d_supervision
            and self.rangeview_foreground is not None
            and feature_dict["rangeview_logit"] is not None
        ):
            losses.update(
                self.rangeview_foreground.get_range_view_mask_loss(
                    feature_dict["bbox_Mask"],
                    feature_dict["segmentation"],
                    feature_dict["rangeview_logit"],
                )
            )
        return losses

    def _format_log_vars(self, log_vars):
        formatted = OrderedDict()
        det_loss_total = 0.0
        aux_loss_total = 0.0
        for key, value in log_vars.items():
            formatted[key] = value
            if key in {"loss_cls", "loss_bbox", "loss_dir"} or key.startswith("loss_"):
                if "depth" in key or "bev_mask" in key or "range" in key:
                    aux_loss_total += float(value)
                elif key != "loss":
                    det_loss_total += float(value)
        if det_loss_total > 0:
            formatted["loss_det"] = det_loss_total
        if aux_loss_total > 0:
            formatted["loss_aux"] = aux_loss_total
        if "loss" in formatted:
            formatted["loss_total"] = formatted["loss"]
        return formatted

    def train_step(self, data, optimizer):
        losses = self(**data)
        loss, log_vars = self._parse_losses(losses)
        outputs = dict(
            loss=loss,
            log_vars=self._format_log_vars(OrderedDict(log_vars)),
            num_samples=len(data["img_metas"]),
        )
        return outputs

    def forward_test(self, points, img_metas, img=None, **kwargs):
        for key in [
            "gt_bboxes_3d",
            "gt_labels_3d",
            "gt_bboxes",
            "gt_labels",
            "gt_bboxes_ignore",
            "proposals",
        ]:
            kwargs.pop(key, None)
        norm_points = self._normalize_points(points)
        norm_metas = self._normalize_img_metas(img_metas)
        if img is None:
            norm_img = None
        else:
            norm_img = self._normalize_img(img)
        return self.simple_test(
            norm_points,
            norm_metas,
            img=norm_img,
            rescale=kwargs.pop("rescale", False),
            **kwargs
        )

    def simple_test(self, points, img_metas, img=None, rescale=False, **kwargs):
        feature_dict = self.extract_feat(points, img=img, img_metas=img_metas)
        bbox_list = [dict() for _ in range(len(self._normalize_img_metas(img_metas)))]
        if self.pts_bbox_head is not None and self.use_box3d_supervision:
            bbox_pts, _ = self.simple_test_pts(feature_dict["pts_feats"], img_metas, rescale=rescale)
            for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
                result_dict["pts_bbox"] = pts_bbox
        return bbox_list
