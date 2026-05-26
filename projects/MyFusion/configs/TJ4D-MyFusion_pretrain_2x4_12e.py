# initialization
custom_imports = dict(imports=["projects.MyFusion.mmdet3d_plugin"])

# dataset settings
dataset_type = "TJ4DDataset"
data_root = "/home/yanzexin/MyFusion/data/TJ4D"
class_names = ["Pedestrian", "Cyclist", "Car", "Truck"]
input_modality = dict(use_lidar=True, use_camera=True)
file_client_args = dict(backend='disk')

point_cloud_range = [0, -39.68, -4, 69.12, 39.68, 2]
post_center_range = [x + y for x, y in zip(point_cloud_range, [-10, -10, -5, 10, 10, 5])]
voxel_size = [0.32, 0.32, 6.0]
grid_config = dict(
    xbound=[point_cloud_range[0], point_cloud_range[3], voxel_size[0]],
    ybound=[point_cloud_range[1], point_cloud_range[4], voxel_size[1]],
    zbound=[point_cloud_range[2], point_cloud_range[5], voxel_size[2]],
    dbound=[1.0, 73.0, 1.0],
)
code_weights = [2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
code_size = len(code_weights)

img_norm_cfg = dict(mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)

ida_aug_conf = dict(
    resize_lim=(0.60, 0.80),
    final_dim=(768, 1024),
    final_dim_test=(768, 1024),
    bot_pct_lim=(0.0, 0.0),
    top_pct_lim=(0.0, 0.3),
    rot_lim=(-2.7, 2.7),
    rand_flip=True,
)

# ida_aug_conf = dict(
#     resize_lim=(0.40, 0.50),
#     final_dim=(480, 640),
#     final_dim_test=(480, 640),
#     bot_pct_lim=(0.0, 0.0),
#     top_pct_lim=(0.0, 0.1),
#     rot_lim=(-2.7, 2.7),
#     rand_flip=True,
# )

bda_aug_conf = dict(
    rot_range=(-0.3925, 0.3925),
    scale_ratio_range=(0.95, 1.05),
    translation_std=(1.0, 1.0, 0.0),
    flip_dx_ratio=0.0,
    flip_dy_ratio=0.5,
)

train_pipeline = [
    dict(type="LoadPointsFromFile", coord_type="LIDAR", load_dim=8, use_dim=[0, 1, 2, 3, 5]),
    dict(type="LoadImageFromFile", to_float32=True),
    dict(type="loadSegmentation", data_root=data_root, dataset="TJ4D", seg_type="detectron2"),
    dict(type="LoadAnnotations3D", with_bbox_3d=True, with_label_3d=True, with_bbox=True, with_label=True),
    dict(type="ImageAug3D", data_aug_conf=ida_aug_conf, is_train=True),
    dict(type="GlobalRotScaleTransFlipAll", bda_aug_conf=bda_aug_conf, is_train=True),
    dict(type="PointsRangeFilter", point_cloud_range=point_cloud_range),
    dict(type="ObjectRangeFilter", point_cloud_range=point_cloud_range),
    dict(type="PointShuffle"),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="CreateDepthFromLiDAR", data_root=data_root, dataset="TJ4D"),
    dict(type="CreateDepthFromRaDAR", filter_min=0.0, filter_max=80.0),
    dict(type="gen2DMask", use_seg=False, use_softlabel=False, is_train=True),
    dict(type="DefaultFormatBundle3D", class_names=class_names),
    dict(type="CustomCollect3D", keys=["points", "img", "gt_bboxes_3d", "gt_labels_3d", "gt_bboxes", "gt_labels"]),
]

test_pipeline = [
    dict(type="LoadPointsFromFile", coord_type="LIDAR", load_dim=8, use_dim=[0, 1, 2, 3, 5]),
    dict(type="LoadImageFromFile", to_float32=True),
    dict(type="loadSegmentation", data_root=data_root, dataset="TJ4D", seg_type="detectron2"),
    dict(type="ImageAug3D", data_aug_conf=ida_aug_conf, is_train=False),
    dict(type="GlobalRotScaleTransFlipAll", bda_aug_conf=bda_aug_conf, is_train=False),
    dict(type="PointsRangeFilter", point_cloud_range=point_cloud_range),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="CreateDepthFromRaDAR", filter_min=0.0, filter_max=80.0),
    dict(type="gen2DMask", use_seg=False, use_softlabel=False, is_train=False),
    dict(type="DefaultFormatBundle3D", class_names=class_names, with_label=False),
    dict(type="CustomCollect3D", keys=["points", "img"]),
]

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type="RepeatDataset",
        times=2,
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            ann_file=data_root + "/TJ4D_infos_train.pkl",
            split="training",
            pts_prefix="velodyne_reduced",
            pipeline=train_pipeline,
            modality=input_modality,
            classes=class_names,
            test_mode=False,
            box_type_3d="LiDAR",
        ),
    ),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + "/TJ4D_infos_val.pkl",
        split="training",
        pts_prefix="velodyne_reduced",
        pipeline=test_pipeline,
        modality=input_modality,
        classes=class_names,
        test_mode=True,
        box_type_3d="LiDAR",
    ),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + "/TJ4D_infos_val.pkl",
        split="training",
        pts_prefix="velodyne_reduced",
        pipeline=test_pipeline,
        modality=input_modality,
        classes=class_names,
        test_mode=True,
        box_type_3d="LiDAR",
    ),
)

# model settings
model_point_cloud_range = [0, -39.68, -4, 69.12, 39.68, 2]
model_voxel_size = [0.32, 0.32, 6.0]
model_grid_config = dict(
    xbound=[model_point_cloud_range[0], model_point_cloud_range[3], model_voxel_size[0]],
    ybound=[model_point_cloud_range[1], model_point_cloud_range[4], model_voxel_size[1]],
    zbound=[model_point_cloud_range[2], model_point_cloud_range[5], model_voxel_size[2]],
    dbound=[1.0, 73.0, 1.0],
)
model_class_names = ["Pedestrian", "Cyclist", "Car", "Truck"]
model_img_norm_cfg = dict(mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)
# model_ida_aug_conf = dict(
#     resize_lim=(0.40, 0.50),
#     final_dim=(480, 640),
#     final_dim_test=(480, 640),
#     bot_pct_lim=(0.0, 0.0),
#     top_pct_lim=(0.0, 0.1),
#     rot_lim=(-2.7, 2.7),
#     rand_flip=True,
# )

model_ida_aug_conf = dict(
    resize_lim=(0.60, 0.80),
    final_dim=(768, 1024),
    final_dim_test=(768, 1024),
    bot_pct_lim=(0.0, 0.0),
    top_pct_lim=(0.0, 0.3),
    rot_lim=(-2.7, 2.7),
    rand_flip=True,
)

bev_h_ = int((model_point_cloud_range[3] - model_point_cloud_range[0]) / model_voxel_size[0])
bev_w_ = int((model_point_cloud_range[4] - model_point_cloud_range[1]) / model_voxel_size[1])
img_channels = 256
rad_channels = 384
downsample = 8
num_in_height = 8
dim_ = 256
loss_range_seg = 0.1

model = dict(
    type="LiteSGDet3D",
    bev_h_=bev_h_,
    bev_w_=bev_w_,
    img_channels=img_channels,
    rad_channels=rad_channels,
    num_classes=len(model_class_names),
    num_in_height=num_in_height,
    use_depth_supervision=True,
    use_props_supervision=True,
    use_box3d_supervision=False,
    use_sa_radarnet=False,
    use_msk2d_supervision=True,
    use_grid_mask=False,
    freeze_images=True,
    freeze_depths=False,
    freeze_radars=False,
    camera_stream="LSS",
    point_cloud_range=model_point_cloud_range,
    grid_config=model_grid_config,
    img_norm_cfg=model_img_norm_cfg,
    img_backbone=dict(
        type="ResNet",
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type="BN", requires_grad=False),
        norm_eval=True,
        style="caffe",
    ),
    img_neck=dict(
        type="FPN",
        in_channels=[256, 512, 1024, 2048],
        out_channels=img_channels,
        norm_cfg=dict(type="BN", requires_grad=False),
        num_outs=5,
    ),
    img_view_transformer=dict(
        type="ViewTransformerLSS",
        downsample=downsample,
        grid_config=model_grid_config,
        data_config=model_ida_aug_conf,
    ),
    depth_net=dict(
        type="GeometryDepth_Net",
        use_radar_depth=False,
        use_extra_depth=False,
        data_config=model_ida_aug_conf,
        downsample=downsample,
        input_channels=img_channels * 5,
        numC_input=dim_,
        numC_Trans=dim_,
        cam_channels=33,
        grid_config=model_grid_config,
        loss_depth_type="kld",
        loss_prob_weight=1.0,
        loss_abs_weight=0.02,
    ),
    rangeview_foreground=dict(
        type="MRF3Net",
        input_channel=dim_,
        output_channel=1,
        base_channel=dim_,
        mask_thre_train=0.95,
        mask_thre_test=0.70,
        loss_box=loss_range_seg,
        loss_seg=loss_range_seg,
    ),
    pts_voxel_layer=dict(
        max_num_points=10,
        point_cloud_range=model_point_cloud_range,
        voxel_size=[model_voxel_size[0] / 2, model_voxel_size[1] / 2, model_voxel_size[2]],
        max_voxels=(16000, 40000),
    ),
    pts_voxel_encoder=dict(
        type="RadarPillarFeatureNet",
        in_channels=5,
        feat_channels=[64],
        with_distance=False,
        voxel_size=[model_voxel_size[0] / 2, model_voxel_size[1] / 2, model_voxel_size[2]],
        point_cloud_range=model_point_cloud_range,
        legacy=False,
        with_velocity_snr_center=True,
    ),
    pts_middle_encoder=dict(
        type="PointPillarsScatter",
        in_channels=64,
        output_shape=[bev_w_ * 2, bev_h_ * 2],
    ),
    pts_backbone=dict(
        type="SECOND",
        in_channels=64,
        layer_nums=[3, 5, 5],
        layer_strides=[2, 2, 2],
        out_channels=[64, 128, 256],
    ),
    pts_neck=dict(
        type="SECONDFPN",
        in_channels=[64, 128, 256],
        upsample_strides=[1, 2, 4],
        out_channels=[128, 128, 128],
    ),
    RCFusion=dict(
        type="Cross_Modal_Fusion",
        kernel_size=3,
        img_channels=dim_,
        rad_channels=rad_channels,
        out_channels=dim_,
    ),
    proposal_layer=dict(
        type="FRPN",
        in_channels=dim_,
        scale_factor=1.0,
        mask_thre=0.4,
        topk_rate_test=0.01,
        loss_weight=1.0,
    ),
    pts_bbox_head=dict(
        type="Anchor3DHead",
        num_classes=len(model_class_names),
        in_channels=dim_,
        feat_channels=dim_,
        use_direction_classifier=True,
        anchor_generator=dict(
            type="Anchor3DRangeGenerator",
            ranges=[
                [0, -40.0, -1.163, 70.4, 40.0, -1.163],
                [0, -40.0, -1.353, 70.4, 40.0, -1.353],
                [0, -40.0, -1.363, 70.4, 40.0, -1.363],
                [0, -40.0, -1.403, 70.4, 40.0, -1.403],
            ],
            sizes=[[0.6, 0.8, 1.69], [0.78, 1.77, 1.60], [1.84, 4.56, 1.70], [2.66, 10.76, 3.47]],
            rotations=[0, 1.57],
            reshape_out=False,
        ),
        assigner_per_size=True,
        diff_rad_by_sin=True,
        assign_per_class=True,
        bbox_coder=dict(type="DeltaXYZWLHRBBoxCoder"),
        loss_cls=dict(type="FocalLoss", use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=1.0),
        loss_bbox=dict(type="SmoothL1Loss", beta=1.0 / 9.0, loss_weight=2.0),
        loss_dir=dict(type="CrossEntropyLoss", use_sigmoid=False, loss_weight=0.2),
    ),
    train_cfg=dict(
        pts=dict(
            assigner=[
                dict(
                    type="MaxIoUAssigner",
                    iou_calculator=dict(type="BboxOverlapsNearest3D"),
                    pos_iou_thr=0.35,
                    neg_iou_thr=0.2,
                    min_pos_iou=0.2,
                    ignore_iof_thr=-1,
                ),
                dict(
                    type="MaxIoUAssigner",
                    iou_calculator=dict(type="BboxOverlapsNearest3D"),
                    pos_iou_thr=0.35,
                    neg_iou_thr=0.2,
                    min_pos_iou=0.2,
                    ignore_iof_thr=-1,
                ),
                dict(
                    type="MaxIoUAssigner",
                    iou_calculator=dict(type="BboxOverlapsNearest3D"),
                    pos_iou_thr=0.5,
                    neg_iou_thr=0.35,
                    min_pos_iou=0.35,
                    ignore_iof_thr=-1,
                ),
                dict(
                    type="MaxIoUAssigner",
                    iou_calculator=dict(type="BboxOverlapsNearest3D"),
                    pos_iou_thr=0.5,
                    neg_iou_thr=0.35,
                    min_pos_iou=0.35,
                    ignore_iof_thr=-1,
                ),
            ],
            allowed_border=0,
            pos_weight=-1,
            debug=False,
        )
    ),
    test_cfg=dict(
        pts=dict(
            use_rotate_nms=True,
            nms_across_levels=False,
            nms_thr=0.01,
            score_thr=0.1,
            min_bbox_size=0,
            nms_pre=100,
            max_num=50,
        )
    ),
)

# training settings
optimizer = dict(type="AdamW", lr=1e-4, betas=(0.95, 0.99), weight_decay=0.01)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
runner = dict(type="EpochBasedRunner", max_epochs=12)
lr_config = dict(
    policy="CosineAnnealing",
    warmup=None,
    warmup_iters=500,
    warmup_ratio=0.1,
    min_lr_ratio=1e-5,
)
momentum_config = None
evaluation = dict(interval=1)
checkpoint_config = dict(interval=1)
log_config = dict(
    interval=10,
    hooks=[dict(type="TextLoggerHook"), dict(type="TensorboardLoggerHook")],
)
dist_params = dict(backend="nccl")
log_level = "INFO"
load_from = None
resume_from = None
workflow = [("train", 1)]

# TJ4D pretrain stage settings
load_from = "/home/yanzexin/SGDet3D/projects/SGDet3D/checkpoints/image_pretrained/mvx_faster_rcnn_detectron2-caffe_20e_coco-pretrain_gt-sample_kitti-3-class_moderate-79.3_20200207-a4a6a3c7.pth"
load_radar_from = "/home/yanzexin/SGDet3D/projects/SGDet3D/checkpoints/vod-radarpillarnet_modified_4x1_80e/epoch_80.pth"
load_img_from = None
resume_from = None

max_epochs = 12
runner = dict(type="EpochBasedRunner", max_epochs=max_epochs)
evaluation = dict(interval=max_epochs)
checkpoint_config = dict(interval=2)
log_config = dict(
    interval=50,
    hooks=[dict(type="TextLoggerHook"), dict(type="TensorboardLoggerHook")],
)
