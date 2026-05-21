from typing import Any, Dict

import numpy as np
import torch
from PIL import Image
from mmdet.datasets.builder import PIPELINES


@PIPELINES.register_module()
class ImageAug3D:
    def __init__(self, data_aug_conf, is_train):
        if not is_train:
            data_aug_conf = dict(
                resize_lim=(1.0, 1.0),
                final_dim=data_aug_conf["final_dim_test"],
                bot_pct_lim=(0.0, 0.0),
                top_pct_lim=(0.0, 0.0),
                rot_lim=(0.0, 0.0),
                rand_flip=False,
            )
        self.data_aug_conf = data_aug_conf
        self.final_dim = data_aug_conf["final_dim"]
        self.resize_lim = data_aug_conf["resize_lim"]
        self.top_pct_lim = data_aug_conf["top_pct_lim"]
        self.rand_flip = data_aug_conf["rand_flip"]
        self.rot_lim = data_aug_conf["rot_lim"]
        self.is_train = is_train

    def sample_augmentation(self, results):
        h, w = results["ori_shape"][:2]
        final_h, final_w = self.final_dim
        results["img_shape"] = (final_h, final_w)
        if self.is_train:
            resize = np.random.uniform(*self.resize_lim)
            resize_dims = (int(w * resize), int(h * resize))
            new_w, new_h = resize_dims
            crop_h = int(np.random.uniform(*self.top_pct_lim) * new_h)
            crop_w = int(np.random.uniform(0, max(0, new_w - final_w)))
            crop = (crop_w, crop_h, crop_w + final_w, crop_h + final_h)
            flip = bool(self.rand_flip and np.random.choice([0, 1]))
            rotate = np.random.uniform(*self.rot_lim)
        else:
            resize = min(final_h / h, final_w / w)
            resize_dims = (int(w * resize), int(h * resize))
            new_w, new_h = resize_dims
            crop_h = int(np.random.uniform(*self.top_pct_lim) * new_h)
            crop_w = int(max(0, new_w - final_w) / 2)
            crop = (crop_w, crop_h, crop_w + final_w, crop_h + final_h)
            flip = False
            rotate = 0.0
        return resize, resize_dims, crop, flip, rotate

    def transform_boxes(self, boxes, resize, crop, flip):
        if boxes is None or len(boxes) == 0:
            return boxes, None

        boxes = boxes.copy()
        boxes *= resize
        boxes[:, [0, 2]] -= crop[0]
        boxes[:, [1, 3]] -= crop[1]
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, self.final_dim[1])
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, self.final_dim[0])

        valid = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
        boxes = boxes[valid]

        if flip and len(boxes) > 0:
            x1 = boxes[:, 0].copy()
            x2 = boxes[:, 2].copy()
            boxes[:, 0] = self.final_dim[1] - x2
            boxes[:, 2] = self.final_dim[1] - x1

        return boxes, valid

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        img = data["img"]
        focal_length = data["focal_length"]
        baseline = data["baseline"]

        post_rot = torch.eye(2)
        post_tran = torch.zeros(2)

        resize, resize_dims, crop, flip, rotate = self.sample_augmentation(data)

        pil = Image.fromarray(img.astype("uint8"))
        pil = pil.resize(resize_dims)
        pil = pil.crop(crop)
        if flip:
            pil = pil.transpose(method=Image.FLIP_LEFT_RIGHT)
        if rotate != 0:
            pil = pil.rotate(rotate)

        post_rot *= resize
        post_tran -= torch.tensor(crop[:2], dtype=torch.float32)
        if flip:
            a = torch.tensor([[-1.0, 0.0], [0.0, 1.0]])
            b = torch.tensor([crop[2] - crop[0], 0.0], dtype=torch.float32)
            post_rot = a.matmul(post_rot)
            post_tran = a.matmul(post_tran) + b

        theta = rotate / 180.0 * np.pi
        a = torch.tensor(
            [[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]],
            dtype=torch.float32,
        )
        b = torch.tensor(
            [crop[2] - crop[0], crop[3] - crop[1]], dtype=torch.float32
        ) / 2
        b = a.matmul(-b) + b
        post_rot = a.matmul(post_rot)
        post_tran = a.matmul(post_tran) + b

        if "gt_bboxes" in data and data["gt_bboxes"] is not None:
            boxes, valid = self.transform_boxes(
                data["gt_bboxes"], resize=resize, crop=crop, flip=flip
            )
            data["gt_bboxes"] = boxes
            if valid is not None:
                data["gt_labels"] = data["gt_labels"][valid]
                data["gt_bboxes_3d"] = data["gt_bboxes_3d"][valid]
                data["gt_labels_3d"] = data["gt_labels_3d"][valid]

        if "segmentation" in data:
            seg = Image.fromarray(data["segmentation"])
            seg = seg.resize(resize_dims)
            seg = seg.crop(crop)
            if flip:
                seg = seg.transpose(method=Image.FLIP_LEFT_RIGHT)
            if rotate != 0:
                seg = seg.rotate(rotate)
            data["segmentation"] = np.array(seg).astype(np.bool_)

        data["img"] = np.array(pil).astype(np.float32)
        data["pad_shape"] = data["img"].shape
        data["scale_factor"] = 1.0
        data["flip"] = flip

        transform = torch.eye(4)
        transform[:2, :2] = post_rot
        transform[:2, 3] = post_tran
        data["img_aug_matrix"] = transform.numpy()

        post_rot3 = torch.eye(3)
        post_tran3 = torch.zeros(3)
        post_rot3[:2, :2] = post_rot
        post_tran3[:2] = post_tran

        intrin = torch.as_tensor(data["cam2img"], dtype=torch.float32)
        lidar2cam = torch.as_tensor(data["lidar2cam"], dtype=torch.float32)
        cam2lidar = torch.inverse(lidar2cam)
        rot = cam2lidar[:3, :3]
        tran = cam2lidar[:3, 3]

        cam_aware = [
            rot,
            tran,
            intrin,
            post_rot3,
            post_tran3,
            torch.zeros(1),
            cam2lidar,
            torch.tensor(float(focal_length), dtype=torch.float32),
            torch.tensor(float(baseline), dtype=torch.float32),
        ]
        data["cam_aware"] = cam_aware
        return data
