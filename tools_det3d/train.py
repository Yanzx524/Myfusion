from __future__ import division

import argparse
import copy
import os
import shutil
import time
import warnings
from collections import OrderedDict
from os import path as osp

import mmcv
import torch
from mmcv import Config, DictAction
from mmcv.runner import get_dist_info, init_dist

from mmdet import __version__ as mmdet_version
from mmdet.apis import set_random_seed
from mmdet3d import __version__ as mmdet3d_version
from mmdet3d.apis import train_model
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model
from mmdet3d.utils import collect_env, get_root_logger
from mmseg import __version__ as mmseg_version


def parse_args():
    parser = argparse.ArgumentParser(description="Train LiteSGDet3D")
    parser.add_argument("--config", required=True, help="train config file path")
    parser.add_argument("--work-dir", help="the dir to save logs and models")
    parser.add_argument("--resume-from", help="the checkpoint file to resume from")
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="whether not to evaluate the checkpoint during training",
    )
    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument(
        "--gpus",
        type=int,
        help="number of gpus to use (only applicable to non-distributed training)",
    )
    group_gpus.add_argument(
        "--gpu-ids",
        type=int,
        nargs="+",
        help="ids of gpus to use (only applicable to non-distributed training)",
    )
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="whether to set deterministic options for CUDNN backend.",
    )
    parser.add_argument(
        "--options",
        nargs="+",
        action=DictAction,
        help="override some settings in the used config, the key-value pair in xxx=yyy format will be merged into config file (deprecated), change to --cfg-options instead.",
    )
    parser.add_argument(
        "--cfg-options",
        nargs="+",
        action=DictAction,
        help="override some settings in the used config, the key-value pair in xxx=yyy format will be merged into config file.",
    )
    parser.add_argument(
        "--launcher",
        choices=["none", "pytorch", "slurm", "mpi"],
        default="none",
        help="job launcher",
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument(
        "--autoscale-lr",
        action="store_true",
        help="automatically scale lr with the number of gpus",
    )
    args = parser.parse_args()

    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = str(args.local_rank)

    if args.options and args.cfg_options:
        raise ValueError(
            "--options and --cfg-options cannot be both specified, --options is deprecated in favor of --cfg-options"
        )
    if args.options:
        warnings.warn("--options is deprecated in favor of --cfg-options")
        args.cfg_options = args.options

    return args


def project_name_from_config(config_path):
    return osp.splitext(osp.basename(config_path))[0]


def resolve_sgdet3d_root():
    env_root = os.environ.get("SGDET3D_ROOT")
    if env_root and osp.isdir(env_root):
        return osp.abspath(env_root)

    candidates = [
        "/home/yanzexin/SGDet3D",
        osp.abspath(osp.join(osp.dirname(__file__), "..", "..", "SGDet3D")),
    ]
    for path in candidates:
        if osp.isdir(path):
            return path
    return None


def maybe_prepare_eval_tools(project):
    dataset_key = None
    project_lower = project.lower()
    if "tj4d" in project_lower:
        dataset_key = "TJ4D"
    elif "vod" in project_lower:
        dataset_key = "vod"
    if dataset_key is None:
        return

    sgdet3d_root = resolve_sgdet3d_root()
    if sgdet3d_root is None:
        print("SGDET3D_ROOT not found, skip dataset-specific eval tool patching.")
        return

    if dataset_key == "TJ4D":
        eval_src = osp.join(sgdet3d_root, "tools_det3d", "eval_tools", "TJ4D-eval.py")
        dataset_src = osp.join(
            sgdet3d_root, "tools_det3d", "eval_tools", "TJ4D-kitti_dataset.py"
        )
        print("USING EVAL TOOLS OF TJ4D DATASET")
    else:
        eval_src = osp.join(sgdet3d_root, "tools_det3d", "eval_tools", "vod-eval.py")
        dataset_src = osp.join(
            sgdet3d_root, "tools_det3d", "eval_tools", "vod-kitti_dataset.py"
        )
        print("USING EVAL TOOLS OF VOD DATASET")

    eval_dst = osp.join(sgdet3d_root, "mmdet3d", "core", "evaluation", "kitti_utils", "eval.py")
    dataset_dst = osp.join(sgdet3d_root, "mmdet3d", "datasets", "kitti_dataset.py")

    if osp.isfile(eval_src) and osp.isfile(dataset_src):
        shutil.copy(eval_src, eval_dst)
        shutil.copy(dataset_src, dataset_dst)
    else:
        print("Eval tool files not found under SGDet3D, skip patching.")


def load_pretrained_model(model, checkpoint_path, mapping_list):
    if checkpoint_path is None:
        print("checkpoint_path is invalid, skip pretrained weight loading.")
        return model

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint

    if list(state_dict.keys())[0].startswith("module."):
        state_dict = {k[7:]: v for k, v in state_dict.items()}

    model_state = model.state_dict()
    new_ckpt = OrderedDict()
    skipped_missing = []
    skipped_shape = []
    for key, value in state_dict.items():
        for old_key, new_key in mapping_list.items():
            if key.startswith(old_key):
                mapped_key = key.replace(old_key, new_key)
                if mapped_key not in model_state:
                    skipped_missing.append(mapped_key)
                    break
                if tuple(model_state[mapped_key].shape) != tuple(value.shape):
                    skipped_shape.append(
                        (
                            mapped_key,
                            tuple(value.shape),
                            tuple(model_state[mapped_key].shape),
                        )
                    )
                    break
                new_ckpt[mapped_key] = value
                break

    incompatible = model.load_state_dict(new_ckpt, strict=False)
    unexpected = getattr(incompatible, "unexpected_keys", [])
    print(f"Loaded {len(new_ckpt)} mapped keys from {checkpoint_path}")
    if skipped_missing:
        print(
            f"Skipped missing mapped keys sample ({min(len(skipped_missing), 20)} / {len(skipped_missing)}): "
            f"{skipped_missing[:20]}"
        )
    if skipped_shape:
        sample = [f"{name}: ckpt{src} != model{dst}" for name, src, dst in skipped_shape[:20]]
        print(
            f"Skipped shape-mismatched keys sample ({min(len(skipped_shape), 20)} / {len(skipped_shape)}): "
            f"{sample}"
        )
    if unexpected:
        print(f"Unexpected keys sample ({min(len(unexpected), 20)} / {len(unexpected)}): {unexpected[:20]}")
    return model


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    project = project_name_from_config(args.config)

    maybe_prepare_eval_tools(project)

    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    if cfg.get("custom_imports", None):
        from mmcv.utils import import_modules_from_strings

        import_modules_from_strings(**cfg["custom_imports"])

    if cfg.get("cudnn_benchmark", False):
        torch.backends.cudnn.benchmark = True

    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    elif cfg.get("work_dir", None) is None:
        cfg.work_dir = osp.join("./work_dirs", osp.splitext(osp.basename(args.config))[0])

    if args.resume_from is not None:
        cfg.resume_from = args.resume_from

    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids
    else:
        cfg.gpu_ids = range(1) if args.gpus is None else range(args.gpus)

    if args.autoscale_lr:
        cfg.optimizer["lr"] = cfg.optimizer["lr"] * len(cfg.gpu_ids) / 8

    if args.launcher == "none":
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)
        _, world_size = get_dist_info()
        cfg.gpu_ids = range(world_size)

    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    cfg.dump(osp.join(cfg.work_dir, osp.basename(args.config)))

    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    log_file = osp.join(cfg.work_dir, f"{timestamp}.log")
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level, name="mmdet")

    meta = dict()
    env_info_dict = collect_env()
    env_info = "\n".join([(f"{k}: {v}") for k, v in env_info_dict.items()])
    dash_line = "-" * 60 + "\n"
    logger.info("Environment info:\n" + dash_line + env_info + "\n" + dash_line)
    meta["env_info"] = env_info
    meta["config"] = cfg.pretty_text

    logger.info(f"Distributed training: {distributed}")
    logger.info(f"Config:\n{cfg.pretty_text}")

    if args.seed is not None:
        logger.info(
            f"Set random seed to {args.seed}, deterministic: {args.deterministic}"
        )
        set_random_seed(args.seed, deterministic=args.deterministic)
    cfg.seed = args.seed
    meta["seed"] = args.seed
    meta["exp_name"] = osp.basename(args.config)

    model = build_model(
        cfg.model,
        train_cfg=cfg.get("train_cfg"),
        test_cfg=cfg.get("test_cfg"),
    )
    model.init_weights()

    if (
        cfg.model.type == "LiteSGDet3D"
        and cfg.get("load_from", None)
        and cfg.get("load_radar_from", None)
        and not cfg.get("load_img_from", None)
    ):
        print("LiteSGDet3D pretrain detected: treating load_from as official image backbone pretrain.")
        print(f"Image checkpoint: {cfg.load_from}")
        model = load_pretrained_model(model, cfg.load_from, {"img_backbone": "img_backbone"})
        cfg.load_from = None

    if "load_img_from" in cfg:
        if cfg.load_img_from:
            print("load_img_from exists, loading pretrained image backbone weights.")
            print(f"Image checkpoint: {cfg.load_img_from}")
            model = load_pretrained_model(model, cfg.load_img_from, {"img_backbone": "img_backbone"})

    if "load_radar_from" in cfg:
        if cfg.load_radar_from:
            print("load_radar_from exists, loading pretrained pure radar weights.")
            print(f"Radar checkpoint: {cfg.load_radar_from}")
            mapping_list = {
                "voxel_encoder": "pts_voxel_encoder",
                "middle_encoder": "pts_middle_encoder",
                "backbone": "pts_backbone",
                "neck": "pts_neck",
            }
            model = load_pretrained_model(model, cfg.load_radar_from, mapping_list)

    logger.info(f"Model:\n{model}")

    datasets = [build_dataset(cfg.data.train)]
    if len(cfg.workflow) == 2:
        val_dataset = copy.deepcopy(cfg.data.val)
        if "dataset" in cfg.data.train:
            val_dataset.pipeline = cfg.data.train.dataset.pipeline
        else:
            val_dataset.pipeline = cfg.data.train.pipeline
        val_dataset.test_mode = False
        datasets.append(build_dataset(val_dataset))

    if cfg.checkpoint_config is not None:
        cfg.checkpoint_config.meta = dict(
            mmdet_version=mmdet_version,
            mmseg_version=mmseg_version,
            mmdet3d_version=mmdet3d_version,
            config=cfg.pretty_text,
            CLASSES=datasets[0].CLASSES,
            PALETTE=datasets[0].PALETTE if hasattr(datasets[0], "PALETTE") else None,
        )

    model.CLASSES = datasets[0].CLASSES

    train_model(
        model,
        datasets,
        cfg,
        distributed=distributed,
        validate=not args.no_validate,
        timestamp=timestamp,
        meta=meta,
    )


if __name__ == "__main__":
    main()
