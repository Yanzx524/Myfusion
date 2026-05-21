import argparse
import os
import shutil
import warnings
from os import path as osp

import mmcv
import torch
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import get_dist_info, init_dist, load_checkpoint, wrap_fp16_model

from mmdet.apis import multi_gpu_test, set_random_seed
from mmdet.datasets import replace_ImageToTensor
from mmdet3d.apis import single_gpu_test
from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_model


def parse_args():
    parser = argparse.ArgumentParser(description="Test MyFusion")
    parser.add_argument("--config", required=True, help="test config file path")
    parser.add_argument("--checkpoint", required=True, help="checkpoint file")
    parser.add_argument("--show", action="store_true", help="show results")
    parser.add_argument("--show-dir", help="directory where results will be saved")
    parser.add_argument("--out", help="output result file in pickle format")
    parser.add_argument(
        "--fuse-conv-bn",
        action="store_true",
        help="Whether to fuse conv and bn, this will slightly increase the inference speed",
    )
    parser.add_argument(
        "--format-only",
        action="store_true",
        help="Format the output results without perform evaluation.",
    )
    parser.add_argument(
        "--eval",
        type=str,
        nargs="+",
        help='evaluation metrics, e.g. "bbox", "segm", "proposal", "mAP"',
    )
    parser.add_argument(
        "--gpu-collect",
        action="store_true",
        help="whether to use gpu to collect results.",
    )
    parser.add_argument(
        "--tmpdir",
        help="tmp directory used for collecting results from multiple workers",
    )
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="whether to set deterministic options for CUDNN backend.",
    )
    parser.add_argument(
        "--cfg-options",
        nargs="+",
        action=DictAction,
        help="override some settings in the used config.",
    )
    parser.add_argument(
        "--options",
        nargs="+",
        action=DictAction,
        help="custom options for evaluation (deprecated), change to --eval-options instead.",
    )
    parser.add_argument(
        "--eval-options",
        nargs="+",
        action=DictAction,
        help="custom options for evaluation, passed to dataset.evaluate().",
    )
    parser.add_argument(
        "--launcher",
        choices=["none", "pytorch", "slurm", "mpi"],
        default="none",
        help="job launcher",
    )
    parser.add_argument("--local_rank", type=int, default=0)
    args = parser.parse_args()

    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = str(args.local_rank)

    if args.options and args.eval_options:
        raise ValueError(
            "--options and --eval-options cannot be both specified, --options is deprecated in favor of --eval-options"
        )
    if args.options:
        warnings.warn("--options is deprecated in favor of --eval-options")
        args.eval_options = args.options
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


def main():
    args = parse_args()
    print("start testing")

    assert args.out or args.eval or args.format_only or args.show or args.show_dir, (
        'Please specify at least one operation with the argument "--out", "--eval", "--format-only", "--show" or "--show-dir"'
    )
    if args.eval and args.format_only:
        raise ValueError("--eval and --format-only cannot be both specified")
    if args.out is not None and not args.out.endswith((".pkl", ".pickle")):
        raise ValueError("The output file must be a pkl file.")

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

    cfg.model.pretrained = None
    samples_per_gpu = 1
    if isinstance(cfg.data.test, dict):
        samples_per_gpu = cfg.data.test.pop("samples_per_gpu", 1)
        if samples_per_gpu > 1:
            cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)
    elif isinstance(cfg.data.test, list):
        samples_per_gpu = max([ds_cfg.pop("samples_per_gpu", 1) for ds_cfg in cfg.data.test])
        if samples_per_gpu > 1:
            for ds_cfg in cfg.data.test:
                ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)

    if args.launcher == "none":
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    if args.seed is not None:
        set_random_seed(args.seed, deterministic=args.deterministic)

    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False,
    )

    cfg.model.train_cfg = None
    model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))
    print("MODEL TOTAL PARAMETERS = %d" % (sum(p.numel() for p in model.parameters())))

    fp16_cfg = cfg.get("fp16", None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location="cpu")

    if args.fuse_conv_bn:
        model = fuse_conv_bn(model)

    if "CLASSES" in checkpoint.get("meta", {}):
        model.CLASSES = checkpoint["meta"]["CLASSES"]
    else:
        model.CLASSES = dataset.CLASSES

    if "PALETTE" in checkpoint.get("meta", {}):
        model.PALETTE = checkpoint["meta"]["PALETTE"]
    elif hasattr(dataset, "PALETTE"):
        model.PALETTE = dataset.PALETTE

    if not distributed:
        model = MMDataParallel(model, device_ids=[0])
        outputs = single_gpu_test(model, data_loader, args.show, args.show_dir)
    else:
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
        )
        outputs = multi_gpu_test(model, data_loader, args.tmpdir, args.gpu_collect)

    rank, _ = get_dist_info()
    if rank == 0:
        if args.out:
            print(f"\nwriting results to {args.out}")
            mmcv.dump(outputs, args.out)
        kwargs = {} if args.eval_options is None else args.eval_options
        if args.format_only:
            dataset.format_results(outputs, **kwargs)
        if args.eval:
            eval_kwargs = cfg.get("evaluation", {}).copy()
            for key in ["interval", "tmpdir", "start", "gpu_collect", "save_best", "rule"]:
                eval_kwargs.pop(key, None)
            eval_kwargs.update(dict(metric=args.eval, **kwargs))
            print(dataset.evaluate(outputs, **eval_kwargs))


if __name__ == "__main__":
    main()
