# MyFusion

`MyFusion` is a lightweight project scaffold for radar-camera BEV fusion.

The goal of this repository is not to port all of `SGDet3D`, but to keep:

- the same `TJ4D` / `VoD` dataset definitions
- the same custom data pipeline shape and metadata contract
- a simplified model decomposition closer to `BEVFusion`

The model side is intentionally reduced to:

- `camera encoder`
- `radar encoder`
- `BEV fuser`
- `decoder`
- `3D bbox head`

The following `SGDet3D`-specific branches are intentionally removed:

- backward projection refinement
- proposal supervision heads
- range-view foreground head
- SA-radar painting branch
- visualization and debugging branches

## Layout

```text
My_fusion/
├── projects/
│   └── MyFusion/
│       ├── configs/
│       └── mmdet3d_plugin/
│           ├── datasets/
│           └── models/
└── tools/
```

## Runtime Assumptions

This scaffold is designed to run on top of an existing `mmdet3d` environment,
such as the one already used by `SGDet3D` in `/home/yanzexin/SGDet3D`.

Before running, make sure the repository root is on `PYTHONPATH`, for example:

```bash
export PYTHONPATH=/home/yanzexin/My_fusion:$PYTHONPATH
```

## Provided Example

An example TJ4D config is provided at:

`projects/MyFusion/configs/tj4d_myfusion.py`

It keeps the SGDet3D-style dataset and pipeline, but the detector is simplified
into a BEVFusion-like structure.
