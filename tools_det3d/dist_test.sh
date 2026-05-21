#!/usr/bin/env bash

CONFIG=$1
CHECKPOINT=$2
GPUS=$3
PORT=${PORT:-29500}
PYTHON_BIN=${PYTHON_BIN:-/home/yanzexin/anaconda3/envs/SGDet3D/bin/python}

if [ -z "$CONFIG" ] || [ -z "$CHECKPOINT" ] || [ -z "$GPUS" ]; then
    echo "Usage: $0 CONFIG CHECKPOINT GPUS [additional test.py args]"
    exit 1
fi

PYTHONPATH="$(dirname "$0")/..":${SGDET3D_ROOT:-/home/yanzexin/SGDet3D}:$PYTHONPATH \
"$PYTHON_BIN" -m torch.distributed.launch \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    $(dirname "$0")/test.py \
    --config "$CONFIG" \
    --checkpoint "$CHECKPOINT" \
    --launcher pytorch ${@:4}
