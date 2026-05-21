#!/usr/bin/env bash

CONFIG=$1
GPUS=$2
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29500}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
PYTHON_BIN=${PYTHON_BIN:-/home/yanzexin/anaconda3/envs/SGDet3D/bin/python}

if [ -z "$CONFIG" ] || [ -z "$GPUS" ]; then
    echo "Usage: $0 CONFIG GPUS [additional train.py args]"
    exit 1
fi

CUDA_VISIBLE_DEVICES="3,4,5,6" \
PYTHONPATH="$(dirname "$0")/..":${SGDET3D_ROOT:-/home/yanzexin/SGDet3D}:$PYTHONPATH \
"$PYTHON_BIN" -m torch.distributed.launch \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    $(dirname "$0")/train.py \
    --config "$CONFIG" \
    --launcher pytorch ${@:3}
