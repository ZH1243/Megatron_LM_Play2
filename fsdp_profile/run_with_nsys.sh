#!/bin/bash
set -euo pipefail

echo "[rank ${RANK:-?} local ${LOCAL_RANK:-?}] NCCL_ALGO=${NCCL_ALGO:-<unset>} NCCL_PROTO=${NCCL_PROTO:-<unset>} NCCL_TUNER_PLUGIN=${NCCL_TUNER_PLUGIN:-<unset>} NCCL_ENV_PLUGIN=${NCCL_ENV_PLUGIN:-<unset>} NCCL_DEBUG=${NCCL_DEBUG:-<unset>} NCCL_DEBUG_SUBSYS=${NCCL_DEBUG_SUBSYS:-<unset>} NCCL_DEBUG_FILE=${NCCL_DEBUG_FILE:-<unset>}"

if [[ "${LOCAL_RANK}" == "0" ]]; then
  exec nsys profile \
    -s none \
    --cpuctxsw=none \
    --trace=cuda,nvtx,cudnn,cublas \
    --capture-range=cudaProfilerApi \
    --capture-range-end=stop \
    --output="${CHECKPOINT_PATH}/profile_node_${NODE_RANK}_rank_${RANK}_local_${LOCAL_RANK}" \
    --gpu-metrics-devices=${LOCAL_RANK} \
    --gpu-metrics-set=gh100 \
    --gpu-metrics-frequency=10000 \
    --force-overwrite=true \
    python pretrain_gpt.py "$@"
else
  exec python pretrain_gpt.py "$@"
fi
