#!/bin/bash
set -euo pipefail

# The original example is for Running Mixtral 8x7B model on 32 H100/A100 GPUs

# export CUDA_DEVICE_MAX_CONNECTIONS=1 
unset CUDA_DEVICE_MAX_CONNECTIONS # The example set CUDA_DEVICE_MAX_CONNECTIONS=1

GPUS_PER_NODE=8
MASTER_ADDR=${MASTER_ADDR:?MASTER_ADDR must be set to rank-0 host}
MASTER_PORT=${MASTER_PORT:-6000}
NNODES=2
export NODE_RANK=${RANK:?RANK must be set to 0 or 1}

CHECKPOINT_PATH=${1:-"./fsdp_profile/Mistral_8_7B"}

DISTRIBUTED_ARGS=(
    --nproc_per_node ${GPUS_PER_NODE}
    --nnodes ${NNODES}
    --node_rank ${NODE_RANK}
    --master_addr ${MASTER_ADDR}
    --master_port ${MASTER_PORT}
)

MODEL_ARGS=(
    --disable-bias-linear
    --seq-length 4096 # The example uses 4096
    --max-position-embeddings 32768 # The example uses 32768
    --num-layers 8 # The example uses 32
    --hidden-size 4096 # The example uses 4096
    --ffn-hidden-size 14336 # The example uses 14336 (***)
    --num-attention-heads 32 # The examples uses 32
    --init-method-std 0.01 # The example uses 0.01
    --attention-dropout 0.0 # The example uses 0.0
    --hidden-dropout 0.0 # The example uses 0.0
    --normalization RMSNorm # The example uses RMSNorm
    --position-embedding-type rope # The example uses rope
    --swiglu # The example uses swiglu
    --untie-embeddings-and-output-weights # The example uses untie-embeddings-and-output-weights
    --group-query-attention # The example uses group-query-attention
    --num-query-groups 8 # The example uses 8 (***)
    --no-masked-softmax-fusion # The example uses no-masked-softmax-fusion
    --no-position-embedding # The example uses no-position-embedding
)

MOE_ARGS=(
    --num-experts 8 # The example uses 8 (***)
    --expert-model-parallel-size 8 # The example uses 8 (****)
    --expert-tensor-parallel-size 1 # The example does not use this. Maybe the default is 1
    --moe-router-load-balancing-type aux_loss # The example uses aux_loss
    --moe-router-topk 2 # The example uses 2
    --moe-aux-loss-coeff 1e-2 # The example uses 1e-2 (***)
    --moe-grouped-gemm # The example uses grouped-gemm
    --moe-permute-fusion # The example uses permute-fusion
    --moe-token-dispatcher-type alltoall # The example uses alltoall (***)
    # --moe-token-dispatcher-type flex
    # --moe-flex-dispatcher-backend hybridep
)

if [[ "${FORCE_UNIFORM_ROUTING:-0}" == "1" ]]; then
    MOE_ARGS+=(--moe-router-force-uniform-routing)
fi

# # The example uses DATA_ARGS as shown below
# DATA_ARGS=(
#     --tokenizer-type Llama2Tokenizer
#     --tokenizer-model ${TOKENIZER_MODEL}
#     --data-path $DATA_PATH
#     --split 99990,8,2
# )


DATA_ARGS=(
    --mock-data
    --tokenizer-type NullTokenizer
    --vocab-size 32000
    --split 100,0,0 # train, validation, test. This does not represent the absolute num.
)


TRAINING_ARGS=(
    --micro-batch-size 2 # The example uses 1
    --global-batch-size 128 # The example uses 128 (***)
    --lr 1e-4 # The example uses 1e-4
    --train-iters 10 # The example uses 500000 (***)
    --lr-decay-iters 10 # The example uses 320000 (***)
    --lr-decay-style cosine # The example uses cosine
    --min-lr 1e-5 # The example uses 1e-5
    --weight-decay 0.0 # The example uses 0.1 (***)
    --lr-warmup-iters 0 # The example uses 500 (***)
    --clip-grad 1.0 # The example uses 1.0
    --bf16 # The example uses bf16
    --overlap-grad-reduce # The example uses this (***). This is for overlapping the gradients sync (all-reduce or reduce-scatter) and backward computation during the backward pass
    --overlap-param-gather # The example uses this (***). Useful if the --use-distributed-optimizer is enabled, which means that reduce-scatter is used for gradient sync during the backward pass and all-gather is needed in the next forward pass. This is for overlapping the gradients all-gather with the forward computation during the forward pass.
    --num-workers 0 # The example does not set this

)

MODEL_PARALLEL_ARGS=(
    --tensor-model-parallel-size 1 # The example uses 1
    --pipeline-model-parallel-size 1 # The example uses 4, but megatron fsdp needs pp = 1
    # --num-layers-per-virtual-pipeline-stage 8 # The example enables vertiual pp
    --sequence-parallel
    --use-distributed-optimizer # The example enables this
)


LOGGING_ARGS=(
    --log-interval 1 # The example uses 1
    --save-interval 1000 # The example uses 10000
    --eval-interval 1000 # The example uses 1000
    --eval-iters 1 # The example uses 10
    --save "${CHECKPOINT_PATH}" # The example enables this
    # --load "${CHECKPOINT_PATH}" # The example enables this
    --tensorboard-dir "${CHECKPOINT_PATH}/tensorboard" # The example enables this
    # --ckpt-format torch_dist # The example enables this, but Megatron fsdp requires "--ckpt-format fsdp_dtensor"
    --auto-detect-ckpt-format # The example enables this
)

# Not shown in the example
FSDP_ARGS=(
    --use-megatron-fsdp
    --data-parallel-sharding-strategy optim_grads_params
    --no-gradient-accumulation-fusion
    --ckpt-format fsdp_dtensor    
    --fsdp-double-buffer
)

# Not shown in the example
PROFILE_ARGS=(
    --profile                    # Enables nsys profiling
    --profile-step-start 5       # Start capturing at this step (skip warmup)
    --profile-step-end 8        # Stop capturing after this step
    --profile-ranks 0 8           # Which ranks to profile (default: all — expensive!)
    --nvtx-ranges
)



export LD_PRELOAD=/jizhicfs/johnnyslin/anaconda3/envs/zh_megatron_312/lib/python3.12/site-packages/transformer_engine/wheel_lib/libtransformer_engine.so${LD_PRELOAD:+:$LD_PRELOAD}

torchrun "${DISTRIBUTED_ARGS[@]}" \
  --no-python \
  nsys profile \
    -s none \
    --cpuctxsw=none \
    --trace=cuda,nvtx,cudnn,cublas \
    --capture-range=cudaProfilerApi \
    --capture-range-end=stop \
    --output="${CHECKPOINT_PATH}/profile_node_${NODE_RANK}_rank_%q{RANK}_local_%q{LOCAL_RANK}" \
    --force-overwrite=true \
  python pretrain_gpt.py \
    "${MODEL_ARGS[@]}" \
    "${MOE_ARGS[@]}" \
    "${DATA_ARGS[@]}" \
    "${TRAINING_ARGS[@]}" \
    "${MODEL_PARALLEL_ARGS[@]}" \
    "${LOGGING_ARGS[@]}" \
    "${PROFILE_ARGS[@]}" \
    "${FSDP_ARGS[@]}"
