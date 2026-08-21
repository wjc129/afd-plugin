#!/usr/bin/env bash
set -eo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "${ROOT_DIR}/tools/dsv4/activate_v023_vllm_cann_runtime.sh"
dsv4_source_ascend_custom_ops
set -u

MODEL_PATH="${MODEL_PATH:-/mnt/workspace/models/DeepSeek-V4-Flash-w8a8-mtp}"
API_HOST="${API_HOST:-127.0.0.1}"
API_PORT="${API_PORT:-8900}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-dsv4-v023-native}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-1024}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-8}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
DATA_PARALLEL_RPC_PORT="${DATA_PARALLEL_RPC_PORT:-29350}"
MASTER_PORT="${MASTER_PORT:-29351}"
ENABLE_MTP="${ENABLE_MTP:-0}"
MTP_NUM_SPECULATIVE_TOKENS="${MTP_NUM_SPECULATIVE_TOKENS:-1}"

MTP_ARGS=()
case "${ENABLE_MTP}" in
  0)
    ;;
  1)
    if [[ ! "${MTP_NUM_SPECULATIVE_TOKENS}" =~ ^[1-9][0-9]*$ ]]; then
      echo "MTP_NUM_SPECULATIVE_TOKENS must be a positive integer" >&2
      exit 2
    fi
    MTP_ARGS=(
      --speculative-config
      "{\"method\":\"mtp\",\"num_speculative_tokens\":${MTP_NUM_SPECULATIVE_TOKENS},\"enforce_eager\":true}"
    )
    ;;
  *)
    echo "ENABLE_MTP must be 0 or 1" >&2
    exit 2
    ;;
esac

export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export HCCL_IF_IP="${HCCL_IF_IP:-192.169.91.106}"
export HCCL_IF_BASE_PORT="${HCCL_IF_BASE_PORT:-53000}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-eth0}"
export HCCL_SOCKET_IFNAME="${HCCL_SOCKET_IFNAME:-eth0}"
export OMP_PROC_BIND=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-10}"
export PYTORCH_NPU_ALLOC_CONF="${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}"
export HCCL_BUFFSIZE="${HCCL_BUFFSIZE:-1024}"
export HCCL_OP_EXPANSION_MODE=AIV
export TASK_QUEUE_ENABLE=1
export SOC_VERSION=ascend910_9362
export VLLM_ENGINE_READY_TIMEOUT_S="${VLLM_ENGINE_READY_TIMEOUT_S:-18000}"
# This is the native control: the AFD plugin must not register or patch vLLM.
export VLLM_PLUGINS=ascend,ascend_model,ascend_model_loader,ascend_kv_connector

exec vllm serve "${MODEL_PATH}" \
  --host "${API_HOST}" \
  --port "${API_PORT}" \
  --api-server-count 1 \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}" \
  --max-num-seqs "${MAX_NUM_SEQS}" \
  --data-parallel-size 8 \
  --data-parallel-rpc-port "${DATA_PARALLEL_RPC_PORT}" \
  --master-port "${MASTER_PORT}" \
  --tensor-parallel-size 1 \
  --enable-expert-parallel \
  --enforce-eager \
  --seed 1024 \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --tokenizer-mode deepseek_v4 \
  --no-enable-prefix-caching \
  --safetensors-load-strategy lazy \
  --quantization ascend \
  --block-size 128 \
  "${MTP_ARGS[@]}"
