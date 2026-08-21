#!/usr/bin/env bash
set -eo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
if [[ -n "${DEPLOY_ENV_FILE:-}" ]]; then
  source "$DEPLOY_ENV_FILE"
fi
export DSV4_VLLM_VENV="${DSV4_RUNTIME_VENV:-/mnt/workspace/code/.venvs/afd-v023-vllm-cann}"
source "${ROOT_DIR}/tools/dsv4/activate_runtime.sh"
DSV4_VLLM_ASCEND_ROOT="${DSV4_VLLM_ASCEND_ROOT:-/mnt/workspace/code/vllm-ascend-rfc-vllm-cann}"
source "${DSV4_VLLM_ASCEND_ROOT}/vllm_ascend/_cann_ops_custom/vendors/custom_transformer/bin/set_env.bash"
set -u

MODEL_PATH="${MODEL_PATH:-/mnt/workspace/models/DeepSeek-V4-Flash-w8a8-mtp}"
API_HOST="${API_HOST:-0.0.0.0}"
API_PORT="${PREFILL_API_PORT:-${API_PORT:-8100}}"
PREFILL_DP_SIZE="${PD_PREFILL_DP_SIZE:-4}"
PREFILL_TP_SIZE="${PD_PREFILL_TP_SIZE:-4}"
DECODE_DP_SIZE="${PD_DECODE_DP_SIZE:-8}"
DECODE_TP_SIZE="${PD_DECODE_TP_SIZE:-1}"
PD_KV_MODE="${PD_KV_MODE:-store}"
PD_KV_PORT="${PD_PREFILL_KV_PORT:-${PD_KV_PORT:-36000}}"
PD_ENGINE_ID="${PD_ENGINE_ID:-0}"
STORE_LOOKUP_RPC_PORT="${PREFILL_STORE_LOOKUP_RPC_PORT:-0}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-16}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"

export ASCEND_RT_VISIBLE_DEVICES="${PREFILL_DEVICES:-${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}}"
export HCCL_IF_IP="${PREFILL_HCCL_IF_IP:-${HCCL_IF_IP:-${PREFILL_IP:-192.169.91.105}}}"
export HCCL_IF_BASE_PORT="${PREFILL_HCCL_IF_BASE_PORT:-42000}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-${NETWORK_INTERFACE:-eth0}}"
export TP_SOCKET_IFNAME="${TP_SOCKET_IFNAME:-${NETWORK_INTERFACE:-eth0}}"
export HCCL_SOCKET_IFNAME="${HCCL_SOCKET_IFNAME:-${NETWORK_INTERFACE:-eth0}}"
export OMP_PROC_BIND=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-10}"
export PYTORCH_NPU_ALLOC_CONF="${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}"
export HCCL_BUFFSIZE="${HCCL_BUFFSIZE:-2560}"
export HCCL_OP_EXPANSION_MODE=AIV
export TASK_QUEUE_ENABLE=1
export SOC_VERSION=ascend910_9362
export VLLM_ENGINE_READY_TIMEOUT_S="${VLLM_ENGINE_READY_TIMEOUT_S:-18000}"
export VLLM_PLUGINS=ascend,ascend_model,ascend_model_loader,ascend_kv_connector
export PYTHONHASHSEED=0
source "${ROOT_DIR}/recipe/npu/P2pHcclAFDConnector/deepseek_v4/mooncake_common.sh"
configure_mooncake_library_path

case "$PD_KV_MODE" in
  direct)
    PD_KV_TRANSFER_CONFIG="$(printf '{"kv_connector":"MooncakeHybridConnector","kv_role":"kv_producer","kv_port":"%s","engine_id":"%s","kv_connector_extra_config":{"prefill":{"dp_size":%s,"tp_size":%s},"decode":{"dp_size":%s,"tp_size":%s}}}' "$PD_KV_PORT" "$PD_ENGINE_ID" "$PREFILL_DP_SIZE" "$PREFILL_TP_SIZE" "$DECODE_DP_SIZE" "$DECODE_TP_SIZE")"
    ;;
  store)
    prepare_mooncake_client_config
    wait_for_mooncake_master
    PD_KV_TRANSFER_CONFIG="$(printf '{"kv_connector":"MultiConnector","kv_role":"kv_producer","kv_load_failure_policy":"recompute","kv_connector_extra_config":{"connectors":[{"kv_connector":"MooncakeHybridConnector","kv_role":"kv_producer","kv_port":"%s","engine_id":"%s","kv_connector_extra_config":{"prefill":{"dp_size":%s,"tp_size":%s},"decode":{"dp_size":%s,"tp_size":%s}}},{"kv_connector":"AscendStoreConnector","kv_role":"kv_producer","kv_connector_extra_config":{"lookup_rpc_port":"%s","backend":"mooncake"}}]}}' "$PD_KV_PORT" "$PD_ENGINE_ID" "$PREFILL_DP_SIZE" "$PREFILL_TP_SIZE" "$DECODE_DP_SIZE" "$DECODE_TP_SIZE" "$STORE_LOOKUP_RPC_PORT")"
    ;;
  *)
    echo "PD_KV_MODE must be direct or store" >&2
    exit 2
    ;;
esac

exec vllm serve "$MODEL_PATH" \
  --host "$API_HOST" \
  --port "$API_PORT" \
  --api-server-count 1 \
  --served-model-name dsv4-afd \
  --max-model-len "$MAX_MODEL_LEN" \
  --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --data-parallel-size "$PREFILL_DP_SIZE" \
  --tensor-parallel-size "$PREFILL_TP_SIZE" \
  --enable-expert-parallel \
  --seed 1024 \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --tokenizer-mode deepseek_v4 \
  --enable-request-id-headers \
  --no-disable-hybrid-kv-cache-manager \
  --no-enable-prefix-caching \
  --safetensors-load-strategy prefetch \
  --quantization ascend \
  --block-size 128 \
  --enforce-eager \
  --kv-transfer-config "$PD_KV_TRANSFER_CONFIG"
