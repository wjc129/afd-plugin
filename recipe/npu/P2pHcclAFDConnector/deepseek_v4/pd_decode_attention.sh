#!/usr/bin/env bash
set -eo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
if [[ -n "${DEPLOY_ENV_FILE:-}" ]]; then
  source "$DEPLOY_ENV_FILE"
fi
export AFD_CONNECTOR=P2pHcclAFDConnector
export EXECUTION_MODE="${EXECUTION_MODE:-eager}"
export U_BATCHES="${DECODE_U_BATCHES:-${U_BATCHES:-2}}"
export DBO_DECODE_TOKEN_THRESHOLD="${DBO_DECODE_TOKEN_THRESHOLD:-2}"
export DBO_PREFILL_TOKEN_THRESHOLD="${DBO_PREFILL_TOKEN_THRESHOLD:-12}"
export ENABLE_MTP="${ENABLE_MTP:-0}"
export AFD_ASYNC_SCHEDULING="${AFD_ASYNC_SCHEDULING:-off}"
export API_HOST="${DECODE_API_HOST:-0.0.0.0}"
export API_PORT="${DECODE_API_PORT:-${API_PORT:-8200}}"
export PD_KV_MODE="${PD_KV_MODE:-store}"
export PD_KV_PORT="${PD_DECODE_KV_PORT:-${PD_KV_PORT:-36200}}"
export PD_ENGINE_ID="${PD_ENGINE_ID:-1}"
export PD_PREFILL_DP_SIZE="${PD_PREFILL_DP_SIZE:-4}"
export PD_PREFILL_TP_SIZE="${PD_PREFILL_TP_SIZE:-4}"
export PD_DECODE_DP_SIZE="${PD_DECODE_DP_SIZE:-8}"
export PD_DECODE_TP_SIZE="${PD_DECODE_TP_SIZE:-1}"
export HCCL_IF_IP="${DECODE_HCCL_IF_IP:-${HCCL_IF_IP:-${DECODE_IP:?Set DECODE_IP or DECODE_HCCL_IF_IP in the server-local environment}}}"
export ATTENTION_HCCL_IF_BASE_PORT="${ATTENTION_HCCL_IF_BASE_PORT:-44000}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-${NETWORK_INTERFACE:?Set NETWORK_INTERFACE or GLOO_SOCKET_IFNAME in the server-local environment}}"
export HCCL_SOCKET_IFNAME="${HCCL_SOCKET_IFNAME:-${NETWORK_INTERFACE:?Set NETWORK_INTERFACE or HCCL_SOCKET_IFNAME in the server-local environment}}"
export PYTHONHASHSEED=0

case "${DECODE_STANDALONE_AF:-0}" in
  0)
    export ENABLE_PD=1
    case "$PD_KV_MODE" in
      direct)
        unset PD_KV_TRANSFER_CONFIG
        ;;
      store)
        source "${ROOT_DIR}/recipe/npu/P2pHcclAFDConnector/deepseek_v4/mooncake_common.sh"
        prepare_mooncake_client_config
        wait_for_mooncake_master
        STORE_LOOKUP_RPC_PORT="${DECODE_STORE_LOOKUP_RPC_PORT:-0}"
        export PD_KV_TRANSFER_CONFIG
        PD_KV_TRANSFER_CONFIG="$(printf '{"kv_connector":"MultiConnector","kv_role":"kv_consumer","engine_id":"%s","kv_load_failure_policy":"recompute","kv_connector_extra_config":{"connectors":[{"kv_connector":"MooncakeHybridConnector","kv_role":"kv_consumer","kv_port":"%s","kv_connector_extra_config":{"prefill":{"dp_size":%s,"tp_size":%s},"decode":{"dp_size":%s,"tp_size":%s}}},{"kv_connector":"AscendStoreConnector","kv_role":"kv_consumer","kv_connector_extra_config":{"lookup_rpc_port":"%s","backend":"mooncake"}}]}}' "$PD_ENGINE_ID" "$PD_KV_PORT" "$PD_PREFILL_DP_SIZE" "$PD_PREFILL_TP_SIZE" "$PD_DECODE_DP_SIZE" "$PD_DECODE_TP_SIZE" "$STORE_LOOKUP_RPC_PORT")"
        ;;
      *)
        echo "PD_KV_MODE must be direct or store" >&2
        exit 2
        ;;
    esac
    ;;
  1)
    export ENABLE_PD=0
    unset PD_KV_TRANSFER_CONFIG
    ;;
  *)
    echo "DECODE_STANDALONE_AF must be 0 or 1" >&2
    exit 2
    ;;
esac

exec bash "${ROOT_DIR}/recipe/npu/CAMP2pAFDConnector/deepseek_v4/afd_attention.sh"
