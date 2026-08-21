#!/usr/bin/env bash
set -eo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
if [[ -n "${DEPLOY_ENV_FILE:-}" ]]; then
  source "$DEPLOY_ENV_FILE"
fi

export PD_KV_MODE="${PD_KV_MODE:-store}"
export AFD_HOST="${AFD_HOST:-${DECODE_IP:-127.0.0.1}}"
export AFD_PORT="${AFD_PORT:-29761}"
export HCCL_IF_IP="${DECODE_HCCL_IF_IP:-${DECODE_IP:-192.169.91.106}}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-${NETWORK_INTERFACE:-eth0}}"
export HCCL_SOCKET_IFNAME="${HCCL_SOCKET_IFNAME:-${NETWORK_INTERFACE:-eth0}}"
export ATTENTION_HCCL_IF_BASE_PORT="${ATTENTION_HCCL_IF_BASE_PORT:-44000}"
export FFN_HCCL_IF_BASE_PORT="${FFN_HCCL_IF_BASE_PORT:-46000}"
export EXECUTION_MODE="${EXECUTION_MODE:-eager}"
export U_BATCHES="${DECODE_U_BATCHES:-${U_BATCHES:-2}}"
export DBO_DECODE_TOKEN_THRESHOLD="${DBO_DECODE_TOKEN_THRESHOLD:-2}"
export DBO_PREFILL_TOKEN_THRESHOLD="${DBO_PREFILL_TOKEN_THRESHOLD:-12}"
export ENABLE_MTP="${ENABLE_MTP:-0}"
export AFD_ASYNC_SCHEDULING="${AFD_ASYNC_SCHEDULING:-off}"

if [[ "$EXECUTION_MODE" != "eager" ]]; then
  echo "PD x AFD Decode supports only EXECUTION_MODE=eager" >&2
  exit 2
fi
if [[ "$U_BATCHES" != "1" && "$U_BATCHES" != "2" ]]; then
  echo "PD x AFD Decode supports U_BATCHES=1 or 2, got $U_BATCHES" >&2
  exit 2
fi
if [[ "$ENABLE_MTP" != "0" ]]; then
  echo "PD x AFD Decode does not support MTP" >&2
  exit 2
fi
if [[ ! "$DBO_DECODE_TOKEN_THRESHOLD" =~ ^[0-9]+$ ]]; then
  echo "DBO_DECODE_TOKEN_THRESHOLD must be a non-negative integer" >&2
  exit 2
fi
if [[ ! "$DBO_PREFILL_TOKEN_THRESHOLD" =~ ^[0-9]+$ ]]; then
  echo "DBO_PREFILL_TOKEN_THRESHOLD must be a non-negative integer" >&2
  exit 2
fi

if [[ "$PD_KV_MODE" == "store" ]]; then
  source "${ROOT_DIR}/recipe/npu/P2pHcclAFDConnector/deepseek_v4/mooncake_common.sh"
  wait_for_mooncake_master
fi

LOG_DIR="${DECODE_LOG_DIR:-/tmp/afd-pd-decode}"
STARTUP_TIMEOUT="${DECODE_STARTUP_TIMEOUT:-1800}"
DECODE_HEALTH_URL="http://127.0.0.1:${DECODE_API_PORT:-8200}/health"
mkdir -p "$LOG_DIR"

{
  echo "afd_plugin_commit=$(git -C "$ROOT_DIR" rev-parse HEAD)"
  echo "execution_mode=$EXECUTION_MODE"
  echo "u_batches=$U_BATCHES"
  echo "dbo_decode_token_threshold=$DBO_DECODE_TOKEN_THRESHOLD"
  echo "dbo_prefill_token_threshold=$DBO_PREFILL_TOKEN_THRESHOLD"
  echo "attention_ranks=${ATTENTION_RANKS:-8}"
  echo "ffn_ranks=${FFN_RANKS:-8}"
  echo "attention_devices=${ATTENTION_DEVICES:-0,1,2,3,4,5,6,7}"
  echo "ffn_devices=${FFN_DEVICES:-8,9,10,11,12,13,14,15}"
  echo "pd_kv_mode=$PD_KV_MODE"
  echo "afd_connector=P2pHcclAFDConnector"
} >"${LOG_DIR}/runtime.env"

echo "Starting PD x AFD Decode: eager/U${U_BATCHES}, A${ATTENTION_RANKS:-8}F${FFN_RANKS:-8}"
echo "DBO thresholds: decode=${DBO_DECODE_TOKEN_THRESHOLD}, prefill=${DBO_PREFILL_TOKEN_THRESHOLD}"

ffn_pid=""
attention_pid=""
shutdown_requested=0

stop_decode_processes() {
  for pid in "$attention_pid" "$ffn_pid"; do
    if [[ -n "$pid" ]]; then
      kill -TERM "$pid" 2>/dev/null || true
    fi
  done
  for pid in "$attention_pid" "$ffn_pid"; do
    if [[ -n "$pid" ]]; then
      wait "$pid" 2>/dev/null || true
    fi
  done
}

request_shutdown() {
  shutdown_requested=1
  stop_decode_processes
}

trap request_shutdown TERM INT

bash "${ROOT_DIR}/recipe/npu/P2pHcclAFDConnector/deepseek_v4/afd_ffn.sh" \
  >"${LOG_DIR}/ffn.log" 2>&1 &
ffn_pid=$!

sleep 3
if ! kill -0 "$ffn_pid" 2>/dev/null; then
  echo "Decode FFN exited during startup; inspect ${LOG_DIR}/ffn.log" >&2
  wait "$ffn_pid"
fi

bash "${ROOT_DIR}/recipe/npu/P2pHcclAFDConnector/deepseek_v4/pd_decode_attention.sh" \
  >"${LOG_DIR}/attention.log" 2>&1 &
attention_pid=$!

ready=0
for ((attempt = 0; attempt < STARTUP_TIMEOUT; attempt++)); do
  if ! kill -0 "$ffn_pid" 2>/dev/null; then
    echo "Decode FFN exited; inspect ${LOG_DIR}/ffn.log" >&2
    stop_decode_processes
    exit 1
  fi
  if ! kill -0 "$attention_pid" 2>/dev/null; then
    echo "Decode Attention exited; inspect ${LOG_DIR}/attention.log" >&2
    stop_decode_processes
    exit 1
  fi
  if curl --fail --silent --max-time 2 "$DECODE_HEALTH_URL" >/dev/null; then
    ready=1
    break
  fi
  sleep 1
done

if [[ "$ready" != "1" ]]; then
  echo "Decode service was not healthy within ${STARTUP_TIMEOUT}s" >&2
  stop_decode_processes
  exit 1
fi

echo "AFD Decode is ready at ${DECODE_HEALTH_URL}"
echo "Decode logs: ${LOG_DIR}/attention.log and ${LOG_DIR}/ffn.log"

set +e
wait -n "$attention_pid" "$ffn_pid"
service_status=$?
set -e
stop_decode_processes

if ((shutdown_requested)); then
  exit 0
fi
exit "$service_status"
