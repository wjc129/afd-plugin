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

if [[ "$PD_KV_MODE" == "store" ]]; then
  source "${ROOT_DIR}/recipe/npu/P2pHcclAFDConnector/deepseek_v4/mooncake_common.sh"
  wait_for_mooncake_master
fi

LOG_DIR="${DECODE_LOG_DIR:-/tmp/afd-pd-decode}"
STARTUP_TIMEOUT="${DECODE_STARTUP_TIMEOUT:-1800}"
DECODE_HEALTH_URL="http://127.0.0.1:${DECODE_API_PORT:-8200}/health"
mkdir -p "$LOG_DIR"

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
