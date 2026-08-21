#!/usr/bin/env bash
set -eo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
if [[ -n "${DEPLOY_ENV_FILE:-}" ]]; then
  source "$DEPLOY_ENV_FILE"
fi

LOG_DIR="${PREFILL_STACK_LOG_DIR:-/tmp/afd-pd-prefill-u2}"
STARTUP_TIMEOUT="${PREFILL_STACK_STARTUP_TIMEOUT:-3600}"
PREFILL_HEALTH_URL="http://127.0.0.1:${PREFILL_API_PORT:-8100}/health"
DECODE_HEALTH_URL="http://${DECODE_IP:?Set DECODE_IP}:${DECODE_API_PORT:-8200}/health"
PROXY_HEALTH_URL="http://127.0.0.1:${PROXY_API_PORT:-8000}/health"
mkdir -p "$LOG_DIR"

master_pid=""
prefill_pid=""
proxy_pid=""
shutdown_requested=0

stop_prefill_stack() {
  for pid in "$proxy_pid" "$prefill_pid" "$master_pid"; do
    if [[ -n "$pid" ]]; then
      kill -TERM "$pid" 2>/dev/null || true
    fi
  done
  for pid in "$proxy_pid" "$prefill_pid" "$master_pid"; do
    if [[ -n "$pid" ]]; then
      wait "$pid" 2>/dev/null || true
    fi
  done
}

request_shutdown() {
  shutdown_requested=1
  stop_prefill_stack
}

wait_for_http() {
  local name="$1"
  local url="$2"
  local pid="${3:-}"
  local attempt
  local owned_pid
  for ((attempt = 0; attempt < STARTUP_TIMEOUT; attempt++)); do
    for owned_pid in "$master_pid" "$prefill_pid" "$proxy_pid"; do
      if [[ -n "$owned_pid" ]] && ! kill -0 "$owned_pid" 2>/dev/null; then
        echo "A prefill-node process exited; inspect ${LOG_DIR}" >&2
        return 1
      fi
    done
    if [[ -n "$pid" ]] && ! kill -0 "$pid" 2>/dev/null; then
      echo "$name exited during startup; inspect ${LOG_DIR}" >&2
      return 1
    fi
    if curl --fail --silent --max-time 2 "$url" >/dev/null; then
      return 0
    fi
    sleep 1
  done
  echo "$name was not healthy within ${STARTUP_TIMEOUT}s: $url" >&2
  return 1
}

trap request_shutdown TERM INT
trap stop_prefill_stack EXIT

bash "${ROOT_DIR}/recipe/npu/P2pHcclAFDConnector/deepseek_v4/start_mooncake_master.sh" \
  >"${LOG_DIR}/mooncake-master.log" 2>&1 &
master_pid=$!

source "${ROOT_DIR}/recipe/npu/P2pHcclAFDConnector/deepseek_v4/mooncake_common.sh"
wait_for_mooncake_master
if ! kill -0 "$master_pid" 2>/dev/null; then
  echo "Mooncake Master exited; inspect ${LOG_DIR}/mooncake-master.log" >&2
  wait "$master_pid"
fi

bash "${ROOT_DIR}/recipe/npu/P2pHcclAFDConnector/deepseek_v4/pd_prefill.sh" \
  >"${LOG_DIR}/prefill.log" 2>&1 &
prefill_pid=$!
wait_for_http "Prefill" "$PREFILL_HEALTH_URL" "$prefill_pid"

echo "Mooncake and Prefill are ready; waiting for Decode U2 at ${DECODE_HEALTH_URL}"
wait_for_http "Decode U2" "$DECODE_HEALTH_URL"

bash "${ROOT_DIR}/recipe/npu/P2pHcclAFDConnector/deepseek_v4/start_proxy.sh" \
  >"${LOG_DIR}/proxy.log" 2>&1 &
proxy_pid=$!
wait_for_http "PD x AFD proxy" "$PROXY_HEALTH_URL" "$proxy_pid"

echo "PD x AFD U2 service is ready at http://127.0.0.1:${PROXY_API_PORT:-8000}"
echo "Prefill-node logs: ${LOG_DIR}"

set +e
wait -n "$proxy_pid" "$prefill_pid" "$master_pid"
service_status=$?
set -e
stop_prefill_stack

if ((shutdown_requested)); then
  exit 0
fi
exit "$service_status"
