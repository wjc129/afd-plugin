#!/usr/bin/env bash
set -eo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
if [[ -n "${DEPLOY_ENV_FILE:-}" ]]; then
  source "$DEPLOY_ENV_FILE"
fi
export DSV4_VLLM_VENV="${DSV4_RUNTIME_VENV:-/mnt/workspace/code/.venvs/afd-v023-vllm-cann}"
source "${ROOT_DIR}/tools/dsv4/activate_runtime.sh"

PREFILL_URL="${PREFILL_URL:-http://${PREFILL_IP:?Set PREFILL_IP}:${PREFILL_API_PORT:-8100}}"
DECODE_URL="${DECODE_URL:-http://${DECODE_IP:?Set DECODE_IP}:${DECODE_API_PORT:-8200}}"

exec python "${ROOT_DIR}/tools/dsv4/pd_afd_proxy.py" \
  --host "${PROXY_HOST:-0.0.0.0}" \
  --port "${PROXY_API_PORT:-8000}" \
  --prefill-url "$PREFILL_URL" \
  --decode-url "$DECODE_URL" \
  --request-timeout "${PROXY_REQUEST_TIMEOUT:-1800}"
