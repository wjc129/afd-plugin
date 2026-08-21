#!/usr/bin/env bash
set -eo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
if [[ -n "${DEPLOY_ENV_FILE:-}" ]]; then
  source "$DEPLOY_ENV_FILE"
fi

source "${ROOT_DIR}/recipe/npu/P2pHcclAFDConnector/deepseek_v4/mooncake_common.sh"
wait_for_mooncake_master

curl --fail --silent --show-error \
  "http://${PREFILL_IP:?Set PREFILL_IP}:${PREFILL_API_PORT:-8100}/health"
echo
curl --fail --silent --show-error \
  "http://${DECODE_IP:?Set DECODE_IP}:${DECODE_API_PORT:-8200}/health"
echo
curl --fail --silent --show-error \
  "http://127.0.0.1:${PROXY_API_PORT:-8000}/health"
echo
