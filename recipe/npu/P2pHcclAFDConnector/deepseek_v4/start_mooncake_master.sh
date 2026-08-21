#!/usr/bin/env bash
set -eo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
if [[ -n "${DEPLOY_ENV_FILE:-}" ]]; then
  source "$DEPLOY_ENV_FILE"
fi

source "${ROOT_DIR}/tools/dsv4/activate_v023_vllm_cann_runtime.sh"
source "${ROOT_DIR}/recipe/npu/P2pHcclAFDConnector/deepseek_v4/mooncake_common.sh"

prepare_mooncake_client_config
configure_mooncake_library_path
if ! command -v mooncake_master >/dev/null 2>&1; then
  echo "mooncake_master is not installed in ${DSV4_VLLM_VENV}" >&2
  exit 2
fi

MOONCAKE_EVICTION_HIGH_WATERMARK="${MOONCAKE_EVICTION_HIGH_WATERMARK:-0.9}"
MOONCAKE_EVICTION_RATIO="${MOONCAKE_EVICTION_RATIO:-0.1}"
MOONCAKE_KV_LEASE_TTL_MS="${MOONCAKE_KV_LEASE_TTL_MS:-120000}"
MOONCAKE_CLIENT_TTL_SECONDS="${MOONCAKE_CLIENT_TTL_SECONDS:-120}"

echo "Starting Mooncake Master at ${MOONCAKE_MASTER_IP}:${MOONCAKE_MASTER_PORT}"
exec mooncake_master \
  --port "$MOONCAKE_MASTER_PORT" \
  --eviction_high_watermark_ratio "$MOONCAKE_EVICTION_HIGH_WATERMARK" \
  --eviction_ratio "$MOONCAKE_EVICTION_RATIO" \
  --default_kv_lease_ttl "$MOONCAKE_KV_LEASE_TTL_MS" \
  --client_ttl "$MOONCAKE_CLIENT_TTL_SECONDS"
