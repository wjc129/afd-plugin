#!/usr/bin/env bash
set -eo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
if [[ -n "${DEPLOY_ENV_FILE:-}" ]]; then
  source "$DEPLOY_ENV_FILE"
fi
source "${ROOT_DIR}/tools/dsv4/activate_v023_vllm_cann_runtime.sh"

LOG_DIR="${DECODE_LOG_DIR:-/tmp/afd-pd-decode}"
OUTPUT="${1:-${LOG_DIR}/u2_evidence.json}"

npu-smi info >"${LOG_DIR}/decode_npu_after_validation.txt" 2>&1 || true
python "${ROOT_DIR}/tools/dsv4/collect_pd_afd_u2_evidence.py" \
  --log-dir "$LOG_DIR" \
  --health-endpoint "http://127.0.0.1:${DECODE_API_PORT:-8200}/health" \
  --output "$OUTPUT"
