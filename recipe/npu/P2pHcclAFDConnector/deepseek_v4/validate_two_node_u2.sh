#!/usr/bin/env bash
set -eo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
if [[ -n "${DEPLOY_ENV_FILE:-}" ]]; then
  source "$DEPLOY_ENV_FILE"
fi
source "${ROOT_DIR}/tools/dsv4/activate_v023_vllm_cann_runtime.sh"

GOLDEN_RESULTS="${1:-${GOLDEN_RESULTS:-/mnt/workspace/validation/dsv4_v023_vllm_cann_native_baseline/golden_results.json}}"
OUTPUT_DIR="${2:-${PD_AFD_U2_VALIDATION_DIR:-/mnt/workspace/validation/dsv4_pd_afd_u2_$(date +%Y%m%d_%H%M%S)}}"
PROXY_PORT="${PROXY_API_PORT:-8000}"

if [[ ! -f "$GOLDEN_RESULTS" ]]; then
  echo "Golden file does not exist: $GOLDEN_RESULTS" >&2
  exit 2
fi

mkdir -p "$OUTPUT_DIR"
bash "${ROOT_DIR}/recipe/npu/P2pHcclAFDConnector/deepseek_v4/check_two_node_service.sh"

{
  echo "afd_plugin_commit=$(git -C "$ROOT_DIR" rev-parse HEAD)"
  echo "execution_mode=eager"
  echo "u_batches=2"
  echo "batch_sizes=1,8,32"
  echo "golden=$GOLDEN_RESULTS"
  echo "endpoint=http://127.0.0.1:${PROXY_PORT}/v1/completions"
} >"${OUTPUT_DIR}/runtime.env"

python "${ROOT_DIR}/recipe/npu/CAMP2pAFDConnector/deepseek_v4/validate_golden.py" \
  --endpoint "http://127.0.0.1:${PROXY_PORT}/v1/completions" \
  --model dsv4-afd \
  --golden "$GOLDEN_RESULTS" \
  --output "${OUTPUT_DIR}/golden.json" \
  --rounds "${PD_AFD_U2_VALIDATION_ROUNDS:-3}" \
  --batch-sizes 1 8 32 \
  --require-batch-token-exact

npu-smi info >"${OUTPUT_DIR}/prefill_npu_after_validation.txt" 2>&1 || true
echo "Node P validation passed: ${OUTPUT_DIR}/golden.json"
echo "Now run collect_decode_u2_evidence.sh on Node D."
