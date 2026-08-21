#!/usr/bin/env bash
set -eo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
if [[ -n "${DEPLOY_ENV_FILE:-}" ]]; then
  source "$DEPLOY_ENV_FILE"
fi

export DECODE_U_BATCHES=2
export EXECUTION_MODE=eager
export ENABLE_MTP=0
export DBO_DECODE_TOKEN_THRESHOLD="${DBO_DECODE_TOKEN_THRESHOLD:-2}"
export DBO_PREFILL_TOKEN_THRESHOLD="${DBO_PREFILL_TOKEN_THRESHOLD:-12}"
export AFD_ASYNC_SCHEDULING="${AFD_ASYNC_SCHEDULING:-off}"

exec bash "${ROOT_DIR}/recipe/npu/P2pHcclAFDConnector/deepseek_v4/start_decode.sh"
