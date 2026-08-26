#!/usr/bin/env bash
set -eo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"

# This entry point intentionally cannot be overridden back to eager/U1.
export EXECUTION_MODE=full-decode-only
export DECODE_U_BATCHES=2
export U_BATCHES=2
export ENABLE_MTP=0
export AFD_ASYNC_SCHEDULING=off
export HCCL_CONNECT_TIMEOUT="${HCCL_CONNECT_TIMEOUT:-1200}"
export MAX_CUDAGRAPH_CAPTURE_SIZE="${MAX_CUDAGRAPH_CAPTURE_SIZE:-8}"
export CUDAGRAPH_CAPTURE_SIZES="${CUDAGRAPH_CAPTURE_SIZES:-1 2 4 8}"

exec bash "${ROOT_DIR}/recipe/npu/P2pHcclAFDConnector/deepseek_v4/pd_decode_attention.sh"
