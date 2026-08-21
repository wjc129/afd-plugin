#!/usr/bin/env bash
# Source this file for the vLLM 0.23 + vllm-ascend rfc/vllm_cann stack.

if [[ -n "${DSV4_RUNTIME_VENV:-}" ]]; then
  export DSV4_VLLM_VENV="$DSV4_RUNTIME_VENV"
fi
export DSV4_EXPECTED_VLLM_VERSION=0.23.0

if ! source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/activate_runtime.sh"; then
  unset DSV4_EXPECTED_VLLM_VERSION
  return 2 2>/dev/null || exit 2
fi

DSV4_DETECTED_VLLM_VERSION="$(python -c 'import importlib.metadata; print(importlib.metadata.version("vllm"))')"
case "$DSV4_DETECTED_VLLM_VERSION" in
  0.23.0|0.23.0+*)
    ;;
  *)
    echo "Expected vLLM 0.23.0, discovered ${DSV4_DETECTED_VLLM_VERSION} in ${DSV4_RUNTIME_VENV}" >&2
    unset DSV4_DETECTED_VLLM_VERSION DSV4_EXPECTED_VLLM_VERSION
    return 2 2>/dev/null || exit 2
    ;;
esac
unset DSV4_DETECTED_VLLM_VERSION DSV4_EXPECTED_VLLM_VERSION
