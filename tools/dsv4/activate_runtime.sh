#!/usr/bin/env bash
# Source this file before building or running the pinned DSV4 AFD stack.

DSV4_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${DSV4_SCRIPT_DIR}/runtime_discovery.sh"

if ! DSV4_RUNTIME_PYTHON="$(dsv4_resolve_runtime_python "$DSV4_SCRIPT_DIR")"; then
  return 2 2>/dev/null || exit 2
fi
if ! DSV4_CANN_ROOT="$(dsv4_resolve_cann_root "$DSV4_SCRIPT_DIR")"; then
  return 2 2>/dev/null || exit 2
fi
if ! DSV4_VLLM_ROOT="$(dsv4_resolve_module_root "$DSV4_RUNTIME_PYTHON" vllm "${DSV4_VLLM_ROOT:-}")"; then
  return 2 2>/dev/null || exit 2
fi
if ! DSV4_VLLM_ASCEND_ROOT="$(dsv4_resolve_module_root "$DSV4_RUNTIME_PYTHON" vllm_ascend "${DSV4_VLLM_ASCEND_ROOT:-}")"; then
  return 2 2>/dev/null || exit 2
fi
DSV4_VLLM_VENV="$("$DSV4_RUNTIME_PYTHON" -c 'import sys; print(sys.prefix)')"
DSV4_PYTHON_LIB_DIR="$("$DSV4_RUNTIME_PYTHON" -c 'import sysconfig; print(sysconfig.get_config_var("LIBDIR") or "")')"

unset ASCEND_AICPU_PATH ASCEND_HOME_PATH ASCEND_OPP_PATH ASCEND_TOOLKIT_HOME
unset ASCEND_CUSTOM_OPP_PATH ATB_HOME_PATH TOOLCHAIN_HOME VIRTUAL_ENV
export CMAKE_PREFIX_PATH=
export LD_LIBRARY_PATH="$DSV4_PYTHON_LIB_DIR"
export PYTHONPATH=
export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

source "${DSV4_CANN_ROOT}/set_env.sh"
if [[ -f "${DSV4_CANN_ROOT}/nnal/atb/set_env.sh" ]]; then
  source "${DSV4_CANN_ROOT}/nnal/atb/set_env.sh"
fi

export VIRTUAL_ENV="${DSV4_VLLM_VENV}"
export PATH="${DSV4_VLLM_VENV}/bin:${PATH}"
export VLLM_PLUGINS=ascend,ascend_model,ascend_model_loader,ascend_kv_connector,afd
export DSV4_CANN_ROOT DSV4_RUNTIME_PYTHON DSV4_RUNTIME_VENV="$DSV4_VLLM_VENV"
export DSV4_VLLM_VENV DSV4_VLLM_ROOT DSV4_VLLM_ASCEND_ROOT

echo "Resolved DSV4 Python: $DSV4_RUNTIME_PYTHON" >&2
echo "Resolved CANN root: $DSV4_CANN_ROOT" >&2
echo "Resolved vLLM root: $DSV4_VLLM_ROOT" >&2
echo "Resolved vLLM-Ascend root: $DSV4_VLLM_ASCEND_ROOT" >&2

unset DSV4_SCRIPT_DIR DSV4_PYTHON_LIB_DIR
