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
DSV4_SITE_PACKAGES="$("$DSV4_RUNTIME_PYTHON" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
export PYTHONPATH="${DSV4_SITE_PACKAGES}${PYTHONPATH:+:${PYTHONPATH}}"

DSV4_TORCH_CXX11_ABI="$("$DSV4_RUNTIME_PYTHON" -c 'import torch; print(int(torch._C._GLIBCXX_USE_CXX11_ABI))')"
DSV4_ATB_ENV_PATH=""
for DSV4_ATB_ENV_CANDIDATE in \
  "${DSV4_CANN_ROOT}/nnal/atb/set_env.sh" \
  "$(dirname "$DSV4_CANN_ROOT")/nnal/atb/set_env.sh"; do
  if [[ -f "$DSV4_ATB_ENV_CANDIDATE" ]]; then
    DSV4_ATB_ENV_PATH="$DSV4_ATB_ENV_CANDIDATE"
    break
  fi
done
if [[ -n "$DSV4_ATB_ENV_PATH" ]]; then
  source "$DSV4_ATB_ENV_PATH" "--cxx_abi=${DSV4_TORCH_CXX11_ABI}"
fi

DSV4_CANN_SEARCH_MAX_DEPTH=8
DSV4_ATB_LIBRARY="$(
  find "$DSV4_CANN_ROOT" "$(dirname "$DSV4_CANN_ROOT")" \
    -maxdepth "$DSV4_CANN_SEARCH_MAX_DEPTH" \
    \( -type f -o -type l \) \
    -path "*/cxx_abi_${DSV4_TORCH_CXX11_ABI}/libatb.so" \
    -print -quit 2>/dev/null
)"
DSV4_RUNTIME_LIBRARY_DIRS=(
  "${DSV4_SITE_PACKAGES}/torch/lib"
  "${DSV4_SITE_PACKAGES}/torch_npu/lib"
)
if [[ -n "$DSV4_ATB_LIBRARY" ]]; then
  DSV4_RUNTIME_LIBRARY_DIRS+=("$(dirname "$DSV4_ATB_LIBRARY")")
fi
DSV4_RUNTIME_LIBRARY_PATH="$(IFS=:; printf '%s' "${DSV4_RUNTIME_LIBRARY_DIRS[*]}")"
export LD_LIBRARY_PATH="${DSV4_RUNTIME_LIBRARY_PATH}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

export VIRTUAL_ENV="${DSV4_VLLM_VENV}"
export PATH="${DSV4_VLLM_VENV}/bin:${PATH}"
export VLLM_PLUGINS=ascend,ascend_model,ascend_model_loader,ascend_kv_connector,afd
export DSV4_CANN_ROOT DSV4_RUNTIME_PYTHON DSV4_RUNTIME_VENV="$DSV4_VLLM_VENV"
export DSV4_VLLM_VENV DSV4_VLLM_ROOT DSV4_VLLM_ASCEND_ROOT

echo "Resolved DSV4 Python: $DSV4_RUNTIME_PYTHON" >&2
echo "Resolved CANN root: $DSV4_CANN_ROOT" >&2
echo "Resolved ATB environment: ${DSV4_ATB_ENV_PATH:-not found}" >&2
echo "Resolved ATB library: ${DSV4_ATB_LIBRARY:-not found}" >&2
echo "Resolved vLLM root: $DSV4_VLLM_ROOT" >&2
echo "Resolved vLLM-Ascend root: $DSV4_VLLM_ASCEND_ROOT" >&2

unset DSV4_SCRIPT_DIR DSV4_PYTHON_LIB_DIR DSV4_SITE_PACKAGES
unset DSV4_TORCH_CXX11_ABI DSV4_ATB_ENV_PATH DSV4_ATB_ENV_CANDIDATE
unset DSV4_CANN_SEARCH_MAX_DEPTH DSV4_ATB_LIBRARY
unset DSV4_RUNTIME_LIBRARY_DIRS DSV4_RUNTIME_LIBRARY_PATH
