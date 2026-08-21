#!/usr/bin/env bash

# Resolve the runtime from the server itself. Explicit paths are preferred, but
# stale paths do not prevent discovery from the active Python environment and
# installed CANN locations.

_dsv4_canonical_dir() {
  local directory="$1"
  (cd "$directory" 2>/dev/null && pwd -P)
}

_dsv4_python_has_stack() {
  local python_bin="$1"
  "$python_bin" -c '
import importlib.util
import importlib.metadata
import os
import sys

required = ("torch_npu", "vllm", "vllm_ascend")
if not all(importlib.util.find_spec(name) is not None for name in required):
    raise SystemExit(1)
expected = os.environ.get("DSV4_EXPECTED_VLLM_VERSION")
if expected:
    version = importlib.metadata.version("vllm")
    if version != expected and not version.startswith(f"{expected}+"):
        raise SystemExit(1)
' >/dev/null 2>&1
}

dsv4_resolve_runtime_python() {
  local script_dir="$1"
  local requested_root
  local python_bin
  local command_name
  local search_root
  local candidate
  local canonical
  local -a compatible_pythons=()
  local -A seen_pythons=()
  local -a requested_roots=(
    "${DSV4_VLLM_VENV:-}"
    "${DSV4_RUNTIME_VENV:-}"
    "${VIRTUAL_ENV:-}"
    "${CONDA_PREFIX:-}"
  )

  for requested_root in "${requested_roots[@]}"; do
    [[ -n "$requested_root" ]] || continue
    python_bin="${requested_root}/bin/python"
    if [[ -x "$python_bin" ]] && _dsv4_python_has_stack "$python_bin"; then
      printf '%s\n' "$python_bin"
      return 0
    fi
    echo "Ignoring incompatible DSV4 runtime: $requested_root" >&2
  done

  for command_name in python python3; do
    python_bin="$(command -v "$command_name" 2>/dev/null || true)"
    if [[ -n "$python_bin" ]] && _dsv4_python_has_stack "$python_bin"; then
      printf '%s\n' "$python_bin"
      return 0
    fi
  done

  local repository_parent
  repository_parent="$(_dsv4_canonical_dir "${script_dir}/../../..")"
  local search_roots="${repository_parent}:${DSV4_RUNTIME_SEARCH_ROOTS:-}"
  IFS=: read -r -a requested_roots <<<"$search_roots"
  for search_root in "${requested_roots[@]}"; do
    [[ -d "$search_root" ]] || continue
    while IFS= read -r candidate; do
      if [[ -x "$candidate" ]] && _dsv4_python_has_stack "$candidate"; then
        canonical="$(readlink -f "$candidate")"
        if [[ -z "${seen_pythons[$canonical]:-}" ]]; then
          seen_pythons[$canonical]=1
          compatible_pythons+=("$canonical")
        fi
      fi
    done < <(
      find "$search_root" -maxdepth 4 -path '*/bin/python' \
        \( -type f -o -type l \) 2>/dev/null
    )
  done

  if (( ${#compatible_pythons[@]} == 1 )); then
    printf '%s\n' "${compatible_pythons[0]}"
    return 0
  fi
  if (( ${#compatible_pythons[@]} > 1 )); then
    echo "Multiple compatible DSV4 Python environments were discovered; refusing to guess:" >&2
    printf '  %s\n' "${compatible_pythons[@]}" >&2
    echo "Activate the intended environment or set DSV4_RUNTIME_VENV." >&2
    return 2
  fi

  echo "Could not discover a compatible Python environment containing torch_npu, vllm, and vllm_ascend." >&2
  echo "Activate it first, set DSV4_RUNTIME_VENV, or set DSV4_RUNTIME_SEARCH_ROOTS." >&2
  return 2
}

dsv4_resolve_cann_root() {
  local script_dir="$1"
  local requested_root="${DSV4_CANN_ROOT:-}"
  local candidate
  local canonical
  local search_root
  local set_env_path
  local -a candidates=()
  local -a search_roots=()
  local -A seen=()

  if [[ -n "$requested_root" ]]; then
    if [[ -f "${requested_root}/set_env.sh" ]]; then
      _dsv4_canonical_dir "$requested_root"
      return 0
    fi
    echo "Ignoring invalid DSV4_CANN_ROOT: $requested_root" >&2
  fi

  for requested_root in "${ASCEND_TOOLKIT_HOME:-}" "${ASCEND_HOME_PATH:-}"; do
    [[ -n "$requested_root" ]] || continue
    for candidate in "$requested_root" "$(dirname "$requested_root")"; do
      if [[ -f "${candidate}/set_env.sh" ]]; then
        _dsv4_canonical_dir "$candidate"
        return 0
      fi
    done
  done

  search_roots+=("$(_dsv4_canonical_dir "${script_dir}/../../..")")
  [[ -d /usr/local/Ascend ]] && search_roots+=(/usr/local/Ascend)
  [[ -d /opt/Ascend ]] && search_roots+=(/opt/Ascend)
  if [[ -n "${DSV4_CANN_SEARCH_ROOTS:-}" ]]; then
    local -a extra_roots=()
    IFS=: read -r -a extra_roots <<<"$DSV4_CANN_SEARCH_ROOTS"
    search_roots+=("${extra_roots[@]}")
  fi

  for search_root in "${search_roots[@]}"; do
    [[ -d "$search_root" ]] || continue
    while IFS= read -r set_env_path; do
      candidate="$(dirname "$set_env_path")"
      canonical="$(_dsv4_canonical_dir "$candidate")"
      [[ -n "$canonical" ]] || continue
      if [[ -z "${seen[$canonical]:-}" ]]; then
        seen[$canonical]=1
        candidates+=("$canonical")
      fi
    done < <(
      find "$search_root" -maxdepth 5 -name set_env.sh \
        \( -type f -o -type l \) \
        -not -path '*/nnal/*' -not -path '*/opp/*' \
        -not -path '*/vendors/*' 2>/dev/null
    )
  done

  if (( ${#candidates[@]} == 1 )); then
    printf '%s\n' "${candidates[0]}"
    return 0
  fi
  if (( ${#candidates[@]} > 1 )); then
    echo "Multiple CANN installations were discovered; refusing to guess:" >&2
    printf '  %s\n' "${candidates[@]}" >&2
    echo "Set DSV4_CANN_ROOT to the intended directory containing set_env.sh." >&2
    return 2
  fi

  echo "Could not discover CANN set_env.sh." >&2
  echo "Set DSV4_CANN_ROOT or DSV4_CANN_SEARCH_ROOTS to a server-local location." >&2
  return 2
}

dsv4_resolve_module_root() {
  local python_bin="$1"
  local module_name="$2"
  local requested_root="$3"

  if [[ -n "$requested_root" && -d "${requested_root}/${module_name}" ]]; then
    _dsv4_canonical_dir "$requested_root"
    return 0
  fi
  if [[ -n "$requested_root" ]]; then
    echo "Ignoring invalid ${module_name} source root: $requested_root" >&2
  fi

  "$python_bin" - "$module_name" <<'PY'
import importlib.util
import pathlib
import sys

module_name = sys.argv[1]
spec = importlib.util.find_spec(module_name)
if spec is None:
    raise SystemExit(f"Could not locate installed module: {module_name}")
if spec.submodule_search_locations:
    package_dir = pathlib.Path(next(iter(spec.submodule_search_locations)))
elif spec.origin:
    package_dir = pathlib.Path(spec.origin).parent
else:
    raise SystemExit(f"Module has no filesystem location: {module_name}")
print(package_dir.resolve().parent)
PY
}

dsv4_source_ascend_custom_ops() {
  local custom_env_path="${DSV4_VLLM_ASCEND_ROOT}/vllm_ascend/_cann_ops_custom/vendors/custom_transformer/bin/set_env.bash"
  if [[ ! -f "$custom_env_path" ]]; then
    echo "Installed vllm_ascend has no custom Transformer environment: $custom_env_path" >&2
    return 2
  fi
  source "$custom_env_path"
}
