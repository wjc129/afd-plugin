#!/usr/bin/env bash

MOONCAKE_MASTER_IP="${MOONCAKE_MASTER_IP:-${PREFILL_IP:-}}"
MOONCAKE_MASTER_PORT="${MOONCAKE_MASTER_PORT:-50088}"
MOONCAKE_CONFIG_PATH="${MOONCAKE_CONFIG_PATH:-/tmp/afd-mooncake/mooncake.json}"
MOONCAKE_GLOBAL_SEGMENT_SIZE="${MOONCAKE_GLOBAL_SEGMENT_SIZE:-1GB}"
MOONCAKE_WAIT_TIMEOUT="${MOONCAKE_WAIT_TIMEOUT:-120}"

require_mooncake_master_address() {
  if [[ -z "$MOONCAKE_MASTER_IP" ]]; then
    echo "Set MOONCAKE_MASTER_IP or PREFILL_IP before starting Mooncake-managed PD" >&2
    return 2
  fi
}

prepare_mooncake_client_config() {
  require_mooncake_master_address
  mkdir -p "$(dirname "$MOONCAKE_CONFIG_PATH")"
  cat >"$MOONCAKE_CONFIG_PATH" <<EOF
{
  "metadata_server": "P2PHANDSHAKE",
  "protocol": "ascend",
  "device_name": "",
  "master_server_address": "${MOONCAKE_MASTER_IP}:${MOONCAKE_MASTER_PORT}",
  "global_segment_size": "${MOONCAKE_GLOBAL_SEGMENT_SIZE}",
  "preferred_segment": false,
  "prefer_alloc_in_same_node": true
}
EOF

  export MOONCAKE_CONFIG_PATH
  export MOONCAKE_MASTER="${MOONCAKE_MASTER_IP}:${MOONCAKE_MASTER_PORT}"
  export ASCEND_ENABLE_USE_FABRIC_MEM="${ASCEND_ENABLE_USE_FABRIC_MEM:-1}"
  export HCCL_RDMA_TIMEOUT="${HCCL_RDMA_TIMEOUT:-17}"
  export ASCEND_CONNECT_TIMEOUT="${ASCEND_CONNECT_TIMEOUT:-30000}"
  export ASCEND_TRANSFER_TIMEOUT="${ASCEND_TRANSFER_TIMEOUT:-30000}"
}

configure_mooncake_library_path() {
  local mooncake_library_dir
  if ! mooncake_library_dir="$(python -c 'import importlib.util, os; spec = importlib.util.find_spec("mooncake"); assert spec is not None and spec.origin is not None; print(os.path.dirname(spec.origin))')"; then
    echo "mooncake-transfer-engine-npu is not installed in the active runtime" >&2
    return 2
  fi
  export LD_LIBRARY_PATH="${mooncake_library_dir}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
}

wait_for_mooncake_master() {
  require_mooncake_master_address
  python - "$MOONCAKE_MASTER_IP" "$MOONCAKE_MASTER_PORT" "$MOONCAKE_WAIT_TIMEOUT" <<'PY'
import socket
import sys
import time

host = sys.argv[1]
port = int(sys.argv[2])
timeout = int(sys.argv[3])
deadline = time.monotonic() + timeout
last_error = "not attempted"

while time.monotonic() < deadline:
    try:
        with socket.create_connection((host, port), timeout=2):
            print(f"Mooncake Master is reachable at {host}:{port}")
            raise SystemExit(0)
    except OSError as error:
        last_error = str(error)
        time.sleep(1)

raise SystemExit(
    f"Mooncake Master {host}:{port} was not reachable within {timeout}s: "
    f"{last_error}"
)
PY
}
