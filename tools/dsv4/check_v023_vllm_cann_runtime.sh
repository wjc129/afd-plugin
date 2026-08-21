#!/usr/bin/env bash
set -eo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "${ROOT_DIR}/tools/dsv4/activate_v023_vllm_cann_runtime.sh"
set -u

EXPECTED_VLLM_COMMIT=0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665
EXPECTED_ASCEND_COMMIT=3da28f9414583d2d0b672a8f06d1fae142404bda
VLLM_ROOT="$DSV4_VLLM_ROOT"
ASCEND_ROOT="$DSV4_VLLM_ASCEND_ROOT"
MODEL_PATH="${MODEL_PATH:?Set MODEL_PATH to the DeepSeek V4 model directory}"
export MODEL_PATH

[[ "$(git -C "${VLLM_ROOT}" rev-parse HEAD)" == "${EXPECTED_VLLM_COMMIT}" ]]
[[ "$(git -C "${ASCEND_ROOT}" rev-parse HEAD)" == "${EXPECTED_ASCEND_COMMIT}" ]]

python - <<'PY'
from importlib.metadata import version
import os

import torch
import torch_npu
import vllm
import vllm_ascend  # noqa: F401

from afd_plugin import register_afd
from afd_plugin.compat.npu import ensure_afd_ascend_ops_loaded
from vllm.engine.arg_utils import EngineArgs

assert "cann-9.1.0" not in repr(dict(os.environ))
assert torch.npu.is_available()
assert torch.npu.device_count() == 16
assert vllm.__version__.startswith("0.23.0")
assert version("torch-npu") == "2.10.0.post2"
assert version("vllm-ascend").endswith("g3da28f941")
assert version("transformers") == "5.5.4"
assert version("numpy") == "2.2.6"
ensure_afd_ascend_ops_loaded()

# Build the real target-branch config once so vllm-ascend's compatibility
# normalization cannot silently turn an AFD U2 launch back into U1.
register_afd()
engine_args = EngineArgs(
    model=os.environ["MODEL_PATH"],
    tokenizer_mode="deepseek_v4",
    quantization="ascend",
    enforce_eager=True,
    data_parallel_size=8,
    tensor_parallel_size=1,
    enable_expert_parallel=True,
    enable_dbo=True,
    dbo_decode_token_threshold=2,
    dbo_prefill_token_threshold=12,
    additional_config={
        "afd": {
            "role": "attention",
            "connector": "P2pHcclAFDConnector",
            "host": "127.0.0.1",
            "port": 29761,
            "num_attention_ranks": 8,
            "num_ffn_ranks": 8,
        }
    },
)
vllm_config = engine_args.create_engine_config()
assert vllm_config.parallel_config.enable_dbo
assert vllm_config.parallel_config.use_ubatching
print("AFD_DBO_CONFIG_PRESERVED")
print("DSV4_AFD_V023_VLLM_CANN_RUNTIME_OK")
print("torch", torch.__version__)
print("torch_npu", torch_npu.__version__)
print("vllm", vllm.__version__)
print("vllm_ascend", version("vllm-ascend"))
print("transformers", version("transformers"))
print("numpy", version("numpy"))
PY
