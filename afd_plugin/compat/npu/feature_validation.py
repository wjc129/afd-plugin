# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Validation for AFD features supported by the Ascend runtime."""

from __future__ import annotations

from typing import TYPE_CHECKING

from afd_plugin.config import (
    AFD_ASYNC_CONNECTOR,
    AFDConfig,
    is_afd_async_dp,
    parse_afd_config,
)

MOONCAKE_HYBRID_CONNECTOR = "MooncakeHybridConnector"
MOONCAKE_MULTI_CONNECTOR = "MultiConnector"
ASCEND_STORE_CONNECTOR = "AscendStoreConnector"
MOONCAKE_STORE_BACKEND = "mooncake"
PD_AFD_CONNECTOR = "P2pHcclAFDConnector"
PD_DECODE_TENSOR_PARALLEL_SIZE = 1
PD_NUM_UBATCHES_U1 = 1
PD_NUM_UBATCHES_U2 = 2

if TYPE_CHECKING:
    from vllm.config import KVTransferConfig, VllmConfig

    from afd_plugin.connectors.base import ConnectorExtraInfo


def fail_if_unsupported_npu_afd_features(
    vllm_config: VllmConfig,
    *,
    afd_config: AFDConfig | None = None,
) -> None:
    """Fail fast for NPU AFD settings that are not currently supported."""

    afd_config = afd_config or parse_afd_config(vllm_config)
    from afd_plugin.connectors.factory import AFDConnectorFactory

    extra_info = AFDConnectorFactory.parse_connector_extra_info(
        afd_config.connector,
        vllm_config,
    )

    if _is_deepseek_v4(vllm_config):
        _fail_if_unsupported_deepseek_v4_features(vllm_config, afd_config)

    if afd_config.connector == AFD_ASYNC_CONNECTOR:
        _fail_if_unsupported_npu_afd_async_features(
            vllm_config,
            afd_config,
            extra_info,
        )
        return

    if afd_config.compute_gate_on_attention:
        raise RuntimeError(
            "AFD NPU runtime does not support compute_gate_on_attention=true yet",
        )
    if afd_config.connector == "CAMP2pAFDConnector":
        from afd_plugin.connectors.npu.camp2p import CAMP2PExtraInfo

        if not isinstance(extra_info, CAMP2PExtraInfo):
            raise TypeError(
                "CAMP2pAFDConnector requires CAMP2PExtraInfo, got "
                f"{type(extra_info).__name__}",
            )
        extra_info.validate_supported()

    uses_ubatching = bool(vllm_config.parallel_config.use_ubatching)
    if uses_ubatching and int(vllm_config.parallel_config.num_ubatches) != 2:
        raise RuntimeError(
            "AFD NPU runtime supports exactly two ubatches when DBO is enabled",
        )
    model_config = vllm_config.model_config
    # Match the pinned NPUModelRunner's sparse-attention backend selection.
    uses_sparse_mla = hasattr(
        model_config.hf_text_config,
        "index_topk",
    )
    cudagraph_mode = vllm_config.compilation_config.cudagraph_mode
    uses_mla_dbo_full_graph = (
        uses_ubatching
        and model_config.use_mla
        and not uses_sparse_mla
        and cudagraph_mode.has_full_cudagraphs()
    )
    if uses_mla_dbo_full_graph and vllm_config.speculative_config is not None:
        raise RuntimeError(
            "AFD NPU MLA DBO FULL graph does not support speculative decoding",
        )
    if uses_mla_dbo_full_graph and cudagraph_mode.name != "FULL_DECODE_ONLY":
        raise RuntimeError(
            "AFD NPU MLA DBO graph execution requires FULL_DECODE_ONLY",
        )


def _fail_if_unsupported_npu_afd_async_features(
    vllm_config: VllmConfig,
    afd_config: AFDConfig,
    extra_info: ConnectorExtraInfo,
) -> None:
    from afd_plugin.connectors.npu.async_cam import AFDAsyncExtraInfo

    if not isinstance(extra_info, AFDAsyncExtraInfo):
        raise TypeError(
            "CAMAsyncAFDConnector requires AFDAsyncExtraInfo, got "
            f"{type(extra_info).__name__}",
        )

    parallel_config = vllm_config.parallel_config
    if not is_afd_async_dp(vllm_config):
        raise RuntimeError(
            "CAMAsyncAFDConnector requires additional_config['afd'] "
            "with async=true and connector='CAMAsyncAFDConnector'",
        )
    if not bool(vllm_config.model_config.enforce_eager):
        raise RuntimeError(
            "CAMAsyncAFDConnector supports only eager Attention/FFN execution",
        )
    if bool(parallel_config.use_ubatching):
        raise RuntimeError(
            "CAMAsyncAFDConnector does not support vLLM native ubatching/DBO",
        )
    if extra_info.async_moe_ubatching:
        _fail_if_unsupported_npu_async_moe_ubatching_features(
            vllm_config,
            afd_config,
            num_ubatches=extra_info.async_moe_num_ubatches,
            split=extra_info.async_moe_split,
        )
    if extra_info.dynamic_quant not in (0, 1):
        raise RuntimeError(
            "CAMAsyncAFDConnector currently supports only dynamicQuant 0 or 1",
        )


def _is_deepseek_v4(vllm_config: VllmConfig) -> bool:
    hf_config = getattr(vllm_config.model_config, "hf_config", None)
    architectures = getattr(hf_config, "architectures", None) or []
    return any(
        architecture in {"DeepseekV4ForCausalLM", "AFDDeepseekV4ForCausalLM"}
        for architecture in architectures
    )


def _fail_if_unsupported_deepseek_v4_features(
    vllm_config: VllmConfig,
    afd_config: AFDConfig,
) -> None:
    """Keep DSV4 AFD inside its validated eager/U2 and graph/U1 boxes."""
    parallel_config = vllm_config.parallel_config
    supported_connectors = {
        "CAMP2pAFDConnector",
        "P2pHcclAFDConnector",
    }
    if afd_config.connector not in supported_connectors:
        raise RuntimeError(
            "DeepSeek-V4 AFD supports only CAMP2pAFDConnector or P2pHcclAFDConnector"
        )
    if (
        afd_config.connector == "CAMP2pAFDConnector"
        and afd_config.num_attention_ranks != afd_config.num_ffn_ranks
    ):
        raise RuntimeError(
            "DeepSeek-V4 CAMP2pAFDConnector requires equal Attention and FFN ranks"
        )
    if parallel_config.tensor_parallel_size != 1:
        raise RuntimeError("DeepSeek-V4 AFD supports only tensor_parallel_size=1")
    if parallel_config.pipeline_parallel_size != 1:
        raise RuntimeError("DeepSeek-V4 AFD supports only pipeline_parallel_size=1")
    if parallel_config.prefill_context_parallel_size != 1:
        raise RuntimeError(
            "DeepSeek-V4 AFD supports only prefill context parallel size 1"
        )
    if parallel_config.decode_context_parallel_size != 1:
        raise RuntimeError(
            "DeepSeek-V4 AFD supports only decode context parallel size 1"
        )
    if parallel_config.use_sequence_parallel_moe:
        raise RuntimeError("DeepSeek-V4 AFD does not support sequence-parallel MoE")
    if afd_config.compute_gate_on_attention:
        raise RuntimeError("DeepSeek-V4 AFD requires FFN-side gate computation")
    speculative_config = vllm_config.speculative_config
    if speculative_config is not None:
        if afd_config.connector != "P2pHcclAFDConnector":
            raise RuntimeError("DeepSeek-V4 AFD MTP supports only P2pHcclAFDConnector")
        if afd_config.num_attention_ranks != afd_config.num_ffn_ranks:
            raise RuntimeError("DeepSeek-V4 AFD MTP requires equal A/F ranks")
        if bool(getattr(parallel_config, "use_ubatching", False)):
            raise RuntimeError("DeepSeek-V4 AFD MTP supports only U1")
        if getattr(speculative_config, "method", None) != "mtp":
            raise RuntimeError("DeepSeek-V4 AFD supports only MTP speculative method")
        if int(getattr(speculative_config, "num_speculative_tokens", 0)) != 1:
            raise RuntimeError("DeepSeek-V4 AFD MTP supports num_speculative_tokens=1")
        draft_enforce_eager = bool(getattr(speculative_config, "enforce_eager", False))
        target_enforce_eager = bool(vllm_config.model_config.enforce_eager)
        if target_enforce_eager and not draft_enforce_eager:
            raise RuntimeError(
                "DeepSeek-V4 AFD MTP eager execution requires draft enforce_eager=true"
            )
        if not target_enforce_eager and not draft_enforce_eager:
            raise RuntimeError(
                "DeepSeek-V4 AFD MTP Graph/U1 currently requires draft "
                "enforce_eager=true; draft ACL Graph is not validated"
            )
        num_mtp_layers = int(
            getattr(vllm_config.model_config.hf_config, "num_nextn_predict_layers", 1)
        )
        if num_mtp_layers != 1:
            raise RuntimeError("DeepSeek-V4 AFD MTP supports exactly one MTP layer")
    if not vllm_config.model_config.enforce_eager:
        if (
            afd_config.connector == "P2pHcclAFDConnector"
            and afd_config.num_attention_ranks != afd_config.num_ffn_ranks
        ):
            raise RuntimeError(
                "DeepSeek-V4 P2pHcclAFDConnector graph execution requires equal "
                "Attention and FFN ranks"
            )
        cudagraph_mode = getattr(
            getattr(vllm_config, "compilation_config", None),
            "cudagraph_mode",
            None,
        )
        mode_name = getattr(cudagraph_mode, "name", None)
        if not isinstance(mode_name, str):
            mode_name = str(cudagraph_mode).rsplit(".", 1)[-1]
        if mode_name != "FULL_DECODE_ONLY":
            raise RuntimeError(
                "DeepSeek-V4 AFD graph execution supports only FULL_DECODE_ONLY"
            )
        if (
            parallel_config.use_ubatching
            and afd_config.connector != PD_AFD_CONNECTOR
        ):
            raise RuntimeError(
                "DeepSeek-V4 AFD Graph/U2 requires P2pHcclAFDConnector"
            )
    if vllm_config.kv_transfer_config is not None:
        _fail_if_unsupported_deepseek_v4_pd_features(vllm_config, afd_config)


def _fail_if_unsupported_deepseek_v4_pd_features(
    vllm_config: VllmConfig,
    afd_config: AFDConfig,
) -> None:
    """Validate the supported DeepSeek-V4 PD x AFD deployment.

    Prefill remains a native vLLM-Ascend service. The AFD service is the
    Decode side, where only Attention owns KV cache and consumes Mooncake data;
    Decode FFN remains a connector-driven daemon without KV state.
    """

    kv_transfer_config = vllm_config.kv_transfer_config
    assert kv_transfer_config is not None
    parallel_config = vllm_config.parallel_config

    if afd_config.role != "attention":
        raise RuntimeError(
            "DeepSeek-V4 PD x AFD attaches KV transfer only to Decode Attention; "
            "FFN must start without --kv-transfer-config"
        )
    if afd_config.connector != PD_AFD_CONNECTOR:
        raise RuntimeError("DeepSeek-V4 PD x AFD baseline requires P2pHcclAFDConnector")
    if kv_transfer_config.kv_connector not in (
        MOONCAKE_HYBRID_CONNECTOR,
        MOONCAKE_MULTI_CONNECTOR,
    ):
        raise RuntimeError(
            "DeepSeek-V4 PD x AFD supports MooncakeHybridConnector directly "
            "or MultiConnector with Mooncake-managed KV Pool"
        )
    if kv_transfer_config.kv_role != "kv_consumer":
        raise RuntimeError(
            "DeepSeek-V4 AFD is the Decode side of PD and requires "
            "kv_role='kv_consumer'"
        )
    uses_ubatching = bool(parallel_config.use_ubatching)
    num_ubatches = int(parallel_config.num_ubatches)
    if uses_ubatching and num_ubatches != PD_NUM_UBATCHES_U2:
        raise RuntimeError(
            "DeepSeek-V4 PD x AFD U2 requires exactly two ubatches"
        )
    if not uses_ubatching and num_ubatches != PD_NUM_UBATCHES_U1:
        raise RuntimeError(
            "DeepSeek-V4 PD x AFD supports only U1 or U2"
        )
    if vllm_config.speculative_config is not None:
        raise RuntimeError("DeepSeek-V4 PD x AFD baseline does not support MTP")

    extra_config = _deepseek_v4_pd_mooncake_extra_config(kv_transfer_config)
    prefill_topology = extra_config.get("prefill")
    decode_topology = extra_config.get("decode")
    if not isinstance(prefill_topology, dict) or not isinstance(decode_topology, dict):
        raise RuntimeError(
            "MooncakeHybridConnector requires prefill/decode topology objects"
        )

    prefill_dp_size = int(prefill_topology.get("dp_size", 0))
    prefill_tp_size = int(prefill_topology.get("tp_size", 0))
    if prefill_dp_size < 1 or prefill_tp_size < 1:
        raise RuntimeError(
            "MooncakeHybridConnector prefill dp_size/tp_size must be positive"
        )

    decode_dp_size = int(decode_topology.get("dp_size", 0))
    decode_tp_size = int(decode_topology.get("tp_size", 0))
    expected_decode_dp_size = int(parallel_config.data_parallel_size)
    if (
        decode_dp_size != expected_decode_dp_size
        or decode_tp_size != PD_DECODE_TENSOR_PARALLEL_SIZE
    ):
        raise RuntimeError(
            "MooncakeHybridConnector decode topology must match AFD Decode "
            f"Attention DP{expected_decode_dp_size}/"
            f"TP{PD_DECODE_TENSOR_PARALLEL_SIZE}; "
            f"got DP{decode_dp_size}/TP{decode_tp_size}"
        )


def _deepseek_v4_pd_mooncake_extra_config(
    kv_transfer_config: KVTransferConfig,
) -> dict[str, object]:
    """Return the direct-transfer topology after validating KV Pool wiring."""

    extra_config = kv_transfer_config.kv_connector_extra_config
    if not isinstance(extra_config, dict):
        raise RuntimeError("DeepSeek-V4 PD x AFD requires kv_connector_extra_config")

    if kv_transfer_config.kv_connector == MOONCAKE_HYBRID_CONNECTOR:
        return extra_config

    connectors = extra_config.get("connectors")
    if not isinstance(connectors, list) or len(connectors) != 2:
        raise RuntimeError(
            "DeepSeek-V4 Mooncake-managed PD requires exactly two MultiConnector "
            "children"
        )

    transfer_connector, store_connector = connectors
    if not isinstance(transfer_connector, dict) or not isinstance(
        store_connector, dict
    ):
        raise RuntimeError(
            "DeepSeek-V4 Mooncake-managed PD connector entries must be objects"
        )
    if (
        transfer_connector.get("kv_connector") != MOONCAKE_HYBRID_CONNECTOR
        or transfer_connector.get("kv_role") != kv_transfer_config.kv_role
    ):
        raise RuntimeError(
            "DeepSeek-V4 Mooncake-managed PD requires MooncakeHybridConnector "
            "as the first child with the same kv_role"
        )
    if (
        store_connector.get("kv_connector") != ASCEND_STORE_CONNECTOR
        or store_connector.get("kv_role") != kv_transfer_config.kv_role
    ):
        raise RuntimeError(
            "DeepSeek-V4 Mooncake-managed PD requires AscendStoreConnector "
            "as the second child with the same kv_role"
        )

    store_extra_config = store_connector.get("kv_connector_extra_config")
    if not isinstance(store_extra_config, dict) or (
        store_extra_config.get("backend") != MOONCAKE_STORE_BACKEND
    ):
        raise RuntimeError(
            "DeepSeek-V4 Mooncake-managed PD requires "
            "AscendStoreConnector backend='mooncake'"
        )

    transfer_extra_config = transfer_connector.get("kv_connector_extra_config")
    if not isinstance(transfer_extra_config, dict):
        raise RuntimeError("MooncakeHybridConnector requires kv_connector_extra_config")
    return transfer_extra_config


def _fail_if_unsupported_npu_async_moe_ubatching_features(
    vllm_config: VllmConfig,
    afd_config: AFDConfig,
    *,
    num_ubatches: int,
    split: str,
) -> None:
    from afd_plugin.connectors.npu.async_cam import ASYNC_MOE_REQUEST_SPLIT

    parallel_config = vllm_config.parallel_config
    if not afd_config.compute_gate_on_attention:
        raise RuntimeError(
            "async_moe_ubatching requires compute_gate_on_attention=true",
        )
    if num_ubatches != 2:
        raise RuntimeError(
            "async_moe_ubatching currently supports exactly two stages; "
            f"got async_moe_num_ubatches={num_ubatches}",
        )
    if split != ASYNC_MOE_REQUEST_SPLIT:
        raise RuntimeError(
            "async_moe_ubatching currently supports only request-boundary split; "
            f"got async_moe_split={split!r}",
        )
    if int(parallel_config.decode_context_parallel_size) > 1:
        raise RuntimeError(
            "async_moe_ubatching does not support decode context parallel metadata yet",
        )


__all__ = ["fail_if_unsupported_npu_afd_features"]
