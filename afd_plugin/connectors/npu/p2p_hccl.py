# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""HCCL Send/Recv connector for Attention/FFN disaggregation on NPU.

``P2pHcclAFDConnector`` deliberately uses the public PyTorch distributed
point-to-point interface for its data path. Hidden states, FFN outputs, and
DeepSeek-V4 input IDs are transferred with ``torch.distributed.send`` and
``torch.distributed.recv`` over HCCL process groups. It does not load or call
the CAMP2P A2E/E2A custom operators.

The connector supports one or more consecutive Attention peers per FFN rank
(``A = k * F``). Each DBO stage owns an independent HCCL group so stages cannot
consume each other's messages. A Gloo control group carries stage token counts
before the FFN side posts receives and prepares its aggregate receive buffers.

Eager U2 keeps the same synchronous ``send/recv`` API but orders decoder
transfers with connector-owned NPU streams and events. DeepSeek-V4 drives its
two stages from one layer-major host loop, avoiding cross-thread DBO handoff
without changing the HCCL message protocol or introducing ``isend/irecv``. U1,
Graph U1, and MTP retain their existing paths.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import timedelta
from typing import TYPE_CHECKING, Any

import torch
import torch.distributed as dist
import torch.distributed.distributed_c10d as c10d
from torch.distributed.distributed_c10d import ProcessGroup
from vllm.forward_context import DPMetadata, get_forward_context

from afd_plugin.config import AFDConfig
from afd_plugin.connectors.base import (
    AFDConnectorBase,
    AFDControlPlane,
    ConnectorExtraInfo,
)
from afd_plugin.connectors.metadata import (
    AFDA2FTransferPayload,
    AFDControlPayload,
    AFDDPMetadata,
    AFDTransferContext,
    AFDTransferMetadata,
    AFDTransferState,
    recv_control_payload,
    send_control_payload,
)
from afd_plugin.distributed import build_rank_mapping, init_afd_process_group
from afd_plugin.v1.worker.dbo import maybe_apply_dbo_yield

if TYPE_CHECKING:
    from vllm.config import VllmConfig


# torch-npu 2.10.0.post2 source:
# torch_npu/dynamo/npugraph_ex/ops/_hcom_send_recv.py.
# Its dist.send/recv tracing wrappers pass torch.Size directly, which the target
# vLLM 0.23 compiler first flattens and then specializes despite a dynamic token
# dimension. Keep the same HCCL lowering without the optional, unused shape.
def _graph_hccl_send(
    tensor: torch.Tensor,
    *,
    dst: int,
    group: ProcessGroup,
) -> None:
    ranks = dist.get_process_group_ranks(group)
    pg_tag = c10d._get_group_tag(group)
    # ### PATCH START: avoid the torch-npu dynamic-shape tracing guard
    torch.ops.npu_define._send.default(
        tensor,
        dst,
        ranks,
        pg_tag,
        0,
        None,
        None,
    )
    # ### PATCH END: avoid the torch-npu dynamic-shape tracing guard


def _graph_hccl_recv(
    tensor: torch.Tensor,
    *,
    src: int,
    group: ProcessGroup,
) -> None:
    ranks = dist.get_process_group_ranks(group)
    pg_tag = c10d._get_group_tag(group)
    # ### PATCH START: avoid the torch-npu dynamic-shape tracing guard
    received = torch.ops.npu_define._recv.default(
        tensor,
        src,
        ranks,
        pg_tag,
        0,
        None,
        None,
    )
    tensor.copy_(received)
    # ### PATCH END: avoid the torch-npu dynamic-shape tracing guard


@dataclass(slots=True)
class HCCLP2PTransferState(AFDTransferState):
    """State retained between one FFN receive and its matching send."""

    stage_idx: int
    num_tokens: int
    peer_ranks: tuple[int, ...] = ()
    seq_lens: tuple[int, ...] = ()
    peer_slices: tuple[tuple[int, int, int], ...] = ()


@dataclass(frozen=True, slots=True)
class HCCLP2PStageLayout:
    """Immutable peer layout parsed once from a stage control payload."""

    peer_ranks: tuple[int, ...]
    seq_lens: tuple[int, ...]
    peer_slices: tuple[tuple[int, int, int], ...]
    num_tokens: int


@dataclass(frozen=True, slots=True)
class HCCLMTPHeader:
    """MTP phase shape and DP layout sent before the draft hidden tensor."""

    num_tokens: int
    speculative_step: int
    num_tokens_across_dp: torch.Tensor


@dataclass(frozen=True, slots=True)
class HCCLAttentionPipelineEvents:
    """NPU events ordering one Attention compute/send/receive exchange."""

    compute_done: Any
    send_done: Any
    recv_done: Any


@dataclass(frozen=True, slots=True)
class HCCLPendingAttentionTransfer:
    """Attention stream state retained between proxy send and receive calls."""

    layer_idx: int
    events: HCCLAttentionPipelineEvents


@dataclass(frozen=True, slots=True)
class HCCLAttentionReceiveDependency:
    """F2A receive that a layer-major Attention stage has not consumed yet."""

    tensor: torch.Tensor
    event: Any


_MTP_HEADER_MAGIC = 0x4D545031
_MTP_HEADER_PREFIX_SIZE = 4
_HCCL_COMMUNICATOR_HANDSHAKE_SIZE = 1


class P2pHcclAFDConnector(AFDConnectorBase):
    """Move AFD tensors with standard HCCL point-to-point operations."""

    yield_after_attn_send = False

    @classmethod
    def parse_extra_config(
        cls,
        raw: Mapping[str, Any] | None,
    ) -> ConnectorExtraInfo:
        if raw is not None and not isinstance(raw, Mapping):
            raise TypeError("HCCL P2P connector_extra_config must be a mapping")
        if raw:
            raise ValueError(
                "P2pHcclAFDConnector does not support connector_extra_config",
            )
        return ConnectorExtraInfo()

    def __init__(
        self,
        rank: int,
        local_rank: int,
        vllm_config: VllmConfig,
        afd_config: AFDConfig,
        role_rank: int,
    ) -> None:
        super().__init__(rank, local_rank, vllm_config, afd_config, role_rank)
        self.mapping = build_rank_mapping(afd_config, role_rank)
        # FFN token-count routing uses the connector topology contract shared
        # with CAMP2P. Here the generic rank mapping is the complete topology.
        self.topology = self.mapping

        self._initialized = False
        self.world_rank = self.mapping.world_rank
        self.p2p_rank = self.mapping.p2p_rank
        self.attn_size = self.mapping.attention_size
        self.ffn_size = self.mapping.ffn_size
        self.min_size = self.mapping.min_size
        self.ratio = self.mapping.ratio
        self.dst_list = list(self.mapping.dp_metadata_destinations)
        self.hidden_size = int(vllm_config.model_config.hf_config.hidden_size)
        self.dtype = vllm_config.model_config.dtype
        self.max_num_batched_tokens = int(
            vllm_config.scheduler_config.max_num_batched_tokens,
        )
        architectures = vllm_config.model_config.hf_config.architectures or []
        self.requires_input_ids = any(
            architecture in {"DeepseekV4ForCausalLM", "AFDDeepseekV4ForCausalLM"}
            for architecture in architectures
        )
        speculative_config = getattr(vllm_config, "speculative_config", None)
        self.requires_mtp = (
            speculative_config is not None
            and getattr(speculative_config, "method", None) == "mtp"
        )
        self.vocab_size = int(vllm_config.model_config.hf_config.vocab_size)
        self.num_stages = max(
            1,
            int(vllm_config.parallel_config.num_ubatches),
        )
        self.stream_overlap_enabled = self.num_stages > 1
        self.preinitialize_graph_groups = (
            vllm_config.compilation_config.cudagraph_mode.has_full_cudagraphs()
        )

        self.data_pg_list: list[ProcessGroup] = []
        self.ids_pg_list: list[ProcessGroup] = []
        self.p2p_pg: ProcessGroup | None = None
        self.input_ids_buffers: list[torch.Tensor] = []
        self.mtp_header_buffers: list[torch.Tensor] = []
        self.hidden_recv_buffers: dict[int, torch.Tensor] = {}
        self.mtp_hidden_recv_buffers: dict[int, torch.Tensor] = {}
        self.dp_metadata_list: dict[int, DPMetadata | AFDDPMetadata] = {}
        self.stage_layouts: dict[int, HCCLP2PStageLayout] = {}
        self.is_graph_capturing = False
        self.is_warmup = False
        self.a2f_send_stream = None
        self.f2a_recv_streams = []
        self.attention_pipeline_events: dict[
            tuple[int, int], HCCLAttentionPipelineEvents
        ] = {}
        self.pending_attention_transfers: dict[int, HCCLPendingAttentionTransfer] = {}
        self.attention_receive_dependencies: dict[
            int, HCCLAttentionReceiveDependency
        ] = {}
        self.control_plane = P2pHcclAFDControlPlane(self)

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    def init_afd_connector(self) -> None:
        """Create one data group per stage and the separate control/ID groups."""
        if self._initialized:
            return

        # Importing torch_npu registers the HCCL backend. This connector does
        # not import or require the afd_ascend custom-op extension.
        import torch_npu  # noqa: F401

        timeout = timedelta(minutes=30)
        try:
            for stage_idx in range(self.num_stages):
                suffix = "" if stage_idx == 0 else f"_{stage_idx}"
                data_group = init_afd_process_group(
                    backend="hccl",
                    init_method=(
                        f"tcp://{self.afd_config.host}:{self.afd_config.port}"
                    ),
                    world_size=self.ffn_size + self.attn_size,
                    rank=self.world_rank,
                    group_name=f"afd_hccl_p2p{suffix}",
                    timeout=timeout,
                )
                self.data_pg_list.append(data_group)

                if self.requires_input_ids or self.requires_mtp:
                    ids_group = init_afd_process_group(
                        backend="hccl",
                        init_method=(
                            f"tcp://{self.afd_config.host}:{self.afd_config.port}"
                        ),
                        world_size=self.ffn_size + self.attn_size,
                        rank=self.world_rank,
                        group_name=f"afd_hccl_p2p_ids{suffix}",
                        timeout=timeout,
                    )
                    self.ids_pg_list.append(ids_group)
                    self.input_ids_buffers.append(
                        torch.empty(
                            self.max_num_batched_tokens,
                            dtype=torch.int32,
                            device=f"npu:{self.local_rank}",
                        ),
                    )
                    self.mtp_header_buffers.append(
                        torch.empty(
                            _MTP_HEADER_PREFIX_SIZE + self.ffn_size,
                            dtype=torch.int32,
                            device=f"npu:{self.local_rank}",
                        ),
                    )

            self.p2p_pg = init_afd_process_group(
                backend="gloo",
                init_method=f"tcp://{self.afd_config.host}:{self.afd_config.port}",
                world_size=self.ffn_size + self.attn_size,
                rank=self.p2p_rank,
                group_name="afd_hccl_p2p_control",
                timeout=timeout,
            )
            if self.preinitialize_graph_groups:
                self._preinitialize_graph_process_groups()
            if self.stream_overlap_enabled and self.afd_config.role == "attention":
                self._initialize_attention_stream_pipeline()
        except BaseException:
            self.close()
            raise
        self._initialized = True

    def close(self) -> None:
        """Destroy every process group and discard connector-owned buffers."""
        groups = [self.p2p_pg, *self.ids_pg_list, *self.data_pg_list]
        destroyed: set[int] = set()
        for group in groups:
            if group is None or id(group) in destroyed:
                continue
            destroyed.add(id(group))
            dist.destroy_process_group(group)

        self.p2p_pg = None
        self.ids_pg_list = []
        self.data_pg_list = []
        self.input_ids_buffers = []
        self.mtp_header_buffers = []
        self.hidden_recv_buffers = {}
        self.mtp_hidden_recv_buffers = {}
        self.dp_metadata_list = {}
        self.stage_layouts = {}
        self.a2f_send_stream = None
        self.f2a_recv_streams = []
        self.attention_pipeline_events = {}
        self.pending_attention_transfers = {}
        self.attention_receive_dependencies = {}
        self._initialized = False

    @property
    def attention_stream_pipeline_ready(self) -> bool:
        # Keep every operand Boolean so TorchDynamo does not need to trace
        # unsupported ``bool(dict)`` during the graph-mode profile run.
        return (
            self.stream_overlap_enabled
            and self.afd_config.role == "attention"
            and self.a2f_send_stream is not None
            and len(self.f2a_recv_streams) == self.num_stages
            and len(self.attention_pipeline_events) > 0
        )

    def _attention_stream_pipeline_active(self) -> bool:
        if not self.attention_stream_pipeline_ready:
            return False
        try:
            forward_context = get_forward_context()
        except AssertionError:
            # Standalone connector component tests intentionally run without
            # a vLLM model forward context and must retain the synchronous path.
            return False
        return bool(
            getattr(forward_context, "dbo_enabled", False)
            and int(getattr(forward_context, "num_ubatches", 1)) > 1
        )

    @staticmethod
    def _layer_major_attention_pipeline_active() -> bool:
        try:
            forward_context = get_forward_context()
        except AssertionError:
            return False
        return bool(getattr(forward_context, "afd_layer_major_u2", False))

    def _initialize_attention_stream_pipeline(self) -> None:
        """Create the eager U2 Attention communication streams and events."""
        device = torch.device("npu", self.local_rank)
        self.a2f_send_stream = torch.npu.Stream(device=device)
        # Keep one receive submission stream per HCCL stage. A slow receive on
        # one process group must not impose stream order on another stage.
        self.f2a_recv_streams = [
            torch.npu.Stream(device=device) for _ in range(self.num_stages)
        ]
        num_layers = int(self.vllm_config.model_config.hf_config.num_hidden_layers)
        self.attention_pipeline_events = {
            (layer_idx, stage_idx): HCCLAttentionPipelineEvents(
                compute_done=torch.npu.Event(),
                send_done=torch.npu.Event(),
                recv_done=torch.npu.Event(),
            )
            for layer_idx in range(num_layers)
            for stage_idx in range(self.num_stages)
        }

    def _preinitialize_graph_process_groups(self) -> None:
        """Initialize graph-mode P2P communicators before graph compilation.

        HCCL creates the pair communicator lazily on the first send/receive.
        Graph warmup transfers input IDs before entering the graph and hidden
        states inside the graph. A one-element round trip for both process-group
        types makes every peer enter communicator creation at connector startup.
        """

        handshake = torch.empty(
            _HCCL_COMMUNICATOR_HANDSHAKE_SIZE,
            dtype=torch.int32,
            device=f"npu:{self.local_rank}",
        )
        graph_groups: list[ProcessGroup] = []
        for stage_idx, data_group in enumerate(self.data_pg_list):
            graph_groups.append(data_group)
            if self.ids_pg_list:
                graph_groups.append(self.ids_pg_list[stage_idx])

        if self.afd_config.role == "attention":
            ffn_rank = self.mapping.subgroup_ranks[0]
            for group in graph_groups:
                dist.send(handshake, dst=ffn_rank, group=group)
                dist.recv(handshake, src=ffn_rank, group=group)
            return

        for group in graph_groups:
            for attention_rank in self._attention_peer_world_ranks():
                dist.recv(handshake, src=attention_rank, group=group)
                dist.send(handshake, dst=attention_rank, group=group)

    def send_attn_output(
        self,
        hidden_states: torch.Tensor,
        context: AFDTransferContext,
        **kwargs: Any,
    ) -> None:
        self._require_initialized()
        metadata = context.metadata
        if not torch.compiler.is_compiling() and not metadata.validate_tensor_shape(
            tuple(hidden_states.shape),
        ):
            raise ValueError(
                f"hidden_states shape {hidden_states.shape!r} does not match "
                f"HCCL P2P metadata token count {metadata.total_tokens}",
            )

        if metadata.phase == "mtp":
            self._validate_mtp_scope(metadata)
            expected_shape = (metadata.total_tokens, self.hidden_size)
            if tuple(hidden_states.shape) != expected_shape:
                raise ValueError(
                    "DSV4 MTP HCCL transfer requires post-HC MoE input shape "
                    f"{expected_shape}, got {tuple(hidden_states.shape)}"
                )
            num_tokens_across_dp = kwargs.get("num_tokens_across_dp")
            if num_tokens_across_dp is None:
                raise RuntimeError(
                    "DSV4 MTP HCCL transfer requires num_tokens_across_dp"
                )
            self.send_mtp_header(
                num_tokens=metadata.total_tokens,
                speculative_step=metadata.speculative_step,
                num_tokens_across_dp=num_tokens_across_dp,
                stage_idx=metadata.stage_idx,
            )
        elif self.requires_input_ids:
            input_ids: torch.Tensor | None = kwargs.get("input_ids")
            if metadata.layer_idx == 0:
                if input_ids is None:
                    raise RuntimeError("DSV4 HCCL P2P layer 0 requires input_ids")
                pretransferred = bool(
                    getattr(
                        get_forward_context(),
                        "afd_input_ids_pretransferred",
                        False,
                    ),
                )
                if not pretransferred:
                    if self._attention_stream_pipeline_active():
                        raise RuntimeError(
                            "HCCL P2P stream overlap requires input_ids to be "
                            "pretransferred before the Attention ubatch threads"
                        )
                    self.send_input_ids(input_ids, ubatch_idx=metadata.stage_idx)
                    maybe_apply_dbo_yield(input_ids, role="attention")
            elif input_ids is not None:
                raise RuntimeError("DSV4 HCCL P2P sends input_ids only at layer 0")

        group = self._data_group(metadata.stage_idx)
        if self._attention_stream_pipeline_active() and metadata.phase == "decoder":
            self._enqueue_attention_send(
                hidden_states,
                layer_idx=metadata.layer_idx,
                stage_idx=metadata.stage_idx,
                dst=self.mapping.subgroup_index,
                group=group,
            )
            return
        self._send_tensor(
            hidden_states,
            dst=self.mapping.subgroup_index,
            group=group,
        )

    def recv_ffn_output(
        self,
        ref_tensor: torch.Tensor,
        ubatch_idx: int = 0,
        **kwargs: Any,
    ) -> torch.Tensor:
        self._require_initialized()
        group = self._data_group(ubatch_idx)
        if self._attention_stream_pipeline_active():
            return self._enqueue_attention_receive(
                ref_tensor,
                stage_idx=ubatch_idx,
                src=self.mapping.subgroup_index,
                group=group,
            )
        self._recv_tensor(
            ref_tensor,
            src=self.mapping.subgroup_index,
            group=group,
        )
        # FFN processes layers in layer-major order (stage 0, then stage 1),
        # while each Attention DBO thread would otherwise continue directly
        # to the next layer. Yield after the blocking receive so the peer
        # stage can send the current layer before this stage sends layer + 1.
        maybe_apply_dbo_yield(ref_tensor, role="attention")
        return ref_tensor

    def _enqueue_attention_send(
        self,
        hidden_states: torch.Tensor,
        *,
        layer_idx: int,
        stage_idx: int,
        dst: int,
        group: ProcessGroup,
    ) -> None:
        if stage_idx in self.pending_attention_transfers:
            raise RuntimeError(
                "HCCL P2P Attention stage already has a pending receive: "
                f"stage={stage_idx}"
            )
        events = self._attention_events(layer_idx, stage_idx)
        compute_stream = torch.npu.current_stream()
        events.compute_done.record(compute_stream)
        assert self.a2f_send_stream is not None
        with torch.npu.stream(self.a2f_send_stream):
            events.compute_done.wait(self.a2f_send_stream)
            self._send_tensor(
                hidden_states,
                dst=dst,
                group=group,
                stream=self.a2f_send_stream,
            )
            events.send_done.record(self.a2f_send_stream)
        self.pending_attention_transfers[stage_idx] = HCCLPendingAttentionTransfer(
            layer_idx=layer_idx,
            events=events,
        )

    def _enqueue_attention_receive(
        self,
        ref_tensor: torch.Tensor,
        *,
        stage_idx: int,
        src: int,
        group: ProcessGroup,
    ) -> torch.Tensor:
        pending = self.pending_attention_transfers.pop(stage_idx, None)
        if pending is None:
            raise RuntimeError(
                "HCCL P2P Attention receive has no matching streamed send: "
                f"stage={stage_idx}"
            )
        recv_tensor = torch.empty_like(ref_tensor)
        recv_stream = self.f2a_recv_streams[stage_idx]
        with torch.npu.stream(recv_stream):
            pending.events.send_done.wait(recv_stream)
            self._recv_tensor(
                recv_tensor,
                src=src,
                group=group,
                stream=recv_stream,
            )
            pending.events.recv_done.record(recv_stream)

        if self._layer_major_attention_pipeline_active():
            if stage_idx in self.attention_receive_dependencies:
                raise RuntimeError(
                    "HCCL P2P Attention stage has an unconsumed F2A receive: "
                    f"stage={stage_idx}"
                )
            self.attention_receive_dependencies[stage_idx] = (
                HCCLAttentionReceiveDependency(
                    tensor=recv_tensor,
                    event=pending.events.recv_done,
                )
            )
            return recv_tensor

        # Let the peer ubatch enqueue its Attention work while this receive is
        # progressing on the connector-owned stream.
        maybe_apply_dbo_yield(recv_tensor, role="attention")
        compute_stream = torch.npu.current_stream()
        pending.events.recv_done.wait(compute_stream)
        self._record_stream(recv_tensor, compute_stream)
        return recv_tensor

    def wait_for_attention_stage_receive(
        self,
        *,
        stage_idx: int,
        tensor: torch.Tensor,
    ) -> None:
        """Make the current compute stream consume one deferred F2A receive."""
        dependency = self.attention_receive_dependencies.pop(stage_idx, None)
        if dependency is None:
            raise RuntimeError(
                "HCCL P2P Attention stage has no deferred F2A receive: "
                f"stage={stage_idx}"
            )
        if dependency.tensor is not tensor:
            raise RuntimeError(
                "HCCL P2P Attention stage received an unexpected tensor: "
                f"stage={stage_idx}"
            )
        compute_stream = torch.npu.current_stream()
        dependency.event.wait(compute_stream)
        self._record_stream(tensor, compute_stream)

    def require_attention_pipeline_idle(self) -> None:
        """Reject a new layer-major step when a prior step left stale state."""
        if self.pending_attention_transfers or self.attention_receive_dependencies:
            raise RuntimeError(
                "HCCL P2P Attention pipeline is not idle: "
                f"pending={tuple(self.pending_attention_transfers)} "
                f"deferred={tuple(self.attention_receive_dependencies)}"
            )

    def reset_attention_pipeline_state(self) -> None:
        """Discard per-step stream bookkeeping after a failed model forward."""
        self.pending_attention_transfers.clear()
        self.attention_receive_dependencies.clear()

    def _attention_events(
        self,
        layer_idx: int,
        stage_idx: int,
    ) -> HCCLAttentionPipelineEvents:
        try:
            return self.attention_pipeline_events[(layer_idx, stage_idx)]
        except KeyError as exc:
            raise RuntimeError(
                "HCCL P2P Attention pipeline event is not initialized: "
                f"layer={layer_idx} stage={stage_idx}"
            ) from exc

    def recv_attn_output(
        self,
        ubatch_idx: int = 0,
        **kwargs: Any,
    ) -> AFDA2FTransferPayload:
        self._require_initialized()
        layer_idx = int(kwargs.get("layer_idx", 0))
        phase = str(kwargs.get("phase", "decoder"))
        speculative_step = int(kwargs.get("speculative_step", 0))
        stream = kwargs.get("_stream")
        if phase == "mtp":
            explicit_num_tokens = int(kwargs.get("num_tokens", 0))
            metadata_probe = AFDTransferMetadata.create_ffn_metadata(
                layer_idx=layer_idx,
                stage_idx=ubatch_idx,
                seq_lens=[explicit_num_tokens],
                phase=phase,
                speculative_step=speculative_step,
            )
            self._validate_mtp_scope(metadata_probe)
            peer_ranks = self._attention_peer_world_ranks()
            layout = HCCLP2PStageLayout(
                peer_ranks=peer_ranks,
                seq_lens=(explicit_num_tokens,),
                peer_slices=_make_peer_slices(
                    peer_ranks,
                    (explicit_num_tokens,),
                ),
                num_tokens=explicit_num_tokens,
            )
        else:
            layout = self._stage_layout(
                ubatch_idx,
                fallback=int(kwargs.get("max_num_tokens", 1)),
            )
        num_tokens = layout.num_tokens
        input_ids = None
        if phase == "decoder" and self.requires_input_ids and layer_idx == 0:
            input_ids = kwargs.get("input_ids")
            if input_ids is None:
                input_ids = self.recv_input_ids(num_tokens, ubatch_idx=ubatch_idx)

        hidden_states = (
            self._mtp_hidden_recv_buffer(ubatch_idx, num_tokens)
            if phase == "mtp"
            else self._hidden_recv_buffer(ubatch_idx, num_tokens)
        )
        group = self._data_group(ubatch_idx)
        for source_rank, start, end in layout.peer_slices:
            peer_slice = hidden_states[start:end]
            self._recv_tensor(
                peer_slice,
                src=source_rank,
                group=group,
                stream=stream,
            )
        metadata = AFDTransferMetadata.create_ffn_metadata(
            layer_idx=layer_idx,
            stage_idx=ubatch_idx,
            seq_lens=layout.seq_lens,
            phase=phase,
            speculative_step=speculative_step,
        )
        return AFDA2FTransferPayload(
            hidden_states=hidden_states,
            context=AFDTransferContext(
                metadata=metadata,
                states=HCCLP2PTransferState(
                    stage_idx=ubatch_idx,
                    num_tokens=num_tokens,
                    peer_ranks=layout.peer_ranks,
                    seq_lens=layout.seq_lens,
                    peer_slices=layout.peer_slices,
                ),
            ),
            input_ids=input_ids,
        )

    def send_ffn_output(
        self,
        ffn_output: torch.Tensor,
        context: AFDTransferContext,
        **kwargs: Any,
    ) -> None:
        self._require_initialized()
        metadata = context.metadata
        if not torch.compiler.is_compiling() and not metadata.validate_tensor_shape(
            tuple(ffn_output.shape),
        ):
            raise ValueError(
                f"ffn_output shape {ffn_output.shape!r} does not match HCCL P2P "
                f"metadata token count {metadata.total_tokens}",
            )
        stage_idx = int(kwargs.get("ubatch_idx", metadata.stage_idx))
        stream = kwargs.get("_stream")
        state = context.states
        if not isinstance(state, HCCLP2PTransferState):
            raise RuntimeError(
                "HCCL P2P FFN output requires its matching receive state",
            )
        if state.stage_idx != stage_idx:
            raise ValueError(
                "HCCL P2P FFN output stage does not match receive state: "
                f"{stage_idx} != {state.stage_idx}",
            )
        if tuple(metadata.seq_lens) != state.seq_lens:
            raise ValueError(
                "HCCL P2P FFN output sequence lengths do not match receive state",
            )
        if len(state.peer_ranks) != len(state.seq_lens):
            raise RuntimeError("HCCL P2P receive state has an invalid peer layout")

        group = self._data_group(stage_idx)
        peer_slices = state.peer_slices or _make_peer_slices(
            state.peer_ranks,
            state.seq_lens,
        )
        for destination_rank, start, end in peer_slices:
            peer_slice = ffn_output[start:end]
            self._send_tensor(
                peer_slice,
                dst=destination_rank,
                group=group,
                stream=stream,
            )

    def recv_attn_output_streamed(
        self,
        *,
        ubatch_idx: int,
        recv_stream,
        wait_event,
        done_event,
        **kwargs: Any,
    ) -> tuple[AFDA2FTransferPayload, Any]:
        """Enqueue an FFN-side receive while retaining synchronous recv APIs."""
        with torch.npu.stream(recv_stream):
            if wait_event is not None:
                wait_event.wait(recv_stream)
            payload = self.recv_attn_output(
                ubatch_idx=ubatch_idx,
                _stream=recv_stream,
                **kwargs,
            )
            self._record_stream(payload.hidden_states, recv_stream)
            if payload.input_ids is not None:
                self._record_stream(payload.input_ids, recv_stream)
            done_event.record(recv_stream)
        return payload, done_event

    def send_ffn_output_streamed(
        self,
        ffn_output: torch.Tensor,
        context: AFDTransferContext,
        *,
        ubatch_idx: int,
        send_stream,
        wait_event,
        done_event,
    ) -> Any:
        """Enqueue an FFN-side send while retaining synchronous send APIs."""
        with torch.npu.stream(send_stream):
            wait_event.wait(send_stream)
            self._record_stream(ffn_output, send_stream)
            self.send_ffn_output(
                ffn_output,
                context,
                ubatch_idx=ubatch_idx,
                _stream=send_stream,
            )
            done_event.record(send_stream)
        return done_event

    def send_input_ids(
        self,
        input_ids: torch.Tensor,
        *,
        ubatch_idx: int,
    ) -> None:
        buffer, group = self._ids_buffer_and_group(ubatch_idx)
        flat_ids = input_ids.reshape(-1)
        num_tokens = int(flat_ids.numel())
        self._validate_input_ids(flat_ids, num_tokens)
        buffer[:num_tokens].copy_(flat_ids, non_blocking=False)
        dist.send(
            buffer[:num_tokens],
            dst=self.mapping.subgroup_index,
            group=group,
        )

    def recv_input_ids(
        self,
        num_tokens: int,
        *,
        ubatch_idx: int,
    ) -> torch.Tensor:
        if num_tokens <= 0 or num_tokens > self.max_num_batched_tokens:
            raise ValueError(
                "DSV4 input_ids token count must be in "
                f"[1, {self.max_num_batched_tokens}], got {num_tokens}",
            )
        layout = self._stage_layout(
            ubatch_idx,
            fallback=num_tokens,
        )
        if layout.num_tokens != num_tokens:
            raise ValueError(
                "DSV4 input_ids token count does not match HCCL P2P peer layout: "
                f"{num_tokens} != {layout.num_tokens}",
            )
        buffer, group = self._ids_buffer_and_group(ubatch_idx)
        input_ids = buffer[:num_tokens]
        for source_rank, start, end in layout.peer_slices:
            peer_slice = input_ids[start:end]
            dist.recv(peer_slice, src=source_rank, group=group)
        return input_ids

    def send_mtp_header(
        self,
        *,
        num_tokens: int,
        speculative_step: int,
        num_tokens_across_dp: torch.Tensor,
        stage_idx: int,
    ) -> None:
        self._validate_mtp_header_values(
            num_tokens=num_tokens,
            speculative_step=speculative_step,
            num_tokens_across_dp=num_tokens_across_dp,
        )
        buffer, group = self._mtp_header_buffer_and_group(stage_idx)
        counts = num_tokens_across_dp.reshape(-1).to(
            device=buffer.device,
            dtype=torch.int32,
        )
        buffer[0] = _MTP_HEADER_MAGIC
        buffer[1] = speculative_step
        buffer[2] = num_tokens
        buffer[3] = self.ffn_size
        buffer[_MTP_HEADER_PREFIX_SIZE:].copy_(counts, non_blocking=False)
        self._send_tensor(
            buffer,
            dst=self.mapping.subgroup_index,
            group=group,
        )

    def recv_mtp_header(self, *, stage_idx: int) -> HCCLMTPHeader:
        if self.afd_config.role != "ffn":
            raise RuntimeError("only the FFN role receives MTP headers")
        self._validate_mtp_topology()
        buffer, group = self._mtp_header_buffer_and_group(stage_idx)
        source_rank = self._attention_peer_world_ranks()[0]
        dist.recv(buffer, src=source_rank, group=group)
        values = [int(value) for value in buffer.cpu().tolist()]
        if values[0] != _MTP_HEADER_MAGIC:
            raise RuntimeError(f"invalid DSV4 MTP HCCL header magic: {values[0]}")
        if values[3] != self.ffn_size:
            raise RuntimeError(
                "DSV4 MTP HCCL header DP size does not match FFN world: "
                f"{values[3]} != {self.ffn_size}"
            )
        counts = torch.tensor(values[_MTP_HEADER_PREFIX_SIZE:], dtype=torch.int32)
        self._validate_mtp_header_values(
            num_tokens=values[2],
            speculative_step=values[1],
            num_tokens_across_dp=counts,
        )
        return HCCLMTPHeader(
            num_tokens=values[2],
            speculative_step=values[1],
            num_tokens_across_dp=counts,
        )

    def prepare_stage_buffer(self, stage_idx: int, num_tokens: int) -> None:
        """Ensure the FFN receive buffer is allocated before posting recv."""
        if self.afd_config.role != "ffn":
            return
        self._hidden_recv_buffer(stage_idx, num_tokens)

    def _send_tensor(
        self,
        tensor: torch.Tensor,
        *,
        dst: int,
        group: ProcessGroup,
        stream=None,
    ) -> None:
        send_tensor = tensor if tensor.is_contiguous() else tensor.contiguous()
        if torch.compiler.is_compiling():
            _graph_hccl_send(send_tensor, dst=dst, group=group)
            return
        if stream is not None:
            self._record_stream(send_tensor, stream)
        dist.send(send_tensor, dst=dst, group=group)

    def _recv_tensor(
        self,
        tensor: torch.Tensor,
        *,
        src: int,
        group: ProcessGroup,
        stream=None,
    ) -> None:
        if torch.compiler.is_compiling():
            _graph_hccl_recv(tensor, src=src, group=group)
            return
        if stream is not None:
            self._record_stream(tensor, stream)
        dist.recv(tensor, src=src, group=group)

    @staticmethod
    def _record_stream(tensor: torch.Tensor, stream) -> None:
        if tensor.device.type == "npu":
            tensor.record_stream(stream)

    def _hidden_recv_buffer(
        self,
        stage_idx: int,
        num_tokens: int,
    ) -> torch.Tensor:
        self._validate_receive_capacity(num_tokens)
        buffer = self.hidden_recv_buffers.get(stage_idx)
        if (
            buffer is None
            or int(buffer.shape[0]) < num_tokens
            or int(buffer.shape[1]) != self.hidden_size
            or buffer.dtype != self.dtype
        ):
            buffer = torch.empty(
                (num_tokens, self.hidden_size),
                dtype=self.dtype,
                device=f"npu:{self.local_rank}",
            )
            self.hidden_recv_buffers[stage_idx] = buffer
        return buffer[:num_tokens]

    def _mtp_hidden_recv_buffer(
        self,
        stage_idx: int,
        num_tokens: int,
    ) -> torch.Tensor:
        self._validate_receive_capacity(num_tokens)
        buffer = self.mtp_hidden_recv_buffers.get(stage_idx)
        if (
            buffer is None
            or int(buffer.shape[0]) < num_tokens
            or tuple(buffer.shape[1:]) != (self.hidden_size,)
            or buffer.dtype != self.dtype
        ):
            buffer = torch.empty(
                (num_tokens, self.hidden_size),
                dtype=self.dtype,
                device=f"npu:{self.local_rank}",
            )
            self.mtp_hidden_recv_buffers[stage_idx] = buffer
        return buffer[:num_tokens]

    def _peer_token_counts_for_stage(
        self,
        stage_idx: int,
        *,
        fallback: int,
    ) -> list[int]:
        return list(self._stage_layout(stage_idx, fallback=fallback).seq_lens)

    def _stage_layout(
        self,
        stage_idx: int,
        *,
        fallback: int,
    ) -> HCCLP2PStageLayout:
        cached = self.stage_layouts.get(stage_idx)
        if cached is not None:
            return cached

        dp_metadata = self.dp_metadata_list.get(stage_idx)
        if dp_metadata is None:
            if self.afd_config.role == "ffn" and self.ratio > 1:
                raise RuntimeError(
                    "HCCL P2P FFN requires DP metadata for unequal topology",
                )
            seq_lens = (max(1, fallback),)
        else:
            attention_counts = _attention_token_counts(
                dp_metadata,
                attention_size=self.attn_size,
                fallback=fallback,
            )
            first_attention_rank = self.mapping.subgroup_index * self.ratio
            seq_lens = tuple(
                max(
                    1,
                    int(
                        attention_counts[first_attention_rank + offset]
                        if first_attention_rank + offset < len(attention_counts)
                        else fallback
                    ),
                )
                for offset in range(self.ratio)
            )

        peer_ranks = self._attention_peer_world_ranks()
        layout = HCCLP2PStageLayout(
            peer_ranks=peer_ranks,
            seq_lens=seq_lens,
            peer_slices=_make_peer_slices(peer_ranks, seq_lens),
            num_tokens=sum(seq_lens),
        )
        if dp_metadata is not None:
            self.stage_layouts[stage_idx] = layout
        return layout

    def _attention_peer_world_ranks(self) -> tuple[int, ...]:
        if self.afd_config.role != "ffn":
            raise RuntimeError("only an FFN rank owns Attention peer ranks")
        return tuple(self.mapping.subgroup_ranks[1:])

    def _validate_receive_capacity(self, num_tokens: int) -> None:
        if num_tokens <= 0 or num_tokens > self.max_num_batched_tokens:
            raise ValueError(
                "HCCL P2P aggregate token count must be in "
                f"[1, {self.max_num_batched_tokens}], got {num_tokens}; "
                "increase FFN max_num_batched_tokens for this A/F ratio",
            )

    def _validate_input_ids(
        self,
        flat_ids: torch.Tensor,
        num_tokens: int,
    ) -> None:
        if num_tokens <= 0 or num_tokens > self.max_num_batched_tokens:
            raise ValueError(
                "DSV4 input_ids token count must be in "
                f"[1, {self.max_num_batched_tokens}], got {num_tokens}",
            )
        if torch.compiler.is_compiling():
            return
        if flat_ids.dtype not in (torch.int32, torch.int64):
            raise TypeError(
                "DSV4 input_ids must use int32 or int64 before HCCL transfer; "
                f"got {flat_ids.dtype}",
            )
        # NPU IDs originate from vLLM's validated scheduler/tokenizer path.
        # Reading min/max back to the host here adds two device-wide syncs to
        # every decode step, so retain the value-domain check for CPU/Mock
        # boundaries and keep the production data path device-local.
        if flat_ids.device.type != "cpu":
            return
        min_id = int(flat_ids.min().item())
        max_id = int(flat_ids.max().item())
        if min_id < -1 or max_id >= self.vocab_size:
            raise ValueError(
                "DSV4 input_ids must contain -1 padding or IDs in "
                f"[0, {self.vocab_size}); got [{min_id}, {max_id}]",
            )

    def _data_group(self, stage_idx: int) -> ProcessGroup:
        if stage_idx < 0 or stage_idx >= len(self.data_pg_list):
            raise RuntimeError(
                f"HCCL P2P data stage {stage_idx} is not initialized",
            )
        return self.data_pg_list[stage_idx]

    def _ids_buffer_and_group(
        self,
        stage_idx: int,
    ) -> tuple[torch.Tensor, ProcessGroup]:
        if stage_idx < 0 or stage_idx >= len(self.ids_pg_list):
            raise RuntimeError(
                f"HCCL P2P input IDs stage {stage_idx} is not initialized",
            )
        return self.input_ids_buffers[stage_idx], self.ids_pg_list[stage_idx]

    def _mtp_header_buffer_and_group(
        self,
        stage_idx: int,
    ) -> tuple[torch.Tensor, ProcessGroup]:
        if stage_idx < 0 or stage_idx >= len(self.mtp_header_buffers):
            raise RuntimeError(
                f"HCCL P2P MTP header stage {stage_idx} is not initialized",
            )
        return self.mtp_header_buffers[stage_idx], self.ids_pg_list[stage_idx]

    def _validate_mtp_scope(self, metadata: AFDTransferMetadata) -> None:
        if metadata.phase != "mtp":
            raise ValueError("MTP scope validation requires phase=mtp")
        if not self.requires_mtp:
            raise RuntimeError("HCCL P2P MTP transfer is not enabled")
        self._validate_mtp_topology()
        if metadata.layer_idx != 0 or metadata.speculative_step != 0:
            raise RuntimeError(
                "DSV4 HCCL P2P MTP supports only layer 0/speculative step 0"
            )
        self._validate_receive_capacity(metadata.total_tokens)

    def _validate_mtp_topology(self) -> None:
        if self.attn_size != self.ffn_size or self.ratio != 1:
            raise RuntimeError("DSV4 HCCL P2P MTP requires equal A/F ranks")
        if len(self.data_pg_list) != 1:
            raise RuntimeError("DSV4 HCCL P2P MTP requires U1")

    def _validate_mtp_header_values(
        self,
        *,
        num_tokens: int,
        speculative_step: int,
        num_tokens_across_dp: torch.Tensor,
    ) -> None:
        self._validate_mtp_topology()
        self._validate_receive_capacity(num_tokens)
        if speculative_step != 0:
            raise RuntimeError("DSV4 HCCL P2P MTP supports only speculative step 0")
        count_size = int(num_tokens_across_dp.numel())
        if count_size != self.ffn_size:
            raise ValueError(
                "DSV4 MTP token-count vector must match FFN world size: "
                f"{count_size} != {self.ffn_size}"
            )
        # ### PATCH START: keep eager MTP traceable beside a target ACL graph.
        # torch.compile traces the draft model even when speculative-config
        # enforce_eager disables draft ACL graph capture. Tensor-to-list value
        # checks are data dependent and cannot be part of that full graph; the
        # scheduler-owned DP counts are validated on every non-compiled entry.
        if torch.compiler.is_compiling():
            return
        # ### PATCH END: keep eager MTP traceable beside a target ACL graph.
        counts = [int(value) for value in num_tokens_across_dp.reshape(-1).tolist()]
        if any(value < 0 for value in counts):
            raise ValueError("DSV4 MTP token counts cannot be negative")

    def _require_initialized(self) -> None:
        if not self._initialized:
            raise RuntimeError("HCCL P2P connector is not initialized")


class P2pHcclAFDControlPlane(AFDControlPlane):
    """Stage metadata transport for ``P2pHcclAFDConnector``."""

    def __init__(self, connector: P2pHcclAFDConnector) -> None:
        self.connector = connector

    def update_state_from_dp_metadata(
        self,
        payload: AFDControlPayload,
    ) -> None:
        connector = self.connector
        connector.dp_metadata_list = payload.dp_metadata_list
        connector.stage_layouts = {}
        connector.is_graph_capturing = payload.is_graph_capturing
        connector.is_warmup = payload.is_warmup
        if connector.afd_config.role != "ffn":
            return
        for stage_idx in payload.dp_metadata_list:
            layout = connector._stage_layout(int(stage_idx), fallback=1)
            connector.prepare_stage_buffer(int(stage_idx), layout.num_tokens)

    def send_dp_metadata_list(
        self,
        payload: AFDControlPayload,
    ) -> None:
        connector = self.connector
        if connector.p2p_pg is None:
            return
        if (
            connector.afd_config.role != "attention"
            or connector.mapping.rank_in_subgroup != 1
        ):
            return
        send_control_payload(
            payload,
            dst=connector.mapping.subgroup_index,
            group=connector.p2p_pg,
            device=torch.device("cpu"),
        )

    def recv_dp_metadata_list(self) -> AFDControlPayload:
        connector = self.connector
        if connector.p2p_pg is None:
            raise RuntimeError(
                "HCCL P2P DP metadata process group is not initialized",
            )
        first_attention_rank = connector.mapping.subgroup_index * connector.ratio
        source_rank = connector.ffn_size + first_attention_rank
        return recv_control_payload(
            src=source_rank,
            group=connector.p2p_pg,
            device=torch.device("cpu"),
        )


def _num_tokens_for_attention_rank(
    dp_metadata: DPMetadata | AFDDPMetadata,
    *,
    attention_rank: int,
    attention_size: int,
    fallback: int = 1,
) -> int:
    counts = _attention_token_counts(
        dp_metadata,
        attention_size=attention_size,
        fallback=fallback,
    )
    if 0 <= attention_rank < len(counts):
        return counts[attention_rank]
    return max(1, fallback)


def _attention_token_counts(
    dp_metadata: DPMetadata | AFDDPMetadata,
    *,
    attention_size: int,
    fallback: int,
) -> list[int]:
    counts = dp_metadata.num_tokens_across_dp_cpu.flatten().tolist()
    if not counts:
        return [max(1, fallback)] * attention_size
    if len(counts) < attention_size and attention_size % len(counts) == 0:
        tp_size = attention_size // len(counts)
        counts = [counts[index // tp_size] for index in range(attention_size)]
    return [max(1, int(count)) for count in counts]


def _make_peer_slices(
    peer_ranks: tuple[int, ...],
    seq_lens: tuple[int, ...],
) -> tuple[tuple[int, int, int], ...]:
    offset = 0
    slices = []
    for peer_rank, peer_tokens in zip(peer_ranks, seq_lens, strict=True):
        end = offset + peer_tokens
        slices.append((peer_rank, offset, end))
        offset = end
    return tuple(slices)


__all__ = [
    "HCCLMTPHeader",
    "HCCLP2PTransferState",
    "P2pHcclAFDConnector",
    "P2pHcclAFDControlPlane",
]
