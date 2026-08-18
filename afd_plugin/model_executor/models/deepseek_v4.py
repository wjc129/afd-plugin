# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""DeepSeek-V4 AFD model wrapper for the pinned Ascend runtime."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from itertools import islice
from typing import Any

import torch
import torch.nn as nn
from vllm.config import VllmConfig
from vllm.forward_context import get_forward_context
from vllm_ascend.models import deepseek_v4 as native

from afd_plugin.config import parse_afd_config
from afd_plugin.model_executor.models.deepseek_v2 import RemoteFFNProxy

_ATTENTION_ROLE = frozenset(("attention",))
_FFN_ROLE = frozenset(("ffn",))
_NO_ROLE = frozenset()


def _checkpoint_weight_roles(name: str) -> frozenset[str]:
    """Return the DSV4 AFD role that owns one raw checkpoint key."""
    normalized = name.removeprefix("model.")
    if normalized.startswith("mtp."):
        return _NO_ROLE

    parts = normalized.split(".")
    if len(parts) >= 3 and parts[0] == "layers" and parts[1].isdigit():
        return _FFN_ROLE if parts[2] == "ffn" else _ATTENTION_ROLE
    return _ATTENTION_ROLE


def _iter_role_weights(
    weights: Iterable[tuple[str, torch.Tensor]],
    *,
    role: str,
) -> Iterator[tuple[str, torch.Tensor]]:
    """Consume a checkpoint iterator once and retain the active role's keys."""
    for name, loaded_weight in weights:
        if role in _checkpoint_weight_roles(name):
            yield name, loaded_weight


class AFDDeepseekV4RemoteMoEProxy(RemoteFFNProxy):
    """Parameter-free DSV4 MoE stage executed by the remote FFN role."""

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_ids = None
        if self.layer_idx == 0:
            input_ids = getattr(get_forward_context(), "input_ids", None)
            if input_ids is None:
                raise RuntimeError(
                    "DSV4 layer 0 requires input_ids in the forward context"
                )
        return self._send_and_receive(hidden_states, input_ids=input_ids)


class AFDDeepseekV4DecoderLayer(native.DeepseekV2DecoderLayer):
    """DSV4 decoder layer that constructs only the active AFD role."""

    # Patch reason: native DSV4 constructs Attention, HC, and MoE for every role.
    # Patch functionality: construct Attention/HC or MoE, never both.
    # Signature: matches the pinned upstream function; no added parameters.
    # Upstream: vllm-ascend/vllm_ascend/models/deepseek_v4.py
    # Commit: f042ad88882e22a43af323b0df5691467bad8553
    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str,
        config: native.DeepseekV2Config | None = None,
        topk_indices_buffer: torch.Tensor | None = None,
        is_draft_layer: bool = False,
    ) -> None:
        # ### PATCH START: role-selective DSV4 construction.
        nn.Module.__init__(self)
        afd_config = parse_afd_config(vllm_config, validate=False)
        self.afd_role = afd_config.role

        if config is None:
            config = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        parallel_config = vllm_config.parallel_config

        self.config = config
        self.hidden_size = config.hidden_size
        self.layer_idx = int(prefix.split(sep=".")[-1])
        self.norm_eps = config.rms_norm_eps
        self.routed_scaling_factor = getattr(config, "routed_scaling_factor", 1.0)

        if self.afd_role == "attention":
            max_position_embeddings = config.rope_parameters[
                "original_max_position_embeddings"
            ]
            self.self_attn = native.DeepseekV4Attention(
                vllm_config=vllm_config,
                config=config,
                max_position_embeddings=max_position_embeddings,
                cache_config=cache_config,
                quant_config=quant_config,
                prefix=f"{prefix}.self_attn",
                topk_indices_buffer=topk_indices_buffer,
            )
            self.mlp = AFDDeepseekV4RemoteMoEProxy(layer_idx=self.layer_idx)
            self.input_layernorm = native.RMSNorm(
                config.hidden_size,
                eps=self.norm_eps,
            )
            self.post_attention_layernorm = native.RMSNorm(
                config.hidden_size,
                eps=self.norm_eps,
            )
            self.hc_mult = hc_mult = config.hc_mult
            self.hc_sinkhorn_iters = config.hc_sinkhorn_iters
            self.hc_eps = config.hc_eps
            mix_hc = (2 + hc_mult) * hc_mult
            hc_dim = hc_mult * config.hidden_size
            self.hc_attn_fn = nn.Parameter(
                torch.empty(mix_hc, hc_dim, dtype=torch.float32)
            )
            self.hc_ffn_fn = nn.Parameter(
                torch.empty(mix_hc, hc_dim, dtype=torch.float32)
            )
            self.hc_attn_base = nn.Parameter(
                torch.empty(mix_hc, dtype=torch.float32)
            )
            self.hc_ffn_base = nn.Parameter(
                torch.empty(mix_hc, dtype=torch.float32)
            )
            self.hc_attn_scale = nn.Parameter(torch.empty(3, dtype=torch.float32))
            self.hc_ffn_scale = nn.Parameter(torch.empty(3, dtype=torch.float32))
        else:
            self.self_attn = native.PPMissingLayer()
            self.mlp = native.DeepseekV4MoE(
                config=config,
                parallel_config=parallel_config,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
                is_draft_layer=is_draft_layer,
            )
        # ### PATCH END

    def hc_pre(
        self,
        x: torch.Tensor,
        hc_fn: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return torch.ops._C_ascend.npu_hc_pre_v2(
            x,
            hc_fn,
            hc_scale,
            hc_base,
            self.hc_mult,
            self.hc_sinkhorn_iters,
            self.norm_eps,
            self.hc_eps,
        )

    def hc_post(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        post: torch.Tensor,
        comb: torch.Tensor,
    ) -> torch.Tensor:
        output = torch.ops._C_ascend.npu_hc_post(
            x.unsqueeze(dim=0),
            residual.unsqueeze(dim=0),
            post.unsqueeze(dim=0),
            comb.unsqueeze(dim=0),
        )
        return output.squeeze(dim=0)

    # Patch reason: native forward invokes the locally constructed MoE.
    # Patch functionality: the Attention role invokes a parameter-free remote proxy.
    # Signature: matches the pinned upstream function; no added parameters.
    # Upstream: vllm-ascend/vllm_ascend/models/deepseek_v4.py
    # Commit: f042ad88882e22a43af323b0df5691467bad8553
    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        llama_4_scaling: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # ### PATCH START: reject accidental FFN full-model execution.
        if self.afd_role != "attention":
            raise RuntimeError("DSV4 FFN layers are connector-driven")
        # ### PATCH END
        residual = hidden_states.clone()
        hidden_states, post, comb = self.hc_pre(
            hidden_states,
            self.hc_attn_fn,
            self.hc_attn_scale,
            self.hc_attn_base,
        )
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            llama_4_scaling=llama_4_scaling,
        )
        hidden_states = self.hc_post(hidden_states, residual, post, comb)

        residual = hidden_states.clone()
        hidden_states, post, comb = self.hc_pre(
            hidden_states,
            self.hc_ffn_fn,
            self.hc_ffn_scale,
            self.hc_ffn_base,
        )
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.hc_post(hidden_states, residual, post, comb)
        return hidden_states, residual

    def compute_ffn_output(
        self,
        hidden_states: torch.Tensor,
        *,
        input_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.afd_role != "ffn":
            raise RuntimeError("DSV4 Attention role does not own local MoE weights")
        return self.mlp(hidden_states, input_ids=input_ids)


@native.support_torch_compile(
    dynamic_arg_dims={
        "input_ids": 0,
        "positions": 0,
        "intermediate_tensors": 0,
        "inputs_embeds": 0,
    }
)
class AFDDeepseekV4Model(native.DeepseekV4Model):
    """Role-aware DSV4 model with a remote-MoE Attention forward path."""

    fall_back_to_pt_during_load = False

    # Patch reason: native DSV4 allocates embedding, every full layer, and head HC.
    # Patch functionality: build role-owned modules and omit the disabled MTP buffer.
    # Signature: matches the pinned upstream function; no added parameters.
    # Upstream: vllm-ascend/vllm_ascend/models/deepseek_v4.py
    # Commit: f042ad88882e22a43af323b0df5691467bad8553
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        # ### PATCH START: initialize role-aware storage without native allocation.
        nn.Module.__init__(self)
        self.afd_config = parse_afd_config(vllm_config, validate=False)
        self.afd_role = self.afd_config.role
        # ### PATCH END

        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.config = config
        self.device = native.current_platform.device_type
        self.vocab_size = config.vocab_size
        self.is_v32 = hasattr(config, "index_topk")

        # ### PATCH START: DSA scratch data belongs only to Attention.
        if self.is_v32 and self.afd_role == "attention":
            self.topk_indices_buffer = torch.empty(
                vllm_config.scheduler_config.max_num_batched_tokens,
                config.index_topk,
                dtype=torch.int32,
                device=self.device,
            )
        else:
            self.topk_indices_buffer = None
        # ### PATCH END

        if self.afd_role == "attention" and native.get_pp_group().is_first_rank:
            self.embed_tokens = native.VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=f"{prefix}.embed_tokens",
            )
        else:
            self.embed_tokens = native.PPMissingLayer()

        self.start_layer, self.end_layer, self.layers = native.make_layers(
            config.num_hidden_layers,
            lambda prefix: AFDDeepseekV4DecoderLayer(
                vllm_config,
                prefix,
                topk_indices_buffer=self.topk_indices_buffer,
            ),
            prefix=f"{prefix}.layers",
        )

        if self.afd_role == "attention" and native.get_pp_group().is_last_rank:
            self.norm = native.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = native.PPMissingLayer()

        self.hc_mult = config.hc_mult

        def make_empty_intermediate_tensors(
            batch_size: int,
            dtype: torch.dtype,
            device: torch.device,
        ) -> native.IntermediateTensors:
            return native.IntermediateTensors(
                {
                    "hidden_states": torch.zeros(
                        (batch_size, self.hc_mult, config.hidden_size),
                        dtype=dtype,
                        device=device,
                    )
                }
            )

        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors

        # ### PATCH START: head HC and normalization belong only to Attention.
        if self.afd_role == "attention":
            self.norm_eps = config.rms_norm_eps
            self.hc_eps = config.hc_eps
            hc_dim = self.hc_mult * config.hidden_size
            self.hc_head_fn = nn.Parameter(
                torch.empty(self.hc_mult, hc_dim, dtype=torch.float32)
            )
            self.hc_head_base = nn.Parameter(
                torch.empty(self.hc_mult, dtype=torch.float32)
            )
            self.hc_head_scale = nn.Parameter(torch.empty(1, dtype=torch.float32))
        # MTP is unsupported in the first DSV4 AFD release, so no target buffer
        # is allocated or updated.
        # ### PATCH END

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        if self.afd_role != "attention":
            raise RuntimeError("DSV4 FFN role does not own token embeddings")
        return self.embed_tokens(input_ids)

    def hc_head(
        self,
        x: torch.Tensor,
        hc_fn: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
    ) -> torch.Tensor:
        shape, dtype = x.size(), x.dtype
        flattened = x.flatten(1).float()
        rsqrt = torch.rsqrt(
            flattened.square().mean(-1, keepdim=True) + self.norm_eps
        )
        mixes = torch.nn.functional.linear(flattened, hc_fn) * rsqrt
        pre = torch.sigmoid(mixes * hc_scale + hc_base) + self.hc_eps
        output = torch.sum(pre.unsqueeze(-1) * flattened.view(shape), dim=1)
        return output.to(dtype)

    # Patch reason: native forward runs native full layers and always updates MTP.
    # Patch functionality: run role-aware layers and omit disabled MTP state.
    # Signature: matches the pinned upstream function; no added parameters.
    # Upstream: vllm-ascend/vllm_ascend/models/deepseek_v4.py
    # Commit: f042ad88882e22a43af323b0df5691467bad8553
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: native.IntermediateTensors | None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | native.IntermediateTensors:
        # ### PATCH START: only Attention runs the complete DSV4 model.
        if self.afd_role != "attention":
            raise RuntimeError("DSV4 FFN model execution is connector-driven")
        # ### PATCH END
        if native.get_pp_group().is_first_rank:
            hidden_states = (
                inputs_embeds
                if inputs_embeds is not None
                else self.embed_input_ids(input_ids)
            )
        else:
            if intermediate_tensors is None:
                raise RuntimeError("pipeline stage requires intermediate tensors")
            hidden_states = intermediate_tensors["hidden_states"]

        llama_4_scaling = None
        aux_hidden_states: list[torch.Tensor] = []
        if native.get_pp_group().is_first_rank:
            hidden_states = hidden_states.unsqueeze(1).repeat(1, self.hc_mult, 1)

        for layer in islice(self.layers, self.start_layer, self.end_layer):
            hidden_states, _ = layer(
                positions,
                hidden_states,
                None,
                llama_4_scaling,
            )
            if layer.layer_idx + 1 in self.aux_hidden_state_layers:
                aux_hidden_states.append(hidden_states.mean(dim=1))

        if not native.get_pp_group().is_last_rank:
            return native.IntermediateTensors({"hidden_states": hidden_states})

        hidden_states = self.hc_head(
            hidden_states,
            self.hc_head_fn,
            self.hc_head_scale,
            self.hc_head_base,
        )
        hidden_states = self.norm(hidden_states)
        if aux_hidden_states:
            return hidden_states, aux_hidden_states
        return hidden_states

    def compute_ffn_output(
        self,
        hidden_states: torch.Tensor,
        layer_idx: int,
        **kwargs: Any,
    ) -> torch.Tensor:
        return self.layers[layer_idx].compute_ffn_output(hidden_states, **kwargs)


class AFDDeepseekV4ForCausalLM(native.AscendDeepseekV4ForCausalLM):
    """DSV4 causal LM wrapper with strict role ownership."""

    model_cls = AFDDeepseekV4Model

    # Patch reason: native construction allocates the LM head for both roles.
    # Patch functionality: build the head only for Attention and register FFN MoE.
    # Signature: matches the pinned upstream function; no added parameters.
    # Upstream: vllm-ascend/vllm_ascend/models/deepseek_v4.py
    # Commit: f042ad88882e22a43af323b0df5691467bad8553
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        # ### PATCH START: establish the AFD role before allocating modules.
        nn.Module.__init__(self)
        self.afd_config = parse_afd_config(vllm_config, validate=False)
        self.afd_role = self.afd_config.role
        # ### PATCH END
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.config = config
        self.quant_config = quant_config
        self.model = self.model_cls(
            vllm_config=vllm_config,
            prefix=native.maybe_prefix(prefix, "model"),
        )
        if self.afd_role == "attention" and native.get_pp_group().is_last_rank:
            self.lm_head = native.ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=native.maybe_prefix(prefix, "lm_head"),
            )
        else:
            self.lm_head = native.PPMissingLayer()
        self.logits_processor = native.LogitsProcessor(config.vocab_size)
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )
        self.num_moe_layers = config.num_hidden_layers
        self.set_moe_parameters()

    def set_moe_parameters(self) -> None:
        self.expert_weights = []
        self.num_expert_groups = getattr(self.config, "n_group", 1)
        self.moe_layers = []
        self.moe_mlp_layers = []
        example_moe = None
        if self.afd_role == "ffn":
            for layer in self.model.layers:
                if isinstance(layer, native.PPMissingLayer):
                    continue
                if isinstance(layer.mlp, native.DeepseekV4MoE):
                    example_moe = layer.mlp
                    self.moe_mlp_layers.append(layer.mlp)
                    self.moe_layers.append(layer.mlp.experts)
        self.extract_moe_parameters(example_moe)

    def compute_ffn_output(
        self,
        hidden_states: torch.Tensor,
        layer_idx: int,
        **kwargs: Any,
    ) -> torch.Tensor:
        return self.model.compute_ffn_output(hidden_states, layer_idx, **kwargs)

    def get_mtp_target_hidden_states(self) -> None:
        return None

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        return super().load_weights(
            _iter_role_weights(weights, role=self.afd_role)
        )


__all__ = ["AFDDeepseekV4ForCausalLM"]
