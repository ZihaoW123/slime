"""Megatron PP/EP/CP actor for reduced GLM-5.3-Flash (TP=1).

Megatron owns the pipeline schedule, distributed optimizer, and expert token
dispatcher. The new KDA/DSA/mHC math deliberately uses the reference eager
Transformers modules until the Ascend fused implementations are validated.
CP first reconstructs the packed sequence with a differentiable all-gather;
this is a correctness path, not a memory-saving CP attention kernel.
"""

from __future__ import annotations

import copy

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from .contract import packed_intervals


class _GatherSequence(torch.autograd.Function):
    """All-gather contiguous CP shards, summing gradients before scattering."""

    @staticmethod
    def forward(ctx, local: torch.Tensor, group):
        ctx.group = group
        ctx.world_size = dist.get_world_size(group)
        ctx.rank = dist.get_rank(group)
        if ctx.world_size == 1:
            return local
        parts = [torch.empty_like(local) for _ in range(ctx.world_size)]
        dist.all_gather(parts, local.contiguous(), group=group)
        return torch.cat(parts, dim=0)

    @staticmethod
    def backward(ctx, full_grad: torch.Tensor):
        if ctx.world_size == 1:
            return full_grad, None
        full_grad = full_grad.contiguous()
        dist.all_reduce(full_grad, group=ctx.group)
        return full_grad.chunk(ctx.world_size, dim=0)[ctx.rank].contiguous(), None


def gather_cp_sequence(local: torch.Tensor, group) -> torch.Tensor:
    return _GatherSequence.apply(local, group)


_LOCAL_EXPERT_KEY = ".routed_moe.experts.local_experts."


def use_expert_data_parallel_replica_ids(sharded_state_dict, expert_dp_rank: int):
    """Mark globally numbered experts as replicas only across expert DP.

    ``hf_model`` is intentionally an ordinary ``nn.Module`` tree, so MCore's
    top-level fallback converts its whole state dict in one pass and never
    reaches ``Glm5NextLocalExperts.sharded_state_dict``.  Repair the replica
    dimension after that conversion, using the same expert-DP rule as
    MCore's ``SequentialMLP``.
    """
    for key, sharding in sharded_state_dict.items():
        if _LOCAL_EXPERT_KEY not in key or not hasattr(sharding, "replica_id"):
            continue
        replica_id = sharding.replica_id
        if not isinstance(replica_id, tuple) or len(replica_id) != 3:
            raise ValueError(f"Expected (PP, TP, DP) replica_id for {key}, got {replica_id!r}")
        sharding.replica_id = (*replica_id[:2], expert_dp_rank)
    return sharded_state_dict


def validate_parallelism(args, hf_config) -> None:
    text = hf_config.text_config
    if getattr(hf_config, "quantization_config", None):
        raise ValueError("Megatron GLM-5.3 actor requires a BF16 checkpoint, not FP8 weights")
    if text.num_hidden_layers != 4 or list(text.layer_types) != ["linear_attention", "linear_attention", "linear_attention", "deepseek_sparse_attention"]:
        raise ValueError("Expected a four-layer checkpoint with 3 KDA layers and 1 DSA layer")
    if list(text.mlp_layer_types) != ["dense", "dense", "dense", "sparse"]:
        raise ValueError("Expected three dense MLP layers followed by one routed MoE layer")
    if text.n_group != 1 or text.topk_group != 1 or not text.norm_topk_prob:
        raise ValueError("The current Megatron router requires GLM-5.3's one-group, normalized top-k layout")
    expected = {
        "num_layers": text.num_hidden_layers,
        "hidden_size": text.hidden_size,
        "ffn_hidden_size": text.intermediate_size,
        "num_attention_heads": text.num_attention_heads,
        "num_experts": text.n_routed_experts,
        "moe_ffn_hidden_size": text.moe_intermediate_size,
        "moe_shared_expert_intermediate_size": text.moe_intermediate_size * text.n_shared_experts,
        "moe_router_topk": text.num_experts_per_tok,
    }
    for name, value in expected.items():
        if getattr(args, name) != value:
            raise ValueError(f"{name} must match the GLM-5.3 checkpoint ({value})")
    router_expected = {
        "moe_router_score_function": "sigmoid",
        "moe_router_enable_expert_bias": True,
        "moe_router_bias_update_rate": 0,
        "moe_router_topk_scaling_factor": text.routed_scaling_factor,
        "expert_tensor_parallel_size": 1,
    }
    for name, value in router_expected.items():
        if getattr(args, name) != value:
            raise ValueError(f"{name} must match the GLM-5.3 BF16 router ({value})")
    if args.tensor_model_parallel_size != 1:
        raise ValueError("GLM-5.3 Megatron actor does not yet support tensor parallelism; set TP=1")
    if getattr(args, "virtual_pipeline_model_parallel_size", None):
        raise ValueError("GLM-5.3 Megatron actor does not yet support virtual pipeline stages")
    if args.num_layers % args.pipeline_model_parallel_size:
        raise ValueError("The reduced model requires an even PP split (PP=1, 2, or 4)")
    if text.n_routed_experts % args.expert_model_parallel_size:
        raise ValueError("The expert count must be divisible by EP")
    if not getattr(args, "allgather_cp", False) and args.context_parallel_size > 1:
        raise ValueError("GLM-5.3 CP requires --allgather-cp for contiguous packed shards")


class _LocalExpert(nn.Module):
    """One published GLM-5.3 expert, including its clamped SwiGLU."""

    def __init__(self, hidden_size: int, intermediate_size: int, limit: float):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.limit = limit

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        gate = self.gate_proj(hidden).clamp(max=self.limit)
        up = self.up_proj(hidden).clamp(min=-self.limit, max=self.limit)
        return self.down_proj(F.silu(gate) * up)


class Glm5NextLocalExperts(nn.Module):
    """Megatron MoELayer expert interface, with globally numbered parameters."""

    def __init__(self, num_local_experts, config, pg_collection=None):
        super().__init__()
        if pg_collection is None:
            raise ValueError("Megatron expert process groups are required")
        ep_rank = dist.get_rank(pg_collection.ep)
        self.tp_group = pg_collection.expt_tp
        self.dp_group = pg_collection.expt_dp
        self.local_experts = nn.ModuleDict()
        first_id = ep_rank * num_local_experts
        for global_id in range(first_id, first_id + num_local_experts):
            expert = _LocalExpert(config.hidden_size, config.moe_ffn_hidden_size, config.glm53_swiglu_limit)
            for parameter in expert.parameters():
                parameter.allreduce = config.expert_model_parallel_size == 1
            self.local_experts[str(global_id)] = expert

    def forward(self, hidden, tokens_per_expert, probs):
        counts = tokens_per_expert.tolist()
        chunks = hidden.split(counts, dim=0)
        prob_chunks = probs.split(counts, dim=0)
        outputs = []
        for expert, tokens, weights in zip(self.local_experts.values(), chunks, prob_chunks, strict=True):
            outputs.append((expert(tokens).float() * weights.float().unsqueeze(-1)).to(hidden.dtype))
        output = torch.cat(outputs, dim=0) if outputs else hidden.new_empty(hidden.shape)
        return output, None

    def backward_dw(self):
        # nn.Linear computes weight gradients during autograd, unlike TE's
        # deferred grouped GEMM path.
        return None

    def sharded_state_dict(self, prefix="", sharded_offsets=(), metadata=None):
        """Checkpoint globally named experts with expert-DP replica IDs.

        The generic Megatron recursion uses the ordinary DP/CP group. With
        EP>1 that incorrectly turns the EP rank into a replica ID even though
        every global expert key exists on exactly one EP rank. Use the expert
        TP/DP groups, matching SequentialMLP's checkpoint contract.
        """
        from megatron.core.transformer.utils import make_sharded_tensors_for_checkpoint

        sharded = {}
        for global_id, expert in self.local_experts.items():
            expert_state = expert.state_dict(prefix="", keep_vars=True)
            sharded.update(
                make_sharded_tensors_for_checkpoint(
                    expert_state,
                    f"{prefix}local_experts.{global_id}.",
                    sharded_offsets=sharded_offsets,
                    tp_group=self.tp_group,
                    dp_cp_group=self.dp_group,
                )
            )
        return sharded


class Glm5NextMegatronMoE(nn.Module):
    """MCore distributed routed experts plus the reference replicated shared MLP."""

    def __init__(self, text_config, mcore_config, layer_number):
        super().__init__()
        from megatron.core.transformer.moe.moe_layer import MoELayer, MoESubmodules
        from transformers.models.glm5_next.modeling_glm5_next import Glm5NextTextMLP

        self.routed_moe = MoELayer(
            config=mcore_config,
            submodules=MoESubmodules(experts=Glm5NextLocalExperts),
            layer_number=layer_number,
        )
        self.shared_experts = Glm5NextTextMLP(
            config=text_config,
            intermediate_size=text_config.moe_intermediate_size * text_config.n_shared_experts,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        routed, bias = self.routed_moe(hidden_states.transpose(0, 1).contiguous())
        if bias is not None:
            routed = routed + bias
        return routed.transpose(0, 1).contiguous() + self.shared_experts(hidden_states)


class Glm5NextMegatronModel(nn.Module):
    """One PP stage; pipeline payload is the flattened four-stream mHC state."""

    def __init__(self, stage_config, text_config, local_layers, pre_process, post_process, mcore_config, recompute=False):
        super().__init__()
        from transformers.models.glm5_next.modeling_glm5_next import (
            Glm5NextTextDecoderLayer,
            Glm5NextTextRMSNorm,
        )

        self.config = stage_config
        self.pre_process = pre_process
        self.post_process = post_process
        self.text_config = text_config
        self.input_tensor = None
        self.hc_mult = text_config.hc_mult
        self.hidden_size = text_config.hidden_size
        self.recompute = recompute

        self.hf_model = nn.Module()
        self.hf_model.model = nn.Module()
        self.hf_model.model.language_model = nn.Module()
        language_model = self.hf_model.model.language_model
        if pre_process:
            language_model.embed_tokens = nn.Embedding(text_config.vocab_size, text_config.hidden_size, padding_idx=text_config.pad_token_id)
        language_model.layers = nn.ModuleDict()
        for layer_id in local_layers:
            # Avoid allocating 288 reference experts before installing MCore MoE.
            init_config = copy.deepcopy(text_config)
            if text_config.mlp_layer_types[layer_id] == "sparse":
                init_config.mlp_layer_types = list(init_config.mlp_layer_types)
                init_config.mlp_layer_types[layer_id] = "dense"
            layer = Glm5NextTextDecoderLayer(init_config, layer_id)
            if text_config.mlp_layer_types[layer_id] == "sparse":
                layer.mlp = Glm5NextMegatronMoE(text_config, mcore_config, layer_id + 1)
            language_model.layers[str(layer_id)] = layer
        if post_process:
            language_model.norm = Glm5NextTextRMSNorm(text_config.hidden_size, eps=text_config.rms_norm_eps)
            self.hf_model.lm_head = nn.Linear(text_config.hidden_size, text_config.vocab_size, bias=False)

    def set_input_tensor(self, input_tensor):
        self.input_tensor = input_tensor[0] if isinstance(input_tensor, (list, tuple)) else input_tensor

    def state_dict_for_save_checkpoint(self, prefix="", keep_vars=False):
        return self.state_dict(prefix=prefix, keep_vars=keep_vars)

    def sharded_state_dict(self, prefix="", sharded_offsets=(), metadata=None):
        from megatron.core import parallel_state
        from megatron.core.transformer.module import MegatronModule

        sharded = MegatronModule.sharded_state_dict(self, prefix, sharded_offsets, metadata)
        return use_expert_data_parallel_replica_ids(
            sharded,
            parallel_state.get_expert_data_parallel_rank(),
        )

    def forward(
        self,
        input_ids,
        position_ids=None,
        attention_mask=None,
        labels=None,
        packed_seq_params=None,
        loss_mask=None,
        **kwargs,
    ):
        from megatron.core import mpu

        if labels is not None or position_ids is not None or attention_mask is not None or kwargs:
            raise ValueError("GLM-5.3 Megatron actor currently supports packed text-only RL")
        if packed_seq_params is None:
            raise ValueError("Packed sequence offsets are required")
        cp_group = mpu.get_context_parallel_group()
        cp_rank = mpu.get_context_parallel_rank()
        local_length = input_ids.shape[1]
        language_model = self.hf_model.model.language_model
        if self.pre_process:
            local = language_model.embed_tokens(input_ids).transpose(0, 1).contiguous()
            full = gather_cp_sequence(local, cp_group)
            full = full.transpose(0, 1).unsqueeze(2).expand(-1, -1, self.hc_mult, -1).contiguous()
        else:
            if self.input_tensor is None:
                raise ValueError("Pipeline input tensor was not set")
            full = gather_cp_sequence(self.input_tensor, cp_group)
            full = full.view(1, -1, self.hc_mult, self.hidden_size)

        intervals = packed_intervals(packed_seq_params.cu_seqlens_q, full.shape[1])
        sequence_outputs = []
        for start, end in intervals:
            hidden = full[:, start:end]
            positions = torch.arange(end - start, device=hidden.device).unsqueeze(0)
            valid = torch.ones(1, end - start, device=hidden.device, dtype=torch.bool)
            topk_indices = None
            for layer in language_model.layers.values():

                def layer_forward(x, block=layer, previous=topk_indices, mask=valid, pos=positions):
                    return block(
                        x,
                        attention_mask=mask,
                        position_ids=pos,
                        use_cache=False,
                        prev_topk_indices=previous,
                    )

                if self.recompute and self.training:
                    hidden, topk_indices = checkpoint(layer_forward, hidden, use_reentrant=False)
                else:
                    hidden, topk_indices = layer_forward(hidden)
            sequence_outputs.append(hidden)
        full = torch.cat(sequence_outputs, dim=1)
        local = full[:, cp_rank * local_length : (cp_rank + 1) * local_length]
        if not self.post_process:
            return local.reshape(local_length, 1, self.hc_mult * self.hidden_size).contiguous()
        collapsed = language_model.norm(local.mean(dim=2))
        return self.hf_model.lm_head(collapsed).float()


def model_provider(pre_process=True, post_process=True, vp_stage=None):
    """Entry point for Megatron's custom model provider."""
    if vp_stage is not None:
        raise ValueError("Virtual PP is not implemented for GLM-5.3")
    from megatron.core.transformer.transformer_block import get_num_layers_to_build
    from megatron.core.transformer.transformer_block import get_transformer_layer_offset
    from megatron.training.arguments import core_transformer_config_from_args
    from megatron.training.global_vars import get_args
    from transformers import AutoConfig

    args = get_args()
    hf_config = AutoConfig.from_pretrained(args.hf_checkpoint, trust_remote_code=True, local_files_only=True)
    validate_parallelism(args, hf_config)
    text_config = hf_config.text_config
    text_config._attn_implementation = "eager"
    mcore_config = core_transformer_config_from_args(args)
    mcore_config.glm53_swiglu_limit = text_config.swiglu_limit
    mcore_config.moe_shared_expert_intermediate_size = None
    stage_config = copy.deepcopy(mcore_config)
    stage_config.hidden_size = text_config.hc_mult * text_config.hidden_size
    offset = get_transformer_layer_offset(mcore_config, vp_stage=None)
    count = get_num_layers_to_build(mcore_config, vp_stage=None)
    if offset + count > text_config.num_hidden_layers:
        raise ValueError("Pipeline stage exceeds the reduced checkpoint layer count")
    original_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.bfloat16)
        return Glm5NextMegatronModel(
            stage_config,
            text_config,
            range(offset, offset + count),
            pre_process,
            post_process,
            mcore_config,
            recompute=getattr(args, "recompute_granularity", None) == "full",
        )
    finally:
        torch.set_default_dtype(original_dtype)
