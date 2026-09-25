"""Megatron PP/EP/CP actor for GLM-5.3-Flash (TP=1).

Megatron owns the pipeline schedule, distributed optimizer, and expert token
dispatcher. KDA and causal-conv1d can use AscendC, Triton-Ascend, or the
reference eager implementation selected by command-line arguments.
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


def _causal_kda_decay_mask(g: torch.Tensor, chunk_size: int) -> torch.Tensor:
    """Exponentiate only finite, causally valid KDA decay exponents.

    The Transformers eager implementation exponentiates the complete pairwise
    matrix and masks its upper triangle afterwards. Since ``g`` is a cumulative
    negative decay, invalid future-token entries can overflow to ``inf``. Their
    forward values are subsequently zeroed, but autograd still encounters
    ``0 * inf`` in ``MulBackward`` and produces NaN gradients.

    Invalid entries are mathematically unused, so setting their exponent to
    zero before ``exp`` preserves every causal entry and makes backward finite.
    """
    exponent = g.unsqueeze(-2) - g.unsqueeze(-3)
    future = torch.triu(
        torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=g.device),
        diagonal=1,
    )
    return exponent.masked_fill(future.unsqueeze(-1), 0).exp().float()


def stable_chunk_kimi_delta_attention(
    query,
    key,
    value,
    g,
    beta,
    chunk_size=64,
    initial_state=None,
    output_final_state=False,
    use_qk_l2norm_in_kernel=False,
    **kwargs,
):
    """Transformers' eager KDA with causal masking applied before ``exp``."""
    from transformers.models.glm5_next.modeling_glm5_next import l2norm

    initial_dtype = query.dtype
    query, key, value, beta, g = [x.transpose(1, 2).contiguous().to(torch.float32) for x in (query, key, value, beta, g)]
    if use_qk_l2norm_in_kernel:
        query = l2norm(query, dim=-1, eps=1e-6)
        key = l2norm(key, dim=-1, eps=1e-6)

    batch_size, num_heads, sequence_length, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    scale = 1 / (query.shape[-1] ** 0.5)
    pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size
    total_sequence_length = sequence_length + pad_size

    query = F.pad(query, (0, 0, 0, pad_size)) * scale
    key = F.pad(key, (0, 0, 0, pad_size))
    value = F.pad(value, (0, 0, 0, pad_size))
    g = F.pad(g, (0, 0, 0, pad_size))
    beta = F.pad(beta, (0, pad_size))
    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)

    query, key, value, g, k_beta, v_beta = [x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1]) for x in (query, key, value, g, k_beta, v_beta)]
    beta = beta.reshape(beta.shape[0], beta.shape[1], -1, chunk_size)

    g = g.cumsum(dim=-2)
    upper_with_diagonal = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=0)
    decay_mask = _causal_kda_decay_mask(g, chunk_size)
    attn = -(k_beta.unsqueeze(-2) * key.unsqueeze(-3) * decay_mask).sum(dim=-1).masked_fill(upper_with_diagonal, 0)
    for i in range(1, chunk_size):
        row = attn[..., i, :i].clone()
        sub = attn[..., :i, :i].clone()
        attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)

    attn = attn + torch.eye(chunk_size, dtype=attn.dtype, device=attn.device)
    value = attn @ v_beta
    k_cumdecay = attn @ (k_beta * g.exp())

    last_recurrent_state = (
        torch.zeros(
            batch_size,
            num_heads,
            k_head_dim,
            v_head_dim,
            dtype=value.dtype,
            device=value.device,
        )
        if initial_state is None
        else initial_state.to(value)
    )
    core_attn_out = torch.zeros_like(value)

    strict_upper = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=1)
    for i in range(total_sequence_length // chunk_size):
        q_i = query[:, :, i]
        k_i = key[:, :, i]
        v_i = value[:, :, i]
        g_i = g[:, :, i]

        attn_inter = (q_i * g_i.exp()) @ last_recurrent_state
        attn_intra = (q_i.unsqueeze(-2) * k_i.unsqueeze(-3) * decay_mask[:, :, i]).sum(dim=-1).masked_fill(strict_upper, 0)
        v_prime = k_cumdecay[:, :, i] @ last_recurrent_state
        v_new = v_i - v_prime

        core_attn_out[:, :, i] = attn_inter + attn_intra @ v_new
        last_recurrent_state = last_recurrent_state * g_i[:, :, -1].exp().unsqueeze(-1) + (k_i * (g_i[:, :, -1:] - g_i).exp()).transpose(-1, -2) @ v_new

    if not output_final_state:
        last_recurrent_state = None

    core_attn_out = core_attn_out.reshape(core_attn_out.shape[0], core_attn_out.shape[1], -1, core_attn_out.shape[-1])
    core_attn_out = core_attn_out[:, :, :sequence_length]
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
    return core_attn_out, last_recurrent_state


def install_stable_eager_kda() -> None:
    """Use the numerically stable eager KDA implementation for training."""
    from transformers.models.glm5_next import modeling_glm5_next

    modeling_glm5_next.chunk_kimi_delta_attention = stable_chunk_kimi_delta_attention


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


def make_pipeline_payload(local: torch.Tensor, hc_mult: int, hidden_size: int) -> torch.Tensor:
    """Flatten the mHC streams into an owning tensor for Megatron PP.

    Megatron pseudo-deallocates a stage output after sending it downstream.
    ``reshape(...).contiguous()`` can still return a view when the reshape is
    already contiguous, so clone the payload to give the schedule independent
    storage while preserving its autograd edge.
    """
    return local.reshape(local.shape[1], 1, hc_mult * hidden_size).clone()


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
    expected_attention = ["deepseek_sparse_attention" if (layer + 1) % 4 == 0 else "linear_attention" for layer in range(text.num_hidden_layers)]
    if list(text.layer_types) != expected_attention:
        raise ValueError("Expected a leading-layer GLM-5.3 checkpoint with KDA/KDA/KDA/DSA attention blocks")
    expected_mlp = ["dense" if layer < 3 else "sparse" for layer in range(text.num_hidden_layers)]
    if list(text.mlp_layer_types) != expected_mlp:
        raise ValueError("Expected the first three MLPs to be dense and all remaining MLPs to be routed MoE")
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
    pp_size = args.pipeline_model_parallel_size
    pipeline_layout = getattr(args, "pipeline_model_parallel_layout", None)
    first_stage_layers = getattr(args, "decoder_first_pipeline_num_layers", None)
    last_stage_layers = getattr(args, "decoder_last_pipeline_num_layers", None)
    if pipeline_layout is not None:
        if first_stage_layers is not None or last_stage_layers is not None:
            raise ValueError("pipeline layout cannot be combined with edge-stage layer counts")
    elif first_stage_layers is None and last_stage_layers is None:
        if args.num_layers % pp_size:
            raise ValueError("num_layers must divide PP unless an edge-stage layer count is set")
    else:
        if pp_size < 2:
            raise ValueError("uneven pipeline stages require PP > 1")
        if first_stage_layers is not None and not 1 <= first_stage_layers < args.num_layers:
            raise ValueError("the first pipeline stage must contain between 1 and num_layers-1 decoder layers")
        if last_stage_layers is not None and not 0 <= last_stage_layers < args.num_layers:
            raise ValueError("the last pipeline stage must contain between 0 and num_layers-1 decoder layers")
        edge_layers = [value for value in (first_stage_layers, last_stage_layers) if value is not None]
        middle_stage_count = pp_size - len(edge_layers)
        middle_layers = args.num_layers - sum(edge_layers)
        if middle_stage_count == 0:
            valid_split = middle_layers == 0
        else:
            valid_split = middle_layers > 0 and middle_layers % middle_stage_count == 0
        if not valid_split:
            raise ValueError("pipeline edge stages must leave an equal positive layer count for every middle PP stage")
    if text.n_routed_experts % args.expert_model_parallel_size:
        raise ValueError("The expert count must be divisible by EP")
    if not getattr(args, "allgather_cp", False) and args.context_parallel_size > 1:
        raise ValueError("GLM-5.3 CP requires --allgather-cp for contiguous packed shards")


def configure_npu_attention(text_config) -> None:
    """Select the memory-efficient Transformers attention path on Ascend."""
    # GLM-5.3's DSA layers still build a dense boolean visibility mask.  The
    # eager implementation materializes the full FP32 attention matrix and
    # needs about 15 GiB per 8K sequence.  Transformers' NPU SDPA adapter keeps
    # the mask boolean and dispatches FlashAttentionScore instead.
    text_config._attn_implementation = "sdpa"


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
        self.recompute = (
            getattr(config, "recompute_granularity", None) == "full"
            and not getattr(config, "glm53_outer_moe_recompute", False)
        )
        # Keep the expert MLP and fp32 router weighting inside one checkpointed
        # region.  For an 8K sequence, converting a complete expert output to
        # fp32 otherwise creates a ~0.5 GiB transient tensor per EP rank.
        self.expert_token_chunk_size = 1024
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
        # Avoid retaining every expert output and then allocating a second
        # full-size buffer in torch.cat.  At 8K/top-k=8 that transient copy is
        # about 0.5 GiB per EP rank and is enough to OOM a middle PP stage.
        output = hidden.new_empty(hidden.shape)
        offset = 0
        for expert, tokens, weights in zip(self.local_experts.values(), chunks, prob_chunks, strict=True):
            for token_chunk, weight_chunk in zip(
                tokens.split(self.expert_token_chunk_size, dim=0),
                weights.split(self.expert_token_chunk_size, dim=0),
                strict=True,
            ):

                def weighted_expert(token_input, router_weight, module=expert):
                    expert_output = module(token_input)
                    return (expert_output.float() * router_weight.float().unsqueeze(-1)).to(hidden.dtype)

                if self.recompute and self.training and token_chunk.requires_grad:
                    weighted = checkpoint(
                        weighted_expert,
                        token_chunk,
                        weight_chunk,
                        use_reentrant=False,
                    )
                else:
                    weighted = weighted_expert(token_chunk, weight_chunk)
                next_offset = offset + token_chunk.shape[0]
                output[offset:next_offset] = weighted
                offset = next_offset
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
        self.recompute = (
            getattr(mcore_config, "recompute_granularity", None) == "full"
            and not getattr(mcore_config, "glm53_outer_moe_recompute", False)
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        routed, bias = self.routed_moe(hidden_states.transpose(0, 1).contiguous())
        if bias is not None:
            routed = routed + bias
        if self.recompute and self.training and hidden_states.requires_grad:
            shared = checkpoint(self.shared_experts, hidden_states, use_reentrant=False)
        else:
            shared = self.shared_experts(hidden_states)
        return routed.transpose(0, 1).contiguous() + shared


def forward_packed_moe_layer(
    layer,
    sequences,
    previous_topk_indices,
    recompute=False,
    moe_dispatch_token_chunk_size=512,
    expert_parallel_group=None,
):
    """Run attention per packed sequence and dispatch all tokens through MoE once.

    Dynamic batching can assign a different number of packed sequences to each
    EP rank.  Calling the complete decoder layer once per sequence would then
    issue a different number of MoE collectives on each rank and deadlock the
    expert all-to-all.  Attention must stay sequence-local, while the token
    dispatcher must see one concatenated tensor per layer and rank.
    """
    if len(sequences) != len(previous_topk_indices):
        raise ValueError("Packed hidden states and top-k state must have the same length")

    ffn_inputs = []
    residuals = []
    ffn_posts = []
    ffn_combs = []
    next_topk_indices = []
    lengths = []

    for hidden_states, previous in zip(sequences, previous_topk_indices, strict=True):
        dtype = hidden_states.dtype
        length = hidden_states.shape[1]
        positions = torch.arange(length, device=hidden_states.device).unsqueeze(0)
        valid = torch.ones(1, length, device=hidden_states.device, dtype=torch.bool)

        # Bind sequence-local metadata in the closure.  Checkpoint invokes this
        # function again during backward, after the loop has advanced to later
        # packed sequences.
        def attention_block(x, valid=valid, positions=positions, previous=previous, dtype=dtype):
            residual = x
            post, comb, attention_input = layer.attn_hc(x)
            attention_input = layer.input_layernorm(attention_input)
            if layer.block_type == "linear_attention":
                attention_output = layer.self_attn(
                    hidden_states=attention_input,
                    cache_params=None,
                    attention_mask=valid,
                )
                topk = None
            else:
                attention_output, _, topk = layer.self_attn(
                    hidden_states=attention_input,
                    attention_mask=valid,
                    position_ids=positions,
                    past_key_values=None,
                    use_cache=False,
                    position_embeddings=None,
                    prev_topk_indices=previous,
                )
            output = post.to(dtype).unsqueeze(-1) * attention_output.unsqueeze(-2)
            output = output + torch.matmul(comb.to(dtype).transpose(-1, -2), residual)
            return output if topk is None else (output, topk)

        if recompute and hidden_states.requires_grad:
            attention_result = checkpoint(attention_block, hidden_states, use_reentrant=False)
        else:
            attention_result = attention_block(hidden_states)
        if isinstance(attention_result, tuple):
            hidden_states, topk_indices = attention_result
        else:
            hidden_states, topk_indices = attention_result, None

        residual = hidden_states
        post, comb, ffn_input = layer.ffn_hc(hidden_states)
        ffn_inputs.append(layer.post_attention_layernorm(ffn_input))
        residuals.append(residual)
        ffn_posts.append(post)
        ffn_combs.append(comb)
        next_topk_indices.append(topk_indices)
        lengths.append(length)

    packed_ffn_input = torch.cat(ffn_inputs, dim=1)
    original_packed_length = packed_ffn_input.shape[1]
    if moe_dispatch_token_chunk_size < 1:
        raise ValueError("GLM-5.3 MoE dispatch token chunk size must be positive")
    # Dynamic batching may assign a different token count to each data rank,
    # while those ranks form one EP group. Synchronize and pad to the maximum
    # so every EP peer issues identical all-to-all chunk shapes. Dummy zero
    # tokens have zero expert/shared-MLP output and are removed before
    # reconstructing the packed sequences.
    max_packed_length = original_packed_length
    if expert_parallel_group is not None and torch.distributed.get_world_size(expert_parallel_group) > 1:
        length_tensor = torch.tensor(max_packed_length, device=packed_ffn_input.device, dtype=torch.int64)
        torch.distributed.all_reduce(length_tensor, op=torch.distributed.ReduceOp.MAX, group=expert_parallel_group)
        max_packed_length = int(length_tensor.item())
    if original_packed_length < max_packed_length:
        padding = packed_ffn_input.new_zeros(
            packed_ffn_input.shape[0],
            max_packed_length - original_packed_length,
            packed_ffn_input.shape[2],
        )
        packed_ffn_input = torch.cat((packed_ffn_input, padding), dim=1)
    packed_ffn_outputs = []
    for ffn_input_chunk in packed_ffn_input.split(moe_dispatch_token_chunk_size, dim=1):
        if recompute and ffn_input_chunk.requires_grad:
            # Top-k routing expands every token into multiple expert-token
            # rows.  Dispatching the whole 8K context at once creates a large
            # masked-select/all-to-all working set on every EP rank.  All EP
            # ranks have the same packed sequence length, so identical token
            # chunks preserve collective ordering while bounding that working
            # set. Checkpoint the complete routed block to release its router
            # and dispatcher activations between pipeline layers.
            ffn_output_chunk = checkpoint(layer.mlp, ffn_input_chunk, use_reentrant=False)
        else:
            ffn_output_chunk = layer.mlp(ffn_input_chunk)
        packed_ffn_outputs.append(ffn_output_chunk)
    packed_ffn_output = torch.cat(packed_ffn_outputs, dim=1)[:, :original_packed_length]
    sequence_ffn_outputs = packed_ffn_output.split(lengths, dim=1)
    outputs = []
    for ffn_output, residual, post, comb in zip(sequence_ffn_outputs, residuals, ffn_posts, ffn_combs, strict=True):
        dtype = residual.dtype
        outputs.append(post.to(dtype).unsqueeze(-1) * ffn_output.unsqueeze(-2) + torch.matmul(comb.to(dtype).transpose(-1, -2), residual))
    return outputs, next_topk_indices


class Glm5NextMegatronModel(nn.Module):
    """One PP stage; pipeline payload is the flattened four-stream mHC state."""

    # The published GLM-5.3 checkpoint has distinct input embedding and output
    # head weights. Megatron's PP gradient finalizer queries this standard
    # model interface even when the weights are not shared.
    share_embeddings_and_output_weights = False

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
        self.recompute_num_layers = max(1, getattr(mcore_config, "recompute_num_layers", 1))
        self.moe_dispatch_token_chunk_size = getattr(mcore_config, "glm53_moe_dispatch_token_chunk_size", 512)

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

    @property
    def deferred_output_layer(self):
        """Return the untied LM head for response-only projection in the loss.

        Applying the 155K-token output head to every prompt token would retain
        an 8K-by-vocabulary activation on the final PP stage.  The policy loss
        only consumes response positions, so the Megatron loss closure applies
        this layer after selecting those positions.
        """
        return self.hf_model.lm_head if self.post_process else None

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
        expert_parallel_group = mpu.get_expert_model_parallel_group()
        recompute_layers = self.recompute and self.training

        def run_layer_group(group_input, group_layers):
            sequence_outputs = [group_input[:, start:end] for start, end in intervals]
            topk_indices = [None] * len(sequence_outputs)
            # A one-layer outer checkpoint already drops the complete layer
            # graph. Nesting non-reentrant attention/MoE checkpoints inside
            # its replay runs the memory-heavy AscendC KDA forward twice and
            # raises the backward peak. Inner checkpoints are useful only when
            # an outer group contains multiple layers.
            inner_recompute = recompute_layers and len(group_layers) > 1
            for layer in group_layers:
                if isinstance(layer.mlp, Glm5NextMegatronMoE):
                    sequence_outputs, topk_indices = forward_packed_moe_layer(
                        layer,
                        sequence_outputs,
                        topk_indices,
                        # The outer reentrant group checkpoint runs its first
                        # forward under no_grad, so these inner checkpoints are
                        # only materialized while the group is replayed for
                        # backward.  Checkpointing attention and each routed
                        # token chunk during that replay prevents a group of
                        # MoE layers from retaining all dispatcher activations
                        # at once on the final PP stage.
                        recompute=inner_recompute,
                        moe_dispatch_token_chunk_size=self.moe_dispatch_token_chunk_size,
                        expert_parallel_group=expert_parallel_group,
                    )
                    continue

                next_outputs = []
                next_topk_indices = []
                for hidden, previous in zip(sequence_outputs, topk_indices, strict=True):
                    length = hidden.shape[1]
                    positions = torch.arange(length, device=hidden.device).unsqueeze(0)
                    valid = torch.ones(1, length, device=hidden.device, dtype=torch.bool)
                    hidden, current_topk = layer(
                        hidden,
                        attention_mask=valid,
                        position_ids=positions,
                        use_cache=False,
                        prev_topk_indices=previous,
                    )
                    next_outputs.append(hidden)
                    next_topk_indices.append(current_topk)
                sequence_outputs = next_outputs
                topk_indices = next_topk_indices
            return torch.cat(sequence_outputs, dim=1)

        local_layers = list(language_model.layers.values())
        for group_start in range(0, len(local_layers), self.recompute_num_layers):
            group_layers = tuple(local_layers[group_start : group_start + self.recompute_num_layers])
            if recompute_layers and full.requires_grad:
                # One-layer checkpoints retain too many 4x-HC boundaries at
                # 8K, while checkpointing the complete PP stage makes backward
                # replay all local activations together.  Small layer groups
                # bound both sides of that memory tradeoff.  Reentrant replay
                # also guarantees identical EP collective ordering.
                def group_forward(x, layers=group_layers):
                    return run_layer_group(x, layers)

                full = checkpoint(group_forward, full, use_reentrant=True)
            else:
                full = run_layer_group(full, group_layers)
        local = full[:, cp_rank * local_length : (cp_rank + 1) * local_length]
        if not self.post_process:
            return make_pipeline_payload(local, self.hc_mult, self.hidden_size)
        collapsed = language_model.norm(local.mean(dim=2))
        # The loss closure projects response positions in small chunks.  Do
        # not materialize full-sequence logits here: prompt positions never
        # contribute to PPO and dominate final-stage HBM at an 8K sequence.
        return collapsed


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
    from .kernels import install_glm53_kernels

    install_glm53_kernels(
        kda_backend=args.glm53_kda_backend,
        causal_conv1d_backend=args.glm53_causal_conv1d_backend,
        num_heads=text_config.linear_num_heads,
        safe_gate_lower_bound=text_config.linear_lower_bound,
        eager_kda_kernel=stable_chunk_kimi_delta_attention,
    )
    configure_npu_attention(text_config)
    mcore_config = core_transformer_config_from_args(args)
    mcore_config.glm53_swiglu_limit = text_config.swiglu_limit
    mcore_config.glm53_moe_dispatch_token_chunk_size = args.glm53_moe_dispatch_token_chunk_size
    # forward_packed_moe_layer checkpoints the complete routed chunk. Avoid
    # nested expert/shared-MLP checkpoints during its backward replay.
    mcore_config.glm53_outer_moe_recompute = getattr(args, "recompute_granularity", None) == "full"
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
