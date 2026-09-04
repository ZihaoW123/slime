import re

import torch

_MEGATRON_PREFIX = "module.module."

_VISION_HIDDEN_SIZE = 1152
_VISION_HEAD_DIM = 72
_VISION_NUM_HEADS = _VISION_HIDDEN_SIZE // _VISION_HEAD_DIM

_VISION_PREFIX_MAPPINGS = [
    (f"{_MEGATRON_PREFIX}vision_model.patch_embed.proj.", "model.visual.patch_embed.proj."),
    (f"{_MEGATRON_PREFIX}vision_model.merger.patch_norm.", "model.visual.merger.norm."),
    (f"{_MEGATRON_PREFIX}vision_model.merger.linear_fc2.", "model.visual.merger.linear_fc2."),
]

_VISION_DECODER_SIMPLE_MAPPINGS = {
    "self_attention.linear_proj.weight": "attn.proj.weight",
    "self_attention.linear_proj.bias": "attn.proj.bias",
    "self_attention.linear_qkv.layer_norm_weight": "norm1.weight",
    "self_attention.linear_qkv.layer_norm_bias": "norm1.bias",
    "mlp.linear_fc1.bias": "mlp.linear_fc1.bias",
    "mlp.linear_fc2.bias": "mlp.linear_fc2.bias",
    "mlp.linear_fc1.layer_norm_weight": "norm2.weight",
    "mlp.linear_fc1.layer_norm_bias": "norm2.bias",
}

_TOP_LEVEL_MAPPINGS = {
    f"{_MEGATRON_PREFIX}embedding.word_embeddings.weight": "model.language_model.embed_tokens.weight",
    f"{_MEGATRON_PREFIX}output_layer.weight": "lm_head.weight",
    f"{_MEGATRON_PREFIX}decoder.final_layernorm.weight": "model.language_model.norm.weight",
}

_MTP_MAPPINGS = {
    "enorm.weight": "mtp.pre_fc_norm_embedding.weight",
    "hnorm.weight": "mtp.pre_fc_norm_hidden.weight",
    "final_layernorm.weight": "mtp.norm.weight",
    "eh_proj.weight": "mtp.fc.weight",
}

_DECODER_SIMPLE_MAPPINGS = {
    "self_attention.linear_proj.weight": "self_attn.o_proj.weight",
    "self_attention.linear_qkv.layer_norm_weight": "input_layernorm.weight",
    "mlp.linear_fc1.layer_norm_weight": "post_attention_layernorm.weight",
    "pre_mlp_layernorm.weight": "post_attention_layernorm.weight",
    "mlp.router.weight": "mlp.gate.weight",
    "mlp.router.expert_bias": "mlp.gate.e_score_correction_bias",
    "self_attention.dt_bias": "linear_attn.dt_bias",
    "self_attention.out_proj.weight": "linear_attn.out_proj.weight",
    "self_attention.q_layernorm.weight": "self_attn.q_norm.weight",
    "self_attention.k_layernorm.weight": "self_attn.k_norm.weight",
    "self_attention.in_proj.layer_norm_weight": "input_layernorm.weight",
}

_LINEAR_ATTN_PASSTHROUGH_KEYS = {"A_log"}

_IN_PROJ_SPLIT_SIZES = [1024, 1024, 1024, 1024, 2048, 16, 16]


def _normalize_param_name(name: str) -> str:
    if name.startswith("module.module.language_model."):
        name = _MEGATRON_PREFIX + name[len("module.module.language_model."):]
    while name.startswith("module.module.module."):
        name = name.replace("module.module.module.", "module.module.", 1)
    return name


def _reorder_gate_up(param: torch.Tensor) -> torch.Tensor:
    q = param.shape[0] // 4
    return torch.cat([param[0:q], param[2 * q:3 * q], param[q:2 * q], param[3 * q:4 * q]], dim=0)


def _convert_vision_param(name, param):
    if name == f"{_MEGATRON_PREFIX}vision_model.pos_embed.weight":
        return [("model.visual.pos_embed.weight", param)]

    for mg_prefix, hf_prefix in _VISION_PREFIX_MAPPINGS:
        if name.startswith(mg_prefix):
            return [(hf_prefix + name[len(mg_prefix):], param)]

    if name == f"{_MEGATRON_PREFIX}vision_model.merger.linear_fc1.weight":
        return [("model.visual.merger.linear_fc1.weight", _reorder_gate_up(param))]
    if name == f"{_MEGATRON_PREFIX}vision_model.merger.linear_fc1.bias":
        return [("model.visual.merger.linear_fc1.bias", param)]

    match = re.match(r"module\.module\.vision_model\.decoder\.layers\.(\d+)\.(.+)", name)
    if not match:
        return None

    layer_idx, rest = match.groups()
    prefix = f"model.visual.blocks.{layer_idx}"

    if rest == "self_attention.linear_qkv.weight":
        param = (
            param.view(_VISION_NUM_HEADS, 3, _VISION_HEAD_DIM, _VISION_HIDDEN_SIZE)
            .permute(1, 0, 2, 3)
            .contiguous()
            .view(3 * _VISION_NUM_HEADS * _VISION_HEAD_DIM, _VISION_HIDDEN_SIZE)
        )
        return [(f"{prefix}.attn.qkv.weight", param)]

    if rest == "self_attention.linear_qkv.bias":
        param = param.view(_VISION_NUM_HEADS, 3, _VISION_HEAD_DIM).permute(1, 0, 2).contiguous().view(-1)
        return [(f"{prefix}.attn.qkv.bias", param)]

    if rest == "mlp.linear_fc1.weight":
        return [(f"{prefix}.mlp.linear_fc1.weight", _reorder_gate_up(param))]
    if rest == "mlp.linear_fc2.weight":
        return [(f"{prefix}.mlp.linear_fc2.weight", param)]

    if rest in _VISION_DECODER_SIMPLE_MAPPINGS:
        return [(f"{prefix}.{_VISION_DECODER_SIMPLE_MAPPINGS[rest]}", param)]

    if name.startswith(f"{_MEGATRON_PREFIX}vision_model."):
        suffix = name[len(f"{_MEGATRON_PREFIX}vision_model."):]
        return [(f"model.visual.{suffix}", param)]

    return None


def _convert_mtp_layer(args, name, param, layer_idx):
    for mg_key, hf_name in _MTP_MAPPINGS.items():
        if mg_key in name:
            return [(hf_name, param)]

    if "transformer_layer" not in name and "mtp_model_layer" not in name:
        return None

    proxy_name = name.replace(f"mtp.layers.{layer_idx}.transformer_layer", f"decoder.layers.{layer_idx}")
    proxy_name = proxy_name.replace(f"mtp.layers.{layer_idx}.mtp_model_layer", f"decoder.layers.{layer_idx}")

    llm_prefix = f"model.language_model.layers.{layer_idx}"
    mtp_prefix = f"mtp.layers.{layer_idx}"
    return [
        (hf_name.replace(llm_prefix, mtp_prefix) if llm_prefix in hf_name else hf_name, tensor)
        for hf_name, tensor in convert_qwen3_5_to_hf(args, proxy_name, param)
    ]


def _split_in_proj_weight(param, prefix):
    chunk1, chunk2 = param.chunk(2, dim=0)
    p1 = torch.split(chunk1, _IN_PROJ_SPLIT_SIZES, dim=0)
    p2 = torch.split(chunk2, _IN_PROJ_SPLIT_SIZES, dim=0)

    qkv = torch.cat([p1[0], p2[0], p1[1], p2[1], p1[2], p1[3], p2[2], p2[3]], dim=0)
    z = torch.cat([p1[4], p2[4]], dim=0)
    b = torch.cat([p1[5], p2[5]], dim=0)
    a = torch.cat([p1[6], p2[6]], dim=0)

    return [
        (f"{prefix}.linear_attn.in_proj_a.weight", a),
        (f"{prefix}.linear_attn.in_proj_b.weight", b),
        (f"{prefix}.linear_attn.in_proj_qkv.weight", qkv),
        (f"{prefix}.linear_attn.in_proj_z.weight", z),
    ]


def _reshape_conv1d_weight(param, prefix):
    c_half1, c_half2 = param.chunk(2, dim=-1)
    blocks_h1 = c_half1.chunk(4, dim=0)
    blocks_h2 = c_half2.chunk(4, dim=0)
    new_param = torch.cat([
        blocks_h1[0], blocks_h2[0],
        blocks_h1[1], blocks_h2[1],
        blocks_h1[2], blocks_h1[3],
        blocks_h2[2], blocks_h2[3],
    ], dim=0)
    return [(f"{prefix}.linear_attn.conv1d.weight", new_param)]


def _convert_expert_param(rest, param, prefix):
    match = re.match(r"mlp\.experts\.(.+)\.weight(\d+)", rest)
    if not match:
        return None
    expert_type, expert_idx = match.groups()
    if expert_type == "linear_fc1":
        gate_weight, up_weight = param.chunk(2, dim=0)
        return [
            (f"{prefix}.mlp.experts.{expert_idx}.gate_proj.weight", gate_weight),
            (f"{prefix}.mlp.experts.{expert_idx}.up_proj.weight", up_weight),
        ]
    if expert_type == "linear_fc2":
        return [(f"{prefix}.mlp.experts.{expert_idx}.down_proj.weight", param)]
    raise ValueError(f"Unknown expert parameter name: {prefix} {expert_type}")


def _convert_shared_expert_param(rest, param, prefix):
    match = re.match(r"mlp\.shared_experts\.(.+)", rest)
    if not match:
        return None
    rest = match.groups()[0]
    if rest == "linear_fc1.weight":
        gate_weight, up_weight = param.chunk(2, dim=0)
        return [
            (f"{prefix}.mlp.shared_expert.gate_proj.weight", gate_weight),
            (f"{prefix}.mlp.shared_expert.up_proj.weight", up_weight),
        ]
    if rest == "linear_fc2.weight":
        return [(f"{prefix}.mlp.shared_expert.down_proj.weight", param)]
    if rest == "gate_weight":
        return [(f"{prefix}.mlp.shared_expert_gate.weight", param)]
    raise ValueError(f"Unknown shared expert parameter name: {prefix} {rest}")


def _convert_qkv_weight(args, param, prefix, head_dim, value_num_per_group):
    param = param.view(args.num_query_groups, -1, head_dim, args.hidden_size)
    q_param, k_param, v_param = torch.split(
        param, split_size_or_sections=[2 * value_num_per_group, 1, 1], dim=1
    )
    q_param = (
        q_param.reshape(args.num_query_groups, 2, value_num_per_group, head_dim, args.hidden_size)
        .transpose(1, 2)
        .reshape(-1, args.hidden_size)
    )
    k_param = k_param.reshape(-1, args.hidden_size)
    v_param = v_param.reshape(-1, args.hidden_size)
    return [
        (f"{prefix}.self_attn.q_proj.weight", q_param),
        (f"{prefix}.self_attn.k_proj.weight", k_param),
        (f"{prefix}.self_attn.v_proj.weight", v_param),
    ]


def _convert_qkv_bias(args, param, prefix, head_dim, value_num_per_group):
    param = param.view(args.num_query_groups, -1)
    q_bias, k_bias, v_bias = torch.split(
        param,
        split_size_or_sections=[value_num_per_group * head_dim, head_dim, head_dim],
        dim=1,
    )
    return [
        (f"{prefix}.self_attn.q_proj.bias", q_bias.contiguous().flatten()),
        (f"{prefix}.self_attn.k_proj.bias", k_bias.contiguous().flatten()),
        (f"{prefix}.self_attn.v_proj.bias", v_bias.contiguous().flatten()),
    ]


def convert_qwen3_5_to_hf(args, name, param):
    name = _normalize_param_name(name)

    vision_params = _convert_vision_param(name, param)
    if vision_params is not None:
        return vision_params

    if "mtp.layers" in name:
        parts = name.split(".")
        try:
            layer_idx = parts[parts.index("layers") + 1]
        except (ValueError, IndexError) as e:
            raise ValueError(f"Invalid MTP layer name format: {name}") from e
        result = _convert_mtp_layer(args, name, param, layer_idx)
        if result is not None:
            return result

    if name in _TOP_LEVEL_MAPPINGS:
        return [(_TOP_LEVEL_MAPPINGS[name], param)]

    try:
        head_dim = args.kv_channels if args.kv_channels is not None else args.hidden_size // args.num_attention_heads
    except AttributeError:
        head_dim = args.hidden_size // args.num_attention_heads
    value_num_per_group = args.num_attention_heads // args.num_query_groups

    match = re.match(r"module\.module\.decoder\.layers\.(\d+)\.(.+)", name)
    if not match:
        raise ValueError(f"Unknown parameter name: {name} with weight shape: {param.shape}")

    layer_idx, rest = match.groups()
    prefix = f"model.language_model.layers.{layer_idx}"

    if rest in _DECODER_SIMPLE_MAPPINGS:
        return [(f"{prefix}.{_DECODER_SIMPLE_MAPPINGS[rest]}", param)]

    if rest == "mlp.experts.linear_fc1":
        return [(f"{prefix}.mlp.experts.gate_up_proj", param)]
    if rest == "mlp.experts.linear_fc2":
        return [(f"{prefix}.mlp.experts.down_proj", param)]

    if rest == "self_attention.in_proj.weight":
        return _split_in_proj_weight(param, prefix)
    if rest == "self_attention.conv1d.weight":
        return _reshape_conv1d_weight(param, prefix)
    if rest == "self_attention.out_norm.weight":
        return [(f"{prefix}.linear_attn.norm.weight", param.float() + 1)]

    expert_result = _convert_expert_param(rest, param, prefix)
    if expert_result is not None:
        return expert_result

    shared_expert_result = _convert_shared_expert_param(rest, param, prefix)
    if shared_expert_result is not None:
        return shared_expert_result

    if rest == "self_attention.linear_qkv.weight":
        return _convert_qkv_weight(args, param, prefix, head_dim, value_num_per_group)
    if rest == "self_attention.linear_qkv.bias":
        return _convert_qkv_bias(args, param, prefix, head_dim, value_num_per_group)

    if rest == "mlp.linear_fc1.weight":
        gate_weight, up_weight = param.chunk(2, dim=0)
        return [
            (f"{prefix}.mlp.gate_proj.weight", gate_weight),
            (f"{prefix}.mlp.up_proj.weight", up_weight),
        ]
    if rest == "mlp.linear_fc2.weight":
        return [(f"{prefix}.mlp.down_proj.weight", param)]

    sub_key = rest[len("self_attention."):]
    if rest.startswith("self_attention.") and sub_key in _LINEAR_ATTN_PASSTHROUGH_KEYS:
        return [(f"{prefix}.linear_attn.{sub_key}", param.float())]

    raise ValueError(f"Unknown parameter name: {name} with weight shape: {param.shape}")
