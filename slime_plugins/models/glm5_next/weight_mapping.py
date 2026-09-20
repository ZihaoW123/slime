"""Reversible mapping between released GLM-5.3 weights and Transformers 5.16.

The published checkpoint uses separate depthwise KDA convolutions and per-expert
MoE tensors. Transformers stacks both, while the Megatron actor keeps globally
numbered experts. Keep this mapping testable on a CPU-only development machine.
"""

from __future__ import annotations

import re

import torch


_EXPERTS = re.compile(r"^(.*\.mlp\.experts)\.(gate_up_proj|down_proj)$")
_MCORE_EXPERT = re.compile(r"^(.*\.mlp)\.routed_moe\.experts\.local_experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight$")


def checkpoint_name(hf_name: str) -> str:
    """Map a one-to-one Transformers name to its published checkpoint name."""
    match = _MCORE_EXPERT.fullmatch(hf_name)
    if match:
        prefix, expert, projection = match.groups()
        return f"{prefix}.experts.{expert}.{projection}.weight"
    hf_name = hf_name.replace(".mlp.routed_moe.router.weight", ".mlp.gate.weight")
    hf_name = hf_name.replace(".mlp.routed_moe.router.expert_bias", ".mlp.gate.e_score_correction_bias")
    return hf_name.replace(".attn_hc.", ".hc_attn_").replace(".ffn_hc.", ".hc_ffn_").replace(".forget_gate.", ".")


def load_hf_tensor(hf_name: str, reader, config) -> torch.Tensor:
    """Build the tensor expected by the eager Transformers model."""
    if hf_name.endswith(".self_attn.conv1d.weight"):
        prefix = hf_name.removesuffix("conv1d.weight")
        return torch.cat(
            [reader.get_tensor(f"{prefix}{part}_conv1d.weight") for part in ("q", "k", "v")],
            dim=0,
        )

    match = _EXPERTS.fullmatch(hf_name)
    if match:
        prefix, projection = match.groups()
        num_experts = config.text_config.n_routed_experts
        if projection == "down_proj":
            return torch.stack([reader.get_tensor(f"{prefix}.{i}.down_proj.weight") for i in range(num_experts)])
        return torch.stack(
            [
                torch.cat(
                    (
                        reader.get_tensor(f"{prefix}.{i}.gate_proj.weight"),
                        reader.get_tensor(f"{prefix}.{i}.up_proj.weight"),
                    ),
                    dim=0,
                )
                for i in range(num_experts)
            ]
        )

    return reader.get_tensor(checkpoint_name(hf_name))


def export_hf_tensor(hf_name: str, tensor: torch.Tensor) -> list[tuple[str, torch.Tensor]]:
    """Return released-checkpoint names accepted by SGLang's GLM-5.3 loader."""
    if hf_name.endswith(".self_attn.conv1d.weight"):
        if tensor.shape[0] % 3:
            raise ValueError(f"KDA convolution is not divisible by three: {hf_name}")
        prefix = hf_name.removesuffix("conv1d.weight")
        return [(f"{prefix}{part}_conv1d.weight", chunk.contiguous()) for part, chunk in zip(("q", "k", "v"), tensor.chunk(3, dim=0), strict=True)]

    match = _EXPERTS.fullmatch(hf_name)
    if match:
        prefix, projection = match.groups()
        if projection == "down_proj":
            return [(f"{prefix}.{i}.down_proj.weight", part) for i, part in enumerate(tensor.unbind(0))]
        if tensor.shape[1] % 2:
            raise ValueError(f"MoE gate-up projection is not divisible by two: {hf_name}")
        converted = []
        for i, expert in enumerate(tensor.unbind(0)):
            gate, up = expert.chunk(2, dim=0)
            converted.extend(((f"{prefix}.{i}.gate_proj.weight", gate.contiguous()), (f"{prefix}.{i}.up_proj.weight", up.contiguous())))
        return converted

    return [(checkpoint_name(hf_name), tensor)]
