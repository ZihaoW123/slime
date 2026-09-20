"""Names and packed-sequence contract shared by GLM-5.3 Megatron actors."""

from __future__ import annotations


def hf_weight_name(adapter_name: str) -> str:
    """Strip Megatron wrappers while keeping globally numbered HF layers."""
    while adapter_name.startswith("module."):
        adapter_name = adapter_name.removeprefix("module.")
    if not adapter_name.startswith("hf_model."):
        raise KeyError(f"Not a GLM-5.3 model parameter: {adapter_name}")
    return adapter_name.removeprefix("hf_model.")


def packed_intervals(cu_seqlens, token_count: int) -> list[tuple[int, int]]:
    offsets = [int(value) for value in cu_seqlens]
    if len(offsets) < 2 or offsets[0] != 0 or offsets[-1] != token_count:
        raise ValueError(f"Invalid packed offsets for {token_count} tokens: {offsets}")
    if any(end <= start for start, end in zip(offsets, offsets[1:], strict=False)):
        raise ValueError(f"Packed offsets must be strictly increasing: {offsets}")
    return list(zip(offsets, offsets[1:], strict=False))
