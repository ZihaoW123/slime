"""Selectable GLM-5.3 KDA and causal-conv1d training kernels.

The Transformers reference functions are module globals.  Installing the
selected implementations before decoder layers are constructed keeps the
model/checkpoint contract unchanged while allowing the expensive eager
operators to be replaced on Ascend.
"""

from __future__ import annotations

from collections.abc import Callable

import torch


GLM53_KERNEL_BACKENDS = ("ascendc", "triton", "eager")
_original_causal_conv1d: Callable | None = None


def _dependency_error(backend: str, operator: str) -> RuntimeError:
    return RuntimeError(f"GLM-5.3 {operator} backend '{backend}' is unavailable. Run slime-ascend/scripts/quick_install_a5_glm53flash.sh in the target environment, or explicitly select the eager backend for debugging.")


def load_kda_kernel(
    backend: str,
    eager_kernel: Callable,
    safe_gate_lower_bound: float | None = None,
) -> Callable:
    """Resolve a KDA implementation lazily so CPU tooling has no NPU dependency."""
    if backend == "eager":
        return eager_kernel
    try:
        if backend == "ascendc":
            from .ascendc_kda import chunk_kda_ascendc

            if safe_gate_lower_bound is None:
                raise ValueError("AscendC KDA backward requires GLM-5.3 safe-gate mode")
            return lambda *args, **kwargs: chunk_kda_ascendc(
                *args,
                **kwargs,
                safe_gate=True,
                lower_bound=safe_gate_lower_bound,
            )
        if backend == "triton":
            from triton_ascend_kernels.attention.fla.kda.chunk import chunk_kda

            return chunk_kda
    except ImportError as exc:
        raise _dependency_error(backend, "KDA") from exc
    raise ValueError(f"Unknown GLM-5.3 KDA backend: {backend!r}")


def _triton_causal_conv1d(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    activation: str | None,
    operator: Callable,
) -> torch.Tensor:
    """Adapt Transformers' BDT layout to MindSpeed-Ops' BTD contract."""
    x = hidden_states.transpose(1, 2).contiguous()
    y, _ = operator(
        x=x,
        weight=weight.transpose(0, 1).contiguous(),
        bias=bias,
        activation=activation,
    )
    return y.transpose(1, 2).contiguous().to(hidden_states.dtype)


def _ascendc_causal_conv1d(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    activation: str | None,
    num_heads: int,
    operator: Callable,
) -> torch.Tensor:
    """Run the three Q/K/V depthwise convolutions with the AscendC kernel."""
    if hidden_states.ndim != 3 or weight.ndim != 2:
        raise ValueError("GLM-5.3 causal_conv1d expects hidden [B,D,T] and weight [D,W]")
    channels = hidden_states.shape[1]
    if channels % 3 or weight.shape[0] != channels:
        raise ValueError(f"Expected concatenated Q/K/V channels, got {channels}")
    qkv_dim = channels // 3
    if qkv_dim % num_heads:
        raise ValueError(f"QKV channels ({qkv_dim}) must divide linear heads ({num_heads})")

    outputs = []
    for index in range(3):
        channel_slice = slice(index * qkv_dim, (index + 1) * qkv_dim)
        part_bias = None if bias is None else bias[channel_slice]
        y, _ = operator(
            x=hidden_states[:, channel_slice].transpose(1, 2).contiguous(),
            weight=weight[channel_slice].contiguous(),
            H=num_heads,
            bias=part_bias,
            activation=activation,
        )
        # AscendC returns [B,H,T,D/H].
        outputs.append(y.transpose(1, 2).reshape(hidden_states.shape[0], hidden_states.shape[2], qkv_dim).transpose(1, 2).contiguous())
    return torch.cat(outputs, dim=1).to(hidden_states.dtype)


def load_causal_conv1d_kernel(backend: str, num_heads: int, eager_kernel: Callable) -> Callable:
    """Resolve and adapt the selected causal-conv1d training implementation."""
    if backend == "eager":
        return eager_kernel
    try:
        if backend == "triton":
            from mindspeed_ops.api.triton.convolution import causal_conv1d

            return lambda hidden_states, weight, bias=None, activation=None, **kwargs: _triton_causal_conv1d(hidden_states, weight, bias, activation, causal_conv1d)
        if backend == "ascendc":
            from .ascendc_causal_conv1d import causal_conv1d_ascendc

            return lambda hidden_states, weight, bias=None, activation=None, **kwargs: _ascendc_causal_conv1d(
                hidden_states,
                weight,
                bias,
                activation,
                num_heads,
                causal_conv1d_ascendc,
            )
    except ImportError as exc:
        raise _dependency_error(backend, "causal_conv1d") from exc
    raise ValueError(f"Unknown GLM-5.3 causal_conv1d backend: {backend!r}")


def install_glm53_kernels(
    *,
    kda_backend: str,
    causal_conv1d_backend: str,
    num_heads: int,
    safe_gate_lower_bound: float | None,
    eager_kda_kernel: Callable,
) -> None:
    """Patch the Transformers dispatch points with the requested implementations."""
    global _original_causal_conv1d

    from transformers.models.glm5_next import modeling_glm5_next

    if _original_causal_conv1d is None:
        _original_causal_conv1d = modeling_glm5_next.causal_conv1d_fn
    modeling_glm5_next.chunk_kimi_delta_attention = load_kda_kernel(
        kda_backend,
        eager_kda_kernel,
        safe_gate_lower_bound,
    )
    modeling_glm5_next.causal_conv1d_fn = load_causal_conv1d_kernel(
        causal_conv1d_backend,
        num_heads,
        _original_causal_conv1d,
    )
