# Copyright © 2026 Huawei Technologies Co., Ltd.
# Based on flash-linear-attention: https://github.com/fla-org/flash-linear-attention
#
# This file contains code copied and/or modified from the flash-linear-attention project.
# The original source code was licensed under the MIT license and included
# the following copyright notice:
# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
"""AscendC KDA: AscendC forward plus the fused AscendC backward.

Supported: chunk_size=64, K=128, V in {128, 256}, H == HV, dense inputs,
post-sigmoid beta, no initial/final state.
"""

import os

import torch
from torch.library import custom_op

from triton_ascend_kernels.attention.fla.kda.fla_utils import (
    autocast_custom_bwd,
    autocast_custom_fwd,
    input_guard,
)
from triton_ascend_kernels.attention.fla.kda.l2norm_kda import l2norm_bwd, l2norm_fwd

from fla_npu.ops.ascendc import chunk_kda_fwd as ascendc_chunk_kda_fwd

try:
    from fla_npu.ops.ascendc import chunk_kda_bwd as ascendc_chunk_kda_bwd
except (AttributeError, ImportError) as exc:
    raise ImportError(
        "fla_npu does not expose chunk_kda_bwd. The installed wheel is either too "
        "old or was built for a different SoC. Inspect the installed operators with:\n"
        "  ls $(python -c 'import fla_npu, os; print(os.path.dirname(fla_npu.__file__))')"
        "/opp/vendors/*/op_impl/*/ | grep chunk_kda\n"
        "Select --glm53-kda-backend=triton to keep training on the Triton path."
    ) from exc
import torch.nn.functional as F

CHUNK_SIZE = 64


@custom_op("slime_glm53::chunk_kda_fwd", mutates_args=())
def _chunk_kda_fwd_op(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    safe_gate: bool,
    lower_bound: float | None,
    use_gate_in_kernel: bool,
    A_log: torch.Tensor | None,
    dt_bias: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    (o, _final_state, g_cumsum, Aqk, Akk, w, _u, qg, kg, v_new, h, _) = ascendc_chunk_kda_fwd(
        q,
        k,
        v,
        g,
        beta,
        float(scale),
        CHUNK_SIZE,
        layout="BSND",
        initial_state=None,
        output_final_state=False,
        cu_seqlens=None,
        chunk_indices=None,
        safe_gate=bool(safe_gate),
        lower_bound=lower_bound,
        use_gate_in_kernel=bool(use_gate_in_kernel),
        A_log=A_log,
        dt_bias=dt_bias,
        disable_recompute=True,
        return_intermediate_states=False,
        state_v_first=False,
    )
    return o, g_cumsum, Aqk, Akk, w, qg, kg, v_new, h


def _bsnd_to_bnsd(tensor):
    if tensor is None:
        return None
    if tensor.dim() != 4:
        raise RuntimeError(f"Expected a rank-4 BSND tensor, got shape {tuple(tensor.shape)}.")
    return tensor.permute(0, 2, 1, 3).contiguous()


def _bnsd_to_bsnd(tensor):
    if tensor is None:
        return None
    if tensor.dim() != 4:
        raise RuntimeError(f"Expected a rank-4 BNSD tensor, got shape {tuple(tensor.shape)}.")
    return tensor.permute(0, 2, 1, 3).contiguous()


def _bsh_to_bhs(tensor):
    if tensor is None:
        return None
    if tensor.dim() != 3:
        raise RuntimeError(f"Expected a rank-3 BSH tensor, got shape {tuple(tensor.shape)}.")
    return tensor.permute(0, 2, 1).contiguous()


def _bhs_to_bsh(tensor):
    if tensor is None:
        return None
    if tensor.dim() != 3:
        raise RuntimeError(f"Expected a rank-3 BHS tensor, got shape {tuple(tensor.shape)}.")
    return tensor.permute(0, 2, 1).contiguous()


def _check_intermediates(q, v, gk, Aqk, Akk, w, qg, kg, v_new, h):
    batch, tokens, heads, key_dim = q.shape
    value_dim = v.shape[3]
    chunks = (tokens + CHUNK_SIZE - 1) // CHUNK_SIZE
    expected = (
        ("g_cumsum", gk, (batch, heads, tokens, key_dim), torch.float32),
        ("Aqk", Aqk, (batch, heads, tokens, CHUNK_SIZE), q.dtype),
        ("Akk", Akk, (batch, heads, tokens, CHUNK_SIZE), q.dtype),
        ("w", w, (batch, heads, tokens, key_dim), q.dtype),
        ("qg", qg, (batch, heads, tokens, key_dim), q.dtype),
        ("kg", kg, (batch, heads, tokens, key_dim), q.dtype),
        ("v_new", v_new, (batch, heads, tokens, value_dim), q.dtype),
        ("h", h, (batch, chunks, heads, key_dim, value_dim), q.dtype),
    )
    for name, tensor, shape, dtype in expected:
        if tensor is None:
            raise RuntimeError(f"AscendC forward did not produce {name}.")
        if tuple(tensor.shape) != shape:
            raise RuntimeError(f"{name} has shape {tuple(tensor.shape)}, expected {shape}.")
        if tensor.dtype != dtype:
            raise RuntimeError(f"{name} has dtype {tensor.dtype}, expected {dtype}.")
        if not tensor.is_contiguous():
            raise RuntimeError(f"{name} is not contiguous.")


class ChunkKDAAscendCFunction(torch.autograd.Function):
    @staticmethod
    @input_guard
    @autocast_custom_fwd
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        scale: float,
        use_qk_l2norm_in_kernel: bool = False,
        use_gate_in_kernel: bool = False,
        safe_gate: bool = False,
        lower_bound: float | None = None,
    ):
        q_rstd = k_rstd = None
        if use_qk_l2norm_in_kernel:
            q, q_rstd = l2norm_fwd(q)
            k, k_rstd = l2norm_fwd(k)

        heads = q.shape[2]
        head_chunk_size = int(os.getenv("GLM53_KDA_FWD_HEAD_CHUNK_SIZE", "4"))
        if head_chunk_size <= 0:
            raise ValueError("GLM53_KDA_FWD_HEAD_CHUNK_SIZE must be positive")

        # AscendC KDA heads are independent. During activation recomputation,
        # the LM-head gradient is already resident on the final PP stage, so a
        # full-head forward workspace can exhaust physical HBM before backward
        # starts. Fill full-sized saved tensors from bounded head launches to
        # avoid both the full-head ACL workspace and concat temporaries.
        outputs = None
        output_head_dims = (2, 1, 1, 1, 1, 1, 1, 1, 2)
        for start in range(0, heads, head_chunk_size):
            end = min(start + head_chunk_size, heads)

            def head_slice(tensor, dim):
                if tensor is None:
                    return None
                return tensor.narrow(dim, start, end - start).contiguous()

            chunk_outputs = _chunk_kda_fwd_op(
                head_slice(q, 2),
                head_slice(k, 2),
                head_slice(v, 2),
                head_slice(g, 2),
                head_slice(beta, 2),
                float(scale),
                bool(safe_gate),
                lower_bound,
                bool(use_gate_in_kernel),
                head_slice(A_log, 0) if use_gate_in_kernel else None,
                head_slice(dt_bias, 0) if use_gate_in_kernel else None,
            )
            if outputs is None:
                outputs = []
                for tensor, head_dim in zip(chunk_outputs, output_head_dims, strict=True):
                    shape = list(tensor.shape)
                    shape[head_dim] = heads
                    outputs.append(tensor.new_empty(shape))
            for output, tensor, head_dim in zip(outputs, chunk_outputs, output_head_dims, strict=True):
                output.narrow(head_dim, start, end - start).copy_(tensor)

        assert outputs is not None
        (o, g_cumsum, Aqk, Akk, w, qg, kg, v_new, h) = outputs
        final_state = None

        _check_intermediates(q, v, g_cumsum, Aqk, Akk, w, qg, kg, v_new, h)

        ctx.save_for_backward(
            q,
            q_rstd,
            k,
            k_rstd,
            v,
            g,
            beta,
            A_log,
            dt_bias,
            g_cumsum,
            Aqk,
            Akk,
            w,
            qg,
            kg,
            v_new,
            h,
        )
        ctx.scale = scale
        ctx.safe_gate = safe_gate
        ctx.lower_bound = lower_bound
        ctx.use_qk_l2norm_in_kernel = use_qk_l2norm_in_kernel
        ctx.use_gate_in_kernel = use_gate_in_kernel
        return o.type_as(q), final_state

    @staticmethod
    @input_guard
    @autocast_custom_bwd
    def backward(ctx, do: torch.Tensor, dht: torch.Tensor):
        (q, q_rstd, k, k_rstd, v, g_input, beta, A_log, dt_bias, g_cumsum, Aqk, Akk, w, qg, kg, v_new, h) = ctx.saved_tensors

        if dht is not None:
            raise RuntimeError("The fused AscendC KDA backward has no final-state gradient.")

        gate_dt_bias = None
        if ctx.use_gate_in_kernel and dt_bias is not None:
            gate_dt_bias = dt_bias.reshape(q.shape[2], q.shape[3]).contiguous()

        q_h, k_h, v_h = map(_bsnd_to_bnsd, (q, k, v))
        beta_h = _bsh_to_bhs(beta)
        do_h = _bsnd_to_bnsd(do)
        raw_g_h = _bsnd_to_bnsd(g_input) if ctx.use_gate_in_kernel else None
        heads = q_h.shape[1]
        head_chunk_size = int(os.getenv("GLM53_KDA_BWD_HEAD_CHUNK_SIZE", "4"))
        if head_chunk_size <= 0:
            raise ValueError("GLM53_KDA_BWD_HEAD_CHUNK_SIZE must be positive")

        # KDA heads are mathematically independent. Splitting only the fused
        # backward launch bounds the ACL workspace on 64-GiB A3 devices while
        # preserving the exact full-head forward and gradients.
        gradients = None
        dA = dbias = None
        for start in range(0, heads, head_chunk_size):
            end = min(start + head_chunk_size, heads)

            def head_slice(tensor, dim=1):
                if tensor is None:
                    return None
                return tensor.narrow(dim, start, end - start).contiguous()

            chunk_result = ascendc_chunk_kda_bwd(
                head_slice(q_h),
                head_slice(k_h),
                head_slice(v_h),
                head_slice(beta_h),
                head_slice(g_cumsum),
                head_slice(Aqk),
                head_slice(Akk),
                head_slice(w),
                head_slice(qg),
                head_slice(kg),
                head_slice(v_new),
                head_slice(h, dim=2),
                head_slice(do_h),
                float(ctx.scale),
                raw_g=head_slice(raw_g_h),
                A_log=head_slice(A_log, dim=0) if ctx.use_gate_in_kernel else None,
                dt_bias=head_slice(gate_dt_bias, dim=0),
                initial_state=None,
                dht=None,
                cu_seqlens=None,
                chunk_indices=None,
                chunk_size=CHUNK_SIZE,
                safe_gate=ctx.safe_gate,
                lower_bound=ctx.lower_bound,
                use_gate_in_kernel=ctx.use_gate_in_kernel,
                disable_recompute=True,
                use_exp2=True,
                state_v_first=False,
            )
            if gradients is None:
                gradients = []
                for tensor in chunk_result[:5]:
                    shape = list(tensor.shape)
                    shape[1] = heads
                    gradients.append(tensor.new_empty(shape))
                if chunk_result[6] is not None:
                    dA = torch.empty_like(A_log)
                if chunk_result[7] is not None:
                    dbias = torch.empty_like(gate_dt_bias)
            for output, tensor in zip(gradients, chunk_result[:5], strict=True):
                output.narrow(1, start, end - start).copy_(tensor)
            if dA is not None:
                dA.narrow(0, start, end - start).copy_(chunk_result[6])
            if dbias is not None:
                dbias.narrow(0, start, end - start).copy_(chunk_result[7])

        assert gradients is not None
        dq_h, dk_h, dv_h, db_h, dg_h = gradients
        dh0 = None

        dq, dk, dv = map(_bnsd_to_bsnd, (dq_h, dk_h, dv_h))
        db, dg = _bhs_to_bsh(db_h), _bnsd_to_bsnd(dg_h)
        if dbias is not None:
            dbias = dbias.reshape(dt_bias.shape)

        if ctx.use_qk_l2norm_in_kernel:
            dq = l2norm_bwd(q, q_rstd, dq)
            dk = l2norm_bwd(k, k_rstd, dk)

        # One gradient per positional argument of apply().
        return (dq.to(q), dk.to(k), dv.to(v), dg.to(g_input), db.to(beta), dA, dbias, None, None, None, None, None)


@torch.compiler.disable
def chunk_kda_ascendc(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    use_gate_in_kernel: bool = False,
    safe_gate: bool = False,
    lower_bound: float | None = None,
    cu_seqlens: torch.LongTensor | None = None,
    **kwargs,
):
    if cu_seqlens is not None:
        raise NotImplementedError("chunk_kda_ascendc supports dense inputs only.")
    if initial_state is not None:
        raise NotImplementedError("chunk_kda_ascendc does not support initial_state.")
    if output_final_state:
        raise NotImplementedError("chunk_kda_ascendc does not support output_final_state.")
    if kwargs.get("use_beta_sigmoid_in_kernel"):
        raise NotImplementedError("chunk_kda_ascendc expects post-sigmoid beta.")
    chunk_size = kwargs.pop("chunk_size", CHUNK_SIZE)
    if chunk_size != CHUNK_SIZE:
        raise ValueError(f"chunk_size must be {CHUNK_SIZE}, got {chunk_size}.")

    A_log = dt_bias = None
    if use_gate_in_kernel:
        A_log, dt_bias = kwargs["A_log"], kwargs.get("dt_bias")
        if A_log.dtype != torch.float32:
            raise TypeError(f"A_log must be float32, got {A_log.dtype}.")
        if dt_bias is not None and dt_bias.dtype != torch.float32:
            raise TypeError(f"dt_bias must be float32, got {dt_bias.dtype}.")
        if safe_gate:
            if lower_bound is None:
                raise ValueError("lower_bound is required when safe_gate=True.")
            if not -5 <= lower_bound < 0:
                raise ValueError(f"lower_bound must be in [-5, 0), got {lower_bound}.")

    B, T, H, K = q.shape
    HV, V = v.shape[2], v.shape[3]

    if q.shape != k.shape:
        raise ValueError(f"q and k must match, got {tuple(q.shape)} vs {tuple(k.shape)}.")
    if H != HV:
        raise NotImplementedError(f"GVA is unsupported: H={H}, HV={HV}.")
    if K != 128 or V not in (128, 256):
        raise NotImplementedError(f"Requires K=128 and V in (128, 256), got K={K}, V={V}.")
    if q.dtype not in (torch.float16, torch.bfloat16):
        raise TypeError(f"q must be float16 or bfloat16, got {q.dtype}.")
    if g.shape != (B, T, HV, K):
        raise ValueError(f"g must be {(B, T, HV, K)}, got {tuple(g.shape)}.")
    if beta.shape != (B, T, HV):
        raise ValueError(f"beta must be {(B, T, HV)}, got {tuple(beta.shape)}.")

    if scale is None:
        scale = K**-0.5

    pad_len = (-T) % CHUNK_SIZE
    if pad_len:
        q = F.pad(q, (0, 0, 0, 0, 0, pad_len), value=1.0)
        k = F.pad(k, (0, 0, 0, 0, 0, pad_len), value=1.0)
        v = F.pad(v, (0, 0, 0, 0, 0, pad_len))
        g = F.pad(g, (0, 0, 0, 0, 0, pad_len))
        beta = F.pad(beta, (0, 0, 0, pad_len))

    o, final_state = ChunkKDAAscendCFunction.apply(
        q,
        k,
        v,
        g,
        beta,
        A_log,
        dt_bias,
        scale,
        use_qk_l2norm_in_kernel,
        use_gate_in_kernel,
        safe_gate,
        lower_bound,
    )
    if pad_len:
        o = o[:, :T]
    return o, final_state
