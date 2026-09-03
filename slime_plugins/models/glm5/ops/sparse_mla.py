import torch

try:
    from .tilelang_sparse_mla_bwd import sparse_mla_bwd
    from .tilelang_sparse_mla_fwd import sparse_mla_fwd_interface
except ModuleNotFoundError as exc:
    if exc.name != "tilelang":
        raise
    sparse_mla_bwd = None
    sparse_mla_fwd_interface = None


def _pytorch_sparse_mla(q, kv, indices, scaling, d_v=512):
    """Differentiable sparse MLA reference path for accelerators without TileLang."""
    if kv.ndim != 3 or kv.shape[1] != 1:
        raise ValueError(f"Expected kv shape [tokens, 1, dim], got {tuple(kv.shape)}")
    if indices.ndim == 3:
        if indices.shape[1] != 1:
            raise ValueError(f"Expected one KV group in indices, got {tuple(indices.shape)}")
        indices = indices.squeeze(1)

    valid = indices >= 0
    safe_indices = indices.clamp(min=0).long()
    selected = kv.squeeze(1)[safe_indices]
    logits = torch.einsum("qhd,qkd->qhk", q.float(), selected.float()) * scaling
    logits = logits.masked_fill(~valid.unsqueeze(1), float("-inf"))
    probs = torch.softmax(logits, dim=-1).to(q.dtype)
    output = torch.einsum("qhk,qkv->qhv", probs, selected[..., :d_v])
    lse = torch.logsumexp(logits, dim=-1)
    return output, lse


class SparseMLA(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, kv, indices, scaling):
        """
        Args:
            q: Query tensor (seq_len, heads, dim_plus_tail_dim)
            kv: Key-Value tensor (seq_len_kv, kv_group, dim_plus_tail_dim)
            indices: Sparse indices tensor (seq_len, kv_group, topk)

        Returns:
            out: Output tensor (seq_len, heads, dim)
        """
        indices = indices.contiguous()
        q, kv = q.contiguous(), kv.contiguous()
        ctx.scaling = scaling
        if sparse_mla_fwd_interface is None:
            tl_out, tl_lse = _pytorch_sparse_mla(q, kv, indices, scaling)
        else:
            tl_out, tl_lse = sparse_mla_fwd_interface(q, kv, indices, sm_scale=scaling)

        # Save tensors for backward pass
        ctx.save_for_backward(q, kv, indices, tl_out, tl_lse)
        ctx.use_pytorch_fallback = sparse_mla_fwd_interface is None

        return tl_out, tl_lse

    @staticmethod
    def backward(ctx, grad_output, grad_lse):
        """
        Args:
            grad_output: Gradient of the loss with respect to output

        Returns:
            Gradients for q, kv, and indices (None for indices)
        """
        q, kv, indices, tl_out, tl_lse = ctx.saved_tensors
        scaling = ctx.scaling

        if ctx.use_pytorch_fallback:
            with torch.enable_grad():
                q_replay = q.detach().requires_grad_(True)
                kv_replay = kv.detach().requires_grad_(True)
                out, _ = _pytorch_sparse_mla(q_replay, kv_replay, indices, scaling)
                tl_dq, tl_dkv = torch.autograd.grad(
                    out,
                    (q_replay, kv_replay),
                    grad_output,
                )
        else:
            tl_dq, tl_dkv = sparse_mla_bwd(
                q, kv, tl_out, grad_output.contiguous(), indices, tl_lse, sm_scale=scaling
            )

        # Return gradients for each input (None for indices as it's not differentiable)
        return tl_dq, tl_dkv, None, None


class SGLangSparseMLA(torch.autograd.Function):
    """SGLang FlashMLA forward with the trainable TileLang backward."""

    @staticmethod
    def forward(ctx, q, kv, indices, scaling, d_v=512):
        from sgl_kernel.flash_mla import flash_mla_sparse_fwd

        q = q.contiguous()
        kv = kv.contiguous()
        indices = indices.contiguous()

        # flash_mla_sparse requires num_heads to be a multiple of 64 on Hopper
        # (sm90) and 128 on Blackwell (sm100/sm103). The kernel is NOT
        # padding-invariant on sm103: padding q from 64 -> 128 heads changes the
        # bf16 rounding of the real heads (~1 bf16 ULP). The SGLang rollout
        # (dsa_backend._forward_flashmla_sparse) always applies this padding on
        # Blackwell, so the train side MUST pad identically or train/rollout
        # logprobs diverge (0.027 on B300 vs 1.9e-7 on H100). Hopper needs no
        # padding for 64 heads (64 % 64 == 0), so this branch is a no-op there.
        num_heads = q.shape[1]
        required_padding = 128 if torch.cuda.get_device_capability(q.device)[0] >= 10 else 64
        need_padding = num_heads % required_padding != 0
        if need_padding:
            assert required_padding % num_heads == 0, (
                f"flash_mla_sparse num_heads {num_heads} cannot be padded to " f"{required_padding}"
            )
            q_input = q.new_zeros((q.shape[0], required_padding, q.shape[2]))
            q_input[:, :num_heads, :] = q
        else:
            q_input = q

        output, _, lse = flash_mla_sparse_fwd(
            q=q_input,
            kv=kv,
            indices=indices,
            sm_scale=scaling,
            d_v=d_v,
        )
        if need_padding:
            output = output[:, :num_heads, :].contiguous()
            lse = lse[:, :num_heads].contiguous()
        ctx.scaling = scaling
        ctx.d_v = d_v
        ctx.save_for_backward(q, kv, indices, output, lse.contiguous())
        return output, lse.to(torch.bfloat16)

    @staticmethod
    def backward(ctx, grad_output, grad_lse):
        q, kv, indices, output, lse = ctx.saved_tensors
        if grad_output is None:
            grad_output = torch.zeros_like(output)
        dq, dkv = sparse_mla_bwd(
            q,
            kv,
            output,
            grad_output.contiguous(),
            indices,
            lse,
            sm_scale=ctx.scaling,
        )
        return dq, dkv, None, None, None
