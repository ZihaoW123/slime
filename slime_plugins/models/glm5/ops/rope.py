import torch

try:
    from apex.transformer.functional import (
        fused_apply_rotary_pos_emb_thd as _apex_apply_rotary_pos_emb_thd,
    )
except ImportError:
    _apex_apply_rotary_pos_emb_thd = None


def apply_rotary_pos_emb_thd(
    tensor: torch.Tensor,
    cu_seqlens: torch.Tensor,
    freqs: torch.Tensor,
) -> torch.Tensor:
    """Apply non-interleaved RoPE to packed THD tensors.

    GLM's caller has already converted the even/odd MLA layout into the
    first-half/second-half layout expected here.  Apex is optional on Ascend;
    use Megatron's differentiable PyTorch primitive when it is unavailable.
    """

    if _apex_apply_rotary_pos_emb_thd is not None:
        return _apex_apply_rotary_pos_emb_thd(tensor, cu_seqlens, freqs)

    from megatron.core.models.common.embeddings.rope_utils import (
        _apply_rotary_pos_emb_bshd,
    )

    sequence_lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
    packed_freqs = torch.cat([freqs[:length] for length in sequence_lengths], dim=0)
    if packed_freqs.shape[0] != tensor.shape[0]:
        raise ValueError(
            "Packed RoPE lengths do not match the token dimension: "
            f"{packed_freqs.shape[0]} != {tensor.shape[0]}"
        )
    return _apply_rotary_pos_emb_bshd(
        tensor.unsqueeze(1),
        packed_freqs,
        rotary_interleaved=False,
        multi_latent_attention=False,
    ).squeeze(1)
