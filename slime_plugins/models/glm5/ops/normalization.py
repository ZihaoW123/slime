import torch
import torch.nn.functional as F


def layer_norm_fp32(module: torch.nn.Module, tensor: torch.Tensor) -> torch.Tensor:
    """Run LayerNorm in FP32 even when the module parameters are BF16."""

    bias = getattr(module, "bias", None)
    output = F.layer_norm(
        tensor.float(),
        module.normalized_shape,
        module.weight.float(),
        bias.float() if bias is not None else None,
        module.eps,
    )
    return output.to(tensor.dtype)
