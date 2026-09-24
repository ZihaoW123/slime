import logging
import math
import os


logger = logging.getLogger(__name__)

_DEFAULT_CHUNK_NUMEL = 32 * 1024 * 1024
_DEFAULT_NORM_CHUNK_NUMEL = 4 * 1024 * 1024
_PATCH_MARKER = "_slime_npu_chunked_grad_scaling"
_NORM_PATCH_MARKER = "_slime_npu_chunked_grad_norm"


def _chunk_numel() -> int:
    value = int(os.getenv("SLIME_NPU_GRAD_SCALE_CHUNK_NUMEL", _DEFAULT_CHUNK_NUMEL))
    if value <= 0:
        raise ValueError("SLIME_NPU_GRAD_SCALE_CHUNK_NUMEL must be positive")
    return value


def scale_gradients_in_chunks(grad_data, scaling_factor, chunk_numel: int | None = None) -> None:
    """Scale a flat gradient buffer without materializing a buffer-sized NPU temporary.

    torch_npu may lower an in-place multiply on a multi-gigabyte flat tensor to an
    operator that requests output-sized workspace. Scaling bounded views keeps the
    peak workspace independent of the model shard size while preserving the exact
    Megatron normalization operation.
    """
    chunk_numel = _chunk_numel() if chunk_numel is None else chunk_numel
    if chunk_numel <= 0:
        raise ValueError("chunk_numel must be positive")

    for start in range(0, grad_data.numel(), chunk_numel):
        grad_data[start : start + chunk_numel].mul_(scaling_factor)


def local_squared_norm_in_chunks(grads, chunk_numel: int | None = None):
    """Accumulate an FP32 squared norm without the NPU multi-tensor BF16 kernel."""
    chunk_numel = (
        int(os.getenv("SLIME_NPU_GRAD_NORM_CHUNK_NUMEL", _DEFAULT_NORM_CHUNK_NUMEL))
        if chunk_numel is None
        else chunk_numel
    )
    if chunk_numel <= 0:
        raise ValueError("chunk_numel must be positive")

    grads = list(grads)
    if not grads:
        import torch

        return torch.zeros(1, dtype=torch.float32, device="cuda")

    import torch

    total = torch.zeros(1, dtype=torch.float32, device=grads[0].device)
    for grad in grads:
        flat_grad = grad.reshape(-1)
        for start in range(0, flat_grad.numel(), chunk_numel):
            chunk = flat_grad[start : start + chunk_numel].to(dtype=torch.float32)
            total.add_(torch.sum(chunk.square()))
    return total


def npu_get_grad_norm_fp32(grads_for_norm, norm_type=2, grad_stats_parallel_group=None) -> float:
    """NPU-safe replacement for Megatron's BF16 multi-tensor L2 norm."""
    import torch
    from megatron.core.utils import get_data_parallel_group_if_dtensor, to_local_if_dtensor

    if isinstance(grads_for_norm, torch.Tensor):
        grads_for_norm = [grads_for_norm]
    else:
        grads_for_norm = list(grads_for_norm)

    if float(norm_type) != 2.0:
        raise NotImplementedError("The NPU chunked gradient norm currently supports only L2 norm")

    data_parallel_group = None
    for grad in grads_for_norm:
        data_parallel_group = get_data_parallel_group_if_dtensor(grad, data_parallel_group)
    grads_for_norm = [to_local_if_dtensor(grad) for grad in grads_for_norm]

    total = local_squared_norm_in_chunks(grads_for_norm)
    local_total = total.item()
    if not math.isfinite(local_total):
        for grad_index, grad in enumerate(grads_for_norm):
            finite = torch.isfinite(grad)
            if not finite.all().item():
                logger.error(
                    "Non-finite NPU gradient before norm reduction: index=%s shape=%s dtype=%s "
                    "nan=%s inf=%s",
                    grad_index,
                    tuple(grad.shape),
                    grad.dtype,
                    torch.isnan(grad).sum().item(),
                    torch.isinf(grad).sum().item(),
                )
                break
        else:
            logger.error("NPU gradient squared-norm accumulation is non-finite: %s", local_total)
    if data_parallel_group:
        torch.distributed.all_reduce(total, op=torch.distributed.ReduceOp.SUM, group=data_parallel_group)
    torch.distributed.all_reduce(total, op=torch.distributed.ReduceOp.SUM, group=grad_stats_parallel_group)
    reduced_total = total.item()
    if not math.isfinite(reduced_total) and math.isfinite(local_total):
        logger.error("NPU gradient squared norm became non-finite during parallel reduction")
    return reduced_total**0.5


def install_npu_chunked_grad_scaling() -> bool:
    """Install the bounded-workspace Megatron gradient scaler on NPU actors."""
    from slime.utils import accelerator

    if accelerator.device_type() != "npu":
        return False

    from megatron.core.distributed.param_and_grad_buffer import _ParamAndGradBuffer

    installed = False
    if not getattr(_ParamAndGradBuffer, _PATCH_MARKER, False):

        def scale_gradients(self, scaling_factor) -> None:
            scale_gradients_in_chunks(self.grad_data, scaling_factor)

        _ParamAndGradBuffer.scale_gradients = scale_gradients
        setattr(_ParamAndGradBuffer, _PATCH_MARKER, True)
        installed = True

    import megatron.core.optimizer.clip_grads as clip_grads
    import megatron.core.optimizer.optimizer as optimizer

    if not getattr(clip_grads, _NORM_PATCH_MARKER, False):
        clip_grads.get_grad_norm_fp32 = npu_get_grad_norm_fp32
        optimizer.get_grad_norm_fp32 = npu_get_grad_norm_fp32
        setattr(clip_grads, _NORM_PATCH_MARKER, True)
        installed = True

    if installed:
        logger.info("Enabled chunked NPU scaling and L2 norm for Megatron gradient buffers")
    return installed
