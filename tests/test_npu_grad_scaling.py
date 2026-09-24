import pytest
import torch

from slime.backends.megatron_utils.npu_grad_scaling import (
    local_squared_norm_in_chunks,
    scale_gradients_in_chunks,
)


@pytest.mark.unit
def test_scale_gradients_in_chunks_matches_full_tensor_scaling():
    actual = torch.arange(23, dtype=torch.float32)
    expected = actual.clone().mul_(0.125)

    scale_gradients_in_chunks(actual, 0.125, chunk_numel=5)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.unit
def test_scale_gradients_in_chunks_rejects_invalid_chunk_size():
    with pytest.raises(ValueError, match="positive"):
        scale_gradients_in_chunks(torch.ones(2), 0.5, chunk_numel=0)


@pytest.mark.unit
def test_local_squared_norm_in_chunks_uses_fp32_accumulation():
    grads = [torch.tensor([3, 4], dtype=torch.bfloat16), torch.tensor([12], dtype=torch.bfloat16)]

    squared_norm = local_squared_norm_in_chunks(grads, chunk_numel=1)

    torch.testing.assert_close(squared_norm, torch.tensor([169.0]), rtol=0, atol=0)
