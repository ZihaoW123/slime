"""Ascend NPU accelerator implementation."""

from __future__ import annotations

import importlib
from typing import Any

import torch

from .torch_accelerator import TorchAccelerator


def npu_module() -> Any:
    module = getattr(torch, "npu", None)
    if module is not None:
        return module
    try:
        importlib.import_module("torch_npu")
    except ModuleNotFoundError as exc:
        if exc.name == "torch_npu":
            return None
        raise
    return getattr(torch, "npu", None)


def is_npu_available() -> bool:
    module = npu_module()
    checker = getattr(module, "is_available", None)
    return bool(module is not None and checker is not None and checker())


class NPUAccelerator(TorchAccelerator):
    name = "npu"
    device_type = "npu"
    communication_backend_name = "hccl"
    ray_resource_name = "NPU"

    @property
    def visible_devices_env(self) -> str:
        return "ASCEND_RT_VISIBLE_DEVICES"

    def _module(self) -> Any:
        module = npu_module()
        if module is None:
            raise RuntimeError("Ascend NPU backend requires torch_npu and torch.npu")
        return module

    def is_available(self) -> bool:
        return is_npu_available()

    def post_import_torch(self) -> None:
        """Load Ascend's Megatron compatibility shim after torch_npu setup."""
        try:
            importlib.import_module("megatron_adaptor")
        except ModuleNotFoundError as exc:
            if exc.name != "megatron_adaptor":
                raise
            # The shim is supplied by the Ascend Megatron stack. Keep Slime
            # importable when that optional dependency is not installed.
            return

    def distributed_device_id(self, index: int | str | torch.device | None = None) -> None:
        # HCCL selects the local device through torch.npu.set_device/LOCAL_RANK.
        return None

    def empty_cache(self) -> None:
        """Avoid allocator cache destruction while an NPU graph is captured.

        Ascend raises ``captures_underway.empty()`` when ``empty_cache`` is
        called during graph capture. Weight-update paths still synchronize and
        perform IPC cleanup; cache destruction is only an optimization.
        """
        return None

    def supports(self, capability: str) -> bool:
        if capability in {"nvml_affinity", "cuda_int4_extension", "sglang_fp8_utils", "triton_kernels"}:
            return False
        if capability == "requires_cpu_initialization":
            return True
        if capability == "bf16":
            checker = getattr(self._module(), "is_bf16_supported", None)
            return bool(checker and checker())
        return super().supports(capability)
