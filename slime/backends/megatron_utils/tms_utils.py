import os
from contextlib import contextmanager


_TRUE_VALUES = {"1", "true", "yes", "on"}
_NPU_TMS_TEMPORARY_POOL_DEPTH = 0


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.lower() in _TRUE_VALUES


def empty_cache_unless_npu_tms_pool_active() -> bool:
    """Empty the device cache unless NPU TMS is recording a temporary pool.

    ``torch_memory_saver.disable()`` enters ``torch.npu.use_mem_pool``. Calling
    ``torch.npu.empty_cache`` before that context exits trips torch_npu's
    ``captures_underway.empty()`` assertion. The pool destructor releases its
    own cached blocks on supported torch_npu versions, so skipping cache
    eviction in this narrow state is both required and sufficient.

    Returns whether cache eviction was performed.
    """
    from slime.utils import accelerator

    if accelerator.device_type() == "npu" and _env_flag("TMS_INIT_ENABLE"):
        try:
            from torch_memory_saver import torch_memory_saver
        except ImportError:
            pass
        else:
            impl = torch_memory_saver._impl
            if impl is not None and not impl._binary_wrapper.cdll.tms_get_interesting_region():
                return False

    accelerator.empty_cache()
    return True


@contextmanager
def npu_tms_temporary_allocation_pool(enabled: bool):
    """Keep ephemeral train allocations out of the persistent TMS region.

    Preload mode must remain active while Megatron constructs model, parameter,
    and gradient buffers so they can be paused for colocated rollout. During a
    train step, however, autograd and HCCL request temporary caching-allocator
    segments. Tracking those segments as persistent NPU virtual-memory regions
    can turn a small request into a multi-gigabyte physical allocation.

    ``torch_memory_saver.disable()`` provides an isolated NPU memory pool and
    disposes it after the step. Existing tracked model allocations are not
    affected and remain available to ``pause()``/``resume()``.
    """

    if not enabled or not _env_flag("TMS_INIT_ENABLE"):
        yield
        return

    from slime.utils import accelerator

    if accelerator.device_type() != "npu":
        yield
        return

    from torch_memory_saver import torch_memory_saver

    global _NPU_TMS_TEMPORARY_POOL_DEPTH
    _NPU_TMS_TEMPORARY_POOL_DEPTH += 1
    try:
        with torch_memory_saver.disable():
            yield
    finally:
        _NPU_TMS_TEMPORARY_POOL_DEPTH -= 1


def npu_tms_temporary_allocation_pool_active() -> bool:
    """Return whether this actor process is inside the disposable NPU pool.

    This is process-scoped rather than a ``ContextVar`` because PyTorch's
    autograd engine can invoke Python backward callbacks from another execution
    context in the same actor process.
    """

    return _NPU_TMS_TEMPORARY_POOL_DEPTH > 0


@contextmanager
def allow_tms_initial_region_subregions(enabled: bool):
    """Allow explicit TMS regions while preload-mode initial tracking is active.

    Train actors start with ``TMS_INIT_ENABLE=1`` so model allocations are
    tracked before Python can enter a context manager. Megatron creates
    dedicated param/grad buffer regions without CPU backup. NPU TMS 0.0.8 does
    not support entering those regions while the preload initial region is
    active, so temporarily suspend the initial configuration for each explicit
    region and restore it afterwards.

    The wrapper is installed only during model/optimizer initialization. It is
    intentionally not a general nested-region implementation because TMS does
    not expose getters for an arbitrary outer region's tag or backup policy.
    """
    if not enabled or not _env_flag("TMS_INIT_ENABLE"):
        yield
        return

    from torch_memory_saver import torch_memory_saver

    torch_memory_saver._ensure_initialized()
    impl = torch_memory_saver._impl
    binary_wrapper = impl._binary_wrapper
    cdll = binary_wrapper.cdll
    initial_cpu_backup = _env_flag("TMS_INIT_ENABLE_CPU_BACKUP")
    original_region = torch_memory_saver.region
    instance_dict = vars(torch_memory_saver)
    had_region_override = "region" in instance_dict
    previous_region_override = instance_dict.get("region")

    @contextmanager
    def initial_region_compatible_region(*args, **kwargs):
        initial_region_active = bool(cdll.tms_get_interesting_region())
        if initial_region_active:
            binary_wrapper.set_config(
                tag="default",
                interesting_region=False,
                enable_cpu_backup=False,
            )
        try:
            with original_region(*args, **kwargs):
                yield
        finally:
            if initial_region_active:
                binary_wrapper.set_config(
                    tag="default",
                    interesting_region=True,
                    enable_cpu_backup=initial_cpu_backup,
                )

    torch_memory_saver.region = initial_region_compatible_region
    try:
        yield
    finally:
        if had_region_override:
            torch_memory_saver.region = previous_region_override
        else:
            del torch_memory_saver.region
