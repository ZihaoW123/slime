import sys
import threading
import types
from contextlib import contextmanager

import pytest

from slime.backends.megatron_utils.tms_utils import (
    allow_tms_initial_region_subregions,
    empty_cache_unless_npu_tms_pool_active,
    npu_tms_temporary_allocation_pool,
    npu_tms_temporary_allocation_pool_active,
)
from slime.utils.memory_utils import clear_memory


class _FakeCdll:
    def __init__(self, events):
        self.events = events
        self.interesting_region = True

    def tms_get_interesting_region(self):
        self.events.append(("get", self.interesting_region))
        return self.interesting_region


class _FakeBinaryWrapper:
    def __init__(self, events):
        self.events = events
        self.cdll = _FakeCdll(events)

    def set_config(self, *, tag, interesting_region, enable_cpu_backup):
        self.events.append(("set", tag, interesting_region, enable_cpu_backup))
        self.cdll.interesting_region = interesting_region


class _FakeTorchMemorySaver:
    def __init__(self, events):
        self.events = events
        self._impl = types.SimpleNamespace(_binary_wrapper=_FakeBinaryWrapper(events))

    def _ensure_initialized(self):
        self.events.append(("init",))

    @contextmanager
    def region(self, *, tag, enable_cpu_backup):
        assert not self._impl._binary_wrapper.cdll.interesting_region
        self.events.append(("enter", tag, enable_cpu_backup))
        self._impl._binary_wrapper.cdll.interesting_region = True
        try:
            yield
        finally:
            self.events.append(("exit", tag))
            self._impl._binary_wrapper.cdll.interesting_region = False


@pytest.mark.unit
def test_initial_tms_region_allows_no_backup_subregion_and_restores(monkeypatch):
    events = []
    saver = _FakeTorchMemorySaver(events)
    module = types.ModuleType("torch_memory_saver")
    module.torch_memory_saver = saver
    monkeypatch.setitem(sys.modules, "torch_memory_saver", module)
    monkeypatch.setenv("TMS_INIT_ENABLE", "1")
    monkeypatch.setenv("TMS_INIT_ENABLE_CPU_BACKUP", "1")

    with allow_tms_initial_region_subregions(enabled=True):
        with saver.region(tag="param_buffer", enable_cpu_backup=False):
            events.append(("allocate",))

    assert saver._impl._binary_wrapper.cdll.interesting_region is True
    assert "region" not in vars(saver)
    assert events == [
        ("init",),
        ("get", True),
        ("set", "default", False, False),
        ("enter", "param_buffer", False),
        ("allocate",),
        ("exit", "param_buffer"),
        ("set", "default", True, True),
    ]


@pytest.mark.unit
def test_initial_tms_region_restores_after_subregion_failure(monkeypatch):
    events = []
    saver = _FakeTorchMemorySaver(events)
    module = types.ModuleType("torch_memory_saver")
    module.torch_memory_saver = saver
    monkeypatch.setitem(sys.modules, "torch_memory_saver", module)
    monkeypatch.setenv("TMS_INIT_ENABLE", "true")
    monkeypatch.setenv("TMS_INIT_ENABLE_CPU_BACKUP", "true")

    with pytest.raises(RuntimeError, match="allocation failed"):
        with allow_tms_initial_region_subregions(enabled=True):
            with saver.region(tag="grad_buffer", enable_cpu_backup=False):
                raise RuntimeError("allocation failed")

    assert saver._impl._binary_wrapper.cdll.interesting_region is True
    assert "region" not in vars(saver)
    assert events[-2:] == [
        ("exit", "grad_buffer"),
        ("set", "default", True, True),
    ]


@pytest.mark.unit
def test_npu_tms_disabled_pool_skips_empty_cache(monkeypatch):
    events = []
    saver = _FakeTorchMemorySaver(events)
    saver._impl._binary_wrapper.cdll.interesting_region = False
    module = types.ModuleType("torch_memory_saver")
    module.torch_memory_saver = saver
    monkeypatch.setitem(sys.modules, "torch_memory_saver", module)
    monkeypatch.setenv("TMS_INIT_ENABLE", "1")

    from slime.utils import accelerator

    monkeypatch.setattr(accelerator, "device_type", lambda: "npu")
    monkeypatch.setattr(accelerator, "empty_cache", lambda: events.append(("empty_cache",)))

    assert empty_cache_unless_npu_tms_pool_active() is False
    assert ("empty_cache",) not in events


@pytest.mark.unit
def test_empty_cache_runs_outside_npu_tms_disabled_pool(monkeypatch):
    events = []
    saver = _FakeTorchMemorySaver(events)
    module = types.ModuleType("torch_memory_saver")
    module.torch_memory_saver = saver
    monkeypatch.setitem(sys.modules, "torch_memory_saver", module)
    monkeypatch.setenv("TMS_INIT_ENABLE", "1")

    from slime.utils import accelerator

    monkeypatch.setattr(accelerator, "device_type", lambda: "npu")
    monkeypatch.setattr(accelerator, "empty_cache", lambda: events.append(("empty_cache",)))

    assert empty_cache_unless_npu_tms_pool_active() is True
    assert events[-2:] == [("get", True), ("empty_cache",)]


@pytest.mark.unit
def test_npu_train_temporary_allocations_use_disabled_pool(monkeypatch):
    events = []

    class Saver:
        @contextmanager
        def disable(self):
            events.append(("disable",))
            try:
                yield
            finally:
                events.append(("restore",))

    module = types.ModuleType("torch_memory_saver")
    module.torch_memory_saver = Saver()
    monkeypatch.setitem(sys.modules, "torch_memory_saver", module)
    monkeypatch.setenv("TMS_INIT_ENABLE", "1")

    from slime.utils import accelerator

    monkeypatch.setattr(accelerator, "device_type", lambda: "npu")
    monkeypatch.setattr(accelerator, "synchronize", lambda: events.append(("sync",)))
    monkeypatch.setattr(accelerator, "empty_cache", lambda: events.append(("empty_cache",)))

    with npu_tms_temporary_allocation_pool(enabled=True):
        assert npu_tms_temporary_allocation_pool_active() is True
        events.append(("train",))
        seen_from_autograd_thread = []
        thread = threading.Thread(
            target=lambda: seen_from_autograd_thread.append(npu_tms_temporary_allocation_pool_active())
        )
        thread.start()
        thread.join()
        assert seen_from_autograd_thread == [True]
        clear_memory()

    assert npu_tms_temporary_allocation_pool_active() is False
    assert events == [("disable",), ("train",), ("sync",), ("restore",)]


@pytest.mark.unit
def test_clear_memory_skips_npu_cache_while_tms_pool_is_active(monkeypatch):
    events = []
    saver = _FakeTorchMemorySaver(events)
    saver._impl._binary_wrapper.cdll.interesting_region = False
    module = types.ModuleType("torch_memory_saver")
    module.torch_memory_saver = saver
    monkeypatch.setitem(sys.modules, "torch_memory_saver", module)
    monkeypatch.setenv("TMS_INIT_ENABLE", "1")

    from slime.utils import accelerator

    monkeypatch.setattr(accelerator, "device_type", lambda: "npu")
    monkeypatch.setattr(accelerator, "synchronize", lambda: events.append(("sync",)))
    monkeypatch.setattr(accelerator, "empty_cache", lambda: events.append(("empty_cache",)))

    clear_memory()

    assert ("sync",) in events
    assert ("empty_cache",) not in events
