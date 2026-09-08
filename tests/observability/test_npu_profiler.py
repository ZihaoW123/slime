from types import SimpleNamespace

import pytest

from slime.observability.npu_profiler import NPUProfilerConfig, NPUProfilerContents
from slime.observability.profile_utils import get_profiled_rollout_step, is_step_in_profile_window


@pytest.mark.unit
def test_npu_profiler_contents_enable_cpu_dependent_options():
    config = NPUProfilerConfig(contents=["npu", "memory", "shapes", "module", "stack"])

    assert config.uses_content("npu")
    assert config.with_memory
    assert config.record_shapes
    assert config.with_modules
    assert config.with_stack


@pytest.mark.unit
def test_npu_profiler_contents_reject_unknown_value():
    with pytest.raises(ValueError, match="unsupported"):
        NPUProfilerContents.from_contents(["npu", "unsupported"])


@pytest.mark.unit
@pytest.mark.parametrize(
    ("ranks", "current_rank", "expected"),
    [([-1], 7, True), ([0, 3], 3, True), ([0, 3], 2, False)],
)
def test_npu_profiler_rank_selection(ranks, current_rank, expected):
    config = NPUProfilerConfig(enabled=True, ranks=ranks, current_rank=current_rank)

    assert config.is_profiling_rank() is expected


@pytest.mark.unit
@pytest.mark.parametrize(
    ("step", "start", "end", "expected"),
    [(1, 1, 3, True), (2, 1, 3, True), (3, 1, 3, False), (10, 1, -1, True)],
)
def test_profile_step_window_is_start_inclusive_and_end_exclusive(step, start, end, expected):
    assert is_step_in_profile_window(step, start, end) is expected


@pytest.mark.unit
def test_profiled_rollout_step_uses_train_step_mapping():
    args = SimpleNamespace(
        wandb_always_use_train_step=True,
        rollout_batch_size=8,
        n_samples_per_prompt=2,
        global_batch_size=4,
    )

    assert get_profiled_rollout_step(args, rollout_id=2, step_start=8, step_end=12) == 8
    assert get_profiled_rollout_step(args, rollout_id=3, step_start=8, step_end=12) is None
