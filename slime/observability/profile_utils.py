import logging
import time
import traceback
from pathlib import Path

import torch

from slime.observability.metric_utils import compute_rollout_step
from slime.observability.npu_profiler import NPUProfilerConfig, NPUProfilerManager
from slime.utils import accelerator
from slime.utils.memory_utils import print_memory

logger = logging.getLogger(__name__)


def is_step_in_profile_window(step: int, step_start: int, step_end: int) -> bool:
    return step >= step_start and (step_end == -1 or step < step_end)


def get_profiled_rollout_step(args, rollout_id: int, step_start: int, step_end: int) -> int | None:
    step = compute_rollout_step(args, rollout_id)
    return step if is_step_in_profile_window(step, step_start, step_end) else None


class TrainProfiler:
    def __init__(self, args, role: str = "actor"):
        self.args = args
        self.role = role
        self._torch_profiler_overall = None
        self._memory_profiler_overall = None
        self._npu_profiler = None
        self._actor_npu_profile_active = False

        if args.use_pytorch_profiler:
            self._torch_profiler_overall = _create_torch_profiler(args, name="train_overall")

        if args.record_memory_history:
            self._memory_profiler_overall = _BaseMemoryProfiler.create(args)
            self._memory_profiler_overall.start()

        if role == "actor" and getattr(args, "npu_profile_actor", False):
            self._npu_profiler = NPUProfilerManager(_create_npu_profiler_config(args))

    def on_init_end(self):
        if self._torch_profiler_overall is not None:
            self._torch_profiler_overall.start()

    def _should_profile_actor_step(self, rollout_id: int) -> bool:
        if self._npu_profiler is None:
            return False
        return (
            get_profiled_rollout_step(
                self.args,
                rollout_id,
                self.args.npu_profile_actor_global_step_start,
                self.args.npu_profile_actor_global_step_end,
            )
            is not None
        )

    def start_actor_profile(self, rollout_id: int) -> None:
        if not self._should_profile_actor_step(rollout_id) or self._actor_npu_profile_active:
            return
        step = compute_rollout_step(self.args, rollout_id)
        logger.info("Starting actor-side NPU profiling at global step %s (rollout_id=%s).", step, rollout_id)
        self._npu_profiler.start()
        self._actor_npu_profile_active = True

    def step(self, rollout_id: int):
        if self._torch_profiler_overall is not None:
            self._torch_profiler_overall.step()

        if (
            self._memory_profiler_overall is not None
            and ((s := self.args.memory_snapshot_num_steps) is not None)
            and (rollout_id == s - 1)
        ):
            self._memory_profiler_overall.stop()

        if self._actor_npu_profile_active:
            self._npu_profiler.step()
            has_next_rollout = self.args.num_rollout is not None and rollout_id + 1 < self.args.num_rollout
            should_continue = has_next_rollout and self._should_profile_actor_step(rollout_id + 1)
            if not should_continue:
                step = compute_rollout_step(self.args, rollout_id)
                self._npu_profiler.stop()
                self._actor_npu_profile_active = False
                logger.info("Stopping actor-side NPU profiling at global step %s (rollout_id=%s).", step, rollout_id)


def _create_npu_profiler_config(args) -> NPUProfilerConfig:
    current_rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    global_step_start = args.npu_profile_actor_global_step_start
    global_step_end = args.npu_profile_actor_global_step_end
    local_step_end = -1 if global_step_end == -1 else global_step_end - global_step_start
    return NPUProfilerConfig(
        enabled=True,
        step_start=0,
        step_end=local_step_end,
        ranks=args.npu_profile_actor_ranks,
        level=args.npu_profile_actor_level,
        export_type=args.npu_profile_actor_export_type,
        contents=args.npu_profile_actor_contents,
        analysis=args.npu_profile_actor_analysis,
        save_path=args.npu_profile_actor_save_path,
        current_rank=current_rank,
    )


def _create_torch_profiler(args, name):
    activities = [torch.profiler.ProfilerActivity.CPU]
    activity_name = accelerator.device_type().upper()
    if hasattr(torch.profiler.ProfilerActivity, activity_name):
        activities.append(getattr(torch.profiler.ProfilerActivity, activity_name))

    return torch.profiler.profile(
        activities=activities,
        schedule=torch.profiler.schedule(
            wait=max(args.profile_step_start - 1, 0),
            warmup=1 if args.profile_step_start > 0 else 0,
            active=args.profile_step_end - args.profile_step_start,
            repeat=1,
        ),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(
            args.tensorboard_dir,
            worker_name=f"{name}_rank_{torch.distributed.get_rank()}",
            use_gzip=True,
        ),
        record_shapes=True,
        with_stack=True,
        profile_memory=True,
        with_flops=True,
    )


class _BaseMemoryProfiler:
    @staticmethod
    def create(args):
        c = {
            "torch": _TorchMemoryProfiler,
            "memray": _MemrayMemoryProfiler,
        }[args.memory_recorder]
        return c(args)

    def __init__(self, args):
        self._path_dump = (
            Path(args.memory_snapshot_dir)
            / f"memory_snapshot_time{time.time()}_rank{torch.distributed.get_rank()}_{args.memory_snapshot_path}"
        )

    def start(self):
        raise NotImplementedError

    def stop(self):
        raise NotImplementedError


class _TorchMemoryProfiler(_BaseMemoryProfiler):
    def __init__(self, args):
        super().__init__(args)
        self._recording = False

    @staticmethod
    def _memory_module():
        return accelerator.memory_module()

    def start(self):
        logger.info("Attach OOM dump memory history.")
        memory_module = self._memory_module()
        if memory_module is None or not hasattr(memory_module, "_record_memory_history"):
            logger.warning("Accelerator memory history is unavailable; skip torch memory profiler.")
            return
        if not hasattr(memory_module, "_dump_snapshot"):
            logger.warning("Accelerator memory snapshot is unavailable; skip torch memory profiler.")
            return

        memory_module._record_memory_history(
            max_entries=1000000,
            stacks="all",
        )
        self._recording = True

        def oom_observer(device, alloc, device_alloc, device_free):
            logger.info(
                f"Observe OOM, will dump snapshot to {self._path_dump}. ({device=} {alloc=} {device_alloc=} {device_free=}; stacktrace is as follows)"
            )
            traceback.print_stack()
            memory_module._dump_snapshot(str(self._path_dump))
            print_memory("when oom")

        attach_oom_observer = getattr(torch._C, "_cuda_attach_out_of_memory_observer", None)
        if attach_oom_observer is not None:
            attach_oom_observer(oom_observer)
        else:
            logger.warning("Accelerator OOM observer is unavailable; memory snapshot on OOM is disabled.")

    def stop(self):
        if not self._recording:
            return
        logger.info(f"Dump memory snapshot to: {self._path_dump}")
        memory_module = self._memory_module()
        if memory_module is None or not hasattr(memory_module, "_dump_snapshot"):
            logger.warning("Accelerator memory snapshot is unavailable; skip dump.")
            return
        memory_module._dump_snapshot(str(self._path_dump))
        memory_module._record_memory_history(enabled=None)
        self._recording = False


class _MemrayMemoryProfiler(_BaseMemoryProfiler):
    def __init__(self, args):
        super().__init__(args)
        assert args.memory_snapshot_num_steps is not None, "In memray, must provide --memory-snapshot-num-steps"

    def start(self):
        logger.info("Memray tracker started.")
        import memray

        self._tracker = memray.Tracker(
            file_name=self._path_dump,
            native_traces=True,
        )
        self._tracker.__enter__()

    def stop(self):
        logger.info(f"Memray tracker stopped and dump snapshot to: {self._path_dump}")
        self._tracker.__exit__(None, None, None)
