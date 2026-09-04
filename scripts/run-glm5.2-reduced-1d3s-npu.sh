#!/usr/bin/env bash

# Single-node GLM-5.2 reduced-model RL smoke run on physical NPUs 8..15.
# The reduced HF directory contains metadata plus symlinks to the immutable
# full checkpoint. Set DRY_RUN=1 to validate and print the launch command.

set -euo pipefail
ulimit -n 65535

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." >/dev/null 2>&1 && pwd)"
WORK_ROOT="${WORK_ROOT:-/mnt/share/w00934247/project/glm_032}"
MEGATRON_ROOT="${WORK_ROOT}/Megatron-LM"
SGLANG_PYTHON_ROOT="${WORK_ROOT}/sglang/python"
MODEL_PATH="${GLM52_REDUCED_MODEL:-${WORK_ROOT}/GLM-5.2-reduced-1d3s}"
PROMPT_DATA="${GLM52_PROMPT_DATA:-${SCRIPT_DIR}/data/glm52-reduced-smoke.jsonl}"
SAVE_PATH="${GLM52_SAVE_PATH:-${PROJECT_ROOT}/outputs/glm52-reduced-1d3s}"
NPU_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-8,9,10,11,12,13,14,15}"

if [[ ! -f "${MODEL_PATH}/config.json" || ! -f "${MODEL_PATH}/model.safetensors.index.json" ]]; then
    echo "Reduced model is missing or incomplete: ${MODEL_PATH}" >&2
    exit 1
fi
if [[ ! -f "${PROMPT_DATA}" ]]; then
    echo "Prompt dataset does not exist: ${PROMPT_DATA}" >&2
    exit 1
fi

source "${SCRIPT_DIR}/models/glm5.2-reduced-1d3s.sh"
set +u
eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV:-/mnt/share/w00934247/conda_envs/glm52_env}"

# Conda activation can replace loader variables, so load CANN afterwards.
# Prefer the A5 installation used by the known-good smoke launcher, while
# retaining the 179 installation as a fallback.
if [[ -z "${CANN_ROOT:-}" ]]; then
    for candidate in \
        /mnt/share/w00934247/cann/cann910/ascend-toolkit \
        /mnt/share/w00934247/cann_envs/cann910/cann-9.1.0; do
        if [[ -f "${candidate}/set_env.sh" ]]; then
            CANN_ROOT="${candidate}"
            break
        fi
    done
fi
if [[ -z "${CANN_ROOT:-}" || ! -f "${CANN_ROOT}/set_env.sh" ]]; then
    echo "Cannot find CANN set_env.sh; set CANN_ROOT explicitly." >&2
    exit 1
fi

CANN_BASE="$(dirname -- "${CANN_ROOT}")"
ATB_ENV="${ATB_ENV:-${CANN_BASE}/nnal/atb/set_env.sh}"
ASDSIP_ENV="${ASDSIP_ENV:-${CANN_BASE}/nnal/asdsip/set_env.sh}"
source "${CANN_ROOT}/set_env.sh"
if [[ -f "${ATB_ENV}" ]]; then
    source "${ATB_ENV}"
fi
if [[ -f "${ASDSIP_ENV}" ]]; then
    source "${ASDSIP_ENV}"
fi
set -u

if [[ -z "${LD_LIBRARY_PATH:-}" ]]; then
    echo "LD_LIBRARY_PATH is empty after loading CANN." >&2
    exit 1
fi

export ASCEND_RT_VISIBLE_DEVICES="${NPU_DEVICES}"
# A5 requires standard-format weights; ACL-format weights fail at runtime.
export SGLANG_NPU_DISABLE_ACL_FORMAT_WEIGHT=1
export PYTHONUNBUFFERED=1
export PYTHONPATH="${PROJECT_ROOT}:${MEGATRON_ROOT}:${SGLANG_PYTHON_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

# A5 must use its routable host interface for HCCL root-info detection.  Keep
# loopback as a fallback for machines (such as the original 179 setup) that do
# not expose the A5 interface.
SOCKET_IFNAME="${SOCKET_IFNAME:-}"
if [[ -z "${SOCKET_IFNAME}" ]]; then
    if ip -4 -o addr show dev enp35s0f2 scope global 2>/dev/null | grep -q .; then
        SOCKET_IFNAME=enp35s0f2
    else
        SOCKET_IFNAME=lo
    fi
fi
if [[ "${SOCKET_IFNAME}" == lo ]]; then
    CURRENT_IP=127.0.0.1
else
    CURRENT_IP="$(
        ip -4 -o addr show dev "${SOCKET_IFNAME}" scope global 2>/dev/null |
            awk 'NR == 1 {split($4, ip, "/"); print ip[1]}'
    )"
fi
if [[ -z "${CURRENT_IP}" ]]; then
    echo "Cannot obtain an IPv4 address from ${SOCKET_IFNAME}." >&2
    exit 1
fi

export MASTER_ADDR="${MASTER_ADDR:-${CURRENT_IP}}"
export MASTER_PORT="${MASTER_PORT:-29500}"
RAY_PORT="${RAY_PORT:-6382}"
RAY_DASHBOARD_PORT="${RAY_DASHBOARD_PORT:-8267}"
RAY_TMP_DIR="${RAY_TMP_DIR:-/tmp/ray-glm52-reduced-1d3s}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-${SOCKET_IFNAME}}"
export HCCL_SOCKET_IFNAME="${HCCL_SOCKET_IFNAME:-${SOCKET_IFNAME}}"
export TP_SOCKET_IFNAME="${TP_SOCKET_IFNAME:-${SOCKET_IFNAME}}"
export HCCL_BUFFSIZE="${HCCL_BUFFSIZE:-1024}"
export HCCL_HOST_SOCKET_PORT_RANGE="${HCCL_HOST_SOCKET_PORT_RANGE:-50000-52999}"
export HCCL_NPU_SOCKET_PORT_RANGE="${HCCL_NPU_SOCKET_PORT_RANGE:-60000-62999}"
export HCCL_CONNECT_TIMEOUT="${HCCL_CONNECT_TIMEOUT:-7200}"
export HCCL_EXEC_TIMEOUT="${HCCL_EXEC_TIMEOUT:-7200}"
export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1
export RAY_DEDUP_LOGS=0
export INDEXER_ROPE_NEOX_STYLE=0
# SGLang must put target/draft weights and KV caches in its NPU memory-saver
# pool so /release_memory_occupation can make room for the colocated trainer.
export SLIME_NPU_ENABLE_SGLANG_MEMORY_SAVER=1
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY

CKPT_ARGS=(
    --hf-checkpoint "${MODEL_PATH}"
    --load "${MODEL_PATH}"
    # --save "${SAVE_PATH}"
    # --save-interval 20
)

ROLLOUT_ARGS=(
    --prompt-data "${PROMPT_DATA}"
    --input-key prompt
    --label-key label
    --apply-chat-template
    --rollout-shuffle
    --rm-type deepscaler
    --num-rollout 1
    --rollout-batch-size 4
    --n-samples-per-prompt 2
    --rollout-max-response-len 128
    --rollout-temperature 1.0
    --global-batch-size 8
)

PERF_ARGS=(
    --tensor-model-parallel-size 1
    --pipeline-model-parallel-size 1
    --context-parallel-size 1
    --expert-model-parallel-size 8
    --expert-tensor-parallel-size 1
    --grad-reduce-in-bf16
    --main-grads-dtype bf16
    --recompute-granularity full
    --recompute-method uniform
    --recompute-num-layers 1
    --use-dynamic-batch-size
    --max-tokens-per-gpu 1024
    --data-pad-size-multiplier 128
    --log-probs-chunk-size 1024
)

GRPO_ARGS=(
    --advantage-estimator grpo
    --kl-loss-coef 0.00
    --kl-loss-type low_var_kl
    --kl-coef 0.00
    --entropy-coef 0.00
    --eps-clip 0.2
    --eps-clip-high 0.28
    --use-tis
    --tis-clip-low 0.5
    --tis-clip 2.0
)

OPTIMIZER_ARGS=(
    --optimizer adam
    --lr 1e-6
    --lr-decay-style constant
    --weight-decay 0.1
    --adam-beta1 0.9
    --adam-beta2 0.98
    --optimizer-cpu-offload
    --overlap-cpu-optimizer-d2h-h2d
    --use-precision-aware-optimizer
)

SGLANG_ARGS=(
    --rollout-num-gpus 8
    --rollout-num-gpus-per-engine 8
    --sglang-device npu
    # Target + EAGLE draft weights need at least 52.5% with this SGLang build.
    --sglang-mem-fraction-static 0.55
    # The model advertises a 1M context window, which otherwise makes SGLang
    # reserve ~2.15M KV-cache tokens per DP rank.  This smoke workload needs
    # only short sequences; cap the pool so it can be restored while the
    # colocated Megatron gradient buffer remains resident.
    --sglang-max-total-tokens 32768
    --sglang-enable-dp-attention
    --sglang-dp-size 8
    --sglang-ep-size 8
    --sglang-enable-dp-lm-head
    --sglang-moe-dense-tp-size 1
    --sglang-moe-a2a-backend deepep
    --sglang-disable-cuda-graph
    --sglang-max-running-requests 16
    --sglang-watchdog-timeout 1200
    --sglang-router-request-timeout-secs 1200
    --sglang-speculative-algorithm EAGLE
    --sglang-speculative-num-steps 3
    --sglang-speculative-eagle-topk 1
    --sglang-speculative-num-draft-tokens 4
    --sglang-speculative-moe-a2a-backend deepep
)

MISC_ARGS=(
    --attention-dropout 0.0
    --hidden-dropout 0.0
    # Keep the resident gradient buffer in BF16.  FP32 doubles it to ~26 GiB
    # per rank and leaves too little room to wake the colocated SGLang engine.
    --attention-softmax-in-fp32
    --attention-backend flash
    --actor-num-nodes 1
    --actor-num-gpus-per-node 8
    --colocate
    # Slime keeps the BF16 gradient buffer resident while the parameter buffer
    # is pauseable, avoiding the NPU allocator's large contiguous-grad failure.
    --offload-train
    --moe-token-dispatcher-type alltoall
    --no-check-for-nan-in-loss-and-grad
    --update-weight-buffer-size 2147483648
)

# Ray actors have their own runtime environment.  Pass the CANN loader and
# host-communication settings explicitly so a reused daemon or nested Ray job
# cannot silently fall back to lo or an empty LD_LIBRARY_PATH.
TRAIN_ENV_VARS=$(python3 - <<'PY_TRAIN_ENV'
import json
import os

env = {
    key: os.environ[key]
    for key in (
        "PATH",
        "PYTHONPATH",
        "LD_LIBRARY_PATH",
        "ASCEND_HOME_PATH",
        "ASCEND_OPP_PATH",
        "HCCL_SOCKET_IFNAME",
        "GLOO_SOCKET_IFNAME",
        "TP_SOCKET_IFNAME",
    )
    if key in os.environ
}
env.update(
    {
        "HCCL_HOST_SOCKET_PORT_RANGE": os.environ.get(
            "TRAIN_HCCL_HOST_SOCKET_PORT_RANGE", "53000-53999"
        ),
        "HCCL_NPU_SOCKET_PORT_RANGE": os.environ.get(
            "TRAIN_HCCL_NPU_SOCKET_PORT_RANGE", "63000-63999"
        ),
        "HCCL_CONNECT_TIMEOUT": os.environ["HCCL_CONNECT_TIMEOUT"],
        "HCCL_EXEC_TIMEOUT": os.environ["HCCL_EXEC_TIMEOUT"],
    }
)
print(json.dumps(env))
PY_TRAIN_ENV
)
MISC_ARGS+=(--train-env-vars "${TRAIN_ENV_VARS}")

TRAIN_ARGS=(
    "${MODEL_ARGS[@]}"
    "${CKPT_ARGS[@]}"
    "${ROLLOUT_ARGS[@]}"
    "${OPTIMIZER_ARGS[@]}"
    "${GRPO_ARGS[@]}"
    "${PERF_ARGS[@]}"
    "${SGLANG_ARGS[@]}"
    "${MISC_ARGS[@]}"
)

if [[ "${PARSE_ONLY:-0}" == 1 ]]; then
    cd "${PROJECT_ROOT}"
    python -c 'import sys; from slime.utils.arguments import parse_args; args = parse_args(); print("validated", args.num_layers, args.expert_model_parallel_size, args.sglang_device)' "${TRAIN_ARGS[@]}"
    exit 0
fi

if [[ "${DRY_RUN:-0}" == 1 ]]; then
    printf 'python train.py'
    printf ' %q' "${TRAIN_ARGS[@]}"
    printf '\n'
    exit 0
fi

if [[ "${ALLOW_BUSY_NPUS:-0}" != 1 ]] && \
    npu-smi info | sed -n '/Process id/,$p' | grep -E '^\| [4-7][[:space:]]' >/dev/null; then
    echo "Physical NPUs 8..15 are busy; refusing to disrupt another workload." >&2
    echo "Wait for the devices to become idle, or set ALLOW_BUSY_NPUS=1 explicitly." >&2
    exit 2
fi

mkdir -p "${SAVE_PATH}"
cd "${PROJECT_ROOT}"
if [[ "${RESET_RAY:-0}" == 1 ]]; then
    ray stop --force >/dev/null 2>&1 || true
fi
if [[ "${REUSE_RAY:-0}" != 1 ]]; then
    ray start --head --port "${RAY_PORT}" --node-ip-address "${MASTER_ADDR}" \
        --resources='{"NPU": 8}' \
        --num-gpus 8 \
        --num-cpus 32 \
        --temp-dir="${RAY_TMP_DIR}" --min-worker-port=30000 --max-worker-port=30999 \
        --disable-usage-stats --dashboard-host=127.0.0.1 --dashboard-port="${RAY_DASHBOARD_PORT}"
fi

echo "HCCL bootstrap: interface=${HCCL_SOCKET_IFNAME}, master=${MASTER_ADDR}"

# Direct launch matches the A5 configuration known to work and preserves the
# fully initialized CANN environment.  Ray Job mode remains available for
# callers that explicitly need it.
if [[ "${USE_RAY_JOB:-0}" == 1 ]]; then
    for _ in $(seq 1 30); do
        if curl --silent --fail "http://127.0.0.1:${RAY_DASHBOARD_PORT}/api/jobs/" >/dev/null; then
            break
        fi
        sleep 2
    done
    curl --silent --fail "http://127.0.0.1:${RAY_DASHBOARD_PORT}/api/jobs/" >/dev/null || {
        echo "Ray dashboard job agent did not become ready on port ${RAY_DASHBOARD_PORT}." >&2
        exit 1
    }

    RUNTIME_ENV_JSON=$(python3 - <<'PY_RUNTIME_ENV'
import json
import os

names = {
    "ASCEND_RT_VISIBLE_DEVICES",
    "SGLANG_NPU_DISABLE_ACL_FORMAT_WEIGHT",
    "PYTHONPATH",
    "PYTHONUNBUFFERED",
    "PATH",
    "LD_LIBRARY_PATH",
    "MASTER_ADDR",
    "MASTER_PORT",
    "GLOO_SOCKET_IFNAME",
    "HCCL_SOCKET_IFNAME",
    "TP_SOCKET_IFNAME",
    "HCCL_BUFFSIZE",
    "HCCL_HOST_SOCKET_PORT_RANGE",
    "HCCL_NPU_SOCKET_PORT_RANGE",
    "HCCL_CONNECT_TIMEOUT",
    "HCCL_EXEC_TIMEOUT",
    "INDEXER_ROPE_NEOX_STYLE",
    "SLIME_NPU_ENABLE_SGLANG_MEMORY_SAVER",
}
names.update(
    key
    for key in os.environ
    if key.startswith(("ASCEND_", "ATB_", "TBE_"))
)
print(json.dumps({"env_vars": {key: os.environ[key] for key in names if key in os.environ}}))
PY_RUNTIME_ENV
    )

    ray job submit --address="http://127.0.0.1:${RAY_DASHBOARD_PORT}" \
        --runtime-env-json="${RUNTIME_ENV_JSON}" \
        -- python train.py "${TRAIN_ARGS[@]}"
else
    python3 train.py "${TRAIN_ARGS[@]}"
fi
