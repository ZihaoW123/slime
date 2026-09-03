#!/usr/bin/env bash

# Single-node GLM-5.2 reduced-model RL smoke run on physical NPUs 8..15.
# The reduced HF directory contains metadata plus symlinks to the immutable
# full checkpoint. Set DRY_RUN=1 to validate and print the launch command.

set -euo pipefail
ulimit -n 65535

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." >/dev/null 2>&1 && pwd)"
WORK_ROOT="/mnt/share/w00934247/project/glm_032"
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
source /mnt/share/w00934247/cann_envs/cann910/cann-9.1.0/set_env.sh
source /mnt/share/w00934247/cann_envs/cann910/nnal/atb/set_env.sh
source /mnt/share/w00934247/cann_envs/cann910/nnal/asdsip/set_env.sh
eval "$(conda shell.bash hook)"
conda activate /mnt/share/w00934247/conda_envs/glm52_env
set -u

export ASCEND_RT_VISIBLE_DEVICES="${NPU_DEVICES}"
export PYTHONUNBUFFERED=1
export PYTHONPATH="${PROJECT_ROOT}:${WORK_ROOT}/Megatron-LM:${WORK_ROOT}/MegatronAdaptor:${WORK_ROOT}/sglang/python${PYTHONPATH:+:${PYTHONPATH}}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-29500}"
RAY_PORT="${RAY_PORT:-6382}"
RAY_DASHBOARD_PORT="${RAY_DASHBOARD_PORT:-8267}"
RAY_TMP_DIR="${RAY_TMP_DIR:-/tmp/ray-glm52-reduced-1d3s}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-lo}"
export HCCL_SOCKET_IFNAME="${HCCL_SOCKET_IFNAME:-lo}"
export HCCL_BUFFSIZE="${HCCL_BUFFSIZE:-1024}"
export HCCL_HOST_SOCKET_PORT_RANGE="${HCCL_HOST_SOCKET_PORT_RANGE:-auto}"
export HCCL_NPU_SOCKET_PORT_RANGE="${HCCL_NPU_SOCKET_PORT_RANGE:-auto}"
export INDEXER_ROPE_NEOX_STYLE=0
# SGLang must put target/draft weights and KV caches in its NPU memory-saver
# pool so /release_memory_occupation can make room for the colocated trainer.
export SLIME_NPU_ENABLE_SGLANG_MEMORY_SAVER=1
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY

CKPT_ARGS=(
    --hf-checkpoint "${MODEL_PATH}"
    --load "${MODEL_PATH}"
    --save "${SAVE_PATH}"
    --save-interval 20
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
        --num-cpus 32 \
        --temp-dir="${RAY_TMP_DIR}" --min-worker-port=30000 --max-worker-port=30999 \
        --disable-usage-stats --dashboard-host=127.0.0.1 --dashboard-port="${RAY_DASHBOARD_PORT}"
fi

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

RUNTIME_ENV_JSON=$(python -c 'import json, os; print(json.dumps({"env_vars": {key: os.environ[key] for key in ["ASCEND_RT_VISIBLE_DEVICES", "PYTHONPATH", "PYTHONUNBUFFERED", "MASTER_ADDR", "MASTER_PORT", "GLOO_SOCKET_IFNAME", "HCCL_SOCKET_IFNAME", "HCCL_BUFFSIZE", "HCCL_HOST_SOCKET_PORT_RANGE", "HCCL_NPU_SOCKET_PORT_RANGE", "INDEXER_ROPE_NEOX_STYLE", "SLIME_NPU_ENABLE_SGLANG_MEMORY_SAVER"]}}))')

ray job submit --address="http://127.0.0.1:${RAY_DASHBOARD_PORT}" \
    --runtime-env-json="${RUNTIME_ENV_JSON}" \
    -- python train.py "${TRAIN_ARGS[@]}"
