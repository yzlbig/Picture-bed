#!/usr/bin/env bash
set -euo pipefail

# Report baseline: vLLM 2cf0a69 + vllm-ascend 43f69a69.
export PYTHONPATH="/home/yzl/vllm:/home/yzl/vllm-ascend:${PYTHONPATH:-}"
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
export VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_ENGINE_READY_TIMEOUT_S=3600
export TASK_QUEUE_ENABLE=1
export CPU_AFFINITY_CONF=1
export HCCL_OP_EXPANSION_MODE=AIV
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

LOG_DIR=/home/yzl/precision/tp4/gemma-4-31b-it/server
mkdir -p "$LOG_DIR"

exec vllm serve /home/weight/gemma-4-31B-it \
  --host 0.0.0.0 \
  --port 8009 \
  --served-model-name gemma-4-31b-it \
  --tensor-parallel-size 4 \
  --gpu-memory-utilization 0.8 \
  --max-model-len 16384 \
  --max-num-seqs 16 \
  --max-num-batched-tokens 4096 \
  --dtype bfloat16 \
  --async-scheduling \
  --no-enable-prefix-caching \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
  --additional-config '{"enable_cpu_binding":true}' \
  --trust-remote-code \
  2>&1 | tee "$LOG_DIR/server.log"
