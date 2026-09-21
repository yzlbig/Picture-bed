export TASK_QUEUE_ENABLE=1
export CPU_AFFINITY_CONF=1
export HCCL_OP_EXPANSION_MODE=AIV
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export VLLM_USE_V2_MODEL_RUNNER=1

ASCEND_RT_VISIBLE_DEVICES=4,5,6,7 vllm serve /mnt/share/weights/gemma4-31B-quant-0903 \
  --host 127.0.0.1 \
  --port 8666 \
  --served-model-name gemma4 \
  --max_model_len 8192 \
  --max-num-batched-tokens 8192 \
  --tensor-parallel-size 4 \
  --enable-auto-tool-choice \
  --tool-call-parser gemma4 \
  --reasoning-parser gemma4 \
  --no-enable-prefix-caching \
  --async-scheduling \
  --quantization ascend \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
  --additional-config '{"enable_cpu_binding": true, "weight_nz_mode": 1, "enable_reduce_sample": false, "ascend_compilation_config": {"enable_npugraph_ex": true, "enable_static_kernel": true}}' \
  --trust-remote-code
