# Launches sglang server for Qwen2.5-1.5B-Instruct on 4 GPUs.
# bash examples/remote_inference_engine/run_sglang_server.sh
set -x

export NCCL_P2P_DISABLE=1          # Disable P2P (often helps with PCIe-only setups)
export NCCL_SHM_DISABLE=0          # Enable shared memory
export NCCL_NET_GDR_LEVEL=0        # Disable GPUDirect RDMA
export NCCL_IB_DISABLE=1           # Disable InfiniBand
export NCCL_DEBUG=INFO             # Get more debug info


CUDA_VISIBLE_DEVICES=2 uv run --isolated --extra sglang -m \
    skyrl_train.inference_engines.sglang.sglang_server \
    --model-path Qwen/Qwen2.5-1.5B-Instruct \
    --tp 1 \
    --host 127.0.0.1 \
    --port 8001 \
    --context-length 4096 \
    --dtype bfloat16