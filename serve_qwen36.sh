#!/usr/bin/env bash
# Caovan vLLM SM75 Turbo3 v0.1.3 复现启动脚本 —— 2×RTX 2080Ti 22G (NVLink), Qwen3.6-27B-AWQ-INT4
# 用法:
#   bash serve_qwen36.sh                 # 默认保守配置(MAXLEN=32768)先验证管线
#   MAXLEN=262144 MEMUTIL=0.96 bash serve_qwen36.sh   # 复现文章的完整 256K 配置
#   TURBO3=0 bash serve_qwen36.sh         # 关闭插件跑 baseline(对照测速)
set -euo pipefail
cd /mnt/hdd_storage/vllm_2080ti

# ===== 设备硬隔离:只用两张 NVLink 直连的 2080Ti(物理 GPU0/1),绝不碰 GPU2/3 的训练任务 =====
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0,1

export OMP_NUM_THREADS=12
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTORCH_NVML_BASED_CUDA_CHECK=1

# ===== 缓存全部落项目目录,不污染全局 ~/.cache =====
export HF_HOME=/mnt/hdd_storage/vllm_2080ti/cache/hf
export TORCHINDUCTOR_CACHE_DIR=/mnt/hdd_storage/vllm_2080ti/cache/torchinductor
export TORCHINDUCTOR_COMPILE_THREADS=1
export TRITON_CACHE_DIR=/mnt/hdd_storage/vllm_2080ti/cache/triton
export TRITON_CACHE_AUTOTUNING=1
export TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas

# v0.1.3 不依赖 FlashQLA,清理可能遗留的环境变量
unset CAOVAN_FLASHQLA_PRECOMPILE 2>/dev/null || true
unset CAOVAN_FLASHQLA_FORCE 2>/dev/null || true
unset CAOVAN_FLASHQLA_DISABLE 2>/dev/null || true

MODEL=/mnt/hdd_storage/vllm_2080ti/models/Qwen3.6-27B-AWQ-INT4
MAXLEN="${MAXLEN:-32768}"     # 首启保守;验证通过后设 262144 复现文章
KVDTYPE="${KVDTYPE:-fp8}"     # SM75 上如报错可改 auto
MEMUTIL="${MEMUTIL:-0.92}"
MAXSEQS="${MAXSEQS:-4}"
TURBO3="${TURBO3:-1}"         # 1=启用插件, 0=baseline 对照

ADDCFG='{"caovan_sm75_turbo3":true}'
SPEC_ARGS=(--speculative-config '{"method":"mtp","num_speculative_tokens":2}')
if [ "$TURBO3" = "0" ]; then
  ADDCFG='{}'
  # baseline 仍保留 MTP=2 以做同条件对照(只是不启用 Turbo3 kernel)
fi

echo "[serve] MODEL=$MODEL MAXLEN=$MAXLEN KV=$KVDTYPE MEMUTIL=$MEMUTIL TURBO3=$TURBO3"
exec /mnt/hdd_storage/vllm_2080ti/.venv/bin/vllm serve "$MODEL" \
  --host 0.0.0.0 --port 8000 \
  --served-model-name Qwen3.6-27B-AWQ-INT4 \
  --tensor-parallel-size 2 \
  --dtype half \
  --max-model-len "$MAXLEN" \
  --max-num-seqs "$MAXSEQS" \
  --max-num-batched-tokens 4096 \
  --gpu-memory-utilization "$MEMUTIL" \
  --kv-cache-dtype "$KVDTYPE" \
  --disable-custom-all-reduce \
  --enable-prefix-caching \
  --mamba-cache-mode align \
  --additional-config "$ADDCFG" \
  "${SPEC_ARGS[@]}" \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder
