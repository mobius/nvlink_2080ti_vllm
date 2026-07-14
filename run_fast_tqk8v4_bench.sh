#!/usr/bin/env bash
# 一键启动 weicj fast/tqk8v4 服务并跑 PP4096/TG128 benchmark
# 用法：bash run_fast_tqk8v4_bench.sh [label_suffix]
# 环境变量覆盖：
#   MODEL_DIR             模型目录（默认 Qwen3.6-27B-AWQ-INT4）
#   SERVED_NAME           服务名（默认带 MTP 的原版模型名）
#   MTP_K                 MTP 投机 token 数（默认 3；模型无 MTP 头时请设为 0）
#   PROFILE               weicj profile 路径（默认 fast/int4/tqk8v4-256K-mtp3-text-only.env）
#   TRITON_CACHE_SUFFIX   区分不同模型的 Triton 缓存后缀（默认空）
#   SERVICE_SCOPE         服务监听范围：local=127.0.0.1, lan=0.0.0.0（默认 local）
set -euo pipefail

ROOT_DIR="/mnt/hdd_storage/vllm_2080ti"
WEICJ_DIR="${ROOT_DIR}/weicj-vllm-2080ti"
# 允许通过环境变量 MODEL_DIR 切换模型；默认使用原版 Qwen3.6-27B-AWQ-INT4
MODEL_DIR="${MODEL_DIR:-${ROOT_DIR}/models/Qwen3.6-27B-AWQ-INT4}"
LOG_DIR="${ROOT_DIR}/logs"
TRITON_CACHE="${ROOT_DIR}/cache/triton_weicj_tqk8v4${TRITON_CACHE_SUFFIX:-_v2}"

LABEL_SUFFIX="${1:-$(date +%Y%m%d-%H%M%S)}"
LABEL="my-fast-tqk8v4-pp4096-tg128-${LABEL_SUFFIX}"
SERVED_NAME="${SERVED_NAME:-qwen27b-int4-tqk8v4-256K-mtp3-text-only-cu128}"
MTP_K="${MTP_K:-3}"
PROFILE="${PROFILE:-qwen27b/fast/int4/tqk8v4-256K-mtp3-text-only.env}"
SERVICE_SCOPE="${SERVICE_SCOPE:-local}"
# Qwen3/3.5/3.6 使用 XML 工具调用格式；开启 auto tool choice 后 agent 的 tool_choice: auto 才能工作
ENABLE_AUTO_TOOL_CHOICE="${ENABLE_AUTO_TOOL_CHOICE:-1}"
TOOL_CALL_PARSER="${TOOL_CALL_PARSER:-qwen3_xml}"
VLLM_ENFORCE_STRICT_TOOL_CALLING="${VLLM_ENFORCE_STRICT_TOOL_CALLING:-0}"

# 清理可能残留的缓存（谨慎：会重置 JIT 缓存，首次启动变慢）
# rm -rf "${TRITON_CACHE}"/* ~/.cache/torch_extensions/py311_cu128/flash_qla_legacy_gdn 2>/dev/null || true

echo "==> 停止已有 vLLM 服务..."
ps aux | grep 'vllm.entrypoints.openai.api_server' | grep -v grep | awk '{print $2}' | xargs -r kill -9 || true
sleep 3

# 确认 GPU0/1 空闲
if nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | awk -F',' '$1 ~ /^(0|1)$/ && int($2) > 100000 {exit 1}'; then
  echo "==> GPU0/1 显存占用低，继续启动"
else
  echo "WARNING: GPU0/1 仍有较高显存占用，请检查 nvidia-smi"
fi

echo "==> 启动 weicj fast/tqk8v4 服务..."
echo "    MODEL_DIR=${MODEL_DIR}"
echo "    SERVED_NAME=${SERVED_NAME}"
echo "    PROFILE=${PROFILE}"
echo "    MTP_K=${MTP_K}"
echo "    SERVICE_SCOPE=${SERVICE_SCOPE}"
echo "    ENABLE_AUTO_TOOL_CHOICE=${ENABLE_AUTO_TOOL_CHOICE}"
echo "    TOOL_CALL_PARSER=${TOOL_CALL_PARSER}"
cd "${WEICJ_DIR}"
env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,1 GPU_DEVICES=0,1 TP_SIZE=2 \
    CC=/usr/bin/gcc-12 CXX=/usr/bin/g++-12 CUDAHOSTCXX=/usr/bin/g++-12 \
    QUANTIZATION=compressed-tensors \
    HF_HOME="${ROOT_DIR}/cache/hf" \
    TRITON_CACHE_DIR="${TRITON_CACHE}" \
    MODEL_DIR="${MODEL_DIR}" \
    SERVED_NAME="${SERVED_NAME}" \
    PROFILE="${PROFILE}" \
    MODE=fast PORT=8000 SERVICE_SCOPE="${SERVICE_SCOPE}" \
    MTP_K="${MTP_K}" \
    ENABLE_AUTO_TOOL_CHOICE="${ENABLE_AUTO_TOOL_CHOICE}" \
    TOOL_CALL_PARSER="${TOOL_CALL_PARSER}" \
    VLLM_ENFORCE_STRICT_TOOL_CALLING="${VLLM_ENFORCE_STRICT_TOOL_CALLING}" \
    ./launcher.sh --non-interactive > "${LOG_DIR}/${LABEL}-launch.log" 2>&1 &

LAUNCHER_PID=$!
echo "  launcher PID: ${LAUNCHER_PID}"
echo "  启动日志: ${LOG_DIR}/${LABEL}-launch.log"

# 等待服务就绪（最多 5 分钟）
echo "==> 等待服务就绪..."
timeout=300
elapsed=0
while ! curl -s http://127.0.0.1:8000/health >/dev/null 2>&1; do
  if ! kill -0 "${LAUNCHER_PID}" 2>/dev/null; then
    echo "ERROR: launcher 进程已退出，请查看日志"
    tail -n 50 "${LOG_DIR}/${LABEL}-launch.log"
    exit 1
  fi
  sleep 5
  elapsed=$((elapsed + 5))
  if [ "${elapsed}" -ge "${timeout}" ]; then
    echo "ERROR: 服务在 ${timeout}s 内未就绪"
    exit 1
  fi
  echo "  已等待 ${elapsed}s..."
done

echo "==> 服务就绪，运行 PP4096/TG128 benchmark..."
.venv/bin/python tools/profile_request.py \
  --model-dir "${MODEL_DIR}" \
  --served-name "${SERVED_NAME}" \
  --base-url http://127.0.0.1:8000/v1 \
  --endpoint completions \
  --prompt-tokens 4096 \
  --gen-tokens 128 \
  --label "${LABEL}" \
  --out "${LOG_DIR}/${LABEL}-bench.jsonl" \
  --gpu-log "${LOG_DIR}/${LABEL}-gpu.log" \
  --ignore-eos --pure-filler

echo "==> benchmark 完成，结果："
.venv/bin/python - <<PY
import json
path = "${LOG_DIR}/${LABEL}-bench.jsonl"
with open(path) as f:
    r = json.loads(f.readlines()[-1])
print(f"  label: {r['label']}")
print(f"  prompt_tokens: {r['prompt_tokens']}")
print(f"  completion_tokens: {r['completion_tokens']}")
print(f"  ttft_s: {r['ttft_s']:.3f}")
print(f"  prefill_tok_s: {r['prefill_tok_s']:.1f}")
print(f"  decode_tok_s: {r['decode_tok_s']:.2f}")
print(f"  原始数据: {path}")
PY

echo "==> 停止服务..."
ps aux | grep 'vllm.entrypoints.openai.api_server' | grep -v grep | awk '{print $2}' | xargs -r kill -9 || true

echo "==> 完成"
