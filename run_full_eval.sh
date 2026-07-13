#!/usr/bin/env bash
# 一键完整评估：性能 benchmark + 质量评估
# 用法：bash run_full_eval.sh [label后缀]
set -euo pipefail

ROOT_DIR="/mnt/hdd_storage/vllm_2080ti"
WEICJ_DIR="${ROOT_DIR}/weicj-vllm-2080ti"
# 允许通过环境变量 MODEL_DIR 切换模型；默认使用原版 Qwen3.6-27B-AWQ-INT4
MODEL_DIR="${MODEL_DIR:-${ROOT_DIR}/models/Qwen3.6-27B-AWQ-INT4}"
LOG_DIR="${ROOT_DIR}/logs"
TRITON_CACHE="${ROOT_DIR}/cache/triton_weicj_tqk8v4_v2"

LABEL_SUFFIX="${1:-$(date +%Y%m%d-%H%M%S)}"
LABEL="full-eval-${LABEL_SUFFIX}"
SERVED_NAME="${SERVED_NAME:-qwen27b-int4-tqk8v4-256K-mtp3-text-only-cu128}"
BASE_URL="${BASE_URL:-http://127.0.0.1:8000/v1}"

echo "=========================================="
echo "  完整评估：性能 + 质量"
echo "  label: ${LABEL}"
echo "  model_dir: ${MODEL_DIR}"
echo "  served_name: ${SERVED_NAME}"
echo "=========================================="

# 1. 停止已有服务
echo ""
echo "==> [1/5] 停止已有 vLLM 服务..."
ps aux | grep 'vllm.entrypoints.openai.api_server' | grep -v grep | awk '{print $2}' | xargs -r kill -9 || true
sleep 3

# 2. 启动服务
echo ""
echo "==> [2/5] 启动 weicj fast/tqk8v4 服务..."
echo "  MODEL_DIR=${MODEL_DIR}"
echo "  SERVED_NAME=${SERVED_NAME}"
cd "${WEICJ_DIR}"
env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,1 GPU_DEVICES=0,1 TP_SIZE=2 \
    CC=/usr/bin/gcc-12 CXX=/usr/bin/g++-12 CUDAHOSTCXX=/usr/bin/g++-12 \
    QUANTIZATION=compressed-tensors \
    HF_HOME="${ROOT_DIR}/cache/hf" \
    TRITON_CACHE_DIR="${TRITON_CACHE}" \
    MODEL_DIR="${MODEL_DIR}" \
    PROFILE=qwen27b/fast/int4/tqk8v4-256K-mtp3-text-only.env \
    MODE=fast PORT=8000 SERVICE_SCOPE=local \
    REASONING_PARSER=off \
    DEFAULT_CHAT_TEMPLATE_KWARGS='{"enable_thinking":false}' \
    ./launcher.sh --non-interactive > "${LOG_DIR}/${LABEL}-launch.log" 2>&1 &

LAUNCHER_PID=$!
echo "  launcher PID: ${LAUNCHER_PID}"

# 等待服务就绪
echo "  等待服务就绪..."
timeout=300
elapsed=0
while ! curl -s http://127.0.0.1:8000/health >/dev/null 2>&1; do
  if ! kill -0 "${LAUNCHER_PID}" 2>/dev/null; then
    echo "ERROR: launcher 进程已退出"
    tail -n 50 "${LOG_DIR}/${LABEL}-launch.log"
    exit 1
  fi
  sleep 5
  elapsed=$((elapsed + 5))
  if [ "${elapsed}" -ge "${timeout}" ]; then
    echo "ERROR: 服务在 ${timeout}s 内未就绪"
    exit 1
  fi
  echo "    已等待 ${elapsed}s..."
done
echo "  服务就绪"

# 3. 性能 benchmark
echo ""
echo "==> [3/5] 运行 PP4096/TG128 性能 benchmark..."
.venv/bin/python tools/profile_request.py \
  --model-dir "${MODEL_DIR}" \
  --served-name "${SERVED_NAME}" \
  --base-url "${BASE_URL}" \
  --endpoint completions \
  --prompt-tokens 4096 \
  --gen-tokens 128 \
  --label "${LABEL}-pp4096-tg128" \
  --out "${LOG_DIR}/${LABEL}-bench.jsonl" \
  --gpu-log "${LOG_DIR}/${LABEL}-gpu.log" \
  --ignore-eos --pure-filler

echo ""
echo "  性能结果："
.venv/bin/python - <<PY
import json
path = "${LOG_DIR}/${LABEL}-bench.jsonl"
with open(path) as f:
    r = json.loads(f.readlines()[-1])
print(f"    TTFT: {r['ttft_s']:.3f}s")
print(f"    Prefill: {r['prefill_tok_s']:.1f} tok/s")
print(f"    Decode: {r['decode_tok_s']:.2f} tok/s")
print(f"    Elapsed: {r['elapsed_s']:.3f}s")
PY

# 4. 质量评估
echo ""
echo "==> [4/5] 运行质量评估..."
cd "${ROOT_DIR}"
QUALITY_OUT="${LOG_DIR}/${LABEL}-quality.jsonl"
.venv/bin/python - <<PY
import json
import requests
from pathlib import Path

ROOT_DIR = "${ROOT_DIR}"
LOG_DIR = Path(ROOT_DIR) / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
OUT_FILE = Path("${QUALITY_OUT}")
SERVED_NAME = "${SERVED_NAME}"
BASE_URL = "${BASE_URL}"
LABEL = "${LABEL}"

def chat(messages, max_tokens=512, temperature=0.0):
    payload = {
        "model": SERVED_NAME,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False,
    }
    try:
        r = requests.post(f"{BASE_URL}/chat/completions", json=payload, timeout=120)
        r.raise_for_status()
        data = r.json()
        choice = data.get("choices", [{}])[0]
        msg = choice.get("message", {}) or {}
        content = msg.get("content") or msg.get("reasoning_content") or ""
        usage = data.get("usage", {}) or {}
        return {
            "content": content,
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
        }
    except Exception as e:
        return {"content": f"ERROR: {type(e).__name__}: {e}", "prompt_tokens": 0, "completion_tokens": 0}

def completions(prompt, max_tokens=256, temperature=0.0):
    payload = {
        "model": SERVED_NAME,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False,
    }
    try:
        r = requests.post(f"{BASE_URL}/completions", json=payload, timeout=120)
        r.raise_for_status()
        data = r.json()
        choice = data.get("choices", [{}])[0]
        content = choice.get("text", "")
        usage = data.get("usage", {})
        return {
            "content": content,
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
        }
    except Exception as e:
        return {"content": f"ERROR: {type(e).__name__}: {e}", "prompt_tokens": 0, "completion_tokens": 0}

records = []

tests = [
    ("chinese_qa", "请简要解释什么是光合作用，50字以内。", 128),
    ("english_qa", "What is the capital of France? Answer in one sentence.", 64),
    ("math_reasoning", "A train travels 120 km in 2 hours. How far will it travel in 5 hours at the same speed? Show your reasoning.", 256),
    ("code_generation", "Write a Python function that reverses a string without using slicing. Include a docstring.", 256),
]

for category, question, max_tokens in tests:
    print(f"    测试: {category}...")
    result = chat([{"role": "user", "content": question}], max_tokens=max_tokens)
    records.append({
        "label": LABEL,
        "category": category,
        "question": question,
        "answer": result["content"],
        "prompt_tokens": result["prompt_tokens"],
        "completion_tokens": result["completion_tokens"],
    })

# 大海捞针
print("    测试: needle_in_haystack...")
needle = "秘密代码是 K7P-2080Ti-NVLink。"
filler = "The quick brown fox jumps over the lazy dog. " * 200
question = "上文提到的秘密代码是什么？只回答代码本身，不要思考过程。"
result = chat([
    {"role": "user", "content": filler + "\n" + needle + "\n" + filler + "\n\n" + question}
], max_tokens=64, temperature=0.0)
needle_found = "K7P-2080Ti-NVLink" in (result["content"] or "")
records.append({
    "label": LABEL,
    "category": "needle_in_haystack",
    "question": question,
    "answer": result["content"],
    "needle": needle,
    "needle_found": needle_found,
    "prompt_tokens": result["prompt_tokens"],
    "completion_tokens": result["completion_tokens"],
})

with open(OUT_FILE, "w", encoding="utf-8") as f:
    for r in records:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

print("\n    质量评估汇总：")
for r in records:
    status = ""
    if r["category"] == "needle_in_haystack":
        status = f" [needle_found={r['needle_found']}]"
    print(f"      {r['category']}: {len(r['answer'] or '')} chars{status}")
PY

# 5. 停止服务
echo ""
echo "==> [5/5] 停止服务..."
ps aux | grep 'vllm.entrypoints.openai.api_server' | grep -v grep | awk '{print $2}' | xargs -r kill -9 || true

echo ""
echo "=========================================="
echo "  完整评估完成"
echo "  label: ${LABEL}"
echo "  性能结果: ${LOG_DIR}/${LABEL}-bench.jsonl"
echo "  质量结果: ${LOG_DIR}/${LABEL}-quality.jsonl"
echo "  服务日志: ${LOG_DIR}/${LABEL}-launch.log"
echo "  GPU 日志: ${LOG_DIR}/${LABEL}-gpu.log"
echo "=========================================="
