#!/usr/bin/env bash
# 评估 Qwen3.6-27B / ThinkingCap 等模型在 weicj vLLM 服务上的输出质量
# 用法：
#   1. 先启动 vLLM 服务
#   2. bash run_quality_eval.sh [label后缀]
# 可通过环境变量覆盖：SERVED_NAME, BASE_URL, DISABLE_THINKING
set -euo pipefail

ROOT_DIR="/mnt/hdd_storage/vllm_2080ti"
WEICJ_DIR="${ROOT_DIR}/weicj-vllm-2080ti"
LOG_DIR="${ROOT_DIR}/logs"

SERVED_NAME="${SERVED_NAME:-qwen27b-int4-tqk8v4-256K-mtp3-text-only-cu128}"
BASE_URL="${BASE_URL:-http://127.0.0.1:8000/v1}"
DISABLE_THINKING="${DISABLE_THINKING:-1}"
LABEL_SUFFIX="${1:-$(date +%Y%m%d-%H%M%S)}"
LABEL="quality-eval-${LABEL_SUFFIX}"
OUT_FILE="${LOG_DIR}/${LABEL}.jsonl"

echo "==> 检查服务是否就绪..."
if ! curl -s "${BASE_URL/\$1/}/models" >/dev/null 2>&1; then
  echo "ERROR: 服务未就绪，请先启动 vLLM 服务"
  exit 1
fi

cd "${WEICJ_DIR}"

.venv/bin/python - <<PY
import json
import requests
import time
import sys
import re
from pathlib import Path

ROOT_DIR = "${ROOT_DIR}"
LOG_DIR = Path(ROOT_DIR) / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
OUT_FILE = Path("${OUT_FILE}")
SERVED_NAME = "${SERVED_NAME}"
BASE_URL = "${BASE_URL}"
LABEL = "${LABEL}"
DISABLE_THINKING = "${DISABLE_THINKING}" == "1"

def chat(messages, max_tokens=512, temperature=0.0):
    payload = {
        "model": SERVED_NAME,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False,
    }
    if DISABLE_THINKING:
        payload["chat_template_kwargs"] = {"enable_thinking": False}
    try:
        r = requests.post(f"{BASE_URL}/chat/completions", json=payload, timeout=120)
        r.raise_for_status()
        data = r.json()
        choice = data.get("choices", [{}])[0]
        msg = choice.get("message", {}) or {}
        content = msg.get("content") or ""
        reasoning = msg.get("reasoning") or msg.get("reasoning_content") or ""
        # If thinking wasn't disabled at engine level, prefer direct content over reasoning
        if not content and reasoning:
            content = reasoning
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

# 1. 中文常识
print("==> 测试 1/5: 中文常识问答")
q1 = "请简要解释什么是光合作用，50字以内。"
result = chat([{"role": "user", "content": q1}], max_tokens=128)
records.append({
    "label": LABEL,
    "category": "chinese_qa",
    "question": q1,
    "answer": result["content"],
    "prompt_tokens": result["prompt_tokens"],
    "completion_tokens": result["completion_tokens"],
})
print(f"  回答: {(result['content'] or '')[:120]}...")

# 2. 英文常识
print("==> 测试 2/5: 英文常识问答")
q2 = "What is the capital of France? Answer in one sentence."
result = chat([{"role": "user", "content": q2}], max_tokens=64)
records.append({
    "label": LABEL,
    "category": "english_qa",
    "question": q2,
    "answer": result["content"],
    "prompt_tokens": result["prompt_tokens"],
    "completion_tokens": result["completion_tokens"],
})
print(f"  回答: {(result['content'] or '')[:120]}...")

# 3. 数学推理
print("==> 测试 3/5: 数学推理")
q3 = "A train travels 120 km in 2 hours. How far will it travel in 5 hours at the same speed? Show your reasoning."
result = chat([{"role": "user", "content": q3}], max_tokens=256)
records.append({
    "label": LABEL,
    "category": "math_reasoning",
    "question": q3,
    "answer": result["content"],
    "prompt_tokens": result["prompt_tokens"],
    "completion_tokens": result["completion_tokens"],
})
print(f"  回答: {(result['content'] or '')[:120]}...")

# 4. 代码生成
print("==> 测试 4/5: 代码生成")
q4 = "Write a Python function that reverses a string without using slicing. Include a docstring."
result = chat([{"role": "user", "content": q4}], max_tokens=256)
records.append({
    "label": LABEL,
    "category": "code_generation",
    "question": q4,
    "answer": result["content"],
    "prompt_tokens": result["prompt_tokens"],
    "completion_tokens": result["completion_tokens"],
})
print(f"  回答: {(result['content'] or '')[:120]}...")

# 5. 大海捞针（needle in haystack）
print("==> 测试 5/5: 大海捞针（长上下文信息提取）")
needle = "秘密代码是 K7P-2080Ti-NVLink。"
filler = "The quick brown fox jumps over the lazy dog. " * 200  # 约 11K tokens
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
print(f"  回答: {(result['content'] or '')[:120]}...")
print(f"  是否找回 needle: {needle_found}")

# 保存结果
with open(OUT_FILE, "w", encoding="utf-8") as f:
    for r in records:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

print(f"\n==> 质量评估完成，结果保存到: {OUT_FILE}")
print("\n=== 汇总 ===")
for r in records:
    status = ""
    if r["category"] == "needle_in_haystack":
        status = f"[needle_found={r['needle_found']}]"
    print(f"  {r['category']}: {len(r['answer'] or '')} chars {status}")
PY
