#!/usr/bin/env bash
# Tau agent 本地 vLLM 功能验证脚本
# 前置：vLLM 服务已启动，且已运行 setup_tau_local_vllm.sh
set -euo pipefail

ROOT_DIR="/mnt/hdd_storage/vllm_2080ti"
VENV_DIR="${ROOT_DIR}/.venv-tau"
TEST_DIR="${ROOT_DIR}/.tmp-tau-test"
MODEL_NAME="ThinkingCap-Qwen3.6-27B-AWQ-tqk8v4-256K"

source "${VENV_DIR}/bin/activate"
export VLLM_API_KEY=sk-vllm

mkdir -p "${TEST_DIR}"

echo "==> 1. 基础对话"
tau -p "hi" --output text 2>/dev/null
echo ""

echo "==> 2. bash 工具调用"
tau -p "请用 bash 工具列出当前目录下的前 5 个文件" --output text 2>/dev/null
echo ""

echo "==> 3. read 工具调用"
tau -p "请用 read 工具读取 README.md 的前 3 行并总结项目用途" --output text 2>/dev/null
echo ""

echo "==> 4. write + bash 组合工具调用"
rm -f "${TEST_DIR}/test_tau.py"
tau --cwd "${TEST_DIR}" \
  -p "请创建一个 test_tau.py 文件，内容是打印 'Hello from Tau agent!'，然后用 bash 工具运行它" \
  --output text 2>/dev/null
if [ -f "${TEST_DIR}/test_tau.py" ]; then
  echo "  文件已创建：${TEST_DIR}/test_tau.py"
  python "${TEST_DIR}/test_tau.py"
else
  echo "ERROR: 文件未创建"
  exit 1
fi

echo ""
echo "==> 所有验证通过 ✓"
