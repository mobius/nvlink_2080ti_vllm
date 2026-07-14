#!/usr/bin/env bash
# 配置 Tau agent 使用本地 vLLM (ThinkingCap-Qwen3.6-27B-AWQ on 2×2080Ti)
# 用法：bash setup_tau_local_vllm.sh
set -euo pipefail

ROOT_DIR="/mnt/hdd_storage/vllm_2080ti"
VENV_DIR="${ROOT_DIR}/.venv-tau"
MODEL_NAME="ThinkingCap-Qwen3.6-27B-AWQ-tqk8v4-256K"
BASE_URL="http://127.0.0.1:8000/v1"

# 创建 Python 3.12 环境并安装 tau-ai（如果尚未安装）
if [ ! -d "${VENV_DIR}" ]; then
  echo "==> 创建 .venv-tau (Python 3.12)..."
  uv venv --python 3.12 "${VENV_DIR}"
fi

if [ ! -f "${VENV_DIR}/bin/tau" ]; then
  echo "==> 安装 tau-ai..."
  uv pip install --python "${VENV_DIR}/bin/python" tau-ai
fi

# 创建 Tau 配置目录
mkdir -p ~/.tau

# 备份已有配置
if [ -f ~/.tau/catalog.toml ] && [ ! -f ~/.tau/catalog.toml.bak ]; then
  cp ~/.tau/catalog.toml ~/.tau/catalog.toml.bak
fi
if [ -f ~/.tau/providers.json ] && [ ! -f ~/.tau/providers.json.bak ]; then
  cp ~/.tau/providers.json ~/.tau/providers.json.bak
fi

# provider 元数据
cat > ~/.tau/catalog.toml <<EOF
schema_version = 1

[[providers]]
name = "local-vllm"
display_name = "Local vLLM (2080Ti)"
kind = "openai-compatible"
base_url = "${BASE_URL}"
api_key_env = "VLLM_API_KEY"
credential_name = "local-vllm"
models = ["${MODEL_NAME}"]
default_model = "${MODEL_NAME}"
docs_url = "http://30.19.40.129:8000/docs"
api = "openai-completions"
thinking_levels = ["off", "low", "medium", "high"]
thinking_default = "off"
thinking_parameter = "reasoning_effort"

[providers.context_windows]
"${MODEL_NAME}" = 262144

[providers.model_metadata."${MODEL_NAME}"]
name = "ThinkingCap Qwen3.6 27B AWQ"
reasoning = true
input = ["text"]
context_window = 262144
max_tokens = 32000
EOF

# runtime 偏好
cat > ~/.tau/providers.json <<EOF
{
  "default_provider": "local-vllm",
  "provider_preferences": {
    "local-vllm": {
      "default_model": "${MODEL_NAME}",
      "timeout_seconds": 600,
      "max_retries": 1,
      "max_retry_delay_seconds": 1
    }
  },
  "scoped_models": []
}
EOF

echo "==> Tau 配置完成"
echo "    Provider: local-vllm"
echo "    Base URL: ${BASE_URL}"
echo "    Model:    ${MODEL_NAME}"
echo ""
echo "==> 用法示例："
echo "    source ${VENV_DIR}/bin/activate"
echo "    export VLLM_API_KEY=sk-vllm"
echo "    tau -p 'hi'"
echo "    tau -p '请用 bash 工具列出当前目录文件'"
echo ""
echo "==> 验证命令："
echo "    bash run_tau_smoke_test.sh"
