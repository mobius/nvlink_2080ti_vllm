#!/usr/bin/env bash
# 一键启动 Gradio 网页聊天界面，连接本地 vLLM 服务
# 用法：bash serve_gradio_chat.sh
# 停止：Ctrl+C 或 kill 对应 python 进程
set -uo pipefail

ROOT_DIR="/mnt/hdd_storage/vllm_2080ti"
VENV_DIR="${ROOT_DIR}/.venv-gradio"
LOG_DIR="${ROOT_DIR}/logs"
PID_FILE="${ROOT_DIR}/.gradio-chat-server.pid"

# 配置
VLLM_BASE_URL="${VLLM_BASE_URL:-http://127.0.0.1:8000/v1}"
VLLM_API_KEY="${VLLM_API_KEY:-sk-vllm}"
VLLM_MODEL="${VLLM_MODEL:-qwen27b-int4-tqk8v4-256K-mtp3-text-only-cu128}"
GRADIO_PORT="${GRADIO_PORT:-7860}"

mkdir -p "${LOG_DIR}"

case "${1:-}" in
  stop)
    echo "==> 停止 Gradio chat 服务..."
    if [ -f "${PID_FILE}" ]; then
      pid=$(cat "${PID_FILE}")
      if kill -0 "${pid}" 2>/dev/null; then
        kill -9 "${pid}" || true
      fi
      rm -f "${PID_FILE}"
    fi
    ps aux | grep 'tools/gradio_chat.py' | grep -v grep | awk '{print $2}' | xargs -r kill -9 || true
    echo "==> 已停止"
    exit 0
    ;;
esac

# 检查 vLLM 服务是否可达
echo "==> 检查 vLLM 服务..."
if ! curl -s "${VLLM_BASE_URL}/models" >/dev/null 2>&1; then
  echo "ERROR: vLLM 服务未就绪，请先启动服务"
  echo "  例如：bash serve_fast_tqk8v4_web.sh"
  exit 1
fi
echo "  vLLM 服务正常: ${VLLM_BASE_URL}"

# 创建隔离环境并安装依赖
if [ ! -d "${VENV_DIR}" ]; then
  echo "==> 创建 Gradio 虚拟环境..."
  uv venv "${VENV_DIR}" --python 3.11
fi

if ! "${VENV_DIR}/bin/python" -c "import gradio, openai" 2>/dev/null; then
  echo "==> 安装 Gradio + OpenAI SDK..."
  if command -v uv >/dev/null 2>&1; then
    uv pip install --python "${VENV_DIR}/bin/python" gradio openai
  else
    "${VENV_DIR}/bin/python" -m ensurepip && "${VENV_DIR}/bin/python" -m pip install gradio openai
  fi
fi

echo "==> 启动 Gradio chat 服务..."
echo "  VLLM_BASE_URL=${VLLM_BASE_URL}"
echo "  VLLM_MODEL=${VLLM_MODEL}"
echo "  GRADIO_PORT=${GRADIO_PORT}"

cd "${ROOT_DIR}"
env VLLM_BASE_URL="${VLLM_BASE_URL}" \
    VLLM_API_KEY="${VLLM_API_KEY}" \
    VLLM_MODEL="${VLLM_MODEL}" \
    GRADIO_SERVER_PORT="${GRADIO_PORT}" \
    GRADIO_SERVER_NAME="0.0.0.0" \
    GRADIO_ANALYTICS_ENABLED="0" \
    "${VENV_DIR}/bin/python" "${ROOT_DIR}/tools/gradio_chat.py" \
    > "${LOG_DIR}/gradio-chat-server.log" 2>&1 &

PYTHON_PID=$!
echo "${PYTHON_PID}" > "${PID_FILE}"
echo "  Python PID: ${PYTHON_PID}"

# 等待就绪
echo "==> 等待 Gradio 就绪..."
timeout=60
elapsed=0
while ! curl -s "http://127.0.0.1:${GRADIO_PORT}/" >/dev/null 2>&1; do
  if ! kill -0 "${PYTHON_PID}" 2>/dev/null; then
    echo "ERROR: Gradio 进程已退出，请查看日志"
    tail -n 80 "${LOG_DIR}/gradio-chat-server.log"
    rm -f "${PID_FILE}"
    exit 1
  fi
  sleep 2
  elapsed=$((elapsed + 2))
  if [ "${elapsed}" -ge "${timeout}" ]; then
    echo "ERROR: Gradio 在 ${timeout}s 内未就绪"
    rm -f "${PID_FILE}"
    exit 1
  fi
  echo "  已等待 ${elapsed}s..."
done

lan_ip=$(hostname -I 2>/dev/null | awk '{print $1}')
echo ""
echo "=========================================="
echo "  Gradio 网页聊天界面已就绪"
echo "  本机访问: http://127.0.0.1:${GRADIO_PORT}"
if [ -n "${lan_ip}" ]; then
  echo "  局域网访问: http://${lan_ip}:${GRADIO_PORT}"
fi
echo "  模型: ${VLLM_MODEL}"
echo "  API: ${VLLM_BASE_URL}"
echo "=========================================="
echo ""
echo "  停止服务:"
echo "    bash serve_gradio_chat.sh stop"
echo ""
echo "  日志: ${LOG_DIR}/gradio-chat-server.log"
echo "=========================================="

wait "${PYTHON_PID}" || true
rm -f "${PID_FILE}"
