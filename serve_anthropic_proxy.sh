#!/usr/bin/env bash
# 启动 Anthropic Messages API -> vLLM OpenAI API 代理
# 用法：bash serve_anthropic_proxy.sh
# 停止：bash serve_anthropic_proxy.sh stop
set -uo pipefail

ROOT_DIR="/mnt/hdd_storage/vllm_2080ti"
VENV_DIR="${ROOT_DIR}/.venv-anthropic-proxy"
LOG_DIR="${ROOT_DIR}/logs"
PID_FILE="${ROOT_DIR}/.anthropic-proxy-server.pid"

UPSTREAM_BASE_URL="${UPSTREAM_BASE_URL:-http://127.0.0.1:8000/v1}"
UPSTREAM_API_KEY="${UPSTREAM_API_KEY:-sk-vllm}"
PROXY_PORT="${PROXY_PORT:-18081}"
PROXY_HOST="${PROXY_HOST:-0.0.0.0}"
# 如需对外部访问做鉴权，设置 PROXY_API_KEY；留空则不校验
PROXY_API_KEY="${PROXY_API_KEY:-}"
# 允许通过环境变量指定代理对外暴露的模型名
MODEL_NAME="${MODEL_NAME:-qwen27b-int4-tqk8v4-256K-mtp3-text-only-cu128}"

mkdir -p "${LOG_DIR}"

# 安全停止代理：只杀 PID 文件记录的进程，以及监听 PROXY_PORT 的进程。
stop_proxy() {
  echo "==> 停止 Anthropic proxy 服务 (PORT=${PROXY_PORT})..."
  if [ -f "${PID_FILE}" ]; then
    pid=$(cat "${PID_FILE}")
    if kill -0 "${pid}" 2>/dev/null; then
      pkill -P "${pid}" 2>/dev/null || true
      kill -9 "${pid}" 2>/dev/null || true
    fi
    rm -f "${PID_FILE}"
  fi
  listener_pid=$(ss -ltnp "sport = :${PROXY_PORT}" 2>/dev/null | grep -oP 'pid=\K[0-9]+' | head -1)
  if [ -n "${listener_pid}" ]; then
    if kill -0 "${listener_pid}" 2>/dev/null; then
      kill -9 "${listener_pid}" 2>/dev/null || true
      echo "  已停止端口 ${PROXY_PORT} 的监听进程 PID=${listener_pid}"
    fi
  fi
  echo "==> 已停止"
}

case "${1:-}" in
  stop)
    stop_proxy
    exit 0
    ;;
esac

# 启动前先停止已有实例，避免端口冲突
echo "==> 停止已有 Anthropic proxy 实例..."
stop_proxy >/dev/null 2>&1 || true

# 检查 vLLM 上游
if ! curl -s "${UPSTREAM_BASE_URL}/models" >/dev/null 2>&1; then
  echo "ERROR: 上游 vLLM 服务未就绪: ${UPSTREAM_BASE_URL}"
  echo "  请先启动: bash serve_fast_tqk8v4_web.sh"
  exit 1
fi

if [ ! -d "${VENV_DIR}" ]; then
  echo "==> 创建 anthropic-proxy 虚拟环境..."
  uv venv "${VENV_DIR}" --python 3.11
  uv pip install --python "${VENV_DIR}/bin/python" fastapi uvicorn httpx python-dotenv pydantic
fi

echo "==> 启动 Anthropic proxy..."
echo "  UPSTREAM_BASE_URL=${UPSTREAM_BASE_URL}"
echo "  PROXY_HOST=${PROXY_HOST}"
echo "  PROXY_PORT=${PROXY_PORT}"
echo "  PROXY_API_KEY=${PROXY_API_KEY:-(未设置，不鉴权)}"

cd "${ROOT_DIR}"
env UPSTREAM_BASE_URL="${UPSTREAM_BASE_URL}" \
    UPSTREAM_API_KEY="${UPSTREAM_API_KEY}" \
    PROXY_API_KEY="${PROXY_API_KEY}" \
    HOST="${PROXY_HOST}" \
    PORT="${PROXY_PORT}" \
    "${VENV_DIR}/bin/python" "${ROOT_DIR}/tools/anthropic_proxy.py" \
    > "${LOG_DIR}/anthropic-proxy-server.log" 2>&1 &

PYTHON_PID=$!
echo "${PYTHON_PID}" > "${PID_FILE}"
echo "  anthropic-proxy PID: ${PYTHON_PID}"

echo "==> 等待 Anthropic proxy 就绪..."
timeout=60
elapsed=0
while ! curl -s "http://127.0.0.1:${PROXY_PORT}/health" >/dev/null 2>&1; do
  if ! kill -0 "${PYTHON_PID}" 2>/dev/null; then
    echo "ERROR: Anthropic proxy 进程已退出"
    tail -n 50 "${LOG_DIR}/anthropic-proxy-server.log"
    rm -f "${PID_FILE}"
    exit 1
  fi
  sleep 2
  elapsed=$((elapsed + 2))
  if [ "${elapsed}" -ge "${timeout}" ]; then
    echo "ERROR: Anthropic proxy 在 ${timeout}s 内未就绪"
    rm -f "${PID_FILE}"
    exit 1
  fi
  echo "  已等待 ${elapsed}s..."
done

lan_ip=$(hostname -I 2>/dev/null | awk '{print $1}')
echo ""
echo "=========================================="
echo "  Anthropic proxy 已就绪"
echo "  本机访问: http://127.0.0.1:${PROXY_PORT}/v1/messages"
if [ -n "${lan_ip}" ]; then
  echo "  局域网访问: http://${lan_ip}:${PROXY_PORT}/v1/messages"
fi
echo "  上游 vLLM: ${UPSTREAM_BASE_URL}"
echo "=========================================="
echo ""
echo "  Agent 配置示例:"
echo "    base_url: http://${lan_ip:-127.0.0.1}:${PROXY_PORT}"
echo "    api_key:  (任意值，或未设置 PROXY_API_KEY 时不校验)"
echo "    model:    ${MODEL_NAME}"
echo ""
echo "  停止服务:"
echo "    bash serve_anthropic_proxy.sh stop"
echo ""
echo "  日志: ${LOG_DIR}/anthropic-proxy-server.log"
echo "=========================================="

wait "${PYTHON_PID}" || true
rm -f "${PID_FILE}"
