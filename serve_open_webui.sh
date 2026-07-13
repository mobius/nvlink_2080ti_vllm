#!/usr/bin/env bash
# 一键安装并启动 Open WebUI，连接本地 vLLM OpenAI API
# 用法：bash serve_open_webui.sh
# 停止：bash serve_open_webui.sh stop
set -uo pipefail

ROOT_DIR="/mnt/hdd_storage/vllm_2080ti"
VENV_DIR="${ROOT_DIR}/.venv-openwebui"
LOG_DIR="${ROOT_DIR}/logs"
DATA_DIR="${ROOT_DIR}/cache/open-webui"
PID_FILE="${ROOT_DIR}/.open-webui-server.pid"

# 配置
VLLM_BASE_URL="${VLLM_BASE_URL:-http://127.0.0.1:8000/v1}"
VLLM_API_KEY="${VLLM_API_KEY:-sk-vllm}"
WEBUI_PORT="${WEBUI_PORT:-18080}"
WEBUI_HOST="${WEBUI_HOST:-0.0.0.0}"
# 注意：Open WebUI 在有用户后不支持完全匿名访问。
# 首次启动会自动创建默认管理员（若数据库为空），之后用该账号登录即可聊天。
# 出于安全考虑，管理员密码必须从环境变量传入，脚本不再硬编码默认值。
WEBUI_AUTH="${WEBUI_AUTH:-true}"
WEBUI_ADMIN_NAME="${WEBUI_ADMIN_NAME:-Admin}"
WEBUI_ADMIN_EMAIL="${WEBUI_ADMIN_EMAIL:-admin@example.com}"
WEBUI_ADMIN_PASSWORD="${WEBUI_ADMIN_PASSWORD:-}"

# 安全校验：若启用认证且未提供管理员密码，拒绝启动
if [ "${WEBUI_AUTH}" = "true" ] && [ -z "${WEBUI_ADMIN_PASSWORD}" ]; then
  echo "ERROR: 请设置 WEBUI_ADMIN_PASSWORD 环境变量后再启动。"
  echo "  示例: WEBUI_ADMIN_PASSWORD=YourStrongPassword bash serve_open_webui.sh"
  exit 1
fi

case "${1:-}" in
  stop)
    echo "==> 停止 Open WebUI 服务..."
    if [ -f "${PID_FILE}" ]; then
      pid=$(cat "${PID_FILE}")
      if kill -0 "${pid}" 2>/dev/null; then
        kill -9 "${pid}" || true
      fi
      rm -f "${PID_FILE}"
    fi
    pgrep -f 'open-webui serve' | xargs -r kill -9 || true
    pgrep -f 'open_webui' | xargs -r kill -9 || true
    echo "==> 已停止"
    exit 0
    ;;
esac

# 检查 vLLM
echo "==> 检查 vLLM 服务..."
if ! curl -s "${VLLM_BASE_URL}/models" >/dev/null 2>&1; then
  echo "ERROR: vLLM 服务未就绪，请先启动服务"
  echo "  例如：bash serve_fast_tqk8v4_web.sh"
  exit 1
fi
echo "  vLLM 服务正常: ${VLLM_BASE_URL}"

# 创建隔离环境
if [ ! -d "${VENV_DIR}" ]; then
  echo "==> 创建 Open WebUI 虚拟环境..."
  uv venv "${VENV_DIR}" --python 3.11
fi

# 安装 open-webui
if ! "${VENV_DIR}/bin/python" -c "import open_webui" 2>/dev/null; then
  echo "==> 安装 Open WebUI（首次需要几分钟）..."
  uv pip install --python "${VENV_DIR}/bin/python" open-webui
fi

echo "==> 启动 Open WebUI..."
echo "  VLLM_BASE_URL=${VLLM_BASE_URL}"
echo "  WEBUI_PORT=${WEBUI_PORT}"
echo "  WEBUI_AUTH=${WEBUI_AUTH}"

cd "${ROOT_DIR}"
env OPENAI_API_BASE_URL="${VLLM_BASE_URL}" \
    OPENAI_API_KEY="${VLLM_API_KEY}" \
    WEBUI_AUTH="${WEBUI_AUTH}" \
    WEBUI_ADMIN_NAME="${WEBUI_ADMIN_NAME}" \
    WEBUI_ADMIN_EMAIL="${WEBUI_ADMIN_EMAIL}" \
    WEBUI_ADMIN_PASSWORD="${WEBUI_ADMIN_PASSWORD}" \
    DATA_DIR="${DATA_DIR}" \
    "${VENV_DIR}/bin/open-webui" serve \
    --host "${WEBUI_HOST}" \
    --port "${WEBUI_PORT}" \
    > "${LOG_DIR}/open-webui-server.log" 2>&1 &

PYTHON_PID=$!
echo "${PYTHON_PID}" > "${PID_FILE}"
echo "  open-webui PID: ${PYTHON_PID}"

# 等待就绪
echo "==> 等待 Open WebUI 就绪..."
timeout=180
elapsed=0
while ! curl -s "http://127.0.0.1:${WEBUI_PORT}/" >/dev/null 2>&1; do
  if ! kill -0 "${PYTHON_PID}" 2>/dev/null; then
    echo "ERROR: Open WebUI 进程已退出，请查看日志"
    tail -n 80 "${LOG_DIR}/open-webui-server.log"
    rm -f "${PID_FILE}"
    exit 1
  fi
  sleep 5
  elapsed=$((elapsed + 5))
  if [ "${elapsed}" -ge "${timeout}" ]; then
    echo "ERROR: Open WebUI 在 ${timeout}s 内未就绪"
    rm -f "${PID_FILE}"
    exit 1
  fi
  echo "  已等待 ${elapsed}s..."
done

lan_ip=$(hostname -I 2>/dev/null | awk '{print $1}')
echo ""
echo "=========================================="
echo "  Open WebUI 已就绪"
echo "  本机访问: http://127.0.0.1:${WEBUI_PORT}"
if [ -n "${lan_ip}" ]; then
  echo "  局域网访问: http://${lan_ip}:${WEBUI_PORT}"
fi
echo "  模型 API: ${VLLM_BASE_URL}"
echo "=========================================="
echo ""
echo "  使用方式："
echo "    1. 浏览器打开上述地址"
echo "    2. 用默认管理员账号登录："
echo "       邮箱: ${WEBUI_ADMIN_EMAIL}"
echo "       密码: ${WEBUI_ADMIN_PASSWORD}"
echo "    3. 登录后在左上角选择模型：qwen27b-int4-tqk8v4-256K-mtp3-text-only-cu128"
echo "    4. 直接开始聊天"
echo ""
echo "  提示："
echo "    - 若首次启动时数据库为空，会自动创建上述默认管理员账号"
echo "    - 模型 API 已自动配置为 ${VLLM_BASE_URL}"
echo ""
echo "  停止服务:"
echo "    bash serve_open_webui.sh stop"
echo ""
echo "  日志: ${LOG_DIR}/open-webui-server.log"
echo "=========================================="

wait "${PYTHON_PID}" || true
rm -f "${PID_FILE}"
