#!/usr/bin/env bash
# 启动 weicj fast/tqk8v4 服务并暴露 OpenAI API，供 web/chat 测试
# 用法：bash serve_fast_tqk8v4_web.sh
# 停止服务：bash serve_fast_tqk8v4_web.sh stop
set -uo pipefail

ROOT_DIR="/mnt/hdd_storage/vllm_2080ti"
WEICJ_DIR="${ROOT_DIR}/weicj-vllm-2080ti"
# 允许通过环境变量 MODEL_DIR 切换模型；默认使用原版 Qwen3.6-27B-AWQ-INT4
MODEL_DIR="${MODEL_DIR:-${ROOT_DIR}/models/Qwen3.6-27B-AWQ-INT4}"
LOG_DIR="${ROOT_DIR}/logs"
TRITON_CACHE="${ROOT_DIR}/cache/triton_weicj_tqk8v4_v2"

PID_FILE="${ROOT_DIR}/.vllm-web-server.pid"

# 允许通过环境变量覆盖关键参数
GPU_UTIL="${GPU_UTIL:-0.85}"
PORT="${PORT:-8000}"
# 模型对外名称，可自定义；默认与路径中的模型标识保持一致
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen27b-int4-tqk8v4-256K-mtp3-text-only-cu128}"
REASONING_PARSER="${REASONING_PARSER:-qwen3}"
DEFAULT_CHAT_TEMPLATE_KWARGS="${DEFAULT_CHAT_TEMPLATE_KWARGS:-{\"enable_thinking\":true}}"
# 工具调用：开启 auto tool choice，让前端 tool_choice: auto 正常工作
ENABLE_AUTO_TOOL_CHOICE="${ENABLE_AUTO_TOOL_CHOICE:-1}"
TOOL_CALL_PARSER="${TOOL_CALL_PARSER:-qwen3_xml}"
VLLM_ENFORCE_STRICT_TOOL_CALLING="${VLLM_ENFORCE_STRICT_TOOL_CALLING:-1}"
# TurboQuant 长上下文 continuation prefill workspace 预分配 tokens。
# fast mode CUDA graph 锁定 workspace 后无法增长；若实际 cached_len 超过预分配值会断言崩溃。
# 默认值 262144 与模型最大上下文对齐，覆盖 256K 长上下文请求（会线性增加显存占用）。
VLLM_TURBOQUANT_CONTINUATION_WORKSPACE_RESERVE_TOKENS="${VLLM_TURBOQUANT_CONTINUATION_WORKSPACE_RESERVE_TOKENS:-262144}"
# 关闭 CUDA graph memory profiling 以释放少量有效显存
VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS="${VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS:-0}"

# 默认用 SERVICE_SCOPE=lan 让服务监听 0.0.0.0，方便局域网浏览器/其他机器访问
# 若只想本地访问，可 SERVICE_SCOPE=local bash serve_fast_tqk8v4_web.sh
SERVICE_SCOPE="${SERVICE_SCOPE:-lan}"

# 用于脚本内健康检查的地址（服务监听 0.0.0.0 时本地仍可用 127.0.0.1）
HEALTH_HOST="127.0.0.1"
# 安全停止本服务：只杀 PID 文件里的 launcher 及其后代，以及监听本服务 PORT 的进程。
# 不杀其它 GPU/端口上的 vLLM 实例（例如 3080 上的训练/评估）。
stop_service() {
  echo "==> 停止 vLLM web 服务 (PORT=${PORT})..."
  if [ -f "${PID_FILE}" ]; then
    pid=$(cat "${PID_FILE}")
    if kill -0 "${pid}" 2>/dev/null; then
      # 先杀 launcher 的后代进程，再杀 launcher 本身
      pkill -P "${pid}" 2>/dev/null || true
      kill -9 "${pid}" 2>/dev/null || true
    fi
    rm -f "${PID_FILE}"
  fi
  # 兜底：杀掉仍占用本服务 PORT 的孤儿进程
  listener_pid=$(ss -ltnp "sport = :${PORT}" 2>/dev/null | grep -oP 'pid=\K[0-9]+' | head -1)
  if [ -n "${listener_pid}" ]; then
    if kill -0 "${listener_pid}" 2>/dev/null; then
      kill -9 "${listener_pid}" 2>/dev/null || true
      echo "  已停止端口 ${PORT} 的监听进程 PID=${listener_pid}"
    fi
  fi
  echo "==> 已停止"
}

case "${1:-}" in
  stop)
    stop_service
    exit 0
    ;;
esac

LABEL="web-server-$(date +%Y%m%d-%H%M%S)"

# launcher.sh 的 apply_profile_overrides 会无条件覆盖环境变量，
# 因此复制一份 profile 并把 GPU_UTIL 等可覆盖项写死为当前脚本想要值，
# 再透过 PROFILE_FILE 让 launcher 读取。
ORIG_PROFILE="${WEICJ_DIR}/profiles/qwen27b/fast/int4/tqk8v4-256K-mtp3-text-only.env"
TMP_PROFILE_DIR="${ROOT_DIR}/.tmp-profiles"
TMP_PROFILE="${TMP_PROFILE_DIR}/qwen27b-fast-int4-tqk8v4-${LABEL}.env"
mkdir -p "${TMP_PROFILE_DIR}"
cp "${ORIG_PROFILE}" "${TMP_PROFILE}"
sed -i "s/^GPU_UTIL=.*/GPU_UTIL=${GPU_UTIL}/" "${TMP_PROFILE}"
# 定期清理 7 天前的临时 profile
find "${TMP_PROFILE_DIR}" -name 'qwen27b-fast-int4-tqk8v4-*.env' -mtime +7 -delete 2>/dev/null || true

# 先停止已有服务（只杀本脚本启动的 2080Ti 服务，不动 3080 等其它实例）
echo "==> 停止已有 vLLM 服务..."
stop_service >/dev/null 2>&1 || true
sleep 3

echo "==> 启动 weicj fast/tqk8v4 web 服务..."
echo "  MODEL_DIR=${MODEL_DIR}"
echo "  SERVED_MODEL_NAME=${SERVED_MODEL_NAME}"
echo "  GPU_UTIL=${GPU_UTIL}, PORT=${PORT}, SERVICE_SCOPE=${SERVICE_SCOPE}"
echo "  REASONING_PARSER=${REASONING_PARSER}"
echo "  DEFAULT_CHAT_TEMPLATE_KWARGS=${DEFAULT_CHAT_TEMPLATE_KWARGS}"
echo "  ENABLE_AUTO_TOOL_CHOICE=${ENABLE_AUTO_TOOL_CHOICE}"
echo "  TOOL_CALL_PARSER=${TOOL_CALL_PARSER}"
echo "  VLLM_TURBOQUANT_CONTINUATION_WORKSPACE_RESERVE_TOKENS=${VLLM_TURBOQUANT_CONTINUATION_WORKSPACE_RESERVE_TOKENS}"
echo "  VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=${VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS}"
echo "  PROFILE_FILE=${TMP_PROFILE}"
cd "${WEICJ_DIR}"
env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,1 GPU_DEVICES=0,1 TP_SIZE=2 \
    CC=/usr/bin/gcc-12 CXX=/usr/bin/g++-12 CUDAHOSTCXX=/usr/bin/g++-12 \
    QUANTIZATION=compressed-tensors \
    HF_HOME="${ROOT_DIR}/cache/hf" \
    TRITON_CACHE_DIR="${TRITON_CACHE}" \
    MODEL_DIR="${MODEL_DIR}" \
    PROFILE_FILE="${TMP_PROFILE}" \
    MODE=fast PORT="${PORT}" SERVICE_SCOPE="${SERVICE_SCOPE}" \
    REASONING_PARSER="${REASONING_PARSER}" \
    DEFAULT_CHAT_TEMPLATE_KWARGS="${DEFAULT_CHAT_TEMPLATE_KWARGS}" \
    ENABLE_AUTO_TOOL_CHOICE="${ENABLE_AUTO_TOOL_CHOICE}" \
    TOOL_CALL_PARSER="${TOOL_CALL_PARSER}" \
    VLLM_ENFORCE_STRICT_TOOL_CALLING="${VLLM_ENFORCE_STRICT_TOOL_CALLING}" \
    VLLM_TURBOQUANT_CONTINUATION_WORKSPACE_RESERVE_TOKENS="${VLLM_TURBOQUANT_CONTINUATION_WORKSPACE_RESERVE_TOKENS}" \
    VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS="${VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS}" \
    ./launcher.sh --non-interactive > "${LOG_DIR}/${LABEL}-launch.log" 2>&1 &

LAUNCHER_PID=$!
echo "${LAUNCHER_PID}" > "${PID_FILE}"
echo "  launcher PID: ${LAUNCHER_PID}"
echo "  启动日志: ${LOG_DIR}/${LABEL}-launch.log"

# 等待服务就绪
echo "==> 等待服务就绪..."
timeout=300
elapsed=0
while ! curl -s "http://${HEALTH_HOST}:${PORT}/health" >/dev/null 2>&1; do
  if ! kill -0 "${LAUNCHER_PID}" 2>/dev/null; then
    echo "ERROR: launcher 进程已退出，请查看日志"
    tail -n 80 "${LOG_DIR}/${LABEL}-launch.log"
    rm -f "${PID_FILE}"
    exit 1
  fi
  sleep 5
  elapsed=$((elapsed + 5))
  if [ "${elapsed}" -ge "${timeout}" ]; then
    echo "ERROR: 服务在 ${timeout}s 内未就绪"
    rm -f "${PID_FILE}"
    exit 1
  fi
  echo "  已等待 ${elapsed}s..."
done

echo ""
echo "=========================================="
echo "  vLLM web 服务已就绪"
echo "  OpenAI API: http://0.0.0.0:${PORT}/v1"
# 列出本机所有 IPv4 地址，方便用户找到实际可访问的 IP
lan_ips=$(ip -4 -o addr show 2>/dev/null | awk '{print $4}' | cut -d'/' -f1 | grep -v '^127\.' | tr '\n' ' ')
if [ -n "$lan_ips" ]; then
  echo "  本机 IP 列表: ${lan_ips}"
  for ip in ${lan_ips}; do
    echo "    http://${ip}:${PORT}/v1"
  done
fi
echo "  Health check: http://127.0.0.1:${PORT}/health"
echo "  Model name: ${SERVED_MODEL_NAME}"
echo "  Model path: ${MODEL_DIR}"
echo "  最大上下文: 262144 tokens"
echo "=========================================="
echo ""
echo "  示例 curl:"
echo "    curl http://127.0.0.1:${PORT}/v1/chat/completions \\"
echo "      -H 'Content-Type: application/json' \\"
echo "      -d '{\"model\":\"qwen27b-int4-tqk8v4-256K-mtp3-text-only-cu128\",\"messages\":[{\"role\":\"user\",\"content\":\"你好\"}],\"max_tokens\":512,\"temperature\":0.0}'"
echo ""
echo "  停止服务:"
echo "    bash serve_fast_tqk8v4_web.sh stop"
echo ""
echo "  启动日志: ${LOG_DIR}/${LABEL}-launch.log"
echo "=========================================="

# 前台等待，保持脚本不退出
wait "${LAUNCHER_PID}" || true
rm -f "${PID_FILE}"
