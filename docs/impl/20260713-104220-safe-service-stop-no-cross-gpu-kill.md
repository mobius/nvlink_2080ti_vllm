# 服务停止逻辑改造：避免误杀 3080 训练进程

**时间**: 2026-07-13 10:42  
**背景**: 在重启 2080Ti 上的 vLLM web 服务时，用户反馈 3080 GPU 上正在运行的训练/评估进程被意外终止。经排查，旧版启动脚本使用 `ps aux | grep 'vllm.entrypoints.openai.api_server'` 全局匹配并 `kill -9` 所有命中进程，过于宽泛；若 3080 训练也启动了 vLLM api_server 实例，会被一并杀掉。

## 改造前的问题

`serve_fast_tqk8v4_web.sh` 停止逻辑：

```bash
ps aux | grep 'vllm.entrypoints.openai.api_server' | grep -v grep | awk '{print $2}' | xargs -r kill -9 || true
```

这行命令会无差别杀死本机**所有**包含 `vllm.entrypoints.openai.api_server` 的 Python 进程，不管它跑在哪个 GPU、监听哪个端口、属于哪个任务。

## 改造方案

改为"双重精确锁定"：

1. **PID 文件**: 启动时把 launcher PID 写入 `.vllm-web-server.pid`，停止时只杀该 PID 及其子进程。
2. **端口兜底**: 如果 PID 文件丢失或孤儿进程残留，用 `ss -ltnp "sport = :${PORT}"` 找出监听本服务端口（默认 8000）的进程，仅杀该进程。

对 Anthropic 代理脚本 `serve_anthropic_proxy.sh` 做同样改造，避免 `pgrep -f 'anthropic_proxy.py'` 误伤其它代理实例。

## 修改后的停止函数示例

```bash
stop_service() {
  echo "==> 停止 vLLM web 服务 (PORT=${PORT})..."
  if [ -f "${PID_FILE}" ]; then
    pid=$(cat "${PID_FILE}")
    if kill -0 "${pid}" 2>/dev/null; then
      pkill -P "${pid}" 2>/dev/null || true
      kill -9 "${pid}" 2>/dev/null || true
    fi
    rm -f "${PID_FILE}"
  fi
  listener_pid=$(ss -ltnp "sport = :${PORT}" 2>/dev/null | grep -oP 'pid=\K[0-9]+' | head -1)
  if [ -n "${listener_pid}" ]; then
    if kill -0 "${listener_pid}" 2>/dev/null; then
      kill -9 "${listener_pid}" 2>/dev/null || true
    fi
  fi
  echo "==> 已停止"
}
```

## 验证

改造后检查当前状态：

- 2080Ti GPU 0/1: `VLLM::Worker_TP` 进程正常，显存 19227 MiB，health ok。
- 3080 GPU 2/3: `run_local.py --gpu_index 2/3` 训练/评估进程正常，显存 18193 MiB，利用率 33%/71%。
- 外部访问 `http://<YOUR_SERVER_IP>:8000/health` 与 `http://<YOUR_SERVER_IP>:18081/health` 均返回 ok。
- Anthropic 代理端到端请求正常返回 `thinking` + `text`。

## 涉及文件

- `serve_fast_tqk8v4_web.sh`
- `serve_anthropic_proxy.sh`
