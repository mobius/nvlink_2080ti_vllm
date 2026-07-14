# 修复 vLLM `tool_choice: "auto"` 报错并重启服务

**时间**：2026-07-14 09:54  
**模型**：ThinkingCap-Qwen3.6-27B-AWQ  
**关联脚本**：`run_fast_tqk8v4_bench.sh`、`serve_fast_tqk8v4_web.sh`、`serve_qwen36.sh`  

---

## 现象

用户 Agent 调用时收到错误：

```text
"auto" tool choice requires --enable-auto-tool-choice and --tool-call-parser to be set
```

同时观察到 vLLM 监听地址为 `127.0.0.1:8000`，而 Agent 配置里写的是 `http://30.19.40.129:8000/v1`，存在局域网访问需求。

## 根因

当前运行中的 vLLM 进程（PID 2067248）启动参数里缺少：

- `--enable-auto-tool-choice`
- `--tool-call-parser qwen3_xml`

因此任何带 `tool_choice: "auto"` 或 `tool_choice: "required"` 的请求都会被 vLLM 直接拒绝。

## 修复步骤

### 1. 仅停止 2080Ti 上的 vLLM，不伤 3080 训练

- 目标 PID：2067248（`vllm.entrypoints.openai.api_server`）
- 发送 SIGTERM，等待 6 秒后正常退出
- 确认 3080 训练进程 1876084/1876085 仍在运行

### 2. 用 tool choice 参数重启

启动命令（通过 `weicj-vllm-2080ti/launcher.sh`）：

```bash
cd /mnt/hdd_storage/vllm_2080ti/weicj-vllm-2080ti
env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,1 GPU_DEVICES=0,1 TP_SIZE=2 \
    CC=/usr/bin/gcc-12 CXX=/usr/bin/g++-12 CUDAHOSTCXX=/usr/bin/g++-12 \
    QUANTIZATION=compressed-tensors \
    HF_HOME=/mnt/hdd_storage/vllm_2080ti/cache/hf \
    TRITON_CACHE_DIR=/mnt/hdd_storage/vllm_2080ti/cache/triton_weicj_tqk8v4_thinkingcap \
    MODEL_DIR=/mnt/hdd_storage/vllm_2080ti/models/ThinkingCap-Qwen3.6-27B-AWQ \
    SERVED_NAME=ThinkingCap-Qwen3.6-27B-AWQ-tqk8v4-256K \
    PROFILE=qwen27b/fast/int4/tqk8v4-256K-mtp3-text-only.env \
    MODE=fast PORT=8000 SERVICE_SCOPE=lan \
    MTP_K=0 \
    ENABLE_AUTO_TOOL_CHOICE=1 \
    TOOL_CALL_PARSER=qwen3_xml \
    VLLM_ENFORCE_STRICT_TOOL_CALLING=0 \
    ./launcher.sh --non-interactive
```

关键变化：

- `SERVICE_SCOPE=lan`：监听 `0.0.0.0:8000`，局域网可直接访问
- `ENABLE_AUTO_TOOL_CHOICE=1` + `TOOL_CALL_PARSER=qwen3_xml`：支持 `tool_choice: auto`
- `VLLM_ENFORCE_STRICT_TOOL_CALLING=0`：避免严格格式校验导致部分工具调用失败
- `MTP_K=0`：ThinkingCap 无 MTP 头，保持关闭

### 3. 启动耗时

- 模型加载：`~4 分 32 秒`（25 GB safetensors 从 HDD 读取）
- torch.compile / CUDA graph warmup：`~1 分 30 秒`
- 总计约 `6 分钟` 后 `/health` 就绪

### 4. 验证工具调用

请求：

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "ThinkingCap-Qwen3.6-27B-AWQ-tqk8v4-256K",
    "messages": [{"role": "user", "content": "What is the weather in Beijing?"}],
    "tools": [{
      "type": "function",
      "function": {
        "name": "get_weather",
        "description": "Get current weather for a location",
        "parameters": {
          "type": "object",
          "properties": {"location": {"type": "string"}},
          "required": ["location"]
        }
      }
    }],
    "tool_choice": "auto",
    "max_tokens": 512,
    "temperature": 0.0
  }'
```

响应：

```json
{
  "choices": [{
    "message": {
      "role": "assistant",
      "content": null,
      "tool_calls": [{
        "id": "chatcmpl-tool-b6ccb981ab3a2d85",
        "type": "function",
        "function": {
          "name": "get_weather",
          "arguments": "{\"location\": \"Beijing\"}"
        }
      }],
      "reasoning": "Thinking Process:\n1. ...\n"
    },
    "finish_reason": "tool_calls"
  }]
}
```

- `finish_reason: tool_calls` 表示工具调用生效
- 未再出现 `"auto" tool choice requires ...` 错误

## 脚本更新

- `run_fast_tqk8v4_bench.sh`：默认启用 `ENABLE_AUTO_TOOL_CHOICE=1`、`TOOL_CALL_PARSER=qwen3_xml`、`VLLM_ENFORCE_STRICT_TOOL_CALLING=0`；新增 `SERVICE_SCOPE` 环境变量覆盖。
- `serve_fast_tqk8v4_web.sh`：已默认开启工具调用，无需修改。
- `serve_qwen36.sh`：已开启工具调用，无需修改。
- `README.md`：快速开始示例中加入工具调用相关环境变量。

## 注意事项

1. ThinkingCap 仍然会输出 reasoning/thinking 内容（响应里有 `reasoning` 字段）。如果 Agent 不需要思考过程，客户端可以在请求中传 `chat_template_kwargs: {"enable_thinking": false}`，或在启动时设 `DEFAULT_CHAT_TEMPLATE_KWARGS='{"enable_thinking":false}'`。
2. 当前服务监听 `0.0.0.0:8000`，若只需要本地访问，可设 `SERVICE_SCOPE=local`。
3. 重启后首次请求仍可能有短暂编译/缓存预热，之后恢复正常速度。

## 服务状态

- OpenAI API：`http://30.19.40.129:8000/v1`
- 模型名：`ThinkingCap-Qwen3.6-27B-AWQ-tqk8v4-256K`
- 最大上下文：262144 tokens
- 工具调用：已启用
- 3080 训练：未受影响
