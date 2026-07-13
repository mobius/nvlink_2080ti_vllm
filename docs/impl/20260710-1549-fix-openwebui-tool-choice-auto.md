# 解决 Open WebUI `tool_choice: auto` 报错

**时间**: 2026-07-10 15:49  
**问题**: 通过 Open WebUI 调用模型时，vLLM 返回错误：

```
"auto" tool choice requires --enable-auto-tool-choice and --tool-call-parser to be set
```

**原因**: Open WebUI 默认会在请求中携带 `tool_choice: "auto"` 和 `tools` 列表，但 vLLM 服务端必须显式开启自动工具选择并指定一个工具调用解析器（tool-call-parser）才能处理这种请求。

**状态**: ✅ 已解决

---

## 1. 问题复现

通过 Open WebUI 发起普通聊天时，如果前端启用了工具/函数调用相关功能，请求会包含：

```json
{
  "tool_choice": "auto",
  "tools": [...]
}
```

vLLM 在未配置 `--enable-auto-tool-choice` 和 `--tool-call-parser` 时直接拒绝：

```json
{
  "detail": "\"auto\" tool choice requires --enable-auto-tool-choice and --tool-call-parser to be set"
}
```

---

## 2. 解决方案

在 `serve_fast_tqk8v4_web.sh` 中新增以下环境变量，传入 weicj launcher：

```bash
ENABLE_AUTO_TOOL_CHOICE=1
TOOL_CALL_PARSER=qwen3_xml
VLLM_ENFORCE_STRICT_TOOL_CALLING=1
```

weicj launcher 会把它们转成 vLLM 启动参数：

```
--tool-call-parser qwen3_xml --enable-auto-tool-choice
```

同时日志中显示：

```
Strict tool calling: VLLM_ENFORCE_STRICT_TOOL_CALLING=1
```

### 2.1 修改后的启动脚本关键片段

```bash
REASONING_PARSER="${REASONING_PARSER:-off}"
DEFAULT_CHAT_TEMPLATE_KWARGS="${DEFAULT_CHAT_TEMPLATE_KWARGS:-{\"enable_thinking\":false}}"
# 工具调用：开启 auto tool choice，让前端 tool_choice: auto 正常工作
ENABLE_AUTO_TOOL_CHOICE="${ENABLE_AUTO_TOOL_CHOICE:-1}"
TOOL_CALL_PARSER="${TOOL_CALL_PARSER:-qwen3_xml}"
VLLM_ENFORCE_STRICT_TOOL_CALLING="${VLLM_ENFORCE_STRICT_TOOL_CALLING:-1}"
```

并在 launcher 调用处导出：

```bash
MODE=fast PORT="${PORT}" SERVICE_SCOPE="${SERVICE_SCOPE}" \
REASONING_PARSER="${REASONING_PARSER}" \
DEFAULT_CHAT_TEMPLATE_KWARGS="${DEFAULT_CHAT_TEMPLATE_KWARGS}" \
ENABLE_AUTO_TOOL_CHOICE="${ENABLE_AUTO_TOOL_CHOICE}" \
TOOL_CALL_PARSER="${TOOL_CALL_PARSER}" \
VLLM_ENFORCE_STRICT_TOOL_CALLING="${VLLM_ENFORCE_STRICT_TOOL_CALLING}" \
./launcher.sh --non-interactive > "${LOG_DIR}/${LABEL}-launch.log" 2>&1 &
```

---

## 3. 验证

### 3.1 直接调用 vLLM（带 tool_choice: auto）

```bash
curl -s -X POST http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen27b-int4-tqk8v4-256K-mtp3-text-only-cu128",
    "messages": [{"role": "user", "content": "你好，测试一下"}],
    "max_tokens": 30,
    "tool_choice": "auto",
    "tools": [{"type": "function", "function": {"name": "get_weather", "description": "Get weather", "parameters": {"type": "object", "properties": {"location": {"type": "string"}}, "required": ["location"]}}}]
  }'
```

返回正常，无错误：

```json
{
  "choices": [{
    "message": {
      "role": "assistant",
      "content": "你好！有什么我可以帮你的吗？",
      "tool_calls": []
    }
  }]
}
```

### 3.2 通过 Open WebUI 调用

```bash
curl -s -X POST http://<YOUR_SERVER_IP>:18080/openai/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <token>" \
  -d '{
    "model": "qwen27b-int4-tqk8v4-256K-mtp3-text-only-cu128",
    "messages": [{"role": "user", "content": "你好，现在工具调用好了吗？"}],
    "max_tokens": 50
  }'
```

返回正常回答，不再报错。

---

## 4. 相关参数说明

- **ENABLE_AUTO_TOOL_CHOICE=1**：开启 vLLM 的自动工具选择，允许请求中使用 `tool_choice: "auto"`。
- **TOOL_CALL_PARSER=qwen3_xml**：指定工具调用解析器。`qwen3_xml` 是 weicj fork 针对 Qwen3 系列模型推荐的 XML 格式工具调用解析器。
- **VLLM_ENFORCE_STRICT_TOOL_CALLING=1**：强制模型输出严格符合工具调用格式，减少无效/格式错误的工具调用响应。

---

## 5. 注意事项

- 如果不需要工具调用功能，也可以在 Open WebUI 前端关闭工具/Functions，这样请求中不会带 `tool_choice: auto`。
- 开启工具调用后，prompt token 数会略有增加（因为要在 system prompt 中注入工具描述）。
- 当前模型是否真正会触发工具调用取决于具体问题和工具描述；测试用例中模型没有触发工具调用，属于正常行为。

---

## 6. 相关文件

- 启动脚本：`/mnt/hdd_storage/vllm_2080ti/serve_fast_tqk8v4_web.sh`
- 本次启动日志：`/mnt/hdd_storage/vllm_2080ti/logs/web-server-20260710-154514-launch.log`
- 前端地址：`http://<YOUR_SERVER_IP>:18080`
- 后端 API：`http://<YOUR_SERVER_IP>:8000/v1`
