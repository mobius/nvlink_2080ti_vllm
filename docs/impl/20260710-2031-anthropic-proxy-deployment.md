# Anthropic Messages API 代理部署说明

**时间**: 2026-07-10 20:31  
**目标**: 为只支持 Anthropic/Claude API 格式的 agent 提供兼容层，把本地 vLLM 的 OpenAI API 包装成 `/v1/messages`。  
**状态**: ✅ 已部署并验证

---

## 1. 部署了什么

- 一个 FastAPI 代理：`/mnt/hdd_storage/vllm_2080ti/tools/anthropic_proxy.py`
- 一键启动脚本：`/mnt/hdd_storage/vllm_2080ti/serve_anthropic_proxy.sh`
- 隔离 Python 环境：`.venv-anthropic-proxy`
- 监听端口：**18081**

代理会把 Anthropic 的 `POST /v1/messages` 请求转成 vLLM 的 `POST /v1/chat/completions` 请求，并把 OpenAI 响应再转回 Anthropic Messages API 格式。

---

## 2. Agent 配置

| 配置项 | 值 |
|--------|-----|
| base_url | `http://<YOUR_SERVER_IP>:18081` |
| api_key | 任意非空字符串，例如 `sk-test` |
| model | `qwen27b-int4-tqk8v4-256K-mtp3-text-only-cu128` |
| max_tokens | 最大 262144 上下文，单次生成按需设置 |

> 当前未设置 `PROXY_API_KEY`，所以代理本身不强制鉴权。如需鉴权，在启动脚本里设置 `PROXY_API_KEY=xxx` 即可。

---

## 3. 支持的 Anthropic API 特性

| 特性 | 支持情况 |
|------|----------|
| `messages`（text content） | ✅ |
| `system` prompt | ✅ |
| `max_tokens` | ✅ |
| `temperature` | ✅ |
| `top_p` | ✅ |
| `stop_sequences` | ✅ |
| `stream`（SSE） | ✅ |
| `tools` / `tool_choice: auto` | ✅ |
| `tool_result` / `tool_use` 多轮 | ✅ |
| `top_k` | 透传但 vLLM 可能忽略 |
| `thinking` | ❌ 不处理，会被忽略 |
| `image` / 多模态 | ❌ 当前模型为 text-only |

---

## 4. 验证结果

### 4.1 基础对话

```python
import anthropic
client = anthropic.Anthropic(base_url="http://<YOUR_SERVER_IP>:18081", api_key="sk-test")
msg = client.messages.create(
    model="qwen27b-int4-tqk8v4-256K-mtp3-text-only-cu128",
    max_tokens=50,
    temperature=0.0,
    messages=[{"role": "user", "content": "你好，用 anthropic SDK 测试"}],
)
print(msg.content[0].text)
```

输出正常。

### 4.2 工具调用

```python
msg = client.messages.create(
    model="qwen27b-int4-tqk8v4-256K-mtp3-text-only-cu128",
    max_tokens=100,
    temperature=0.0,
    tools=[{
        "name": "get_weather",
        "description": "查询指定城市天气",
        "input_schema": {
            "type": "object",
            "properties": {"location": {"type": "string"}},
            "required": ["location"],
        },
    }],
    tool_choice={"type": "auto"},
    messages=[{"role": "user", "content": "北京今天天气怎么样？"}],
)
print(msg)
```

返回 `ToolUseBlock`，`stop_reason: tool_use`。

### 4.3 流式输出

```python
with client.messages.stream(
    model="qwen27b-int4-tqk8v4-256K-mtp3-text-only-cu128",
    max_tokens=30,
    temperature=0.0,
    messages=[{"role": "user", "content": "你好，stream 测试"}],
) as stream:
    for text in stream.text_stream:
        print(text, end="", flush=True)
```

流式事件格式符合 Anthropic SSE 规范。

### 4.4 tool_result 多轮

```python
msg = client.messages.create(
    model="qwen27b-int4-tqk8v4-256K-mtp3-text-only-cu128",
    max_tokens=50,
    temperature=0.0,
    messages=[
        {"role": "user", "content": "北京天气？"},
        {"role": "assistant", "content": [
            {"type": "tool_use", "id": "tool-1", "name": "get_weather", "input": {"location": "北京"}}
        ]},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "tool-1", "content": "晴天，25度"}
        ]},
    ],
    tools=[...],
)
print(msg.content[0].text)
```

输出：`北京今天天气晴朗，气温25度。`

---

## 5. 启动 / 停止

```bash
# 启动
bash serve_anthropic_proxy.sh

# 停止
bash serve_anthropic_proxy.sh stop
```

---

## 6. 相关文件

- 代理源码：`/mnt/hdd_storage/vllm_2080ti/tools/anthropic_proxy.py`
- 启动脚本：`/mnt/hdd_storage/vllm_2080ti/serve_anthropic_proxy.sh`
- 运行日志：`/mnt/hdd_storage/vllm_2080ti/logs/anthropic-proxy-server.log`
- 后端 vLLM：`http://<YOUR_SERVER_IP>:8000/v1`
- 前端 Open WebUI：`http://<YOUR_SERVER_IP>:18080`

---

## 7. 注意事项

- 代理只做了格式转换，**模型本身还是 Qwen，不是 Claude**。如果 agent 硬编码了 Claude 特有行为（比如期望特定拒绝话术），需要自行调整。
- Anthropic SDK 某些字段（如 `thinking`）会被代理忽略。
- 当前未开启代理层鉴权；如需暴露到公网，建议设置 `PROXY_API_KEY`。
- 并发能力受限于后端 vLLM 的 `max-num-seqs=1` 配置。

---
