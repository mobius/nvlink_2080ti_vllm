# Agent 接入配置说明

**时间**: 2026-07-10 20:24  
**目标**: 把当前部署的 Qwen27B 模型通过 OpenAI 兼容 API 接入外部 agent/客户端。  
**状态**: ✅ 已生成 API key 并验证可用

---

## 1. 可用端点

### 方案 A：直接连 vLLM（推荐，最轻量）

| 项目 | 值 |
|------|-----|
| Base URL | `http://<YOUR_SERVER_IP>:8000/v1` |
| API Key | 任意非空字符串即可，例如 `sk-vllm` |
| Model | `qwen27b-int4-tqk8v4-256K-mtp3-text-only-cu128` |
| Max context | 262144 tokens（256K） |

### 方案 B：经 Open WebUI 中转

| 项目 | 值 |
|------|-----|
| Base URL | `http://<YOUR_SERVER_IP>:18080/openai` |
| API Key | `sk-9003d86ef5f44f83acba0136f76c9eb5`（已通过 admin 账号生成） |
| Model | `qwen27b-int4-tqk8v4-256K-mtp3-text-only-cu128` |
| 登录 Web | `http://<YOUR_SERVER_IP>:18080`（账号 admin@example.com / <ADMIN_PASSWORD>） |

> 推荐 **方案 A**，少一层转发、延迟更低、稳定性更好。方案 B 适合需要用到 Open WebUI 的用户管理、聊天记录、RAG 等能力时再用。

---

## 2. 关键参数建议

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| `temperature` | 0.0 - 0.7 | 0.0 适合确定性任务；创意任务可提高到 0.7 |
| `max_tokens` | 按需，最大 8192 | 长回答可适当增大，但会占用更多显存/时间 |
| `top_p` | 1.0 或 0.9 |  nucleus sampling，配合 temperature 使用 |
| `stream` | true / false | Agent 聊天建议 `true` 以获得流式响应 |
| `tool_choice` | `auto` / `none` | 已开启 `--enable-auto-tool-choice`，支持 `auto` |

> 当前已关闭模型 thinking 输出（`enable_thinking=false`），模型会直接给出最终答案。

---

## 3. 示例代码

### Python (openai SDK)

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://<YOUR_SERVER_IP>:8000/v1",  # 或 Open WebUI: http://<YOUR_SERVER_IP>:18080/openai
    api_key="sk-vllm",  # 直接连 vLLM 可任意填；连 Open WebUI 填真实 API key
)

response = client.chat.completions.create(
    model="qwen27b-int4-tqk8v4-256K-mtp3-text-only-cu128",
    messages=[
        {"role": "system", "content": "你是一个有帮助的助手。"},
        {"role": "user", "content": "请介绍一下自己"},
    ],
    temperature=0.0,
    max_tokens=512,
    stream=False,
)

print(response.choices[0].message.content)
```

### cURL

```bash
curl -s -X POST http://<YOUR_SERVER_IP>:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-vllm" \
  -d '{
    "model": "qwen27b-int4-tqk8v4-256K-mtp3-text-only-cu128",
    "messages": [
      {"role": "user", "content": "请介绍一下自己"}
    ],
    "temperature": 0.0,
    "max_tokens": 512
  }'
```

### LangChain

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    base_url="http://<YOUR_SERVER_IP>:8000/v1",
    api_key="sk-vllm",
    model="qwen27b-int4-tqk8v4-256K-mtp3-text-only-cu128",
    temperature=0.0,
    max_tokens=512,
)

result = llm.invoke("请介绍一下自己")
print(result.content)
```

---

## 4. 工具调用示例

因为已开启 `--enable-auto-tool-choice`，可以像 OpenAI 一样传入 `tools` 和 `tool_choice`。

```python
response = client.chat.completions.create(
    model="qwen27b-int4-tqk8v4-256K-mtp3-text-only-cu128",
    messages=[{"role": "user", "content": "北京今天天气怎么样？"}],
    tools=[{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "查询指定城市的天气",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "城市名"}
                },
                "required": ["location"]
            }
        }
    }],
    tool_choice="auto",
)

print(response.choices[0].message)
```

> 注意：模型是否真正触发工具调用取决于问题本身和工具描述质量。简单问题模型可能直接文本回答。

---

## 5. 长上下文使用建议

- 最大窗口 **262144 tokens**，约 20-40 万汉字（取决于 tokenization）。
- 长文本一次性输入时，prefill 阶段会较慢，但 decode 阶段可正常生成。
- 如需测试长文本召回，可运行 `bash run_quality_eval.sh` 中的 needle-in-a-haystack 任务。
- 不建议单次对话把上下文压到极限，留 10-20% 余量给生成输出更稳。

---

## 6. 服务状态检查

```bash
# vLLM 健康检查
curl http://<YOUR_SERVER_IP>:8000/health

# 模型列表
curl http://<YOUR_SERVER_IP>:8000/v1/models

# Open WebUI 健康检查
curl http://<YOUR_SERVER_IP>:18080/
```

---

## 7. 已知限制

- 当前只部署了文本模型，不支持图像/多模态。
- `max_tokens` 设置过大会显著增加首 token 等待时间。
- 并发能力有限：当前 profile 为 `max-num-seqs=1`，更适合单用户/agent 低延迟场景，不是高并发服务。
- 如需更高并发，可切换 `normal` mode 或调整 `max-num-seqs`/`max-num-batched-tokens`。

---

## 8. 相关文件

- 后端启动脚本：`/mnt/hdd_storage/vllm_2080ti/serve_fast_tqk8v4_web.sh`
- 前端启动脚本：`/mnt/hdd_storage/vllm_2080ti/serve_open_webui.sh`
- Open WebUI API Key：`sk-9003d86ef5f44f83acba0136f76c9eb5`
- 后端日志：`/mnt/hdd_storage/vllm_2080ti/logs/web-server-20260710-154514-launch.log`
- 前端日志：`/mnt/hdd_storage/vllm_2080ti/logs/open-webui-server.log`

---
