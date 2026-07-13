# 启用 vLLM reasoning/thinking 输出并透传给 Anthropic 代理

时间：2026-07-10 21:14 CST  
作者：Kimi Code CLI  
关联提交：tools/anthropic_proxy.py、serve_fast_tqk8v4_web.sh

---

## 1. 目标

让已部署的 `qwen27b-int4-tqk8v4-256K-mtp3-text-only-cu128` 模型：

1. 在 vLLM 层输出 thinking/reasoning 内容；
2. 通过 OpenAI API 以 `reasoning_content` 字段暴露；
3. 通过 Anthropic 兼容代理以 `thinking` content block 暴露，供只支持 Anthropic API 的 agent 使用。

---

## 2. 改动内容

### 2.1 vLLM 启动脚本 `serve_fast_tqk8v4_web.sh`

将默认参数改为启用 thinking 生成并解析：

```bash
REASONING_PARSER="${REASONING_PARSER:-qwen3}"
DEFAULT_CHAT_TEMPLATE_KWARGS="${DEFAULT_CHAT_TEMPLATE_KWARGS:-{\"enable_thinking\":true}}"
```

- `REASONING_PARSER=qwen3`：vLLM 把 Qwen3 的 `<think>...</think>` 内容从 `content` 拆到 `reasoning_content`。
- `DEFAULT_CHAT_TEMPLATE_KWARGS={"enable_thinking":true}`：chat template 在构造 prompt 时保留 thinking 指令，让模型实际生成思考过程。

其余参数保持不变（tool calling、MTP3、TurboQuant KV、CUDA graph 等）。

### 2.2 Anthropic 代理 `tools/anthropic_proxy.py`

新增对 reasoning 内容的透传：

#### 非流式响应

从 OpenAI `message.reasoning_content` 映射到 Anthropic `content` 数组的第一个 `thinking` block：

```python
reasoning = msg.get("reasoning_content") or ""
if reasoning:
    content.append({"type": "thinking", "thinking": reasoning})
```

随后追加普通 `text` block 和 `tool_use` block。

#### 流式响应

从 OpenAI delta `reasoning_content` 映射到 Anthropic 的 `thinking_delta`：

```python
reasoning_delta = delta.get("reasoning_content") or ""
if reasoning_delta:
    if not started:
        yield content_block_start(type="thinking")
        started = True
    yield content_block_delta(type="thinking_delta", thinking=reasoning_delta)
```

这样 Anthropic SDK / 客户端会收到与普通 Claude 一致的 `thinking` 流事件。

---

## 3. 启动与验证

### 3.1 重启 vLLM

```bash
bash /mnt/hdd_storage/vllm_2080ti/serve_fast_tqk8v4_web.sh stop
nohup bash /mnt/hdd_storage/vllm_2080ti/serve_fast_tqk8v4_web.sh \
  > /mnt/hdd_storage/vllm_2080ti/logs/web-server-restart.log 2>&1 &
```

服务在约 4 分半后就绪（`curl http://127.0.0.1:8000/health` 返回 200）。

### 3.2 OpenAI API 验证

请求：

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model":"qwen27b-int4-tqk8v4-256K-mtp3-text-only-cu128",
    "messages":[{"role":"user","content":"1+1=?"}],
    "max_tokens":128,
    "temperature":0.0
  }'
```

响应中 `choices[0].message` 同时包含：

- `reasoning_content`: 模型的思考过程；
- `content`: 最终答案（如 `"2"`）。

### 3.3 Anthropic API 代理验证

重启代理：

```bash
bash /mnt/hdd_storage/vllm_2080ti/serve_anthropic_proxy.sh stop
nohup bash /mnt/hdd_storage/vllm_2080ti/serve_anthropic_proxy.sh \
  > /mnt/hdd_storage/vllm_2080ti/logs/anthropic-proxy-restart.log 2>&1 &
```

非流式请求：

```bash
curl http://127.0.0.1:18081/v1/messages \
  -H 'Content-Type: application/json' \
  -H 'anthropic-version: 2023-06-01' \
  -d '{
    "model": "qwen27b-int4-tqk8v4-256K-mtp3-text-only-cu128",
    "max_tokens": 128,
    "messages": [{"role": "user", "content": "1+1=?"}]
  }'
```

响应 `content` 数组：

```json
[
  {"type": "thinking", "thinking": "Thinking Process:\n1. Analyze..."},
  {"type": "text", "text": "2"}
]
```

流式请求同样先收到 `thinking_delta` 事件，再收到 `text_delta` 事件。

---

## 4. 接入 agent 的配置建议

若 agent 只支持 Anthropic API，指向代理即可：

```yaml
base_url: http://<YOUR_SERVER_IP>:18081
api_key: sk-anything          # 未设置 PROXY_API_KEY 时不校验，可任意填
model: qwen27b-int4-tqk8v4-256K-mtp3-text-only-cu128
max_tokens: 262144            # 模型最大上下文
```

针对你给出的配置片段，修正如下：

```json
{
  "openai_compatible": {
    "qwen3.6-27b": {
      "api_url": "http://<YOUR_SERVER_IP>:18081/v1",
      "available_models": [
        {
          "name": "qwen27b-int4-tqk8v4-256K-mtp3-text-only-cu128",
          "max_tokens": 262144,
          "max_output_tokens": 32000,
          "max_completion_tokens": 200000,
          "reasoning_effort": "medium",
          "capabilities": {
            "tools": true,
            "images": false,
            "parallel_tool_calls": false,
            "prompt_cache_key": false,
            "chat_completions": true,
            "interleaved_reasoning": true,
            "max_tokens_parameter": false
          }
        }
      ]
    }
  }
}
```

注意：

- `name` 字段前有一个多余空格，已去掉；
- `reasoning_effort` 是 OpenAI/客户端概念，当前代理层暂未把它转发成 vLLM 的 `enable_thinking` 开关。因为 vLLM 启动时已固定 `enable_thinking=true`，模型默认就会输出 thinking。后续如需按请求关闭 thinking，再扩展代理；
- `interleaved_reasoning: true` 可保留，表示客户端能处理 thinking 与 text 交错出现的内容块。

---

## 5. 遇到的问题

### 5.1 启动 vLLM 时偶发 `Free memory on device cuda:0 (0.47/21.48 GiB)`

现象：第一次重启时 worker init 阶段报显存不足，但 `nvidia-smi` 显示 2080Ti 几乎空闲。

处理：直接 `stop` 后重新 `nohup` 启动，第二次成功。推测是前一次异常退出时某段 GPU 内存/上下文未及时释放，或 NCCL/cudaMemGetInfo 在瞬时状态读到了被占用的值。后续若再出现，可先确认无残留 vLLM 进程再重试。

---

## 6. 结论

- vLLM 已启用 thinking 输出；
- OpenAI API 可用 `reasoning_content` 拿到思考过程；
- Anthropic 代理已升级，可像 Claude 一样返回 `thinking` content block；
- agent 可直接用 `http://<YOUR_SERVER_IP>:18081` 作为 Anthropic 兼容端点。

---

## 7. 后续可选项

1. 在 Anthropic 代理中支持 `thinking: {type: "enabled", budget_tokens: N}` 参数，按请求动态关闭/限制 thinking；
2. 在 Open WebUI 中测试 reasoning 输出是否正常显示；
3. 评估启用 thinking 后对 PP4096/TG128 峰值吞吐的影响。
