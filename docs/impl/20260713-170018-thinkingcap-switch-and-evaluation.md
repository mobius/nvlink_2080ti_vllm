# ThinkingCap-Qwen3.6-27B-AWQ 切换与评估

**时间**: 2026-07-13 16:00–17:00

## 目标

将服务从原模型 `Qwen3.6-27B-AWQ-INT4` 切换到 `sahilchachra/ThinkingCap-Qwen3.6-27B-AWQ`，并在相同 fast/tqk8v4-256K 配置下完成性能与质量评估。

## 实施步骤

1. **停止旧服务**：只 kill 占用 GPU0/1 的 `vllm.entrypoints.openai.api_server`，确认 GPU2/3 训练任务不受影响。
2. **用 weicj launcher 启动新模型**：
   - `MODEL_DIR=/mnt/hdd_storage/vllm_2080ti/models/ThinkingCap-Qwen3.6-27B-AWQ`
   - `SERVED_NAME=ThinkingCap-Qwen3.6-27B-AWQ-tqk8v4-256K-mtp3`
   - `PROFILE=qwen27b/fast/int4/tqk8v4-256K-mtp3-text-only.env`
   - `MODE=fast`, `MTP_K=3`（默认）
3. **验证模型健康**：`curl /health` OK，服务绑定 `127.0.0.1:8000`。

## 关键发现：ThinkingCap 没有保留 MTP 头

- 检查 safetensors key：ThinkingCap 共 1696 个 key，**0 个 mtp 相关 key**。
- 对比旧模型 `Qwen3.6-27B-AWQ-INT4`：共 2420 个 key，含 36 个 mtp key（`mtp.layers.*`）。
- 结果：启用 `MTP_K=3` 时 vLLM 无法真正做 MTP 投机解码，反而因投机框架 overhead 导致 decode 吞吐暴跌。

## 性能对比（PP4096/TG128）

| 模型 | MTP | TTFT | Prefill | Decode | 说明 |
|---|---|---|---|---|---|
| Qwen3.6-27B-AWQ-INT4 | K=3 | 2.6–3.1 s | 1304–1570 tok/s | **90 tok/s** | 原版，含 MTP 头 |
| ThinkingCap-AWQ | K=3 | 3.4 s | 1191 tok/s | **19.4 tok/s** | 无 MTP 头，投机框架拖慢 |
| ThinkingCap-AWQ | K=0 | 1.3–3.3 s | 1209–1244 tok/s | **32 tok/s** | 关闭投机解码后的真实速度 |

- ThinkingCap 的真实 decode 速度约为旧模型的 **35%**。
- 无 MTP 时 chat-text benchmark 工具计数异常（completion_tokens=0），手动 OpenAI SDK 流式测试稳定得到 32–33 tok/s。

## 质量评估（关闭 thinking）

通过 `chat_template_kwargs={"enable_thinking": false}` 关闭 thinking 后跑质量评估：

| 类别 | 结果 | 备注 |
|---|---|---|
| 中文常识 | 正确 | 光合作用解释准确 |
| 英文常识 | 正确 | 法国首都是巴黎 |
| 数学推理 | 正确 | 火车问题给出完整推导 |
| 代码生成 | 正确 | 生成无切片反转字符串函数 |
| 大海捞针 | **needle_found=True** | 从约 11K token 上下文中准确提取代码 |

不关闭 thinking 时，所有输出都是 "Here's a thinking process..." 的推理过程，说明关闭 thinking 对 agent/qa 场景是必需的。

## 脚本更新

1. **`run_quality_eval.sh`**：新增 `DISABLE_THINKING` 环境变量（默认 `1`），请求时自动附加 `chat_template_kwargs={"enable_thinking": false}`；若响应只有 reasoning 无 content，则 fallback 到 reasoning 文本。
2. **`run_fast_tqk8v4_bench.sh`**：新增 `MTP_K`、`PROFILE`、`TRITON_CACHE_SUFFIX` 环境变量覆盖，支持无 MTP 模型正确 benchmark。
3. **`README.md`**：更新关键成果与快速开始说明，强调 ThinkingCap 需 `MTP_K=0`。

## 结论

- `ThinkingCap-Qwen3.6-27B-AWQ` 可以在现有 weicj vLLM 栈上跑通，256K 上下文可用，质量 OK。
- 由于模型本身**没有保留 MTP 头**，不能享受投机解码加速；在 fast/tqk8v4 配置下 decode 约 **32 tok/s**。
- 如果追求更高吞吐，应换用保留 MTP 头的 checkpoint（如 `shawnw3i/Qwen3.6-27B-AWQ-MTP` 或 GPTQ-MTP 版本），或对 ThinkingCap BF16 重新量化并保留 MTP。

## 新术语

- **MTP head / MTP 头**：Multi-Token Prediction 所需的额外解码层权重。量化/微调时若被剥离，模型将失去一次预测多个 token 的能力，无法与 vLLM 的 `speculative-config method=mtp` 配合加速。
- **投机解码 overhead**：当模型没有合适 draft 权重但投机框架仍被启用时，vLLM 会反复走 fallback 路径，反而增加调度/内存开销，导致吞吐低于普通解码。
