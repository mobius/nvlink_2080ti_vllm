# 2026-07-02 full eval 一键脚本迭代：关闭 thinking 输出并验证 needle 召回

时间戳：2026-07-02 17:22 CST  
标签：`full-eval-agent-test-20260702-v4-retry1`  
文档类型：impl

---

## 1. 背景

上一版 `run_full_eval.sh`（标签 `full-eval-agent-test-20260702-v2`）虽然能跑通性能+质量全流程，但质量评估结果异常：

- 所有 chat 回答都是模型的 `<think>` 思考过程（如 "Here's a thinking process..."）。
- `content` 字段为空，脚本被迫 fallback 到 `reasoning_content`。
- 大海捞针（needle-in-haystack）返回的是思考文本，没有提取出 needle，`needle_found=False`。

根本原因：**服务默认启用了 Qwen3 reasoning parser，且模型默认开启 thinking 输出**。只把 parser 关掉并不能阻止模型生成 thinking token，还需要通过 chat template kwargs 显式关闭 thinking。

---

## 2. 本次改动

### 2.1 `run_full_eval.sh`

在启动服务的环境变量中新增：

```bash
REASONING_PARSER=off \
DEFAULT_CHAT_TEMPLATE_KWARGS='{"enable_thinking":false}' \
```

这样服务同时满足：

1. **不解析 reasoning token**：vLLM 不会把 thinking 内容拆到单独的 `reasoning_content` 字段。
2. **模型不生成 thinking**：通过 Qwen3 的 chat template 参数 `enable_thinking=false`，让模型直接给出最终答案。

### 2.2 `run_quality_eval.sh`

- 修复了一个 typo：`>/devdev/null` → `>/dev/null`。
- 服务未就绪的提示里补充了 `REASONING_PARSER=off`，提醒用户若要做质量评估，启动服务时应关闭 reasoning parser/思考输出。

---

## 3. 启动过程中的偶发问题

### 3.1 现象

第一次启动（`agent-test-20260702-v4`）失败，worker 报错：

```text
ValueError: Free memory on device cuda:1 (0.98/21.48 GiB) on startup is
less than desired GPU memory utilization (0.9, 19.34 GiB).
```

但此时 `nvidia-smi` 显示 GPU 0/1 几乎空闲（仅 Xorg 占用 4 MiB）。

### 3.2 处理

- 确认无残留 vLLM / api_server / EngineCore / Worker 进程。
- 未修改任何配置，直接重跑一次（`agent-test-20260702-v4-retry1`），服务正常启动。
- 判断为偶发的显存状态/上下文释放延迟问题，非配置错误。

---

## 4. 评估结果

### 4.1 性能（PP4096 / TG128）

| 指标 | 数值 |
|------|------|
| TTFT | 2.640 s |
| Prefill | 1551.5 tok/s |
| Decode | 86.06 tok/s |
| Elapsed | 4.127 s |

与之前 `v3-retry1` 结果（decode 85.77 tok/s）一致，说明关闭 thinking 不影响吞吐。

### 4.2 质量

| 任务 | 结果 | 说明 |
|------|------|------|
| chinese_qa | 光合作用是植物利用光能，将二氧化碳和水转化为有机物并释放氧气的过程。 | 34 字，符合 50 字以内要求 |
| english_qa | The capital of France is Paris. | 正确 |
| math_reasoning | 给出 60 km/h × 5 h = 300 km 的完整推导 | 正确（在 256 tokens 处截断） |
| code_generation | 给出无切片 reverse_string 函数及 docstring | 正确 |
| needle_in_haystack | **K7P-2080Ti-NVLink** | `needle_found=True`，召回成功 |

---

## 5. 结论

- `run_full_eval.sh` 现在可以一键完成「性能 + 质量」完整评估。
- 关闭 `REASONING_PARSER` 和 `enable_thinking=false` 后，质量评估结果正常，大海捞针召回成功。
- 当前 AWQ-INT4 + fast/tqk8v4-256K-mtp3 配置下，PP4096/TG128 decode 吞吐稳定在 **~86 tok/s**（低于 weicj 宣称的 GPTQ-INT4 峰值 ~101 tok/s，差距主要来自模型格式）。

---

## 6. 相关文件

- 脚本：`/mnt/hdd_storage/vllm_2080ti/run_full_eval.sh`
- 脚本：`/mnt/hdd_storage/vllm_2080ti/run_quality_eval.sh`
- 性能日志：`/mnt/hdd_storage/vllm_2080ti/logs/full-eval-agent-test-20260702-v4-retry1-bench.jsonl`
- 质量日志：`/mnt/hdd_storage/vllm_2080ti/logs/full-eval-agent-test-20260702-v4-retry1-quality.jsonl`
- 启动日志：`/mnt/hdd_storage/vllm_2080ti/logs/full-eval-agent-test-20260702-v4-retry1-launch.log`
- GPU 日志：`/mnt/hdd_storage/vllm_2080ti/logs/full-eval-agent-test-20260702-v4-retry1-gpu.log`
