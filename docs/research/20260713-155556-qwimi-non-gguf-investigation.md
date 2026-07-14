# Qwimi-3.6-27B 非 GGUF 方案调研

**时间**: 2026-07-13 15:55  
**背景**: 用户希望找到 `trjxter/Qwimi-3.6-27B-Coder-MTP-GGUF` 的非 GGUF 版本，以便在现有双 2080Ti (SM75, 22GB ×2, NVLink) vLLM 栈上运行，而不使用 llama.cpp。

---

## 1. 调研方法

1. Hugging Face API 搜索 `Qwimi`、`Qwopus`、`trjxter` 等关键词，过滤 `transformers`/量化标签。
2. 读取候选仓库的 `config.json`，确认 `architectures`、`quantization_config`、`vision_config` 等关键字段。
3. 网络搜索验证模型关系与社区量化版本。

---

## 2. 核心发现

### 2.1 Qwimi 官方非 GGUF 版本存在，但无法直接部署

- **仓库**: `trjxter/Qwimi-3.6-27B-Coder-MTP-BF16`
- **格式**: BF16 safetensors，15 个分片
- **大小**: 约 54 GB（估算）
- **结论**: 双 2080Ti 总显存 44 GB，BF16 模型放不下；必须再量化。
- **额外限制**: `config.json` 含 `vision_config`，说明这是**多模态（vision-language）模型**，即使能放下，也需在 vLLM 中处理视觉模块兼容性。

### 2.2 Qwimi/Qwopus 系列全为多模态

对以下仓库逐一核对 `config.json` 中的 `vision_config`：

| 仓库 | 类型 | 是否多模态 | 量化方式 | 备注 |
|---|---|---|---|---|
| `trjxter/Qwimi-3.6-27B-Coder-MTP-BF16` | 官方非 GGUF | 是 | BF16 | 54 GB，放不下 |
| `mconcat/Qwopus3.6-27B-v2-AWQ-4bit` | 社区 AWQ | 是 | AWQ INT4 | 含 `model.visual.*` 与 MTP |
| `cpatonn/Qwopus3.6-27B-Coder-AWQ-INT4` | 社区 AWQ | 是 | compressed-tensors | 5 分片 |
| `XReyRobert/Qwopus3.6-27B-Coder-GPTQ-Pro` | 社区 GPTQ | 是 | GPTQ | 含 `model-mtp-aware-gptq.safetensors` |
| `Jackrong/Qwopus3.6-27B-Coder` | 基础 SFT | 是 | BF16 | 放不下 |

说明：**Qwimi/Qwopus 整条线都是基于 `Qwen3_5ForConditionalGeneration` 的多模态模型**，没有纯文本-only 的 AWQ/GPTQ 社区版本。

### 2.3 与 Qwimi 定位相近的纯文本替代方案

如果目标是「减少 Qwen3.6-27B 过度 thinking + 代码/Agent 能力强」，以下**纯文本 Qwen3.6-27B 量化版**更适合现有 vLLM 栈：

| 仓库 | 格式 | 是否 MTP | 适合 2080Ti | 备注 |
|---|---|---|---|---|
| `sahilchachra/ThinkingCap-Qwen3.6-27B-AWQ` | AWQ INT4 | 否 | 是 | 正在下载，目标模型 |
| `cyankiwi/Qwen3.6-27B-AWQ-INT4` | AWQ INT4 | 否 | 是 | 下载量高 |
| `QuantTrio/Qwen3.6-27B-AWQ` | AWQ INT4 | 否 | 是 | 下载量高 |
| `shawnw3i/Qwen3.6-27B-AWQ-MTP` | AWQ INT4 | 是 | 待测 | 保留 MTP 头 |
| `groxaxo/Qwen3.6-27B-GPTQ-Pro-4bit` | GPTQ INT4 | 否 | 是 | weicj fork 峰值测试基于 GPTQ |
| `llmfan46/Qwen3.6-27B-uncensored-heretic-v2-Native-MTP-Preserved-GPTQ-Int4` | GPTQ INT4 | 是 | 待测 | 保留原生 MTP |

### 2.4 当前下载状态

- `ThinkingCap-Qwen3.6-27B-AWQ/model.safetensors` 已下载约 22 GB / 25 GB，wget PID 仍在运行。

---

## 3. 结论与建议

1. **Qwimi 的精确非 GGUF 版本存在，但不适合 2080Ti**：`trjxter/Qwimi-3.6-27B-Coder-MTP-BF16` 是 BF16 多模态模型，约 54 GB，超出 44 GB 总显存；且没有社区 AWQ/GPTQ 纯文本版本。
2. **社区 AWQ/GPTQ Qwopus 仍为多模态**：`mconcat/Qwopus3.6-27B-v2-AWQ-4bit` 等虽为 INT4，但含视觉模块，在 SM75 vLLM 上兼容性未知，且本次复现栈围绕纯文本服务搭建。
3. **推荐路径**：
   - 继续使用已下载的 `ThinkingCap-Qwen3.6-27B-AWQ`（减少过度 thinking 的 Qwen3.6-27B AWQ）。
   - 若后续想追峰值吞吐，可再测 `shawnw3i/Qwen3.6-27B-AWQ-MTP` 或 `llmfan46/...-Native-MTP-Preserved-GPTQ-Int4` 等纯文本 MTP 量化版。
   - 若确实需要 Qwimi 本身，只能本地对 BF16 做 AWQ/GPTQ 量化（耗时且需校准数据），或接受 GGUF + llama.cpp。

---

## 4. 新术语（同步写入 glossary.md）

- **Qwimi / Qwopus**：基于 Qwen3.6-27B 的社区微调/衍生模型系列，主打代码、Agent、推理能力。二者多为多模态（vision-language）。
- **BF16（bfloat16）**：16-bit 浮点格式，动态范围与 FP32 相同但精度较低。27B 模型 BF16 约需 54 GB 显存，2080Ti 双卡放不下。
- **多模态模型 / VLM（Vision-Language Model）**：同时处理文本与图像输入的模型。`Qwen3_5ForConditionalGeneration` 架构在 Qwimi/Qwopus 中含 `vision_config`，即属此类。
- **compressed-tensors**：一种通用量化权重存储格式，不同于原生 AWQ/GPTQ，可能走 Marlin 等混合精度 kernel 路径。
