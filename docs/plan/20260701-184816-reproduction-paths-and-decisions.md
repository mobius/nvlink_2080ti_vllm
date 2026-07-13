# 计划文档：复现路径与决策点

- 迭代时间：2026-07-01 18:48:16 CST
- 阶段：plan（第 1 轮）

---

## 1. 当前状态

- 环境评估完成（见 research 文档）：硬件匹配，但指南核心插件与模型均为付费私有资源，本机缺失。
- 因此在提交任何安装/下载动作前，需用户在两条路径中选择。

## 2. 决策点 D1：走哪条复现路径？

### 路径 P1 —— 逐字复现原指南（需用户提供付费资源）
前置条件（缺一不可）：
1. 用户提供 `caovan-vllm-sm75-turbo3-v0.4.33-external-plugin.zip`（付费插件）。
2. 用户提供 `Qwen3.6-27B-AWQ-INT4` 模型权重（或可下载来源）。

若满足：
- 用 uv/conda 建隔离环境（Python 3.11）。
- 装 PyTorch + vLLM + 插件 wheel。
- 跑 `caovan-sm75-doctor` → 按指南启动 → 验证 API。
- 逐轮把 doctor 输出、启动日志、吞吐验证记录进 impl 文档。

风险：版本号(PyTorch 2.11 / vLLM 0.21)可能与真实公开发布不符，需按插件实际要求对齐；FP8/256K 配置在 2080Ti 上仍需实测。

### 路径 P2 —— 等价开源复现（不依赖付费插件，推荐先做可行性验证）
在同样 2×2080Ti + NVLink 上，用**公开模型 + 原版 vLLM** 跑 TP=2 OpenAI 兼容服务。
- 环境：`uv venv`（项目内隔离）。
- 模型候选（需选一个 SM75 上能跑、显存能容纳的）：
  - 例如 Qwen2.5-14B/32B 的 AWQ/GPTQ INT4 量化版，或 Qwen3 系列可得的量化版。
  - 27B@INT4 + 长上下文能否在 44G 内放下需实算 KV cache 预算。
- 已知 SM75 难点（需在 impl 中逐一验证/降级）：
  - AWQ Marlin kernel 常要求 SM80+ → 可能需非 Marlin 路径或改量化格式。
  - FP8 KV cache 不被 SM75 原生支持 → 退回 fp16/auto KV。
  - 新版 vLLM 对 Turing 支持逐步收紧 → 可能需选定某个兼容 vLLM 版本。
- 交付：一个真实可用的 `/v1/chat/completions` 服务，附吞吐实测。

### 路径 P3 —— 仅做环境/可行性论证，不实际拉起服务
若用户此刻只想要「判断环境能否支持 + 文档」，则到此为止，等资源到位再继续。

## 3. 推荐

- 若用户能拿到付费插件+模型 → P1。
- 否则 → **P2**（用公开资源做等价复现，最能体现"在这套硬件上把大模型服务跑起来"的真实目标）。
- P2 的第一步一定是**可行性验证**：先确认 vLLM 版本 × SM75 × 量化 kernel × 显存预算这条链路能通，再选定具体模型。

## 4. 环境隔离方案（三路径通用）

- 首选 uv：`cd /mnt/hdd_storage/vllm_2080ti && uv venv --python 3.11 .venv`，之后 `uv pip install ...`。
- 强制 `CUDA_VISIBLE_DEVICES=0,1`，绝不触碰 GPU2/3 上的训练任务。
- 所有缓存(TORCHINDUCTOR/TRITON/HF)指向项目目录或用户 cache，不写系统全局。

## 5. 待用户确认后展开的下一轮

- 选定路径后，新建 `docs/plan/<ts>-<path>-detailed-steps.md` 细化步骤，并开始 `docs/impl/<ts>-...` 记录实际执行与日志。
