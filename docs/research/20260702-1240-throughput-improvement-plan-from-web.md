# 提升方案：2×RTX 2080Ti 22G 上 Qwen3.6-27B-AWQ-INT4 推理加速（网络/文献整理）

- 整理时间：2026-07-02 12:40 CST
- 来源：GitHub weicj/2080Ti-LLM-Toolbox、weicj/vLLM-2080Ti-Definitive、docs/mtp-task-sensitivity.md、公开技术文档及通用 vLLM 调优指南。

---

## 1. 当前实测基线

| 路线 | 配置 | 实测 decode tok/s | 来源/证据 |
|---|---|---|---|
| caovan v0.1.3 插件 | 256K / fp8 KV / MTP=2 | **~40.1** | 本机实测，`_caovan_sm75_*` kernel 生效 |
| weicj Definitive fork | 256K / fp16 KV / MTP=3 / FlashQLA legacy | **~52.0** | 本机实测，FlashQLA legacy kernel 生效 |
| weicj 宣称峰值 | PP4096/TG128 / AWQ / FlashQLA / MTP=3 | **101.3** | GitHub weicj/2080Ti-LLM-Toolbox |

差距说明：峰值 101.3 tok/s 是在 **PP4096(prompt 4K)/TG128(gen 128)**、排除冷启动/模型加载/CUDA graph/Triton JIT 的“warm single-request peak”口径；本机 52 tok/s 是 **端到端 wall-clock**（含 prefill/reasoning/JIT 已排除但仍为长输出 600 tokens）。若严格对齐 PP4096/TG128 或更短输出，应能接近 80-100。

---

## 2. 官方推荐冲峰路线（weicj/2080Ti-LLM-Toolbox）

```
vLLM 0.21.0
torch 2.11 cu130    (本机实际 cu128)
TP=2
AWQ Marlin
FlashInfer/FA2 attention
FlashQLA SM70/SM75 legacy GDN prefill
MTP K=3
```

要点：
- **核心解锁是 FlashQLA**。没有 FlashQLA legacy GDN 时 noMTP decode 在 GPTQ-INT4 上约 43.59 tok/s；启用 FlashQLA 后 MTP3 在 LongGen3 上达 60.62；PP4096/TG128 峰值 101.3。
- **MTP K=3 是 Qwen3.6 的安全甜点**：weicj `mtp-task-sensitivity.md` 显示，Qwen3.6-27B-GPTQ-INT4 LongGen3 4096/1024 中，MTP3 60.62、MTP4 60.77、MTP5 59.55；但接受率从 MTP3 67.9% 降至 MTP5 51.3%。MTP3 是混合工作负载默认。
- **TurboQuant KV 是下一步容量/速度杠杆**：`tq4nc` 可把 256K 的 KV cache 从 FP16 的 ~272K tokens 扩到 **735,084 tokens**，同显存下支持更长上下文，且在 PP4096/TG128 峰值中用到。

## 3. 可直接尝试的提升项（按投入/风险排序）

### 3.1 低投入：换用 fast profile + TurboQuant KV（最可能接近 100 tok/s）

weicj 提供 `fast/int4/tqk8v4-256K-mtp3-text-only.env`：
- `KV_CACHE_DTYPE=turboquant_k8v4`（K 8-bit / V 4-bit 混合精度 KV）
- MTP=3, cudagraph=FULL_AND_PIECEWISE
- 宣称 **100.81 tok/s** decode（PP4096/TG128 峰值）

风险：TurboQuant 是实验压缩格式，可能引入数值漂移；长上下文任务需做质量验收。

### 3.2 低投入：用 GPTQ-INT4 替代 AWQ（weicj 文档中更强的 MTP 基线）

weicj `mtp-task-sensitivity.md` 显示：
- Qwen3.6-27B-GPTQ-INT4 noMTP 43.59 vs AWQ noMTP 约 31-32 tok/s
- GPTQ-INT4 MTP3 60.62 vs AWQ 路线更快

原因：GPTQ 在该 fork 的 Marlin/量化 kernel 路径上更高效。本机当前模型是 AWQ；若可下载 `llmfan46/Qwen3.6-27B-...-GPTQ-Int4` 或 `QuantTrio/Qwen3.6-27B-AWQ` 之外经 weicj 锁定的 GPTQ checkpoint，可能再提升。

### 3.3 中投入：修复本机 g++-12 缺失

- 方案：`apt install g++-12`（需 sudo）。
- 效果：免去 `CC=/usr/bin/gcc-11` 等绕路，FlashQLA/FlashInfer/Triton 全用系统默认 CUDA 偏好编译器，减少补丁维护。
- 风险：低；只是改系统编译器包，不影响训练。

### 3.4 中投入：应用 weicj vLLM patch queue

weicj/2080Ti-LLM-Toolbox 提供 `engines/vllm/patches/`，例如：
- `0001-sm75-flashqla-gdn-ragged-prefill.patch`：修复 FlashQLA legacy 对 multi-prefill batch（packed cu_seqlens）的兼容性，使并发服务可用。
- 其他 patch 可能涉及 Marlin sub-tile 填充、TurboQuant decode clamp、MTP 在线量化传播等。

效果：多并发 / 多 prefill 场景更稳，peak 吞吐也略有提升。风险：需要重新编译 vLLM。

### 3.5 中投入：对齐 weicj 最佳 recipe 的 lock file

- 仓库：`engines/vllm/recipes/qwen36-27b-awq-best-sm75.md`
- lock：`manifests/dual-2080ti-vllm-qwen27-awq-best-sm75.lock`

按 lock 文件精确安装依赖、应用 patches，可复现仓库测试过的 101.3 tok/s 峰值；因为目前本机是手动修复的“最接近能跑”环境，可能还差一些 lock 中的细微版本/patch。

### 3.6 高投入/高风险：上 SGLang 或升级 vLLM 主线

- SGLang + Qwen3.6-27B-AWQ 在 weicj  STATUS 中仅 smoke，不稳定。
- vLLM 主线 0.21+ 对 GatedDeltaNet/FlashQLA 仍在快速迭代，但 SM75 不一定是主线优先支持目标；升级可能回退。

当前结论：weicj fork 是最现实的 SM75 优化 runtime。

### 3.7 通用 vLLM 调优（已在两条路线部分应用）

- `--max-num-batched-tokens` 越大通常吞吐越好（只要显存够），通用指南建议能高则高。
- `--enable-chunked-prefill`：开启后可混合 prefill/decode，降低长 prompt 对正在 decode 请求的阻塞。
- `--enable-prefix-caching`：命中共享前缀时减少重复 prefill。
- CUDA Graph： warmed-up 后减少 kernel launch overhead；MTP 场景用 PIECEWISE 更稳。
- `--disable-custom-all-reduce`：SM75 上更稳（已在 caovan 使用）。

## 4. 推荐优先顺序（针对本机）

1. **先对齐测速口径**：用 weicj 自带 benchmark 或固定 PP4096/TG128 短生成长度重测，确认真实 peak（预计会从当前 52 显著提升到 70-90+）。
2. **安装 g++-12** 并去掉 `CC=gcc-11` 补丁（最干净，避免后续 patch 冲突）。
3. **试 fast/int4/tqk8v4-256K-mtp3 profile**：预期最快接近 100 tok/s，但需做质量验收。
4. **如质量可接受**：换用 TurboQuant KV / 应用 patches 并 lock 文件对齐，可冲击宣称峰值。
5. **不推荐的捷径**：盲目提高 MTP K 到 4/5（接受率下降，LongGen3 上反而持平或下降）、或非 weicj 优化的 baseline（noMTP 仅 ~30-43 tok/s）。

## 5. 引用

- weicj/2080Ti-LLM-Toolbox README — Peak Result 101.3 tok/s: https://github.com/weicj/2080Ti-LLM-Toolbox
- weicj/vLLM-2080Ti-Definitive docs/mtp-task-sensitivity.md — MTP K sensitivity: https://github.com/weicj/vLLM-2080Ti-Definitive/blob/sm75-tp2-cu128-stable/docs/mtp-task-sensitivity.md
- caovan v0.1.3 install guide — AWQ MTP=2 baseline: https://caovan.com/rtx-2080-ti-bendedamoxingtuilitisujin50caovan-vllm-sm75-turbo3-waibuchajiananzhuangjiaochengq/.html
- vLLM blog/Qwen3-Next — GatedDeltaNet + MTP native support: https://vllm.ai/blog/2025-09-11-qwen3-next
- General vLLM tuning guide — tensor parallel / chunked prefill / max batched tokens: https://willitrunai.com/blog/vllm-multi-gpu-setup-guide
