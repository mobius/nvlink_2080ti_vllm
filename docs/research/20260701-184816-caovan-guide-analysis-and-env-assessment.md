# 研究文档：Caovan vLLM SM75 Turbo3 v0.4.33 指南分析 + 环境可行性评估

- 迭代时间：2026-07-01 18:48:16 CST
- 阶段：research（第 1 轮）
- 来源指南：https://caovan.com/caovan-vllm-sm75-turbo3-v0433-install-guide/.html
- 目标：在本机 2×RTX 2080Ti 22G（NVLink）上复现 Qwen3.6-27B-AWQ-INT4 推理服务

---

## 1. 指南要点摘录

指南描述的是一个面向 SM75 / RTX 2080Ti 的**第三方付费插件** `caovan-vllm-sm75-turbo3 v0.4.33`，其核心命令是：

- 安装：`pip install --force-reinstall dist/caovan_vllm_sm75_turbo3-0.4.33-py3-none-any.whl`
- 自检：`caovan-sm75-doctor`
- 启动：`caovan-vllm-serve /data/qwen/Qwen3.6-27B-AWQ-INT4 ...`（TP=2, fp8 kv-cache, max-model-len 262144, MTP=3）

指南声称的软件栈：

| 组件 | 指南声称版本 |
|---|---|
| OS | Ubuntu 22.04 |
| Python | 3.11 |
| PyTorch | 2.11.0+cu130 |
| CUDA runtime | 13.0 |
| Triton | 3.6.0 |
| vLLM | 0.21.0 |
| 插件 | caovan-vllm-sm75-turbo3 v0.4.33 |

---

## 2. 本机环境实测

| 项目 | 实测结果 | 与指南目标是否匹配 |
|---|---|---|
| OS | Ubuntu 22.04.5 LTS | ✅ 匹配 |
| CPU | 2× Xeon E5-2682 v4，64 线程，62Gi 内存 | ✅ 足够 |
| GPU0/1 | 2× RTX 2080 Ti **22528 MiB**，compute cap **7.5 (SM75)** | ✅ 精确匹配目标硬件 |
| NVLink | GPU0↔GPU1 = **NV2**（2 条 bonded NVLink，各 25.781 GB/s，合计 ~51 GB/s） | ✅ 已启用 NVLink |
| GPU2/3 | 2× RTX 3080 20G（SM86），**正被占用** | ⚠️ 见下 |
| CUDA 工具链 | `/usr/local/cuda` → **12.8**（ptxas 12.8, 2025-02），driver 580.159.03 支持 CUDA 13.0 运行时 | ⚠️ 工具链是 12.8，非指南的 13.0 |
| uv | ✅ `/home/<USER>/.local/bin/uv` | ✅ 首选方案可用 |
| conda | ❌ 未安装 | — |
| docker / podman | ✅ 均可用 | ✅ 备选 |
| 系统 Python | 3.10.12 | 需在隔离环境中改用 3.11/3.12 |
| 模型 `/data/qwen/...` | ❌ 不存在（`/data/qwen` 无此目录） | ❌ 模型未就位 |
| 插件 zip | ❌ 不存在 | ❌ 见阻塞点 |
| 磁盘 | `/mnt/hdd_storage` 可用 2.4T | ✅ 充足 |

### 2.1 GPU2/3 占用情况（重要约束）

GPU2、GPU3（两张 RTX 3080）当前被 **另一进程占满**（100% util、~11GB 显存）：

```
PID 128563/128564  user=<USER>  已运行 8h30m
CMD: /mnt/hdd_storage/reason-lite/.venv/bin/python3 src/open_r1/sft.py --config .../config_stage1_l2b.yaml
```

这是一个正在进行的 SFT 训练任务。**本任务绝不可干扰 GPU2/3**。复现工作必须严格限制在 `CUDA_VISIBLE_DEVICES=0,1`（两张 2080Ti）上。好在指南目标本就是 2×2080Ti，天然隔离。

---

## 3. 阻塞点与技术疑点（必须先跟用户对齐）

### 阻塞点 A：核心插件是付费会员内容，无法公开获取 —— 致命阻塞
指南正文里插件下载处标注 **「会员专属内容 / PREMIUM ACCESS，开通会员后可查看完整内容、下载资源」**。
即：`caovan_vllm_sm75_turbo3-0.4.33-py3-none-any.whl`、`caovan-vllm-serve`、`caovan-sm75-doctor` 都来自这个**未公开、需付费**的私有包。
没有这个 zip，指南里从安装、doctor 到启动的每一步都无法执行。这是无法绕过的根本阻塞。

### 阻塞点 B：模型 `Qwen3.6-27B-AWQ-INT4` 并非已知的公开发布
- 通义千问公开发布过 Qwen2.5、Qwen3、Qwen3-Next（后者正是采用 **GDN(Gated DeltaNet) + MTP** 混合架构，与指南术语吻合）。
- 但 **「Qwen3.6-27B」这个名字目前没有对应的公开发布**。它很可能是同一付费来源重新打包/改名的模型，本机 `/data/qwen` 下也没有它。

### 技术疑点 C：指南声称的版本号多数不对应真实公开发布
- **PyTorch 2.11.0+cu130**：PyTorch 尚无 2.11 公开版本（公开线在 2.5–2.8 附近）。
- **vLLM 0.21.0**：vLLM 公开版本号形态为 0.x（近期 0.6–0.11），0.21 不符。
- 这些要么来自私有 fork，要么是文案夸大，需保持怀疑。

### 技术疑点 D：SM75 上 FP8 KV cache
RTX 2080Ti(Turing/SM75) **硬件不原生支持 FP8**（FP8 原生支持始于 Ada/SM89、Hopper/SM90）。原版 vLLM 在 SM75 上开 `--kv-cache-dtype fp8` 通常会报不支持。插件声称能接管，但这是需要验证的强假设。

### 技术疑点 E：44GB 总显存跑 27B 且 max-model-len 262144（256K 上下文）
即便有 FP8 KV cache 和 AWQ-INT4 权重，256K 上下文在 2×22G 上仍是非常激进的配置，真实可用性存疑。

---

## 4. 结论

- **硬件侧完全匹配**：本机确有 2×RTX 2080Ti 22G + NVLink，且这两张卡空闲，是指南的精确目标环境。
- **软件/模型侧无法照搬**：指南的核心插件与模型均为付费私有资源，本机都没有；且多处版本号不对应真实公开发布。
- 因此「逐字复现该指南」在当前条件下**不可行**，除非用户能提供那个付费插件 zip 与对应模型权重。
- 可行的替代路径是做一个**等价的开源复现**：在同样的 2×2080Ti + NVLink 上，用**公开可得的模型 + 原版 vLLM** 跑起 TP=2 推理服务（不依赖付费插件）。但这条路本身在 SM75 上有真实工程挑战（见后续 plan 文档）。

下一步需用户在两条路径间做选择（见 plan 文档与交互提问）。
