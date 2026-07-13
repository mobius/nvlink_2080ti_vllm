# 研究文档：weicj/vLLM-2080Ti-Definitive 分析与双轨复现决策

- 迭代时间：2026-07-01 19:22:23 CST
- 阶段：research（第 4 轮）
- 来源：https://github.com/weicj/vLLM-2080Ti-Definitive（branch `vllm-2080ti-deifinitive`，release v0.1.11）
- 触发：用户要求"重点关注这篇实现"

---

## 1. 它是什么

一个**面向双 RTX 2080Ti / SM75 的完整 vLLM fork**（不是插件），base vLLM 0.21.0，Apache-2.0。
- 实测头条：双 2080Ti TP=2，Qwen3.6-27B 单请求 decode **100+ tok/s**。
- 集成：Marlin 权重路线（FP8/INT4/NVFP4）、FlashQLA-SM70/SM75 + FlashInfer 快速 prefill、TurboQuant/INT8 KV、MTP、CUDAGraph、256K 原生上下文、YaRN 扩展、图像多模态。
- 定位：极限**单并发**性能（个人 agent 场景），非多租户吞吐集群。

## 2. 与 caovan 两条路线的区别（关键）

| 维度 | caovan v0.1.3（插件，本地已有） | weicj Definitive（完整 fork） |
|---|---|---|
| 形态 | 装在原版 vLLM 上的小插件（monkey-patch GDN+MTP decode） | 从源码重编译整个 vLLM + CUDA 扩展 |
| 安装 | `pip install` 一个 70KB wheel（秒级） | `./build.sh` 全量编译（~20-45 分钟） |
| torch/CUDA | torch 2.11.0**+cu130**（我已装） | torch 2.11.0**+cu128** + CUDA 12.8 toolkit（自建独立 .venv） |
| 加速面 | 仅 GDN+MTP=2 split-QK decode kernel | Marlin + FlashQLA prefill + TurboQuant KV + CUDAGraph 全链路 |
| 声称速度 | ~60 tok/s | ~100 tok/s |
| 依赖复杂度 | 极低（纯 Triton，无外部 kernel） | 高（CUTLASS v4.4.2 / Triton v3.6.0 / FlashQLA-SM70-SM75 源码编译） |
| 模型 | 通用 Qwen3.6 GDN+MTP | 同上（profile 与模型路径解耦，两者可共用同一权重） |

**共用点**：两者都基于 vLLM 0.21.0 + Qwen3.6-27B AWQ-INT4，我下载的 `cyankiwi/Qwen3.6-27B-AWQ-INT4` **两条路线都能用**。

## 3. build.sh 做什么（子代理审阅 + 本地确认）

15 步一键源码构建：
1. preflight（CPU/内存/磁盘/GPU/拓扑/下载源 benchmark）—— 本机已全部通过，识别出"双 2080Ti 目标"。
2. `uv venv --python 3.11 .venv`（在 repo 内，独立隔离）。
3. 装 `requirements/build/cuda.txt` + `requirements/cuda.txt`（`torch==2.11.0`、`flashinfer-python==0.6.8.post1`、`nvidia-cutlass-dsl==4.4.2`、`tilelang==0.1.9` 等）。
4. clone 到 `.deps/`：FlashQLA-SM70-SM75、CUTLASS `v4.4.2`、Triton `v3.6.0`。
5. 用 `tools/flashqla_sm75_patches/{sm_legacy.py,gdn_forward.cu}` 打补丁，编译 FlashQLA legacy GDN CUDA 扩展。
6. **重步骤**：`TORCH_CUDA_ARCH_LIST=7.5 MAX_JOBS=.. FLASHINFER_ENABLE_AOT=1 ... uv pip install --no-build-isolation --no-deps -e .` —— 单架构 sm_75 全量重编译 vLLM `csrc/`（CUTLASS w8a8/w4a8 GEMM、Marlin/AWQ/GPTQ、paged-attn、Mamba SSM、MoE…）。
7. `tools/validate_runtime_components.py` 严格校验 `vllm._C`、FlashInfer、FlashQLA `.so` 可导入。

**无预编译 Release**（所有 release assets 为空），必须现编。

## 4. 本机可行性：满足，无硬阻塞

- nvcc 12.8 ✓（`/usr/local/cuda-12.8/bin/nvcc` V12.8.93）、gcc/g++ 11.4 ✓。
- CUDA dev 静态库齐全 ✓（`libcudadevrt.a / libcudart_static.a / libculibos.a`）、`cuda_runtime.h` ✓。
- 51GB 可用内存、2.4TB 磁盘、64 线程 ✓。
- driver 580（宣称 CUDA13）> 验证的 590 无关紧要，cu128 wheel 对更新 driver 兼容 ✓。
- 风险控制：`MAX_JOBS` 默认 62 会 OOM/挤占训练 → 本次**手动设 `MAX_JOBS=24`**，并 `ASSUME_YES=1` 非交互；`CUDA_HOME=/usr/local/cuda-12.8`。

## 5. Profile 体系与目标速度

命名：`profiles/<model>/<mode>/<precision>/<kv>-<ctx>-<mtp>-<msgtype>.env`；模式 `safe/normal/fast/aggressive`。
- `qwen27b/normal/int4/fp16kv-256K-mtp3-text-only.env` → 97.79 tok/s decode
- `qwen27b/fast/int4/tqk8v4-256K-mtp3-text-only.env` → **100.81 tok/s**（头条，MTP_K=3、KV=turboquant_k8v4、MAX_MODEL_LEN=262144、GPU_UTIL=0.90、MAX_NUM_SEQS=1）
- 推荐 int4 权重：`QuantTrio/Qwen3.6-27B-AWQ`（我下的 cyankiwi 同为 AWQ-INT4，应可直接用；如遇 MTP/结构问题再换）。

## 6. 双轨复现决策

用户先给 caovan 插件（已推进到差模型下载），又要求重点关注 weicj。二者独立、共用模型 → **并行推进**：
- **轨道 A（caovan v0.1.3）**：模型下载完 → `serve_qwen36.sh` 起服务 → 验 API + 插件 ACTIVE DISPATCH。轻量、先出可用结果。
- **轨道 B（weicj Definitive）**：`build.sh` 后台编译中（独立 cu128 .venv，`MAX_JOBS=24`）→ 编译完用 `launcher.sh --non-interactive` + 上述 profile 起服务 → 对比 100 tok/s。
- 两条长任务（下载、编译）已挂后台监视，完成自动通知。
- 全程 `CUDA_VISIBLE_DEVICES=0,1`，GPU2/3 训练不受影响；所有产物落项目目录，不污染全局。

## 7. 署名（遵 repo AGENTS.md）
本项目复现基于 upstream vLLM 与 `vLLM 2080 Ti Definitive Edition`（github.com/weicj）。文档中的性能数字均标注来源（作者实测 vs 我方实测），不混淆、不虚构。
