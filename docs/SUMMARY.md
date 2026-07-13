# 复现总结：2×RTX 2080Ti 22G (NVLink) 上的 Qwen3.6-27B-AWQ-INT4

- 完成时间：2026-07-02 12:37 CST
- 状态：**两条路线均复现成功**

---

## 1. 目标与环境

在本机 2×RTX 2080Ti 22G + NVLink(NV2) 上复现 Qwen3.6-27B-AWQ-INT4 推理服务（源自 caovan.com 两篇文章 + weicj GitHub fork 三个参考）。

- 硬件：2×RTX 2080Ti 22G(SM75, NVLink) + 2×RTX 3080(**他人训练占用，全程隔离未碰**)；2×Xeon E5-2682v4 64线程；62G 内存。
- 隔离：全部落项目目录 `/mnt/hdd_storage/vllm_2080ti/`，`uv`/独立 `.venv`，不污染全局；`CUDA_DEVICE_ORDER=PCI_BUS_ID + CUDA_VISIBLE_DEVICES=0,1` 硬隔离到 2080Ti。
- 模型：`cyankiwi/Qwen3.6-27B-AWQ-INT4`（19GiB，含 INT4 MTP 头，两轨共用）。
- 编译器：已从 `g++-11` 升级到 `g++-12` 12.3.0，并验证 weicj/FlashQLA 可正常编译。

## 2. 两条路线对比（均在 256K 上下文实测）

| 维度 | 轨道 A: caovan v0.1.3 插件 | 轨道 B: weicj Definitive fork v0.1.11 |
|---|---|---|
| 形态 | 原版 vLLM 0.21.0 + 70KB 外部插件(monkey-patch) | 完整 vLLM fork 源码重编译(+FlashQLA/CUTLASS/Triton) |
| 安装成本 | 秒级 pip install | ~15min 编译(MAX_JOBS=8) + 大量依赖/编译器排障 |
| torch/CUDA | 2.11.0+cu130 / CUDA13 运行时 | 2.11.0+cu128 / CUDA12.8 toolkit(独立 .venv) |
| 加速面 | 仅 GDN+MTP=2 split-QK decode kernel | FlashQLA legacy prefill + Marlin + MTP=3 + CUDAGraph 全链 |
| 生效证据 | `_caovan_sm75_prepare_qk_conv_spec_decode_kernel` / `_caovan_sm75_fused_v_gdn_spec_decode_kernel` | `Using FlashQLA legacy SM70/SM75 GDN prefill kernel` |
| **decode 吞吐(256K)** | **~40.1 tok/s** | **~52 tok/s** |
| decode 吞吐(32K) | ~48.9 tok/s | (未单测) |
| **decode 吞吐(PP4096/TG128)** | (未测) | **~68 tok/s (normal/fp16kv)** / **~90 tok/s (fast/tqk8v4)** |
| draft acceptance | 75–95% | MTP=3 生效(未单独统计) |
| 稳定性 | max_num_seqs=1 稳定 | normal/PIECEWISE 稳定；fast/FULL_AND_PIECEWISE 亦通过 PP4096/TG128 短生成测试 |

结论：weicj fork 因多了 FlashQLA prefill + Marlin + MTP=3，256K 下比 caovan 插件快约 30%（52 vs 40 tok/s）。对齐到 PP4096/TG128 短生成口径后，weicj normal/fp16kv 约 68 tok/s，fast/tqk8v4 约 90 tok/s，仍未达到 weicj 宣称的 101.3 tok/s。**核心原因是本机模型为 AWQ-INT4，而 weicj 仓库 int4 profile 的峰值数据基于 GPTQ-INT4**。

## 3. 关于三篇参考

- caovan v0.4.33（第一篇，付费会员插件）：本地无，跳过。
- caovan v0.1.3（第二篇，公开插件，目录中已有）：**轨道 A，复现成功**。
- weicj/vLLM-2080Ti-Definitive（GitHub 开源 fork）：**轨道 B，复现成功**。

## 4. 关键工程结论（SM75/2080Ti 复现要点）

1. **CUDA13/torch2.11 仍含 sm_75**，2080Ti 可正常跑（arch_list 含 sm_75）。
2. Qwen3.6-27B 是 64 层混合架构：**48 层 GDN 线性注意力 + 16 层全注意力**，只有全注意力层持增长 KV → 配 fp8/fp16 KV 后 256K 上下文能塞进 2×22G。
3. **设备隔离必须 `CUDA_DEVICE_ORDER=PCI_BUS_ID` + `CUDA_VISIBLE_DEVICES=0,1` 同时设**（否则 CUDA FASTEST_FIRST 可能把 Ampere 3080 当 device0）。
4. SM75 上 FA2 不可用 → 自动回退 FLASHINFER/Triton；AWQ 走 compressed-tensors + Marlin(WNA16) 可用。
5. 重型源码编译必须限并行 + 内存看门狗（MAX_JOBS=24 曾致整机 OOM 重启）。
6. weicj 运行时三处 JIT（FlashQLA/FlashInfer/Triton）依赖本机编译器；本机已安装 `g++-12`，可去掉 `CC=gcc-11` 绕路。
7. **模型格式决定峰值上限**：weicj 仓库 int4 profile 的 throughput 数据全部基于 GPTQ-INT4；本机 AWQ-INT4 在相同 profile 下约低 10%（fast/tqk8v4 90 vs 100.81 tok/s）到 30%（normal/fp16kv 68 vs 97.79 tok/s）。

## 5. 复现入口

- 轨道 A：`bash serve_qwen36.sh`（默认 32K；`MAXLEN=262144 MEMUTIL=0.96 bash serve_qwen36.sh` 跑 256K）。
- 轨道 B：见 `docs/impl/20260702-123758-weicj-repro-success-and-fix-chain.md` 第 4 节启动命令。
- 全过程文档：`docs/research`、`docs/plan`、`docs/architecture`、`docs/impl`（按时间戳），术语表 `docs/glossary.md`。

## 6. 待办/建议
- 如需 weicj 冲更高吞吐：换用 **GPTQ-INT4** checkpoint（weicj 峰值基于该格式），例如 `llmfan46/Qwen3.6-27B-uncensored-heretic-v2-Native-MTP-Preserved-GPTQ-Int4`（保留原生 MTP 头）或 `palmfuture/Qwen3.6-27B-GPTQ-Int4`，预计 fast/tqk8v4 可达 ~100 tok/s。
- 一键测试脚本：`run_fast_tqk8v4_bench.sh`（仅性能），`run_quality_eval.sh`（仅质量，需服务已启动），`run_full_eval.sh`（性能 + 质量，全自动）。**注意做质量评估时需同时关闭 `REASONING_PARSER` 和 `enable_thinking`，否则模型会返回思考过程而非最终答案**。
- 最新完整评估结果（AWQ-INT4 / fast/tqk8v4 / PP4096/TG128）：TTFT 2.64s、Prefill 1551.5 tok/s、Decode 86.06 tok/s；5 项质量测试全部通过，大海捞针 needle 召回成功。详见 `docs/impl/20260702-1722-full-eval-thinking-off-needle-recovered.md`。
