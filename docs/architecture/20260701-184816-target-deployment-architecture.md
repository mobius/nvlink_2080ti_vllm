# 架构文档：2×RTX 2080Ti 22G NVLink 推理部署（目标架构）

- 迭代时间：2026-07-01 18:48:16 CST
- 阶段：architecture（第 1 轮）

---

## 1. 硬件拓扑（实测）

```
NUMA node0 (CPU 0-15,32-47)
  ├── GPU0  RTX 2080Ti 22G  SM75  ─┐
  │                                ├─ NV2 (2× NVLink, ~51 GB/s)  ← 本项目使用
  └── GPU1  RTX 2080Ti 22G  SM75  ─┘

NUMA node1 (CPU 16-31,48-63)
  ├── GPU2  RTX 3080 20G  SM86   ← 被占用(<USER> SFT 训练)，禁止触碰
  └── GPU3  RTX 3080 20G  SM86   ← 被占用(<USER> SFT 训练)，禁止触碰
```

- GPU0↔GPU1 通过 NVLink(NV2) 直连，是 TP=2 的理想通信通道。
- GPU0/1 与 GPU2/3 跨 NUMA、仅 PCIe/QPI 相连(SYS)，且 2/3 在训练中，故本项目**硬隔离到 GPU0,1**。

## 2. 软件分层（目标）

```
┌─────────────────────────────────────────────┐
│ 客户端 (curl / OpenAI SDK)                    │
├─────────────────────────────────────────────┤
│ OpenAI 兼容 HTTP API  :8000                   │
├─────────────────────────────────────────────┤
│ vLLM 引擎 (TP=2, PagedAttention, cont. batch) │
│   ├─ 张量并行分片 → GPU0 / GPU1               │
│   └─ (可选) 投机解码 / 量化 kernel            │
├─────────────────────────────────────────────┤
│ PyTorch + CUDA runtime + Triton               │
├─────────────────────────────────────────────┤
│ NVIDIA driver 580.159.03 (支持 CUDA 13 运行时)│
│ 本地 CUDA 工具链: /usr/local/cuda → 12.8      │
├─────────────────────────────────────────────┤
│ 隔离环境: uv venv (首选) / conda / 容器        │
└─────────────────────────────────────────────┘
```

## 3. 环境隔离原则（用户要求）

1. **首选 uv**：`uv venv` 建本地虚拟环境，`uv pip install` 装依赖，全部落在项目目录内，不污染全局。
2. 次选 conda（本机未装，需先装 miniforge，成本更高）。
3. 再次 podman/docker（vLLM 官方镜像，但需处理 NVIDIA container runtime 与 SM75 兼容）。
4. 任何安装均不写入系统全局 site-packages，不 `sudo pip`。

## 4. 与指南原架构的差异

- 指南用 conda 环境 + 付费插件接管 GDN/prefill/decode 热路径。
- 本机若走开源路径，则用 uv venv + 原版 vLLM，无付费插件；GDN/MTP 等特性取决于所选模型与 vLLM 版本对 SM75 的支持程度。
- FP8 KV cache、AWQ Marlin 等在 SM75 上可能不可用，需实测降级方案（fp16 KV / 非 Marlin 量化 kernel）。
