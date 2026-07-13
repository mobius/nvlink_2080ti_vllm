# 实施文档（迭代 3）：SM75 可行性验证通过 + 模型分析 + 启动脚本

- 迭代时间：2026-07-01 19:0x CST
- 阶段：impl（第 3 轮）

---

## 1. SM75 / CUDA13 可行性门槛：✅ 通过（本项目最大不确定性已消除）

在 uv 隔离环境（`.venv`, Python 3.11）安装 `vllm==0.21.0` 后实测：

```
torch: 2.11.0+cu130      torch.version.cuda: 13.0     cudnn: 91900
arch_list (compiled): ['sm_75', 'sm_80', 'sm_86', 'sm_90', 'sm_100', 'sm_120']
gpu0/1: NVIDIA GeForce RTX 2080 Ti  cap=7.5  vram=21.5GiB
fp16 matmul on cuda:0 -> OK（真实执行）
```

结论：**torch 2.11 / CUDA 13 的预编译包仍含 `sm_75` 代码**，2080Ti 上真实 kernel 可执行。第 1 轮担心的"CUDA13 移除 Turing"不成立。

安装组件（关键）：`vllm 0.21.0 / torch 2.11.0 / torchvision 0.26.0 / triton 3.6.0 / transformers 5.12.1 / flashinfer-python`。

## 2. 插件安装与自检：✅ 全部 PASS

```
caovan-vllm-sm75-turbo3==0.1.3  安装于 .venv
caovan-sm75-doctor:
  GDN 接口检查：PASS family=legacy-gdn-linear-attn-v020
  插件入口发现：PASS (caovan_sm75_turbo3)
  GPU0/1: capability=(7,5)
  (modular-qwen-gdn-v021 路径在 0.21.0 不存在 → 正常跳过,与文章说明一致)
caovan-sm75-verify:  18 组合成张量用例全 PASS
  conv_exact=True, deterministic=True, finite=True, out_max=0
  → "PASS: Caovan SM75 Turbo3 外部插件内核通过结构安全门禁"
```

即：Turbo3 的 split-QK GDN spec-decode Triton kernel 在 2080Ti 上能编译并产出与参考路径逐位一致的结果。

## 3. 模型选定：`cyankiwi/Qwen3.6-27B-AWQ-INT4`（19.06 GiB, 4 shard）

选择理由：文章精确同名、体积最小（对 2×22G 更安全）、含 INT4 量化 MTP 头。

模型结构（从 config + safetensors index 实测）：

| 项 | 值 |
|---|---|
| 架构 | `Qwen3_5ForConditionalGeneration`（多模态，qwen3_5） |
| 层数 | 64；`full_attention_interval=4` → **48 层 GDN 线性注意力 + 16 层全注意力** |
| GDN 字段 | `linear_conv_kernel_dim=4`、key/value heads=16/48、head_dim=128/128 ✓ 符合插件候选 |
| MTP | `mtp.*` 36 个 tensor（INT4）✓ 满足 MTP=2 spec-decode |
| 视觉塔 | `visual.*` 333 个 tensor，vision_config depth=27/hidden=1152（多模态入口保留） |
| 上下文 | max_position_embeddings=262144（256K） |
| 量化 | compressed-tensors（AWQ-INT4） |

**关键洞察**：只有 16 层全注意力持有随上下文增长的 KV cache，48 层 GDN 仅维护固定大小递归状态 —— 这是 256K 上下文能在 44G 显存内成立的结构性原因。

## 4. 显存预算（粗估，2×21.5GiB，TP=2）

- 权重 19GiB INT4，TP 切分 → ~9.5GiB/GPU。
- 视觉塔 + MTP + embedding ~1-2GiB/GPU。
- `gpu-memory-utilization` 0.92 → 每卡 ~19.8GiB 上限，扣权重后 ~8GiB/卡 给 KV+激活+CUDA graph。
- 仅 16 层全注意力 + fp8 KV → 256K 上下文的 KV 占用被大幅压缩。文章称可行，但仍紧张。
- 策略:首启 `MAXLEN=32768, MEMUTIL=0.92` 验证管线与插件 dispatch；通过后再 `MAXLEN=262144, MEMUTIL=0.96` 复现完整配置。

## 5. 启动脚本 `serve_qwen36.sh`（已写入项目根）

- 用**原版 `vllm serve`**（v0.1.3 路线，不需要付费的 `caovan-vllm-serve`），插件经 `--additional-config '{"caovan_sm75_turbo3":true}'` + `--speculative-config '{"method":"mtp","num_speculative_tokens":2}'` 启用。
- 硬隔离 `CUDA_DEVICE_ORDER=PCI_BUS_ID` + `CUDA_VISIBLE_DEVICES=0,1`。
- 全部缓存指向 `cache/`。
- 支持 `TURBO3=0` 跑 baseline 对照。
- 已用 `vllm serve --help=all` 校验全部 flag 在 0.21.0 存在，`--mamba-cache-mode` 取值含 `align`，`--kv-cache-dtype` 含 `fp8`。

## 6. 下一步

1. 等 19GiB 下载完成。
2. `caovan-sm75-check-model models/Qwen3.6-27B-AWQ-INT4` 静态校验。
3. `bash serve_qwen36.sh`（保守配置）后台启动 → 等 warmup。
4. `curl /v1/models`、`/v1/chat/completions` 验证。
5. `grep` 服务日志确认 `ACTIVE DISPATCH confirmed` 与两个 caovan kernel 名。
6. 记录吞吐;若稳定,再推 256K。
