# 实施文档（迭代 7）：caovan 256K 完整配置复现成功

- 迭代时间：2026-07-02 10:12 CST
- 阶段：impl（轨道 A 终极配置）

---

## 1. 配置（对齐 caovan 文章头条）

```
bash serve_qwen36.sh  with  MAXLEN=262144 MEMUTIL=0.96 MAXSEQS=2
vllm serve Qwen3.6-27B-AWQ-INT4
  --tensor-parallel-size 2 --dtype half
  --max-model-len 262144            # 256K 原生上下文
  --kv-cache-dtype fp8
  --gpu-memory-utilization 0.96
  --mamba-cache-mode align --enable-prefix-caching --disable-custom-all-reduce
  --additional-config '{"caovan_sm75_turbo3":true}'
  --speculative-config '{"method":"mtp","num_speculative_tokens":2}'
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,1  (仅 2×2080Ti)
```

## 2. 结果（实测）

| 指标 | 值 |
|---|---|
| 启动就绪 | ~165s（torch.compile 缓存命中，比首次 688s 快很多） |
| GPU KV cache | **519,819 tokens** |
| 256K 上下文并发 | 1.98x |
| 每卡 Available KV | 9.06 GiB |
| 每卡显存占用 | 21,197 / 22,528 MiB（util 0.96，紧但稳定，无 OOM） |
| decode 吞吐 | **40.1 tok/s**（600 tokens / 14.95s） |
| 插件 kernel | 两个 `_caovan_sm75_*` 在 256K 下同样 JIT dispatch ✓ |
| Draft acceptance | 75.6% – 95.5% |

## 3. 结论

- caovan v0.1.3 插件路线在 **2×RTX 2080Ti 22G + NVLink** 上，用 **Qwen3.6-27B-AWQ-INT4** 完整复现了文章的关键配置：
  - 256K 原生上下文 ✓
  - FP8 KV cache ✓
  - TP=2 ✓
  - MTP=2 speculative decode ✓
  - Turbo3 GDN+MTP split-QK kernel 真实生效 ✓
- 256K 能塞进 44G 的结构性原因再次印证：64 层中仅 16 层全注意力持 KV（fp8），48 层 GDN 只存固定递归状态。
- 吞吐：32K 下 ~48.9 tok/s，256K 下 ~40.1 tok/s（单请求）。与 caovan 文章"60+ tok/s"有差距，原因：本插件只优化 GDN+MTP=2 decode，未做 prefill/Marlin/KV 的 SM75 深度优化；且未开 FlashInfer autotune / 更激进 profile。真正冲 100 tok/s 是 weicj fork 的 MTP=3+TurboQuant 路线（轨道 B，编译中）。

## 4. 两种上下文配置对比（轨道 A 内部）

| 配置 | KV cache tokens | 每卡显存 | decode tok/s | 用途 |
|---|---|---|---|---|
| MAXLEN=32768, util=0.90, seqs=1 | 297,890 | ~19.9G | 48.9 | 日常/低延迟 |
| MAXLEN=262144, util=0.96, seqs=2 | 519,819 | ~21.2G | 40.1 | 长上下文极限复现 |

## 5. 服务保持
256K 服务当前存活(port 8000)，作为轨道 A 的最终交付。GPU2/3 训练全程未受影响。
