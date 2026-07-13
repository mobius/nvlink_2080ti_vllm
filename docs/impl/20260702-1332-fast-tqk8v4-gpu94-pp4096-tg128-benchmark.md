# fast/tqk8v4 + GPU_UTIL=0.94 实测

- 时间：2026-07-02 13:32 CST
- 路线：weicj vLLM-2080Ti-Definitive v0.1.11
- 模型：Qwen3.6-27B-AWQ-INT4
- GPU：2×RTX 2080Ti 22G (NVLink)，TP=2
- Profile：`qwen27b/fast/int4/tqk8v4-256K-mtp3-text-only.env`
- 模式：fast
- 变量：GPU_UTIL 从 0.90 提升到 0.94

## 启动命令

```bash
cd /mnt/hdd_storage/vllm_2080ti/weicj-vllm-2080ti
env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,1 GPU_DEVICES=0,1 TP_SIZE=2 \
    CC=/usr/bin/gcc-12 CXX=/usr/bin/g++-12 CUDAHOSTCXX=/usr/bin/g++-12 \
    QUANTIZATION=compressed-tensors \
    HF_HOME=/mnt/hdd_storage/vllm_2080ti/cache/hf \
    TRITON_CACHE_DIR=/mnt/hdd_storage/vllm_2080ti/cache/triton_weicj_tqk8v4_v3 \
    MODEL_DIR=/mnt/hdd_storage/vllm_2080ti/models/Qwen3.6-27B-AWQ-INT4 \
    PROFILE=qwen27b/fast/int4/tqk8v4-256K-mtp3-text-only.env \
    MODE=fast PORT=8000 SERVICE_SCOPE=local GPU_UTIL=0.94 \
    ./launcher.sh --non-interactive
```

## 实测结果

| 运行 | prompt tokens | gen tokens | TTFT (s) | prefill tok/s | decode tok/s |
|------|---------------|------------|----------|---------------|--------------|
| run1 | 4096 | 128 | 2.670 | 1534.3 | 89.61 |
| run2 | 4096 | 128 | 2.628 | 1558.9 | 90.02 |
| run3 | 4096 | 128 | 2.612 | 1568.1 | 89.67 |

- **稳定 decode 吞吐：约 89.6-90.0 tok/s**

## 结论

1. 将 `GPU_UTIL` 从 0.90 提升到 0.94 并未带来提升，反而略低于 0.90 时的 90.5-91.1 tok/s。
2. 原因可能是更高的 GPU util 被 CUDA graph memory profiling 抵消，或 TurboQuant KV 在该 AWQ 模型上的收益已饱和。
3. 当前 AWQ 模型在 fast/tqk8v4 下的甜点约为 **90 tok/s**。

## 关键发现：模型格式不匹配

查看 `weicj-vllm-2080ti/profiles/README.md` 发现：

> Tested checkpoint: **GPTQ-INT4**, about 19G.

weicj 仓库所有 `int4` profile 的宣称吞吐（包括 fast/tqk8v4 的 **100.81 tok/s**）都是基于 **GPTQ-INT4** 模型，而非本机的 AWQ-INT4。

| 配置 | weicj 宣称 (GPTQ-INT4) | 本机实测 (AWQ-INT4) | 差距 |
|------|------------------------|---------------------|------|
| normal/int4/fp16kv MTP3 | 97.79 tok/s | ~68 tok/s | -30% |
| fast/int4/tqk8v4 MTP3 | 100.81 tok/s | ~90 tok/s | -11% |

因此，要真正复现 weicj 宣称的 101.3 tok/s 峰值，需要换用 **GPTQ-INT4** checkpoint。

## 相关文件

- 测速原始数据：`/mnt/hdd_storage/vllm_2080ti/logs/20260702-1332-weicj-fast-tqk8v4-gpu94-pp4096-tg128-bench.jsonl`
- 服务启动日志：`/mnt/hdd_storage/vllm_2080ti/logs/20260702-1332-weicj-fast-tqk8v4-gpu94-pp4096-tg128-launch.log`
