# 测速口径对齐：PP4096/TG128 实测 weicj normal/fp16kv

- 时间：2026-07-02 13:00 CST
- 目标：对齐 weicj/2080Ti-LLM-Toolbox 宣称的 101.3 tok/s 峰值测速口径
- 路线：weicj vLLM-2080Ti-Definitive v0.1.11
- 模型：Qwen3.6-27B-AWQ-INT4
- GPU：2×RTX 2080Ti 22G (NVLink)，TP=2
- 编译器：g++-12 12.3.0（已替代之前的 g++-11 补丁）
- Profile：`qwen27b/normal/int4/fp16kv-256K-mtp3-text-only.env`
- 模式：normal
- 关键参数：MTP=3, FlashQLA legacy GDN prefill, FP16 KV, chunked prefill, prefix caching, cudagraph PIECEWISE

## 启动命令

```bash
cd /mnt/hdd_storage/vllm_2080ti/weicj-vllm-2080ti
env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,1 GPU_DEVICES=0,1 TP_SIZE=2 \
    CC=/usr/bin/gcc-12 CXX=/usr/bin/g++-12 CUDAHOSTCXX=/usr/bin/g++-12 \
    QUANTIZATION=compressed-tensors \
    HF_HOME=/mnt/hdd_storage/vllm_2080ti/cache/hf \
    TRITON_CACHE_DIR=/mnt/hdd_storage/vllm_2080ti/cache/triton_weicj \
    MODEL_DIR=/mnt/hdd_storage/vllm_2080ti/models/Qwen3.6-27B-AWQ-INT4 \
    PROFILE=qwen27b/normal/int4/fp16kv-256K-mtp3-text-only.env \
    MODE=normal PORT=8000 SERVICE_SCOPE=local \
    ./launcher.sh --non-interactive
```

## 测速方法

使用仓库自带 `tools/profile_request.py`，endpoint=`completions`，`--pure-filler --ignore-eos`：

```bash
.venv/bin/python tools/profile_request.py \
  --model-dir /mnt/hdd_storage/vllm_2080ti/models/Qwen3.6-27B-AWQ-INT4 \
  --served-name qwen27b-int4-fp16kv-256K-mtp3-text-only-cu128 \
  --base-url http://127.0.0.1:8000/v1 \
  --endpoint completions --prompt-tokens 4096 --gen-tokens 128 \
  --label weicj-normal-fp16kv-mtp3-pp4096-tg128-runX \
  --out /mnt/hdd_storage/vllm_2080ti/logs/20260702-1300-weicj-normal-pp4096-tg128-bench.jsonl \
  --gpu-log /mnt/hdd_storage/vllm_2080ti/logs/20260702-1300-weicj-normal-pp4096-tg128-gpu.log \
  --ignore-eos --pure-filler
```

## 实测结果

| 运行 | prompt tokens | gen tokens | TTFT (s) | prefill tok/s | decode tok/s | 备注 |
|------|---------------|------------|----------|---------------|--------------|------|
| warmup1 | 4096 | 128 | 3.493 | 1172.7 | 72.80 | 含 cudagraph 冷启动 |
| run2 | 4096 | 128 | 0.663 | 6182.2 | 61.97 | warm 后 |
| run3 | 4096 | 128 | 0.652 | 6284.6 | 69.54 | warm 后 |

- **稳定 decode 吞吐：约 65-70 tok/s**
- 峰值 warm-up 单点：72.80 tok/s

## 结论

1. 对齐到 PP4096/TG128 短生成口径后，normal/fp16kv 路线比此前长输出端到端 52 tok/s 提升约 30%。
2. 仍低于 weicj 宣称的 101.3 tok/s；差距来源可能是 fast/tqk8v4 KV 压缩、更高 GPU util、FULL_AND_PIECEWISE cudagraph。
3. g++-12 编译路径验证成功，FlashQLA legacy kernel 正常加载。
4. 下一步：切换 `fast/int4/tqk8v4-256K-mtp3-text-only.env` 继续冲击 ~100 tok/s。

## 相关文件

- 测速原始数据：`/mnt/hdd_storage/vllm_2080ti/logs/20260702-1300-weicj-normal-pp4096-tg128-bench.jsonl`
- GPU 监控日志：`/mnt/hdd_storage/vllm_2080ti/logs/20260702-1300-weicj-normal-pp4096-tg128-gpu.log`
- 服务启动日志：`/mnt/hdd_storage/vllm_2080ti/logs/20260702-1300-weicj-normal-gcc12-pp4096-tg128-launch.log`
