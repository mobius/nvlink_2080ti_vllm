# 实施文档（迭代 6）：caovan 插件成功复现 + 首次 API 请求与吞吐

- 迭代时间：2026-07-02 09:53 CST
- 阶段：impl（轨道 A 验证成功）

---

## 1. 配置

- 命令: `bash serve_qwen36.sh` (`CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,1`)
- 环境: `vllm==0.21.0`, `torch==2.11.0+cu130`, `caovan-vllm-sm75-turbo3==0.1.3`
- 模型: `cyankiwi/Qwen3.6-27B-AWQ-INT4`, TP=2, `max_model_len=32768`, `kv-cache-dtype=fp8`
- 插件启用: `--additional-config '{"caovan_sm75_turbo3":true}'`
- 投机解码: `--speculative-config '{"method":"mtp","num_speculative_tokens":2}'` (MTP=2)
- 关键改动: `max_num_seqs=1`(避免第一次因空闲等待导致的 worker 超时误判)

## 2. 关键日志证据

### 2.1 服务启动成功
```
EngineCore: GPU KV cache size: 297,890 tokens
Maximum concurrency for 32,768 tokens per request: 9.09x
Graph capturing finished in 2 secs
core.py: init engine ... took 155.79 s (compilation: 114.62 s)
api_server.py: Supported tasks: ['generate']
Application startup complete.
```

### 2.2 插件进入真实推理路径(最重要)
```
jit_monitor: Triton kernel JIT compilation during inference:
    _caovan_sm75_prepare_qk_conv_spec_decode_kernel
    _caovan_sm75_fused_v_gdn_spec_decode_kernel
```
与 v0.1.3 文章一致，说明 Turbo3 真实接管了 GDN+MTP speculative decode 热路径。

### 2.3 模型/硬件路径
```
Resolved architecture: Qwen3_5ForConditionalGeneration / Qwen3_5MTP
compressed_tensors_wNa16: Using MarlinLinearKernel for CompressedTensorsWNA16
gdn_linear_attn: Using Triton/FLA GDN prefill kernel
Using FLASHINFER attention backend
FA2 not supported on compute capability 7.5 → fallback OK
Device capability 7.5 (SM75) confirmed
```

## 3. API 验证结果

| 请求 | 耗时 | 输出 | 观察 |
|---|---|---|---|
| 1 (prompt "用三句话介绍你自己", max_tokens=128) | 87.5s | finish_reason=length | 首请求触发 caovan Triton JIT，符合"首次慢"预期 |
| 2 (prompt "写一段200字的科幻小说开头", max_tokens=200) | 4.75s | 200 completion tokens | **decode ~42.1 tok/s** |

投机解码接受率:
- 请求1: Mean acceptance length 2.87, **Avg Draft acceptance rate 93.3%**
- 请求2: Mean acceptance length 2.54, **Avg Draft acceptance rate 77.2%**

## 4. 解释

- **42–49 tok/s**（多请求实测稳定，600 tokens/12.28s = **48.9 tok/s**，vLLM 日志 `Avg generation throughput: 48.9 tokens/s` 印证）是在 *MAXLEN=32768 / MTP=2 / fp8 KV / compressed-tensors Marlin / no-eager with PIECEWISE cudagraph* 下的真实单请求 decode 速度。
- 这不是文章的 ~60 tok/s 峰值，也不是 weicj fork 的 100 tok/s，因为：
  1. caovan v0.1.3 只优化了 GDN+MTP=2 的 spec-decode kernel，未触及 prefill/FlashQLA/Marlin SM75 深层优化。
  2. 当前上下文预算保守(32K)，未启用更激进的 profile/256K/MTP=3/TurboQuant。
- 本次核心交付是"成功复现插件生效的真实路径"，而非极限速度。

## 5. 第一次崩溃(迭代 5)的根因

首次启动(`MAXSEQS=4`)在 warmup 成功、就绪后约 4.5 分钟无请求，EngineCore 反复报 "No available shared memory broadcast block in 60 seconds"，随后 `WorkerProc died unexpectedly`。 

最可能解释：V1 EngineCore 对 RPC 响应有超时(60s)，worker 在图捕获/compile 期间耗时超过 60s 无心跳，被误判为死亡；或在长等待期间触发了 NCCL/通信 watchdog。降到 `MAXSEQS=1` 后稳定，可能是因为减少了调度器复杂度、worker 响应更快。

## 6. 备注

- `/v1/models` 正常返回。
- 服务仍存活；GPU0/1 各约 19.9GB 显存，无 OOM。
- `CUDA_VISIBLE_DEVICES=0,1` 必须配合 `CUDA_DEVICE_ORDER=PCI_BUS_ID` 使用，否则 CUDA 可能按 FASTEST_FIRST 把 3080 当 device 0。已再次确认当前服务确实运行在 cc7.5(SM75)上。
