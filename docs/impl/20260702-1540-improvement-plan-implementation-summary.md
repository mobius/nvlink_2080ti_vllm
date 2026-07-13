# 提升方案实施小结（截止 2026-07-02 15:40 CST）

## 已完成项

1. **g++-12 安装并验证**
   - 版本：`g++-12 (Ubuntu 12.3.0-1ubuntu1~22.04.3) 12.3.0`
   - 已将 `weicj-vllm-2080ti/.deps/FlashQLA-SM70-SM75/flash_qla/ops/gated_delta_rule/legacy/sm_legacy.py` 中的 `-ccbin /usr/bin/g++-11` 改为 `-ccbin /usr/bin/g++-12`。
   - weicj 服务可正常编译启动，FlashQLA legacy kernel 生效。

2. **对齐测速口径 PP4096/TG128**
   - 使用仓库自带 `tools/profile_request.py`，endpoint=`completions`，`--pure-filler --ignore-eos`。
   - prompt=4096 tokens，generation=128 tokens。

3. **实测 weicj 各 profile 吞吐**

| profile | GPU_UTIL | 模式 | KV | 实测 decode tok/s | 备注 |
|---------|----------|------|----|-------------------|------|
| normal/int4/fp16kv-256K-mtp3 | 0.90 | normal | FP16 | ~68 | 稳定 |
| fast/int4/tqk8v4-256K-mtp3 | 0.90 | fast | TurboQuant k8v4 | ~90.5-91.1 | 当前 AWQ 下最佳 |
| fast/int4/tqk8v4-256K-mtp3 | 0.94 | fast | TurboQuant k8v4 | ~89.6-90.0 | 提升 GPU util 无收益 |

4. **关键发现：模型格式决定峰值上限**
   - `weicj-vllm-2080ti/profiles/README.md` 写明所有 int4 profile 的吞吐数据基于 **GPTQ-INT4** checkpoint。
   - 本机模型为 **AWQ-INT4**（`cyankiwi/Qwen3.6-27B-AWQ-INT4`）。
   - 因此 fast/tqk8v4 在 AWQ 下跑到 ~90 tok/s 已接近该格式上限，无法达到 weicj 宣称的 100.81/101.3 tok/s。

## 与宣称值对比

| 配置 | weicj 宣称 (GPTQ-INT4) | 本机实测 (AWQ-INT4) | 差距 |
|------|------------------------|---------------------|------|
| normal/int4/fp16kv MTP3 | 97.79 tok/s | ~68 tok/s | -30% |
| fast/int4/tqk8v4 MTP3 | 100.81 tok/s | ~90 tok/s | -11% |

## 结论

- 在 AWQ-INT4 模型下，当前最优配置为 `fast/int4/tqk8v4-256K-mtp3-text-only.env`，PP4096/TG128 口径稳定 decode 吞吐约 **90 tok/s**。
- 这一结果显著优于 caovan 插件路线的 ~40 tok/s 和此前长输出端到端的 ~52 tok/s。
- 用户决定不再下载 GPTQ-INT4 模型，当前进展已归档。

## 相关文档

- 正常测速记录：`docs/impl/20260702-1300-pp4096-tg128-benchmark-normal-fp16kv.md`
- fast/tqk8v4 测速记录：`docs/impl/20260702-1325-fast-tqk8v4-pp4096-tg128-benchmark.md`
- GPU_UTIL=0.94 测速记录：`docs/impl/20260702-1332-fast-tqk8v4-gpu94-pp4096-tg128-benchmark.md`
- 汇总更新：`docs/SUMMARY.md`
- 术语表更新：`docs/glossary.md`

## 原始数据

- normal 测速：`logs/20260702-1300-weicj-normal-pp4096-tg128-bench.jsonl`
- fast/tqk8v4 测速：`logs/20260702-1325-weicj-fast-tqk8v4-pp4096-tg128-bench.jsonl`
- GPU_UTIL=0.94 测速：`logs/20260702-1332-weicj-fast-tqk8v4-gpu94-pp4096-tg128-bench.jsonl`
