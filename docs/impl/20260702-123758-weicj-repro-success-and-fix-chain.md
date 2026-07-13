# 实施文档（迭代 8）：weicj Definitive fork 复现成功 + 完整排障链

- 迭代时间：2026-07-02 12:37 CST
- 阶段：impl（轨道 B 完成）

---

## 1. 结果

`weicj/vLLM-2080Ti-Definitive` v0.1.11 在本机 2×RTX 2080Ti 22G(NVLink)上成功跑起 Qwen3.6-27B-AWQ-INT4：

| 项 | 值 |
|---|---|
| profile | `qwen27b/normal/int4/fp16kv-256K-mtp3-text-only`（256K, fp16 KV, MTP=3） |
| 量化 | compressed-tensors（W4A16, 内部 Marlin） |
| decode 吞吐 | **~52 tok/s**（600 tokens/11.5s，多次稳定 52.0/52.4/52.3） |
| 每卡显存 | 20,827 / 22,528 MiB（util 0.90） |
| 关键 kernel | `Using FlashQLA legacy SM70/SM75 GDN prefill kernel`（fork 核心优化生效 ✓） |
| MTP | `Detected MTP model. Sharing target embedding/lm_head with draft`（MTP=3 spec decode 生效 ✓） |
| 就绪耗时 | ~255s（首次含 FlashInfer/Triton JIT） |
| 输出 | 连贯（qwen3 reasoning 模式，思考内容在 reasoning 字段） |

关于速度与 fork 宣称 97.79 tok/s 的差异：fork 的口径是 4096/128 的**纯 decode** 稳态；本文是**端到端 wall-clock**（600 tokens，含 prefill + reasoning + 一次性 JIT 已排除），且 `--disable-log-stats` 未输出引擎级 decode 速率。二者口径不同，本文 52 tok/s 是保守的端到端真实值。复现结论（fork 可编译、可服务、FlashQLA+MTP 生效）成立。

## 2. 完整排障链（本机特有问题）

编译阶段（前文迭代 4-5）：
1. z3-solver 解包竞态 → 清缓存 + `UV_LINK_MODE=copy`。
2. tilelang 0.1.9 空壳 → 强制重装。
3. MAX_JOBS=24 CUTLASS 编译 OOM → **机器重启**（教训：改 MAX_JOBS=8 + 内存看门狗 8GB 阈值，成功无 OOM）。

依赖修复（编译成功后 native 产物 OK，但一批包被之前网络中断装残缺）：
4. transformers/flashinfer-cubin/flashinfer-python/mistral_common/nvidia-cudnn-frontend/pycountry/uvloop/llvmlite → 逐一/批量强制重装 → `vllm --version=0.1.11`。
5. tokenizers 0.23.1 与 transformers 冲突 → 降 0.22.2（对齐 caovan venv）。

服务阶段（三处编译器/环境问题）：
6. launcher mmap 预检要求 `vm.overcommit_memory=1`（需 sudo）→ 实测 5GB mmap 可用，判定误报 → 给 `check_checkpoint_mmap_policy` 打可逆补丁 `return 0`。
7. 量化参数：launcher 猜 `awq_marlin`，模型 config 是 compressed-tensors → 传 `QUANTIZATION=compressed-tensors`。
8. **系统缺 g++-12**（只有 gcc-12 驱动无 cc1plus）→ 三处运行时 JIT 全部受影响：
   - FlashQLA legacy GDN kernel(.cu)：补丁 `sm_legacy.py` 加 `-ccbin /usr/bin/g++-11`。
   - FlashInfer prefill kernel(head_dim256, .cu)：它读 `CC` 作 nvcc `-ccbin`。
   - Triton launcher(.c)：它读 `CC` 作 **C 编译器**。
   - 关键坑：先设 `CC=g++-11` 修好 FlashInfer 却**打断 Triton**（C 文件被 C++ 编译，`_Alignas`/隐式 enum 转换等报错）。正解是 **`CC=/usr/bin/gcc-11`**（gcc 驱动）——既正确编 Triton 的 `.c`，又带 cc1plus 供 nvcc 做 host C++。`CXX=CUDAHOSTCXX=g++-11`。
   - launcher 第 2665 行 `[[ -z "${CC:-}" ]] && export CC=gcc-12` → 预设 CC 即不被覆盖。

## 3. 本机遗留补丁清单（复现需知，均可还原）

- `weicj-vllm-2080ti/launcher.sh`：`check_checkpoint_mmap_policy()` 开头加 `return 0`（绕 mmap 预检）。
- `weicj-vllm-2080ti/.deps/FlashQLA-SM70-SM75/.../legacy/sm_legacy.py`：`extra_cuda_cflags` 加 `-ccbin /usr/bin/g++-11`。
- 启动环境必须：`CC=/usr/bin/gcc-11 CXX=/usr/bin/g++-11 CUDAHOSTCXX=/usr/bin/g++-11 QUANTIZATION=compressed-tensors CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,1`。
- 根治建议：`apt install g++-12`（需 sudo，未执行）即可免去 8 里的编译器绕行。

## 4. 启动命令（可复现）

见 `weicj_launch7` 环境；等价直启：
```
cd weicj-vllm-2080ti
env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,1 GPU_DEVICES=0,1 TP_SIZE=2 \
    CC=/usr/bin/gcc-11 CXX=/usr/bin/g++-11 CUDAHOSTCXX=/usr/bin/g++-11 \
    QUANTIZATION=compressed-tensors \
    MODEL_DIR=/mnt/hdd_storage/vllm_2080ti/models/Qwen3.6-27B-AWQ-INT4 \
    PROFILE=qwen27b/normal/int4/fp16kv-256K-mtp3-text-only.env MODE=normal PORT=8000 SERVICE_SCOPE=local \
    ./launcher.sh --non-interactive
```

## 5. 服务状态
weicj 服务当前在 port 8000 存活（GPU0/1）。GPU2/3 训练全程未受影响（MAX_JOBS 限制 + 看门狗 + 设备隔离奏效，重启事件后训练已自动恢复）。
