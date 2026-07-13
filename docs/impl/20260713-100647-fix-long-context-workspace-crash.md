# 修复 vLLM 长上下文 workspace 崩溃并恢复服务

时间：2026-07-13 10:06 CST  
作者：Kimi Code CLI  
关联文件：serve_fast_tqk8v4_web.sh、tools/anthropic_proxy.py

---

## 1. 故障现象

用户报告调用 `http://<YOUR_SERVER_IP>:8000/v1/chat/completions` 时报错：

```
error sending request for url (...): client error (SendRequest):
connection closed before message completed
```

检查发现 vLLM 服务已崩溃退出，Anthropic 代理 `18081` 仍在运行但上游不可用。

---

## 2. 根因分析

查看 vLLM 崩溃日志 `/mnt/hdd_storage/vllm_2080ti/weicj-vllm-2080ti/run-logs/vllm-qwen27b-int4-tqk8v4-256K-mtp3-text-only-cu128-20260710-210715.log`，关键错误：

```
AssertionError: Workspace is locked but allocation from
'turboquant_attn.py:1913:_continuation_prefill' requires 128.75 MB,
current size is 128.00 MB. Workspace growth is not allowed after locking.
```

触发条件：

- 请求已有 `num_computed_tokens=65920` 在 KV cache 中；
- 本次续写 `num_scheduled_tokens=383`；
- TurboQuant attention 进入 `_continuation_prefill` 路径，需要 dequant 65920 个 cached tokens；
- `fast` mode 启用 CUDA graph，`WorkspaceManager` 在图捕获后被锁定，无法动态增长；
- 预分配的 continuation workspace 只有约 128 MB（基于 `max_num_batched_tokens=2560`），不足。

结论：`fast` mode + CUDA graph + TurboQuant + 长上下文 continuation 的组合存在 workspace 预分配不足的问题。

---

## 3. 修复措施

### 3.1 调整启动脚本 `serve_fast_tqk8v4_web.sh`

1. **降低默认 GPU_UTIL**：从 `0.90` 改为 `0.85`，避免启动阶段因显存检查阈值过高而偶发失败。
2. **真正让 `GPU_UTIL` 生效**：`launcher.sh` 的 `apply_profile_overrides` 会无条件用 profile 文件里的值覆盖环境变量，因此脚本现在：
   - 复制原 profile 到 `.tmp-profiles/`；
   - 用 `sed` 把临时 profile 里的 `GPU_UTIL` 改为脚本想要的值；
   - 通过 `PROFILE_FILE` 环境变量让 launcher 读取临时 profile。
   这样不改 weicj fork 的原 profile，也能让 `GPU_UTIL=0.85` 真正生效。
3. **预分配更大的 TurboQuant continuation workspace**：
   - 环境变量 `VLLM_TURBOQUANT_CONTINUATION_WORKSPACE_RESERVE_TOKENS=131072`
   - 默认值从 `0` 改为 `131072` tokens，覆盖绝大多数长上下文请求。
4. **关闭 CUDA graph memory profiling**：
   - `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0`
   - 释放少量有效显存，提高启动成功率。

改动后脚本关键片段：

```bash
GPU_UTIL="${GPU_UTIL:-0.85}"
VLLM_TURBOQUANT_CONTINUATION_WORKSPACE_RESERVE_TOKENS="${VLLM_TURBOQUANT_CONTINUATION_WORKSPACE_RESERVE_TOKENS:-131072}"
VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS="${VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS:-0}"

# 复制 profile 并覆盖 GPU_UTIL，避免 launcher.sh 用原 profile 的 0.90 覆盖环境变量
ORIG_PROFILE="${WEICJ_DIR}/profiles/qwen27b/fast/int4/tqk8v4-256K-mtp3-text-only.env"
TMP_PROFILE="${ROOT_DIR}/.tmp-profiles/qwen27b-fast-int4-tqk8v4-${LABEL}.env"
mkdir -p "${ROOT_DIR}/.tmp-profiles"
cp "${ORIG_PROFILE}" "${TMP_PROFILE}"
sed -i "s/^GPU_UTIL=.*/GPU_UTIL=${GPU_UTIL}/" "${TMP_PROFILE}"

env ... \
    PROFILE_FILE="${TMP_PROFILE}" \
    GPU_UTIL="${GPU_UTIL}" \
    VLLM_TURBOQUANT_CONTINUATION_WORKSPACE_RESERVE_TOKENS="..." \
    VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS="..." \
    ./launcher.sh --non-interactive
```

### 3.2 服务已恢复

重启后验证：

- `curl http://127.0.0.1:8000/health` ✅
- `curl http://127.0.0.1:18081/health` ✅
- OpenAI API 返回 `reasoning_content` ✅
- Anthropic 代理返回 `thinking` + `text` content blocks ✅

---

## 4. 当前状态

```
vLLM OpenAI API:  http://<YOUR_SERVER_IP>:8000/v1
Anthropic proxy:  http://<YOUR_SERVER_IP>:18081/v1/messages
Model:            qwen27b-int4-tqk8v4-256K-mtp3-text-only-cu128
GPU:              RTX 2080 Ti x2 (NVLink)
GPU_UTIL:         0.85
Continuation WS:  131072 tokens
```

---

## 5. 注意事项

1. **Thinking 输出占用 tokens**：启用 `enable_thinking=true` 后，模型会先生成 thinking 内容。如果 `max_tokens` 设得太小（如 64），可能只有 thinking 没有正文答案。建议对话场景 `max_tokens >= 512`。

2. **长上下文上限**：当前 workspace 预分配到 131072 tokens，可安全支持约 128K 的 continuation 请求。若需完整 256K 上下文，可增大 `VLLM_TURBOQUANT_CONTINUATION_WORKSPACE_RESERVE_TOKENS`，但会线性增加显存占用（约每 131072 tokens 几百 MB）。

3. **fast mode 的 trade-off**：fast mode 吞吐高，但 CUDA graph 对动态 shape/workspace 不友好。若仍遇到类似崩溃，可切到 `MODE=normal`（牺牲约 10-20% 吞吐换取稳定性）。

4. **3080 上 reason-lite 进程**：故障期间 `/mnt/hdd_storage/reason-lite/run_dual_vllm_3080.sh` 在 GPU 2/3 运行，但不影响 2080Ti 服务。当前已结束。

---

## 6. 后续可优化

1. 在 Anthropic 代理中支持按请求关闭 thinking（`thinking: {type: "disabled"}`），避免短问答时浪费 tokens。
2. 若 128K workspace 仍不足，可进一步提升到 262144 或评估切 normal mode。
3. 监控服务稳定性，记录再次崩溃时的请求长度和 workspace 需求。
