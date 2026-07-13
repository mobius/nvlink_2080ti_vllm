# 2026-07-02 术语补充：PP4096 / TG128 评估口径

时间戳：2026-07-02 17:30 CST  
文档类型：research（术语说明）

---

## PP4096 / TG128 的含义

这是 LLM serving 性能 benchmark 中常用的**固定输入/输出长度测速口径**，便于不同方案之间横向对比吞吐。

| 缩写 | 全称 | 含义 |
|------|------|------|
| **PP** | **Prompt / Prefill Phase** | 输入阶段，即一次性并行处理输入 prompt |
| **4096** | prompt 长度 | 输入序列包含 **4096 个 token** |
| **TG** | **Token Generation / Decode Phase** | 输出阶段，即自回归逐 token 生成 |
| **128** | generation 长度 | 要求模型再生成 **128 个 token** |

因此 **PP4096/TG128** 指的是：
> 给模型喂 4096 tokens 的输入，让它生成 128 tokens 的输出，测量端到端延迟、prefill 吞吐、decode 吞吐等指标。

---

## 为什么用这个口径

1. **消除序列长度差异**：不同任务、不同用户的输入/输出长度差异极大。固定 PP/TG 后，不同配置、不同 fork、不同硬件之间的对比才有意义。
2. **关注 decode 瓶颈**：短输出（TG=128）时，prefill 占比很小，整体吞吐主要由 decode 阶段决定，因此常被用作**峰值 decode 吞吐**的测速口径。
3. **与 caovan/weicj 文章对齐**：很多 2080Ti 优化文章的吞吐数字（如 weicj 宣称的 101.3 tok/s）都基于 PP4096/TG128，所以本项目复现时也采用这一口径来验证是否达到宣称性能。

---

## 真实长输出场景会慢很多

PP4096/TG128 只能代表「短生成」峰值，不代表实际聊天/长文生成的端到端吞吐。原因：

- 真实对话往往要生成 1024 / 2048 / 4096 tokens 甚至更多。
- prefill 是一次性的，而 decode 与输出长度成正比；输出越长，prefill 的"一次性加速"被摊得越薄。
- 端到端吞吐（总 token / 总时间）会因 prefill 占比降低而显著下降。

举例：
- PP4096/TG128 下 decode 86 tok/s，但端到端平均也接近 86 tok/s（因为 prefill 很快）。
- 同样是 4096 prompt，若 TG=4096，端到端平均可能只有 40–50 tok/s 甚至更低。

---

## 本项目相关

- `run_fast_tqk8v4_bench.sh` 和 `run_full_eval.sh` 默认都跑 PP4096/TG128。
- 工具脚本：`tools/profile_request.py --prompt-tokens 4096 --gen-tokens 128`。
- 输出字段：
  - `ttft_s`：Time To First Token，首 token 返回时间（主要反映 prefill 延迟）
  - `prefill_tok_s`：`prompt_tokens / prefill_time`
  - `decode_tok_s`：`completion_tokens / decode_time`（即 PP4096/TG128 口径下的核心指标）

---

## 相关术语

- **Prefill**：输入 prompt 的一次性前向计算，算力密集，可高并行。
- **Decode**：自回归逐 token 生成，访存/延迟敏感，是吞吐瓶颈所在。
- **Throughput（吞吐）**：通常指 decode 阶段每秒生成的 token 数（tok/s）。
- **TTFT**：从收到请求到返回第一个生成 token 的时间，影响"首字延迟"体验。
