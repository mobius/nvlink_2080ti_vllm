# ThinkingCap-Qwen3.6-27B-AWQ 下载完成

**时间**: 2026-07-13 16:02

## 状态

- `models/ThinkingCap-Qwen3.6-27B-AWQ/model.safetensors` 下载完成，大小 **25 GB**。
- 目录总大小约 **29 GB**，包含 tokenizer、chat template、config 等全部元数据。
- 当前服务器仍运行旧模型 `qwen27b-int4-tqk8v4-256K-mtp3-text-only-cu128`（路径 `models/Qwen3.6-27B-AWQ-INT4`）。
- GPU0/1（2080Ti）显存占用约 19.2 GB / 22.5 GB，服务正常。
- GPU2/3（3080）利用率 100%，正在跑训练任务，**不触碰**。

## 下一步

1. 更新 `serve_qwen36.sh` 的 `MODEL` 指向 `models/ThinkingCap-Qwen3.6-27B-AWQ`。
2. 安全停止当前只占用 GPU0/1 的 vLLM api_server 进程。
3. 使用 weicj fork fast/tqk8v4 启动新模型，验证健康后跑 PP4096/TG128 benchmark。
