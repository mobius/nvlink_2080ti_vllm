# 2026-07-07 部署 Gradio 网页聊天界面

时间戳：2026-07-07 19:38 CST  
文档类型：impl

---

## 目标

在 vLLM OpenAI API 之上部署一个可直接在浏览器里聊天的网页界面，方便手动测试模型效果。

---

## 方案选择

- **Open WebUI**：功能完整，但依赖重、安装大、首次启动慢。
- **Gradio ChatInterface**：极简、安装小、流式输出、足够单用户测试。

最终选择 **Gradio**，在隔离虚拟环境 `.venv-gradio` 中运行，不污染 weicj vLLM 环境。

---

## 新增文件

- `tools/gradio_chat.py`：Gradio 聊天界面脚本，连接 `http://127.0.0.1:8000/v1`。
- `serve_gradio_chat.sh`：一键启动/停止脚本，自动创建 venv、安装依赖、启动服务。

---

## 安装过程

1. `uv venv .venv-gradio --python 3.11`
2. `uv pip install --python .venv-gradio/bin/python gradio openai`

安装依赖：gradio 6.19.0、openai 2.44.0。

---

## 运行状态

- 访问地址：**http://<YOUR_SERVER_IP>:7860**
- 后端 API：**http://<YOUR_SERVER_IP>:8000/v1**
- 模型：`qwen27b-int4-tqk8v4-256K-mtp3-text-only-cu128`
- Python PID：3682037
- Health check：`200 OK`

界面可调：
- Max tokens：64–4096
- Temperature：0.0–2.0

---

## 使用方式

```bash
# 启动
bash serve_gradio_chat.sh

# 停止
bash serve_gradio_chat.sh stop
```

环境变量可覆盖：
- `VLLM_BASE_URL`：vLLM API 地址（默认 `http://127.0.0.1:8000/v1`）
- `VLLM_MODEL`：模型名
- `GRADIO_PORT`：Gradio 端口（默认 7860）

---

## 备注

- 首次用 `serve_gradio_chat.sh` 启动时因 venv 内无 pip 导致安装失败，已修复为使用 `uv pip install --python`。
- 服务通过 Kimi background task 机制持久化运行，避免 shell 退出后进程被回收。
