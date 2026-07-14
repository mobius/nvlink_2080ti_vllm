# nvlink_2080ti_vllm

在 **2×RTX 2080Ti 22G (NVLink, SM75)** 上复现并部署 **Qwen3.6-27B-AWQ-INT4** 长上下文推理服务的完整记录与一键脚本。

---

## 仓库用途

本项目用于：

1. **复现 caovan.com 的 SM75 2080Ti vLLM 优化方案**（caovan v0.1.3 插件 + weicj/vLLM-2080Ti-Definitive fork 两条路线）。
2. **部署可对外提供 API 的本地 LLM 服务**：
   - OpenAI 兼容 API：`http://<YOUR_SERVER_IP>:8000/v1`
   - Anthropic Messages 兼容代理：`http://<YOUR_SERVER_IP>:18081/v1/messages`
   - Gradio 网页聊天界面：`http://<YOUR_SERVER_IP>:7860`
   - Open WebUI：`http://<YOUR_SERVER_IP>:18080`
3. **记录完整工程过程**：环境评估、路线选择、编译踩坑、性能基准、质量评估、WebUI 接入、Agent 接入、长上下文崩溃修复等，按 `docs/{research,plan,impl,architecture}` 归档。

所有环境均落在项目目录内（`.venv*` 隔离），不污染系统全局环境；GPU 使用硬隔离到 2080Ti（`CUDA_VISIBLE_DEVICES=0,1`），避免影响本机其它 GPU 上的训练任务。

---

## 快速开始

### 1. 环境要求

- 2×RTX 2080Ti 22G + NVLink（SM75）
- CUDA Toolkit 12.8
- `g++-12`
- `uv`（Python 包管理，优先）
- 模型：`cyankiwi/Qwen3.6-27B-AWQ-INT4` 或等效 AWQ/GPTQ INT4 checkpoint

### 2. 启动核心服务（weicj 路线，256K 上下文）

```bash
cd /mnt/hdd_storage/vllm_2080ti
bash serve_fast_tqk8v4_web.sh
```

可选环境变量覆盖：

```bash
MODEL_DIR=/mnt/hdd_storage/vllm_2080ti/models/ThinkingCap-Qwen3.6-27B-AWQ \
SERVED_NAME=thinkingcap-qwen3.6-27b-awq \
GPU_UTIL=0.85 \
PORT=8000 \
SERVICE_SCOPE=lan \
REASONING_PARSER=qwen3 \
DEFAULT_CHAT_TEMPLATE_KWARGS='{"enable_thinking":true}' \
ENABLE_AUTO_TOOL_CHOICE=1 \
TOOL_CALL_PARSER=qwen3_xml \
bash serve_fast_tqk8v4_web.sh
```

说明：
- 所有服务脚本和评估脚本都支持通过 `MODEL_DIR` / `SERVED_NAME` / `MTP_K` 等环境变量切换模型。例如换成 `sahilchachra/ThinkingCap-Qwen3.6-27B-AWQ` 时，设置 `MODEL_DIR` 指向下载后的本地目录，并设置 `MTP_K=0`（该模型无 MTP 头）。
- `ENABLE_AUTO_TOOL_CHOICE=1` + `TOOL_CALL_PARSER=qwen3_xml` 已默认开启，支持 Agent 发送 `tool_choice: auto`；如果客户端不需要工具调用，可设 `ENABLE_AUTO_TOOL_CHOICE=0` 关闭。

### 3. 启动 Anthropic 兼容代理（给只支持 Claude API 的 Agent 用）

```bash
bash serve_anthropic_proxy.sh
```

Agent 配置示例：

```yaml
base_url: http://<YOUR_SERVER_IP>:18081
api_key: sk-test            # 任意值，未设置 PROXY_API_KEY 时不校验
model: qwen27b-int4-tqk8v4-256K-mtp3-text-only-cu128
```

### 4. 启动 Tau coding agent（终端 Agent）

需要先确保 vLLM 服务已在 `0.0.0.0:8000` 运行，然后：

```bash
bash setup_tau_local_vllm.sh   # 创建 .venv-tau、安装 tau-ai、配置 ~/.tau/
bash run_tau_smoke_test.sh     # 运行基础对话 + bash/read/write 验证
```

日常非交互式使用：

```bash
source .venv-tau/bin/activate
export VLLM_API_KEY=sk-vllm
tau -p "请用 bash 工具查看当前目录" --output text
```

进入 TUI：

```bash
tau
```

详细说明见 `docs/impl/20260714-103000-tau-agent-local-vllm-validation.md`。

### 5. 启动网页聊天界面（二选一）

```bash
# Gradio 轻量界面
bash serve_gradio_chat.sh

# Open WebUI（功能更全，支持工具调用、RAG 等）
WEBUI_ADMIN_PASSWORD=YourStrongPassword bash serve_open_webui.sh
```

---

## 目录结构

```
.
├── docs/
│   ├── research/          # 参考文章、环境评估、术语研究
│   ├── plan/              # 复现路线与决策
│   ├── impl/              # 每次迭代的实施记录（时间戳命名）
│   ├── architecture/      # 部署架构与隔离方案
│   ├── glossary.md        # 术语表
│   └── SUMMARY.md         # 复现总结
├── tools/
│   ├── anthropic_proxy.py # Anthropic Messages API -> vLLM OpenAI API 代理
│   └── gradio_chat.py     # Gradio 聊天前端
├── serve_*.sh             # 各类一键启动脚本
├── run_*.sh               # 性能/质量/Tau 评估脚本
├── setup_tau_local_vllm.sh # Tau agent 一键安装配置脚本
├── serve_qwen36.sh        # caovan 插件路线启动脚本
└── weicj-vllm-2080ti/     # git submodule，weicj/vLLM-2080Ti-Definitive fork
```

---

## 关键成果

- 两条路线均复现成功：caovan v0.1.3 插件路线与 weicj Definitive fork 路线。
- 256K 上下文在 2×2080Ti 22G 上可运行。
- weicj 路线 `fast/tqk8v4` profile 在 PP4096/TG128 口径下约 **90 tok/s**（AWQ-INT4）。
- 大海捞针（needle-in-haystack）长上下文测试通过。
- 完整支持 reasoning/thinking 输出、auto tool choice、Anthropic API 代理、Tau terminal agent 接入。
- 服务脚本与评估脚本已模型无关化：通过 `MODEL_DIR` / `SERVED_NAME` / `MTP_K` 等环境变量即可切换同系列 AWQ/GPTQ INT4 checkpoint，已验证可接入 `sahilchachra/ThinkingCap-Qwen3.6-27B-AWQ`。
  - 该模型**未保留 MTP 头**，启用 `MTP_K=3` 会反噬吞吐；正确用法是 `MTP_K=0`，此时 PP4096/TG128 decode 约 **32 tok/s**。

详细数据与踩坑过程见 `docs/impl/` 各文件。

---

## 安全与隐私

- 仓库中不包含真实密码、私钥或模型权重。
- 默认 API key（如 `sk-vllm`、`sk-test`）仅为占位符，生产环境请通过环境变量覆盖。
- `serve_open_webui.sh` 不再硬编码管理员密码，必须通过 `WEBUI_ADMIN_PASSWORD` 环境变量传入。
- 本地运行时产生的日志、缓存、虚拟环境、secret key 等已加入 `.gitignore`。

---

## 参考

- caovan.com SM75 2080Ti 安装教程
- [weicj/vLLM-2080Ti-Definitive](https://github.com/weicj/vLLM-2080Ti-Definitive)
- 模型：`cyankiwi/Qwen3.6-27B-AWQ-INT4`（默认）或 `sahilchachra/ThinkingCap-Qwen3.6-27B-AWQ` 等同级别 Qwen3.5/3.6 AWQ-INT4 checkpoint

---

## License

父仓库文档与脚本按仓库实际需要自行管理；`weicj-vllm-2080ti/` 子模块遵循其原仓库许可证。
