# 接入 Tau agent 并验证本地 LLM 工具调用

**时间**：2026-07-14 10:30  
**目标**：根据 [twotimespi.dev/guides](https://twotimespi.dev/guides) 安装/配置 Tau coding agent，使用本地部署的 `ThinkingCap-Qwen3.6-27B-AWQ-tqk8v4-256K` 模型完成基础对话、bash、read、write 工具调用验证。  

---

## Tau 简介

[Tau](https://twotimespi.dev/) 是一个教育性的 Python terminal coding agent，由 HuggingFace 发布。架构分三层：

- `tau_ai`：provider 适配层，把各家模型 API 转成统一事件流。
- `tau_agent`：可复用的 agent loop（messages、tools、transcript、session）。
- `tau_coding`：coding 应用层，提供 CLI/TUI、文件/Shell 工具、项目管理等。

Tau 支持 OpenAI、Anthropic、OpenRouter、Hugging Face 以及自定义 OpenAI-compatible 端点，因此可以接入本地 vLLM。

---

## 环境准备

### 1. 安装 tau-ai

Tau 需要 **Python 3.12+**。本机系统 Python 为 3.10，因此用 uv 创建独立环境：

```bash
cd /mnt/hdd_storage/vllm_2080ti
uv venv --python 3.12 .venv-tau
source .venv-tau/bin/activate
uv pip install tau-ai
```

安装后 `tau --help` 可正常使用。

### 2. 创建 provider 配置

Tau 通过用户级配置目录 `~/.tau/` 加载 provider：

- `~/.tau/catalog.toml`：provider 与模型元数据
- `~/.tau/providers.json`：运行时偏好（timeout、默认模型等）
- `~/.tau/credentials.json`：可选，存储 API key

本配置把本地 vLLM 注册为 `local-vllm` provider：

**~/.tau/catalog.toml**

```toml
schema_version = 1

[[providers]]
name = "local-vllm"
display_name = "Local vLLM (2080Ti)"
kind = "openai-compatible"
base_url = "http://127.0.0.1:8000/v1"
api_key_env = "VLLM_API_KEY"
credential_name = "local-vllm"
models = ["ThinkingCap-Qwen3.6-27B-AWQ-tqk8v4-256K"]
default_model = "ThinkingCap-Qwen3.6-27B-AWQ-tqk8v4-256K"
docs_url = "http://30.19.40.129:8000/docs"
api = "openai-completions"
thinking_levels = ["off", "low", "medium", "high"]
thinking_default = "off"
thinking_parameter = "reasoning_effort"

[providers.context_windows]
"ThinkingCap-Qwen3.6-27B-AWQ-tqk8v4-256K" = 262144

[providers.model_metadata."ThinkingCap-Qwen3.6-27B-AWQ-tqk8v4-256K"]
name = "ThinkingCap Qwen3.6 27B AWQ"
reasoning = true
input = ["text"]
context_window = 262144
max_tokens = 32000
```

**~/.tau/providers.json**

```json
{
  "default_provider": "local-vllm",
  "provider_preferences": {
    "local-vllm": {
      "default_model": "ThinkingCap-Qwen3.6-27B-AWQ-tqk8v4-256K",
      "timeout_seconds": 600,
      "max_retries": 1,
      "max_retry_delay_seconds": 1
    }
  },
  "scoped_models": []
}
```

说明：
- `base_url` 指向本地 vLLM OpenAI 兼容端点。
- `api_key_env = "VLLM_API_KEY"` 配合环境变量 `export VLLM_API_KEY=sk-vllm` 使用；vLLM 本身不校验 key，但 Tau 会发送 Authorization header。
- `thinking_default = "off"`：ThinkingCap 仍会输出 thinking 内容，但 Tau 默认不请求 reasoning_effort，减少前端解析负担。

### 3. 一键配置脚本

项目已新增 `setup_tau_local_vllm.sh`，自动完成环境创建、tau-ai 安装和配置写入，并备份已有的 `~/.tau` 配置。

---

## 功能验证

使用 `run_tau_smoke_test.sh` 一键跑验证，覆盖 4 类场景：

### 1. 基础对话

```bash
export VLLM_API_KEY=sk-vllm
tau -p "hi" --output text
```

输出：

```text
Hi! How can I help you today?
```

### 2. bash 工具调用

```bash
tau -p "请用 bash 工具列出当前目录下的前 5 个文件" --output text
```

输出：

```text
1. `cache`
2. `docs`
3. `logs`
4. `models`
5. `_plugin_inspect`
```

### 3. read 工具调用

```bash
tau -p "请用 read 工具读取 README.md 的前 3 行并总结项目用途" --output text
```

输出：对项目用途的准确中文总结。

### 4. write + bash 组合工具调用

```bash
tau --cwd /mnt/hdd_storage/vllm_2080ti/.tmp-tau-test \
  -p "请创建一个 test_tau.py 文件，内容是打印 'Hello from Tau agent!'，然后用 bash 工具运行它" \
  --output text
```

输出：

```text
文件已创建并成功运行，输出 `Hello from Tau agent!`。
```

实际生成的 `test_tau.py`：

```python
def main():
    print('Hello from Tau agent!')

if __name__ == '__main__':
    main()
```

运行结果：

```text
Hello from Tau agent!
```

---

## 验证结论

- Tau 成功接入本地 vLLM ThinkingCap 模型。
- 基础对话、bash、read、write 四类工具调用均正常工作。
- 本地 LLM 的 `tool_choice: auto` 由 Tau 的 agent loop 自动管理，vLLM 后端已开启 `--enable-auto-tool-choice --tool-call-parser qwen3_xml`。

---

## 注意事项

1. **环境隔离**：`.venv-tau` 是独立环境，Python 3.12，不影响 vLLM 的 Python 3.11 环境。
2. **配置位置**：Tau 的 provider 配置固定在 `~/.tau/`（用户级），无法放到项目目录。`setup_tau_local_vllm.sh` 会备份已有配置。
3. **TUI 模式**：直接运行 `tau` 会进入交互式 Textual TUI。在 SSH/无终端环境下使用 `tau -p "..." --output text` 非交互模式。
4. **thinking 内容**：模型默认仍会输出 reasoning/thinking，如果 Tau 前端解析异常，可在 vLLM 启动时关闭：`DEFAULT_CHAT_TEMPLATE_KWARGS='{"enable_thinking":false}'`。
5. **timeout**：本地模型首 token 延迟较高，providers.json 中 `timeout_seconds` 设为 600 秒。

---

## 新增文件

- `setup_tau_local_vllm.sh`：一键安装 Tau 并配置本地 vLLM provider。
- `run_tau_smoke_test.sh`：Tau 功能验证脚本。
- `docs/impl/20260714-103000-tau-agent-local-vllm-validation.md`：本文档。
