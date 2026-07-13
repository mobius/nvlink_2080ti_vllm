# 模型无关化改造与 ThinkingCap-Qwen3.6-27B-AWQ 接入准备

**时间**: 2026-07-13 15:05  
**背景**: 用户希望把评估/部署的模型从默认的 `cyankiwi/Qwen3.6-27B-AWQ-INT4` 替换为 `sahilchachra/ThinkingCap-Qwen3.6-27B-AWQ`，并验证 2080Ti 方案可泛化到同级别的 Qwen3.5/3.6 AWQ INT4 模型。

## 改造内容

### 1. 服务脚本支持 `MODEL_DIR` / `SERVED_NAME` 环境变量覆盖

- `serve_fast_tqk8v4_web.sh`
  - `MODEL_DIR` 默认保持 `/mnt/hdd_storage/vllm_2080ti/models/Qwen3.6-27B-AWQ-INT4`
  - 新增 `SERVED_MODEL_NAME` 默认保持 `qwen27b-int4-tqk8v4-256K-mtp3-text-only-cu128`
  - 启动日志打印当前 `MODEL_DIR` 和 `SERVED_MODEL_NAME`
- `serve_anthropic_proxy.sh`
  - 新增 `MODEL_NAME` 环境变量，默认同上
  - Agent 配置示例中的模型名改为动态输出
- `serve_gradio_chat.sh` / `serve_open_webui.sh`
  - 已通过 `VLLM_MODEL` 支持覆盖

### 2. 评估脚本支持模型切换

- `run_fast_tqk8v4_bench.sh`
  - `MODEL_DIR` 可覆盖
  - `SERVED_NAME` 可覆盖
- `run_full_eval.sh`
  - `MODEL_DIR` 可覆盖
  - `SERVED_NAME` 可覆盖
  - `BASE_URL` 可覆盖
- `run_quality_eval.sh`
  - 已通过 `SERVED_NAME` / `BASE_URL` 支持覆盖，补充注释

### 3. Open WebUI 默认密码安全改造

- `serve_open_webui.sh` 不再硬编码 `Vllm2080Ti!`
- 若 `WEBUI_AUTH=true` 且未设置 `WEBUI_ADMIN_PASSWORD`，启动失败并提示用户设置

### 4. 文档脱敏

- 所有硬编码公网 IP `30.19.40.129` 替换为 `<YOUR_SERVER_IP>`
- 局域网 IP `192.168.100.1` 替换为 `<YOUR_LAN_IP>`
- 用户名 `joey` / 主机名 `bingxiaoliu` 替换为占位符
- 历史密码 `admin12345` / `Vllm2080Ti!` 替换为 `<ADMIN_PASSWORD>`
- 重新 force push 到 GitHub，commit `5c4d369`

## 新模型信息

- **模型 ID**: `sahilchachra/ThinkingCap-Qwen3.6-27B-AWQ`
- **架构**: `Qwen3_5ForConditionalGeneration`
- **模型类型**: `qwen3_5`
- **量化**: `compressed-tensors` / `pack-quantized` / INT4 / group_size=128
- **最大上下文**: 262144 tokens
- **格式**: 单文件 `model.safetensors`
- **视觉模块**: 配置中包含 `vision_config`，`language_model_only: false`
- **用途**: 据称减轻 Qwen3.6 过度 thinking

## 下载状态

使用 wget 从 Hugging Face 主站断点续传下载中：

```bash
cd /mnt/hdd_storage/vllm_2080ti/models/ThinkingCap-Qwen3.6-27B-AWQ
wget -c -O model.safetensors \
  "https://huggingface.co/sahilchachra/ThinkingCap-Qwen3.6-27B-AWQ/resolve/main/model.safetensors?download=true"
```

下载速度约 40 MB/s，预计总大小约 10 GB，完成时间约 1 小时 50 分钟。

## 待验证项

1. 新模型下载完成后，用 `MODEL_DIR=/mnt/hdd_storage/vllm_2080ti/models/ThinkingCap-Qwen3.6-27B-AWQ SERVED_NAME=thinkingcap-qwen3.6-27b-awq bash serve_fast_tqk8v4_web.sh` 启动
2. 由于新模型含视觉配置，需测试是否必须去掉 `--language-model-only` / `--skip-mm-profiling`
3. 短对话验证 reasoning/thinking 输出是否更简洁
4. 跑 PP4096/TG128 benchmark 对比吞吐
5. 跑质量评估对比答案质量

## 使用示例

```bash
# 启动新模型服务
MODEL_DIR=/mnt/hdd_storage/vllm_2080ti/models/ThinkingCap-Qwen3.6-27B-AWQ \
SERVED_NAME=thinkingcap-qwen3.6-27b-awq \
REASONING_PARSER=qwen3 \
DEFAULT_CHAT_TEMPLATE_KWARGS='{"enable_thinking":true}' \
bash serve_fast_tqk8v4_web.sh

# 启动 Anthropic 代理（模型名同步）
MODEL_NAME=thinkingcap-qwen3.6-27b-awq bash serve_anthropic_proxy.sh

# 跑完整评估
MODEL_DIR=/mnt/hdd_storage/vllm_2080ti/models/ThinkingCap-Qwen3.6-27B-AWQ \
SERVED_NAME=thinkingcap-qwen3.6-27b-awq \
bash run_full_eval.sh thinkingcap
```
