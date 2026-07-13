# Open WebUI 部署与认证配置

**时间**: 2026-07-10 10:35  
**目标**: 在双 2080Ti 服务器上部署 Open WebUI，作为 vLLM 后端的网页聊天界面，支持直接通过浏览器聊天。  
**状态**: ✅ 已完成并验证

---

## 1. 当前环境

- **vLLM 后端**: `http://<YOUR_SERVER_IP>:8000/v1`
- **服务模型**: `qwen27b-int4-tqk8v4-256K-mtp3-text-only-cu128`
- **Open WebUI 前端**: `http://<YOUR_SERVER_IP>:18080`
- **Open WebUI 版本**: `0.10.2`
- **部署方式**: 本地隔离 venv（`.venv-openwebui`），使用 `uv` 管理

---

## 2. 部署步骤

### 2.1 一键脚本

已创建 `/mnt/hdd_storage/vllm_2080ti/serve_open_webui.sh`：

- 自动检查 vLLM 服务是否就绪
- 用 `uv` 创建 Python 3.11 隔离环境
- 自动安装 `open-webui`
- 自动配置 OpenAI API 连接到本地 vLLM
- 支持 `stop` 参数停止服务

### 2.2 关键环境变量

```bash
OPENAI_API_BASE_URL=http://127.0.0.1:8000/v1
OPENAI_API_KEY=sk-vllm
WEBUI_AUTH=true
WEBUI_ADMIN_NAME=Admin
WEBUI_ADMIN_EMAIL=admin@example.com
WEBUI_ADMIN_PASSWORD=<ADMIN_PASSWORD>
DATA_DIR=/mnt/hdd_storage/vllm_2080ti/cache/open-webui
WEBUI_PORT=18080
HOST=0.0.0.0
```

> 注：Open WebUI 的 `serve` 命令使用 typer 显式参数 `--host`/`--port`，不会自动读取 `PORT`/`HOST` 环境变量。脚本中必须写成 `open-webui serve --host ${WEBUI_HOST} --port ${WEBUI_PORT}` 才能生效。

> 注：当前数据库中已有的管理员密码为 `<ADMIN_PASSWORD>`（早期手动注册时设置）。脚本中的 `WEBUI_ADMIN_PASSWORD` 仅在新数据库首次启动时生效。

---

## 3. 认证问题与处理

### 3.1 初次尝试：WEBUI_AUTH=false

最初希望实现"免登录直接聊天"，将 `WEBUI_AUTH=false`。但发现：

- Open WebUI 在数据库中存在用户后，会完全关闭认证相关端点（signin/signup）。
- 但前端 API（如 `/api/models`）仍要求有效的用户 token。
- 结果：已有用户 + `WEBUI_AUTH=false` 会导致前端无法正常工作，signin 返回 `400`：`"You can't turn off authentication because there are existing users."`

### 3.2 解决方案

改为 `WEBUI_AUTH=true`，并预先创建默认管理员账号：

- 首次启动时，Open WebUI 检测到数据库为空，会根据 `WEBUI_ADMIN_EMAIL`/`WEBUI_ADMIN_PASSWORD` 自动创建管理员。
- 用户打开网页后用默认账号登录即可聊天。
- 登录后获得 JWT token，后续 API 调用通过 `Authorization: Bearer <token>` 认证。

### 3.3 验证登录与模型列表

```bash
# 登录
curl -X POST http://<YOUR_SERVER_IP>:18080/api/v1/auths/signin \
  -H "Content-Type: application/json" \
  -d '{"email":"admin@example.com","password":"<ADMIN_PASSWORD>"}'

# 返回 JWT token

# 获取模型列表
curl http://<YOUR_SERVER_IP>:18080/api/models \
  -H "Authorization: Bearer <token>"
```

模型列表正常返回，包含已连接的 vLLM 模型。

---

## 4. 聊天功能验证

通过 Open WebUI 的 OpenAI 兼容端点测试：

```bash
curl -X POST http://<YOUR_SERVER_IP>:18080/openai/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <token>" \
  -d '{
    "model": "qwen27b-int4-tqk8v4-256K-mtp3-text-only-cu128",
    "messages": [{"role": "user", "content": "你好，请用一句话介绍自己"}],
    "max_tokens": 50,
    "stream": false
  }'
```

返回：

```json
{
  "choices": [{
    "message": {
      "role": "assistant",
      "content": "我是通义千问，由阿里巴巴集团通义实验室独立开发的大型语言模型。"
    }
  }]
}
```

聊天链路正常。

---

## 5. 使用说明

1. 浏览器访问：`http://<YOUR_SERVER_IP>:18080`
2. 使用默认管理员账号登录：
   - 邮箱：`admin@example.com`
   - 密码：`<ADMIN_PASSWORD>`
3. 登录后在左上角选择模型 `qwen27b-int4-tqk8v4-256K-mtp3-text-only-cu128`
4. 直接开始聊天

---

## 6. 相关文件

- 一键脚本：`/mnt/hdd_storage/vllm_2080ti/serve_open_webui.sh`
- 数据目录：`/mnt/hdd_storage/vllm_2080ti/cache/open-webui`
- 运行日志：`/mnt/hdd_storage/vllm_2080ti/logs/open-webui-server.log`
- 安装日志：`/mnt/hdd_storage/vllm_2080ti/logs/open-webui-install.log`

---

## 7. 已知限制

- Open WebUI 在有用户后不支持完全匿名访问；必须至少登录一次。
- 默认管理员账号建议用户在首次登录后到设置中修改密码。
- 若需重置为全新安装，可停止服务、删除 `cache/open-webui` 目录后重新启动。
