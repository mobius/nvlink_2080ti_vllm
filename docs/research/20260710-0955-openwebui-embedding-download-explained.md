# 2026-07-10 术语补充：Open WebUI 启动时下载 embedding 模型的原因

时间戳：2026-07-10 09:55 CST  
文档类型：research

---

## 现象

Open WebUI 首次启动时会从 HuggingFace 下载 `sentence-transformers/all-MiniLM-L6-v2`，即使我们只是用它来聊天。

## 原因

Open WebUI 默认把 **RAG（Retrieval-Augmented Generation）** 功能一起启动了。RAG 需要：

1. **Embedding 模型**：把上传的文档/知识库变成向量。
2. **向量数据库**（ChromaDB）：存储这些向量。

`all-MiniLM-L6-v2` 就是 Open WebUI 默认的轻量 embedding 模型，用于把文本转成向量。只要 RAG 功能开启，启动时就会尝试加载/下载它。

## 与纯聊天的关系

- 如果你只是聊天、不用"上传文档问答"，这个 embedding 模型**完全用不上**。
- 但它默认启用，所以首次启动必须下载完才能进入 Web UI。

## 如何避免

有几种方式：

### 方式 1：等它下完
第一次下完后会缓存，后续启动不再下载。

### 方式 2：禁用 RAG
在 Open WebUI 配置文件或环境变量里关闭 RAG/embedding：
- 环境变量：`RAG_EMBEDDING_MODEL=""` 或相关配置
- 但 Open WebUI 的配置比较复杂，有时禁用不完全。

### 方式 3：换更轻的聊天框架
如果只是纯聊天，不需要 RAG：
- **Gradio ChatInterface**（已部署）：最简单，无额外下载。
- **ChatGPT-Next-Web**：静态/单文件部署，无 embedding 下载。
- **LobeChat**：功能接近 Open WebUI，但默认不强制下载 embedding。

## 本项目选择

当前因为 Open WebUI 首次启动被 embedding 下载阻塞，如果用户只需要聊天，建议：
- 继续等待（适合后续想体验完整功能）
- 或换用 ChatGPT-Next-Web / LobeChat 等更轻量框架

已留档。
