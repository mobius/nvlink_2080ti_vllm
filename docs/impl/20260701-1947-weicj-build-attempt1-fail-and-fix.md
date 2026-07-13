# 实施文档（迭代 5）：weicj build 第 1 次失败 + 修复方案

- 迭代时间：2026-07-01 19:47 CST
- 阶段：impl（第 5 轮，轨道 B）

---

## 1. 现象

`build.sh`（`ASSUME_YES=1 MAX_JOBS=24 CUDA_HOME=/usr/local/cuda-12.8`）在第 3 步「Install CUDA runtime requirements」失败：

```
Selected-route index attempt after 00:01:00  FAILED   (x2)
Mirror index attempt ... Resolved 179 packages
error: Failed to install: z3_solver-4.15.4.0-...whl (z3-solver==4.15.4.0)
  Caused by: RECORD file doesn't match wheel contents, could not find entry for:
  z3_solver-4.15.4.0.data/data/bin/.tmp2SpB0H/z3
BUILD FAILED — Step failed: Install CUDA runtime requirements
```

重步骤（CUDA 源码编译）尚未开始，无编译产物损失；`.venv/bin/vllm` 未生成。

## 2. 归因

1. **带宽竞争**：与 19GiB 模型下载并发，官方索引源两次 60s 超时 → 切镜像。
2. **z3-solver 解包竞态/缓存问题**：`.tmp2SpB0H` 是本次解包的随机临时名，属 uv 处理 z3-solver data-scripts 时的竞态；叠加 uv cache 与项目跨文件系统（hardlink 失败回退 copy）更易触发。

均为**环境/瞬时**问题，非本机能力或 fork 本身缺陷。

## 3. 修复方案（待模型下载完、带宽释放后执行）

1. 等 `bash-m6rr9jv7` 通知（模型下完，带宽释放）。
2. 重跑 build，追加：
   - `UV_LINK_MODE=copy`（消除跨文件系统 hardlink 警告与相关竞态）。
   - 保持 `ASSUME_YES=1 MAX_JOBS=24 CUDA_HOME=/usr/local/cuda-12.8`。
3. 若 z3 仍失败：先 `uv cache clean z3-solver`（给足超时），再重跑；必要时在 requirements 中固定一个可用 z3-solver 版本（最后手段，尽量不改 fork 源）。
4. build.sh 会重建/复用 `.venv` 并从失败步继续；重步骤（sm_75 全量编译）预计 20-45 分钟。

## 4. 不影响的部分

- 轨道 A（caovan 插件）已独立验证通过，与本失败无关。
- GPU2/3 训练不受影响（编译是 CPU 活）。
- 所有产物仍在 `weicj-vllm-2080ti/`（独立 .venv/.deps），可随时清理重来。
