# 实施文档（迭代 2）：插件源码审计 + 版本核实 + 安装启动

- 迭代时间：2026-07-01 18:55:55 CST
- 阶段：impl（第 2 轮）
- 触发：用户提供第二篇文章 + 本地已存在插件 zip `caovan-vllm-sm75-turbo3-v0.1.3-external-plugin.zip`

---

## 0. 对第 1 轮结论的重要订正（诚实留痕）

第 1 轮 research 文档基于我**过时的知识**，把以下判为"不存在/疑似伪造"。经在 PyPI / HuggingFace **实际查询**后，均证实为 2026 年的**真实发布**，特此订正：

| 项目 | 第 1 轮误判 | 实测（2026-07-01） |
|---|---|---|
| vLLM 0.21.0 | "版本号不符" | ✅ PyPI 存在（最新已到 0.24.0） |
| PyTorch 2.11.0 | "尚无此版本" | ✅ PyPI 存在（最新已到 2.12.1） |
| CUDA 13.0 | 存疑 | ✅ 真实，driver 580 支持 |
| Qwen3.6-27B | "无对应公开发布" | ✅ 官方 `Qwen/Qwen3.6-27B` 在 HF；有多个 AWQ-INT4 量化版 |

教训：当前系统日期是 2026-07，训练知识早于此，涉及"最新版本是否存在"必须用工具查证，不能凭记忆下结论。

第 1 轮仍然成立的部分：硬件匹配、GPU2/3 被占用需隔离、SM75 无原生 FP8、256K 上下文激进——这些是硬件/架构事实，不受知识时效影响。

---

## 1. 两篇文章 / 两个插件版本的关系

- 第一篇：v0.4.33（付费会员，本地**没有**）。
- 第二篇：**v0.1.3（公开版，本地已有 zip）** ← 本次实际采用。
- v0.1.3 定位："外部插件修复版"，只优化 Qwen GDN + MTP=2 speculative decode 的解码路径；不改 vLLM 源码、不依赖 NVLink、不绑定模型路径/量化格式。

## 2. 插件源码安全审计结论：通过

对 `src/caovan_vllm_sm75_turbo3/` 全部源码审阅：

- **入口**：`plugin.py:register()`，经 `vllm.general_plugins` 入口加载。仅当 `additional_config.caovan_sm75_turbo3=true` **且** 运行在 SM75 CUDA **且** 该 GDN 层 `num_spec==2`(MTP=2) 时才启用；否则回退 upstream 原路径。
- **机制**：对 Qwen GatedDeltaNet 类的 `__init__` 与 `_forward_core` 做 monkey-patch（进程内，不落盘、不改 vLLM 文件）。`compat.py` 运行时动态识别两代 GDN 接口（legacy / modular-qwen），不写死单一模块路径。
- **kernel**：`kernels/gdn.py`（896 行）为真实 Triton kernel，`import from vllm.triton_utils` 与 `vllm.model_executor.layers.fla.ops.op`，做 split-QK 复用 + V/GDN 状态更新融合。
- **安全扫描**：无 `subprocess`/`os.system`/`eval`/`exec`/`socket`/`pickle`/`base64`。唯一网络调用在 `benchmark.py`，用 `urllib` POST 到**本地** vLLM API 做测速（正当用途）。dispatch 失败时 `raise`（不隐藏错误），设计谨慎。
- 结论：可安全安装。

## 3. 版本兼容矩阵（来自代码常量 + PyPI 元数据）

- 插件 `__init__.py`：
  - `PERF_VERIFIED_VLLM_VERSIONS = {"0.20.2rc1.dev118+g10ebb40d6"}`（作者测速用的开发版）
  - `INTERFACE_VALIDATED_VLLM_VERSIONS = {"0.21.1rc1.dev387+g5d126dd15"}`
  - 注意：**稳定版 0.21.0 不在这两个集合里** → doctor 会显示"未实测；仅接口检查通过"。文章正文却以 0.21.0 为推荐环境，属于文案与代码常量的轻微不一致，但不影响接口层可用性（compat 动态探测）。
- vLLM 0.21.0 依赖：`torch==2.11.0`、`torchvision==0.26.0`、`flashinfer-python==0.6.8.post1`；`requires-python >=3.10,<3.15`。
- 模型静态候选条件（`model_check.py`）：config 需含 `linear_num_key_heads/linear_num_value_heads/linear_key_head_dim/linear_value_head_dim/linear_conv_kernel_dim`，且 `linear_conv_kernel_dim==4`。

## 4. 本轮动作

1. `uv venv --python 3.11 .venv`（已完成，隔离环境）。
2. 项目内缓存目录：`cache/hf`、`cache/torchinductor`、`cache/triton`、`logs/`、`models/`（避免污染全局 `~/.cache`）。
3. 后台安装：`uv pip install vllm==0.21.0`（日志 `logs/install_vllm.log`）。
4. 待装完 → 验证 `torch.version.cuda` 与 **SM75 kernel 能否在 2080Ti 上实际执行**（CUDA 13 是否保留 sm_75 是本步最大不确定性）。
5. 通过后再 `pip install` 插件 wheel → `caovan-sm75-doctor` / `caovan-sm75-verify`。

## 5. 关键约束（贯穿全程）

- `CUDA_VISIBLE_DEVICES=0,1`，**绝不触碰 GPU2/3**（<USER> 的 SFT 训练仍在跑）。
- 所有下载/缓存落项目目录，不 `sudo`，不写系统全局。
- 模型下载（~15-20GB）放到 SM75 验证通过之后，避免无谓流量。
