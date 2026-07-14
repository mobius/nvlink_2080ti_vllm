# 术语表 Glossary

本文件记录复现过程中出现的技术词汇。每条给出简明解释 + 与本项目的关系。按首次出现时间追加。

---

## 2026-07-01 首批（来自 Caovan 指南）

- **SM75 / Compute Capability 7.5**：NVIDIA GPU 架构代号。SM75 = Turing 架构，RTX 2080Ti 属于此代。决定了支持哪些 CUDA 特性（例如 SM75 无原生 FP8）。`capability=(7,5)` 就是指它。

- **NVLink / NV2**：NVIDIA 的 GPU 间高速直连总线，带宽远高于走 CPU 的 PCIe。`nvidia-smi topo -m` 里 `NV2` 表示两张卡之间有 2 条 bonded NVLink。本机 GPU0↔GPU1 为 NV2，约 51 GB/s，利于张量并行(TP)时的 all-reduce 通信。

- **TP / tensor-parallel（张量并行）**：把单层权重矩阵切分到多张 GPU 上并行计算，用于让单卡放不下的大模型跨卡运行。`--tensor-parallel-size 2` 即用 2 张卡切分。跨卡通信频繁，因此 NVLink 很关键。

- **vLLM**：高吞吐 LLM 推理/服务引擎，核心特性是 PagedAttention（显存分页）与连续批处理(continuous batching)，提供 OpenAI 兼容 API。

- **AWQ-INT4**：Activation-aware Weight Quantization，把权重量化到 4-bit 整数以省显存，同时用激活感知策略保护精度。INT4 权重让 27B 级模型能塞进小显存。

- **KV cache / `--kv-cache-dtype fp8`**：自回归推理时缓存历史 token 的 Key/Value 张量以避免重复计算，是长上下文显存的大头。用 FP8(8-bit 浮点)存 KV 可减半显存。注意 SM75 无原生 FP8 支持，此项在 2080Ti 上有兼容性风险。

- **GDN（Gated DeltaNet）**：一种线性注意力/状态空间类机制（DeltaNet 的门控变体），计算随序列长度线性增长而非二次方，适合超长上下文。Qwen3-Next 等混合架构用它替换部分标准注意力层。指南中的 "GDNCore"、"GDN prefill" 均围绕它。

- **Mamba / `--mamba-cache-mode`**：Mamba 是状态空间模型(SSM)，与 GDN 同属线性序列建模家族，需要维护一个「状态缓存」而非传统 KV cache。`--mamba-cache-mode align` 用于对齐这类缓存布局。

- **MTP（Multi-Token Prediction，多 token 预测）**：模型一次前向预测未来多个 token，配合投机解码(speculative decoding)提升生成吞吐。`MTP=3` 表示一次预测 3 个草稿 token。MTP 越大越省前向次数，但草稿越可能被拒、且更吃显存（指南称 MTP=4 在 2080Ti 上易 OOM）。

- **Speculative decoding（投机解码）/ draft acceptance rate（草稿接受率）**：用一个便宜的"草稿"过程先猜若干 token，再由主模型一次性校验、接受或拒绝。接受率越高，加速越明显。指南的 "AcceptanceLock" 声称用于稳定接受率。

- **Prefill vs Decode**：LLM 推理两阶段。Prefill = 一次性并行处理输入 prompt（算力密集）；Decode = 逐 token 自回归生成（访存/延迟敏感）。指南的 "real-prefill"、"decode path" 指分别优化这两条路径。

- **prefix caching / `--enable-prefix-caching`**：缓存不同请求共享的相同前缀的 KV，避免重复 prefill，提升多请求场景吞吐。

- **ptxas**：NVIDIA CUDA 工具链里把 PTX 中间码汇编成 GPU 机器码(SASS)的组件。Triton 编译 kernel 时会调用它，`TRITON_PTXAS_PATH` 用于指定其路径。

- **Triton**：OpenAI 出的 GPU kernel 编译器/DSL，vLLM 很多算子用 Triton 写。首次运行会 JIT 编译并缓存(故"第一次很慢")。

- **TorchInductor / `TORCHINDUCTOR_CACHE_DIR`**：PyTorch 2.x 的编译后端(torch.compile)，把计算图编译成高效 kernel。缓存目录避免每次重编译。

- **AWQ Marlin / kernel 与 SM 版本约束**：vLLM 中量化(AWQ/GPTQ)的高速 Marlin kernel 通常要求 SM80+(Ampere 及以上)。这意味着在 SM75 的 2080Ti 上，某些量化加速路径不可用，需退回较慢实现——这是 SM75 复现的真实难点之一。

- **`--disable-custom-all-reduce`**：关闭 vLLM 自定义 all-reduce 实现，改用 NCCL 标准通信。在部分老架构/驱动组合上更稳。

- **`gpu_memory_utilization`**：vLLM 参数，设定单卡显存使用上限比例（默认 0.9），影响 KV cache 可用空间与 OOM 风险。

- **uv**：Rust 写的高速 Python 包/环境管理器，可创建隔离 venv（`uv venv`）并安装依赖，不污染全局环境。本项目安装的首选工具。

---

## 2026-07-01 第二批（来自 v0.1.3 文章 + 插件源码审计）

- **vLLM general_plugins / 外部插件（out-of-tree plugin）**：vLLM 提供的 `vllm.general_plugins` entry point 机制，第三方包可在 vLLM 启动时被自动 import 并执行注册函数，从而在不修改 vLLM 源码的前提下挂接/替换内部实现。caovan 插件即用此机制，通过 `--additional-config '{"caovan_sm75_turbo3":true}'` 显式开启。

- **monkey-patch（猴子补丁）**：运行时替换类/函数的实现。插件对 Qwen GatedDeltaNet 的 `_forward_core` 做替换，仅在满足 SM75+MTP=2+开关 时走自研 kernel，否则调用原实现。进程内生效，不落盘。

- **speculative decoding config / `--speculative-config '{"method":"mtp","num_speculative_tokens":2}'`**：vLLM 里配置投机解码的方式。此处用模型自带的 MTP 头作为 draft，一次投机 2 个 token（即 MTP=2 / num_spec=2）。插件正是绑定 `num_spec==2` 才生效。

- **FlashInfer**：一套面向 LLM serving 的高性能 GPU attention kernel 库，vLLM 可选用作 attention backend（日志里 `FLASHINFER`）。vLLM 0.21.0 依赖 `flashinfer-python==0.6.8.post1`。

- **compressed-tensors / Marlin**：`compressed-tensors` 是一种通用量化权重存储格式；Marlin 是高性能 INT4/FP 混合精度 GEMM kernel。AWQ-INT4 模型在 vLLM 运行时可能被识别为 compressed-tensors 并走 Marlin 路径。注意 Marlin 常要求 SM80+，SM75 上是否可用需实测（这是 2080Ti 复现的关键风险点之一）。

- **CUDA Graph / `--compilation-config '{"cudagraph_mode":"PIECEWISE"}'`**：把一段 GPU 操作序列录制成图，之后重放以消除逐 kernel 的 CPU 启动开销。PIECEWISE 表示分段捕获（对动态 shape 更友好）。首次启动的 warmup 慢很大程度来自图捕获。

- **torch.compile / warmup**：PyTorch 2.x 编译；首次启动会编译+捕获+JIT，故"第一次很慢"，之后走缓存变快。

- **abi3 wheel（`cp38-abi3`）**：使用 CPython 稳定 ABI 构建的 wheel，一个包可跨多个 Python 版本（3.8+）使用。vLLM 0.21.0 的 wheel 即 `cp38-abi3`，故 Python 3.11 可直接安装。

- **split-QK vs fused（single-launch）decode path**：GDN+MTP 解码的两种 kernel 组织方式。fused/single-launch 把步骤合成一次 kernel 启动（少一次调度但可能降低投机接受率）；split-QK 保留 Q/K 卷积准备的共享复用 + V/GDN 融合，v0.1.3/AcceptanceLock 路线选择后者以保护 draft acceptance。

- **num_accepted_tokens / draft acceptance**：投机解码中主模型对 draft token 的接受数量/比例。接受率高才有加速；kernel 数值差异过大会拉低接受率，反而变慢。

- **NVFP4**：NVIDIA 的 4-bit 浮点量化格式（如 `nvidia/Qwen3.6-27B-NVFP4`）。需要较新架构支持，2080Ti(SM75) 一般不适用，仅作对比了解。

- **sm_75 kernel 与 CUDA 13 架构支持**：GPU 二进制按 compute capability(如 sm_75) 编译。CUDA 各大版本会逐步弃用老架构。本项目最大不确定性即：torch 2.11.0 / CUDA 13.0 的预编译包是否仍含 sm_75 代码——需在 2080Ti 上跑真实张量运算实测确认。

---

## 2026-07-01 第三批（来自 weicj/vLLM-2080Ti-Definitive fork）

- **硬件定向 fork（hardware-targeted fork）**：针对特定 GPU 架构裁剪/优化的 vLLM 源码分支。weicj 这个 fork 专为 SM75 双 2080Ti，编译时 `TORCH_CUDA_ARCH_LIST=7.5` 只出 sm_75 单架构代码（省编译时间）。

- **Marlin**：高性能混合精度 GEMM kernel（INT4/FP8 权重 × FP16 激活），vLLM 量化推理的关键加速。原生偏好 SM80+，此 fork 做了 SM75 适配。

- **FlashQLA（Flash Qwen Linear Attention）**：Qwen 官方的 GatedDeltaNet/线性注意力 prefill kernel 库（基于 TileLang）。`weicj/FlashQLA-SM70-SM75` 是其 SM70/SM75 适配版，专供 2080Ti 的 Qwen3.6 prefill 热路径。

- **TileLang**：一种面向 GPU kernel 的 tile 级编程 DSL/编译器，FlashQLA 用它实现线性注意力 kernel。

- **CUTLASS**：NVIDIA 的 CUDA 模板库，提供高性能 GEMM/卷积原语。此 fork 编译时用 CUTLASS v4.4.2 生成 w8a8/w4a8 量化 GEMM。

- **TurboQuant / K8V4 KV（`turboquant_k8v4`）**：一种 KV cache 压缩方案，对 Key 用 8-bit、Value 用 4-bit（K8V4）等混合精度存储，进一步压缩长上下文 KV 显存，用于 fast 压缩路线。

- **INT8 KV cache**：把 KV cache 以 8-bit 整数存储，介于 FP16（质量优先）与更激进压缩之间，用于平衡型长上下文服务。

- **YaRN**：一种 RoPE 位置编码的上下文长度外推方法，可把模型有效上下文扩展到训练长度以上（如 256K→512K），此 fork 在 INT8 KV 路线上支持。

- **MTP=3 / MTP_K=3**：一次投机预测 3 个 token（比 caovan 的 MTP=2 更激进）。weicj 的 Qwen27B profile 默认 MTP3。

- **CUDAGraph / no-eager**：把整段解码计算录成 CUDA 图重放以消除 kernel 启动开销；no-eager 即启用图捕获（相对 eager 逐算子执行）。启动 warmup 慢部分来自图捕获。

- **AOT 编译（`FLASHINFER_ENABLE_AOT=1`）**：Ahead-Of-Time 预编译 FlashInfer kernel（相对运行时 JIT），构建期一次性编好，运行期免首次 JIT 卡顿。

- **PCIe P2P（peer-to-peer）**：GPU 之间不经 CPU 直接互访显存的能力。TP=2 通信的底线要求；有 NVLink 时走 NVLink，否则退回 PCIe P2P（需实测拓扑）。

- **safe / normal / fast / aggressive 模式**：weicj launcher 的运行档位，速度递增、稳定性/质量风险递增。normal 为生产推荐，safe 用于排障。

- **`vllm._C` / native 扩展**：vLLM 编译出的 C++/CUDA 原生扩展模块（PagedAttention、cache ops 等）。源码构建成功的标志之一是它能被 import。

---

## 2026-07-02 第四批（weicj 复现排障中出现）

- **ccbin / nvcc host 编译器**：nvcc 编译 `.cu` 时把 host 端 C++ 代码交给 `-ccbin` 指定的编译器（如 gcc/g++）。选错或缺失会导致 kernel 编译失败。

- **cc1plus**：GCC 的 C++ 实际编译后端（`gcc`/`g++` 只是驱动，真正编译 C++ 的是 cc1plus）。系统只装 `gcc-12` 驱动而无 `g++-12` 时，`cc1plus`(12) 缺失 → gcc-12 编 C++ 报 "cannot execute cc1plus"。

- **CC vs CXX（关键区别）**：`CC`=C 编译器，`CXX`=C++ 编译器。本项目里 Triton 用 `CC` 编译它的 `.c` launcher（必须是 C 编译器,设成 g++ 会因 C++ 严格性报错），FlashInfer 用 `CC` 作 nvcc `-ccbin`。因此通用正确值是 `CC=gcc-11`(gcc 驱动)，而非 g++-11。

- **FlashQLA legacy SM70/SM75 GDN prefill kernel**：weicj fork 为 2080Ti 适配的 FlashQLA GDN prefill CUDA 扩展（`gdn_prefill_backend=flashqla_legacy` 启用）。日志出现 "Using FlashQLA legacy SM70/SM75 GDN prefill kernel" 即生效。

- **vm.overcommit_memory / commit headroom**：Linux 内存提交策略。0=启发式(对只读 file-backed mmap 宽松)，1=总是允许，2=严格(按 CommitLimit)。weicj launcher 在 overcommit=0 且 CommitLimit-Committed_AS < 最大权重文件时预检报警要求 sudo 设 1；但启发式模式下大文件 mmap 实测可行，属误报。

- **torch cpp_extension JIT / build.ninja**：torch 运行时把 C++/CUDA 扩展 JIT 编译成 `.so` 并缓存(含 build.ninja 记录编译命令)。改编译器需删旧构建目录让其重生 build.ninja。

- **flashinfer JIT cached_ops**：FlashInfer 按 kernel 形状(如 head_dim_qk_256)在 `~/.cache/flashinfer/<ver>/<sm>/cached_ops` 下 JIT 编译并缓存；未被 AOT 覆盖的形状在首个请求现编译。

- **MTP draft weight sharing**：MTP 投机解码时 draft(nextn)层与主模型共享 embedding/lm_head 权重（日志 "Sharing target model embedding/lm_head weights with the draft model"），省显存。

- **内存看门狗(build watchdog)**：本项目自建的后台脚本，编译期每 5s 查 MemAvailable，低于阈值(8GB)即杀整个编译进程组，防止再次 OOM 拖垮整机。

---

## 2026-07-02 第五批（PP4096/TG128 峰值测速与模型格式发现）

- **PP4096/TG128 测速口径**：prompt（输入）长度 4096 tokens，generation（输出）长度 128 tokens 的合成 benchmark 口径。weicj/2080Ti-LLM-Toolbox 宣称的 101.3 tok/s 即在此口径下测得；长输出端到端 wall-clock 会因 prefill 占比降低而显著拉低平均吞吐。

- **TurboQuant / TQK8V4**：weicj fork 支持的实验性 KV cache 混合精度压缩格式。`k8v4` 表示 Key 8-bit、Value 4-bit；用更小的 KV 占用换取更大上下文容量，同时在短生成口径下通常比 FP16 KV 更快（因显存带宽压力下降）。

- **GPU_UTIL (`--gpu-memory-utilization`)**：vLLM 允许引擎占用的单卡显存比例。提高它能增加 KV cache 容量，但过高会与 CUDA graph memory profiling、运行时开销冲突，反而可能不提升甚至降低吞吐。

- **CUDA graph memory profiling**：vLLM 0.21.0 起默认在初始化阶段为 CUDA graph 预留/估算显存。日志提示 "0.90 GPU memory utilization is equivalent to 0.8974 without profiling"，说明它会吃掉一部分有效显存；可设 `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0` 关闭。

- **cudagraph_mode**：vLLM 编译配置中的 CUDA graph 捕获策略。
  - `PIECEWISE`：按算子子图捕获，内存占用小，MTP/投机解码下更稳。
  - `FULL_AND_PIECEWISE`：同时捕获完整 forward 与子图，kernel launch overhead 更低，是 fast mode 的默认，吞吐更高但内存占用更大。

- **AWQ-INT4 vs GPTQ-INT4**：两种 INT4 权重量化格式。在本项目/weicj fork 中，GPTQ-INT4 配合 Marlin kernel 路径在 SM75 上的 decode 效率显著高于 AWQ-INT4；weicj 仓库 int4 profile 的宣称峰值均基于 GPTQ-INT4。

- **模型格式决定峰值**：同一 profile、同一硬件下，仅因 checkpoint 从 AWQ-INT4 换成 GPTQ-INT4，PP4096/TG128 decode 吞吐可从 ~90 tok/s 提升到 ~100 tok/s（weicj 宣称 100.81 tok/s）。

- **HuggingFace GPTQ-INT4 checkpoint**：适合本项目的候选包括 `llmfan46/Qwen3.6-27B-uncensored-heretic-v2-Native-MTP-Preserved-GPTQ-Int4`（保留原生 MTP 头）、`palmfuture/Qwen3.6-27B-GPTQ-Int4`、`AxisQuant/Qwen3.6-27b-gptq-int4` 等。

---

## 2026-07-02 第六批（一键脚本相关）

- **`run_fast_tqk8v4_bench.sh`**：放置于项目根目录的一键脚本，自动完成停止旧服务、启动 weicj fast/tqk8v4 服务、等待就绪、跑 PP4096/TG128 benchmark、输出结果、停止服务全流程。

- **一键脚本中的等待就绪逻辑**：通过轮询 `http://127.0.0.1:8000/health` 判断服务是否可用，同时监控 launcher 进程是否异常退出，避免空等。

- **benchmark label 自定义**：脚本支持传入后缀参数，例如 `bash run_fast_tqk8v4_bench.sh test1`，生成的结果文件和日志会带对应 label。

---

## 2026-07-02 第七批（质量评估相关）

- **`run_quality_eval.sh`**：项目根目录下的一键质量评估脚本。在 vLLM 服务启动后运行，覆盖中文问答、英文问答、数学推理、代码生成、大海捞针五类任务，输出 JSONL 结果。

- **Needle-in-a-Haystack（大海捞针）**：长上下文评估方法。在长文本中随机/固定位置插入一条关键信息，然后让模型回答相关问题，检验模型能否从长上下文中准确定位并提取关键信息。

- **TurboQuant KV 质量风险**：KV cache 压缩可能带来数值漂移，需在长上下文、重复生成、中文输出等场景下做质量回归。

- **MTP 投机解码质量风险**：MTP 通过 draft model 预测多个 token 再验证，可能降低接受率或导致重复/乱码；需在实际聊天任务中观察。

- **性能 vs 质量的 trade-off**：`fast` mode + TurboQuant 吞吐更高，但稳定性/质量风险高于 `normal` mode；生产部署前建议跑过质量评估。

---

## 2026-07-02 第九批（reasoning/thinking 输出控制）

- **Reasoning parser / `--reasoning-parser`**：vLLM 提供的机制，用于把模型生成的 thinking/reasoning 内容从普通 `content` 中拆分出来，放到响应的 `reasoning_content` 字段。对 Qwen3 等原生带 `<think>...</think>` 标签的模型，默认可能自动启用。设为 `off`/`none`/`disabled` 可关闭自动解析。

- **Chat template kwargs / `enable_thinking`**：通过 `--default-chat-template-kwargs` 传给模型 chat template 的额外参数。对 Qwen3 系列，`{"enable_thinking":false}` 可在模板层面关闭思考输出，使模型直接生成最终答案；`true` 则保留思考过程。

- **Thinking 输出 vs Reasoning parser 的区别**：
  - `enable_thinking` 控制**模型是否生成** thinking token。
  - `REASONING_PARSER` 控制 **vLLM 是否把 thinking token 从 `content` 拆出**到 `reasoning_content`。
  - 若只关 parser 不关 `enable_thinking`，客户端仍会收到 thinking 文本；两者都关才能获得直接答案。

- **Needle-in-a-Haystack 召回验证**：长上下文质量测试。只有当模型输出中确实包含目标 needle 字符串时，`needle_found=True`。关闭 thinking 后，模型按指令只返回代码本身，召回成功。

---

## 2026-07-10 第十批（Open WebUI 部署相关）

- **Open WebUI**：一个开源的 LLM 聊天 Web 界面，支持连接 OpenAI 兼容 API、Ollama、本地模型等。提供多用户、聊天记录、RAG、插件市场等功能。本项目用它作为 vLLM 后端的网页聊天前端。

- **WEBUI_AUTH**：Open WebUI 的环境变量开关。`true` 时启用用户认证（登录/注册）；`false` 时尝试禁用认证，但**仅在没有用户的新安装时有效**。一旦数据库中存在用户，再设 `false` 会导致认证端点被关闭而前端 API 仍要 token，造成无法使用。

- **默认管理员自动创建（`WEBUI_ADMIN_EMAIL`/`WEBUI_ADMIN_PASSWORD`/`WEBUI_ADMIN_NAME`）**：Open WebUI 启动时，若数据库为空且这些环境变量已设置，会自动创建第一个管理员账号，并自动关闭公开注册。适合无人值守部署。

- **JWT（JSON Web Token）/ Bearer 认证**：Open WebUI 登录成功后返回一段签名的 token，后续前端请求在 HTTP header 中携带 `Authorization: Bearer <token>` 以证明身份。token 有过期时间，过期后需重新登录。

- **Embedding 模型（`sentence-transformers/all-MiniLM-L6-v2`）**：Open WebUI 默认下载的轻量级文本嵌入模型，用于 RAG（检索增强生成）、知识库、长记忆等功能。即使不做 RAG，首次启动也会自动下载并加载。本项目已将其缓存到本地，避免重复下载。

- **bcrypt**：一种自适应的密码哈希算法。Open WebUI 默认用它存储用户密码，哈希值以 `$2b$` 开头。bcrypt 对 GPU/ASIC 暴力破解有抵抗性，但只处理密码前 72 字节。

- **OpenAI 兼容端点**：Open WebUI 暴露 `/openai/chat/completions` 等端点，请求/响应格式与 OpenAI API 一致。本项目把后端指向本地 vLLM 的 `/v1` 端点，从而用 Open WebUI 调用 vLLM 模型。

- **Gradio vs Open WebUI**：两者都是 LLM Web 前端。
  - Gradio：更轻量、开箱即用、适合快速 demo，但功能较简单。
  - Open WebUI：功能更丰富（多会话、RAG、Admin 设置、用户管理），但需要初始配置和认证。
  本项目已同时部署两者，用户可按需选择。

- **RAG（Retrieval-Augmented Generation，检索增强生成）**：让 LLM 在回答前先从一个知识库/文档集合中检索相关片段，再把片段作为上下文输入模型。Open WebUI 的 RAG 功能依赖 embedding 模型把文档转成向量。本项目暂未启用 RAG，但 embedding 模型已就绪。

---

## 2026-07-10 第十一批（工具调用相关）

- **Tool calling / Function calling（工具调用/函数调用）**：LLM 的一种能力，模型可以根据用户输入和预定义的工具描述，决定调用哪个工具、传入什么参数。典型场景包括查询天气、调用计算器、操作数据库等。工具调用的输出通常是结构化的 JSON，由客户端解析并执行。

- **tool_choice / `tool_choice: "auto"`**：OpenAI 兼容 API 的参数，控制模型是否/如何选择工具。
  - `auto`：模型自己决定是否需要调用工具。
  - `none`：不调用工具，只生成文本。
  - `required`：必须调用至少一个工具。
  - `{"type": "function", "function": {"name": "xxx"}}`：强制调用指定工具。

- **--enable-auto-tool-choice**：vLLM 的启动参数。开启后，vLLM 才允许请求中使用 `tool_choice: "auto"` 或 `tool_choice: "required"`，并让模型在生成文本和生成工具调用之间自动选择。

- **--tool-call-parser**：vLLM 的启动参数，指定一个工具调用解析器，用于从模型生成的文本中提取工具调用。不同模型家族使用不同的工具调用格式，因此需要匹配对应的解析器。本项目使用 `qwen3_xml`，即 Qwen3 系列推荐的 XML 格式工具调用解析器。

- **VLLM_ENFORCE_STRICT_TOOL_CALLING**：vLLM 环境变量/参数。设为 `1` 时，强制模型输出严格符合工具调用格式，减少格式错误/无效的工具调用，对前端可靠性有帮助。

- **qwen3_xml parser**：针对 Qwen3 系列模型的 XML 格式工具调用解析器。模型生成类似 `<tool>...<name>...</name><parameters>...</parameters>...</tool>` 的 XML 片段，解析器把它转成 OpenAI 格式的 `tool_calls` 数组返回给客户端。

- **Open WebUI 默认工具调用**：Open WebUI 在某些配置下会默认给模型请求附加 `tool_choice: "auto"` 和空/默认 `tools` 列表。如果后端没开 `--enable-auto-tool-choice`，就会直接报错。解决方式要么在后端开启支持，要么在前端关闭工具/Functions 开关。

---

## 2026-07-10 第十二批（Anthropic API 代理相关）

- **Anthropic Messages API / Claude API**：Anthropic 公司提供的 LLM API 格式。端点为 `POST /v1/messages`，请求体包含 `model`、`messages`、`max_tokens`、`system`、`tools`、`tool_choice`、`stream` 等；响应中内容以 `content` 数组形式返回（text block / tool_use block），usage 字段为 `input_tokens`/`output_tokens`。

- **OpenAI API vs Anthropic API 差异**：两者请求/响应格式不同。OpenAI 用 `/v1/chat/completions`，响应 `choices[0].message.content/tool_calls`；Anthropic 用 `/v1/messages`，响应 `content: [{type:"text", text:"..."}, {type:"tool_use", ...}]`。Agent 若只支持其中一种，需要代理层做格式转换。

- **API 代理 / API proxy**：一个中间服务，接收一种 API 格式的请求，内部翻译成另一种格式调用后端，再把响应翻译回原始格式返回。本项目用 FastAPI 写了 anthropic_proxy，把 Anthropic `/v1/messages` 转给 vLLM 的 OpenAI `/v1/chat/completions`。

- **anthropic Python SDK**：Anthropic 官方 Python 客户端，用法类似 `anthropic.Anthropic(base_url=..., api_key=...).messages.create(...)`。把它指向本地代理的 base_url，即可像调用 Claude 一样调用本地 Qwen 模型。

- **SSE（Server-Sent Events）**：一种服务器向客户端单向推送事件的 HTTP 协议。Anthropic 和 OpenAI 的流式 API 都用 SSE，但事件名/数据格式不同（如 Anthropic 用 `event: content_block_delta`，OpenAI 用 `data: {...}`）。代理层需要把 OpenAI 的 SSE 转义成 Anthropic 的 SSE 事件序列。

- **Anthropic `thinking` content block**：Anthropic Messages API 中用于承载模型思考过程的内容块，非流式响应格式为 `{type:"thinking", thinking:"..."}`，流式增量格式为 `{type:"thinking_delta", thinking:"..."}`。本项目代理把 vLLM 的 `reasoning_content` 映射成该格式，让只支持 Anthropic API 的 agent 能看到 Qwen3 的思考过程。

- **WorkspaceManager / CUDA graph workspace 锁定**：vLLM V1 引擎中管理临时 GPU buffer 的组件。启用 CUDA graph 后，workspace 在图捕获完成会被锁定，后续不能再扩容。如果运行时算子（如 TurboQuant 的 `_continuation_prefill`）需要比预分配更大的 workspace，会触发 `Workspace growth is not allowed after locking` 断言崩溃。本项目通过环境变量 `VLLM_TURBOQUANT_CONTINUATION_WORKSPACE_RESERVE_TOKENS` 预分配更大空间来规避。

- **TurboQuant continuation prefill**：TurboQuant KV cache 在长上下文续写时，需要把缓存的 K/V 从压缩格式反量化到 FP16 再做 attention 的路径。该路径一次性反量化所有已缓存 tokens，因此 workspace 需求随 cached_len 线性增长，是长上下文场景下最容易触发 workspace 不足的热点。

---

## 2026-07-13 第十三批（Qwimi 非 GGUF 调研）

- **Qwimi / Qwopus**：基于 `Qwen/Qwen3.6-27B` 的社区衍生模型系列，主打代码、Agent 与推理能力。二者多采用 `Qwen3_5ForConditionalGeneration` 架构，含 `vision_config`，属于**多模态（vision-language）模型**。
- **BF16（bfloat16）**：16-bit 浮点权重格式，动态范围与 FP32 相同但精度较低。27B 模型以 BF16 加载约需 54 GB 显存，超过双 2080Ti 22GB 总显存（44 GB），必须再做量化才能部署。
- **多模态模型 / VLM（Vision-Language Model）**：同时接受文本与图像输入、在语言模型之外还包含视觉编码器（vision encoder）的模型。Qwimi/Qwopus 的 `config.json` 中出现 `vision_config` 即表明该属性；纯文本推理栈接入时可能遇到视觉模块兼容性/显存占用问题。
- **compressed-tensors**：一种通用量化权重存储格式（常与 Neural Compressor / AutoRound 相关），不同于原生 AWQ/GPTQ 布局。加载时可能走 Marlin 等混合精度 kernel 路径，在 SM75 上的兼容性需实测验证。
- **MTP preserved / Native-MTP-Preserved 量化版**：在量化过程中保留模型原生 Multi-Token Prediction（MTP）头的 checkpoint。对比被 strip 掉 MTP 头的量化版，这类模型可在 vLLM 中直接启用 `speculative-config` 进行投机解码，无需额外 draft 模型。

---

---

---

---

---

## 2026-07-13 第十四批（ThinkingCap 切换）

- **MTP head / MTP 头**：Multi-Token Prediction 所需的额外解码层权重。量化/微调时若被剥离，模型将失去一次预测多个 token 的能力，无法与 vLLM 的 `speculative-config method=mtp` 配合加速。
- **投机解码 overhead**：当模型没有合适 draft 权重但投机框架仍被启用时，vLLM 会反复走 fallback 路径，反而增加调度/内存开销，导致吞吐低于普通解码。

---

## 2026-07-14 第十五批（tool_choice: auto 报错修复）

- **`"auto" tool choice requires --enable-auto-tool-choice and --tool-call-parser to be set`**：vLLM 在收到 `tool_choice: "auto"` 请求但启动时未开启 `--enable-auto-tool-choice` 时返回的错误。解决方式是在启动命令中加入 `--enable-auto-tool-choice --tool-call-parser <parser>`（Qwen3 系列用 `qwen3_xml`）。
- **`SERVICE_SCOPE=lan`**：weicj launcher 的环境变量，`local` 时 vLLM 只监听 `127.0.0.1`，`lan` 时监听 `0.0.0.0`，便于局域网或其他机器直接访问 `http://<服务器IP>:8000/v1`。
- **工具调用 smoke test**：启动后发送一个带 `tools` 和 `tool_choice: "auto"` 的 chat completion 请求，验证后端能正确返回 `tool_calls` 而不是报错。本项目使用 "查询北京天气" 作为固定测试用例。

---

## 2026-07-14 第十六批（Tau agent 接入）

- **Tau / tau-ai**：HuggingFace 发布的 Python terminal coding agent，架构分为 `tau_ai`（provider 适配层）、`tau_agent`（agent loop）、`tau_coding`（coding 应用层）三层。
- **provider catalog**：Tau 在 `~/.tau/catalog.toml` 中注册的模型供应商配置，用于把 OpenAI-compatible 端点（如本地 vLLM）接入 Tau 的 agent loop。
- **durable session**：Tau 默认把每次对话以 JSONL 形式持久化到 `~/.tau/sessions/`，可随时恢复上下文。
- **slash command**：Tau TUI 中的 `/` 命令（如 `/clear` 清上下文、`/compact` 压缩历史）。
- **event stream**：Tau agent 与模型之间的统一事件抽象（文本增量、工具调用、工具结果、错误等），provider 层负责把各厂商 API 转换为同一事件流。

---

---

---

---
