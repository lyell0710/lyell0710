## 李玉章 · Yuzhang Li

**M.S. ECE @ UC Davis**　|　GPU 性能优化 · LLM 推理系统

把「算子 → 引擎 → 框架」这条链做了一遍：实现 CUDA 与 Triton 两套算子栈和三架构推理引擎，再深入 vLLM / SGLang 做部署选型与源码级机理实验。

性能结论都重测 3 轮以上，对照物取**同算子的官方实现**，**负结果照登**。

---

### 项目

| 仓库 | 内容 | 关键结果 |
|---|---|---|
| **[Kernel_Optimazation-public](https://github.com/lyell0710/Kernel_Optimazation-public)** | CUDA kernel 版本梯，逐级归因 | Tensor Core GEMM **140.5 TFLOPS = cuBLAS 的 91.6%**；reduce 在 HBM-bound 区间达理论带宽 **94.5%**、与官方 CUB 持平（数据装进 L2 后 CUB 反快 12.1%）；Ada → Hopper 跨代复测，同一改动在 H100 上 **+160.7%** |
| **[triton-kernels](https://github.com/lyell0710/triton-kernels)** | Triton LLM 推理算子库 | FA2 前向达 **SDPA-flash 的 87%**；流水线 GEMM **159.4±1.2 TFLOPS** 与 cuBLAS 持平；FP8 per-block GEMM **1.5×**；CUDA Graph 把小核 launch 塌缩 **11.6×** |
| **[llm-engine](https://github.com/lyell0710/llm-engine)** | dense LLM 推理引擎 | 与 HuggingFace **逐位一致（max_abs_err = 0）**，llama / qwen2 / qwen3 三架构一套 config 驱动；接入融合算子后 decode 吞吐 **+44.1%** |
| **[vllmExperience](https://github.com/lyell0710/vllmExperience)** | vLLM 部署形态选型 + MoE kernel 级分解 | 四臂全矩阵对照给出选型边界；request 级关联测得 **KV 传输等待占 TTFT 54–64%**；nsys 分解定位 fused MoE grouped GEMM 占 serving GPU 时间 **56%** |
| **[LLM_Quantization](https://github.com/lyell0710/LLM_Quantization)** | GPTQ / AWQ / SmoothQuant 实现 | 同模型、同评测协议、同量化网格的控制变量对照：GPTQ 收回 RTN 损失 **61.6%**、AWQ **31.9%**、叠加只再 **+1.0pp**——一阶启发式与二阶补偿的差距被量出来了 |
| **[sglang-prefix-lab](https://github.com/lyell0710/sglang-prefix-lab)** | SGLang RadixAttention 机理与收益边界 | 共享前缀下 TTFT p50 **−77%**（Qwen3-8B，含 disable-radix 反例臂与逐 token 计数器闭环）；LRU 命中不是衰减而是**重用距离越线即崩塌** |

### 方法

- **同算子、同区间才可比** —— reduce 的对照是 CUB 不是 `cublasSasum`，softmax 的对照是 cuDNN 不是自写参照。
- **双侧各 3 轮，mean±std 落盘** —— 进结论的数字都能从 raw 重算，文本结果文件首行带 provenance。
- **跑前锁定可证伪假设** —— 判定阈值写在实验记录里，跑完不改；被数据推翻的假设原样保留。
- **反例臂** —— 关掉待测机制再测一遍，把相关性钉成因果。

### 边界

结论主要来自 **2×RTX 4090（无 NVLink，P2P 驱动级禁用）** 的消费级平台，算子另在单卡 H100 上做过跨代复测；跨互联的外推没有做，也不声称。

---

📮 lyell0710@gmail.com
