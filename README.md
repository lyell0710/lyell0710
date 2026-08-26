## 李玉章 · Yuzhang Li

**M.S. ECE @ UC Davis**　|　GPU 性能优化 · LLM 推理系统

把「算子 → 引擎 → 框架」这条链亲手做了一遍：手写 CUDA 与 Triton 两套算子栈，从零实现三架构推理引擎，再深入 vLLM / SGLang 做部署选型与源码级机理实验。

所有性能结论都在同一台机器上重测三轮，对照物取**同算子的官方实现**，**负结果照登**。

---

### 项目

| 仓库 | 内容 | 关键结果 |
|---|---|---|
| **[Kernel_Optimazation](https://github.com/lyell0710/Kernel_Optimazation)** | 手写 CUDA kernel 版本梯，逐级归因 | Tensor Core GEMM **133.1 TFLOPS = 真 cuBLAS 的 85.6%**；reduce 在 HBM-bound 区间达理论带宽 **93.9%**、与官方 CUB 差 **0.7%**（数据装进 L2 后 CUB 反快 33.3%） |
| **[triton-kernels](https://github.com/lyell0710/triton-kernels)** | Triton 手写 LLM 算子库 | FA2 前向达 **SDPA-flash 的 87%**；流水线 GEMM **160.5 TFLOPS** 与 cuBLAS 打平；FP8 per-block GEMM **1.50×**；CUDA Graph 把小核 launch 塌缩 **11.8×** |
| **[llm-engine](https://github.com/lyell0710/llm-engine)** | 从零手写 dense LLM 推理引擎 | 与 HuggingFace **逐位一致（max_abs_err = 0）**，llama / qwen2 / qwen3 三架构一套 config 驱动；接入自研融合算子后 decode 吞吐 **+44.1%** |
| **[vllmExperience](https://github.com/lyell0710/vllmExperience)** | vLLM 部署形态选型 + MoE kernel 级分解 | 四臂全矩阵对照给出选型边界；request 级关联测得 **KV 传输等待占 TTFT 54–64%**；nsys 分解定位 fused MoE grouped GEMM 占 serving GPU 时间 **56%** |
| **[LLM_Quantization](https://github.com/lyell0710/LLM_Quantization)** | GPTQ / AWQ / SmoothQuant 从零实现 | 同模型、同评测协议、同量化网格的控制变量对照：GPTQ 收回 RTN 损失 **61.6%**、AWQ **31.9%**、叠加只再 **+1.0pp**——一阶启发式与二阶补偿的差距被量出来了 |
| **[sglang-prefix-lab](https://github.com/lyell0710/sglang-prefix-lab)** | SGLang RadixAttention 机理与收益边界 | 共享前缀下 TTFT p50 **−77%**（Qwen3-8B，含 disable-radix 反例臂与逐 token 计数器闭环）；LRU 命中不是衰减而是**重用距离越线即崩塌** |

### 方法

- **同算子、同区间才可比** —— 手写 reduce 的对照是 CUB 不是 `cublasSasum`，softmax 的对照是 cuDNN 不是自写参照。
- **双侧各 3 轮，mean±std 落盘** —— 进结论的数字都能从 raw 重算，文本结果文件首行带 provenance。
- **跑前锁定可证伪假设** —— 判定阈值写在实验记录里，跑完不改；被数据推翻的假设原样保留。
- **反例臂** —— 关掉待测机制再测一遍，把相关性钉成因果。

### 边界

结论均限于 **2×RTX 4090（无 NVLink，P2P 驱动级禁用）** 的消费级平台；跨机器、跨互联的外推没有做，也不声称。

---

📮 lyell0710@gmail.com
