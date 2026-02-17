# 🚀 推理引擎与服务化

探索 vLLM、NVIDIA Triton Inference Server、TensorRT 等推理框架与部署方案。

---

<div class="paper-grid">

<a class="paper-card" href="./vllm-paper">
  <div class="paper-icon">📄</div>
  <div class="paper-body">
    <h3>vLLM: Easy, Fast, and Cheap LLM Serving with PagedAttention</h3>
    <p class="paper-meta">Woosuk Kwon et al. · UC Berkeley · 2023</p>
    <p class="paper-desc">提出 PagedAttention 机制，通过虚拟内存分页管理 KV Cache，大幅提升 LLM 推理吞吐量，减少内存浪费。</p>
    <div class="paper-tags">
      <span class="tag">vLLM</span>
      <span class="tag">PagedAttention</span>
      <span class="tag">KV Cache</span>
      <span class="tag">LLM Serving</span>
    </div>
  </div>
</a>

<a class="paper-card" href="./tensorrt-llm">
  <div class="paper-icon">📄</div>
  <div class="paper-body">
    <h3>TensorRT-LLM: A High-Performance Inference Framework for LLMs</h3>
    <p class="paper-meta">NVIDIA · 2024</p>
    <p class="paper-desc">NVIDIA 推出的高性能 LLM 推理框架，支持量化、Kernel 融合、In-flight Batching 等核心优化技术。</p>
    <div class="paper-tags">
      <span class="tag">TensorRT</span>
      <span class="tag">量化</span>
      <span class="tag">Kernel 融合</span>
      <span class="tag">NVIDIA</span>
    </div>
  </div>
</a>

<a class="paper-card" href="./triton-inference-server">
  <div class="paper-icon">📄</div>
  <div class="paper-body">
    <h3>NVIDIA Triton Inference Server: 模型服务化部署实践</h3>
    <p class="paper-meta">NVIDIA · Triton Inference Server</p>
    <p class="paper-desc">学习 Triton Inference Server 的架构设计、模型仓库管理、动态批处理与多模型编排等生产级部署方案。</p>
    <div class="paper-tags">
      <span class="tag">Triton Server</span>
      <span class="tag">Model Serving</span>
      <span class="tag">Dynamic Batching</span>
    </div>
  </div>
</a>

<a class="paper-card" href="./sglang">
  <div class="paper-icon">📄</div>
  <div class="paper-body">
    <h3>SGLang: Efficient Execution of Structured Language Model Programs</h3>
    <p class="paper-meta">Lianmin Zheng et al. · UC Berkeley & Stanford · 2024</p>
    <p class="paper-desc">提出结构化生成语言 SGLang，通过 RadixAttention（KV Cache 自动复用）、压缩有限状态机（高速约束解码）和 API 推测执行三大优化，将复杂 LLM 程序加速最高 6.4 倍。</p>
    <div class="paper-tags">
      <span class="tag">SGLang</span>
      <span class="tag">RadixAttention</span>
      <span class="tag">Constrained Decoding</span>
      <span class="tag">LLM Programming</span>
    </div>
  </div>
</a>

<a class="paper-card" href="./flash-attention">
  <div class="paper-icon">📄</div>
  <div class="paper-body">
    <h3>FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness</h3>
    <p class="paper-meta">Tri Dao et al. · Stanford University · NeurIPS 2022</p>
    <p class="paper-desc">提出 IO 感知的精确注意力算法，通过分块计算（Tiling）和在线 Softmax 避免实体化 N² 注意力矩阵，将内存复杂度从 O(N²) 降至 O(N)，墙钟时间快 2-4 倍。</p>
    <div class="paper-tags">
      <span class="tag">FlashAttention</span>
      <span class="tag">IO-Aware</span>
      <span class="tag">Tiling</span>
      <span class="tag">Kernel Fusion</span>
    </div>
  </div>
</a>

<a class="paper-card" href="./clipper">
  <div class="paper-icon">📄</div>
  <div class="paper-body">
    <h3>Clipper: A Low-Latency Online Prediction Serving System</h3>
    <p class="paper-meta">Daniel Crankshaw et al. · UC Berkeley · NSDI 2017</p>
    <p class="paper-desc">最早系统性地将 ML 模型推向在线推理服务的通用 Serving 系统之一，通过模型抽象层（容器化 + 自适应批处理）和模型选择层（Bandit 算法 + 集成学习）解决框架碎片化与在线模型选优问题。</p>
    <div class="paper-tags">
      <span class="tag">Model Serving</span>
      <span class="tag">Adaptive Batching</span>
      <span class="tag">Bandit Algorithm</span>
      <span class="tag">Ensemble</span>
    </div>
  </div>
</a>

<a class="paper-card" href="./smoothquant">
  <div class="paper-icon">📄</div>
  <div class="paper-body">
    <h3>SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models</h3>
    <p class="paper-meta">Guangxuan Xiao et al. · MIT & NVIDIA · ICML 2023</p>
    <p class="paper-desc">通过数学等价的平滑变换将激活值中的离群值迁移到权重上，实现 W8A8 全量化 INT8 推理，在 OPT-175B 上精度损失不到 1%，推理加速 1.56 倍，显存节省近 2 倍。</p>
    <div class="paper-tags">
      <span class="tag">Quantization</span>
      <span class="tag">INT8</span>
      <span class="tag">Post-Training</span>
      <span class="tag">Outlier Migration</span>
    </div>
  </div>
</a>

<a class="paper-card" href="./speculative-decoding">
  <div class="paper-icon">📄</div>
  <div class="paper-body">
    <h3>Fast Inference from Transformers via Speculative Decoding</h3>
    <p class="paper-meta">Yaniv Leviathan et al. · Google Research · ICML 2023</p>
    <p class="paper-desc">用小模型猜测、大模型并行验证的方式无损加速自回归解码，通过精心设计的拒绝采样保证输出分布与原始大模型完全一致，实现 2-3 倍推理加速。</p>
    <div class="paper-tags">
      <span class="tag">Speculative Decoding</span>
      <span class="tag">Rejection Sampling</span>
      <span class="tag">Lossless Acceleration</span>
      <span class="tag">Autoregressive</span>
    </div>
  </div>
</a>

<a class="paper-card" href="./flash-decoding">
  <div class="paper-icon">📝</div>
  <div class="paper-body">
    <h3>Flash-Decoding for Long-Context Inference</h3>
    <p class="paper-meta">Tri Dao et al. · Stanford CRFM · 2023 Blog</p>
    <p class="paper-desc">在 FlashAttention 基础上增加 KV 序列长度维度的并行拆分，通过 log-sum-exp 归约合并各分块结果，让解码阶段注意力计算充分利用 GPU，长序列端到端加速 8 倍。</p>
    <div class="paper-tags">
      <span class="tag">Flash-Decoding</span>
      <span class="tag">KV Split</span>
      <span class="tag">GPU Utilization</span>
      <span class="tag">Long Context</span>
    </div>
  </div>
</a>

<a class="paper-card" href="./flash-decoding-pp">
  <div class="paper-icon">📄</div>
  <div class="paper-body">
    <h3>FlashDecoding++: Faster Large Language Model Inference on GPUs</h3>
    <p class="paper-meta">Ke Hong et al. · 清华大学 & 上交 · arXiv 2023</p>
    <p class="paper-desc">通过统一最大值实现异步 Softmax 消除 ~20% 同步开销、pad8+双缓冲优化扁平 GEMM、启发式数据流自适应硬件，在 Flash-Decoding 基础上端到端再加速 1.37 倍。</p>
    <div class="paper-tags">
      <span class="tag">Async Softmax</span>
      <span class="tag">Flat GEMM</span>
      <span class="tag">Double Buffering</span>
      <span class="tag">Cross-Hardware</span>
    </div>
  </div>
</a>

</div>

::: tip 💡 持续更新中
更多推理引擎与服务化相关的论文解读和学习笔记将陆续更新，敬请关注！
:::

<style>
.paper-grid {
  display: grid;
  grid-template-columns: 1fr;
  gap: 16px;
  margin: 24px 0;
}

@media (min-width: 768px) {
  .paper-grid {
    grid-template-columns: repeat(2, 1fr);
  }
}

.paper-card {
  display: flex;
  gap: 16px;
  padding: 20px;
  border: 1px solid var(--vp-c-divider);
  border-radius: 12px;
  text-decoration: none !important;
  color: inherit !important;
  transition: all 0.3s ease;
  background: var(--vp-c-bg-soft);
}

.paper-card:hover {
  border-color: var(--vp-c-brand-1);
  box-shadow: 0 4px 16px rgba(0, 0, 0, 0.08);
  transform: translateY(-2px);
}

.paper-icon {
  font-size: 28px;
  flex-shrink: 0;
  margin-top: 2px;
}

.paper-body h3 {
  margin: 0 0 6px 0;
  font-size: 16px;
  font-weight: 600;
  line-height: 1.4;
  color: var(--vp-c-text-1);
}

.paper-meta {
  margin: 0 0 8px 0;
  font-size: 13px;
  color: var(--vp-c-text-3);
}

.paper-desc {
  margin: 0 0 12px 0;
  font-size: 14px;
  color: var(--vp-c-text-2);
  line-height: 1.6;
}

.paper-tags {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
}

.tag {
  padding: 2px 10px;
  font-size: 12px;
  border-radius: 999px;
  background: var(--vp-c-brand-soft);
  color: var(--vp-c-brand-1);
  font-weight: 500;
}
</style>
