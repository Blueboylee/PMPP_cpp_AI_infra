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
