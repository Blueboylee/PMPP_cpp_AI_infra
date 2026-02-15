# 🏋️ 分布式训练

探索 DeepSpeed ZeRO、Megatron-LM、FSDP 等分布式训练框架与大规模模型训练技术。

---

<div class="paper-grid">

<a class="paper-card" href="./deepspeed-zero">
  <div class="paper-icon">📄</div>
  <div class="paper-body">
    <h3>ZeRO: Memory Optimizations Toward Training Trillion Parameter Models</h3>
    <p class="paper-meta">Samyam Rajbhandari et al. · Microsoft · SC 2020</p>
    <p class="paper-desc">提出 ZeRO（Zero Redundancy Optimizer），通过三阶段递进地分片优化器状态、梯度和参数，消除数据并行中的内存冗余，使可训练模型参数量推向万亿级别。</p>
    <div class="paper-tags">
      <span class="tag">DeepSpeed</span>
      <span class="tag">ZeRO</span>
      <span class="tag">Data Parallelism</span>
      <span class="tag">Memory Optimization</span>
    </div>
  </div>
</a>

</div>

::: tip 💡 持续更新中
更多分布式训练相关的论文解读和学习笔记将陆续更新，敬请关注！
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
