---
title: "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"
date: 2026-02-14
---

# FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness

<p style="color: var(--vp-c-text-2); font-size: 14px;">
📅 2026-02-14 &nbsp;·&nbsp; 🏷️ 推理引擎 &nbsp;·&nbsp; 📖 论文精读
</p>

> **论文信息**
> - **作者**: Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, Christopher Ré
> - **机构**: Stanford University, University at Buffalo
> - **发表**: NeurIPS 2022
> - **链接**: [arXiv:2205.14135](https://arxiv.org/abs/2205.14135)

## 一句话总结

FlashAttention 提出了一种 **IO 感知（IO-Aware）** 的精确注意力算法，通过 **分块计算（Tiling）** 和 **核函数融合（Kernel Fusion）** 避免在 GPU 高带宽内存（HBM）中实体化巨大的注意力矩阵，将注意力计算的内存复杂度从 \(O(N^2)\) 降至 \(O(N)\)，同时在墙钟时间上比标准注意力快 **2-4 倍**。

---

## Introduction：为什么需要 FlashAttention？

### 1. Transformer 的核心瓶颈：Self-Attention

Transformer 已成为 NLP、CV、语音等领域的基础架构。然而，自注意力机制（Self-Attention）在序列长度 \(N\) 上具有 **\(O(N^2)\) 的时间和空间复杂度**，这成为 Transformer 处理长序列的根本瓶颈。

**标准注意力的计算流程**：

$$
\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d}}\right) V
$$

在标准实现中，这个过程需要：
1. 计算 \(S = QK^\top\)，生成一个 \(N \times N\) 的注意力分数矩阵
2. 对 \(S\) 施加 softmax，得到 \(P = \text{softmax}(S)\)
3. 计算输出 \(O = PV\)

**问题在于**：中间矩阵 \(S\) 和 \(P\) 都是 \(N \times N\) 的，当序列长度 \(N = 8192\) 时，仅这两个矩阵就需要约 **512 MB** 显存（fp32）。这不仅占用大量内存，更关键的是需要频繁在 GPU 的不同存储层级之间搬运数据。

### 2. GPU 内存层次结构：被忽视的瓶颈

论文指出，现有工作几乎都以 **FLOP 数（浮点运算次数）** 作为优化目标，但这忽略了一个关键事实：**现代 GPU 的计算速度远超内存读写速度**。

以 A100 GPU 为例：

| 存储层级 | 容量 | 带宽 |
|---------|------|------|
| **SRAM**（片上缓存，每个 SM） | ~20 MB（总计） | ~19 TB/s |
| **HBM**（高带宽内存） | 40/80 GB | ~2 TB/s |

可以看到，SRAM 的带宽是 HBM 的近 **10 倍**，但容量却小得多。标准注意力实现的问题在于：

```
标准 Attention 的内存访问模式：

  HBM (慢)                    SRAM (快)                  计算单元
┌──────────┐   读取 Q,K    ┌──────────┐   矩阵乘    ┌──────────┐
│  Q, K, V │ ──────────►  │          │ ──────────► │  S=QK^T  │
│          │              │          │             │          │
│  S (N²)  │ ◄────────── │  结果 S  │ ◄────────── │          │
│          │   写回 S     │          │             │          │
│          │   读取 S     │          │   softmax   │          │
│  P (N²)  │ ◄────────── │  结果 P  │ ◄────────── │  P=sm(S) │
│          │   写回 P     │          │             │          │
│          │   读取 P,V   │          │   矩阵乘    │          │
│  O       │ ◄────────── │  结果 O  │ ◄────────── │  O=PV    │
└──────────┘   写回 O     └──────────┘             └──────────┘

问题：S 和 P 各 N² 大小，反复在 HBM ↔ SRAM 之间搬运！
```

**核心洞察**：注意力计算其实是 **内存带宽受限（Memory-bound）** 的操作，而非计算受限。瓶颈不是"算不过来"，而是"数据搬不过来"。

### 3. 现有的近似注意力方法的困境

为了突破 \(O(N^2)\) 的限制，研究社区提出了大量的 **近似注意力（Approximate Attention）** 方法，包括：

- **稀疏注意力（Sparse Attention）**：只计算部分位置对的注意力（如 Longformer、BigBird）
- **低秩近似（Low-rank Approximation）**：用低秩矩阵近似完整注意力矩阵（如 Linformer、Performer）
- **线性注意力（Linear Attention）**：通过核方法将 softmax 近似为可分解形式，实现线性复杂度

然而，论文指出这些方法存在两个共性问题：

1. **精度损失**：近似方法在长序列上经常出现质量退化，尤其是在需要精确建模长距离依赖的任务中
2. **墙钟时间并未真正加速**：虽然 FLOP 数降低了，但由于这些方法往往引入了更多的内存访问开销（如稀疏索引、额外的矩阵变换），在实际 GPU 上跑起来并没有标准注意力快。论文中的实验表明，很多近似方法在序列长度达到 512-2048 之前甚至比标准注意力更慢

::: warning 一个反直觉的事实
减少 FLOP ≠ 减少运行时间。在 GPU 上，如果一个算法 FLOP 更少但内存访问更多，它完全可能比 FLOP 更多但内存访问模式更优的算法更慢。这就是 FlashAttention 的出发点。
:::

### 4. FlashAttention 的核心思路

FlashAttention 不走近似路线，而是从 **IO 复杂度（IO Complexity）** 的角度重新审视标准注意力，通过优化内存访问模式来实现加速，同时保持结果的 **数值精确性**。

核心策略包括两点：

**（1）分块计算（Tiling）**：将 Q、K、V 分成小块，每次只加载一小块到 SRAM 中进行计算，避免实体化完整的 \(N \times N\) 注意力矩阵。

**（2）在线 Softmax（Online Softmax）**：传统 softmax 需要先遍历整行求最大值和求和，再做归一化——这要求整行数据同时在内存中。FlashAttention 采用了 Milakov & Gimelshein (2018) 提出的在线 softmax 技巧，在分块流式处理的过程中 **增量更新** softmax 统计量（running max 和 running sum），无需回头修正。

```
FlashAttention 的内存访问模式：

  HBM (慢)                    SRAM (快)                  计算单元
┌──────────┐   读取 Q块,   ┌──────────┐   一次性     ┌──────────┐
│  Q, K, V │ ──────────►  │ Q块,K块, │ ──────────► │ 分块计算  │
│          │   K块, V块    │  V块     │             │ S块→P块→ │
│          │              │          │   融合计算   │  O块累加  │
│  O       │ ◄────────── │  O块     │ ◄────────── │          │
└──────────┘   只写最终O   └──────────┘             └──────────┘

优势：
  ✅ 中间矩阵 S、P 从不写回 HBM
  ✅ HBM 读写量从 O(N²) 降至 O(N²d²M⁻¹)（M 为 SRAM 大小）
  ✅ 结果与标准注意力完全一致（精确算法）
```

### 5. 论文的主要贡献

论文总结了以下关键贡献：

1. **FlashAttention 算法**：一种 IO 感知的精确注意力实现，通过 Tiling 和在线 Softmax 将 HBM 访问量减少为 \(O(N^2 d^2 M^{-1})\)，其中 \(d\) 是头维度、\(M\) 是 SRAM 大小。论文还证明了在所有精确注意力算法中，这是 **HBM 访问次数的渐近最优下界**

2. **Kernel 融合的扩展**：将 FlashAttention 扩展到支持常用的注意力变体，包括 **带 Mask 的注意力**（如因果掩码）和 **Dropout**，这些操作都在同一个 CUDA Kernel 中完成，避免了额外的内存读写

3. **长序列建模的实际收益**：基于 FlashAttention 的高效实现，论文展示了在多个基准任务上的显著提升：
   - GPT-2 训练速度提升至标准 HuggingFace 实现的 **3 倍**
   - 支持的序列长度从 1K-2K 拓展到 **4K-16K**，使 Transformer 首次在长文档分类（如 MIMIC-III）和长序列生成任务上取得 SOTA 表现
   - Path-X（16K 序列长度的合成任务）上首次达到 **超过随机水平的准确率**

4. **IO 复杂度的理论分析**：论文给出了精确注意力的 HBM 访问下界证明，并分析了常见近似/稀疏注意力的 IO 复杂度，为后续注意力优化研究提供了理论基础

::: tip 为什么叫"Flash"？
Flash 一语双关：既指速度极快（如闪存 Flash Memory），也暗示了算法的核心思想——像闪存一样 **感知和优化 IO 访问模式**，让数据在正确的存储层级被高效处理。
:::

---

::: info 🚧 后续章节持续更新中
FlashAttention 的核心算法细节（Tiling + Online Softmax 的具体实现）、IO 复杂度证明、实验结果分析等内容将在后续更新中补充。
:::

## 推荐阅读

- [FlashAttention 论文](https://arxiv.org/abs/2205.14135)
- [FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning](https://arxiv.org/abs/2307.08691)
- [FlashAttention GitHub](https://github.com/Dao-AILab/flash-attention)
- [Online Softmax (Milakov & Gimelshein, 2018)](https://arxiv.org/abs/1805.02867)
- [vLLM: PagedAttention](./vllm-paper) — FlashAttention 的下游应用之一
