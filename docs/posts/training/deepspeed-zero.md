---
title: "ZeRO: Memory Optimizations Toward Training Trillion Parameter Models"
date: 2026-02-15
---

# ZeRO: Memory Optimizations Toward Training Trillion Parameter Models

<p style="color: var(--vp-c-text-2); font-size: 14px;">
2026-02-15 &nbsp;·&nbsp; 分布式训练 &nbsp;·&nbsp; 论文精读
</p>

> **论文信息**
> - **作者**: Samyam Rajbhandari, Jeff Rasley, Olatunji Ruwase, Yuxiong He
> - **机构**: Microsoft
> - **发表**: SC 2020 (The International Conference for High Performance Computing)
> - **链接**: [arXiv:1910.02054](https://arxiv.org/abs/1910.02054)

## 一句话总结

ZeRO（Zero Redundancy Optimizer）提出了一套 **消除数据并行中冗余状态** 的内存优化方法，通过将模型状态（优化器状态、梯度、参数）**分片（Partition）** 到各个数据并行进程中，使得单卡内存占用随并行度 **线性下降**，在不牺牲数据并行通信效率的前提下，将可训练模型的参数量推向 **万亿级别**。

---

## Introduction：为什么需要 ZeRO？

### 1. 大模型训练的内存墙

2020 年前后，语言模型参数量进入爆发式增长期：GPT-2（1.5B）→ Megatron-LM（8.3B）→ T5（11B）。但训练更大模型面临一个根本性的障碍——**GPU 显存不够用**。

以一个 1.5B 参数的 GPT-2 模型为例，使用 Adam 优化器和混合精度训练：

| 组件 | 精度 | 单参数字节 | 1.5B 参数总量 |
|------|------|-----------|--------------|
| 参数（fp16） | fp16 | 2 B | 3 GB |
| 梯度（fp16） | fp16 | 2 B | 3 GB |
| Adam 动量（fp32） | fp32 | 4 B | 6 GB |
| Adam 方差（fp32） | fp32 | 4 B | 6 GB |
| 参数主副本（fp32） | fp32 | 4 B | 6 GB |
| **合计** | — | **16 B** | **24 GB** |

一个 1.5B 的模型，仅"模型状态"就需要 **24 GB**，几乎吃满一块 V100（32 GB）。更大的模型——比如 100B 参数——需要约 **1.6 TB** 的显存，远超单卡容量。

### 2. 现有并行策略的困境

面对内存墙，主流的两种并行策略各有缺陷：

**数据并行（Data Parallelism, DP）**：
- 每张卡存一份 **完整的** 模型副本，只切分数据
- 通信高效：只需 AllReduce 梯度
- **致命缺陷**：内存没有节省！每张卡都存完整的参数 + 梯度 + 优化器状态

**模型并行（Model Parallelism, MP）**：
- 将模型的不同层（或层内的张量）切分到不同卡上
- 内存确实减少了
- **致命缺陷**：通信开销巨大。以 Megatron-LM 为例，每个 Transformer 层需要 2 次 AllReduce（前向 + 反向各 1 次），在跨节点场景下通信成本极高
- 扩展性差：MP 度超过单节点 GPU 数量后效率急剧下降

```
数据并行 vs 模型并行：

数据并行 (DP)                          模型并行 (MP)
┌─────────────┐ ┌─────────────┐       ┌──────┬──────┬──────┬──────┐
│  完整模型    │ │  完整模型    │       │ 层1  │ 层2  │ 层3  │ 层4  │
│  完整优化器  │ │  完整优化器  │       │GPU0  │GPU1  │GPU2  │GPU3  │
│  数据分片1   │ │  数据分片2   │       │      │      │      │      │
│    GPU 0     │ │    GPU 1     │       └──┬───┴──┬───┴──┬───┴──┬───┘
└──────────────┘ └──────────────┘          │      │      │      │
     ↕ AllReduce 梯度                      ↕ 前向/反向激活通信 ↕
                                      （每层都需要通信！）

DP 优点: 通信高效, 扩展性好            MP 优点: 单卡内存减少
DP 缺点: 每卡内存完整冗余              MP 缺点: 通信密集, 跨节点差
```

论文的核心洞察：**数据并行的通信效率很高，但内存太冗余；模型并行的内存效率不错，但通信太昂贵。能否两全其美？**

### 3. ZeRO 的核心思想

ZeRO 的答案是：**在保持数据并行通信效率的同时，消除数据并行的内存冗余**。

具体做法很直觉——既然数据并行中每张卡都存了一份完整的模型状态，那我们 **让每张卡只存 \(1/N_d\) 份**（\(N_d\) 是数据并行度），需要完整数据时通过集合通信临时聚合即可。

这个思想分三个递进的阶段实现：

| 阶段 | 分片内容 | 内存节省 | 通信开销 |
|------|---------|---------|---------|
| **ZeRO-1** (\(P_{os}\)) | 优化器状态 | **4x** | 不变 |
| **ZeRO-2** (\(P_{os+g}\)) | 优化器状态 + 梯度 | **8x** | 不变 |
| **ZeRO-3** (\(P_{os+g+p}\)) | 优化器状态 + 梯度 + 参数 | **\(N_d\) x** | ~1.5x |

::: tip 命名含义
ZeRO = Zero Redundancy Optimizer。名字本身就揭示了核心——让 **冗余为零**。不是减少冗余，而是 **消除** 冗余。
:::

### 4. 论文的主要贡献

1. **ZeRO-DP**：一套三阶段递进的优化器状态、梯度和参数分片方案，使数据并行从"内存冗余"变为"内存线性扩展"

2. **ZeRO-R**：针对残余内存（激活值、临时缓冲区、内存碎片）的优化方案，包含分片激活检查点、恒定大小缓冲区、内存碎片整理

3. **理论分析**：严格推导了每个阶段的通信量，证明 ZeRO-1 和 ZeRO-2 的通信量与标准数据并行相同，ZeRO-3 仅增加 50%

4. **实验验证**：在最多 400 块 V100 上训练了最高 100B 参数的模型，达到超过 15 PetaFLOPs 的吞吐量，超过 Megatron-LM 同等配置的 10 倍

---

## 内存消耗分析：训练到底吃了多少显存？

在理解 ZeRO 的优化方法之前，我们必须精确分析 **训练一个模型到底需要多少显存**。论文将显存消耗分为两大类：**模型状态（Model States）** 和 **残余状态（Residual States）**。

### 模型状态：内存大户

模型状态包括三部分：**优化器状态（Optimizer States）**、**梯度（Gradients）** 和 **参数（Parameters）**。

以当前主流的 **混合精度训练 + Adam 优化器** 为例，让我们精确计算每个参数的内存开销。

#### 混合精度训练的内存开销

混合精度训练的标准流程是：
1. 用 fp16 参数做前向和反向传播（减少计算时间和激活值内存）
2. 在 fp32 下保存参数主副本和优化器状态（保证数值稳定性）
3. 梯度以 fp16 计算，但更新时转为 fp32

对于 Adam 优化器，每个参数需要保存：

$$
\text{总内存/参数} = \underbrace{2}_{\text{fp16 参数}} + \underbrace{2}_{\text{fp16 梯度}} + \underbrace{4 + 4 + 4}_{\text{fp32 参数副本 + 动量 + 方差}} = 16 \text{ 字节}
$$

论文将其归纳为一个通用公式。设 \(\Psi\) 为参数量，\(K\) 为优化器状态的乘数因子：

$$
\text{模型状态内存} = 2\Psi + 2\Psi + K\Psi \text{ 字节}
$$

对于 Adam，\(K = 12\)（4 字节 fp32 参数副本 + 4 字节动量 + 4 字节方差），因此总计 \(16\Psi\) 字节。

::: warning 优化器状态占主导
在 \(16\Psi\) 总内存中，**优化器状态占了 \(12\Psi\)（75%）**，参数和梯度各占 \(2\Psi\)（12.5%）。这是 ZeRO 优先分片优化器状态的原因——它是内存的大头。
:::

#### 用数字说话

| 模型 | 参数量 \(\Psi\) | 模型状态内存 | V100 32GB 需要几块？ |
|------|--------|------------|---------------------|
| GPT-2 | 1.5B | 24 GB | 1 块（刚好） |
| Megatron-LM | 8.3B | 133 GB | 5 块 |
| T5-11B | 11B | 176 GB | 6 块 |
| GPT-3 175B | 175B | 2.8 TB | 88 块 |
| 万亿参数 | 1T | 16 TB | 500 块 |

### 残余状态：不可忽视的内存开销

除了模型状态，训练过程中还有三类内存开销：

**1. 激活值（Activations）**

前向传播产生的中间结果，需要保留到反向传播时使用。对于 Transformer 模型，每层激活值的大小大约是：

$$
\text{每层激活值} \approx 12 \cdot b \cdot s \cdot h
$$

其中 \(b\) 是 batch size，\(s\) 是序列长度，\(h\) 是隐藏维度。

以一个 100B 参数的类 GPT 模型为例（\(h = 19456, s = 2048, b = 32, L = 128\) 层）：

$$
\text{激活值总量} \approx 12 \times 32 \times 2048 \times 19456 \times 128 \approx 60 \text{ TB}
$$

即使使用激活检查点（Activation Checkpointing），仍需约 **33 GB**。

**2. 临时缓冲区（Temporary Buffers）**

梯度 AllReduce、梯度范数计算等操作需要临时缓冲区。对于大模型，这些缓冲区可达 **数 GB**。

**3. 内存碎片（Memory Fragmentation）**

训练过程中频繁的内存申请和释放会造成碎片化，导致即使总空闲内存足够，也无法分配连续的大块内存，引发 OOM。

### 内存消耗全景图

```
训练一个 Ψ 参数模型的内存布局 (混合精度 + Adam):

┌─────────────────────────────────────────────────┐
│                  GPU 显存                         │
├─────────────────────────────────────────────────┤
│ ┌─────────────────────────────────────────────┐ │
│ │          模型状态 (Model States)              │ │
│ │ ┌───────────┬───────────┬─────────────────┐ │ │
│ │ │ 参数 (fp16)│ 梯度 (fp16)│  优化器状态     │ │ │
│ │ │   2Ψ B     │   2Ψ B     │  (fp32)        │ │ │
│ │ │  12.5%     │  12.5%     │   12Ψ B        │ │ │
│ │ │            │            │   75%          │ │ │
│ │ └───────────┴───────────┴─────────────────┘ │ │
│ │              合计: 16Ψ 字节                   │ │
│ └─────────────────────────────────────────────┘ │
│                                                 │
│ ┌─────────────────────────────────────────────┐ │
│ │         残余状态 (Residual States)            │ │
│ │ ┌─────────────┬──────────┬────────────────┐ │ │
│ │ │ 激活值       │ 临时缓冲  │ 内存碎片        │ │ │
│ │ │ (可检查点)   │ (AllReduce│ (碎片化)       │ │ │
│ │ │              │  梯度范数) │                │ │ │
│ │ └─────────────┴──────────┴────────────────┘ │ │
│ └─────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────┘

对于 1.5B 模型:  模型状态 24 GB + 激活值 ~几 GB = 塞满 V100 32GB
对于 100B 模型:  模型状态 1.6 TB ← 这才是真正的瓶颈!
```

---

## ZeRO-DP：数据并行中的内存优化

ZeRO-DP（ZeRO-powered Data Parallelism）是论文的核心贡献。它通过三个递进阶段，逐步消除数据并行中的冗余内存。

### 基线：标准数据并行

在标准数据并行中，\(N_d\) 张 GPU **每张都保存完整的模型状态**：

```
标准数据并行 (N_d = 4 张 GPU):

GPU 0           GPU 1           GPU 2           GPU 3
┌───────────┐  ┌───────────┐  ┌───────────┐  ┌───────────┐
│ 完整参数   │  │ 完整参数   │  │ 完整参数   │  │ 完整参数   │
│ 完整梯度   │  │ 完整梯度   │  │ 完整梯度   │  │ 完整梯度   │
│ 完整优化器 │  │ 完整优化器 │  │ 完整优化器 │  │ 完整优化器 │
│ 状态      │  │ 状态      │  │ 状态      │  │ 状态      │
├───────────┤  ├───────────┤  ├───────────┤  ├───────────┤
│ 数据分片 0 │  │ 数据分片 1 │  │ 数据分片 2 │  │ 数据分片 3 │
└───────────┘  └───────────┘  └───────────┘  └───────────┘

每张卡的模型状态内存: 16Ψ 字节 (完全冗余!)
通信量: 每步 2Ψ 元素 (AllReduce 梯度)
```

所有 4 张卡存的模型状态 **完全相同**——这就是"冗余"。

### ZeRO Stage 1（\(P_{os}\)）：分片优化器状态

**核心思想**：优化器状态占 \(12\Psi\) 字节（75% 的模型状态内存），但每个参数的优化器状态只在 **参数更新时** 需要。既然每张卡最终得到的更新后参数是一样的，为什么不让每张卡只负责更新 \(1/N_d\) 的参数？

**具体做法**：

1. **前向 + 反向传播**：照常进行，每张卡都持有完整参数和完整梯度
2. **梯度聚合**：执行 **Reduce-Scatter**（而非 AllReduce），使得每张卡只得到自己负责那 \(1/N_d\) 参数的聚合梯度
3. **参数更新**：每张卡只用自己的 \(1/N_d\) 优化器状态更新自己负责的 \(1/N_d\) 参数
4. **参数同步**：执行 **AllGather**，让所有卡都获得完整的更新后参数

```
ZeRO Stage 1: 分片优化器状态 (N_d = 4)

GPU 0           GPU 1           GPU 2           GPU 3
┌───────────┐  ┌───────────┐  ┌───────────┐  ┌───────────┐
│ 完整参数   │  │ 完整参数   │  │ 完整参数   │  │ 完整参数   │
│ 完整梯度   │  │ 完整梯度   │  │ 完整梯度   │  │ 完整梯度   │
│           │  │           │  │           │  │           │
│ 优化器 1/4 │  │ 优化器 2/4 │  │ 优化器 3/4 │  │ 优化器 4/4 │
│ ████░░░░  │  │ ░░░░████░ │  │ ░░░░░████ │  │ ░░░░░░███ │
└───────────┘  └───────────┘  └───────────┘  └───────────┘

每张卡的内存: 2Ψ + 2Ψ + 12Ψ/N_d = 4Ψ + 12Ψ/N_d 字节
当 N_d = 4:    4Ψ + 3Ψ = 7Ψ 字节  (vs 标准 DP 的 16Ψ)
当 N_d = 64:   4Ψ + 0.19Ψ ≈ 4.2Ψ 字节  → 约 4x 节省
```

**内存节省**：随着 \(N_d\) 增大，优化器状态的内存 \(\frac{12\Psi}{N_d}\) 趋近于 0，总内存趋近 \(4\Psi\)（参数 + 梯度），约为标准 DP 的 **1/4**。

### ZeRO Stage 2（\(P_{os+g}\)）：分片优化器状态 + 梯度

**核心思想**：在 Stage 1 中，每张卡仍然保存完整梯度。但 Stage 1 的 Reduce-Scatter 已经说明了——每张卡最终只需要自己负责的 \(1/N_d\) 参数的梯度。既然如此，每张卡只保留自己需要的那 \(1/N_d\) 梯度不就行了？

**具体做法**：

1. **反向传播时逐步聚合**：当一个参数的梯度计算完毕后，立即执行 **Reduce-Scatter**
2. **只保留分片梯度**：Reduce-Scatter 后，每张卡只保留自己负责的 \(1/N_d\) 梯度，其余部分 **立即释放**
3. **参数更新**：每张卡用分片的优化器状态和分片的梯度更新 \(1/N_d\) 参数
4. **参数同步**：AllGather 完整参数

```
ZeRO Stage 2: 分片优化器状态 + 梯度 (N_d = 4)

GPU 0           GPU 1           GPU 2           GPU 3
┌───────────┐  ┌───────────┐  ┌───────────┐  ┌───────────┐
│ 完整参数   │  │ 完整参数   │  │ 完整参数   │  │ 完整参数   │
│           │  │           │  │           │  │           │
│ 梯度 1/4  │  │ 梯度 2/4  │  │ 梯度 3/4  │  │ 梯度 4/4  │
│ ████░░░░  │  │ ░░░░████░ │  │ ░░░░░████ │  │ ░░░░░░███ │
│           │  │           │  │           │  │           │
│ 优化器 1/4 │  │ 优化器 2/4 │  │ 优化器 3/4 │  │ 优化器 4/4 │
│ ████░░░░  │  │ ░░░░████░ │  │ ░░░░░████ │  │ ░░░░░░███ │
└───────────┘  └───────────┘  └───────────┘  └───────────┘

每张卡的内存: 2Ψ + 2Ψ/N_d + 12Ψ/N_d = 2Ψ + 14Ψ/N_d 字节
当 N_d = 4:    2Ψ + 3.5Ψ = 5.5Ψ 字节
当 N_d = 64:   2Ψ + 0.22Ψ ≈ 2.2Ψ 字节  → 约 8x 节省
```

**内存节省**：随着 \(N_d\) 增大，总内存趋近 \(2\Psi\)（仅参数），约为标准 DP 的 **1/8**。

::: tip 梯度的生命周期优化
Stage 2 的一个精妙之处在于 **梯度的及时释放**。在标准 DP 中，所有梯度必须等到反向传播完成后才能统一 AllReduce。在 Stage 2 中，每个参数的梯度一旦 Reduce-Scatter 完成（只保留分片），其余部分立刻释放。这意味着在反向传播的任意时刻，显存中 **最多只有一小部分梯度是完整的**，大部分已经被释放了。
:::

### ZeRO Stage 3（\(P_{os+g+p}\)）：分片一切

**核心思想**：在 Stage 2 中，参数仍然是冗余的——每张卡存了完整的 \(2\Psi\) 字节参数。但参数在前向和反向传播中只是被 **读取**（不会被原地修改），因此可以像梯度一样分片存储，需要时再 AllGather。

**具体做法**：

1. **每张卡只持久存储 \(1/N_d\) 的参数**
2. **前向传播**：逐层执行 AllGather 收集当前层的完整参数，用完后立即释放非本卡分片
3. **反向传播**：同样逐层 AllGather 参数 → 计算梯度 → 释放参数 → Reduce-Scatter 梯度
4. **参数更新**：每张卡只更新自己的 \(1/N_d\) 分片

```
ZeRO Stage 3: 分片一切 (N_d = 4)

GPU 0           GPU 1           GPU 2           GPU 3
┌───────────┐  ┌───────────┐  ┌───────────┐  ┌───────────┐
│ 参数 1/4  │  │ 参数 2/4  │  │ 参数 3/4  │  │ 参数 4/4  │
│ ████░░░░  │  │ ░░░░████░ │  │ ░░░░░████ │  │ ░░░░░░███ │
│           │  │           │  │           │  │           │
│ 梯度 1/4  │  │ 梯度 2/4  │  │ 梯度 3/4  │  │ 梯度 4/4  │
│ ████░░░░  │  │ ░░░░████░ │  │ ░░░░░████ │  │ ░░░░░░███ │
│           │  │           │  │           │  │           │
│ 优化器 1/4 │  │ 优化器 2/4 │  │ 优化器 3/4 │  │ 优化器 4/4 │
│ ████░░░░  │  │ ░░░░████░ │  │ ░░░░░████ │  │ ░░░░░░███ │
└───────────┘  └───────────┘  └───────────┘  └───────────┘

每张卡的内存: 16Ψ/N_d 字节  ← 线性扩展!
当 N_d = 4:    4Ψ 字节     → 4x 节省
当 N_d = 64:   0.25Ψ 字节  → 64x 节省!
```

**内存节省**：**线性于 \(N_d\)**！64 张卡就能把内存降到标准 DP 的 1/64。理论上，只要 GPU 数量足够，任意大的模型都可以训练。

### Stage 3 的前向传播详细流程

Stage 3 的前向传播是理解整个 ZeRO-3 的关键。让我们逐步追踪：

```
ZeRO Stage 3 前向传播 (逐层处理):

时间步 1: 计算第 1 层
  ┌──────────────────────────────────────────┐
  │ 所有 GPU 执行 AllGather 收集第 1 层完整参数  │
  │                                          │
  │ GPU 0: [██░░] ──┐                        │
  │ GPU 1: [░░██] ──┤── AllGather ──→ [████]  │
  │ GPU 2: [██░░] ──┤     所有 GPU 都得到     │
  │ GPU 3: [░░██] ──┘     完整的第 1 层参数     │
  │                                          │
  │ → 使用完整参数执行第 1 层前向传播           │
  │ → 丢弃非本卡分片 (只保留 1/N_d)           │
  └──────────────────────────────────────────┘
          ↓
时间步 2: 计算第 2 层
  ┌──────────────────────────────────────────┐
  │ AllGather 第 2 层参数 → 前向传播 → 释放     │
  └──────────────────────────────────────────┘
          ↓
        ...
          ↓
时间步 L: 计算第 L 层
  ┌──────────────────────────────────────────┐
  │ AllGather 第 L 层参数 → 前向传播 → 释放     │
  └──────────────────────────────────────────┘

关键: 任意时刻最多只有 1-2 层的完整参数在内存中!
```

### 三阶段对比总结

| 特性 | 标准 DP | ZeRO-1 | ZeRO-2 | ZeRO-3 |
|------|---------|--------|--------|--------|
| **分片优化器** | ✗ | ✓ | ✓ | ✓ |
| **分片梯度** | ✗ | ✗ | ✓ | ✓ |
| **分片参数** | ✗ | ✗ | ✗ | ✓ |
| **每卡内存** | \(16\Psi\) | \(4\Psi + \frac{12\Psi}{N_d}\) | \(2\Psi + \frac{14\Psi}{N_d}\) | \(\frac{16\Psi}{N_d}\) |
| **极限节省** (\(N_d \to \infty\)) | 1x | 4x | 8x | \(N_d\) x |
| **通信量** | \(2\Psi\) | \(2\Psi\) | \(2\Psi\) | \(3\Psi\) |

下面的代码可以直观计算和对比不同阶段的内存消耗：

```cpp-run title="ZeRO 三阶段内存消耗计算器"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <algorithm>
#include <string>

// ZeRO 内存计算器: 精确模拟三个阶段的内存消耗

struct MemoryProfile {
    double params_gb;      // 参数内存 (GB)
    double grads_gb;       // 梯度内存 (GB)
    double optim_gb;       // 优化器状态内存 (GB)
    double total_gb;       // 总模型状态内存 (GB)
    double comm_elements;  // 通信量 (以 Ψ 为单位)
};

MemoryProfile calc_standard_dp(double psi_B, int Nd) {
    // 标准数据并行: 每卡都存完整的模型状态
    double params = 2.0 * psi_B;   // fp16 参数
    double grads  = 2.0 * psi_B;   // fp16 梯度
    double optim  = 12.0 * psi_B;  // fp32 参数副本 + 动量 + 方差
    return {params, grads, optim, params + grads + optim, 2.0};
}

MemoryProfile calc_zero_stage1(double psi_B, int Nd) {
    // ZeRO-1: 优化器状态分片
    double params = 2.0 * psi_B;
    double grads  = 2.0 * psi_B;
    double optim  = 12.0 * psi_B / Nd;  // 分片!
    return {params, grads, optim, params + grads + optim, 2.0};
}

MemoryProfile calc_zero_stage2(double psi_B, int Nd) {
    // ZeRO-2: 优化器状态 + 梯度分片
    double params = 2.0 * psi_B;
    double grads  = 2.0 * psi_B / Nd;   // 分片!
    double optim  = 12.0 * psi_B / Nd;  // 分片!
    return {params, grads, optim, params + grads + optim, 2.0};
}

MemoryProfile calc_zero_stage3(double psi_B, int Nd) {
    // ZeRO-3: 一切分片
    double params = 2.0 * psi_B / Nd;   // 分片!
    double grads  = 2.0 * psi_B / Nd;   // 分片!
    double optim  = 12.0 * psi_B / Nd;  // 分片!
    return {params, grads, optim, params + grads + optim, 3.0};
}

int main() {
    std::cout << "============================================================\n";
    std::cout << "         ZeRO 三阶段内存消耗分析 (混合精度 + Adam)\n";
    std::cout << "============================================================\n\n";

    // ---- 场景 1: 7.5B 参数模型, 不同 GPU 数量 ----
    double psi = 7.5e9;  // 7.5B 参数
    double psi_B = psi / (1024.0 * 1024 * 1024);  // 转换为"G 个参数"

    std::cout << "场景: 7.5B 参数模型 (类 GPT)\n";
    std::cout << "混合精度 + Adam, 每参数 16 字节模型状态\n";
    std::cout << "基线内存: " << std::fixed << std::setprecision(1)
              << 16.0 * psi_B << " GB\n\n";

    std::cout << std::setw(6)  << "N_d"
              << std::setw(14) << "标准 DP"
              << std::setw(14) << "ZeRO-1"
              << std::setw(14) << "ZeRO-2"
              << std::setw(14) << "ZeRO-3"
              << std::setw(14) << "ZeRO-3节省" << "\n";
    std::cout << std::string(76, '-') << "\n";

    for (int Nd : {1, 4, 8, 16, 32, 64, 128, 256}) {
        auto dp = calc_standard_dp(psi_B, Nd);
        auto z1 = calc_zero_stage1(psi_B, Nd);
        auto z2 = calc_zero_stage2(psi_B, Nd);
        auto z3 = calc_zero_stage3(psi_B, Nd);

        std::cout << std::setw(6)  << Nd
                  << std::setw(11) << std::fixed << std::setprecision(1)
                  << dp.total_gb << " GB"
                  << std::setw(11) << z1.total_gb << " GB"
                  << std::setw(11) << z2.total_gb << " GB"
                  << std::setw(11) << z3.total_gb << " GB"
                  << std::setw(11) << std::setprecision(0)
                  << dp.total_gb / z3.total_gb << "x"
                  << "\n";
    }

    std::cout << "\n";

    // ---- 场景 2: 固定 64 GPU, 不同模型大小 ----
    int Nd = 64;
    std::cout << "============================================================\n";
    std::cout << "固定 " << Nd << " 张 GPU, 不同模型大小的每卡内存 (GB)\n";
    std::cout << "============================================================\n\n";

    std::cout << std::setw(12) << "模型大小"
              << std::setw(14) << "标准 DP"
              << std::setw(14) << "ZeRO-1"
              << std::setw(14) << "ZeRO-2"
              << std::setw(14) << "ZeRO-3"
              << std::setw(16) << "可用32GB?" << "\n";
    std::cout << std::string(84, '-') << "\n";

    struct ModelConfig { const char* name; double params; };
    ModelConfig models[] = {
        {"1.5B",   1.5e9},
        {"7.5B",   7.5e9},
        {"20B",    20e9},
        {"100B",   100e9},
        {"175B",   175e9},
        {"1T",     1e12},
    };

    for (auto& mc : models) {
        double pb = mc.params / (1024.0 * 1024 * 1024);
        auto dp = calc_standard_dp(pb, Nd);
        auto z1 = calc_zero_stage1(pb, Nd);
        auto z2 = calc_zero_stage2(pb, Nd);
        auto z3 = calc_zero_stage3(pb, Nd);

        std::string fit = z3.total_gb < 32.0 ? "YES" :
                         (z3.total_gb < 80.0 ? "A100" : "NO");

        std::cout << std::setw(12) << mc.name
                  << std::setw(11) << std::fixed << std::setprecision(1)
                  << dp.total_gb << " GB"
                  << std::setw(11) << z1.total_gb << " GB"
                  << std::setw(11) << z2.total_gb << " GB"
                  << std::setw(11) << z3.total_gb << " GB"
                  << std::setw(12) << fit << "\n";
    }

    std::cout << "\n";

    // ---- 场景 3: 通信量对比 ----
    std::cout << "============================================================\n";
    std::cout << "             通信量对比 (每训练步)\n";
    std::cout << "============================================================\n\n";

    std::cout << "  标准 DP:  2Ψ 元素 (1x AllReduce = ReduceScatter + AllGather)\n";
    std::cout << "  ZeRO-1:  2Ψ 元素 (ReduceScatter 梯度 + AllGather 参数)\n";
    std::cout << "  ZeRO-2:  2Ψ 元素 (ReduceScatter 梯度 + AllGather 参数)\n";
    std::cout << "  ZeRO-3:  3Ψ 元素 (AllGather 参数×2 + ReduceScatter 梯度)\n";
    std::cout << "\n";
    std::cout << "  结论: ZeRO-1/2 通信量与标准 DP 完全相同!\n";
    std::cout << "        ZeRO-3 仅增加 50% 通信量, 换来 N_d 倍内存节省.\n";

    // ---- 场景 4: 内存组成饼图 (文字版) ----
    std::cout << "\n============================================================\n";
    std::cout << "     7.5B 模型, 64 张 GPU 各方案内存组成\n";
    std::cout << "============================================================\n\n";

    double pb = 7.5e9 / (1024.0 * 1024 * 1024);
    struct { const char* name; MemoryProfile mp; } stages[] = {
        {"标准 DP", calc_standard_dp(pb, 64)},
        {"ZeRO-1", calc_zero_stage1(pb, 64)},
        {"ZeRO-2", calc_zero_stage2(pb, 64)},
        {"ZeRO-3", calc_zero_stage3(pb, 64)},
    };

    for (auto& s : stages) {
        double total = s.mp.total_gb;
        int bar_p = std::max(1, (int)(s.mp.params_gb / total * 40));
        int bar_g = std::max(1, (int)(s.mp.grads_gb / total * 40));
        int bar_o = 40 - bar_p - bar_g;
        if (bar_o < 1) bar_o = 1;

        std::cout << "  " << std::setw(8) << s.name << " ["
                  << std::string(bar_p, 'P')
                  << std::string(bar_g, 'G')
                  << std::string(bar_o, 'O')
                  << "] " << std::fixed << std::setprecision(2)
                  << total << " GB\n";
        std::cout << "           P=参数 " << std::setprecision(2) << s.mp.params_gb
                  << "GB  G=梯度 " << s.mp.grads_gb
                  << "GB  O=优化器 " << s.mp.optim_gb << "GB\n\n";
    }

    return 0;
}
```

---

## 通信量分析：ZeRO 的代价是什么？

ZeRO 节省了大量内存，但分布式系统中没有免费的午餐——分片必然意味着更多的通信。论文用严格的分析证明了一个令人振奋的结果：**ZeRO-1 和 ZeRO-2 的通信量与标准数据并行完全相同**。

### 标准数据并行的通信量

标准 DP 在每一步训练中执行一次 AllReduce 操作来同步梯度。AllReduce 可以分解为 Reduce-Scatter + AllGather，其通信量为：

$$
\text{标准 DP 通信量} = 2\Psi \text{ 元素} \quad (\text{Reduce-Scatter: } \Psi + \text{AllGather: } \Psi)
$$

### ZeRO-1 和 ZeRO-2 的通信量

在 ZeRO-1 和 ZeRO-2 中，每一步训练执行两个通信操作：

1. **Reduce-Scatter 梯度**（\(\Psi\) 元素）：每张卡得到自己负责的 \(1/N_d\) 梯度的聚合结果
2. **AllGather 更新后参数**（\(\Psi\) 元素）：参数更新后，每张卡需要获得完整参数

总通信量：\(2\Psi\) 元素——**与标准 DP 完全一样！**

::: tip 为什么通信量不变？
标准 DP 的 AllReduce 本质上也是 Reduce-Scatter + AllGather。ZeRO-1/2 只是把 AllGather 从"聚合梯度后立刻执行"改为了"参数更新后再执行"。两个操作还在，只是 **时间点不同**。
:::

### ZeRO-3 的通信量

ZeRO-3 需要额外的 AllGather 来收集前向和反向传播所需的参数：

1. **前向传播 AllGather 参数**（\(\Psi\) 元素）：逐层收集完整参数
2. **反向传播 AllGather 参数**（\(\Psi\) 元素）：再次逐层收集
3. **Reduce-Scatter 梯度**（\(\Psi\) 元素）：分片聚合梯度

总通信量：\(3\Psi\) 元素——比标准 DP 多 **50%**。

```
通信量对比:

标准 DP:    [ReduceScatter Ψ] + [AllGather Ψ] = 2Ψ
             ↑ 梯度聚合         ↑ 梯度广播

ZeRO-1/2:  [ReduceScatter Ψ] + [AllGather Ψ] = 2Ψ
             ↑ 梯度聚合         ↑ 更新后参数广播

ZeRO-3:    [AllGather Ψ] + [AllGather Ψ] + [ReduceScatter Ψ] = 3Ψ
             ↑ 前向参数     ↑ 反向参数      ↑ 梯度聚合

ZeRO-3 额外增加的 50% 通信量, 换来了 N_d 倍的内存节省.
对于通信带宽充足的场景 (如 NVLink 连接), 这是非常划算的交易.
```

### 通信与计算的重叠

ZeRO-3 的通信开销可以通过 **通信-计算重叠（Communication-Computation Overlap）** 进一步隐藏。在前向传播中：

1. 当第 \(l\) 层在计算时，提前发起第 \(l+1\) 层的 AllGather
2. 只要 AllGather 在第 \(l\) 层计算完成之前结束，通信时间就完全被隐藏

在反向传播中类似——当前层的梯度 Reduce-Scatter 可以与下一层的反向计算重叠。

---

## ZeRO-R：残余内存优化

ZeRO-DP 解决了模型状态的冗余问题。但如前文分析，训练中还有 **残余状态**（激活值、临时缓冲区、内存碎片）消耗大量显存。ZeRO-R 针对这三类残余状态分别提出了优化方案。

### 1. 分片激活检查点（Partitioned Activation Checkpointing，\(P_a\)）

**问题**：激活检查点（Activation Checkpointing）已经大幅减少了激活值内存，但对于非常大的模型，即使检查点后的激活值仍然很大。而且在使用模型并行时，每张卡保存的是完整激活值的副本（因为每张卡都需要完整激活值来做反向传播）。

**解决方案**：
- 将检查点激活值 **分片** 到不同的数据并行进程
- 每张卡只存 \(1/N_d\) 的激活值检查点
- 需要时通过 AllGather 收集

**额外技巧**：当 CPU 内存充足时，可以将分片的激活值 **卸载到 CPU 内存**，几乎完全消除激活值的 GPU 内存开销。

### 2. 恒定大小缓冲区（Constant Size Buffers，\(C_B\)）

**问题**：一些操作（如 AllReduce、梯度范数计算）需要把所有参数拼接到一个连续缓冲区中。如果模型很大，这个缓冲区也会很大。

**解决方案**：使用固定大小的缓冲区。当需要处理的数据超过缓冲区大小时，分多次处理。这保证了缓冲区内存与模型大小 **解耦**。

### 3. 内存碎片整理（Memory Defragmentation，\(M_D\)）

**问题**：训练过程中，激活值和梯度的生命周期不同——激活值在前向传播中创建、反向传播中消费，梯度则相反。这种交错的申请/释放模式导致严重的内存碎片化。

**解决方案**：预分配连续的内存池，为激活值和梯度分别管理。通过预分配避免运行时的频繁 malloc/free，从根本上消除碎片。

```
内存碎片问题与 ZeRO-R 解决方案:

碎片化的内存布局:
┌────┬──┬────┬──┬────┬──┬────┬──┬────┐
│Act1│空│Grad│空│Act2│空│Grad│空│Act3│
└────┴──┴────┴──┴────┴──┴────┴──┴────┘
  总空闲内存足够, 但无法分配连续大块 → OOM!

ZeRO-R 的内存池方案:
┌────────────────────┬────────────────────┐
│   激活值内存池       │   梯度内存池        │
│ ┌────┬────┬────┐   │ ┌────┬────┬────┐   │
│ │Act1│Act2│Act3│   │ │Grd1│Grd2│Grd3│   │
│ └────┴────┴────┘   │ └────┴────┴────┘   │
│   连续、无碎片      │   连续、无碎片       │
└────────────────────┴────────────────────┘
```

---

## ZeRO 与模型并行的关系

论文特别强调了 ZeRO 与模型并行（MP）的互补关系。

### ZeRO 何时可以替代 MP？

**当通信带宽足够时**（如单节点内 NVLink），ZeRO-3 可以完全替代模型并行：

- ZeRO-3 的通信量为 \(3\Psi\)，且是 AllGather/Reduce-Scatter（大块数据传输，带宽利用率高）
- 模型并行每层需要 2 次 AllReduce，通信更频繁但单次数据量更小

在单节点 8 卡场景下，ZeRO-3 通常比同等 GPU 数量的模型并行更高效。

### ZeRO 何时需要与 MP 配合？

**当模型的单层参数就超过单卡容量时**，ZeRO-3 的 AllGather 虽然分摊了持久存储，但前向/反向传播时仍需要短暂地持有一层的完整参数。如果单层参数就超过单卡显存，则必须使用模型并行（张量并行）来切分层内参数。

最佳实践：
- **节点内**：使用模型并行（利用 NVLink 的高带宽）
- **跨节点**：使用 ZeRO-DP（AllGather/Reduce-Scatter 对带宽要求更友好）

```
混合并行策略 (ZeRO + MP):

           节点 0                        节点 1
    ┌───────────────────┐        ┌───────────────────┐
    │ GPU0  GPU1  GPU2  GPU3│    │ GPU4  GPU5  GPU6  GPU7│
    │ ←── MP (NVLink) ──→│        │ ←── MP (NVLink) ──→│
    │ 模型并行度 = 4      │        │ 模型并行度 = 4      │
    └─────────┬─────────┘        └─────────┬─────────┘
              │                            │
              └────── ZeRO-DP (以太网) ─────┘
                    数据并行度 = 2
```

---

## 实验结果与关键发现

### 实验设置

论文在以下环境中进行评估：
- **硬件**：最多 400 块 NVIDIA V100 GPU（32 GB），跨 25 个 DGX-2 节点
- **互连**：节点内 NVSwitch（300 GB/s），节点间 InfiniBand EDR（100 Gb/s）
- **模型**：基于 GPT-2 架构，参数量从 8.3B 到 100B
- **基线**：Megatron-LM（当时 SOTA 的模型并行实现）

### 关键结果

**1. 模型规模突破**

| 配置 | 最大可训练参数量 |
|------|----------------|
| Megatron-LM (MP only, 16 GPUs) | 40B |
| ZeRO-1 + MP (64 GPUs) | 60B |
| ZeRO-2 + MP (64 GPUs) | **170B** |
| ZeRO-3 (理论, 1024 GPUs) | **> 1T** |

**2. 训练吞吐量**

- **100B 模型在 400 GPUs 上**：超过 **15 PetaFLOPs**（每 GPU 约 38 TFLOPS）
- 这相当于峰值性能的 **30%+**——对于大模型训练是非常好的利用率

**3. 超线性加速**

论文发现了一个有趣的现象：**ZeRO 有时候在增加 GPU 数量时表现出超线性加速**。

原因：更多的 GPU → 每卡内存更少 → 可以使用更大的 batch size per GPU（因为有更多空闲内存给激活值）→ 更高的计算效率。

**4. 与 Megatron-LM 的对比**

在 8.3B 模型（Megatron-LM 的最佳配置）上：
- **Megatron-LM（MP=8）**：在 8 卡内效率高，但跨节点后效率急剧下降
- **ZeRO-1 + MP=4**：在相同总 GPU 数下，吞吐量更高，因为减少了跨节点的 MP 通信

::: warning 论文的局限性
论文发表时（2020 年），ZeRO-3 的完整实现尚未完成。论文中的实验主要基于 ZeRO-1 和 ZeRO-2，ZeRO-3 的结果为理论分析。ZeRO-3 后来在 DeepSpeed 库中被完整实现，并在实践中得到了广泛验证。
:::

---

## 深入理解：为什么 AllReduce = Reduce-Scatter + AllGather？

这是理解 ZeRO 通信量分析的基础。让我们用一个具体的例子来说明。

### AllReduce 的两步分解

设 4 张 GPU 各有一个长度为 4 的梯度向量需要求和：

```
初始状态:
  GPU 0: [a0, a1, a2, a3]
  GPU 1: [b0, b1, b2, b3]
  GPU 2: [c0, c1, c2, c3]
  GPU 3: [d0, d1, d2, d3]

目标: 每张 GPU 得到 [a0+b0+c0+d0, a1+b1+c1+d1, a2+b2+c2+d2, a3+b3+c3+d3]
```

**Step 1: Reduce-Scatter**（每卡发送 \(3\Psi/4\)，接收 \(3\Psi/4\)）

每张卡负责一个分片的归约结果：

```
Reduce-Scatter 后:
  GPU 0: [a0+b0+c0+d0,  _,  _,  _]   ← 只得到分片 0 的结果
  GPU 1: [_, a1+b1+c1+d1,  _,  _]     ← 只得到分片 1 的结果
  GPU 2: [_,  _, a2+b2+c2+d2,  _]     ← 只得到分片 2 的结果
  GPU 3: [_,  _,  _, a3+b3+c3+d3]     ← 只得到分片 3 的结果

通信量: 每卡发送 N(1-1/N_d) ≈ N 个元素
```

**Step 2: AllGather**（每卡发送 \(\Psi/4\)，接收 \(3\Psi/4\)）

每张卡把自己的归约结果广播给所有人：

```
AllGather 后:
  GPU 0: [a0+b0+c0+d0, a1+b1+c1+d1, a2+b2+c2+d2, a3+b3+c3+d3]
  GPU 1: [a0+b0+c0+d0, a1+b1+c1+d1, a2+b2+c2+d2, a3+b3+c3+d3]
  GPU 2: [a0+b0+c0+d0, a1+b1+c1+d1, a2+b2+c2+d2, a3+b3+c3+d3]
  GPU 3: [a0+b0+c0+d0, a1+b1+c1+d1, a2+b2+c2+d2, a3+b3+c3+d3]

通信量: 每卡发送 N/N_d ≈ N/N_d 个元素, 但总传输约 N 个元素
```

总通信量 ≈ \(2\Psi\)（Reduce-Scatter \(\Psi\) + AllGather \(\Psi\)）。

### ZeRO 的巧妙之处

标准 DP：`Reduce-Scatter(梯度)` → `AllGather(梯度)` → 更新参数

ZeRO-1/2：`Reduce-Scatter(梯度)` → 更新参数 → `AllGather(参数)`

**通信操作完全一样，只是 AllGather 的对象从"聚合后的梯度"变成了"更新后的参数"，总量不变！**

---

## ZeRO 的工程实现：DeepSpeed 库

ZeRO 的算法由微软开源的 **DeepSpeed** 库实现。使用方式非常简洁——只需几行配置就能启用不同阶段的优化。

### 基本使用方式

使用 DeepSpeed 只需要三步：

1. 在 `deepspeed_config.json` 中配置 ZeRO 阶段
2. 用 `deepspeed.initialize()` 包装模型和优化器
3. 用 `deepspeed` 命令启动训练

```python
# 1. deepspeed_config.json
{
    "zero_optimization": {
        "stage": 3,                      # ZeRO 阶段: 0, 1, 2, 3
        "overlap_comm": true,            # 通信-计算重叠
        "contiguous_gradients": true,    # 连续梯度 (减少碎片)
        "reduce_bucket_size": 5e8,       # Reduce-Scatter 桶大小
        "stage3_prefetch_bucket_size": 5e8,
        "stage3_param_persistence_threshold": 1e6
    },
    "fp16": {
        "enabled": true
    },
    "train_batch_size": 32,
    "train_micro_batch_size_per_gpu": 4
}
```

```python
# 2. 训练代码
import deepspeed

model = MyModel()
model, optimizer, _, _ = deepspeed.initialize(
    model=model,
    model_parameters=model.parameters(),
    config="deepspeed_config.json"
)

for batch in dataloader:
    loss = model(batch)
    model.backward(loss)
    model.step()
```

```bash
# 3. 启动命令
deepspeed --num_gpus=8 train.py
```

### ZeRO-3 特有的配置项

ZeRO-3 引入了一些额外的配置来平衡内存和性能：

| 配置项 | 含义 | 推荐值 |
|--------|------|--------|
| `stage3_prefetch_bucket_size` | 预取参数的桶大小 | 模型参数量的 1%-5% |
| `stage3_param_persistence_threshold` | 小于此阈值的参数不分片 | 1e5 ~ 1e6 |
| `stage3_max_live_parameters` | 同时存在的最大参数量 | 根据显存调整 |
| `stage3_max_reuse_distance` | 参数复用距离阈值 | 根据模型结构调整 |

`stage3_param_persistence_threshold` 的含义是：对于元素数量小于阈值的参数（如 LayerNorm 的 bias），保持每卡一份而不分片。因为这些小参数的 AllGather 开销（延迟）可能大于内存节省。

---

## 总结与启示

### ZeRO 的核心贡献

1. **重新定义了数据并行**：传统观点认为数据并行只能用于模型放得下单卡的情况。ZeRO 证明了通过消除冗余，数据并行可以处理 **任意大** 的模型

2. **提供了一个内存-通信权衡的连续谱**：从 ZeRO-1（少量内存节省，无额外通信）到 ZeRO-3（最大内存节省，50% 额外通信），用户可以根据实际环境选择最优平衡点

3. **与模型并行正交互补**：ZeRO 不是模型并行的替代品，而是互补品。两者结合形成的混合并行策略是当前大模型训练的标准范式

### 对后续工作的影响

ZeRO 的思想深刻影响了后续的分布式训练研究：

- **ZeRO-Offload / ZeRO-Infinity**：将分片的模型状态进一步卸载到 CPU 内存甚至 NVMe SSD，使单 GPU 也能训练大模型
- **FSDP（Fully Sharded Data Parallel）**：PyTorch 官方实现的类 ZeRO-3 方案
- **3D 并行**：数据并行（ZeRO）+ 模型并行（张量并行）+ 流水线并行的组合，成为 GPT-3、PaLM 等万亿参数模型训练的标准方案

::: tip 一个深刻的观察
ZeRO 的核心洞察与 FlashAttention 异曲同工——**优化的关键不在于减少计算量，而在于减少数据搬运**。FlashAttention 优化的是 GPU 内部 SRAM 与 HBM 之间的数据搬运；ZeRO 优化的是 GPU 之间的冗余数据存储。两者都是从"数据在哪里"而非"计算有多少"的角度出发，找到了突破性的优化方案。
:::

---

## 参考文献

1. Rajbhandari, S., Rasley, J., Ruwase, O., & He, Y. (2020). **ZeRO: Memory Optimizations Toward Training Trillion Parameter Models**. SC 2020. [arXiv:1910.02054](https://arxiv.org/abs/1910.02054)

2. Shoeybi, M., et al. (2019). **Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism**. [arXiv:1909.08053](https://arxiv.org/abs/1909.08053)

3. Rajbhandari, S., et al. (2021). **ZeRO-Infinity: Breaking the GPU Memory Wall for Extreme Scale Deep Learning**. SC 2021. [arXiv:2104.07857](https://arxiv.org/abs/2104.07857)

4. Ren, J., et al. (2021). **ZeRO-Offload: Democratizing Billion-Scale Model Training**. USENIX ATC 2021. [arXiv:2101.06840](https://arxiv.org/abs/2101.06840)

5. Rasley, J., et al. (2020). **DeepSpeed: System Optimizations Enable Training Deep Learning Models with Over 100 Billion Parameters**. KDD 2020.
