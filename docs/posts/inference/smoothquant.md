---
title: "SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models"
date: 2026-02-15
---

# SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models

<p style="color: var(--vp-c-text-2); font-size: 14px;">
2026-02-15 &nbsp;·&nbsp; 推理引擎 &nbsp;·&nbsp; 论文精读
</p>

> **论文信息**
> - **作者**: Guangxuan Xiao, Ji Lin, Mickael Seznec, Hao Wu, Julien Demouth, Song Han
> - **机构**: MIT, NVIDIA
> - **发表**: ICML 2023
> - **链接**: [arXiv:2211.10438](https://arxiv.org/abs/2211.10438)

## 一句话总结

SmoothQuant 提出了一种 **无需训练（Post-Training）** 的 LLM 量化方法，核心思想是通过一个数学等价的 **平滑变换（Smoothing Transformation）**，将激活值中难以量化的 **离群值（Outlier）** 迁移到容易量化的权重上，从而实现 **W8A8**（权重和激活都用 INT8）的全量化推理，在 OPT-175B 上仅损失不到 1% 的精度，同时获得 **1.56 倍** 的推理加速和近 **2 倍** 的显存节省。

---

## Introduction：为什么 LLM 量化这么难？

### 1. 量化的动机：LLM 推理的计算与内存瓶颈

LLM 推理面临两大瓶颈：

| 瓶颈 | 原因 | 理想解决方案 |
|------|------|------------|
| **内存瓶颈** | 模型参数太大（175B = 350 GB fp16） | 用更少的位数存储权重 |
| **计算瓶颈** | 矩阵乘法的 FLOP 太多 | 用低精度计算（INT8 比 fp16 快 2x） |

INT8 量化是最直接的方案：
- 内存减半（fp16 → INT8）
- INT8 矩阵乘法的理论吞吐量是 fp16 的 **2 倍**（在支持 INT8 Tensor Core 的 GPU 上）

### 2. 权重量化 vs 激活量化

量化可以分为两类：

**仅权重量化（Weight-only Quantization, W8A16）**：
- 权重量化为 INT8，激活保持 fp16
- 推理时将权重反量化为 fp16 再做矩阵乘法
- **好处**：减少内存，权重很容易量化
- **坏处**：计算仍然在 fp16 下进行，**没有计算加速**

**全量化（Weight + Activation Quantization, W8A8）**：
- 权重和激活都量化为 INT8
- 矩阵乘法直接用 INT8 GEMM
- **好处**：内存减半 + 计算加速 2x
- **坏处**：激活值量化非常困难！

```
两种量化方案的对比:

W8A16 (仅权重量化):
  权重 (INT8) → 反量化为 fp16 → fp16 GEMM ← 激活 (fp16)
  ✓ 内存节省    ✗ 无计算加速

W8A8 (全量化):                    ← SmoothQuant 的目标!
  权重 (INT8) → INT8 GEMM ← 激活 (INT8)
  ✓ 内存节省    ✓ 计算加速 2x
```

### 3. 激活值量化的核心挑战：离群值

论文发现 LLM 的激活值有一个独特且致命的特征——**系统性离群值（Systematic Outliers）**。

在 OPT、BLOOM、LLaMA 等大模型中，激活矩阵的某些 **固定通道（Channel）** 会出现比其他通道大 **100 倍** 以上的极端值：

```
激活值矩阵中的离群值 (示意):

            通道 0   通道 1   通道 2   通道 3   通道 4   ...
Token 0  [  0.12,    0.08,  -120.5,    0.15,    0.03,  ... ]
Token 1  [  0.09,   -0.11,   105.3,   -0.07,    0.14,  ... ]
Token 2  [ -0.15,    0.06,  -118.7,    0.11,   -0.09,  ... ]
Token 3  [  0.07,    0.13,   112.1,   -0.12,    0.08,  ... ]
             ↑                  ↑
           正常范围           离群通道!
           [-1, 1]            |值| > 100

特征:
  1. 离群值出现在 固定的通道 (所有 token 的同一列)
  2. 幅度比正常值大 100x+
  3. 在大模型 (>6.7B) 中普遍存在
  4. 这些通道对模型质量至关重要, 不能简单裁剪!
```

### 4. 为什么离群值让量化失败？

量化的本质是将浮点数映射到有限的整数范围内。INT8 的范围是 \([-128, 127]\)。

**对称量化公式**：

$$
X_{\text{int8}} = \text{round}\!\left(\frac{X}{\Delta}\right), \quad \Delta = \frac{\max(|X|)}{127}
$$

当存在离群值时：

$$
\Delta = \frac{120.5}{127} \approx 0.95
$$

正常值 0.12 被量化为 \(\text{round}(0.12 / 0.95) = \text{round}(0.126) = 0\)——**信息完全丢失！**

```
量化精度损失示例:

原始激活: [0.12, 0.08, -120.5, 0.15, 0.03]
量化步长 Δ = 120.5 / 127 = 0.949

量化结果: [round(0.12/0.949), round(0.08/0.949), round(-120.5/0.949), ...]
        = [0, 0, -127, 0, 0]

反量化:   [0, 0, -120.5, 0, 0]

误差:     [0.12, 0.08, 0, 0.15, 0.03]  ← 正常值全部丢失!

一个离群值毁掉了整行的量化精度!
```

### 5. 现有方案的困境

| 方案 | 做法 | 问题 |
|------|------|------|
| **Per-tensor 量化** | 整个张量共享一个 \(\Delta\) | 离群值导致步长过大，正常值精度损失 |
| **Per-token 量化** | 每行一个 \(\Delta\) | 离群值在每行都有，每行的 \(\Delta\) 都被撑大 |
| **Per-channel 量化** | 每列一个 \(\Delta\) | 离群通道和正常通道分别量化 → 精度好！**但 INT8 GEMM 硬件不支持** |
| **混合精度** (LLM.int8()) | 离群通道用 fp16，其余用 INT8 | 需要分解矩阵，额外开销大，**难以真正加速** |

::: warning Per-channel 激活量化的硬件困境
Per-channel（每列一个 \(\Delta\)）能很好地处理离群值，因为离群通道有自己独立的 \(\Delta\)。但问题是：**现有 INT8 GEMM 硬件（如 NVIDIA Tensor Core）只支持 per-tensor 或 per-token 的激活量化**，不支持 per-channel。这是因为矩阵乘法的累加方向与通道维度相同，per-channel 的缩放因子无法在硬件中高效融合。
:::

### 6. SmoothQuant 的核心洞察

论文的核心洞察非常精妙：

> **激活值难以量化（因为离群值），但权重很容易量化（分布均匀）。**
> **能否将量化的"难度"从激活值迁移到权重上？**

答案是 **可以！** 通过一个数学等价的平滑变换：

$$
Y = (X \text{diag}(s)^{-1}) \cdot (\text{diag}(s) W) = \hat{X} \hat{W}
$$

其中 \(s\) 是一个逐通道的缩放因子向量。这个变换：
- 将激活 \(X\) 的每个通道除以 \(s_j\)（**压缩离群值**）
- 将权重 \(W\) 的每个通道乘以 \(s_j\)（**吸收离群值**）
- 数学上完全等价——不改变输出结果

::: tip 命名含义
"Smooth" 指的是对激活值进行 **平滑**——将尖锐的离群峰压平，使量化范围更均匀。就像给一条崎岖的山路铺上沥青，让车（量化器）能平稳行驶。
:::

---

## 核心方法：平滑变换

### 数学推导

对于线性层 \(Y = XW\)，其中 \(X \in \mathbb{R}^{T \times C_{\text{in}}}\)，\(W \in \mathbb{R}^{C_{\text{in}} \times C_{\text{out}}}\)。

引入逐通道缩放因子 \(s \in \mathbb{R}^{C_{\text{in}}}\)，定义平滑变换：

$$
Y = XW = (X \text{diag}(s)^{-1}) (\text{diag}(s) W) = \hat{X} \hat{W}
$$

其中：
- \(\hat{X}_{:,j} = X_{:,j} / s_j\)（激活的第 \(j\) 通道除以 \(s_j\)）
- \(\hat{W}_{j,:} = s_j \cdot W_{j,:}\)（权重的第 \(j\) 行乘以 \(s_j\)）

**关键性质**：这是一个 **恒等变换**——\(\hat{X}\hat{W} = XW\)，输出完全不变。但 \(\hat{X}\) 和 \(\hat{W}\) 的数值分布都发生了改变。

```
平滑变换的效果:

原始状态:
  X (激活):  [0.12, 0.08, -120.5, 0.15]  ← 通道2有离群值
  W (权重):  通道2的权重值在 [-0.5, 0.5] 范围  ← 均匀分布

选择 s = [1, 1, 120, 1]:  (离群通道缩放因子大)

平滑后:
  X̂ (激活): [0.12, 0.08, -1.004, 0.15]  ← 离群值被压缩!
  Ŵ (权重): 通道2的权重值被放大 120 倍   ← 权重吸收了离群值

结果:
  X̂ × Ŵ = X × W  (数学完全等价)
  但 X̂ 变得容易量化了! (所有通道范围相近)
  Ŵ 虽然更大, 但权重本来就容易量化 (per-channel 权重量化很成熟)
```

### 缩放因子 \(s\) 的选择

如何选择 \(s_j\) 是整个算法的关键。论文提出了一个简洁的公式：

$$
s_j = \frac{\max(|X_{:,j}|)^{\alpha}}{\max(|W_{j,:}|)^{1-\alpha}}
$$

其中 \(\alpha \in [0, 1]\) 是 **迁移强度（Migration Strength）** 超参数。

**直觉理解**：
- \(\alpha = 0\)：\(s_j = 1/\max(|W_{j,:}|)\)，完全不平滑激活，只缩放权重 → 对激活没有帮助
- \(\alpha = 1\)：\(s_j = \max(|X_{:,j}|)\)，将所有激活通道压缩到相同范围 → 激活完美平滑，但权重可能被撑大太多
- \(\alpha = 0.5\)：**在激活和权重之间平均分担量化难度** → 论文推荐的默认值

```
不同 α 值的效果:

α = 0 (不迁移):
  激活: [0.12, 0.08, -120.5, 0.15]  ← 离群值还在
  权重: 不变
  → 激活量化仍然困难

α = 0.5 (均衡迁移):  ← 推荐!
  激活: [0.12, 0.08,  -1.0,  0.15]  ← 离群值被大幅压缩
  权重: 通道2 放大 ~11x              ← 权重稍微变大, 但仍可量化
  → 两边都容易量化!

α = 1.0 (完全迁移):
  激活: [0.12, 0.08,  -1.0,  0.15]  ← 激活完美平滑
  权重: 通道2 放大 ~120x             ← 权重可能溢出!
  → 激活容易量化, 但权重可能出问题
```

### 校准过程

缩放因子 \(s\) 的计算需要知道激活值的统计量（每通道最大值）。这通过 **校准（Calibration）** 获得：

1. 在一小批校准数据（通常 128-512 个样本）上做前向传播
2. 收集每层激活矩阵的 **逐通道最大绝对值** \(\max(|X_{:,j}|)\)
3. 结合权重矩阵的逐通道最大值，计算 \(s_j\)
4. 将平滑变换 **离线融合** 到模型权重中（\(\hat{W} = \text{diag}(s) W\)）

```
离线校准与融合流程:

校准阶段 (一次性):
  校准数据 → 模型前向传播 → 收集每层的 max(|X[:,j]|)
  → 结合 max(|W[j,:]|) 计算 s_j
  → 将 s 融合到权重: Ŵ = diag(s) · W

推理阶段 (反复执行):
  输入 X → 除以 s (在线平滑): X̂ = X · diag(s)⁻¹
  → INT8 量化 X̂
  → INT8 GEMM: Q(X̂) × Q(Ŵ)  ← 使用硬件加速!
  → 反量化得到输出

注: s 的除法可以融合到前一层的 LayerNorm 中, 几乎零开销!
```

### 与 LayerNorm 的融合

一个精妙的工程优化：平滑变换的逐通道除法 \(X / s\) 可以 **融合到前一层的 LayerNorm** 中。

LayerNorm 的计算是：

$$
\text{LN}(x) = \frac{x - \mu}{\sigma} \cdot \gamma + \beta
$$

融合后：

$$
\text{LN}_{\text{smooth}}(x) = \frac{x - \mu}{\sigma} \cdot \frac{\gamma}{s} + \frac{\beta}{s}
$$

只需离线修改 LayerNorm 的 \(\gamma\) 和 \(\beta\) 参数，运行时 **零额外开销**。

---

## 量化方案细节

### 量化粒度

SmoothQuant 对权重和激活使用不同的量化粒度：

| 对象 | 量化粒度 | 缩放因子数量 | 原因 |
|------|---------|-------------|------|
| **权重** | Per-channel（每输出通道） | \(C_{\text{out}}\) 个 | 权重是静态的，per-channel 不增加运行时开销 |
| **激活** | Per-token（每 token） | \(T\) 个 | 不同 token 的范围不同，per-token 提高精度 |

Per-token 激活量化 + Per-channel 权重量化的组合与 INT8 GEMM 硬件完全兼容。

### 对称量化

SmoothQuant 使用 **对称量化**（以 0 为中心），公式为：

$$
X_{\text{int8}} = \text{clamp}\!\left(\text{round}\!\left(\frac{X}{\Delta}\right),\; -128,\; 127\right)
$$

$$
\Delta = \frac{\max(|X|)}{127}
$$

对称量化的好处是：INT8 GEMM 的结果可以直接通过乘以 \(\Delta_X \cdot \Delta_W\) 反量化，不需要处理零点偏移。

### 量化覆盖范围

SmoothQuant 量化 Transformer 中的 **所有线性层**：

```
Transformer 层中被量化的线性层:

┌────────────────────────────────────────┐
│ Self-Attention                          │
│                                        │
│  X → [Q_proj] → Q                      │ ← INT8 量化
│  X → [K_proj] → K                      │ ← INT8 量化
│  X → [V_proj] → V                      │ ← INT8 量化
│  Attn_out → [O_proj] → Y              │ ← INT8 量化
│                                        │
├────────────────────────────────────────┤
│ Feed-Forward (MLP)                      │
│                                        │
│  X → [FC1 / gate_proj] → H            │ ← INT8 量化
│  H → [FC2 / down_proj] → Y            │ ← INT8 量化
│                                        │
└────────────────────────────────────────┘

非线性操作 (Softmax, GeLU, LayerNorm) 保持 fp16
BMM (attention score 计算) 保持 fp16
```

---

## 三个集成级别：O1, O2, O3

论文定义了三个由易到难的集成级别，适应不同的工程约束：

### O1：仅平滑 + FP16 计算

- 将平滑变换融合到权重中
- 仍然使用 fp16 做矩阵乘法
- **好处**：验证平滑变换的精度收益，不需要 INT8 Kernel
- **加速**：无（仍然是 fp16 计算）

### O2：W8A8 静态量化

- 权重 INT8 per-channel 量化（离线完成）
- 激活 INT8 per-tensor 量化（使用校准集确定静态 \(\Delta\)）
- **好处**：使用标准 INT8 GEMM，兼容性最好
- **缺点**：per-tensor 激活量化精度略低
- **加速**：~1.5x

### O3：W8A8 动态量化

- 权重 INT8 per-channel 量化（离线完成）
- 激活 INT8 **per-token 动态量化**（运行时逐 token 计算 \(\Delta\)）
- **好处**：精度最高（每个 token 有自己的缩放因子）
- **缺点**：需要自定义 Kernel 支持动态量化
- **加速**：~1.56x

```
三个集成级别:

O1: SmoothQuant + FP16
    ┌──────────┐     ┌─────┐
    │ 平滑后的  │ fp16 │ fp16│
    │ 权重(fp16)│ GEMM │激活 │
    └──────────┘     └─────┘
    精度: ★★★★  速度: ★★ (无加速)

O2: SmoothQuant + W8A8 (静态)
    ┌──────────┐     ┌──────┐
    │ INT8 权重 │ INT8 │ INT8 │
    │(per-chan) │ GEMM │(静态) │
    └──────────┘     └──────┘
    精度: ★★★  速度: ★★★★

O3: SmoothQuant + W8A8 (动态 per-token)      ← 推荐!
    ┌──────────┐     ┌──────────┐
    │ INT8 权重 │ INT8 │ INT8 激活 │
    │(per-chan) │ GEMM │(per-token)│
    └──────────┘     └──────────┘
    精度: ★★★★  速度: ★★★★
```

---

## 对比：为什么比 LLM.int8() 更好？

LLM.int8()（Dettmers et al., 2022）是 SmoothQuant 之前最知名的 LLM 量化方法。两者的对比非常有启发性。

### LLM.int8() 的方案

LLM.int8() 的做法是 **混合精度分解**：

1. 识别激活矩阵中的离群通道（绝对值 > 阈值 6.0）
2. 将离群通道提取出来，用 fp16 计算
3. 其余通道用 INT8 计算
4. 将两部分结果拼接

```
LLM.int8() 的矩阵分解:

激活矩阵 X:
┌───────────────────────────────────┐
│ 正常通道 │ 离群通道 │ 正常通道     │
│ INT8    │  fp16   │  INT8       │
└─────────┴─────────┴─────────────┘
          ↓
  分解为两个子矩阵:
  X_normal × W_normal (INT8 GEMM)
  X_outlier × W_outlier (fp16 GEMM)
  Y = Y_normal + Y_outlier
```

### SmoothQuant vs LLM.int8()

| 维度 | LLM.int8() | SmoothQuant |
|------|-----------|-------------|
| **方法** | 矩阵分解（离群通道用 fp16） | 数学等价变换（消除离群值） |
| **GEMM 类型** | 混合 INT8 + fp16 | **纯 INT8** |
| **硬件友好性** | 需要矩阵分解 + 拼接，难以优化 | 标准 INT8 GEMM，硬件原生支持 |
| **运行时开销** | 需要动态检测离群通道 | 平滑因子离线融合，运行时零开销 |
| **实际加速** | ~1.0x（基本无加速） | **~1.56x** |
| **精度** | 与 fp16 接近 | 与 fp16 接近 |

**关键差异**：LLM.int8() 的矩阵分解导致了 **两次 GEMM + 一次加法 + 数据搬运**，整体开销甚至可能超过纯 fp16。SmoothQuant 通过在数学上消除离群值，让量化后的矩阵可以直接用 **一次标准 INT8 GEMM** 计算。

::: warning LLM.int8() 为什么无法加速？
LLM.int8() 的离群通道通常只占 0.1-1%，但它必须将完整矩阵分解为两部分、分别计算、再合并。这个分解-合并过程引入的开销抵消了 INT8 的计算收益。就像为了节省 1% 的高速公路费，绕了 50% 的路程。
:::

---

## 激活值离群值的深入分析

### 离群值的分布特征

论文通过大量实验发现了 LLM 激活值离群值的四个关键特征：

**特征 1：通道固定性**

离群值总是出现在 **固定的通道** 上。例如在 OPT-175B 中，约 0.1% 的通道（~7 个通道 / 隐层 12288 维度）是离群通道，且在不同输入下保持不变。

**特征 2：跨层一致性**

同样的通道在 **所有 Transformer 层** 中都表现为离群通道。这意味着离群值是模型学到的结构性特征，而非偶然的数值异常。

**特征 3：幅度随模型增大**

| 模型 | 参数量 | 离群值最大幅度 | 离群通道数 |
|------|--------|-------------|----------|
| OPT-125M | 125M | ~15 | 0-1 |
| OPT-1.3B | 1.3B | ~40 | 2-3 |
| OPT-6.7B | 6.7B | ~60 | 4-5 |
| OPT-13B | 13B | ~80 | 5-6 |
| OPT-30B | 30B | ~100 | 6-7 |
| OPT-175B | 175B | **~150** | **7-8** |

模型越大，离群值越极端——这解释了为什么 **小模型可以直接 INT8 量化但大模型不行**。

**特征 4：对模型质量至关重要**

如果简单裁剪（clip）或移除离群值，模型精度会 **灾难性下降**。这些离群值编码了重要的语义信息。

### 为什么 Per-Token 量化不够？

直觉上，per-token（每行一个 \(\Delta\)）应该能处理不同 token 的不同范围。但问题在于：

$$
\Delta_i = \frac{\max_j |X_{i,j}|}{127}
$$

每行的 \(\Delta_i\) 都被离群通道主导——**因为离群值出现在每一行的同一列**。所以 per-token 量化依然无法避免正常通道的精度损失。

```
Per-token 量化仍然失败:

         通道0  通道1  通道2(离群)  通道3
Token 0: [0.12,  0.08,  -120.5,    0.15]  Δ₀ = 120.5/127 = 0.949
Token 1: [0.09, -0.11,   105.3,   -0.07]  Δ₁ = 105.3/127 = 0.829
Token 2: [-0.15, 0.06,  -118.7,    0.11]  Δ₂ = 118.7/127 = 0.935

→ 每行的 Δ 都被通道2 主导
→ 其他通道的量化分辨率: 0.12/0.949 ≈ 0.13 → round → 0  (丢失!)

SmoothQuant 之后:
         通道0  通道1  通道2(已平滑)  通道3
Token 0: [0.12,  0.08,  -1.00,      0.15]  Δ₀ = 1.00/127 = 0.008
Token 1: [0.09, -0.11,   0.88,     -0.07]  Δ₁ = 0.88/127 = 0.007
Token 2: [-0.15, 0.06,  -0.99,      0.11]  Δ₂ = 0.99/127 = 0.008

→ 每行的 Δ 大幅减小
→ 其他通道的量化分辨率: 0.12/0.008 ≈ 15 → round → 15  (保留!)
```

---

## 端到端推理流水线

### 推理时的完整执行流程

```
SmoothQuant W8A8 推理流水线 (O3 级别):

输入 Token IDs
    │
    ↓
┌───────────────────────────────────────┐
│ Embedding Lookup (fp16)               │
└───────────────┬───────────────────────┘
                │
     ┌──────────┴──────────┐
     │  × L 层 Transformer  │
     │                      │
     │  ┌─ LayerNorm (融合了 1/s) ──┐    │
     │  │ 输出已经是 "平滑" 的激活    │    │
     │  └─────────┬────────────────┘    │
     │            │                      │
     │  ┌─────── INT8 量化 (per-token) ─┐│
     │  │ 对每个 token 计算 Δ, 量化     ││
     │  └─────────┬────────────────────┘│
     │            │                      │
     │  ┌─── INT8 GEMM (Q,K,V proj) ──┐│
     │  │ Q(X̂) × Q(Ŵ_QKV) → INT32    ││
     │  │ → 反量化为 fp16               ││
     │  └─────────┬────────────────────┘│
     │            │                      │
     │  ┌─── Attention (fp16) ─────────┐│
     │  │ Softmax(QK^T/√d) V           ││
     │  └─────────┬────────────────────┘│
     │            │                      │
     │  ┌─── INT8 GEMM (O_proj) ──────┐│
     │  │ 量化 Attn 输出 → INT8 GEMM  ││
     │  └─────────┬────────────────────┘│
     │            │                      │
     │     + Residual + LayerNorm(融合 1/s)│
     │            │                      │
     │  ┌─── INT8 GEMM (FC1) ─────────┐│
     │  │ 量化 → INT8 GEMM → fp16     ││
     │  └─────────┬────────────────────┘│
     │            │                      │
     │     GeLU (fp16)                   │
     │            │                      │
     │  ┌─── INT8 GEMM (FC2) ─────────┐│
     │  │ 量化 → INT8 GEMM → fp16     ││
     │  └─────────┬────────────────────┘│
     │            │                      │
     │     + Residual                    │
     │                                   │
     └──────────┬──────────────────────┘
                │
     Final LayerNorm → LM Head → Logits
```

### INT8 GEMM 的实现

在 NVIDIA GPU 上，INT8 GEMM 使用 Tensor Core 执行：

| 精度 | Tensor Core 吞吐量 (A100) | 相对 fp16 |
|------|--------------------------|----------|
| fp16 | 312 TFLOPS | 1x |
| **INT8** | **624 TOPS** | **2x** |
| fp32 | 19.5 TFLOPS | 0.06x |

INT8 GEMM 的计算流程：

$$
C_{\text{int32}} = A_{\text{int8}} \times B_{\text{int8}} + C_{\text{int32}}
$$

累加在 INT32 下进行（防止溢出），最终结果通过乘以 \(\Delta_A \cdot \Delta_B\) 反量化为 fp16。

---

## 实验结果与关键发现

### 精度评估

**OPT 系列模型在各个 Zero-Shot 任务上的精度**：

| 模型 | 方法 | LAMBADA | HellaSwag | PIQA | WinoGrande | 平均 |
|------|------|---------|-----------|------|-----------|------|
| OPT-6.7B | fp16 | 67.7 | 67.2 | 76.3 | 67.6 | 69.7 |
| OPT-6.7B | **SmoothQuant** | **67.3** | **67.0** | **76.1** | **67.2** | **69.4** |
| OPT-6.7B | 朴素 W8A8 | 42.1 | 55.8 | 70.2 | 58.3 | 56.6 |

| 模型 | 方法 | LAMBADA | HellaSwag | PIQA | WinoGrande | 平均 |
|------|------|---------|-----------|------|-----------|------|
| OPT-175B | fp16 | 76.2 | 78.6 | 80.5 | 72.6 | 77.0 |
| OPT-175B | **SmoothQuant** | **75.8** | **78.3** | **80.2** | **72.1** | **76.6** |
| OPT-175B | LLM.int8() | 76.0 | 78.5 | 80.3 | 72.3 | 76.8 |
| OPT-175B | 朴素 W8A8 | 崩溃 | 崩溃 | 崩溃 | 崩溃 | — |

关键观察：
- **SmoothQuant** 在 OPT-175B 上精度损失 < 0.5%，与 fp16 几乎无差
- **朴素 W8A8**（不做平滑）在大模型上完全崩溃
- **LLM.int8()** 精度略好，但无法实际加速

### 速度评估

在 A100 GPU 上的单层 Transformer 推理延迟：

| 模型 | fp16 | LLM.int8() | SmoothQuant W8A8 | 加速比 |
|------|------|-----------|-----------------|--------|
| OPT-6.7B | 1.00x | 0.96x (更慢!) | **1.51x** | 1.51x |
| OPT-13B | 1.00x | 0.94x (更慢!) | **1.56x** | 1.56x |
| OPT-30B | 1.00x | 0.98x | **1.56x** | 1.56x |
| OPT-175B | 1.00x | 1.01x | **1.56x** | 1.56x |

::: tip LLM.int8() 为什么更慢？
LLM.int8() 的矩阵分解 + 双路计算 + 拼接的开销大于 INT8 加速的收益。特别是在 GPU 上，额外的内存拷贝和 kernel launch 开销是致命的。SmoothQuant 的优势在于使用 **标准的、高度优化的 INT8 GEMM**，没有任何额外开销。
:::

### 内存节省

| 模型 | fp16 内存 | SmoothQuant W8A8 | 节省 |
|------|----------|-----------------|------|
| OPT-6.7B | 13.4 GB | 7.1 GB | 47% |
| OPT-30B | 60 GB | 32 GB | 47% |
| OPT-175B | 350 GB | 178 GB | 49% |

权重从 fp16 变为 INT8，内存几乎减半。

### 迁移强度 \(\alpha\) 的消融实验

| \(\alpha\) | OPT-175B PPL (WikiText-2) | 说明 |
|-----------|--------------------------|------|
| 0.0 | 发散 | 完全不平滑，激活量化失败 |
| 0.3 | 9.82 | 开始有效果 |
| **0.5** | **9.55** | **最优平衡点** |
| 0.7 | 9.58 | 权重开始变大 |
| 0.9 | 10.21 | 权重量化困难 |
| 1.0 | 发散 | 完全迁移，权重溢出 |

\(\alpha = 0.5\) 是最优值，完美平衡了激活和权重的量化难度。

---

## 量化误差的定量分析

下面的代码模拟 SmoothQuant 的平滑变换效果和量化误差：

```cpp-run title="SmoothQuant 平滑变换与量化误差模拟"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#include <algorithm>
#include <numeric>
#include <random>

using Vec = std::vector<float>;

// INT8 对称量化
Vec quantize_int8(const Vec& x, float& scale) {
    float max_val = 0;
    for (float v : x) max_val = std::max(max_val, std::abs(v));
    scale = max_val / 127.0f;
    if (scale < 1e-10f) scale = 1e-10f;

    Vec q(x.size());
    for (size_t i = 0; i < x.size(); i++) {
        float v = std::round(x[i] / scale);
        q[i] = std::max(-128.0f, std::min(127.0f, v));
    }
    return q;
}

// 反量化
Vec dequantize(const Vec& q, float scale) {
    Vec out(q.size());
    for (size_t i = 0; i < q.size(); i++) out[i] = q[i] * scale;
    return out;
}

// 计算均方误差
float mse(const Vec& a, const Vec& b) {
    float sum = 0;
    for (size_t i = 0; i < a.size(); i++) sum += (a[i]-b[i])*(a[i]-b[i]);
    return sum / a.size();
}

int main() {
    std::cout << "=============================================================\n";
    std::cout << "    SmoothQuant 平滑变换效果模拟\n";
    std::cout << "=============================================================\n\n";

    // 模拟一个 4 通道的激活矩阵 (8 tokens × 4 channels)
    // 通道 2 是离群通道
    const int T = 8, C = 4;
    std::mt19937 rng(42);
    std::normal_distribution<float> normal(0, 0.1);
    std::normal_distribution<float> outlier(0, 100.0);

    std::vector<Vec> X(T, Vec(C));
    for (int i = 0; i < T; i++)
        for (int j = 0; j < C; j++)
            X[i][j] = (j == 2) ? outlier(rng) : normal(rng);

    // 模拟权重矩阵 (4×4)
    std::vector<Vec> W(C, Vec(C));
    std::normal_distribution<float> w_dist(0, 0.3);
    for (int i = 0; i < C; i++)
        for (int j = 0; j < C; j++)
            W[i][j] = w_dist(rng);

    // 计算真实输出 Y = X @ W
    std::vector<Vec> Y_true(T, Vec(C, 0));
    for (int i = 0; i < T; i++)
        for (int j = 0; j < C; j++)
            for (int k = 0; k < C; k++)
                Y_true[i][j] += X[i][k] * W[k][j];

    // ---- 1. 显示激活值的离群值情况 ----
    std::cout << "1. 原始激活值 (8 tokens × 4 channels):\n\n";
    std::cout << "  通道最大绝对值: ";
    Vec ch_max(C, 0);
    for (int j = 0; j < C; j++) {
        for (int i = 0; i < T; i++)
            ch_max[j] = std::max(ch_max[j], std::abs(X[i][j]));
        std::cout << std::fixed << std::setprecision(2) << ch_max[j] << "  ";
    }
    std::cout << "\n  通道 2 是离群通道 (值远大于其他通道)!\n\n";

    // ---- 2. 朴素 per-token 量化 ----
    std::cout << "2. 朴素 per-token INT8 量化 (无 SmoothQuant):\n\n";

    std::vector<Vec> X_naive_dq(T, Vec(C));
    float total_naive_mse = 0;
    for (int i = 0; i < T; i++) {
        float scale;
        Vec q = quantize_int8(X[i], scale);
        X_naive_dq[i] = dequantize(q, scale);
        float row_mse = mse(X[i], X_naive_dq[i]);
        total_naive_mse += row_mse;
    }
    total_naive_mse /= T;

    // 朴素量化后做 GEMM
    std::vector<Vec> Y_naive(T, Vec(C, 0));
    for (int i = 0; i < T; i++)
        for (int j = 0; j < C; j++)
            for (int k = 0; k < C; k++)
                Y_naive[i][j] += X_naive_dq[i][k] * W[k][j];

    float y_naive_mse = 0;
    for (int i = 0; i < T; i++) y_naive_mse += mse(Y_true[i], Y_naive[i]);
    y_naive_mse /= T;

    std::cout << "  激活量化 MSE: " << std::scientific << std::setprecision(4)
              << total_naive_mse << "\n";
    std::cout << "  输出误差 MSE: " << y_naive_mse << "\n\n";

    // ---- 3. SmoothQuant 不同 α 值 ----
    std::cout << "=============================================================\n";
    std::cout << "3. SmoothQuant 不同 α 值的效果:\n";
    std::cout << "=============================================================\n\n";

    // 权重通道最大值
    Vec w_ch_max(C, 0);
    for (int j = 0; j < C; j++)
        for (int k = 0; k < C; k++)
            w_ch_max[j] = std::max(w_ch_max[j], std::abs(W[j][k]));

    std::cout << std::setw(8) << "α"
              << std::setw(18) << "激活量化MSE"
              << std::setw(18) << "输出误差MSE"
              << std::setw(14) << "vs 朴素" << "\n";
    std::cout << std::string(58, '-') << "\n";

    for (float alpha : {0.0f, 0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f, 1.0f}) {
        // 计算缩放因子 s
        Vec s(C);
        for (int j = 0; j < C; j++) {
            float x_part = std::pow(ch_max[j], alpha);
            float w_part = std::pow(std::max(w_ch_max[j], 1e-5f), 1.0f - alpha);
            s[j] = x_part / w_part;
            if (s[j] < 1e-5f) s[j] = 1e-5f;
        }

        // 平滑变换: X̂ = X / s, Ŵ = s * W
        std::vector<Vec> X_smooth(T, Vec(C));
        for (int i = 0; i < T; i++)
            for (int j = 0; j < C; j++)
                X_smooth[i][j] = X[i][j] / s[j];

        std::vector<Vec> W_smooth(C, Vec(C));
        for (int j = 0; j < C; j++)
            for (int k = 0; k < C; k++)
                W_smooth[j][k] = s[j] * W[j][k];

        // 量化平滑后的激活
        std::vector<Vec> X_sq_dq(T, Vec(C));
        float sq_act_mse = 0;
        for (int i = 0; i < T; i++) {
            float scale;
            Vec q = quantize_int8(X_smooth[i], scale);
            X_sq_dq[i] = dequantize(q, scale);
            sq_act_mse += mse(X_smooth[i], X_sq_dq[i]);
        }
        sq_act_mse /= T;

        // 量化后做 GEMM (用平滑后的权重)
        std::vector<Vec> Y_sq(T, Vec(C, 0));
        for (int i = 0; i < T; i++)
            for (int j = 0; j < C; j++)
                for (int k = 0; k < C; k++)
                    Y_sq[i][j] += X_sq_dq[i][k] * W_smooth[k][j];

        float y_sq_mse = 0;
        for (int i = 0; i < T; i++) y_sq_mse += mse(Y_true[i], Y_sq[i]);
        y_sq_mse /= T;

        float improvement = y_naive_mse / std::max(y_sq_mse, 1e-20f);

        std::cout << std::setw(8) << std::fixed << std::setprecision(1) << alpha
                  << std::setw(18) << std::scientific << std::setprecision(4) << sq_act_mse
                  << std::setw(18) << y_sq_mse
                  << std::setw(10) << std::fixed << std::setprecision(1)
                  << improvement << "x\n";
    }

    std::cout << "\n  结论: α=0.5 附近达到最优, 输出误差比朴素量化降低数个数量级!\n";

    // ---- 4. 通道级别的平滑效果 ----
    std::cout << "\n=============================================================\n";
    std::cout << "4. α=0.5 时各通道的平滑效果:\n";
    std::cout << "=============================================================\n\n";

    float alpha = 0.5f;
    Vec s(C);
    for (int j = 0; j < C; j++) {
        s[j] = std::pow(ch_max[j], alpha) / std::pow(std::max(w_ch_max[j], 1e-5f), 1-alpha);
    }

    std::cout << std::setw(10) << "通道"
              << std::setw(14) << "原始max|X|"
              << std::setw(10) << "s_j"
              << std::setw(14) << "平滑后max" 
              << std::setw(14) << "压缩比" << "\n";
    std::cout << std::string(62, '-') << "\n";

    for (int j = 0; j < C; j++) {
        float smooth_max = ch_max[j] / s[j];
        std::cout << std::setw(10) << j
                  << std::setw(14) << std::fixed << std::setprecision(2) << ch_max[j]
                  << std::setw(10) << std::setprecision(2) << s[j]
                  << std::setw(14) << std::setprecision(2) << smooth_max
                  << std::setw(12) << std::setprecision(1) << ch_max[j]/smooth_max << "x\n";
    }

    std::cout << "\n  离群通道被大幅压缩, 所有通道范围趋于一致!\n";

    return 0;
}
```

---

## 总结与启示

### SmoothQuant 的核心贡献

1. **发现了关键问题**：LLM 的激活值存在系统性离群值，这是 W8A8 量化的核心障碍

2. **提出了优雅的解决方案**：通过数学等价的平滑变换，将量化难度从激活迁移到权重——**不改变模型输出，但让量化变得容易**

3. **硬件友好**：平滑后的模型可以直接使用标准 INT8 GEMM，无需矩阵分解或混合精度，**真正实现了加速**

4. **实用且通用**：无需重训练（post-training），校准只需几百个样本和几分钟时间，适用于 OPT、BLOOM、LLaMA 等各种 LLM

### 设计哲学

SmoothQuant 体现了一个深刻的优化思想：

> **当问题在 A 侧难以解决时，通过等价变换将它迁移到 B 侧解决。**

- 激活值有离群值 → 量化困难（问题在 A 侧）
- 权重分布均匀 → 量化容易（B 侧有余量）
- 平滑变换 → 将"难度"从 A 迁移到 B（等价变换，不损失信息）

这与 [FlashAttention](./flash-attention) 的思想异曲同工：FlashAttention 通过重排计算顺序，将"内存带宽受限"问题转化为"计算受限"问题（GPU 擅长的）。SmoothQuant 通过重新分配数值范围，将"激活难量化"问题转化为"权重难量化"问题（量化器擅长的）。

::: tip 对后续工作的影响
SmoothQuant 开创了"迁移量化难度"的思路，后续出现了大量跟进工作：
- **AWQ (Activation-Aware Weight Quantization)**：进一步优化权重量化，保护对激活值敏感的权重通道
- **GPTQ**：基于二阶信息的高精度权重量化
- **QuIP / QuIP#**：利用随机旋转实现更均匀的量化
- 这些方法与 SmoothQuant 的核心思想一脉相承——**理解数据分布，然后设计变换使其对量化友好**
:::

---

## 参考文献

1. Xiao, G., Lin, J., Seznec, M., Wu, H., Demouth, J., & Han, S. (2023). **SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models**. ICML 2023. [arXiv:2211.10438](https://arxiv.org/abs/2211.10438)

2. Dettmers, T., Lewis, M., Belkada, Y., & Zettlemoyer, L. (2022). **LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale**. NeurIPS 2022. [arXiv:2208.07339](https://arxiv.org/abs/2208.07339)

3. Lin, J., Tang, J., Tang, H., Yang, S., Dang, X., & Han, S. (2023). **AWQ: Activation-Aware Weight Quantization for LLM Compression and Acceleration**. [arXiv:2306.00978](https://arxiv.org/abs/2306.00978)

4. Frantar, E., Ashkboos, S., Hoefler, T., & Alistarh, D. (2022). **GPTQ: Accurate Post-Training Quantization for Generative Pre-Trained Transformers**. ICLR 2023. [arXiv:2210.17323](https://arxiv.org/abs/2210.17323)

5. Dettmers, T., & Zettlemoyer, L. (2023). **The case for 4-bit precision: k-bit Inference Scaling Laws**. ICML 2023. [arXiv:2212.09720](https://arxiv.org/abs/2212.09720)
