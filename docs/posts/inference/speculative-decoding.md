---
title: "Fast Inference from Transformers via Speculative Decoding"
date: 2026-02-15
---

# Fast Inference from Transformers via Speculative Decoding

<p style="color: var(--vp-c-text-2); font-size: 14px;">
2026-02-15 &nbsp;·&nbsp; 推理引擎 &nbsp;·&nbsp; 论文精读
</p>

> **论文信息**
> - **作者**: Yaniv Leviathan, Matan Kalman, Yossi Matias
> - **机构**: Google Research
> - **发表**: ICML 2023
> - **链接**: [arXiv:2211.17192](https://arxiv.org/abs/2211.17192)

## 一句话总结

Speculative Decoding 提出了一种 **无损加速** 自回归解码的方法：用一个快速的 **小模型（Draft Model）** 先猜测未来若干 token，再用 **大模型（Target Model）** 并行验证这些猜测，通过精心设计的 **拒绝采样（Rejection Sampling）** 方案保证输出分布与原始大模型 **完全一致**——不改变任何一个 bit 的输出质量，却能获得 **2-3 倍** 的推理加速。

---

## Introduction：为什么自回归解码这么慢？

### 1. 自回归解码的根本瓶颈

LLM 的推理（生成阶段）是 **自回归（Autoregressive）** 的——每次只生成一个 token，且每个 token 的生成依赖所有前面的 token：

$$
x_{t+1} \sim p(x \mid x_1, x_2, \ldots, x_t)
$$

这意味着 \(T\) 个 token 的生成需要串行调用大模型 \(T\) 次。每次调用都需要：
1. 加载完整的模型权重到 GPU 计算单元
2. 执行一次完整的前向传播
3. 只输出 **一个** token

```
自回归解码的串行瓶颈:

步骤 1: "The"     → 加载模型 → 前向传播 → 采样 → "cat"
步骤 2: "The cat" → 加载模型 → 前向传播 → 采样 → "sat"
步骤 3: "The cat sat" → 加载模型 → 前向传播 → 采样 → "on"
...
步骤 T: 前 T-1 个 token → 加载模型 → 前向传播 → 采样 → 第 T 个 token

问题: 每步加载 100B+ 参数的模型权重, 却只产出 1 个 token!
      GPU 算力严重利用不足 (memory-bound)!
```

### 2. 内存带宽瓶颈

自回归解码是典型的 **内存带宽受限（Memory-bound）** 操作。以一个 175B 参数的模型为例：

| 指标 | 数值 |
|------|------|
| 模型权重 (fp16) | 350 GB |
| A100 HBM 带宽 | 2 TB/s |
| 读取所有权重的时间 | 350 / 2000 = **175 ms** |
| 实际计算时间（单 token） | < 1 ms |
| **GPU 计算利用率** | **< 1%** |

每生成一个 token，GPU 花 99% 的时间在"搬数据"，只花 1% 的时间在"算"。增加 batch size 可以提高利用率（更多 token 共享同一次权重加载），但在 **低延迟场景**（单请求）中 batch size = 1 是常态。

### 3. 现有加速方法的局限

| 方法 | 原理 | 局限 |
|------|------|------|
| 量化 (INT8/INT4) | 减少权重大小 → 减少内存读取 | 有精度损失 |
| KV Cache | 避免重复计算历史 token 的 KV | 已是标准做法，无法进一步加速 |
| 模型蒸馏 | 用小模型替代大模型 | 质量下降 |
| 并行计算 | 更多 GPU | 成本高，单请求延迟受通信限制 |

所有这些方法要么 **有损**（精度/质量下降），要么 **受限于硬件成本**。有没有一种方法能 **不改变输出质量** 地加速推理？

### 4. Speculative Decoding 的核心洞察

论文的核心洞察来自一个简单的观察：

> **LLM 生成的大部分 token 其实是"容易的"——一个小得多的模型就能正确预测。只有少数"难"的 token 才真正需要大模型。**

例如，生成 "The capital of France is Paris." 时：
- "The"、"of"、"is"、"." 等 token 非常 trivial，几乎任何语言模型都能预测
- "capital"、"France"、"Paris" 可能需要更强的模型，但也不一定需要 175B 参数

**关键思路**：让一个小模型（如 7B）先快速"猜"若干 token，然后让大模型（如 175B）**一次性并行验证** 这些猜测。如果猜对了，就等于跳过了多步串行解码；如果猜错了，用大模型的正确分布修正。

```
传统自回归解码 vs Speculative Decoding:

传统 (每步 1 个 token):
  大模型: [step1] → [step2] → [step3] → [step4] → [step5]
  产出:      t1       t2       t3       t4       t5
  时间:   ████████████████████████████████████████████████

Speculative Decoding (猜 γ=4 个):
  小模型: [guess t1,t2,t3,t4]  ← 很快! 小模型便宜
  大模型: [verify t1,t2,t3,t4 + sample t5]  ← 1 次并行前向传播
  产出:      t1  t2  t3  t4  t5  (假设全部接受)
  时间:   ██████████████

  如果 t3 被拒绝:
  产出:      t1  t2  t3'  (从大模型分布重新采样)
  时间:   ██████████████   (仍然比传统快! 3 个 token / 1 次大模型调用)
```

### 5. 论文的主要贡献

1. **Speculative Decoding 算法**：形式化定义了"猜测-验证"框架，保证输出分布与原始大模型完全一致

2. **修正的拒绝采样方案**：设计了一种特殊的拒绝采样策略，当小模型的猜测被拒绝时，使用修正后的分布采样新 token——确保最终输出的 **每一个 token** 都严格来自大模型的条件分布

3. **加速比的理论分析**：推导了期望加速比的闭式表达，取决于小模型与大模型的分布 "匹配度"

4. **实验验证**：在多个任务上实现了 2-3 倍的无损加速

---

## 核心算法：猜测与验证

### 算法概览

Speculative Decoding 的每一轮由三步组成：

1. **Draft（猜测）**：用小模型 \(M_q\) 自回归生成 \(\gamma\) 个候选 token
2. **Verify（验证）**：用大模型 \(M_p\) **一次并行前向传播** 计算这 \(\gamma\) 个位置的概率分布
3. **Accept/Reject（接受/拒绝）**：逐个检查候选 token 是否"足够好"，接受的保留，拒绝的用大模型分布重新采样

### 形式化定义

设：
- \(M_p\)：目标大模型，条件分布为 \(p(x \mid x_1, \ldots, x_t)\)
- \(M_q\)：草稿小模型，条件分布为 \(q(x \mid x_1, \ldots, x_t)\)
- \(\gamma\)：每轮猜测的 token 数（猜测窗口大小）
- \(x_1, \ldots, x_n\)：已经生成的 prefix

**Step 1：Draft**

用 \(M_q\) 自回归生成 \(\gamma\) 个候选 token：

$$
\tilde{x}_{n+1} \sim q(\cdot \mid x_1, \ldots, x_n)
$$
$$
\tilde{x}_{n+2} \sim q(\cdot \mid x_1, \ldots, x_n, \tilde{x}_{n+1})
$$
$$
\vdots
$$
$$
\tilde{x}_{n+\gamma} \sim q(\cdot \mid x_1, \ldots, x_n, \tilde{x}_{n+1}, \ldots, \tilde{x}_{n+\gamma-1})
$$

同时保存每步的概率分布 \(q_1, q_2, \ldots, q_\gamma\)。

**Step 2：Verify**

用 \(M_p\) 对 prefix + 全部 \(\gamma\) 个候选 token **一次性** 做前向传播：

$$
p_1, p_2, \ldots, p_{\gamma+1} = M_p(x_1, \ldots, x_n, \tilde{x}_{n+1}, \ldots, \tilde{x}_{n+\gamma})
$$

注意 \(M_p\) 输出 \(\gamma + 1\) 个位置的分布——\(\gamma\) 个验证位置 + 1 个新位置（如果全部接受，可以额外得到一个 token）。

::: tip 为什么大模型可以并行验证？
Transformer 的前向传播天然支持并行处理多个 token（就像处理一个 batch 的 prompt）。验证 \(\gamma\) 个候选 token 的计算量与处理一个长度为 \(\gamma\) 的 prompt 相同——只需要 **一次** 前向传播，而不是 \(\gamma\) 次。
:::

**Step 3：Accept/Reject**

对每个候选 token \(\tilde{x}_{n+i}\)（\(i = 1, \ldots, \gamma\)），按以下规则决定是否接受：

$$
\text{接受概率} = \min\!\left(1, \frac{p_i(\tilde{x}_{n+i})}{q_i(\tilde{x}_{n+i})}\right)
$$

具体来说：
1. 生成随机数 \(r \sim \text{Uniform}(0, 1)\)
2. 如果 \(r < \min\!\left(1, \frac{p_i(\tilde{x}_{n+i})}{q_i(\tilde{x}_{n+i})}\right)\)，**接受** \(\tilde{x}_{n+i}\)，继续验证下一个
3. 否则，**拒绝** \(\tilde{x}_{n+i}\)，从修正分布采样替代 token，**停止本轮验证**

### 完整算法流程

```
Algorithm: Speculative Decoding
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
输入: 目标模型 M_p, 草稿模型 M_q, 猜测窗口 γ, 已有前缀 x₁...xₙ

循环:
  ┌─ Step 1: Draft (小模型猜 γ 个 token)
  │   for i = 1 to γ:
  │     x̃_{n+i} ~ q(· | x₁,...,xₙ, x̃_{n+1},...,x̃_{n+i-1})
  │     保存 q_i 分布
  │
  ├─ Step 2: Verify (大模型并行检查)
  │   p₁,...,p_{γ+1} = M_p(x₁,...,xₙ, x̃_{n+1},...,x̃_{n+γ})
  │
  └─ Step 3: Accept / Reject
      accepted = 0
      for i = 1 to γ:
        r ~ Uniform(0, 1)
        if r < min(1, p_i(x̃_{n+i}) / q_i(x̃_{n+i})):
          接受 x̃_{n+i}
          accepted += 1
        else:
          从修正分布采样: x_{n+i} ~ norm(max(0, p_i - q_i))
          break
      if 全部接受 (accepted == γ):
        额外采样: x_{n+γ+1} ~ p_{γ+1}  ← 白赚一个 token!

  本轮产出: accepted + 1 个 token (至少 1 个, 至多 γ+1 个)
  更新 n, 进入下一轮
```

---

## 数学保证：输出分布完全一致

这是整篇论文最核心的理论贡献——证明 Speculative Decoding 的输出与直接从大模型采样 **分布完全一致**。

### 拒绝采样的正确性

对于候选 token \(\tilde{x}\)，接受概率为 \(\min(1, p(\tilde{x})/q(\tilde{x}))\)。被拒绝时，从 **修正分布** 采样：

$$
x \sim \text{norm}\!\left(\max(0,\; p(x) - q(x))\right)
$$

其中 \(\text{norm}(\cdot)\) 表示归一化为概率分布。

**定理（论文 Theorem 1）**：通过上述拒绝采样方案采样的 token，其边际分布恰好等于 \(p(x)\)。

### 直觉理解

考虑词表中的某个 token \(x\)：

**Case 1：\(q(x) \leq p(x)\)**（小模型低估了这个 token 的概率）
- 如果小模型采到了 \(x\)，接受概率 = \(\min(1, p(x)/q(x)) = 1\)，必然接受
- 来自 \(q\) 的贡献 = \(q(x) \cdot 1 = q(x)\)
- 还需要来自修正分布的贡献 = \(p(x) - q(x)\)（由拒绝后采样补充）

**Case 2：\(q(x) > p(x)\)**（小模型高估了这个 token 的概率）
- 如果小模型采到了 \(x\)，接受概率 = \(p(x)/q(x) < 1\)，部分拒绝
- 来自 \(q\) 的贡献 = \(q(x) \cdot p(x)/q(x) = p(x)\)
- 修正分布中 \(\max(0, p(x) - q(x)) = 0\)，不额外贡献

两种情况合计：token \(x\) 被最终采样的概率恰好 = \(p(x)\)。

```
拒绝采样的直觉 (某个 token x):

  概率
  ↑
  │
p(x)─┤ ┌───────┐
  │   │ 接受    │ ← q(x) ≤ p(x) 时, 100% 接受
  │   │ (来自q) │
q(x)─┤ ├───────┤
  │   │ 修正    │ ← 拒绝后从修正分布补充 p(x)-q(x)
  │   │ 分布补充│
  0───┴─┴───────┴──→

  概率
  ↑
  │
q(x)─┤ ┌───────┐
  │   │ 部分   │
  │   │ 拒绝   │ ← q(x) > p(x) 时, 以 p(x)/q(x) 的概率接受
p(x)─┤ ├───────┤
  │   │ 接受   │ ← 接受的贡献 = q(x) · p(x)/q(x) = p(x) ✓
  │   │ (来自q)│
  0───┴─┴───────┴──→

两种情况下, 最终采样概率都恰好 = p(x)!
```

### 修正分布的推导

当 token 被拒绝时，我们需要从 \(\text{norm}(\max(0, p - q))\) 中采样。这个分布的直觉是：

$$
p'(x) = \frac{\max(0, p(x) - q(x))}{\sum_x \max(0, p(x) - q(x))}
$$

它只对 \(p(x) > q(x)\) 的 token 有非零概率——即大模型认为比小模型更可能的 token。这些正是小模型"遗漏"的部分。

归一化常数为：

$$
Z = \sum_x \max(0, p(x) - q(x)) = 1 - \sum_x \min(p(x), q(x))
$$

而拒绝发生的概率恰好也是：

$$
P(\text{reject}) = 1 - \sum_x q(x) \cdot \min\!\left(1, \frac{p(x)}{q(x)}\right) = 1 - \sum_x \min(p(x), q(x)) = Z
$$

所以修正分布的贡献 \(Z \cdot p'(x) = \max(0, p(x) - q(x))\)，加上接受的贡献 \(\min(p(x), q(x))\)，总计 = \(p(x)\)。

::: tip 完美的数学构造
接受贡献 + 拒绝后修正贡献 = 目标分布。这不是近似，是 **精确等式**。这就是为什么 Speculative Decoding 是"无损"的——输出分布没有任何偏差。
:::

---

## 加速比的理论分析

### 期望接受长度

每轮最多猜 \(\gamma\) 个 token。设小模型与大模型在位置 \(i\) 的分布匹配度为：

$$
\alpha_i = \sum_x \min(p_i(x), q_i(x)) = 1 - D_{\text{TV}}(p_i, q_i)
$$

其中 \(D_{\text{TV}}\) 是总变差距离（Total Variation Distance）。\(\alpha_i\) 越接近 1，表示两个分布越相似。

假设各位置的 \(\alpha_i\) 近似相同，记为 \(\alpha\)。则期望接受的 token 数为：

$$
E[\text{accepted}] = \frac{1 - \alpha^{\gamma+1}}{1 - \alpha} - 1
$$

每轮总产出（包括拒绝后采样或额外采样的 1 个 token）：

$$
E[\text{tokens per round}] = \frac{1 - \alpha^{\gamma+1}}{1 - \alpha}
$$

### 期望加速比

设大模型一次前向传播耗时 \(T_p\)，小模型一次前向传播耗时 \(T_q\)。每轮需要：
- 小模型：\(\gamma\) 次前向传播 → 耗时 \(\gamma T_q\)
- 大模型：1 次前向传播（验证 \(\gamma\) 个 token）→ 耗时 \(T_p\)

加速比：

$$
\text{Speedup} = \frac{E[\text{tokens per round}]}{\gamma T_q + T_p} \cdot T_p = \frac{1 - \alpha^{\gamma+1}}{(1 - \alpha)(\gamma c + 1)}
$$

其中 \(c = T_q / T_p\) 是小模型与大模型的速度比。

### 最优猜测窗口 \(\gamma^*\)

给定 \(\alpha\) 和 \(c\)，可以求解最优的 \(\gamma\)。论文给出近似解：

$$
\gamma^* \approx \frac{\log c}{\log \alpha}
$$

直觉：
- 当 \(\alpha\) 很大（分布很相似）→ \(\gamma^*\) 大（可以猜更多，大多数会被接受）
- 当 \(c\) 很小（小模型很快）→ \(\gamma^*\) 大（猜测成本低，多猜几个划算）

下面的代码可以计算不同配置下的理论加速比：

```cpp-run title="Speculative Decoding 加速比理论分析"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <algorithm>

// 期望每轮产出 token 数
double expected_tokens(double alpha, int gamma) {
    if (std::abs(alpha - 1.0) < 1e-10) return gamma + 1.0;
    return (1.0 - std::pow(alpha, gamma + 1)) / (1.0 - alpha);
}

// 加速比
double speedup(double alpha, int gamma, double c) {
    double tokens = expected_tokens(alpha, gamma);
    double cost = gamma * c + 1.0;  // γ 次小模型 + 1 次大模型
    return tokens / cost;
}

// 最优 γ (暴力搜索)
int optimal_gamma(double alpha, double c, int max_gamma = 40) {
    int best_g = 1;
    double best_s = 0;
    for (int g = 1; g <= max_gamma; g++) {
        double s = speedup(alpha, g, c);
        if (s > best_s) { best_s = s; best_g = g; }
    }
    return best_g;
}

int main() {
    std::cout << "=============================================================\n";
    std::cout << "    Speculative Decoding 加速比理论分析\n";
    std::cout << "=============================================================\n\n";

    // ---- 1. 不同 α 和 γ 下的期望产出 ----
    std::cout << "1. 每轮期望产出 token 数 (不同 α 和 γ):\n\n";
    std::cout << std::setw(8) << "α \\ γ";
    for (int g : {1, 2, 3, 4, 5, 8, 10}) std::cout << std::setw(8) << g;
    std::cout << "\n" << std::string(64, '-') << "\n";

    for (double a : {0.5, 0.6, 0.7, 0.8, 0.9, 0.95}) {
        std::cout << std::setw(8) << std::fixed << std::setprecision(2) << a;
        for (int g : {1, 2, 3, 4, 5, 8, 10}) {
            std::cout << std::setw(8) << std::setprecision(2) << expected_tokens(a, g);
        }
        std::cout << "\n";
    }

    std::cout << "\n  α=0.9, γ=5: 平均每轮产出 "
              << std::setprecision(1) << expected_tokens(0.9, 5) << " 个 token\n\n";

    // ---- 2. 不同配置下的加速比 ----
    std::cout << "=============================================================\n";
    std::cout << "2. 加速比 (c = T_q/T_p = 小模型/大模型 速度比)\n";
    std::cout << "=============================================================\n\n";

    // 典型配置: 大模型 175B, 小模型 7B → c ≈ 0.05
    // 大模型 70B, 小模型 7B → c ≈ 0.1
    // 大模型 13B, 小模型 1B → c ≈ 0.08

    std::cout << "  固定 c=0.05 (小模型速度是大模型的 1/20):\n\n";
    std::cout << std::setw(8) << "α \\ γ";
    for (int g : {1, 2, 3, 4, 5, 8, 10}) std::cout << std::setw(8) << g;
    std::cout << std::setw(10) << "最优γ" << std::setw(10) << "最优加速";
    std::cout << "\n" << std::string(84, '-') << "\n";

    double c = 0.05;
    for (double a : {0.5, 0.6, 0.7, 0.8, 0.9, 0.95}) {
        std::cout << std::setw(8) << std::fixed << std::setprecision(2) << a;
        for (int g : {1, 2, 3, 4, 5, 8, 10}) {
            std::cout << std::setw(7) << std::setprecision(2) << speedup(a, g, c) << "x";
        }
        int og = optimal_gamma(a, c);
        std::cout << std::setw(8) << og;
        std::cout << std::setw(8) << std::setprecision(2) << speedup(a, og, c) << "x";
        std::cout << "\n";
    }

    std::cout << "\n";

    // ---- 3. 不同 c 值的对比 ----
    std::cout << "=============================================================\n";
    std::cout << "3. 不同小模型大小对加速比的影响 (α=0.8):\n";
    std::cout << "=============================================================\n\n";

    double alpha = 0.8;
    std::cout << std::setw(8) << "c"
              << std::setw(24) << "场景"
              << std::setw(10) << "最优γ"
              << std::setw(12) << "加速比" << "\n";
    std::cout << std::string(54, '-') << "\n";

    struct Config { double c; const char* desc; };
    Config configs[] = {
        {0.02, "7B / 175B"},
        {0.05, "7B / 70B"},
        {0.10, "7B / 30B"},
        {0.15, "7B / 13B"},
        {0.25, "1B / 7B"},
        {0.50, "小/中模型"},
    };

    for (auto& cfg : configs) {
        int og = optimal_gamma(alpha, cfg.c);
        double s = speedup(alpha, og, cfg.c);
        std::cout << std::setw(8) << std::fixed << std::setprecision(2) << cfg.c
                  << std::setw(24) << cfg.desc
                  << std::setw(10) << og
                  << std::setw(10) << std::setprecision(2) << s << "x\n";
    }

    std::cout << "\n  结论:\n";
    std::cout << "  - c 越小 (小模型相对越快), 加速比越高\n";
    std::cout << "  - α 越大 (两模型越相似), 加速比越高\n";
    std::cout << "  - 最优 γ 随 α 增大和 c 减小而增大\n\n";

    // ---- 4. 接受率 α 对实际生成的影响 ----
    std::cout << "=============================================================\n";
    std::cout << "4. 生成 100 个 token 的预期大模型调用次数:\n";
    std::cout << "=============================================================\n\n";

    int total_tokens = 100;
    std::cout << std::setw(8) << "α"
              << std::setw(8) << "γ"
              << std::setw(16) << "每轮产出"
              << std::setw(14) << "总轮数"
              << std::setw(18) << "大模型调用数"
              << std::setw(14) << "vs 传统" << "\n";
    std::cout << std::string(78, '-') << "\n";

    for (double a : {0.7, 0.8, 0.9, 0.95}) {
        int g = optimal_gamma(a, 0.05);
        double tpr = expected_tokens(a, g);
        double rounds = total_tokens / tpr;
        double target_calls = rounds;  // 每轮 1 次大模型调用

        std::cout << std::setw(8) << std::fixed << std::setprecision(2) << a
                  << std::setw(8) << g
                  << std::setw(12) << std::setprecision(1) << tpr << " tok"
                  << std::setw(14) << std::setprecision(1) << rounds
                  << std::setw(14) << std::setprecision(0) << target_calls << " 次"
                  << std::setw(12) << std::setprecision(1) << total_tokens / target_calls << "x\n";
    }

    std::cout << "\n  传统方法: 100 个 token = 100 次大模型调用\n";
    std::cout << "  α=0.9, γ=5: 约 " << std::setprecision(0)
              << 100.0 / expected_tokens(0.9, 5) << " 次大模型调用 → "
              << std::setprecision(1) << expected_tokens(0.9, 5) << "x 加速!\n";

    return 0;
}
```

---

## 接受率 \(\alpha\) 的直觉

### 什么决定了 \(\alpha\)？

\(\alpha = \sum_x \min(p(x), q(x))\) 是大模型和小模型分布的重叠面积。影响 \(\alpha\) 的因素：

**1. 小模型的质量**

更好的小模型 → 分布更接近大模型 → \(\alpha\) 更大。但小模型不能太大（否则猜测本身就变慢了）。

**2. Token 的"难度"**

- **简单 token**（如 "the"、"of"、","）：大模型和小模型几乎一致 → \(\alpha \approx 1\)
- **困难 token**（如专业术语、稀有词）：分布可能差异较大 → \(\alpha\) 较小

**3. 采样温度**

- **低温度（贪婪解码）**：两个模型都倾向于输出 top-1 token → 如果 top-1 一致，\(\alpha \approx 1\)
- **高温度**：分布更平坦，重叠面积通常更大 → \(\alpha\) 可能更大

### 实际场景中的典型 \(\alpha\)

| 大模型 | 小模型 | 任务 | 典型 \(\alpha\) |
|--------|--------|------|----------------|
| 175B | 7B | 英语续写 | 0.7 - 0.9 |
| 70B | 7B | 代码生成 | 0.6 - 0.8 |
| 13B | 1B | 翻译 | 0.5 - 0.7 |

---

## 工程实现要点

### 1. KV Cache 的管理

Speculative Decoding 需要同时管理 **两个模型的 KV Cache**：

```
KV Cache 管理:

小模型 KV Cache:
  [x₁...xₙ | x̃_{n+1}...x̃_{n+γ}]  ← 包含猜测的 γ 个 token

大模型 KV Cache:
  [x₁...xₙ | x̃_{n+1}...x̃_{n+γ}]  ← 验证时也缓存了 γ 个 token

如果在位置 n+k 拒绝了:
  小模型 Cache: 回滚到 [x₁...xₙ₊ₖ₋₁]  ← 丢弃 k 之后的缓存
  大模型 Cache: 回滚到 [x₁...xₙ₊ₖ₋₁]  ← 同样回滚
  然后用修正分布采样的 token 更新 Cache
```

KV Cache 的回滚是一个 **截断操作**（只需修改长度计数器），实际开销很小。

### 2. 大模型验证的并行性

关键实现细节：大模型验证 \(\gamma\) 个 token 时，这 \(\gamma\) 个 token 像 prompt 一样 **并行处理**。这意味着：
- 如果已有 KV Cache 到位置 \(n\)，验证只需处理 \(\gamma\) 个新 token
- 计算量约等于处理一个长度为 \(\gamma\) 的 prompt（而非 \(\gamma\) 次自回归）
- 在 batch size = 1 的场景下，由于自回归是内存带宽受限的，处理 1 个 token 和 \(\gamma\) 个 token 的耗时几乎一样（权重加载是主要开销）

### 3. 小模型的选择

| 选择策略 | 优点 | 缺点 |
|---------|------|------|
| 同系列小模型（如 LLaMA-7B + LLaMA-70B） | \(\alpha\) 高（同分布族） | 需要额外部署一个模型 |
| 同模型的量化版本 | 部署简单 | \(\alpha\) 取决于量化精度 |
| N-gram 模型 | 极快，几乎零开销 | \(\alpha\) 较低 |
| 浅层网络 / 退出 | 无需额外模型 | 实现复杂 |

### 4. 与现有优化的兼容性

Speculative Decoding 可以与其他推理优化 **组合使用**：

| 优化 | 兼容性 | 说明 |
|------|--------|------|
| KV Cache | ✓ | 两个模型都使用各自的 KV Cache |
| INT8 量化 | ✓ | 大模型/小模型都可以量化 |
| FlashAttention | ✓ | 验证时的注意力计算可以用 FlashAttention |
| Continuous Batching | ✓ | 不同请求可以独立 speculate |
| Tensor Parallelism | ✓ | 大模型可以使用 TP |

---

## 实验结果与关键发现

### 实验设置

论文在以下配置上评估：
- **大模型（Target）**：T5-XXL (11B), LaMDA (137B)
- **小模型（Draft）**：T5-Small (60M), LaMDA 的小版本
- **任务**：翻译（WMT En-De）、摘要（CNN/DailyMail）、对话（LaMDA 对话任务）
- **硬件**：TPU v4

### 关键结果

**1. 无损加速**

| 任务 | 大模型 | 加速比 | 输出质量 |
|------|--------|--------|---------|
| 英德翻译 | T5-XXL | **2.06x** | BLEU 完全一致 |
| 摘要 | T5-XXL | **1.92x** | ROUGE 完全一致 |
| 对话 | LaMDA 137B | **2.46x** | 完全一致 |

"完全一致" 意味着在相同随机种子下，Speculative Decoding 产生与传统自回归解码 **完全相同的 token 序列**（给定相同的随机数）。

**2. 不同猜测窗口 \(\gamma\) 的影响**

| \(\gamma\) | 翻译加速比 | 摘要加速比 | 对话加速比 |
|-----------|-----------|-----------|-----------|
| 1 | 1.35x | 1.28x | 1.52x |
| 2 | 1.64x | 1.55x | 1.89x |
| 4 | 1.93x | 1.82x | 2.31x |
| 5 | **2.06x** | **1.92x** | **2.46x** |
| 8 | 1.98x | 1.85x | 2.39x |

存在最优 \(\gamma^*\)（约 4-6），过大的 \(\gamma\) 反而降速——因为后面的猜测更可能被拒绝，白白浪费了小模型的计算。

**3. 不同采样温度的影响**

| 温度 \(T\) | 接受率 \(\alpha\) | 加速比 |
|-----------|------------------|--------|
| 0 (贪婪) | ~0.92 | **2.5x+** |
| 0.3 | ~0.88 | 2.3x |
| 0.7 | ~0.82 | 2.0x |
| 1.0 | ~0.75 | 1.8x |

温度越低，加速效果越好——因为贪婪解码时两个模型更容易 "同意" 同一个 top-1 token。

---

## 与后续工作的关系

Speculative Decoding 催生了大量后续研究：

### SpecInfer / Medusa / EAGLE

这些工作尝试 **不用独立的小模型**，而是用更高效的方式生成候选 token：

- **Medusa**：在大模型的倒数第二层加上多个轻量解码头，每个头负责预测不同偏移量的 token
- **EAGLE**：用大模型的最后一层隐状态 + 轻量网络预测未来 token 的特征
- **SpecInfer**：用多个小模型组成 **猜测树（Speculative Tree）**，同时探索多个候选序列

### Staged Speculative Decoding

引入 **多级猜测**：极小模型猜测 → 小模型验证/扩展 → 大模型最终验证。更多级可以进一步提高加速比。

### 与 Lookahead Decoding 的区别

Lookahead Decoding（Jacobi 迭代解码）用 **同一个模型** 做并行猜测，不需要小模型。但它缺少 Speculative Decoding 的精确分布保证。

---

## 总结与启示

### 核心贡献

1. **无损加速的范式**：证明了可以在 **不改变任何输出分布** 的前提下加速自回归解码——这在之前被认为是不可能的

2. **优雅的数学保证**：拒绝采样 + 修正分布的组合保证了精确的分布等价性，不是近似，是严格证明

3. **实用性**：2-3 倍的加速在不需要任何额外训练的情况下实现，直接适用于现有模型

### 设计哲学

Speculative Decoding 的核心思想可以概括为：

> **用便宜的资源（小模型）做猜测，用昂贵的资源（大模型）做验证。验证比生成快（可以并行），猜测比验证便宜（小模型轻量）。**

这是一种 **投机执行（Speculative Execution）** 的思想，源自计算机体系结构中的经典技术：

| 领域 | 投机执行 | 验证 |
|------|---------|------|
| CPU 分支预测 | 猜测分支方向，提前执行 | 分支结果出来后验证 |
| 数据库乐观锁 | 假设无冲突，先执行 | 提交时检查冲突 |
| **Speculative Decoding** | **小模型猜测 token** | **大模型验证分布** |

::: tip 与 FlashAttention 和 ZeRO 的统一视角
这三个系列的工作解决了 LLM 不同阶段的效率问题，但共享相同的哲学——**利用问题的结构特性找到更优的执行方式**：

- **FlashAttention**：利用注意力计算的分块可分解性 → 减少内存搬运
- **ZeRO**：利用数据并行中的状态冗余性 → 减少内存浪费
- **Speculative Decoding**：利用 token 生成的难度不均匀性 → 减少大模型调用次数

每一个突破都来自对 "什么是真正的瓶颈" 的深刻理解。
:::

---

## 参考文献

1. Leviathan, Y., Kalman, M., & Matias, Y. (2023). **Fast Inference from Transformers via Speculative Decoding**. ICML 2023. [arXiv:2211.17192](https://arxiv.org/abs/2211.17192)

2. Chen, C., Borgeaud, S., Irving, G., Lespiau, J.B., Sifre, L., & Jumper, J. (2023). **Accelerating Large Language Model Decoding with Speculative Sampling**. [arXiv:2302.01318](https://arxiv.org/abs/2302.01318)

3. Cai, T., Li, Y., Geng, Z., Peng, H., Lee, J.D., Chen, D., & Dao, T. (2024). **Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads**. ICML 2024. [arXiv:2401.10774](https://arxiv.org/abs/2401.10774)

4. Li, Y., Cai, T., Zhang, Y., Chen, D., & Dao, T. (2024). **EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty**. ICML 2024. [arXiv:2401.15077](https://arxiv.org/abs/2401.15077)

5. Miao, X., Oliaro, G., Zhang, Z., Cheng, X., Wang, Z., Wong, R.Y.Y., ... & Jia, Z. (2023). **SpecInfer: Accelerating Generative Large Language Model Serving with Speculative Inference and Token Tree Verification**. [arXiv:2305.09781](https://arxiv.org/abs/2305.09781)

6. Fu, Y., Bailis, P., Stoica, I., & Zhang, H. (2024). **Break the Sequential Dependency of LLM Inference Using Lookahead Decoding**. [arXiv:2402.02057](https://arxiv.org/abs/2402.02057)
