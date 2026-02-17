---
title: "DistServe: Prefill-Decoding 解耦的 Goodput 优化 LLM 服务"
date: 2026-02-15
---

# DistServe: Prefill-Decoding 解耦的 Goodput 优化 LLM 服务

<p style="color: var(--vp-c-text-2); font-size: 14px;">
2026-02-15 &nbsp;·&nbsp; 推理服务 &nbsp;·&nbsp; 论文精读
</p>

> **论文信息**
> - **作者**: Yinmin Zhong, Shengyu Liu, Junda Chen, Jianbo Hu, Yibo Zhu, Xuanzhe Liu, Xin Jin, Hao Zhang
> - **机构**: 北京大学 / UC San Diego / StepFun
> - **发表**: OSDI 2024
> - **链接**: [arXiv:2401.09670](https://arxiv.org/abs/2401.09670)

## 一句话总结

DistServe 提出将 LLM 推理的 **Prefill（预填充）** 和 **Decoding（解码）** 两个阶段 **解耦到不同 GPU** 上执行，消除两阶段间的相互干扰，并为每个阶段独立优化资源分配与并行策略，在满足 TTFT 和 TPOT 双重 SLO 约束下，最大化每 GPU 的 **Goodput**（有效吞吐），相比 SOTA 系统可服务 **7.4 倍** 更多请求或支持 **12.6 倍** 更严格的延迟约束。

---

## Introduction：为什么需要解耦 Prefill 和 Decoding？

### 1. LLM 推理的两阶段特性

LLM 推理是一个两阶段过程，且两个阶段的计算特征 **截然不同**：

```
LLM 推理的两阶段：

  用户输入 "What is the largest ocean?"
         │
         ▼
  ┌─── Prefill 阶段 ───┐
  │ 并行处理所有输入 token │    → 输出第一个 token "The"
  │ 计算密集 (compute-bound) │
  │ 延迟指标: TTFT         │    TTFT = Time To First Token
  └─────────┬─────────────┘
            ▼
  ┌─── Decoding 阶段 ──┐
  │ 逐个生成后续 token    │    → "Pacific" → "Ocean" → ...
  │ 访存密集 (memory-bound)│
  │ 延迟指标: TPOT         │    TPOT = Time Per Output Token
  └───────────────────────┘
```

| 特征 | Prefill | Decoding |
|------|---------|----------|
| 处理 token 数 | 多（整个 prompt） | 1（每步） |
| 计算特征 | **计算密集** | **访存密集** |
| GPU 利用率 | 高（矩阵乘法饱和） | 低（GEMV，带宽受限） |
| 延迟指标 | TTFT（首 token 延迟） | TPOT（每 token 延迟） |
| 对并行的偏好 | 倾向 **张量并行**（降低执行时间） | 倾向 **流水线并行**（提升吞吐） |
| batch 策略 | 小 batch（已计算饱和） | 大 batch（提升 GPU 利用率） |

### 2. 共置（Colocation）的三大问题

现有系统（如 vLLM、TensorRT-LLM）将两个阶段放在 **同一组 GPU** 上，通过 continuous batching 将 prefill 和 decoding 请求混合执行。这带来严重问题：

**问题 1：Prefill-Decoding 互相干扰**

```
Continuous Batching 的干扰：

  时间线：
  ────────────────────────────────────────────────
  正常 decode batch (4个请求):
  [decode₁][decode₂][decode₃][decode₄]  每步 ~5ms
  
  混入一个 prefill 请求后:
  [prefill(512 tokens) + decode₁₋₄]     这一步 ~40ms!
  
  影响：
  ├─ decode 的 TPOT: 5ms → 40ms  (8x 退化！)
  └─ prefill 的 TTFT: 也被 decode 拖慢 ~15-30%
```

论文测量显示，在 13B 模型、输入长度 512 的场景下：
- 将一个 prefill 请求加入 decode batch → decode 延迟增加 **3-8 倍**
- decode 请求也使 prefill 延迟增加 **15-30%**

**问题 2：资源和并行策略耦合**

```
Prefill 和 Decoding 想要不同的并行策略：

  Prefill（计算密集）：
    ✓ 张量并行 (TP) — 切分矩阵乘到多 GPU → 降低执行时间
    ✗ 流水线并行 (PP) — 增加气泡，不利于单请求延迟
    
  Decoding（访存密集）：
    ✓ 流水线并行 (PP) — 线性扩容吞吐 → 服务更多请求
    ✗ 张量并行 (TP) — 通信开销大于计算节省
    
  共置时两者被迫使用同一套并行策略 → 总有一方受损
```

**问题 3：调度困难**

即使不混合 batch，将 prefill 和 decode 在同一 GPU 上交替执行，也面临困境：

```
排队冲突（共置时）：

  Prefill 请求等待 decode batch 完成:
  [decode][decode][decode] → [prefill] → TTFT 被排队延迟拉长
  
  Decode 请求等待 prefill 完成:
  [prefill (长序列)] → [decode] → TPOT 被排队延迟拉长
  
  优先调度某一方 → 另一方 SLO 被牺牲
  唯一选择: 过度配置 GPU → 成本浪费
```

### 3. 核心思路：解耦到不同 GPU

```
DistServe 的核心思想：

  传统系统（共置）：
  ┌────────── GPU Group ──────────┐
  │  Prefill + Decoding 混合执行    │
  │  共享权重、共享 KV Cache         │
  │  共享并行策略、互相干扰          │
  └───────────────────────────────┘

  DistServe（解耦）：
  ┌─── Prefill GPU(s) ───┐     KV Cache 传输     ┌─── Decoding GPU(s) ──┐
  │ 只做 Prefill           │  ═══════════════►   │ 只做 Decoding          │
  │ 独立资源分配            │                      │ 独立资源分配            │
  │ 独立并行策略 (偏好 TP)  │                      │ 独立并行策略 (偏好 PP)  │
  │ 优化 TTFT              │                      │ 优化 TPOT              │
  └────────────────────────┘                      └────────────────────────┘
```

---

## 核心方法一：两阶段的独立分析

### 1. Prefill 实例分析

Prefill 阶段的特点是 **计算密集**。对于 13B 模型，单个 512 token 的序列就能让 A100 接近计算饱和。

**Batching 策略**：一旦 GPU 计算饱和，增加 batch 只会线性增加延迟，不提升效率。因此 prefill 实例通常使用 **小 batch 甚至 batch=1**。

**排队模型**：Prefill 实例可以建模为 **M/D/1 队列**（泊松到达、确定性服务时间、单服务器）：

$$
\text{Avg\_TTFT} = D + \frac{RD^2}{2(1 - RD)}
$$

其中 \(D\) 是单个 prefill 的执行时间，\(R\) 是到达率。第一项是执行时间，第二项是排队延迟。

**并行策略分析**：

使用 2-way **张量并行（TP）**：执行时间变为 \(D/K\)（\(K \approx 1.5\)，由于通信开销不完美加速），但容量不变：

$$
\text{Avg\_TTFT}_{\text{intra}} = \frac{D}{K} + \frac{R(D/K)^2}{2(1 - RD/K)}
$$

使用 2-way **流水线并行（PP）**：执行时间约不变（\(\approx D\)），但容量翻倍（吞吐率上限从 \(1/D\) 变为 \(2/D\)）：

$$
\text{Avg\_TTFT}_{\text{inter}} = D + \frac{RD^2}{4(2 - RD)}
$$

```
Prefill 的并行策略选择：

  Avg TTFT
    │
    │  TP (张量并行)
    │  ╲
    │    ╲           PP (流水线并行)
    │      ╲       ╱
    │        ╲   ╱
    │          ✕  ← 交叉点
    │        ╱   ╲
    │      ╱       ╲
    │    ╱           ╲
    └─────────────────────► 到达率 R
         低负载        高负载
    
  低负载: TP 更好（降低执行时间，排队少）
  高负载: PP 更好（翻倍容量，减少排队）
```

### 2. Decoding 实例分析

Decoding 阶段的特点是 **访存密集**、每步只处理一个新 token。

**Batching 策略**：由于单个 decode 步骤 GPU 利用率极低，增大 batch 可以 **近乎免费地** 提升吞吐（直到 GPU 饱和）。

```
Decoding 的 batch 效应：

  吞吐 (tokens/s)
    │           ╭──── 饱和 ─────
    │         ╱
    │       ╱
    │     ╱      ← 几乎线性增长
    │   ╱
    │ ╱
    └───────────────────────► Batch Size
    1    8    16   32   64
    
  batch=1:  GPU 利用率 ~5%
  batch=32: GPU 利用率 ~80%   
  batch 增大时延迟几乎不增加！
```

**排队模型**：Decoding 实例可以建模为 **M/G/1 队列**，因为不同请求的生成长度不同（服务时间随机）。batch 的存在使得分析更复杂，但核心结论是：

- 更大的 batch → 更高的吞吐
- 流水线并行（PP）是首选：线性扩容吞吐，且通信开销可与计算重叠

### 3. 两阶段的最优并行策略总结

```
场景：66B 模型，8 GPU A100

共置系统（如 vLLM）：
  8-way TP 或 4-way TP + 2-way PP
  → 两阶段被迫共享同一配置
  → Goodput: ~1.6 rps/GPU

DistServe 解耦后：
  Prefill: 2-way TP × 2 实例     (4 GPUs)
  Decoding: 4-way PP × 1 实例    (4 GPUs)
  → 各自使用最优配置
  → Goodput: ~3.3 rps/GPU (2.1x 提升！)
```

---

## 核心方法二：Goodput 优化的资源分配

### 1. 问题定义

给定：
- 模型规模和硬件配置
- TTFT 和 TPOT 的 SLO 约束（如 P90 TTFT < 200ms, P90 TPOT < 50ms）
- SLO 达成率目标（如 90%）

目标：找到最大化 **per-GPU goodput** 的配置：
- Prefill 实例数量和并行策略
- Decoding 实例数量和并行策略
- 两类实例的比例

$$
\text{Goodput} = \frac{\text{满足 SLO 的最大请求率}}{\text{总 GPU 数}}
$$

### 2. 搜索算法

DistServe 使用 **穷举搜索 + profiling** 的方式寻找最优配置：

```
资源分配搜索算法：

输入: 模型 M, GPU 型号, TTFT_SLO, TPOT_SLO, SLO_attainment
输出: 最优 (prefill_config, decode_config, allocation_ratio)

for each 候选并行策略 (TP_p, PP_p) for prefill:
  for each 候选并行策略 (TP_d, PP_d) for decoding:
    for each prefill:decode GPU 比例:
      1. Profile prefill 实例在 (TP_p, PP_p) 下的延迟曲线
      2. Profile decode 实例在 (TP_d, PP_d) 下的延迟曲线
      3. 找到满足 TTFT_SLO 的最大 prefill 请求率 R_p
      4. 找到满足 TPOT_SLO 的最大 decode 请求率 R_d
      5. 系统 goodput = min(R_p, R_d) × (prefill比例 + decode比例)
      6. per_gpu_goodput = goodput / total_gpus
    记录最优配置

返回 per_gpu_goodput 最高的配置
```

### 3. 搜索空间与约束

```
搜索空间：

  Prefill 并行策略:
    TP ∈ {1, 2, 4, 8}
    PP ∈ {1, 2, 4}
    约束: TP × PP × n_prefill_instances ≤ total_gpus / 2
    
  Decoding 并行策略:
    TP ∈ {1, 2, 4}
    PP ∈ {1, 2, 4, 8}
    约束: TP × PP × n_decode_instances ≤ total_gpus / 2

  GPU 分配比例:
    prefill_gpus : decode_gpus 可以是非对称的
    例如: 4:4, 6:2, 2:6 等

  实际搜索空间不大（~几百种组合），每种配置通过 profiling 评估
```

---

## 核心方法三：KV Cache 传输与布局算法

### 1. 解耦的通信开销

解耦后，prefill 完成的请求需要将 **KV Cache** 从 prefill GPU 传输到 decoding GPU：

```
KV Cache 传输量：

  KV Cache 大小（per request）:
  = 2 × num_layers × 2 × num_kv_heads × head_dim × seq_len × sizeof(dtype)
    ↑                 ↑
    K + V        fp16 = 2 bytes
  
  例: LLaMA2-13B, 输入 512 tokens:
  = 2 × 40 × 2 × 40 × 128 × 512 × 2 bytes
  = 2 × 40 × 20,971,520 bytes
  ≈ 838 MB (使用 GQA, kv_heads = 40 → 实际更小)
  
  实际 LLaMA2-13B (GQA, kv_heads=40):
  = 2 × 40 × 40 × 128 × 512 × 2 = ~420 MB

  传输时间 (不同互联):
  ├─ NVLink (300 GB/s):  ~1.4 ms     ← 同节点
  ├─ PCIe 4.0 (32 GB/s): ~13 ms     ← 同节点跨socket
  └─ InfiniBand (25 GB/s): ~17 ms   ← 跨节点
```

### 2. 布局算法

KV Cache 的传输开销取决于 prefill 和 decoding 实例的 **物理位置**。DistServe 设计了布局算法来最小化通信开销：

```
布局策略：

策略 1: 同节点布局 (NVLink)
  ┌─────────── Node ───────────┐
  │  GPU₀  GPU₁   GPU₂  GPU₃  │
  │  ├─ Prefill ─┤ ├─ Decode ─┤│
  │       NVLink 互联            │
  └────────────────────────────┘
  优点: 传输快 (~1ms)
  缺点: 限制灵活分配
  适用: GPU 数量充足的大节点

策略 2: 跨节点布局 (InfiniBand)
  ┌── Node A ──┐     IB 网络     ┌── Node B ──┐
  │  Prefill   │ ═══════════► │  Decoding   │
  └────────────┘               └─────────────┘
  优点: 灵活分配
  缺点: 传输慢 (~17ms)
  适用: 当传输时间 < TPOT_SLO 时可接受

策略 3: 混合布局
  - 同节点的 GPU 优先用 NVLink
  - 跨节点的传输在 decode 步间异步进行
  - 利用 prefetch 隐藏传输延迟
```

### 3. 传输开销的可控性

论文的关键观察：**传输开销相比 SLO 通常很小**。

```
传输时间 vs 典型 SLO：

                 传输时间     TTFT SLO     TPOT SLO
  NVLink:        ~1 ms       200 ms       50 ms        占比 <2%
  InfiniBand:    ~17 ms      200 ms       50 ms        占比 ~8-34%
  
  且传输可以与 decoding 计算重叠:
  
  时间线 (decoding GPU):
  ────────────────────────────────────────────
  [接收 KV Cache]   [decode step 1] [decode step 2] ...
       ~1ms             ~5ms            ~5ms
  
  或者流水线化:
  [recv KV₁][decode₁ + recv KV₂][decode₂ + recv KV₃]...
```

---

## 核心方法四：调度与请求路由

### 1. 系统架构

```
DistServe 系统架构：

                 ┌─────────────────────┐
                 │     Controller      │
                 │  (请求路由 + 调度)   │
                 └──────┬──────────────┘
                        │
          ┌─────────────┼────────────────┐
          ▼             ▼                ▼
  ┌───────────┐  ┌───────────┐   ┌───────────┐
  │ Prefill   │  │ Prefill   │   │ Prefill   │
  │ Instance 0│  │ Instance 1│   │ Instance 2│
  │ (2-way TP)│  │ (2-way TP)│   │ (2-way TP)│
  └─────┬─────┘  └─────┬─────┘   └─────┬─────┘
        │               │               │
        │  KV Transfer  │               │
        ▼               ▼               ▼
  ┌─────────────────────────────────────────┐
  │         Decoding Instance 0             │
  │         (4-way PP)                      │
  │    持续 batch decoding 所有请求          │
  └─────────────────────────────────────────┘
```

### 2. 请求生命周期

```
一个请求在 DistServe 中的生命周期：

1. 请求到达 → Controller
2. Controller 将请求路由到负载最低的 Prefill Instance
3. Prefill Instance 执行预填充:
   - 加载模型权重
   - 并行处理所有输入 token
   - 生成 KV Cache + 第一个输出 token
   - 将第一个 token 流式返回客户端        ← TTFT 完成
4. KV Cache 传输到对应的 Decoding Instance
5. Decoding Instance 将请求加入 continuous batching:
   - 每步生成一个 token
   - 流式返回给客户端                     ← 每步贡献一个 TPOT
6. 生成终止 token → 请求完成，释放 KV Cache
```

---

## 量化分析：C++ 模拟

以下代码模拟 DistServe 的 Goodput 优化和解耦收益：

::: code-group
```cpp-run [DistServe Goodput 分析]
#include <iostream>
#include <iomanip>
#include <cmath>
#include <algorithm>
#include <vector>

// M/D/1 队列: 平均延迟 = D + R*D^2 / (2*(1-R*D))
double md1_avg_latency(double D, double R) {
    if (R * D >= 1.0) return 1e9;  // 不稳定
    return D + R * D * D / (2.0 * (1.0 - R * D));
}

// M/D/1 P90 近似 (基于指数分布近似排队时间)
double md1_p90_latency(double D, double R) {
    if (R * D >= 1.0) return 1e9;
    double avg_wait = R * D * D / (2.0 * (1.0 - R * D));
    // P90 约为平均排队时间的 2.3 倍 (指数分布)
    return D + avg_wait * 2.3;
}

// Prefill: 2-way 张量并行的延迟
double prefill_tp2_p90(double D, double R) {
    double K = 1.5;  // 2-way TP 加速比 (考虑通信)
    double D_new = D / K;
    return md1_p90_latency(D_new, R);
}

// Prefill: 2-way 流水线并行的延迟  
double prefill_pp2_p90(double D, double R) {
    // PP: 执行时间不变，但容量翻倍
    double D_m = D / 2.0;  // 每阶段延迟
    if (R * D_m >= 1.0) return 1e9;
    double D_s = D;  // 端到端延迟
    double avg_wait = R * D_m * D_m / (2.0 * (1.0 - R * D_m));
    return D_s + avg_wait * 2.3;
}

// Decode 吞吐: batch 增大近似线性提升
double decode_tpot(double base_tpot, int batch_size) {
    // batch 增大时单步延迟缓慢增长 (访存密集, batch 不显著增加计算)
    double overhead = 1.0 + 0.02 * (batch_size - 1);  // ~2% per batch item
    return base_tpot * overhead;
}

// 计算 decode 实例满足 TPOT SLO 时的最大请求率
double decode_max_rate(double base_tpot, double tpot_slo, int max_batch = 256) {
    for (int b = max_batch; b >= 1; b--) {
        double tpot = decode_tpot(base_tpot, b);
        if (tpot <= tpot_slo) {
            return (double)b / tpot * 1000.0;  // rps (tpot in ms)
        }
    }
    return 1.0 / base_tpot * 1000.0;
}

struct Config {
    int prefill_gpus;
    int decode_gpus;
    const char* prefill_parallel;
    const char* decode_parallel;
    double goodput;         // total rps
    double per_gpu_goodput; // rps / gpu
};

int main() {
    std::cout << "═══════════════════════════════════════════════════════════════\n";
    std::cout << "          DistServe: Prefill-Decoding 解耦分析\n";
    std::cout << "═══════════════════════════════════════════════════════════════\n\n";

    // ===== 分析1: Prefill-Decoding 干扰 =====
    std::cout << "【分析1】Prefill-Decoding 干扰量化 (13B 模型, A100)\n";
    std::cout << "─────────────────────────────────────────────────────────────\n\n";
    
    double prefill_time_512 = 35.0;   // ms, 单个 512 token prefill
    double decode_step_time = 5.0;     // ms, 单步 decode (batch=1)
    
    std::cout << "  将 1 个 prefill (input=512) 加入 decode batch 的影响:\n\n";
    std::cout << std::setw(14) << "Decode Batch"
              << std::setw(16) << "纯Decode(ms)"
              << std::setw(18) << "混合Batch(ms)"
              << std::setw(16) << "TPOT退化" << "\n";
    
    int decode_batches[] = {1, 4, 8, 16, 32};
    for (int b : decode_batches) {
        double pure_decode = decode_tpot(decode_step_time, b);
        // 混入 prefill 后: 整个 batch 时间由 prefill 主导
        double mixed = prefill_time_512 + pure_decode * 0.3;  // 额外增加
        double degradation = mixed / pure_decode;
        
        std::cout << std::setw(14) << b
                  << std::setw(14) << std::fixed << std::setprecision(1) << pure_decode << "ms"
                  << std::setw(16) << mixed << "ms"
                  << std::setw(14) << std::setprecision(1) << degradation << "x" << "\n";
    }

    // ===== 分析2: Prefill 并行策略对比 =====
    std::cout << "\n\n【分析2】Prefill 并行策略对比 (66B 模型, 2 GPU)\n";
    std::cout << "─────────────────────────────────────────────────────────────\n";
    std::cout << "  SLO: P90 TTFT < 500ms\n\n";
    
    double D_prefill = 150.0;  // ms, 66B 模型单 GPU prefill 时间
    
    std::cout << std::setw(12) << "到达率(rps)"
              << std::setw(15) << "无并行P90"
              << std::setw(14) << "2-TP P90"
              << std::setw(14) << "2-PP P90"
              << std::setw(12) << "最优策略" << "\n";
    
    double rates[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0};
    for (double R : rates) {
        double no_par = md1_p90_latency(D_prefill, R / 1000.0);
        double tp2 = prefill_tp2_p90(D_prefill, R / 1000.0);
        double pp2 = prefill_pp2_p90(D_prefill, R / 1000.0);
        
        const char* best = tp2 < pp2 ? "TP" : "PP";
        
        std::cout << std::setw(12) << std::fixed << std::setprecision(1) << R
                  << std::setw(13) << std::setprecision(0) << std::min(no_par, 99999.0) << "ms"
                  << std::setw(12) << std::min(tp2, 99999.0) << "ms"
                  << std::setw(12) << std::min(pp2, 99999.0) << "ms"
                  << std::setw(12) << best << "\n";
    }

    // ===== 分析3: Goodput 优化 =====
    std::cout << "\n\n【分析3】Goodput 优化 - 共置 vs 解耦 (13B, 8 GPU)\n";
    std::cout << "─────────────────────────────────────────────────────────────\n";
    std::cout << "  SLO: P90 TTFT < 200ms, P90 TPOT < 50ms\n\n";
    
    // 13B 模型参数
    double prefill_D = 35.0;      // ms, 单 GPU prefill (input=512)
    double decode_base_tpot = 5.0; // ms, 单 GPU 单请求 decode step
    double ttft_slo = 200.0;       // ms
    double tpot_slo = 50.0;        // ms
    int total_gpus = 8;
    
    // 共置方案: 每个 GPU 同时处理 prefill 和 decode
    // 干扰导致有效 TTFT 和 TPOT 都退化
    double colocate_prefill_rate = 0;  // rps per GPU
    {
        // 由于干扰, 有效 prefill 时间增加 ~30%
        double eff_D = prefill_D * 1.3;
        for (double r = 0.1; r <= 50.0; r += 0.1) {
            double p90 = md1_p90_latency(eff_D, r / 1000.0);
            if (p90 > ttft_slo) {
                colocate_prefill_rate = (r - 0.1);
                break;
            }
        }
    }
    
    // 共置: decode 受 prefill 干扰
    double colocate_decode_rate = 0;
    {
        // 干扰使 TPOT 有效增加 ~40%
        double eff_tpot = decode_base_tpot * 1.4;
        colocate_decode_rate = decode_max_rate(eff_tpot, tpot_slo);
    }
    
    double colocate_goodput_per_gpu = std::min(colocate_prefill_rate, 
                                                colocate_decode_rate / total_gpus * total_gpus);
    // 简化: 取 prefill 和 decode 约束中更严的
    double colocate_total = std::min(colocate_prefill_rate * total_gpus, colocate_decode_rate);
    double colocate_per_gpu = colocate_total / total_gpus;
    
    std::cout << "  共置 (8 GPU, 统一配置):\n";
    std::cout << "    Prefill 受限: " << std::fixed << std::setprecision(1) 
              << colocate_prefill_rate << " rps/GPU\n";
    std::cout << "    Decode  受限: " << colocate_decode_rate << " rps (total)\n";
    std::cout << "    Per-GPU Goodput: " << colocate_per_gpu << " rps/GPU\n\n";
    
    // 解耦方案: 搜索最优分配
    std::cout << "  解耦方案搜索:\n";
    std::cout << std::setw(14) << "P:D GPUs"
              << std::setw(16) << "Prefill cap"
              << std::setw(16) << "Decode cap"
              << std::setw(16) << "系统 Goodput"
              << std::setw(16) << "Per-GPU" << "\n";
    
    Config best = {0, 0, "", "", 0, 0};
    
    int splits[][2] = {{1,7}, {2,6}, {3,5}, {4,4}, {5,3}, {6,2}, {7,1}};
    for (auto& s : splits) {
        int p_gpus = s[0], d_gpus = s[1];
        
        // Prefill: 无干扰, 独立运行
        double p_rate_per_gpu = 0;
        for (double r = 0.1; r <= 100.0; r += 0.1) {
            double p90 = md1_p90_latency(prefill_D, r / 1000.0);
            if (p90 > ttft_slo) {
                p_rate_per_gpu = r - 0.1;
                break;
            }
        }
        double total_p_rate = p_rate_per_gpu * p_gpus;
        
        // Decode: 无干扰, 独立运行, 可以大 batch
        double total_d_rate = decode_max_rate(decode_base_tpot, tpot_slo) * d_gpus / 1.0;
        // 简化: decode 大 batch 提升效率
        total_d_rate = std::min(total_d_rate, (double)d_gpus * 10.0);
        
        double system_goodput = std::min(total_p_rate, total_d_rate);
        double per_gpu = system_goodput / total_gpus;
        
        std::cout << std::setw(6) << p_gpus << ":" << d_gpus
                  << std::setw(13) << std::setprecision(1) << total_p_rate << " rps"
                  << std::setw(13) << total_d_rate << " rps"
                  << std::setw(13) << system_goodput << " rps"
                  << std::setw(13) << per_gpu << " rps/GPU"
                  << (per_gpu > best.per_gpu_goodput ? "  ★" : "")
                  << "\n";
        
        if (per_gpu > best.per_gpu_goodput) {
            best = {p_gpus, d_gpus, "", "", system_goodput, per_gpu};
        }
    }
    
    std::cout << "\n  最优配置: Prefill " << best.prefill_gpus 
              << " GPU, Decode " << best.decode_gpus << " GPU\n";
    std::cout << "  解耦 Per-GPU Goodput: " << std::setprecision(1) << best.per_gpu_goodput << " rps/GPU\n";
    std::cout << "  共置 Per-GPU Goodput: " << colocate_per_gpu << " rps/GPU\n";
    std::cout << "  加速比: " << std::setprecision(2) << best.per_gpu_goodput / colocate_per_gpu << "x\n";

    // ===== 分析4: KV Cache 传输开销 =====
    std::cout << "\n\n【分析4】KV Cache 传输开销分析\n";
    std::cout << "─────────────────────────────────────────────────────────────\n\n";
    
    struct ModelKV {
        const char* model;
        int layers;
        int kv_heads;
        int head_dim;
    };
    
    ModelKV models[] = {
        {"LLaMA2-7B",  32, 32, 128},
        {"LLaMA2-13B", 40, 40, 128},
        {"LLaMA2-70B", 80,  8, 128},
        {"GPT-3 175B", 96, 96, 128},
    };
    
    int input_lens[] = {128, 512, 1024, 2048};
    
    std::cout << std::setw(14) << "模型"
              << std::setw(10) << "Input=128"
              << std::setw(10) << "Input=512"
              << std::setw(11) << "Input=1024"
              << std::setw(11) << "Input=2048"
              << "  (MB, fp16)\n";
    
    for (auto& m : models) {
        std::cout << std::setw(14) << m.model;
        for (int L : input_lens) {
            double kv_bytes = 2.0 * m.layers * m.kv_heads * m.head_dim * L * 2;
            double kv_mb = kv_bytes / (1024.0 * 1024.0);
            std::cout << std::setw(10) << std::fixed << std::setprecision(0) << kv_mb;
        }
        std::cout << "\n";
    }
    
    std::cout << "\n  传输时间 (LLaMA2-13B, input=512):\n";
    double kv_13b_512_mb = 2.0 * 40 * 40 * 128 * 512 * 2 / (1024.0 * 1024.0);
    
    struct Interconnect {
        const char* name;
        double bw_GBs;
    };
    Interconnect links[] = {
        {"NVLink 3.0",  300.0},
        {"NVLink 4.0",  450.0},  
        {"PCIe 4.0",     32.0},
        {"PCIe 5.0",     64.0},
        {"InfiniBand HDR", 25.0},
        {"InfiniBand NDR", 50.0},
    };
    
    std::cout << std::setw(20) << "互联"
              << std::setw(14) << "带宽(GB/s)"
              << std::setw(14) << "传输时间"
              << std::setw(16) << "占TTFT_SLO" << "\n";
    
    for (auto& l : links) {
        double time_ms = kv_13b_512_mb / l.bw_GBs;
        double pct = time_ms / 200.0 * 100;
        std::cout << std::setw(20) << l.name
                  << std::setw(12) << std::setprecision(0) << l.bw_GBs << " GB/s"
                  << std::setw(12) << std::setprecision(1) << time_ms << " ms"
                  << std::setw(14) << std::setprecision(1) << pct << "%" << "\n";
    }

    // ===== 分析5: 不同应用场景 =====
    std::cout << "\n\n【分析5】不同应用场景的 SLO 与最优配置\n";
    std::cout << "─────────────────────────────────────────────────────────────\n\n";
    
    struct AppScenario {
        const char* app;
        int avg_input;
        int avg_output;
        double ttft_slo;
        double tpot_slo;
        const char* bottleneck;
        const char* optimal_ratio;
    };
    
    AppScenario apps[] = {
        {"实时聊天",    128,  256, 100, 60,  "TTFT (要求即时响应)",   "P多:D少 (4:4)"},
        {"代码补全",     64,   32,  50, 30,  "TTFT (要求极低延迟)",   "P多:D少 (5:3)"},
        {"文档摘要",   2048,  512, 500, 30,  "TPOT (长输出快生成)",   "P少:D多 (2:6)"},
        {"批量翻译",   1024, 1024, 2000, 50, "吞吐 (高并发)",        "平衡 (3:5)"},
    };
    
    std::cout << std::setw(12) << "应用场景"
              << std::setw(12) << "平均输入"
              << std::setw(12) << "平均输出"
              << std::setw(12) << "TTFT SLO"
              << std::setw(12) << "TPOT SLO"
              << std::setw(24) << "瓶颈"
              << std::setw(16) << "最优P:D" << "\n";
    
    for (auto& a : apps) {
        std::cout << std::setw(12) << a.app
                  << std::setw(12) << a.avg_input
                  << std::setw(12) << a.avg_output
                  << std::setw(10) << a.ttft_slo << "ms"
                  << std::setw(10) << a.tpot_slo << "ms"
                  << std::setw(24) << a.bottleneck
                  << std::setw(16) << a.optimal_ratio << "\n";
    }
    
    std::cout << "\n═══════════════════════════════════════════════════════════════\n";
    std::cout << "结论：\n";
    std::cout << "  1. Prefill-Decoding 共置导致互相干扰 → TPOT 退化 3-8 倍\n";
    std::cout << "  2. 解耦后各阶段可独立优化并行策略 → Goodput 提升 2-4 倍\n";
    std::cout << "  3. KV Cache 传输在 NVLink 下开销极小 (<1% TTFT SLO)\n";
    std::cout << "  4. 不同应用场景的最优 P:D 比例差异很大 → 灵活配置是关键\n";
    std::cout << "═══════════════════════════════════════════════════════════════\n";

    return 0;
}
```
:::

---

## 实验结果

### 1. 端到端 Goodput 对比

DistServe 在三种典型应用场景下与 SOTA 系统对比：

```
Per-GPU Goodput (rps/GPU)，满足 P90 SLO attainment ≥ 90%：

场景: 实时聊天 (TTFT<200ms, TPOT<50ms)
模型           vLLM    TensorRT-LLM    DeepSpeed     DistServe    加速比
─────────────────────────────────────────────────────────────────────────
LLaMA2-7B      2.1        2.5            1.8          5.8         2.3x
LLaMA2-13B     1.2        1.6            1.0          3.8         2.4x
LLaMA2-70B     0.3        0.4            0.2          1.1         2.8x
─────────────────────────────────────────────────────────────────────────

场景: 代码补全 (TTFT<50ms, TPOT<30ms)
模型           vLLM    TensorRT-LLM    DistServe    加速比
────────────────────────────────────────────────────────────
LLaMA2-13B     0.4        0.5           3.0         6.0x
─────────────────────────────────────────────────────────────

场景: 文档摘要 (TTFT<500ms, TPOT<30ms)  
模型           vLLM    TensorRT-LLM    DistServe    加速比
────────────────────────────────────────────────────────────
LLaMA2-13B     0.8        1.0           7.4         7.4x
────────────────────────────────────────────────────────────
```

**关键发现**：
- DistServe 在所有场景下都显著优于共置系统
- **SLO 越严格，解耦优势越大**：代码补全（TTFT<50ms）场景下加速高达 6-7 倍
- 文档摘要场景下加速 7.4 倍——因为 TPOT 要求严格，共置系统的 decode 受 prefill 干扰严重

### 2. SLO Attainment 对比

```
满足 P90 SLO attainment 时可支持的最严 SLO (LLaMA2-13B, 2rps)：

指标         vLLM        DistServe      改善
───────────────────────────────────────────────
TTFT SLO    1000ms        80ms         12.6x 更严
TPOT SLO    200ms         30ms         6.7x  更严
```

### 3. 消融实验

```
消融实验 (13B, 8 GPU, 聊天场景)：

配置                              Per-GPU Goodput    vs 完整版
────────────────────────────────────────────────────────────
完整 DistServe                      3.8 rps/GPU        -
  去掉解耦 (回到共置)                1.6 rps/GPU      -58%
  解耦 + 固定并行 (不独立优化)       2.5 rps/GPU      -34%
  解耦 + 固定 P:D=1:1              2.9 rps/GPU      -24%
────────────────────────────────────────────────────────────

结论:
  - 解耦本身贡献最大 (~56% 的提升来自消除干扰)
  - 独立并行策略贡献 ~24%
  - 灵活的 P:D 比例贡献 ~10%
```

---

## 与相关工作的关系

```
LLM Serving 系统演进：

Orca (2022):
  ├─ 提出 Continuous Batching (iteration-level scheduling)
  └─ 问题: prefill-decode 混合执行

vLLM (2023):
  ├─ PagedAttention → 高效 KV Cache 管理
  ├─ 仍然是共置 + continuous batching
  └─ 问题: prefill-decode 干扰

Sarathi / Sarathi-Serve (2023-2024):
  ├─ Chunked-Prefill with Piggybacking
  ├─ 将长 prefill 拆成小块，与 decode 混合
  ├─ 缓解干扰但不消除（KV Cache 重复加载 O(N²)）
  └─ 仍然是共置

DistServe (2024):                              ← 本文
  ├─ 彻底解耦 prefill 和 decode 到不同 GPU
  ├─ 消除干扰 + 独立并行策略 + 灵活资源分配
  └─ 代价: KV Cache 跨 GPU 传输（开销可控）

Splitwise (2024, 微软):
  ├─ 类似的 prefill-decode 解耦思想
  ├─ 更关注异构硬件（不同代 GPU 混合部署）
  └─ DistServe 和 Splitwise 独立并行提出

Mooncake (2024, 月之暗面):
  ├─ 在 DistServe 基础上进一步发展
  ├─ 将 KV Cache 放到分布式对象存储
  └─ 更大规模的解耦部署
```

---

## 总结与启示

### 核心贡献

1. **发现问题**：首次系统性地识别并量化了 prefill-decoding 共置带来的三大问题（干扰、资源耦合、调度困难）
2. **解耦方案**：将两个阶段分配到不同 GPU，从根本上消除干扰
3. **Goodput 优化**：提出基于排队论模型的资源分配优化算法，自动搜索最优并行策略和 P:D 比例
4. **布局算法**：考虑集群拓扑和带宽，最小化 KV Cache 传输开销
5. **显著效果**：Goodput 提升 2-7 倍，SLO 可收紧 12.6 倍

### 设计哲学

```
DistServe 的核心洞察：

1. 分治法（Divide and Conquer）
   ├─ 两个阶段计算特征不同 → 不要强行放一起
   ├─ 分开后各自优化 → 整体效果远超共置
   └─ 类比: 读写分离（数据库）、计算存储分离（云原生）

2. 以 SLO 为导向优化
   ├─ 不追求最大吞吐量 → 追求满足 SLO 的最大 Goodput
   ├─ Goodput ≠ 吞吐: 超过 SLO 的请求不计入 Goodput
   └─ 这改变了优化目标和资源分配策略

3. 通信开销可控
   ├─ KV Cache 传输看似增加开销
   ├─ 但相比 SLO（百毫秒级）是微小的（毫秒级）
   └─ 消除干扰的收益远超传输代价

4. 对产业的影响
   ├─ 这个思想已被广泛采纳
   ├─ vLLM / SGLang / TensorRT-LLM 后续版本都在探索类似解耦
   └─ 月之暗面的 Mooncake 系统是该思想的大规模实践
```

---

## 参考文献

1. Yinmin Zhong et al. *DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving*. OSDI 2024.
2. Woosuk Kwon et al. *Efficient Memory Management for Large Language Model Serving with PagedAttention*. SOSP 2023.
3. Amey Agrawal et al. *Sarathi: Efficient LLM Inference by Piggybacking Decodes with Chunked Prefills*. 2023.
4. Pratyush Patel et al. *Splitwise: Efficient Generative LLM Inference Using Phase Splitting*. ISCA 2024.
5. Ruoyu Qin et al. *Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving*. 2024.
6. Hugo Touvron et al. *Llama 2: Open Foundation and Fine-Tuned Chat Models*. 2023.
