---
title: Hello World - 第一篇博客
date: 2026-02-10
---

# Hello World - 第一篇博客

<p style="color: var(--vp-c-text-2); font-size: 14px;">📅 2026-02-10 &nbsp;·&nbsp; 🏷️ 入门</p>

这是我的第一篇博客文章！

## 关于这个项目

这个博客用来记录学习 CUDA 并行编程和 AI 基础设施的过程。主要参考教材是 *Programming Massively Parallel Processors (PMPP)*。

## 一个简单的 CUDA 示例

```cpp
#include <stdio.h>

__global__ void helloKernel() {
    printf("Hello from GPU thread %d!\n", threadIdx.x);
}

int main() {
    helloKernel<<<1, 10>>>();
    cudaDeviceSynchronize();
    return 0;
}
```

后续会持续更新学习笔记，敬请关注！
