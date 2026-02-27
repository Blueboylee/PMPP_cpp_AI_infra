import { defineConfig } from 'vitepress'
import { cppPlaygroundPlugin } from './markdown-it-cpp-playground'

export default defineConfig({
  title: 'AI Infrastructure',
  description: 'AI Infra 全栈学习笔记：深入 CUDA 并行编程、vLLM PagedAttention、SGLang、TensorRT-LLM、Triton Inference Server 等推理引擎原理与实践，涵盖 GPU 优化、算子融合、模型服务化部署。',
  base: '/AI-INFRA-ALL-IN-ONE/',
  lang: 'zh-CN',

  markdown: {
    math: true,
    config: (md) => {
      md.use(cppPlaygroundPlugin)
    },
  },

  head: [
    ['link', { rel: 'icon', type: 'image/svg+xml', href: '/AI-INFRA-ALL-IN-ONE/logo.svg' }],
    ['meta', { name: 'keywords', content: 'AI Infrastructure, CUDA, vLLM, PagedAttention, SGLang, TensorRT-LLM, Triton Inference Server, OpenAI Triton, GPU编程, 推理引擎, 模型服务化, 算子优化, LLM推理优化, AI基础设施, 高性能计算' }],
    ['meta', { name: 'author', content: 'Blueboylee' }],
    // Open Graph
    ['meta', { property: 'og:type', content: 'website' }],
    ['meta', { property: 'og:title', content: 'AI Infra 学习笔记 — 从 CUDA 到推理引擎全栈技术' }],
    ['meta', { property: 'og:description', content: '系统学习 AI 基础设施：CUDA 并行编程、vLLM、SGLang、TensorRT-LLM、Triton 推理服务、算子优化等全栈技术笔记与论文精读。' }],
    ['meta', { property: 'og:url', content: 'https://blueboylee.github.io/AI-INFRA-ALL-IN-ONE/' }],
    ['meta', { property: 'og:locale', content: 'zh_CN' }],
    ['meta', { property: 'og:site_name', content: 'AI Infra 学习笔记' }],
    // Twitter Card
    ['meta', { name: 'twitter:card', content: 'summary' }],
    ['meta', { name: 'twitter:title', content: 'AI Infra 学习笔记 — 从 CUDA 到推理引擎全栈技术' }],
    ['meta', { name: 'twitter:description', content: '系统学习 AI 基础设施：CUDA 并行编程、vLLM、SGLang、TensorRT-LLM、Triton 推理服务、算子优化等全栈技术笔记与论文精读。' }],
  ],

  themeConfig: {
    nav: [
      { text: '首页', link: '/' },
      { text: 'PMPP 专栏', link: '/posts/pmpp/' },
      { text: '博客', link: '/posts/' },
      { text: '关于', link: '/about' },
      {
        text: '源码',
        link: 'https://github.com/Blueboylee/AI-INFRA-ALL-IN-ONE/tree/main/src',
      },
    ],

    sidebar: {
      '/posts/pmpp/': [
        {
          text: 'PMPP：并行处理器编程',
          items: [
            { text: '⚡ 专栏介绍', link: '/posts/pmpp/' },
            { text: 'Ch01: Introduction', link: '/posts/pmpp/ch01' },
            { text: 'Ch02: 异构数据并行计算', link: '/posts/pmpp/ch02' },
            { text: 'Ch03: 多维网格与数据', link: '/posts/pmpp/ch03' },
            { text: 'Ch04: 计算架构与调度', link: '/posts/pmpp/ch04' },
            { text: 'Ch05: 内存架构与数据局部性', link: '/posts/pmpp/ch05' },
            { text: 'Ch06: 性能优化', link: '/posts/pmpp/ch06' },
            { text: 'Ch07: 卷积', link: '/posts/pmpp/ch07' },
            { text: 'Ch08: Stencil', link: '/posts/pmpp/ch08' },
            { text: 'Ch09: 并行直方图', link: '/posts/pmpp/ch09' },
            { text: 'Ch10: 归约', link: '/posts/pmpp/ch10' },
            { text: 'Ch11: 前缀和 (Scan)', link: '/posts/pmpp/ch11' },
            { text: 'Ch12: 归并', link: '/posts/pmpp/ch12' },
            { text: 'Ch13: 排序', link: '/posts/pmpp/ch13' },
            { text: 'Ch14: 稀疏矩阵计算', link: '/posts/pmpp/ch14' },
            { text: 'Ch15: 图遍历', link: '/posts/pmpp/ch15' },
            { text: 'Ch16: 深度学习', link: '/posts/pmpp/ch16' },
            { text: 'Ch17: 迭代式 MRI 重建', link: '/posts/pmpp/ch17' },
            { text: 'Ch18: 静电势图', link: '/posts/pmpp/ch18' },
            { text: 'Ch19: 并行编程与计算思维', link: '/posts/pmpp/ch19' },
            { text: 'Ch20: 异构集群编程', link: '/posts/pmpp/ch20' },
            { text: 'Ch21: CUDA 动态并行', link: '/posts/pmpp/ch21' },
            { text: 'Ch22: 高级实践与未来展望', link: '/posts/pmpp/ch22' },
          ],
        },
      ],
      '/posts/inference/': [
        {
          text: '推理引擎与服务化',
          items: [
            { text: '📚 文献列表', link: '/posts/inference/' },
            { text: 'vLLM: PagedAttention', link: '/posts/inference/vllm-paper' },
            { text: 'TensorRT-LLM', link: '/posts/inference/tensorrt-llm' },
            { text: 'Triton Inference Server', link: '/posts/inference/triton-inference-server' },
            { text: 'SGLang: 结构化生成语言', link: '/posts/inference/sglang' },
            { text: 'FlashAttention: IO感知注意力', link: '/posts/inference/flash-attention' },
            { text: 'Clipper: 在线推理服务系统', link: '/posts/inference/clipper' },
            { text: 'SmoothQuant: LLM INT8 量化', link: '/posts/inference/smoothquant' },
            { text: 'Speculative Decoding: 投机解码', link: '/posts/inference/speculative-decoding' },
            { text: 'Flash-Decoding: 长上下文解码加速', link: '/posts/inference/flash-decoding' },
            { text: 'FlashDecoding++: 异步解码加速', link: '/posts/inference/flash-decoding-pp' },
            { text: 'DistServe: Prefill-Decode 解耦', link: '/posts/inference/distserve' },
          ],
        },
      ],
      '/posts/training/': [
        {
          text: '分布式训练',
          items: [
            { text: '📚 文献列表', link: '/posts/training/' },
            { text: 'ZeRO: 零冗余优化器', link: '/posts/training/deepspeed-zero' },
            { text: 'ZeRO-Offload: 异构卸载训练', link: '/posts/training/zero-offload' },
            { text: 'ZeRO-Infinity: NVMe 极限扩展', link: '/posts/training/zero-infinity' },
            { text: 'Megatron-LM: 张量并行', link: '/posts/training/megatron-lm' },
          ],
        },
      ],
      '/posts/': [
        {
          text: '博客文章',
          items: [
            { text: 'DeepSpeed ZeRO 系列总结', link: '/posts/hello-world' },
            { text: 'NCCL Ring / Tree 算法与拓扑自适应', link: '/posts/nccl-topology-ring-tree' },
          ],
        },
      ],
    },

    socialLinks: [
      { icon: 'github', link: 'https://github.com/Blueboylee/AI-INFRA-ALL-IN-ONE' },
    ],

    footer: {
      message: 'Released under the MIT License.',
      copyright: 'Copyright © 2026 Blueboylee',
    },

    outline: {
      label: '目录',
    },

    docFooter: {
      prev: '上一篇',
      next: '下一篇',
    },

    search: {
      provider: 'local',
      options: {
        translations: {
          button: { buttonText: '搜索', buttonAriaLabel: '搜索' },
          modal: {
            noResultsText: '未找到相关结果',
            resetButtonTitle: '清除查询',
            footer: { selectText: '选择', navigateText: '切换', closeText: '关闭' },
          },
        },
      },
    },
  },
})
