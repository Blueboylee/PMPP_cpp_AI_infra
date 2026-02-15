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
      { text: '博客', link: '/posts/' },
      { text: '关于', link: '/about' },
      {
        text: '源码',
        link: 'https://github.com/Blueboylee/AI-INFRA-ALL-IN-ONE/tree/main/src',
      },
    ],

    sidebar: {
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
          ],
        },
      ],
      '/posts/': [
        {
          text: '博客文章',
          items: [
            { text: 'DeepSpeed ZeRO 系列总结', link: '/posts/hello-world' },
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
