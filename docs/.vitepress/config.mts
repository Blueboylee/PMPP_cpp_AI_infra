import { defineConfig } from 'vitepress'
import { cppPlaygroundPlugin } from './markdown-it-cpp-playground'

export default defineConfig({
  title: 'AI Infrastructure',
  description: 'AI 基础设施学习笔记',
  base: '/AI-INFRA-ALL-IN-ONE/',

  markdown: {
    config: (md) => {
      md.use(cppPlaygroundPlugin)
    },
  },

  head: [
    ['link', { rel: 'icon', type: 'image/svg+xml', href: '/AI-INFRA-ALL-IN-ONE/logo.svg' }],
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
          ],
        },
      ],
      '/posts/': [
        {
          text: '博客文章',
          items: [
            { text: 'Hello World - 第一篇博客', link: '/posts/hello-world' },
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
