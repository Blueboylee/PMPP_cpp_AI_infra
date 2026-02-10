import { defineConfig } from 'vitepress'

export default defineConfig({
  title: 'PMPP - C++ & AI Infrastructure',
  description: 'CUDA 并行编程与 AI 基础设施学习笔记',
  base: '/PMPP_cpp_AI_infra/',

  head: [
    ['link', { rel: 'icon', type: 'image/svg+xml', href: '/PMPP_cpp_AI_infra/logo.svg' }],
  ],

  themeConfig: {
    nav: [
      { text: '首页', link: '/' },
      { text: '博客', link: '/posts/' },
      { text: '关于', link: '/about' },
      {
        text: '源码',
        link: 'https://github.com/Blueboylee/PMPP_cpp_AI_infra/tree/main/src',
      },
    ],

    sidebar: {
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
      { icon: 'github', link: 'https://github.com/Blueboylee/PMPP_cpp_AI_infra' },
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
