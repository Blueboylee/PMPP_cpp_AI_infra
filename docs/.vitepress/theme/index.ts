import DefaultTheme from 'vitepress/theme'
import CppPlayground from './components/CppPlayground.vue'
import LangSwitch from './components/LangSwitch.vue'
import type { Theme } from 'vitepress'
import { h } from 'vue'
import './styles/lang-switch.css'

const theme: Theme = {
  extends: DefaultTheme,
  Layout() {
    return h(DefaultTheme.Layout, null, {
      'nav-bar-content-after': () => h(LangSwitch),
    })
  },
  enhanceApp({ app }) {
    app.component('CppPlayground', CppPlayground)
  },
}

export default theme
