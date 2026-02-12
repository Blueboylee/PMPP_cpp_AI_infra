<script setup lang="ts">
import { ref, onMounted, watch } from 'vue'

const STORAGE_KEY = 'preferred-lang'
const lang = ref<'zh' | 'en'>('zh')

onMounted(() => {
  const saved = localStorage.getItem(STORAGE_KEY)
  if (saved === 'zh' || saved === 'en') {
    lang.value = saved
  }
  applyLang(lang.value)
})

watch(lang, (val) => {
  localStorage.setItem(STORAGE_KEY, val)
  applyLang(val)
})

function applyLang(l: 'zh' | 'en') {
  document.documentElement.setAttribute('data-lang', l)
}
</script>

<template>
  <div class="lang-switch-nav">
    <button
      class="lang-btn"
      :class="{ active: lang === 'zh' }"
      @click="lang = 'zh'"
    >
      中
    </button>
    <span class="lang-sep">/</span>
    <button
      class="lang-btn"
      :class="{ active: lang === 'en' }"
      @click="lang = 'en'"
    >
      EN
    </button>
  </div>
</template>

<style scoped>
.lang-switch-nav {
  display: inline-flex;
  align-items: center;
  gap: 2px;
  margin-left: 8px;
  font-size: 13px;
  line-height: 1;
}

.lang-btn {
  padding: 4px 8px;
  border: none;
  border-radius: 6px;
  background: transparent;
  color: var(--vp-c-text-2);
  cursor: pointer;
  font-weight: 500;
  font-size: 13px;
  transition: all 0.2s ease;
}

.lang-btn.active {
  background: var(--vp-c-brand-soft);
  color: var(--vp-c-brand-1);
}

.lang-btn:hover:not(.active) {
  color: var(--vp-c-text-1);
  background: var(--vp-c-bg-soft);
}

.lang-sep {
  color: var(--vp-c-divider);
  font-size: 12px;
  user-select: none;
}
</style>
