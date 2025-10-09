<template>
  <div class="layout-grid">
    <section class="section">
      <h2>System Status</h2>
      <p>Check health status for the backend services.</p>
      <div class="actions">
        <button type="button" @click="loadStatus" :disabled="loading">Load Status</button>
      </div>
      <div v-if="response" class="response-card">
        <pre>{{ response }}</pre>
      </div>
    </section>
  </div>
</template>

<script setup>
import { ref } from 'vue'
import { systemApi } from '../services/api'

const loading = ref(false)
const response = ref('')

function stringify(data) {
  return JSON.stringify(data, null, 2)
}

async function loadStatus() {
  loading.value = true
  try {
    const res = await systemApi.status()
    response.value = stringify(res.data)
  } catch (error) {
    response.value = stringify(error.response?.data || error.message)
  } finally {
    loading.value = false
  }
}
</script>
