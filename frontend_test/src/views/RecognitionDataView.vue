<template>
  <div class="layout-grid">
    <section class="section">
      <h2>Recognition Data</h2>
      <p>Akses cepat ke embedding, sesi enrollment, serta riwayat autentikasi untuk troubleshooting.</p>
      <div class="actions">
        <button type="button" @click="loadEmbeddings" :disabled="loading.embeddings">Embeddings</button>
        <button type="button" @click="loadSessions" :disabled="loading.sessions">Enrollment Sessions</button>
        <button type="button" @click="loadAttempts" :disabled="loading.attempts">Auth Attempts</button>
      </div>
      <div v-if="responses.embeddings" class="response-card">
        <h3>Embeddings</h3>
        <pre>{{ responses.embeddings }}</pre>
      </div>
      <div v-if="responses.sessions" class="response-card">
        <h3>Enrollment Sessions</h3>
        <pre>{{ responses.sessions }}</pre>
      </div>
      <div v-if="responses.attempts" class="response-card">
        <h3>Authentication Attempts</h3>
        <pre>{{ responses.attempts }}</pre>
      </div>
    </section>
  </div>
</template>

<script setup>
import { reactive } from 'vue'
import { recognitionApi } from '../services/api'

const loading = reactive({
  embeddings: false,
  sessions: false,
  attempts: false
})

const responses = reactive({
  embeddings: '',
  sessions: '',
  attempts: ''
})

function stringify(data) {
  return JSON.stringify(data, null, 2)
}

async function loadEmbeddings() {
  loading.embeddings = true
  try {
    const response = await recognitionApi.listEmbeddings()
    responses.embeddings = stringify(response.data)
  } catch (error) {
    responses.embeddings = stringify(error.response?.data || error.message)
  } finally {
    loading.embeddings = false
  }
}

async function loadSessions() {
  loading.sessions = true
  try {
    const response = await recognitionApi.listSessions()
    responses.sessions = stringify(response.data)
  } catch (error) {
    responses.sessions = stringify(error.response?.data || error.message)
  } finally {
    loading.sessions = false
  }
}

async function loadAttempts() {
  loading.attempts = true
  try {
    const response = await recognitionApi.listAttempts()
    responses.attempts = stringify(response.data)
  } catch (error) {
    responses.attempts = stringify(error.response?.data || error.message)
  } finally {
    loading.attempts = false
  }
}
</script>
