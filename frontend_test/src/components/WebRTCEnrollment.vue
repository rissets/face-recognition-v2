<template>
  <div class="webrtc-enroll">
    <h2>WebRTC Enrollment</h2>
    <div class="controls">
      <button @click="start" :disabled="running">Start</button>
      <button @click="stop" :disabled="!running">Stop</button>
      <button @click="switchMode" class="secondary">Mode: {{ modeLabel }}</button>
    </div>
    <div class="status-line">Status: {{ status }}</div>
    <CameraStream ref="cam" />
    <div class="metrics">
      <span>Frames: {{ frames }}</span>
      <span>Embeddings Saved: {{ embeddingsSaved }}</span>
      <span>Liveness Blinks: {{ liveness }}</span>
      <span>Quality Last: {{ lastResult?.quality_score?.toFixed(3) }}</span>
    </div>
  </div>
</template>
<script setup>
import { ref, onBeforeUnmount, computed } from 'vue'
import CameraStream from './CameraStream.vue'
import { enrollmentApi, API_BASE_URL } from '../services/api'
import { useWebRTCSession } from '../composables/useWebRTCSession'

const cam = ref(null)
const running = ref(false)
const loopHandle = ref(null)
const httpFallback = ref(false)
let httpSessionToken = null
const {
  connect, sendFrame, close, status, frames, embeddingsSaved, liveness, lastResult
} = useWebRTCSession()

const modeLabel = computed(() => httpFallback.value ? 'HTTP' : 'WebSocket')

async function start() {
  if (running.value) return
  running.value = true
  httpSessionToken = null
  // Create WebRTC enrollment session
  const { data } = await enrollmentApi.createSession({}) // HTTP session for parallel fallback
  httpSessionToken = data?.session_token || null
  // Also request WebRTC session token (backend endpoint)
  const resp2 = await fetch(`${API_BASE_URL}enrollment/webrtc/create/`, { method: 'POST', headers: authHeaders() })
  if (!resp2.ok) {
    throw new Error(`Failed to create WebRTC enrollment session (${resp2.status})`)
  }
  const data2 = await resp2.json()
  if (!httpFallback.value) connect(data2.session_token)
  await cam.value.start()
  pump()
}

function authHeaders() { const t = localStorage.getItem('access_token'); return { 'Content-Type':'application/json', 'Authorization': t ? `Bearer ${t}`:'' } }

function pump() {
  if (!running.value) return
  loopHandle.value = requestAnimationFrame(async () => {
    try {
      const frame = await cam.value.captureFrame({ quality: 0.75 })
      if (httpFallback.value) {
        if (!httpSessionToken) throw new Error('HTTP enrollment session missing token')
        // HTTP fallback
        const resp = await fetch(`${API_BASE_URL}enrollment/process-frame/`, { method:'POST', headers: authHeaders(), body: JSON.stringify({ session_token: httpSessionToken, frame_data: frame }) })
        if (!resp.ok) throw new Error(`HTTP enrollment frame failed (${resp.status})`)
      } else {
        sendFrame(frame)
      }
    } catch (e) { console.error(e) }
    pump()
  })
}

function stop() {
  running.value = false
  if (loopHandle.value) cancelAnimationFrame(loopHandle.value)
  cam.value.stop()
  close()
}

function switchMode() {
  httpFallback.value = !httpFallback.value
}

onBeforeUnmount(stop)
</script>
