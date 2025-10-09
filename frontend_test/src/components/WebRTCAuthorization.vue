<template>
  <div class="webrtc-auth">
    <h2>WebRTC Authentication / Verification</h2>
    <div class="controls">
      <input v-model="targetEmail" placeholder="Target email (verification optional)" />
      <select v-model="sessionType">
        <option value="authentication">Authentication</option>
        <option value="verification">Verification</option>
        <option value="identification">Identification</option>
      </select>
      <button @click="start" :disabled="running">Start</button>
      <button @click="stop" :disabled="!running">Stop</button>
      <button @click="toggleMode" class="secondary">Mode: {{ modeLabel }}</button>
    </div>
    <div class="status-line">Status: {{ status }} / Frames: {{ frames }} / Liveness: {{ liveness }}</div>
    <CameraStream ref="cam" />
    <div v-if="authSuccess" class="success-box">Success! similarity={{ authSimilarity?.toFixed(3) }}</div>
    <div v-if="errorMsg" class="error-box">{{ errorMsg }}</div>
  </div>
</template>
<script setup>
import { ref, onBeforeUnmount, computed } from 'vue'
import CameraStream from './CameraStream.vue'
import { faceAuthApi } from '../services/api'
import { useWebRTCSession } from '../composables/useWebRTCSession'

const cam = ref(null)
const running = ref(false)
const loopHandle = ref(null)
const httpMode = ref(false)
const targetEmail = ref('')
const sessionType = ref('authentication')
const authSuccess = ref(false)
const errorMsg = ref('')
const authSimilarity = ref(0)
let httpSessionToken = null

const { connect, sendFrame, close, status, frames, liveness } = useWebRTCSession({
  onResult: handleResult,
  onFinal: handleFinal,
  onError: handleWebRTCError
})

const modeLabel = computed(() => (httpMode.value ? 'HTTP' : 'WebSocket'))

function resolveSessionType() {
  if (!targetEmail.value && sessionType.value !== 'identification') {
    return 'identification'
  }
  return sessionType.value
}

async function start() {
  if (running.value) return
  running.value = true
  authSuccess.value = false
  errorMsg.value = ''
  httpSessionToken = null

  const resolvedType = resolveSessionType()
  const payload = {
    session_type: resolvedType,
    email: targetEmail.value || null,
    device_info: { device_id: 'webrtc-auth' }
  }

  try {
    if (httpMode.value) {
      const { data } = await faceAuthApi.createSession(payload)
      httpSessionToken = data.session_token
    } else {
      const { data } = await faceAuthApi.createWebRTCSession(payload)
      connect(data.session_token)
    }

    await cam.value.start()
    pump()
  } catch (error) {
    running.value = false
    errorMsg.value = error?.response?.data?.detail || error?.message || 'Gagal memulai sesi'
    stop()
  }
}

function pump() {
  if (!running.value) return
  loopHandle.value = requestAnimationFrame(async () => {
    try {
      const frame = await cam.value.captureFrame({ quality: 0.75 })
      if (httpMode.value) {
        if (!httpSessionToken) throw new Error('HTTP auth session missing token')
        await faceAuthApi.processFrame({ session_token: httpSessionToken, frame_data: frame })
      } else {
        sendFrame(frame)
      }
    } catch (e) { errorMsg.value = e.message }
    pump()
  })
}

function stop() {
  running.value = false
  if (loopHandle.value) {
    cancelAnimationFrame(loopHandle.value)
    loopHandle.value = null
  }
  httpSessionToken = null
  cam.value.stop()
  close()
}

function toggleMode() {
  if (running.value) return
  httpMode.value = !httpMode.value
}

function handleResult(res) {
  if (res?.success) {
    authSuccess.value = true
    authSimilarity.value = res.similarity_score || 0
  }
}

function handleFinal(message) {
  const result = message?.result || {}
  if (result?.success) {
    authSuccess.value = true
    authSimilarity.value = result.similarity_score || authSimilarity.value
  }
  stop()
}

function handleWebRTCError(error) {
  errorMsg.value = error?.message || 'WebRTC connection error'
}

onBeforeUnmount(stop)
</script>
