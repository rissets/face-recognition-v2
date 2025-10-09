<template>
  <div class="face-login-inline">
    <div class="inline-camera">
      <CameraStream
        ref="cameraRef"
        @started="onCameraStarted"
        @stopped="onCameraStopped"
        @error="handleCameraError"
      />
      <div class="camera-status">
        <span class="status-pill" :class="cameraActive ? 'success' : 'danger'">
          Kamera {{ cameraActive ? 'Aktif' : 'Off' }}
        </span>
        <span class="status-pill" :class="blinkCount >= requiredBlinks ? 'success' : 'warning'">
          Kedipan {{ blinkCount }} / {{ requiredBlinks }}
        </span>
        <span class="status-pill" :class="framesProcessed >= minFrames ? 'success' : 'warning'">
          Frame {{ framesProcessed }} / {{ minFrames }}
        </span>
      </div>
    </div>

    <div class="inline-controls">
      <div class="actions">
        <button type="button" class="secondary" @click="toggleTransport" :disabled="cameraActive || isLoading">
          Mode: {{ transportLabel }}
        </button>
        <button type="button" @click="startCapture" :disabled="cameraActive || isLoading">
          {{ cameraActive ? 'Kamera Aktif' : 'Mulai Kamera' }}
        </button>
        <button type="button" class="secondary" @click="stopCapture" :disabled="!cameraActive">
          Stop
        </button>
        <button type="button" class="danger" @click="cancel">Batal</button>
      </div>
      <div v-if="statusMessage" :class="['inline-status', statusClass]">
        {{ statusMessage }}
      </div>
    </div>
  </div>
</template>

<script setup>
import { computed, onBeforeUnmount, reactive, ref, watch } from 'vue'
import { useAuthStore } from '../stores/auth'
import { faceAuthApi } from '../services/api'
import { useWebRTCSession } from '../composables/useWebRTCSession'
import CameraStream from './CameraStream.vue'

const props = defineProps({
  email: {
    type: String,
    default: ''
  }
})

const emit = defineEmits(['cancel', 'completed'])

const cameraRef = ref(null)
const authStore = useAuthStore()

const state = reactive({
  sessionToken: null,
  transport: 'webrtc',
  cameraActive: false,
  framesProcessed: 0,
  blinkCount: 0,
  statusMessage: '',
  statusLevel: 'info',
  loading: false,
  minFrames: 3,
  requiredBlinks: 2,
  sessionType: props.email ? 'verification' : 'identification'
})

const loopHandle = ref(null)
const loopMode = ref(null)

const transportLabel = computed(() => (state.transport === 'webrtc' ? 'WebRTC' : 'HTTP'))
const cameraActive = computed(() => state.cameraActive)
const framesProcessed = computed(() => state.framesProcessed)
const blinkCount = computed(() => state.blinkCount)
const statusMessage = computed(() => state.statusMessage)
const statusClass = computed(() => {
  if (state.statusLevel === 'success') return 'status-success'
  if (state.statusLevel === 'warning') return 'status-warning'
  if (state.statusLevel === 'danger') return 'status-danger'
  return 'status-info'
})
const minFrames = computed(() => state.minFrames)
const requiredBlinks = computed(() => state.requiredBlinks)
const isLoading = computed(() => state.loading)

const { connect, sendFrame, close: closeWebRTC, status: webrtcStatus } = useWebRTCSession({
  onResult: handleWebRTCResult,
  onFinal: handleWebRTCFinal,
  onError: handleWebRTCError
})

watch(
  () => props.email,
  (value) => {
    const trimmed = (value || '').trim()
    if (state.cameraActive) {
      stopCapture()
    }
    state.sessionType = trimmed ? 'verification' : 'identification'
    state.sessionToken = null
    state.framesProcessed = 0
    state.blinkCount = 0
  }
)

watch(
  () => state.sessionType,
  (mode) => {
    state.requiredBlinks = mode === 'identification' ? 0 : 2
  },
  { immediate: true }
)

function setStatus(message, level = 'info') {
  state.statusMessage = message
  state.statusLevel = level
}

function extractErrorMessage(error, fallback = 'Terjadi kesalahan.') {
  if (!error) return fallback
  const responseData = error.response?.data
  if (typeof responseData === 'string') return responseData
  if (responseData?.error) return responseData.error
  if (responseData?.detail) return responseData.detail
  if (responseData?.message) return responseData.message
  if (Array.isArray(responseData?.messages) && responseData.messages.length) {
    return responseData.messages[0].message || fallback
  }
  if (error.message) return error.message
  return fallback
}

function onCameraStarted() {
  state.cameraActive = true
}

function onCameraStopped() {
  state.cameraActive = false
}

function buildSessionPayload() {
  const trimmedEmail = (props.email || '').trim() || null
  const apiSessionType = state.sessionType === 'identification' ? 'authentication' : state.sessionType
  const payload = {
    session_type: apiSessionType,
    email: trimmedEmail,
    device_info: {
      device_id: 'face-login-inline',
      device_name: 'Inline Face Login',
      device_type: 'web'
    }
  }
  console.debug('[FaceLoginInline] Session payload prepared', payload)
  return payload
}

async function prepareSession() {
  if (state.sessionToken) return state.sessionToken
  state.loading = true
  setStatus('Menyiapkan session autentikasi...', 'info')
  const payload = buildSessionPayload()
  try {
    console.debug('[FaceLoginInline] Creating session with transport', state.transport)
    const response =
      state.transport === 'webrtc'
        ? await faceAuthApi.createPublicWebRTCSession(payload)
        : await faceAuthApi.createPublicSession(payload)
    console.debug('[FaceLoginInline] Session create response', response?.status, response?.data)
    state.sessionToken = response.data?.session_token || null
    if (!state.sessionToken) {
      throw new Error('Server tidak mengembalikan session token')
    }
    if (state.transport === 'webrtc') {
      connect(state.sessionToken)
    }
    setStatus('Session siap. Pastikan wajah berada di tengah frame dan kedipkan mata beberapa kali.', 'info')
    return state.sessionToken
  } catch (error) {
    console.error('[FaceLoginInline] Session creation failed', error?.response?.data || error)
    let message = extractErrorMessage(error, 'Gagal membuat session autentikasi.')
    if (error?.response?.status === 429) {
      const retry = error?.response?.headers?.['retry-after']
      const base = error?.response?.data?.error || message
      message = retry ? `${base} Coba lagi dalam ${retry} detik.` : base
    }
    setStatus(message, 'danger')
    throw error
  } finally {
    state.loading = false
  }
}

async function startCapture() {
  if (state.loading || state.cameraActive) return
  try {
    await prepareSession()
    await cameraRef.value?.start()
    state.cameraActive = true
    state.framesProcessed = 0
    state.blinkCount = 0
    setStatus(
      state.transport === 'webrtc'
        ? 'Kamera aktif. Streaming via WebRTC, lakukan kedipan atau gerakan ringan.'
        : 'Kamera aktif. Mengirim frame via HTTP, lakukan kedipan atau gerakan ringan.',
      'warning'
    )
    sendFrameLoop()
  } catch (error) {
    const message = extractErrorMessage(error, 'Tidak dapat memulai kamera atau session.')
    setStatus(message, 'danger')
  }
}

async function sendFrameLoop() {
  if (!state.cameraActive || !state.sessionToken) return
  try {
    const frame = await cameraRef.value?.captureFrame()
    if (!frame) {
      throw new Error('Frame tidak tersedia')
    }

    if (state.transport === 'http') {
      const response = await faceAuthApi.processFrame({
        session_token: state.sessionToken,
        frame_data: frame
      })
      console.debug('[FaceLoginInline] HTTP frame response', response?.status, response?.data)
      handleHttpResponse(response.data)
    } else {
      if (webrtcStatus.value !== 'connected') {
        queueNextFrame(180)
        return
      }
      sendFrame(frame)
      queueNextFrame(0)
    }
  } catch (error) {
    console.error('[FaceLoginInline] sendFrameLoop error', error?.response?.data || error)
    const message = extractErrorMessage(error, 'Terjadi kesalahan saat mengirim frame.')
    setStatus(message, 'danger')
    queueNextFrame(state.transport === 'webrtc' ? 240 : 1500)
  }
}

function handleHttpResponse(data) {
  console.debug('[FaceLoginInline] Handling HTTP response payload', data)
  state.framesProcessed = data?.frames_processed ?? state.framesProcessed + 1
  if (typeof data?.liveness_blinks === 'number') {
    state.blinkCount = data.liveness_blinks
  }

  if (data?.requires_more_frames) {
    setStatus(data.message || 'Lanjutkan streaming untuk memenuhi persyaratan liveness.', 'warning')
    queueNextFrame(1200)
    return
  }

  if (data?.success) {
    setStatus('Autentikasi berhasil. Mengambil token...', 'success')
    finalizeFaceLogin(data)
    return
  }

  if (data?.requires_new_session) {
    setStatus(data.message || 'Session perlu direset. Menghentikan kamera.', 'danger')
    stopCapture()
    state.sessionToken = null
    emit('cancel')
    return
  }

  setStatus(data?.message || data?.error || 'Autentikasi belum berhasil.', 'warning')
  queueNextFrame(1200)
}

function queueNextFrame(delay = 1200) {
  if (!state.cameraActive) return
  cancelFrameLoop()

  if (state.transport === 'webrtc') {
    if (delay > 0) {
      loopMode.value = 'timeout'
      loopHandle.value = window.setTimeout(() => sendFrameLoop(), delay)
    } else {
      loopMode.value = 'raf'
      loopHandle.value = requestAnimationFrame(() => sendFrameLoop())
    }
  } else {
    loopMode.value = 'timeout'
    loopHandle.value = window.setTimeout(() => sendFrameLoop(), delay)
  }
}

function cancelFrameLoop() {
  if (loopHandle.value == null) return
  if (loopMode.value === 'raf') {
    cancelAnimationFrame(loopHandle.value)
  } else {
    clearTimeout(loopHandle.value)
  }
  loopHandle.value = null
  loopMode.value = null
}

function stopCapture() {
  cancelFrameLoop()
  state.cameraActive = false
  cameraRef.value?.stop()
  closeWebRTC()
  state.sessionToken = null
}

function cancel() {
  stopCapture()
  state.framesProcessed = 0
  state.blinkCount = 0
  setStatus('', 'info')
  emit('cancel')
}

function toggleTransport() {
  if (state.cameraActive || state.loading) return
  state.transport = state.transport === 'webrtc' ? 'http' : 'webrtc'
  state.sessionToken = null
  setStatus(
    state.transport === 'webrtc'
      ? 'Mode WebRTC diaktifkan untuk streaming kamera.'
      : 'Mode HTTP fallback diaktifkan.',
    'info'
  )
}

async function finalizeFaceLogin(payload) {
  stopCapture()
  state.loading = true
  setStatus('Mengambil token login wajah...', 'info')
  try {
    const result = payload?.result || payload
    const accessToken = payload?.access_token || payload?.access
    const refreshToken = payload?.refresh_token || payload?.refresh

    if (!accessToken || !refreshToken) {
      throw new Error('Server tidak mengembalikan token login facial')
    }

    authStore.setTokens({ access: accessToken, refresh: refreshToken })
    const profile = await authStore.fetchProfile().catch(() => null)

    state.sessionToken = null

    emit('completed', {
      accessToken,
      refreshToken,
      user: payload?.user || result?.user || profile || null
    })
  } catch (error) {
    const message = extractErrorMessage(error, 'Gagal mengambil token login.')
    setStatus(message, 'danger')
  } finally {
    state.loading = false
  }
}

function handleWebRTCResult(result, message) {
  console.debug('[FaceLoginInline] WebRTC frame result', { result, message })
  if (!result) return
  state.framesProcessed = message?.frames_processed ?? state.framesProcessed + 1
  if (typeof result?.liveness_data?.blinks_detected === 'number') {
    state.blinkCount = result.liveness_data.blinks_detected
  }

  if (result.requires_more_frames) {
    setStatus(result.message || 'Lanjutkan streaming untuk memenuhi persyaratan liveness.', 'warning')
  } else if (result.error && !result.success) {
    setStatus(result.message || result.error, 'warning')
  } else if (result.success) {
    setStatus('Autentikasi terverifikasi, menunggu token...', 'success')
  }
}

function handleWebRTCFinal(message) {
  console.debug('[FaceLoginInline] WebRTC session final message', message)
  const result = message?.result || {}
  if (typeof message?.frames_processed === 'number') {
    state.framesProcessed = message.frames_processed
  }
  if (typeof result?.liveness_data?.blinks_detected === 'number') {
    state.blinkCount = result.liveness_data.blinks_detected
  }

  if (result?.success) {
    setStatus('Autentikasi berhasil. Mengambil token...', 'success')
    finalizeFaceLogin({
      ...message,
      access_token: message?.access_token || message?.access,
      refresh_token: message?.refresh_token || message?.refresh,
      user: message?.user
    })
  } else {
    setStatus(message?.error || result?.message || 'Autentikasi gagal.', 'danger')
    stopCapture()
    state.sessionToken = null
    emit('cancel')
  }
}

function handleWebRTCError(error) {
  console.error('[FaceLoginInline] WebRTC error', error)
  const message = typeof error?.message === 'string' ? error.message : 'Koneksi WebRTC mengalami kendala.'
  setStatus(message, 'danger')
}

function handleCameraError(error) {
  console.error('Inline face login camera error', error)
  state.cameraActive = false
  const message = typeof error?.message === 'string' ? error.message : 'Kamera mengalami masalah.'
  setStatus(message, 'danger')
}

onBeforeUnmount(() => {
  stopCapture()
})
</script>
