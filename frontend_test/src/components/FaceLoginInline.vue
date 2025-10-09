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
        <button type="button" @click="startCapture" :disabled="cameraActive">
          {{ cameraActive ? 'Kamera Aktif' : 'Mulai Kamera' }}
        </button>
        <button type="button" class="secondary" @click="stopCapture" :disabled="!cameraActive">
          Stop
        </button>
        <button type="button" class="danger" @click="cancel">
          Batal
        </button>
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
  sessionType: 'verification',
  cameraActive: false,
  framesProcessed: 0,
  blinkCount: 0,
  statusMessage: '',
  statusLevel: 'info',
  loading: false,
  minFrames: 3,
  requiredBlinks: 2
})

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

watch(
  () => props.email,
  () => {
    state.sessionToken = null
    state.framesProcessed = 0
    state.blinkCount = 0
    if (!props.email) {
      setStatus('Masukkan email sebelum memulai face login.', 'warning')
    }
  }
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

async function ensureSession() {
  if (state.sessionToken) return state.sessionToken
  if (!props.email) {
    setStatus('Masukkan email akun untuk memulai face login.', 'danger')
    throw new Error('Email required for face login')
  }
  state.loading = true
  setStatus('Menyiapkan session autentikasi...', 'info')
  try {
    const response = await faceAuthApi.createPublicSession({
      session_type: state.sessionType,
      email: props.email,
      device_info: {
        device_id: 'face-login-inline',
        device_name: 'Inline Face Login',
        device_type: 'web'
      }
    })
    state.sessionToken = response.data.session_token
    setStatus('Session siap. Pastikan wajah berada di tengah frame dan kedipkan mata beberapa kali.', 'info')
    return state.sessionToken
  } catch (error) {
    const message = extractErrorMessage(error, 'Gagal membuat session autentikasi.')
    setStatus(message, 'danger')
    console.error('Failed to create face login session', error)
    throw error
  } finally {
    state.loading = false
  }
}

async function startCapture() {
  if (state.loading) return
  try {
    await ensureSession()
    if (!cameraActive.value) {
      await cameraRef.value?.start()
      setStatus('Kamera aktif. Kedipkan mata dan tahan beberapa detik.', 'warning')
    }
    state.cameraActive = true
    state.framesProcessed = 0
    state.blinkCount = 0
    sendFrameLoop()
  } catch (error) {
    const message = extractErrorMessage(error, 'Tidak dapat memulai kamera atau session.')
    setStatus(message, 'danger')
    console.error('Face login start capture failed', error)
  }
}

function stopCapture() {
  state.cameraActive = false
  cameraRef.value?.stop()
}

function cancel() {
  stopCapture()
  state.sessionToken = null
  emit('cancel')
}

async function sendFrameLoop() {
  if (!state.cameraActive || !state.sessionToken) return
  try {
    const frame = await cameraRef.value?.captureFrame()
    if (!frame) {
      throw new Error('Frame tidak tersedia')
    }

    const response = await faceAuthApi.processFrame({
      session_token: state.sessionToken,
      frame_data: frame
    })

    state.framesProcessed = response.data?.frames_processed ?? state.framesProcessed + 1
    state.blinkCount = response.data?.liveness_blinks ?? state.blinkCount

    if (response.data?.requires_more_frames) {
      setStatus(response.data.message || 'Lanjutkan streaming untuk verifikasi.', 'warning')
      queueNextFrame()
      return
    }

    if (response.data?.success) {
      setStatus('Autentikasi berhasil. Mengambil token...', 'success')
      await finalizeFaceLogin(response.data)
      return
    }

    setStatus(response.data?.message || response.data?.error || 'Autentikasi gagal.', 'danger')
    if (response.data?.requires_new_session) {
      setStatus('Session perlu direset. Menghentikan kamera.', 'danger')
      stopCapture()
      emit('cancel')
    } else {
      queueNextFrame()
    }
  } catch (error) {
    const message = extractErrorMessage(error, 'Terjadi kesalahan saat mengirim frame.')
    setStatus(message, 'danger')
    console.error('Face login inline frame error', error)
    queueNextFrame()
  }
}

function queueNextFrame() {
  if (!state.cameraActive) return
  setTimeout(sendFrameLoop, 1200)
}

async function finalizeFaceLogin(responseData) {
  try {
    stopCapture()
    state.loading = true
    setStatus('Mengambil token login wajah...', 'info')

    const authUser = responseData.user
    const access = responseData.access_token
    const refresh = responseData.refresh_token

    if (!authUser?.email || !access || !refresh) {
      throw new Error('Server tidak mengembalikan token login facial')
    }

    authStore.setTokens({ access, refresh })
    await authStore.fetchProfile()

    state.sessionToken = null

    emit('completed', {
      accessToken: access,
      refreshToken: refresh,
      user: authUser
    })
  } catch (error) {
    console.error('Face login finalize failed', error)
    setStatus(error?.response?.data?.detail || error?.message || 'Gagal mengambil token login.', 'danger')
  } finally {
    state.loading = false
  }
}

function handleCameraError(error) {
  console.error('Inline face login camera error', error)
  state.cameraActive = false
  setStatus('Kamera error: ' + (error?.message || error), 'danger')
}

onBeforeUnmount(() => {
  stopCapture()
})
</script>
