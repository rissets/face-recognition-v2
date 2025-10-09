<template>
  <div class="layout-grid">
    <section class="section">
      <header>
        <h2>Login via Face Recognition</h2>
        <p>
          Buat session autentikasi, hidupkan kamera, dan biarkan minimal tiga frame terkirim. Pastikan melakukan kedipan
          mata (idealnya dua kali) agar pemeriksaan liveness terpenuhi sebelum verifikasi dinyatakan berhasil.
        </p>
      </header>
      <div class="layout-split">
        <div>
          <CameraStream
            ref="cameraRef"
            @started="onCameraStarted"
            @stopped="onCameraStopped"
            @error="handleCameraError"
          >
            <template #overlay>
              <div class="camera-overlay-hint" v-if="!isCameraActive">
                <p>Aktifkan kamera & mulai session</p>
              </div>
            </template>
          </CameraStream>

          <div class="pill-group">
            <span class="status-pill" :class="isCameraActive ? 'success' : 'danger'">
              Kamera {{ isCameraActive ? 'Aktif' : 'Off' }}
            </span>
            <span class="status-pill" :class="streamingState.capturing ? 'success' : 'warning'">
              Streaming {{ streamingState.capturing ? 'Berjalan' : 'Idle' }}
            </span>
            <span class="status-pill" :class="lastResult?.success ? 'success' : 'warning'">
              Hasil {{ lastResult ? (lastResult.success ? 'Match' : 'Belum') : 'N/A' }}
            </span>
          </div>

          <div class="actions">
            <button type="button" :disabled="loading.camera || isCameraActive" @click="startCamera">
              {{ loading.camera ? 'Mengaktifkan...' : 'Aktifkan Kamera' }}
            </button>
            <button type="button" class="secondary" :disabled="!isCameraActive" @click="stopCamera">
              Matikan Kamera
            </button>
            <button
              type="button"
              :disabled="!sessionToken || !isCameraActive"
              @click="toggleStreaming"
            >
              {{ streamingState.capturing ? 'Stop Capture' : 'Mulai Face Login' }}
            </button>
          </div>

          <div v-if="errors.session" class="status-error">{{ errors.session }}</div>
          <div v-if="errors.camera" class="status-error">{{ errors.camera }}</div>

          <div v-if="logs.length" class="log-panel">
            <div v-for="log in logs" :key="log.id" class="log-entry">
              <span>{{ log.time }} · {{ log.level.toUpperCase() }}</span>
              <div>{{ log.message }}</div>
              <div v-if="log.detail" class="log-detail">{{ log.detail }}</div>
            </div>
          </div>
        </div>

        <div>
          <form class="form-grid" @submit.prevent="createSession">
            <div class="field">
              <label>Mode</label>
              <select v-model="settings.sessionType">
                <option value="identification">Identification</option>
                <option value="verification">Verification</option>
              </select>
            </div>
            <div class="field" v-if="settings.sessionType === 'verification'">
              <label>Email Target</label>
              <input v-model="settings.email" type="email" placeholder="user@example.com" required />
            </div>
            <div class="field">
              <label>Device Info (JSON)</label>
              <textarea v-model="deviceInfoInput" rows="6"></textarea>
            </div>
            <div class="actions">
              <button type="submit" :disabled="loading.session">
                {{ loading.session ? 'Menyiapkan...' : 'Buat Session' }}
              </button>
              <button type="button" class="secondary" @click="resetSession">Reset</button>
            </div>
          </form>

          <div v-if="authSession" class="layout-grid">
            <div class="session-summary">
              <div class="session-tile">
                <span>Jenis Session</span>
                <strong>{{ authSession.session_type }}</strong>
              </div>
              <div class="session-tile">
                <span>Frame Terkirim</span>
                <strong>{{ streamingState.framesSent }}</strong>
              </div>
              <div class="session-tile">
                <span>Similarity</span>
                <strong>{{ lastResult?.similarity_score?.toFixed(3) ?? '-' }}</strong>
              </div>
              <div class="session-tile">
                <span>Liveness</span>
                <strong>{{ lastLiveness }}</strong>
              </div>
            </div>
            <div class="info-card">
              <strong>Session Token</strong>
              <span class="mono">{{ authSession.session_token }}</span>
            </div>
            <div class="info-card" v-if="authSession.webrtc_config">
              <strong>WebRTC Config</strong>
              <span class="mono">{{ formatConfig(authSession.webrtc_config) }}</span>
            </div>
            <div class="info-card" v-if="recognizedUser">
              <strong>User Terverifikasi</strong>
              <span>{{ recognizedUser.full_name || recognizedUser.email }}</span>
              <span class="mono">{{ recognizedUser.email }}</span>
            </div>
            <div class="info-card" v-if="lastResult?.message">
              <strong>Pesan Sistem</strong>
              <span>{{ lastResult.message }}</span>
            </div>
          </div>
        </div>
      </div>
    </section>

    <section class="section">
      <h2>Mode Manual (Opsional)</h2>
      <p>Untuk pengujian gambar statis atau debugging spesifik alur autentikasi.</p>
      <form class="form-grid" @submit.prevent="processManualFrame">
        <div class="field">
          <label>Session Token</label>
          <input v-model="manualForm.sessionToken" placeholder="UUID session" />
        </div>
        <div class="field">
          <label>Frame Data (base64)</label>
          <textarea v-model="manualForm.frameData" rows="5" placeholder="data:image/jpeg;base64,..."></textarea>
        </div>
        <div class="field">
          <label>Upload Gambar</label>
          <input type="file" accept="image/*" @change="handleManualFile" />
        </div>
        <div class="actions">
          <button type="submit" :disabled="loading.manual">Kirim Frame Manual</button>
        </div>
      </form>
      <div v-if="errors.manual" class="status-error">{{ errors.manual }}</div>
    </section>
  </div>
</template>

<script setup>
import { computed, onBeforeUnmount, reactive, ref } from 'vue'
import CameraStream from '../components/CameraStream.vue'
import { faceAuthApi } from '../services/api'

const cameraRef = ref(null)
const authSession = ref(null)
const recognizedUser = ref(null)
const cameraActive = ref(false)

const settings = reactive({
  sessionType: 'identification',
  email: ''
})

const deviceInfoInput = ref(
  JSON.stringify(
    {
      device_id: 'web-face-login',
      device_name: 'Face Login Tester',
      device_type: 'web',
      browser: navigator.userAgent
    },
    null,
    2
  )
)

const loading = reactive({
  session: false,
  manual: false,
  camera: false
})

const errors = reactive({
  session: '',
  manual: '',
  camera: ''
})

const streamingState = reactive({
  capturing: false,
  framesSent: 0,
  lastResult: null
})

const logs = ref([])
const manualForm = reactive({
  sessionToken: '',
  frameData: ''
})

const captureIntervalMs = 1200
const captureTimer = ref(null)

const isCameraActive = computed(() => cameraActive.value)
const sessionToken = computed(() => authSession.value?.session_token || '')
const lastResult = computed(() => streamingState.lastResult)

const lastLiveness = computed(() => {
  const direct = streamingState.lastResult?.liveness_blinks
  if (typeof direct === 'number') {
    return `${direct} blinks`
  }

  const value = streamingState.lastResult?.liveness_data
  if (!value) return '-'
  if (typeof value === 'number') return value.toFixed(3)
  if (value.blinks_detected != null) {
    return `${value.blinks_detected} blinks`
  }
  return JSON.stringify(value)
})

function addLog(level, message, detail = '') {
  logs.value.unshift({
    id: `${Date.now()}-${Math.random()}`,
    level,
    message,
    detail: typeof detail === 'string' ? detail : detail ? JSON.stringify(detail, null, 2) : '',
    time: new Date().toLocaleTimeString()
  })
  if (logs.value.length > 40) {
    logs.value.pop()
  }
}

function clearTimer() {
  if (captureTimer.value) {
    clearTimeout(captureTimer.value)
    captureTimer.value = null
  }
}

function stopStreaming() {
  streamingState.capturing = false
  clearTimer()
}

async function startCamera() {
  if (isCameraActive.value) return
  loading.camera = true
  errors.camera = ''
  try {
    await cameraRef.value?.start()
    cameraActive.value = true
    addLog('info', 'Kamera aktif')
  } catch (error) {
    errors.camera = formatError(error)
    addLog('error', 'Gagal mengaktifkan kamera', errors.camera)
  } finally {
    loading.camera = false
  }
}

function stopCamera() {
  stopStreaming()
  cameraRef.value?.stop()
  cameraActive.value = false
  addLog('info', 'Kamera dimatikan')
}

function resetSession() {
  stopStreaming()
  authSession.value = null
  streamingState.framesSent = 0
  streamingState.lastResult = null
  recognizedUser.value = null
  manualForm.sessionToken = ''
  errors.session = ''
  addLog('info', 'Session autentikasi direset')
}

function onCameraStarted() {
  cameraActive.value = true
}

function onCameraStopped() {
  cameraActive.value = false
}

function parseDeviceInfo() {
  if (!deviceInfoInput.value) return {}
  try {
    return JSON.parse(deviceInfoInput.value)
  } catch (error) {
    throw new Error('Device info harus JSON valid')
  }
}

async function createSession() {
  loading.session = true
  errors.session = ''
  try {
    const payload = {
      session_type: settings.sessionType,
      device_info: parseDeviceInfo()
    }
    if (settings.sessionType === 'verification') {
      if (!settings.email) {
        throw new Error('Email wajib diisi untuk mode verification')
      }
      payload.email = settings.email
    }
    const response = await faceAuthApi.createSession(payload)
    authSession.value = {
      session_token: response.data.session_token,
      session_type: response.data.session_type,
      webrtc_config: response.data.webrtc_config
    }
    manualForm.sessionToken = authSession.value.session_token
    streamingState.framesSent = 0
    streamingState.lastResult = null
    recognizedUser.value = null
    addLog('success', 'Session autentikasi dibuat', response.data)
  } catch (error) {
    const raw = error.response?.data
    const message = formatError(error)

    if (raw?.session_token) {
      const token = raw.session_token
      const status = (raw.status || raw.session_status || '').toString().toLowerCase()
      const failureReason = raw.failure_reason || raw.message || message

      authSession.value = {
        session_token: token,
        session_type: settings.sessionType,
        webrtc_config: null
      }
      manualForm.sessionToken = token

      if (status.includes('failed') || raw?.requires_new_session) {
        errors.session = 'Session sebelumnya gagal. Tekan Reset untuk menghapus session lama, lalu buat session baru.'
        addLog('warning', 'Session gagal', { session_token: token, reason: failureReason })
      } else if (status.includes('active')) {
        errors.session = 'Session autentikasi masih aktif. Gunakan token tersebut atau lanjutkan proses.'
        addLog('warning', 'Session aktif', { session_token: token })
      } else {
        errors.session = failureReason
        addLog('warning', 'Session existing', { session_token: token, detail: failureReason })
      }
    } else {
      errors.session = message
      addLog('error', 'Gagal membuat session', message)
    }
  } finally {
    loading.session = false
  }
}

async function toggleStreaming() {
  errors.session = ''
  if (streamingState.capturing) {
    stopStreaming()
    addLog('info', 'Streaming dihentikan')
    return
  }

  if (!authSession.value) {
    errors.session = 'Buat session autentikasi terlebih dahulu.'
    return
  }

  if (!isCameraActive.value) {
    await startCamera()
    if (!isCameraActive.value) {
      errors.session = 'Kamera belum aktif.'
      return
    }
  }

  streamingState.capturing = true
  addLog('info', 'Mulai mengirim frame', { interval_ms: captureIntervalMs })
  captureNextFrame()
}

async function captureNextFrame() {
  clearTimer()
  if (!streamingState.capturing) return

  try {
    const frame = await cameraRef.value?.captureFrame()
    await sendFrame(frame)
  } catch (error) {
    const message = formatError(error)
    errors.session = message
    addLog('error', 'Gagal menangkap/kirim frame', message)
    stopStreaming()
    return
  }

  if (streamingState.capturing) {
    captureTimer.value = setTimeout(captureNextFrame, captureIntervalMs)
  }
}

async function sendFrame(frameData, overrideToken) {
  const token = overrideToken || authSession.value?.session_token
  if (!token) {
    throw new Error('Session token tidak tersedia')
  }

  try {
    const response = await faceAuthApi.processFrame({
      session_token: token,
      frame_data: frameData
    })

    streamingState.framesSent += 1
    const data = response.data
    streamingState.lastResult = data

    if (data.requires_more_frames) {
      errors.session = ''
      addLog('info', data.message || 'Lanjutkan streaming untuk verifikasi.', JSON.stringify(data, null, 2))
      return data
    }

    if (data.success) {
      recognizedUser.value = data.user || null
      errors.session = ''
      addLog('success', 'Autentikasi berhasil', JSON.stringify(data, null, 2))
      stopStreaming()
    } else {
      recognizedUser.value = null
      const failureMsg = data.message || data.error || 'Autentikasi gagal'
      errors.session = failureMsg
      addLog('warning', failureMsg, JSON.stringify(data, null, 2))
      if (data.session_finalized) {
        stopStreaming()
      }
      if (data.requires_new_session) {
        addLog('warning', 'Session perlu dibuat ulang. Tekan Reset lalu buat session baru.')
      }
    }

    return data
  } catch (error) {
    const message = formatError(error)
    if (!overrideToken) {
      errors.session = message
    } else {
      errors.manual = message
    }
    addLog('error', 'Frame gagal diproses', message)
    throw error
  }
}

async function processManualFrame() {
  loading.manual = true
  errors.manual = ''
  try {
    if (!manualForm.sessionToken) {
      throw new Error('Session token wajib diisi')
    }
    if (!manualForm.frameData) {
      throw new Error('Frame data wajib diisi')
    }
    await sendFrame(manualForm.frameData, manualForm.sessionToken)
    manualForm.frameData = ''
  } catch (error) {
    errors.manual = formatError(error)
  } finally {
    loading.manual = false
  }
}

function handleManualFile(event) {
  const [file] = event.target.files || []
  if (!file) return
  const reader = new FileReader()
  reader.onload = () => {
    manualForm.frameData = reader.result
  }
  reader.readAsDataURL(file)
}

function handleCameraError(error) {
  errors.camera = formatError(error)
  cameraActive.value = false
  addLog('error', 'Kamera error', errors.camera)
}

function formatConfig(config) {
  try {
    return JSON.stringify(config, null, 2)
  } catch (error) {
    return String(config)
  }
}

function formatError(error) {
  if (!error) return 'Unknown error'
  if (typeof error === 'string') return error
  if (error.response?.data) {
    return JSON.stringify(error.response.data, null, 2)
  }
  if (error.message) return error.message
  return String(error)
}

onBeforeUnmount(() => {
  stopStreaming()
  cameraRef.value?.stop()
})
</script>
