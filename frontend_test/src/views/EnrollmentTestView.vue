<template>
  <div class="layout-grid">
    <section class="section">
      <header>
        <h2>Enrollment Wajah via Streaming</h2>
        <p>Aktifkan kamera, buat session enrollment, lalu kirim sampel secara otomatis sampai target terpenuhi.</p>
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
                <p>Aktifkan kamera untuk memulai.</p>
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
            <span class="status-pill success">
              Sampel {{ completedSamples }} / {{ targetSamplesValue }}
            </span>
          </div>

          <div class="actions">
            <button type="button" :disabled="loading.camera || isCameraActive" @click="startCamera">
              {{ loading.camera ? 'Mengaktifkan...' : 'Aktifkan Kamera' }}
            </button>
            <button
              type="button"
              class="secondary"
              :disabled="!isCameraActive"
              @click="stopCamera"
            >
              Matikan Kamera
            </button>
            <button
              type="button"
              :disabled="!sessionToken || !isCameraActive"
              @click="toggleStreaming"
            >
              {{ streamingState.capturing ? 'Stop Capture' : 'Mulai Capture' }}
            </button>
          </div>
          <div v-if="errors.camera" class="status-error">{{ errors.camera }}</div>
          <div v-if="errors.session" class="status-error">{{ errors.session }}</div>

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
              <label>Target Samples</label>
              <input v-model.number="settings.targetSamples" type="number" min="3" max="10" required />
            </div>
            <div class="field">
              <label>Device Info (JSON)</label>
              <textarea v-model="deviceInfoInput" rows="6"></textarea>
            </div>
            <div class="actions">
              <button type="submit" :disabled="loading.create">
                {{ loading.create ? 'Membuat...' : 'Buat Session' }}
              </button>
              <button type="button" class="secondary" @click="resetSession">Reset</button>
            </div>
          </form>

          <div v-if="session" class="layout-grid">
            <div class="session-summary">
              <div class="session-tile">
                <span>Status</span>
                <strong>{{ sessionStatus }}</strong>
              </div>
              <div class="session-tile">
                <span>Progress</span>
                <strong>{{ progressPercent }}%</strong>
              </div>
              <div class="session-tile">
                <span>Frame Terkirim</span>
                <strong>{{ streamingState.framesSent }}</strong>
              </div>
              <div class="session-tile">
                <span>Quality Terakhir</span>
                <strong>{{ lastQuality ?? '-' }}</strong>
              </div>
            </div>
            <div class="progress-wrapper">
              <div class="progress-track">
                <div class="progress-fill" :style="{ width: progressPercent + '%' }"></div>
              </div>
              <small>
                {{ completedSamples }} dari {{ targetSamplesValue }} sampel · Expires: {{ session?.expires_at || '-' }}
              </small>
            </div>
            <div class="info-card">
              <strong>Session Token</strong>
              <span class="mono">{{ session.session_token }}</span>
            </div>
            <div v-if="lastMessage" class="info-card">
              <strong>Pesan Terakhir</strong>
              <span>{{ lastMessage }}</span>
            </div>
          </div>
        </div>
      </div>
    </section>

    <section class="section">
      <h2>Mode Manual (Opsional)</h2>
      <p>Gunakan mode ini bila ingin menguji dengan gambar statis atau debugging.</p>
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
import { enrollmentApi } from '../services/api'

const cameraRef = ref(null)
const session = ref(null)
const cameraActive = ref(false)
const settings = reactive({
  targetSamples: 5
})
const deviceInfoInput = ref(
  JSON.stringify(
    {
      device_id: 'web-enrollment-tester',
      device_name: 'Enrollment Tester',
      device_type: 'web',
      browser: navigator.userAgent
    },
    null,
    2
  )
)

const loading = reactive({
  create: false,
  manual: false,
  camera: false
})

const errors = reactive({
  camera: '',
  session: '',
  manual: ''
})

const streamingState = reactive({
  capturing: false,
  framesSent: 0,
  lastResponse: null
})

const logs = ref([])
const captureIntervalMs = 1400
const captureTimer = ref(null)

const manualForm = reactive({
  sessionToken: '',
  frameData: ''
})

const isCameraActive = computed(() => cameraActive.value)

const completedSamples = computed(() => {
  return (
    streamingState.lastResponse?.completed_samples ??
    session.value?.completed_samples ??
    0
  )
})

const targetSamplesValue = computed(() => {
  return Number(session.value?.target_samples ?? settings.targetSamples ?? 0)
})

const progressPercent = computed(() => {
  if (!targetSamplesValue.value) return 0
  return Math.min(100, Math.round((completedSamples.value / targetSamplesValue.value) * 100))
})

const sessionToken = computed(() => session.value?.session_token || '')

const sessionStatus = computed(() => {
  return streamingState.lastResponse?.session_status ?? session.value?.status ?? 'pending'
})

const lastQuality = computed(() => streamingState.lastResponse?.quality_score ?? null)
const lastMessage = computed(() => streamingState.lastResponse?.message ?? '')

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
  session.value = null
  streamingState.framesSent = 0
  streamingState.lastResponse = null
  manualForm.sessionToken = ''
  errors.session = ''
  errors.manual = ''
  errors.camera = ''
  addLog('info', 'Session direset')
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
  loading.create = true
  errors.session = ''
  try {
    const payload = {
      target_samples: settings.targetSamples,
      device_info: parseDeviceInfo()
    }
    const response = await enrollmentApi.createSession(payload)
    session.value = {
      ...response.data,
      target_samples: response.data.target_samples ?? settings.targetSamples,
      completed_samples: 0,
      status: 'pending'
    }
    manualForm.sessionToken = session.value.session_token
    streamingState.framesSent = 0
    streamingState.lastResponse = null
    addLog('success', 'Session enrollment dibuat', response.data)
  } catch (error) {
    const raw = error.response?.data
    const message = formatError(error)

    if (raw?.session_token && typeof raw.session_token === 'string') {
      const token = raw.session_token
      const status = (raw.session_status || raw.error || '').toString().toLowerCase()
      const detailMessage = raw.error || message

      session.value = {
        session_token: token,
        status: raw.session_status || 'pending',
        target_samples: raw.target_samples || settings.targetSamples,
        completed_samples: raw.completed_samples ?? session.value?.completed_samples ?? 0
      }
      manualForm.sessionToken = token

      if (status.includes('failed')) {
        errors.session = 'Sesi enrollment sebelumnya berstatus gagal. Tekan Reset untuk menghapus session lama sebelum membuat session baru.'
        addLog('warning', 'Session ditemukan namun gagal', { session_token: token, detail: detailMessage })
      } else if (status.includes('active')) {
        errors.session = 'Session enrollment masih aktif. Gunakan session token yang sudah ada.'
        addLog('warning', 'Session sudah aktif', { session_token: token })
      } else {
        errors.session = detailMessage
        addLog('warning', 'Session existing', { session_token: token, detail: detailMessage })
      }
    } else {
      errors.session = message
      addLog('error', 'Gagal membuat session', message)
    }
  } finally {
    loading.create = false
  }
}

async function toggleStreaming() {
  errors.session = ''
  if (streamingState.capturing) {
    stopStreaming()
    addLog('info', 'Streaming dihentikan')
    return
  }

  if (!session.value) {
    errors.session = 'Buat session enrollment terlebih dahulu.'
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
  if (!streamingState.capturing) {
    return
  }

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
  const token = overrideToken || session.value?.session_token
  if (!token) {
    throw new Error('Session token tidak tersedia')
  }

  try {
    const response = await enrollmentApi.processFrame({
      session_token: token,
      frame_data: frameData
    })

    streamingState.framesSent += 1
    const data = response.data
    streamingState.lastResponse = data

    if (session.value && session.value.session_token === token) {
      session.value.status = data.session_status ?? session.value.status
      session.value.completed_samples = data.completed_samples ?? session.value.completed_samples ?? 0
      session.value.target_samples = data.target_samples ?? session.value.target_samples ?? settings.targetSamples
    }

    if (data.success === false) {
      addLog('warning', data.error || 'Frame ditolak', JSON.stringify(data, null, 2))
      return data
    }

    addLog('success', 'Frame diterima', `samples: ${data.completed_samples}/${session.value?.target_samples ?? settings.targetSamples}`)

    const target = session.value?.target_samples ?? settings.targetSamples
    if (
      (data.session_status && !['pending', 'in_progress'].includes(data.session_status)) ||
      (typeof data.completed_samples === 'number' && data.completed_samples >= target)
    ) {
      addLog('success', 'Enrollment selesai', data.message || 'Target sampel tercapai')
      stopStreaming()
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
