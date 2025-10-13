<template>
  <div class="layout-grid">
    <section class="section">
      <header>
        <h2>Enrollment Wajah via Streaming</h2>
        <div class="guide-section">
          <h3>📋 Panduan Enrollment:</h3>
          <ol>
            <li><strong>Isi External User ID</strong> - Masukkan ID unik untuk user (contoh: john.doe)</li>
            <li><strong>Aktifkan Kamera</strong> - Pastikan wajah terlihat jelas dengan pencahayaan yang baik</li>
            <li><strong>Buat Session</strong> - Sistem akan mulai mengumpulkan sampel wajah</li>
            <li><strong>Mulai Streaming</strong> - Sistem akan menganalisis setiap frame:
              <ul>
                <li>💚 <strong>Lakukan kedipan mata</strong> untuk liveness detection</li>
                <li>🔄 <strong>Gerakkan kepala sedikit</strong> untuk variasi pose</li>
                <li>✋ <strong>Jangan tutup wajah</strong> dengan tangan atau benda lain</li>
                <li>💡 <strong>Pastikan pencahayaan baik</strong> untuk kualitas optimal</li>
              </ul>
            </li>
            <li><strong>Tunggu Completion</strong> - Sistem akan otomatis selesai setelah cukup sampel berkualitas</li>
          </ol>
          <div class="tips">
            <strong>💡 Tips:</strong> Jika progress 100% tapi belum selesai, pastikan sudah kedip mata minimal 1x atau gerakkan kepala sedikit.
          </div>
        </div>
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
              <label>External User ID</label>
              <input
                v-model="settings.userId"
                placeholder="john.doe"
                required
              />
            </div>
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
              <span>Liveness Score</span>
              <strong>{{ (lastLivenessScore || 0).toFixed(2) }}</strong>
              <small v-if="lastLivenessHint">{{ lastLivenessHint }}</small>
              <span class="status-pill" :class="livenessVerified ? 'success' : 'warning'">
                {{ livenessVerified ? 'Verified' : 'Pending' }}
              </span>
            </div>
            <div class="session-tile">
              <span>Quality Terakhir</span>
              <strong>{{ lastQuality ?? '-' }}</strong>
            </div>
            <div class="session-tile" v-if="canComplete">
              <span>Status Enrollment</span>
              <span class="status-pill success">Ready to Complete</span>
            </div>
            <div class="session-tile" v-if="obstaclesDetected.length > 0">
              <span>Obstacles Detected</span>
              <div class="obstacle-list">
                <span v-for="obstacle in obstaclesDetected" :key="obstacle" class="status-pill danger">
                  {{ obstacle }}
                </span>
              </div>
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
            <div class="info-card" v-if="streamingState.lastPreview">
              <strong>Preview Wajah Terakhir</strong>
              <img :src="streamingState.lastPreview" alt="Face preview" class="preview-face" />
            </div>
            <div class="info-card" v-if="sessionFeedback">
              <strong>Session Feedback</strong>
              <span>{{ sessionFeedback }}</span>
            </div>
            <div class="info-card" v-if="lastMessage && lastMessage !== sessionFeedback">
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
import { sessionApi } from '../services/api'

const cameraRef = ref(null)
const session = ref(null)
const cameraActive = ref(false)
const settings = reactive({
  userId: '',
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
  lastResponse: null,
  lastLivenessScore: 0,
  lastLivenessData: null,
  lastPreview: null
})

const logs = ref([])
const captureIntervalMs = 1400
const captureTimer = ref(null)

const manualForm = reactive({
  sessionToken: '',
  frameData: '',
  file: null
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
const sessionFeedback = computed(() => streamingState.lastResponse?.session_feedback ?? '')
const lastLivenessScore = computed(() => streamingState.lastLivenessScore)
const livenessVerified = computed(() => streamingState.lastResponse?.liveness_verified ?? false)
const obstaclesDetected = computed(() => streamingState.lastResponse?.obstacles ?? [])
const canComplete = computed(() => streamingState.lastResponse?.can_complete_enrollment ?? false)
const enrollmentProgress = computed(() => streamingState.lastResponse?.enrollment_progress ?? 0)
const lastLivenessHint = computed(() => {
  const info = streamingState.lastLivenessData
  if (!info) return ''
  const blinks = info.blinks_detected ?? info.total_blinks ?? 0
  const motion = info.motion_events ?? info.liveness_motion_events ?? 0
  return `Blink ${blinks}x · Motion ${motion}x`
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
      user_id: settings.userId,
      session_type: 'webcam',
      metadata: {
        target_samples: settings.targetSamples,
        device_info: parseDeviceInfo()
      }
    }
    const response = await sessionApi.createEnrollment(payload)
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

function ensureBlobFromFrame(frameData) {
  if (frameData instanceof File || frameData instanceof Blob) {
    return frameData
  }
  if (typeof frameData === 'string') {
    if (!frameData.startsWith('data:')) {
      throw new Error('Frame data harus berupa data URL base64')
    }
    const [header, data] = frameData.split(',')
    const mime = header.match(/:(.*?);/)?.[1] || 'image/jpeg'
    const binary = atob(data)
    const length = binary.length
    const buffer = new Uint8Array(length)
    for (let i = 0; i < length; i += 1) {
      buffer[i] = binary.charCodeAt(i)
    }
    return new Blob([buffer], { type: mime })
  }
  throw new Error('Format frame tidak didukung')
}

async function sendFrame(frameData, overrideToken) {
  const token = overrideToken || session.value?.session_token
  if (!token) {
    throw new Error('Session token tidak tersedia')
  }

  try {
    const blob = ensureBlobFromFrame(frameData)
    const formData = new FormData()
    formData.append('session_token', token)
    formData.append('image', blob, `frame-${streamingState.framesSent + 1}.jpg`)
    formData.append('frame_number', streamingState.framesSent + 1)

    const response = await sessionApi.processFrame(formData)

    streamingState.framesSent += 1
    const data = response.data || {}
    streamingState.lastResponse = data
    streamingState.lastLivenessScore = data.liveness_score ?? streamingState.lastLivenessScore
    streamingState.lastLivenessData = data.liveness_data ?? data.last_liveness ?? streamingState.lastLivenessData
    streamingState.lastPreview = data.preview_image || streamingState.lastPreview

    // Enhanced session-based response handling
    const framesProcessed = data.completed_samples ?? data.frames_processed ?? streamingState.framesSent
    const targetSamples = data.target_samples ?? session.value?.target_samples ?? settings.targetSamples
    const sessionStatus = data.session_status ?? data.status ?? session.value?.status ?? 'in_progress'
    const requiresMore = data.requires_more_frames === true
    const progressPercentage = data.enrollment_progress ?? ((framesProcessed / targetSamples) * 100)
    const sessionFeedback = data.session_feedback || data.message || ''

    if (session.value && session.value.session_token === token) {
      session.value.status = sessionStatus
      session.value.completed_samples = framesProcessed
      session.value.target_samples = targetSamples
    }

    if (data.error) {
      addLog('warning', data.error, JSON.stringify(data, null, 2))
      return data
    }

    // Enhanced feedback with session-based information
    const successState = data.success === true ? 'success' : 'info'
    const logMessage = sessionFeedback || 'Frame diterima'
    addLog(successState, logMessage, {
      frames: `${framesProcessed}/${targetSamples}`,
      progress: `${progressPercentage.toFixed(1)}%`,
      liveness_score: data.liveness_score,
      liveness_verified: data.liveness_verified,
      can_complete: data.can_complete_enrollment,
      requires_more_frames: data.requires_more_frames,
      obstacles: data.obstacles || []
    })

    if (sessionStatus === 'failed') {
      addLog('warning', data.message || 'Sesi enrollment gagal.', JSON.stringify(data, null, 2))
      stopStreaming()
      return data
    }

    if (sessionStatus === 'completed' && !requiresMore) {
      addLog('success', 'Enrollment selesai', {
        message: sessionFeedback || data.message || 'Target sampel tercapai dengan embedding averaging',
        liveness_score: data.liveness_score,
        liveness_verified: data.liveness_verified,
        enrolled_user_id: data.enrolled_user_id,
        quality_score: data.quality_score
      })
      stopStreaming()
    }

    return data
 } catch (error) {
   const message = formatError(error)
    if (message.includes('Session is no longer active')) {
      addLog('info', 'Session telah selesai diproses.')
      stopStreaming()
      return
    }
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
    const payload = manualForm.file || manualForm.frameData
    if (!payload) {
      throw new Error('Frame data wajib diisi')
    }
    await sendFrame(payload, manualForm.sessionToken)
    manualForm.frameData = ''
    manualForm.file = null
  } catch (error) {
    errors.manual = formatError(error)
  } finally {
    loading.manual = false
  }
}

function handleManualFile(event) {
  const [file] = event.target.files || []
  if (!file) return

  manualForm.file = file
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

<style scoped>
.obstacle-list {
  display: flex;
  flex-wrap: wrap;
  gap: 0.25rem;
  margin-top: 0.25rem;
}

.obstacle-list .status-pill {
  font-size: 0.75rem;
  padding: 0.125rem 0.5rem;
}

.guide-section {
  background: #f8fafc;
  border: 1px solid #e2e8f0;
  border-radius: 0.5rem;
  padding: 1rem;
  margin: 1rem 0;
}

.guide-section h3 {
  margin-top: 0;
  color: #1e293b;
  font-size: 1.1rem;
}

.guide-section ol {
  margin: 0.5rem 0;
  padding-left: 1.5rem;
}

.guide-section ol li {
  margin-bottom: 0.5rem;
}

.guide-section ul {
  margin: 0.25rem 0;
  padding-left: 1rem;
}

.guide-section ul li {
  margin-bottom: 0.25rem;
}

.tips {
  background: #dbeafe;
  border: 1px solid #3b82f6;
  border-radius: 0.25rem;
  padding: 0.5rem;
  margin-top: 1rem;
  font-size: 0.9rem;
}
</style>
