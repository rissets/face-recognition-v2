<template>
  <div class="layout-grid">
    <section class="section">
      <header>
        <h2>Login via Face Recognition</h2>
        <div class="guide-section">
          <h3>🔐 Panduan Authentication:</h3>
          <ol>
            <li><strong>Pilih Mode Authentication:</strong>
              <ul>
                <li><strong>Identification</strong> - Sistem akan mengenali siapa Anda dari database</li>
                <li><strong>Verification</strong> - Verifikasi identitas dengan memasukkan External User ID</li>
              </ul>
            </li>
            <li><strong>Aktifkan Kamera</strong> - Pastikan wajah terlihat jelas</li>
            <li><strong>Buat Session</strong> - Sistem siap untuk autentikasi</li>
            <li><strong>Mulai Face Login</strong> - Sistem akan memverifikasi identitas:
              <ul>
                <li>👁️ <strong>Kedipkan mata</strong> untuk membuktikan Anda manusia</li>
                <li>📸 <strong>Tatap kamera langsung</strong> dengan wajah jelas</li>
                <li>🚫 <strong>Hindari obstacle</strong> - jangan tutup wajah dengan tangan/benda</li>
                <li>⏱️ <strong>Tunggu beberapa detik</strong> untuk proses verifikasi</li>
              </ul>
            </li>
          </ol>
          <div class="tips">
            <strong>💡 Tips:</strong> Pastikan Anda sudah terdaftar di sistem (sudah melakukan enrollment) sebelum mencoba login.
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
              <label>Mode Autentikasi</label>
              <select v-model="settings.mode">
                <option value="identification">Identification (tanpa target)</option>
                <option value="verification">Verification (butuh external user)</option>
              </select>
            </div>
            <div class="field" v-if="settings.mode === 'verification'">
              <label>External User ID</label>
              <input v-model="settings.userId" placeholder="john.doe" required />
            </div>
            <div class="field">
              <label>
                <input type="checkbox" v-model="settings.requireLiveness" />
                Require liveness check
              </label>
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
                <span class="status-pill" :class="livenessVerified ? 'success' : 'warning'">
                  {{ livenessVerified ? 'Verified' : 'Pending' }}
                </span>
              </div>
              <div class="session-tile" v-if="obstaclesDetected.length > 0">
                <span>Obstacles</span>
                <div class="obstacle-list">
                  <span v-for="obstacle in obstaclesDetected" :key="obstacle" class="status-pill danger">
                    {{ obstacle }}
                  </span>
                </div>
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
              <span>{{ recognizedUser.display_name || recognizedUser.external_user_id }}</span>
              <span class="mono">ID: {{ recognizedUser.id }}</span>
              <span class="mono">External ID: {{ recognizedUser.external_user_id }}</span>
              <span class="mono">Enrolled: {{ recognizedUser.is_enrolled ? 'Yes' : 'No' }}</span>
            </div>
            <div class="info-card" v-if="sessionFeedback">
              <strong>Session Feedback</strong>
              <span>{{ sessionFeedback }}</span>
            </div>
            <div class="info-card" v-if="lastResult?.message && lastResult.message !== sessionFeedback">
              <strong>Pesan Sistem</strong>
              <span>{{ lastResult.message }}</span>
            </div>
            <div class="info-card" v-if="lastResult?.authentication_metadata">
              <strong>Authentication Info</strong>
              <span>Algorithm: {{ lastResult.authentication_metadata.algorithm_used }}</span>
              <span>Confidence: {{ lastResult.authentication_metadata.confidence_level }}</span>
              <span>Liveness: {{ lastResult.authentication_metadata.liveness_method }}</span>
              <span v-if="lastResult.match_fallback_used" class="text-warning">
                ⚠️ {{ lastResult.match_fallback_explanation }}
              </span>
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
import { sessionApi } from '../services/api'

const cameraRef = ref(null)
const authSession = ref(null)
const recognizedUser = ref(null)
const cameraActive = ref(false)

const settings = reactive({
  mode: 'identification',
  userId: '',
  requireLiveness: true
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
  frameData: '',
  file: null
})

const captureIntervalMs = 800 // Lebih cepat untuk authentication
const captureTimer = ref(null)

const isCameraActive = computed(() => cameraActive.value)
const sessionToken = computed(() => authSession.value?.session_token || '')
const lastResult = computed(() => streamingState.lastResult)
const sessionFeedback = computed(() => streamingState.lastResult?.session_feedback || '')
const livenessVerified = computed(() => streamingState.lastResult?.liveness_verified || false)
const obstaclesDetected = computed(() => streamingState.lastResult?.obstacles || [])

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
      session_type: 'webcam',
      require_liveness: settings.requireLiveness,
      metadata: {
        flow_mode: settings.mode,
        device_info: parseDeviceInfo()
      }
    }
    if (settings.mode === 'verification') {
      if (!settings.userId) {
        throw new Error('External user ID wajib diisi untuk mode verification')
      }
      payload.user_id = settings.userId
    }
    const response = await sessionApi.createAuthentication(payload)
    authSession.value = {
      session_token: response.data.session_token,
      session_type: response.data.session_type || settings.mode,
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
        session_type: settings.mode,
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
    const buffer = new Uint8Array(binary.length)
    for (let i = 0; i < binary.length; i += 1) {
      buffer[i] = binary.charCodeAt(i)
    }
    return new Blob([buffer], { type: mime })
  }
  throw new Error('Format frame tidak didukung')
}

async function sendFrame(frameData, overrideToken) {
  const token = overrideToken || authSession.value?.session_token
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
    const data = response.data
    streamingState.lastResult = data

    // Enhanced session-based feedback
    const sessionFeedback = data.session_feedback || data.message || ''
    const livenessVerified = data.liveness_verified || false
    const obstacles = data.obstacles || []

    if (data.requires_more_frames) {
      errors.session = ''
      let message = sessionFeedback || 'Lanjutkan streaming untuk verifikasi.'
      if (obstacles.length > 0) {
        message += ` Obstacle detected: ${obstacles.join(', ')}`
      }
      addLog('info', message, {
        frames_processed: data.frames_processed,
        liveness_score: data.liveness_score,
        liveness_verified: livenessVerified,
        obstacles: obstacles
      })
      return data
    }

    if (data.success) {
      recognizedUser.value = data.matched_user || data.user || null
      errors.session = ''
      
      let successMessage = 'Autentikasi berhasil'
      if (data.matched_user) {
        successMessage += ` - ${data.matched_user.display_name || data.matched_user.external_user_id}`
      }
      
      // Session-based authentication info
      const authMetadata = data.authentication_metadata || {}
      if (authMetadata.algorithm_used === 'session_based') {
        successMessage += ' (session-based liveness detection)'
      }
      
      addLog('success', successMessage, {
        similarity_score: data.similarity_score,
        liveness_score: data.liveness_score,
        liveness_verified: livenessVerified,
        quality_score: data.quality_score,
        algorithm: authMetadata.algorithm_used,
        confidence_level: authMetadata.confidence_level,
        session_feedback: sessionFeedback
      })
      stopStreaming()
    } else {
      recognizedUser.value = null
      const failureMsg = sessionFeedback || data.message || data.error || 'Autentikasi gagal'
      
      // Handle max frames reached - show error immediately and stop
      if (data.max_frames_reached) {
        errors.session = failureMsg
        addLog('warning', failureMsg, {
          similarity_score: data.similarity_score,
          liveness_score: data.liveness_score,
          liveness_verified: livenessVerified,
          obstacles: obstacles,
          session_feedback: sessionFeedback,
          frames_sent: streamingState.framesSent,
          max_frames_reached: true
        })
        stopStreaming()
        return data
      }
      
      // Don't show error immediately for other cases, let it continue for a few frames
      if (streamingState.framesSent > 3) {
        errors.session = failureMsg
      }
      
      addLog('info', failureMsg, {
        similarity_score: data.similarity_score,
        liveness_score: data.liveness_score,
        liveness_verified: livenessVerified,
        obstacles: obstacles,
        session_feedback: sessionFeedback,
        frames_sent: streamingState.framesSent
      })
      
      // Stop if session finalized or requires new session
      if (data.session_finalized) {
        stopStreaming()
      }
      if (data.requires_new_session) {
        addLog('warning', 'Session perlu dibuat ulang. Tekan Reset lalu buat session baru.')
        stopStreaming()
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
  background: #dcfce7;
  border: 1px solid #16a34a;
  border-radius: 0.25rem;
  padding: 0.5rem;
  margin-top: 1rem;
  font-size: 0.9rem;
}
</style>
