<template>
  <div class="enhanced-webrtc-auth">
    <div class="auth-header">
      <h2>Enhanced Face Recognition System</h2>
      <p class="subtitle">With Liveness Detection & Obstacle Recognition</p>
    </div>

    <div class="controls-panel">
      <div class="control-group">
        <label for="email">Target Email (Optional for Verification)</label>
        <input 
          id="email"
          v-model="targetEmail" 
          placeholder="user@example.com" 
          type="email"
          :disabled="running"
        />
      </div>
      
      <div class="control-group">
        <label for="sessionType">Authentication Mode</label>
        <select id="sessionType" v-model="sessionType" :disabled="running">
          <option value="authentication">Authentication (Registered User)</option>
          <option value="verification">Verification (Specific Email)</option>
          <option value="identification">Identification (Any User)</option>
        </select>
      </div>
      
      <div class="button-group">
        <button @click="start" :disabled="running" class="btn-primary">
          {{ running ? 'Processing...' : 'Start Recognition' }}
        </button>
        <button @click="stop" :disabled="!running" class="btn-secondary">Stop</button>
        <button @click="toggleMode" :disabled="running" class="btn-mode">
          Mode: {{ modeLabel }}
        </button>
      </div>
    </div>

    <!-- Status Display -->
    <div class="status-panel">
      <div class="status-item">
        <span class="label">Status:</span>
        <span class="value" :class="statusClass">{{ status }}</span>
      </div>
      <div class="status-item">
        <span class="label">Frames Processed:</span>
        <span class="value">{{ frames }}</span>
      </div>
      <div class="status-item">
        <span class="label">Connection:</span>
        <span class="value">{{ modeLabel }}</span>
      </div>
    </div>

    <!-- Camera Stream -->
    <div class="camera-container">
      <CameraStream ref="cam">
        <template #overlay>
          <div class="face-overlay">
            <!-- Face Detection Box -->
            <div 
              v-if="faceBox" 
              class="face-box"
              :style="faceBoxStyle"
            ></div>
            
            <!-- Liveness Status -->
            <div class="liveness-indicator" :class="livenessClass">
              <div class="indicator-dot"></div>
              <span>{{ livenessStatus }}</span>
            </div>
            
            <!-- Instructions -->
            <div class="instructions">
              {{ currentInstruction }}
            </div>
          </div>
        </template>
      </CameraStream>
    </div>

    <!-- Detection Results -->
    <div class="results-panel">
      <!-- Liveness Detection -->
      <div class="detection-section">
        <h3>
          <span class="icon">👁️</span>
          Liveness Detection
          <span class="status-badge" :class="livenessClass">
            {{ livenessVerified ? 'VERIFIED' : 'CHECKING' }}
          </span>
        </h3>
        <div class="detection-grid">
          <div class="metric">
            <span class="metric-label">Blinks Detected:</span>
            <span class="metric-value">{{ livenessData.blinks_detected || 0 }}</span>
          </div>
          <div class="metric">
            <span class="metric-label">Motion Events:</span>
            <span class="metric-value">{{ livenessData.motion_events || 0 }}</span>
          </div>
          <div class="metric">
            <span class="metric-label">Liveness Score:</span>
            <div class="score-bar">
              <div 
                class="score-fill" 
                :style="{ width: `${(livenessScore * 100)}%` }"
                :class="{ 'score-good': livenessScore > 0.7 }"
              ></div>
              <span class="score-text">{{ (livenessScore * 100).toFixed(1) }}%</span>
            </div>
          </div>
          <div class="metric">
            <span class="metric-label">Eye Aspect Ratio:</span>
            <span class="metric-value">{{ livenessData.ear?.toFixed(3) || '0.000' }}</span>
          </div>
        </div>
      </div>

      <!-- Obstacle Detection -->
      <div class="detection-section">
        <h3>
          <span class="icon">🚫</span>
          Obstacle Detection
          <span class="status-badge" :class="obstacleClass">
            {{ obstacles.length > 0 ? 'OBSTACLES FOUND' : 'CLEAR' }}
          </span>
        </h3>
        <div class="obstacle-list">
          <div 
            v-if="obstacles.length === 0" 
            class="obstacle-item clear"
          >
            <span class="obstacle-icon">✅</span>
            <span>No obstacles detected</span>
          </div>
          <div 
            v-for="obstacle in obstacles" 
            :key="obstacle"
            class="obstacle-item detected"
          >
            <span class="obstacle-icon">{{ getObstacleIcon(obstacle) }}</span>
            <span>{{ getObstacleLabel(obstacle) }}</span>
            <span class="confidence">{{ getObstacleConfidence(obstacle) }}%</span>
          </div>
        </div>
      </div>

      <!-- Quality Metrics -->
      <div class="detection-section">
        <h3>
          <span class="icon">📊</span>
          Quality Metrics
        </h3>
        <div class="detection-grid">
          <div class="metric">
            <span class="metric-label">Image Quality:</span>
            <div class="score-bar">
              <div 
                class="score-fill" 
                :style="{ width: `${(qualityScore * 100)}%` }"
                :class="{ 'score-good': qualityScore > 0.6 }"
              ></div>
              <span class="score-text">{{ (qualityScore * 100).toFixed(1) }}%</span>
            </div>
          </div>
          <div class="metric">
            <span class="metric-label">Similarity Score:</span>
            <div class="score-bar">
              <div 
                class="score-fill" 
                :style="{ width: `${(similarityScore * 100)}%` }"
                :class="{ 'score-good': similarityScore > 0.7 }"
              ></div>
              <span class="score-text">{{ (similarityScore * 100).toFixed(1) }}%</span>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- Success/Error Messages -->
    <div v-if="authSuccess" class="result-box success">
      <h3>🎉 Authentication Successful!</h3>
      <div class="success-details">
        <div class="detail-item">
          <span class="label">Similarity Score:</span>
          <span class="value">{{ (authSimilarity * 100).toFixed(1) }}%</span>
        </div>
        <div class="detail-item">
          <span class="label">Liveness Verified:</span>
          <span class="value">{{ livenessVerified ? 'Yes' : 'No' }}</span>
        </div>
        <div class="detail-item" v-if="recognizedUser">
          <span class="label">User:</span>
          <span class="value">{{ recognizedUser.email }}</span>
        </div>
      </div>
    </div>

    <div v-if="errorMsg" class="result-box error">
      <h3>❌ Authentication Failed</h3>
      <p>{{ errorMsg }}</p>
      <div class="retry-button">
        <button @click="clearError" class="btn-secondary">Try Again</button>
      </div>
    </div>
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
const recognizedUser = ref(null)

// Liveness and detection data
const livenessData = ref({})
const livenessVerified = ref(false)
const livenessScore = ref(0)
const obstacles = ref([])
const obstacleConfidence = ref({})
const qualityScore = ref(0)
const similarityScore = ref(0)
const faceBox = ref(null)

let httpSessionToken = null

const { connect, sendFrame, close, status, frames, liveness } = useWebRTCSession({
  onResult: handleResult,
  onFinal: handleFinal,
  onError: handleWebRTCError,
  onFrame: handleFrameResult
})

const modeLabel = computed(() => (httpMode.value ? 'HTTP' : 'WebSocket'))

const statusClass = computed(() => {
  if (status.value === 'connected') return 'status-connected'
  if (status.value === 'connecting') return 'status-connecting'
  if (status.value === 'disconnected') return 'status-disconnected'
  return 'status-idle'
})

const livenessClass = computed(() => {
  if (livenessVerified.value) return 'status-verified'
  if (livenessScore.value > 0.5) return 'status-progress'
  return 'status-waiting'
})

const obstacleClass = computed(() => {
  return obstacles.value.length > 0 ? 'status-warning' : 'status-clear'
})

const livenessStatus = computed(() => {
  if (livenessVerified.value) return 'Live Detected'
  if (livenessScore.value > 0.5) return 'Verifying...'
  return 'Blink or Move'
})

const currentInstruction = computed(() => {
  if (!running.value) return 'Press Start to begin'
  if (obstacles.value.length > 0) {
    const obstacleNames = obstacles.value.map(o => getObstacleLabel(o)).join(', ')
    return `Please remove: ${obstacleNames}`
  }
  if (!livenessVerified.value) {
    return 'Please blink or move your head slightly'
  }
  if (status.value === 'processing') {
    return 'Processing... Stay still'
  }
  return 'Looking good! Keep your face in frame'
})

const faceBoxStyle = computed(() => {
  if (!faceBox.value) return {}
  return {
    left: `${faceBox.value.x}px`,
    top: `${faceBox.value.y}px`,
    width: `${faceBox.value.width}px`,
    height: `${faceBox.value.height}px`
  }
})

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
  
  // Reset detection data
  livenessData.value = {}
  livenessVerified.value = false
  livenessScore.value = 0
  obstacles.value = []
  obstacleConfidence.value = {}
  qualityScore.value = 0
  similarityScore.value = 0
  faceBox.value = null
  recognizedUser.value = null

  const resolvedType = resolveSessionType()
  const payload = {
    session_type: resolvedType,
    email: targetEmail.value || null,
    device_info: { 
      device_id: 'enhanced-webrtc-auth',
      user_agent: navigator.userAgent,
      screen_resolution: `${screen.width}x${screen.height}`
    }
  }

  console.debug('[EnhancedWebRTCAuthorization] Session payload', {
    mode: httpMode.value ? 'http' : 'webrtc',
    payload
  })

  try {
    if (httpMode.value) {
      const { data, status } = await faceAuthApi.createSession(payload)
      console.debug('[EnhancedWebRTCAuthorization] HTTP session response', status, data)
      httpSessionToken = data.session_token
    } else {
      const { data, status } = await faceAuthApi.createWebRTCSession(payload)
      console.debug('[EnhancedWebRTCAuthorization] WebRTC session response', status, data)
      connect(data.session_token)
    }

    await cam.value.start()
    pump()
  } catch (error) {
    running.value = false
    console.error('[EnhancedWebRTCAuthorization] Session start failed', error?.response?.data || error)
    const status = error?.response?.status
    const data = error?.response?.data
    if (status === 429) {
      const retry = error?.response?.headers?.['retry-after']
      errorMsg.value = (data?.error || 'Rate limit terlampaui.') + (retry ? ` Coba lagi dalam ${retry} detik.` : '')
    } else {
      errorMsg.value = data?.detail || data?.error || error?.message || 'Failed to start session'
    }
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
        const response = await faceAuthApi.processFrame({ 
          session_token: httpSessionToken, 
          frame_data: frame 
        })
        console.debug('[EnhancedWebRTCAuthorization] HTTP frame response', response?.status, response?.data)
        handleFrameResult(response.data)
      } else {
        sendFrame(frame)
      }
    } catch (e) {
      console.error('Frame processing error:', e)
      errorMsg.value = e.message
    }
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
  cam.value?.stop()
  close()
}

function toggleMode() {
  if (running.value) return
  httpMode.value = !httpMode.value
}

function clearError() {
  errorMsg.value = ''
  authSuccess.value = false
}

function handleFrameResult(res) {
  console.debug('[EnhancedWebRTCAuthorization] Frame result payload', res)
  if (!res) return
  
  // Update liveness data
  if (res.liveness_data) {
    livenessData.value = res.liveness_data
  }
  
  // Update detection scores
  livenessVerified.value = res.liveness_verified || false
  livenessScore.value = res.liveness_score || 0
  qualityScore.value = res.quality_score || 0
  similarityScore.value = res.similarity_score || 0
  
  // Update obstacles
  if (res.obstacles) {
    obstacles.value = res.obstacles
  }
  if (res.obstacle_confidence) {
    obstacleConfidence.value = res.obstacle_confidence
  }
  
  // Update face box
  if (res.bbox) {
    const [x1, y1, x2, y2] = res.bbox
    faceBox.value = {
      x: x1,
      y: y1,
      width: x2 - x1,
      height: y2 - y1
    }
  }
}

function handleResult(res) {
  console.debug('[EnhancedWebRTCAuthorization] WebRTC result message', res)
  if (res?.success) {
    authSuccess.value = true
    authSimilarity.value = res.similarity_score || 0
    recognizedUser.value = res.user || null
    livenessVerified.value = true
  } else if (res?.error) {
    errorMsg.value = res.error
  }
}

function handleFinal(message) {
  console.debug('[EnhancedWebRTCAuthorization] Final session message', message)
  const result = message?.result || {}
  if (result?.success) {
    authSuccess.value = true
    authSimilarity.value = result.similarity_score || authSimilarity.value
    recognizedUser.value = result.user || null
    livenessVerified.value = true
  } else if (result?.error) {
    errorMsg.value = result.error
  }
  stop()
}

function handleWebRTCError(error) {
  console.error('[EnhancedWebRTCAuthorization] WebRTC error', error)
  errorMsg.value = error?.message || 'WebRTC connection error'
}

function getObstacleIcon(obstacle) {
  const icons = {
    glasses: '👓',
    mask: '😷',
    hat: '👒',
    hand_covering: '🤚'
  }
  return icons[obstacle] || '❓'
}

function getObstacleLabel(obstacle) {
  const labels = {
    glasses: 'Glasses',
    mask: 'Face Mask',
    hat: 'Hat/Cap',
    hand_covering: 'Hand Covering Face'
  }
  return labels[obstacle] || obstacle
}

function getObstacleConfidence(obstacle) {
  const confidence = obstacleConfidence.value[obstacle] || 0
  return Math.round(confidence * 100)
}

onBeforeUnmount(stop)
</script>

<style scoped>
.enhanced-webrtc-auth {
  max-width: 1200px;
  margin: 0 auto;
  padding: 20px;
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
}

.auth-header {
  text-align: center;
  margin-bottom: 30px;
}

.auth-header h2 {
  color: #2c3e50;
  margin: 0 0 8px 0;
  font-size: 2rem;
}

.subtitle {
  color: #7f8c8d;
  margin: 0;
  font-size: 1.1rem;
}

.controls-panel {
  background: #f8f9fa;
  border-radius: 12px;
  padding: 20px;
  margin-bottom: 20px;
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 20px;
  align-items: end;
}

.control-group {
  display: flex;
  flex-direction: column;
}

.control-group label {
  font-weight: 600;
  color: #2c3e50;
  margin-bottom: 8px;
  font-size: 0.9rem;
}

.control-group input,
.control-group select {
  padding: 12px;
  border: 2px solid #e9ecef;
  border-radius: 8px;
  font-size: 1rem;
  transition: border-color 0.2s;
}

.control-group input:focus,
.control-group select:focus {
  outline: none;
  border-color: #3498db;
}

.button-group {
  grid-column: 1 / -1;
  display: flex;
  gap: 12px;
  justify-content: center;
}

.btn-primary {
  background: linear-gradient(135deg, #3498db, #2980b9);
  color: white;
  border: none;
  padding: 12px 24px;
  border-radius: 8px;
  font-size: 1rem;
  font-weight: 600;
  cursor: pointer;
  transition: all 0.2s;
}

.btn-primary:hover:not(:disabled) {
  transform: translateY(-2px);
  box-shadow: 0 4px 12px rgba(52, 152, 219, 0.3);
}

.btn-primary:disabled {
  opacity: 0.6;
  cursor: not-allowed;
}

.btn-secondary {
  background: #95a5a6;
  color: white;
  border: none;
  padding: 12px 24px;
  border-radius: 8px;
  font-size: 1rem;
  cursor: pointer;
  transition: background 0.2s;
}

.btn-secondary:hover:not(:disabled) {
  background: #7f8c8d;
}

.btn-mode {
  background: #e74c3c;
  color: white;
  border: none;
  padding: 12px 24px;
  border-radius: 8px;
  font-size: 1rem;
  cursor: pointer;
  transition: background 0.2s;
}

.btn-mode:hover:not(:disabled) {
  background: #c0392b;
}

.status-panel {
  display: flex;
  gap: 30px;
  margin-bottom: 20px;
  padding: 15px;
  background: #ecf0f1;
  border-radius: 8px;
  justify-content: center;
}

.status-item {
  display: flex;
  align-items: center;
  gap: 8px;
}

.status-item .label {
  font-weight: 600;
  color: #2c3e50;
}

.status-item .value {
  font-weight: 700;
  padding: 4px 8px;
  border-radius: 4px;
}

.status-connected { background: #d5f4e6; color: #27ae60; }
.status-connecting { background: #fef5d3; color: #f39c12; }
.status-disconnected { background: #fadbd8; color: #e74c3c; }
.status-idle { background: #e8f5e8; color: #2c3e50; }

.camera-container {
  position: relative;
  margin-bottom: 30px;
  border-radius: 12px;
  overflow: hidden;
  box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
}

.face-overlay {
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  pointer-events: none;
}

.face-box {
  position: absolute;
  border: 3px solid #3498db;
  border-radius: 8px;
  box-shadow: 0 0 0 2px rgba(52, 152, 219, 0.3);
}

.liveness-indicator {
  position: absolute;
  top: 20px;
  left: 20px;
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 8px 12px;
  border-radius: 20px;
  font-weight: 600;
  font-size: 0.9rem;
  backdrop-filter: blur(10px);
}

.status-verified {
  background: rgba(46, 204, 113, 0.9);
  color: white;
}

.status-progress {
  background: rgba(241, 196, 15, 0.9);
  color: white;
}

.status-waiting {
  background: rgba(52, 152, 219, 0.9);
  color: white;
}

.indicator-dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: currentColor;
  animation: pulse 1.5s infinite;
}

@keyframes pulse {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.3; }
}

.instructions {
  position: absolute;
  bottom: 20px;
  left: 50%;
  transform: translateX(-50%);
  background: rgba(0, 0, 0, 0.8);
  color: white;
  padding: 12px 20px;
  border-radius: 20px;
  font-size: 1rem;
  font-weight: 500;
  text-align: center;
  backdrop-filter: blur(10px);
  max-width: 90%;
}

.results-panel {
  display: grid;
  grid-template-columns: 1fr;
  gap: 20px;
}

.detection-section {
  background: white;
  border-radius: 12px;
  padding: 20px;
  box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
  border: 2px solid #ecf0f1;
}

.detection-section h3 {
  display: flex;
  align-items: center;
  gap: 12px;
  margin: 0 0 15px 0;
  color: #2c3e50;
  font-size: 1.2rem;
}

.detection-section .icon {
  font-size: 1.5rem;
}

.status-badge {
  font-size: 0.7rem;
  font-weight: 700;
  padding: 4px 8px;
  border-radius: 12px;
  margin-left: auto;
}

.status-verified { background: #d5f4e6; color: #27ae60; }
.status-progress { background: #fef5d3; color: #f39c12; }
.status-waiting { background: #e3f2fd; color: #2196f3; }
.status-warning { background: #fadbd8; color: #e74c3c; }
.status-clear { background: #d5f4e6; color: #27ae60; }

.detection-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 15px;
}

.metric {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.metric-label {
  font-weight: 600;
  color: #2c3e50;
  font-size: 0.9rem;
}

.metric-value {
  font-size: 1.2rem;
  font-weight: 700;
  color: #3498db;
}

.score-bar {
  position: relative;
  height: 24px;
  background: #ecf0f1;
  border-radius: 12px;
  overflow: hidden;
}

.score-fill {
  height: 100%;
  background: linear-gradient(90deg, #e74c3c, #f39c12, #f1c40f);
  transition: width 0.3s ease;
  border-radius: 12px;
}

.score-fill.score-good {
  background: linear-gradient(90deg, #27ae60, #2ecc71);
}

.score-text {
  position: absolute;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
  font-weight: 600;
  font-size: 0.8rem;
  color: #2c3e50;
  text-shadow: 1px 1px 2px rgba(255, 255, 255, 0.8);
}

.obstacle-list {
  display: flex;
  flex-direction: column;
  gap: 10px;
}

.obstacle-item {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 12px;
  border-radius: 8px;
  font-weight: 500;
}

.obstacle-item.clear {
  background: #d5f4e6;
  color: #27ae60;
}

.obstacle-item.detected {
  background: #fadbd8;
  color: #e74c3c;
}

.obstacle-icon {
  font-size: 1.2rem;
}

.confidence {
  margin-left: auto;
  font-weight: 700;
  font-size: 0.9rem;
}

.result-box {
  margin-top: 20px;
  padding: 20px;
  border-radius: 12px;
  border: 2px solid;
}

.result-box.success {
  background: #d5f4e6;
  border-color: #27ae60;
  color: #1e8449;
}

.result-box.error {
  background: #fadbd8;
  border-color: #e74c3c;
  color: #c0392b;
}

.result-box h3 {
  margin: 0 0 15px 0;
  font-size: 1.3rem;
}

.success-details {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 10px;
}

.detail-item {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 8px 12px;
  background: rgba(255, 255, 255, 0.7);
  border-radius: 6px;
}

.detail-item .label {
  font-weight: 600;
}

.detail-item .value {
  font-weight: 700;
}

.retry-button {
  margin-top: 15px;
  text-align: center;
}

@media (max-width: 768px) {
  .controls-panel {
    grid-template-columns: 1fr;
  }
  
  .status-panel {
    flex-direction: column;
    gap: 10px;
    text-align: center;
  }
  
  .detection-grid {
    grid-template-columns: 1fr;
  }
  
  .success-details {
    grid-template-columns: 1fr;
  }
}
</style>
