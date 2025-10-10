import axios from 'axios'

export const API_BASE_URL = 'http://127.0.0.1:8000/api/'

export const apiClient = axios.create({
  baseURL: API_BASE_URL,
  timeout: 20000,
  headers: {
    'Content-Type': 'application/json'
  }
})

export function setClientSession({ apiKey, accessToken }) {
  if (apiKey) {
    apiClient.defaults.headers.common['X-API-Key'] = apiKey
  } else {
    delete apiClient.defaults.headers.common['X-API-Key']
  }

  if (accessToken) {
    apiClient.defaults.headers.common.Authorization = `JWT ${accessToken}`
  } else {
    delete apiClient.defaults.headers.common.Authorization
  }
}

export function clearClientSession() {
  delete apiClient.defaults.headers.common['X-API-Key']
  delete apiClient.defaults.headers.common.Authorization
}

function atobPolyfill(base64) {
  if (typeof window !== 'undefined' && typeof window.atob === 'function') {
    return window.atob(base64)
  }
  if (typeof Buffer === 'function') {
    return Buffer.from(base64, 'base64').toString('binary')
  }
  throw new Error('Base64 decoding not supported in this environment')
}

function dataUrlToBlob(dataUrl) {
  if (dataUrl instanceof Blob) {
    return dataUrl
  }
  if (typeof dataUrl !== 'string') {
    throw new Error('Frame payload requires string or Blob data')
  }
  if (!dataUrl.startsWith('data:')) {
    throw new Error('Frame data must be a data URL (data:<mime>;base64,...)')
  }
  const [header, data] = dataUrl.split(',')
  const mime = header.match(/:(.*?);/)?.[1] || 'image/jpeg'
  const binary = atobPolyfill(data)
  const buffer = new Uint8Array(binary.length)
  for (let i = 0; i < binary.length; i += 1) {
    buffer[i] = binary.charCodeAt(i)
  }
  return new Blob([buffer], { type: mime })
}

function buildFrameForm(payload) {
  if (payload instanceof FormData) {
    return payload
  }

  const form = new FormData()
  const token = payload?.session_token
  if (!token) {
    throw new Error('session_token is required in frame payload')
  }
  form.append('session_token', token)

  if (payload.frame_number != null) {
    form.append('frame_number', payload.frame_number)
  }

  if (payload.image instanceof Blob) {
    form.append('image', payload.image, payload.image.name || `frame-${Date.now()}.jpg`)
  } else if (typeof payload.image === 'string') {
    const blob = dataUrlToBlob(payload.image)
    form.append('image', blob, `frame-${Date.now()}.jpg`)
  } else if (typeof payload.image_base64 === 'string') {
    const blob = dataUrlToBlob(payload.image_base64)
    form.append('image', blob, `frame-${Date.now()}.jpg`)
  } else if (payload.frame_data) {
    const blob = dataUrlToBlob(payload.frame_data)
    form.append('image', blob, `frame-${Date.now()}.jpg`)
  } else {
    throw new Error('Frame payload requires image Blob/File or frame_data string')
  }

  return form
}

export const coreApi = {
  clientInfo() {
    return apiClient.get('core/info/')
  }
}

export const clientAuthApi = {
  authenticate(payload) {
    return apiClient.post('core/auth/client/', payload)
  },
  authenticateUser(payload) {
    return apiClient.post('core/auth/user/', payload)
  }
}

export const clientApi = {
  list() {
    return apiClient.get('clients/clients/')
  },
  stats(clientId) {
    return apiClient.get(`clients/clients/${clientId}/stats/`)
  },
  resetCredentials(clientId, payload) {
    return apiClient.post(`clients/clients/${clientId}/reset_credentials/`, payload)
  }
}

export const clientUsersApi = {
  list(params = {}) {
    return apiClient.get('clients/users/', { params })
  },
  create(payload) {
    return apiClient.post('clients/users/', payload)
  },
  remove(id) {
    return apiClient.delete(`clients/users/${id}/`)
  },
  activate(id) {
    return apiClient.post(`clients/users/${id}/activate/`)
  },
  deactivate(id) {
    return apiClient.post(`clients/users/${id}/deactivate/`)
  },
  enrollments(id) {
    return apiClient.get(`clients/users/${id}/enrollments/`)
  }
}

export const sessionApi = {
  createEnrollment(payload) {
    return apiClient.post('auth/enrollment/create/', payload)
  },
  createAuthentication(payload) {
    return apiClient.post('auth/authentication/create/', payload)
  },
  processFrame(payload) {
    const isFormData = typeof FormData !== 'undefined' && payload instanceof FormData
    const config = isFormData
      ? {
          headers: {
            ...apiClient.defaults.headers.common,
            'Content-Type': 'multipart/form-data'
          }
        }
      : undefined
    return apiClient.post('auth/process-image/', payload, config)
  },
  sessionStatus(token) {
    return apiClient.get(`auth/session/${token}/status/`)
  }
}

export const recognitionApi = {
  listEmbeddings() {
    return apiClient.get('recognition/embeddings/')
  },
  listSessions() {
    return apiClient.get('recognition/sessions/')
  },
  listAttempts() {
    return apiClient.get('recognition/attempts/')
  }
}

export const enrollmentApi = {
  createSession(payload = {}) {
    const { session_type = 'webcam', user_id = null, metadata = {}, target_samples, device_info } = payload
    const requestPayload = {
      session_type,
      metadata: {
        ...metadata,
        target_samples,
        device_info
      }
    }
    if (user_id) {
      requestPayload.user_id = user_id
    }
    return sessionApi.createEnrollment(requestPayload)
  },
  processFrame(payload) {
    const form = buildFrameForm(payload)
    return sessionApi.processFrame(form)
  }
}

export const faceAuthApi = {
  createSession(payload = {}) {
    return sessionApi.createAuthentication({
      session_type: payload.session_type || 'webcam',
      require_liveness: payload.require_liveness !== undefined ? payload.require_liveness : true,
      user_id: payload.user_id || payload.email || null,
      metadata: {
        ...payload.metadata,
        mode: payload.mode || payload.session_type || 'identification',
        transport: payload.transport || 'http',
        device_info: payload.device_info
      }
    })
  },
  createWebRTCSession(payload = {}) {
    return this.createSession({ ...payload, session_type: 'webrtc', transport: 'webrtc' })
  },
  createPublicSession(payload = {}) {
    return this.createSession({ ...payload, metadata: { ...payload.metadata, is_public: true } })
  },
  createPublicWebRTCSession(payload = {}) {
    return this.createWebRTCSession({ ...payload, metadata: { ...payload.metadata, is_public: true } })
  },
  processFrame(payload) {
    const form = buildFrameForm(payload)
    return sessionApi.processFrame(form)
  },
  sessionStatus(token) {
    return sessionApi.sessionStatus(token)
  }
}

export const analyticsApi = {
  auditLogs(params = {}) {
    return apiClient.get('core/audit-logs/', { params })
  },
  securityEvents(params = {}) {
    return apiClient.get('core/security-events/', { params })
  },
  authLogs(params = {}) {
    return apiClient.get('analytics/auth-logs/', { params })
  },
  securityAlerts(params = {}) {
    return apiClient.get('analytics/security-alerts/', { params })
  },
  dashboard(params = {}) {
    return apiClient.get('analytics/dashboard/', { params })
  },
  statistics() {
    return apiClient.get('analytics/statistics/')
  },
  systemMetrics(params = {}) {
    return apiClient.get('analytics/system-metrics/', { params })
  },
  userBehavior(params = {}) {
    return apiClient.get('analytics/user-behavior/', { params })
  },
  faceRecognitionStats(params = {}) {
    return apiClient.get('analytics/face-recognition-stats/', { params })
  },
  modelPerformance(params = {}) {
    return apiClient.get('analytics/model-performance/', { params })
  },
  dataQuality(params = {}) {
    return apiClient.get('analytics/data-quality/', { params })
  },
  monitoringOverview() {
    return apiClient.get('analytics/monitoring/overview/')
  },
  systemStatus() {
    return apiClient.get('core/status/')
  }
}

export const webhookApi = {
  listEvents(params = {}) {
    return apiClient.get('webhooks/events/', { params })
  },
  listEventLogs(params = {}) {
    return apiClient.get('webhooks/event-logs/', { params })
  },
  listDeliveries(params = {}) {
    return apiClient.get('webhooks/deliveries/', { params })
  },
  endpointStats(endpointId) {
    return apiClient.get(`webhooks/endpoints/${endpointId}/stats/`)
  },
  testEndpoint(endpointId, payload) {
    return apiClient.post(`webhooks/endpoints/${endpointId}/test/`, payload)
  },
  regenerateSecret(endpointId) {
    return apiClient.post(`webhooks/endpoints/${endpointId}/regenerate_secret/`)
  },
  failedDeliveries(params = {}) {
    return apiClient.get('webhooks/deliveries/failed/', { params })
  },
  retryDelivery(deliveryId) {
    return apiClient.post(`webhooks/deliveries/${deliveryId}/retry/`)
  },
  stats(params = {}) {
    return apiClient.get('webhooks/stats/', { params })
  },
  retryFailed() {
    return apiClient.post('webhooks/retry-failed/')
  },
  clearLogs(params = {}) {
    return apiClient.delete('webhooks/clear-logs/', { params })
  }
}

export const streamingApi = {
  createSession(payload) {
    return apiClient.post('streaming/sessions/create/', payload)
  },
  listSessions() {
    return apiClient.get('streaming/sessions/')
  },
  getSession(id) {
    return apiClient.get(`streaming/sessions/${id}/`)
  },
  sendSignal(payload) {
    return apiClient.post('streaming/signaling/', payload)
  },
  getSignals() {
    return apiClient.get('streaming/signaling/')
  }
}

export const systemApi = {
  status() {
    return apiClient.get('core/status/')
  }
}

export const authApi = {
  register() {
    return Promise.reject(new Error('User registration is disabled in third-party client mode'))
  }
}
