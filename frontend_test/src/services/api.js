import axios from 'axios'

export const API_BASE_URL = 'http://127.0.0.1:8000/api/'

export const apiClient = axios.create({
  baseURL: API_BASE_URL,
  timeout: 20000,
  headers: {
    'Content-Type': 'application/json'
  }
})

const refreshClient = axios.create({
  baseURL: API_BASE_URL,
  timeout: 20000,
  headers: {
    'Content-Type': 'application/json'
  }
})

let refreshTokenProvider = null
let tokenUpdateHandler = null
let logoutHandler = null
let isRefreshing = false
let refreshPromise = null
const pendingQueue = []

function processQueue(error, token = null) {
  while (pendingQueue.length) {
    const { resolve, reject, config } = pendingQueue.shift()
    if (token) {
      const retryConfig = {
        ...config,
        __isRetryRequest: true,
        headers: {
          ...config.headers,
          Authorization: `Bearer ${token}`
        }
      }
      resolve(apiClient.request(retryConfig))
    } else {
      reject(error)
    }
  }
}

export function configureAuthHandlers({ getRefreshToken, onTokenRefreshed, onLogout }) {
  refreshTokenProvider = typeof getRefreshToken === 'function' ? getRefreshToken : null
  tokenUpdateHandler = typeof onTokenRefreshed === 'function' ? onTokenRefreshed : null
  logoutHandler = typeof onLogout === 'function' ? onLogout : null
}

export function setAuthToken(token) {
  if (token) {
    apiClient.defaults.headers.common.Authorization = `Bearer ${token}`
  } else {
    delete apiClient.defaults.headers.common.Authorization
  }
}

function shouldAttemptRefresh(response) {
  if (!response) return false
  if (response.status !== 401) return false

  const code = response.data?.code
  if (code === 'token_not_valid') return true

  const messageArray = response.data?.messages
  if (Array.isArray(messageArray)) {
    return messageArray.some((msg) =>
      typeof msg?.message === 'string' && msg.message.toLowerCase().includes('token is expired')
    )
  }

  const detail = response.data?.detail
  if (typeof detail === 'string' && detail.toLowerCase().includes('token is expired')) {
    return true
  }

  return false
}

apiClient.interceptors.response.use(
  (response) => response,
  (error) => {
    const { config, response } = error

    if (!config || config.__isRetryRequest || !shouldAttemptRefresh(response)) {
      return Promise.reject(error)
    }

    if (!refreshTokenProvider) {
      logoutHandler?.()
      return Promise.reject(error)
    }

    const refreshToken = refreshTokenProvider()
    if (!refreshToken) {
      logoutHandler?.()
      return Promise.reject(error)
    }

    if (!isRefreshing) {
      isRefreshing = true
      refreshPromise = refreshClient
        .post('auth/token/refresh/', { refresh: refreshToken })
        .then((res) => {
          const newAccess = res.data?.access
          const newRefresh = res.data?.refresh || refreshToken

          if (!newAccess) {
            throw new Error('Refresh response missing access token')
          }

          tokenUpdateHandler?.(newAccess, newRefresh)
          processQueue(null, newAccess)
          return newAccess
        })
        .catch((refreshError) => {
          processQueue(refreshError, null)
          logoutHandler?.()
          throw refreshError
        })
        .finally(() => {
          isRefreshing = false
        })
    }

    return new Promise((resolve, reject) => {
      pendingQueue.push({ resolve, reject, config })
    })
  }
)

export const authApi = {
  login(payload) {
    return apiClient.post('auth/token/', payload)
  },
  register(payload) {
    return apiClient.post('auth/register/', payload)
  },
  profile() {
    return apiClient.get('auth/profile/')
  },
  updateProfile(payload) {
    return apiClient.put('auth/profile/', payload)
  },
  userDevices() {
    return apiClient.get('user/devices/')
  },
  authHistory() {
    return apiClient.get('user/auth-history/')
  },
  securityAlerts() {
    return apiClient.get('user/security-alerts/')
  }
}

export const enrollmentApi = {
  createSession(payload) {
    return apiClient.post('enrollment/create/', payload)
  },
  processFrame(payload) {
    return apiClient.post('enrollment/process-frame/', payload)
  }
}

export const faceAuthApi = {
  createSession(payload) {
    return apiClient.post('auth/face/create/', payload)
  },
  createWebRTCSession(payload) {
    return apiClient.post('auth/face/webrtc/create/', payload)
  },
  processFrame(payload) {
    return apiClient.post('auth/face/process-frame/', payload)
  },
  createPublicSession(payload) {
    return apiClient.post('auth/face/public/create/', payload)
  },
  createPublicWebRTCSession(payload) {
    return apiClient.post('auth/face/webrtc/public/create/', payload)
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

export const analyticsApi = {
  authLogs(params) {
    return apiClient.get('analytics/auth-logs/', { params })
  },
  securityAlerts(params) {
    return apiClient.get('analytics/security-alerts/', { params })
  },
  dashboard(params) {
    return apiClient.get('analytics/dashboard/', { params })
  },
  statistics() {
    return apiClient.get('analytics/statistics/')
  }
}

export const systemApi = {
  status() {
    return apiClient.get('system/status/')
  }
}

export const detectionApi = {
  // Get liveness detection history
  livenessHistory() {
    return apiClient.get('detection/liveness-history/')
  },
  
  // Get obstacle detection history
  obstacleHistory() {
    return apiClient.get('detection/obstacle-history/')
  },
  
  // Get detection analytics
  analytics(days = 30) {
    return apiClient.get('detection/analytics/', { params: { days } })
  }
}
