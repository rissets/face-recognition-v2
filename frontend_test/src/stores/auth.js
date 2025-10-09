import { defineStore } from 'pinia'
import { authApi, setAuthToken, configureAuthHandlers } from '../services/api'

const STORAGE_KEY = 'face_tester_auth'

export const useAuthStore = defineStore('auth', {
  state: () => ({
    accessToken: null,
    refreshToken: null,
    profile: null,
    loading: false,
    error: null
  }),
  getters: {
    isAuthenticated: (state) => Boolean(state.accessToken)
  },
  actions: {
    initFromStorage() {
      this.setupApiHandlers()
      try {
        const raw = localStorage.getItem(STORAGE_KEY)
        if (!raw) {
          return
        }
        const data = JSON.parse(raw)
        this.accessToken = data.accessToken
        this.refreshToken = data.refreshToken
        this.profile = data.profile || null
        setAuthToken(this.accessToken)
      } catch (error) {
        console.warn('Failed to restore auth session', error)
        this.logout()
      }
    },
    persist() {
      const payload = {
        accessToken: this.accessToken,
        refreshToken: this.refreshToken,
        profile: this.profile
      }
      localStorage.setItem(STORAGE_KEY, JSON.stringify(payload))
    },
    setTokens({ access, refresh }) {
      this.accessToken = access
      this.refreshToken = refresh || null
      setAuthToken(access)
      this.persist()
      this.setupApiHandlers()
    },
    async login({ email, password, deviceInfo = {} }) {
      this.loading = true
      this.error = null
      try {
        const response = await authApi.login({
          email,
          password,
          device_info: {
            device_id: deviceInfo.device_id || 'web-tester',
            device_name: deviceInfo.device_name || 'API Tester',
            device_type: deviceInfo.device_type || 'web',
            browser: deviceInfo.browser || navigator.userAgent
          }
        })
        const { access, refresh } = response.data
        this.setTokens({ access, refresh })
        await this.fetchProfile()
        return response.data
      } catch (error) {
        this.error = error.response?.data || error.message
        throw error
      } finally {
        this.loading = false
      }
    },
    async fetchProfile() {
      try {
        const response = await authApi.profile()
        this.profile = response.data
        this.persist()
        return response.data
      } catch (error) {
        this.error = error.response?.data || error.message
        throw error
      }
    },
    logout() {
      this.accessToken = null
      this.refreshToken = null
      this.profile = null
      this.error = null
      setAuthToken(null)
      localStorage.removeItem(STORAGE_KEY)
      this.setupApiHandlers()
    },
    setupApiHandlers() {
      configureAuthHandlers({
        getRefreshToken: () => this.refreshToken,
        onTokenRefreshed: (access, refresh) => {
          this.accessToken = access
          if (refresh) {
            this.refreshToken = refresh
          }
          setAuthToken(access)
          this.persist()
        },
        onLogout: () => {
          this.logout()
        }
      })
    }
  }
})
