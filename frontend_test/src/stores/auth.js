import { defineStore } from 'pinia'
import { clientAuthApi, clientApi, setClientSession, clearClientSession } from '../services/api'

const STORAGE_KEY = 'face_tester_client_session'

export const useAuthStore = defineStore('client-session', {
  state: () => ({
    apiKey: null,
    accessToken: null,
    client: null,
    loading: false,
    error: null
  }),
  getters: {
    isAuthenticated: (state) => Boolean(state.apiKey && state.accessToken)
  },
  actions: {
    initFromStorage() {
      try {
        const raw = localStorage.getItem(STORAGE_KEY)
        if (!raw) return
        const data = JSON.parse(raw)
        this.apiKey = data.apiKey
        this.accessToken = data.accessToken
        this.client = data.client || null
        setClientSession({ apiKey: this.apiKey, accessToken: this.accessToken })
      } catch (error) {
        console.warn('Failed to restore client session', error)
        this.disconnect()
      }
    },
    persist() {
      localStorage.setItem(
        STORAGE_KEY,
        JSON.stringify({
          apiKey: this.apiKey,
          accessToken: this.accessToken,
          client: this.client
        })
      )
    },
    async connect({ apiKey, apiSecret }) {
      this.loading = true
      this.error = null
      try {
        const response = await clientAuthApi.authenticate({
          api_key: apiKey,
          api_secret: apiSecret
        })
        this.apiKey = response.data.api_key
        this.accessToken = response.data.access_token
        this.client = {
          id: response.data.client_id,
          name: response.data.client_name,
          tier: response.data.tier,
          rate_limits: response.data.rate_limits,
          features: response.data.features
        }
        setClientSession({ apiKey: this.apiKey, accessToken: this.accessToken })
        this.persist()
        await this.refreshClientDetails()
        return response.data
      } catch (error) {
        this.error = error.response?.data || error.message
        throw error
      } finally {
        this.loading = false
      }
    },
    async refreshClientDetails() {
      if (!this.apiKey) return
      try {
        const response = await clientApi.list()
        const firstClient = Array.isArray(response.data) ? response.data[0] : null
        if (firstClient) {
          this.client = firstClient
          this.persist()
        }
      } catch (error) {
        console.warn('Failed to refresh client details', error)
      }
    },
    async fetchProfile() {
      await this.refreshClientDetails()
      return this.client
    },
    setTokens() {
      console.warn('setTokens is not used in third-party client mode')
    },
    disconnect() {
      this.apiKey = null
      this.accessToken = null
      this.client = null
      this.error = null
      clearClientSession()
      localStorage.removeItem(STORAGE_KEY)
    }
  }
})
