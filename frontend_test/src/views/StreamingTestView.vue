<template>
  <div class="layout-grid">
    <section class="section">
      <h2>Create Streaming Session</h2>
      <p>Gunakan panel ini untuk membuat sesi WebRTC terpisah dan menguji signaling backend.</p>
      <form class="form-grid" @submit.prevent="createSession">
        <div class="field">
          <label>Session Type</label>
          <select v-model="createForm.sessionType">
            <option value="enrollment">Enrollment</option>
            <option value="authentication">Authentication</option>
            <option value="verification">Verification</option>
          </select>
        </div>
        <div class="field">
          <label>Session Token (optional)</label>
          <input v-model="createForm.sessionToken" placeholder="Auto-generate if empty" />
        </div>
        <div class="field">
          <label>Initial Status</label>
          <select v-model="createForm.status">
            <option value="initiating">Initiating</option>
            <option value="connecting">Connecting</option>
            <option value="connected">Connected</option>
          </select>
        </div>
        <div class="field">
          <label>Session Data (JSON)</label>
          <textarea v-model="createForm.sessionData" rows="4"></textarea>
        </div>
        <div class="actions">
          <button type="submit" :disabled="loading.create">Create Session</button>
          <span v-if="errors.create" class="status-error">{{ errors.create }}</span>
        </div>
      </form>
      <div v-if="responses.create" class="response-card">
        <pre>{{ responses.create }}</pre>
      </div>
    </section>

    <section class="section">
      <h2>Sessions</h2>
      <div class="actions">
        <button type="button" @click="listSessions" :disabled="loading.list">List Sessions</button>
        <div class="field" style="max-width: 260px;">
          <label>Session ID</label>
          <input v-model="detailId" placeholder="UUID" />
        </div>
        <button type="button" @click="loadSessionDetail" :disabled="loading.detail">Load Detail</button>
      </div>
      <div v-if="responses.list" class="response-card">
        <h3>Sessions</h3>
        <pre>{{ responses.list }}</pre>
      </div>
      <div v-if="responses.detail" class="response-card">
        <h3>Session Detail</h3>
        <pre>{{ responses.detail }}</pre>
      </div>
    </section>

    <section class="section">
      <h2>WebRTC Signaling</h2>
      <form class="form-grid" @submit.prevent="sendSignal">
        <div class="field">
          <label>Session Token</label>
          <input v-model="signalForm.sessionToken" placeholder="Token from session" />
        </div>
        <div class="field">
          <label>Signal Type</label>
          <select v-model="signalForm.signalType">
            <option value="offer">Offer</option>
            <option value="answer">Answer</option>
            <option value="ice_candidate">ICE Candidate</option>
          </select>
        </div>
        <div class="field">
          <label>Direction</label>
          <select v-model="signalForm.direction">
            <option value="outbound">Outbound</option>
            <option value="inbound">Inbound</option>
          </select>
        </div>
        <div class="field">
          <label>Signal Payload (JSON)</label>
          <textarea v-model="signalForm.signalData" rows="4"></textarea>
        </div>
        <div class="actions">
          <button type="submit" :disabled="loading.signal">Send Signal</button>
          <button type="button" class="secondary" @click="loadSignals">Recent Signals</button>
          <span v-if="errors.signal" class="status-error">{{ errors.signal }}</span>
        </div>
      </form>
      <div v-if="responses.signal" class="response-card">
        <pre>{{ responses.signal }}</pre>
      </div>
      <div v-if="responses.recentSignals" class="response-card">
        <h3>Recent Signals</h3>
        <pre>{{ responses.recentSignals }}</pre>
      </div>
    </section>
  </div>
</template>

<script setup>
import { reactive, ref } from 'vue'
import { streamingApi } from '../services/api'

const createForm = reactive({
  sessionType: 'enrollment',
  sessionToken: '',
  status: 'initiating',
  sessionData: JSON.stringify({ note: 'sample streaming session' }, null, 2)
})

const signalForm = reactive({
  sessionToken: '',
  signalType: 'offer',
  direction: 'outbound',
  signalData: JSON.stringify({
    type: 'offer',
    sdp: 'sample sdp data'
  }, null, 2)
})

const responses = reactive({
  create: '',
  list: '',
  detail: '',
  signal: '',
  recentSignals: ''
})

const errors = reactive({
  create: '',
  signal: ''
})

const loading = reactive({
  create: false,
  list: false,
  detail: false,
  signal: false
})

const detailId = ref('')

function stringify(data) {
  return JSON.stringify(data, null, 2)
}

function parseJson(raw, fallback = {}) {
  if (!raw) return fallback
  try {
    return JSON.parse(raw)
  } catch (error) {
    throw new Error('JSON payload invalid: ' + error.message)
  }
}

function ensureSessionToken(value) {
  if (value) return value
  if (typeof crypto !== 'undefined' && typeof crypto.randomUUID === 'function') {
    return crypto.randomUUID()
  }
  return `session-${Math.random().toString(36).slice(2, 10)}`
}

async function createSession() {
  loading.create = true
  errors.create = ''
  try {
    const payload = {
      session_type: createForm.sessionType,
      status: createForm.status,
      session_data: parseJson(createForm.sessionData)
    }
    const token = ensureSessionToken(createForm.sessionToken)
    payload.session_token = token
    createForm.sessionToken = token

    const response = await streamingApi.createSession(payload)
    responses.create = stringify(response.data)
    if (response.data.session_token) {
      createForm.sessionToken = response.data.session_token
      signalForm.sessionToken = response.data.session_token
    }
  } catch (error) {
    errors.create = stringify(error.response?.data || error.message)
  } finally {
    loading.create = false
  }
}

async function listSessions() {
  loading.list = true
  try {
    const response = await streamingApi.listSessions()
    responses.list = stringify(response.data)
  } catch (error) {
    responses.list = stringify(error.response?.data || error.message)
  } finally {
    loading.list = false
  }
}

async function loadSessionDetail() {
  if (!detailId.value) {
    errors.create = 'Session ID required for detail lookup.'
    return
  }
  loading.detail = true
  try {
    const response = await streamingApi.getSession(detailId.value)
    responses.detail = stringify(response.data)
  } catch (error) {
    responses.detail = stringify(error.response?.data || error.message)
  } finally {
    loading.detail = false
  }
}

async function sendSignal() {
  loading.signal = true
  errors.signal = ''
  try {
    if (!signalForm.sessionToken) {
      throw new Error('Session token required.')
    }
    const signalPayload = parseJson(signalForm.signalData, {})
    signalPayload.type = signalPayload.type || signalForm.signalType

    const payload = {
      session_token: signalForm.sessionToken,
      type: signalForm.signalType,
      direction: signalForm.direction,
      signal_data: signalPayload
    }
    const response = await streamingApi.sendSignal(payload)
    responses.signal = stringify(response.data)
  } catch (error) {
    errors.signal = stringify(error.response?.data || error.message)
  } finally {
    loading.signal = false
  }
}

async function loadSignals() {
  try {
    const response = await streamingApi.getSignals()
    responses.recentSignals = stringify(response.data)
  } catch (error) {
    responses.recentSignals = stringify(error.response?.data || error.message)
  }
}
</script>
