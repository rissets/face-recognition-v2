<template>
  <div class="layout-grid">
    <section class="section">
      <h2>Authentication Logs</h2>
      <form class="form-grid" @submit.prevent="loadAuthLogs">
        <div class="field">
          <label>Success</label>
          <select v-model="authFilters.success">
            <option value="">Any</option>
            <option value="true">True</option>
            <option value="false">False</option>
          </select>
        </div>
        <div class="field">
          <label>Auth Method</label>
          <select v-model="authFilters.auth_method">
            <option value="">Any</option>
            <option value="face">Face</option>
            <option value="password">Password</option>
            <option value="2fa">2FA</option>
            <option value="social">Social</option>
          </select>
        </div>
        <div class="field">
          <label>Date From</label>
          <input v-model="authFilters.date_from" type="date" />
        </div>
        <div class="field">
          <label>Date To</label>
          <input v-model="authFilters.date_to" type="date" />
        </div>
        <div class="actions">
          <button type="submit" :disabled="loading.authLogs">Fetch Logs</button>
        </div>
      </form>
      <div v-if="responses.authLogs" class="response-card">
        <pre>{{ responses.authLogs }}</pre>
      </div>
    </section>

    <section class="section">
      <h2>Security Alerts</h2>
      <form class="form-grid" @submit.prevent="loadSecurityAlerts">
        <div class="field">
          <label>Severity</label>
          <select v-model="alertFilters.severity">
            <option value="">Any</option>
            <option value="low">Low</option>
            <option value="medium">Medium</option>
            <option value="high">High</option>
            <option value="critical">Critical</option>
          </select>
        </div>
        <div class="field">
          <label>Alert Type</label>
          <select v-model="alertFilters.alert_type">
            <option value="">Any</option>
            <option value="suspicious_login">Suspicious Login</option>
            <option value="face_mismatch">Face Mismatch</option>
            <option value="unusual_activity">Unusual Activity</option>
            <option value="brute_force">Brute Force</option>
            <option value="device_change">Device Change</option>
          </select>
        </div>
        <div class="field">
          <label>Resolved</label>
          <select v-model="alertFilters.resolved">
            <option value="">Any</option>
            <option value="true">Resolved</option>
            <option value="false">Unresolved</option>
          </select>
        </div>
        <div class="actions">
          <button type="submit" :disabled="loading.alerts">Fetch Alerts</button>
        </div>
      </form>
      <div v-if="responses.alerts" class="response-card">
        <pre>{{ responses.alerts }}</pre>
      </div>
    </section>

    <section class="section">
      <h2>Dashboard & Statistics</h2>
      <div class="form-grid">
        <div class="field">
          <label>Dashboard Period</label>
          <select v-model="dashboardPeriod">
            <option value="7_days">7 days</option>
            <option value="30_days">30 days</option>
            <option value="90_days">90 days</option>
            <option value="1_year">1 year</option>
          </select>
        </div>
        <div class="actions">
          <button type="button" @click="loadDashboard" :disabled="loading.dashboard">Load Dashboard</button>
          <button type="button" class="secondary" @click="loadStatistics" :disabled="loading.statistics">Load Statistics</button>
        </div>
      </div>
      <div v-if="responses.dashboard" class="response-card">
        <h3>Dashboard</h3>
        <pre>{{ responses.dashboard }}</pre>
      </div>
      <div v-if="responses.statistics" class="response-card">
        <h3>Statistics</h3>
        <pre>{{ responses.statistics }}</pre>
      </div>
    </section>

    <section class="section">
      <h2>System Metrics & Monitoring</h2>
      <form class="form-grid" @submit.prevent="loadSystemMetrics">
        <div class="field">
          <label>Metric Name</label>
          <input v-model="metricsFilters.metric_name" placeholder="e.g. enrollment.completed" />
        </div>
        <div class="field">
          <label>Metric Type</label>
          <select v-model="metricsFilters.metric_type">
            <option value="">Any</option>
            <option value="counter">Counter</option>
            <option value="gauge">Gauge</option>
            <option value="histogram">Histogram</option>
            <option value="timer">Timer</option>
          </select>
        </div>
        <div class="field">
          <label>Limit</label>
          <input v-model.number="metricsFilters.limit" min="1" type="number" />
        </div>
        <div class="actions">
          <button type="submit" :disabled="loading.systemMetrics">Load Metrics</button>
          <button type="button" class="secondary" @click="loadMonitoring" :disabled="loading.monitoring">
            Monitoring Overview
          </button>
        </div>
      </form>
      <div v-if="responses.systemMetrics" class="response-card">
        <h3>Latest Metrics</h3>
        <pre>{{ responses.systemMetrics }}</pre>
      </div>
      <div v-if="responses.monitoring" class="response-card">
        <h3>Monitoring Overview</h3>
        <pre>{{ responses.monitoring }}</pre>
      </div>
    </section>

    <section class="section">
      <h2>User Behaviour & Face Stats</h2>
      <div class="form-grid">
        <div class="field">
          <label>Risk Level</label>
          <select v-model="behaviorFilters.risk_level">
            <option value="">Any</option>
            <option value="low">Low</option>
            <option value="medium">Medium</option>
            <option value="high">High</option>
            <option value="critical">Critical</option>
          </select>
        </div>
        <div class="actions">
          <button type="button" @click="loadUserBehavior" :disabled="loading.userBehavior">Load Behaviour</button>
        </div>
      </div>
      <div class="form-grid">
        <div class="field">
          <label>Stats From</label>
          <input v-model="faceStatsFilters.date_from" type="date" />
        </div>
        <div class="field">
          <label>Stats To</label>
          <input v-model="faceStatsFilters.date_to" type="date" />
        </div>
        <div class="actions">
          <button type="button" @click="loadFaceStats" :disabled="loading.faceStats">Load Face Stats</button>
        </div>
      </div>
      <div v-if="responses.userBehavior" class="response-card">
        <h3>Behaviour Analytics</h3>
        <pre>{{ responses.userBehavior }}</pre>
      </div>
      <div v-if="responses.faceStats" class="response-card">
        <h3>Face Recognition Stats</h3>
        <pre>{{ responses.faceStats }}</pre>
      </div>
    </section>

    <section class="section">
      <h2>Model Performance & Data Quality</h2>
      <div class="form-grid">
        <div class="field">
          <label>Environment</label>
          <select v-model="modelFilters.environment">
            <option value="">Any</option>
            <option value="development">Development</option>
            <option value="staging">Staging</option>
            <option value="production">Production</option>
          </select>
        </div>
        <div class="field">
          <label>Model ID</label>
          <input v-model="modelFilters.model" placeholder="Optional model UUID" />
        </div>
        <div class="actions">
          <button type="button" @click="loadModelPerformance" :disabled="loading.modelPerformance">
            Load Model Performance
          </button>
        </div>
      </div>
      <div class="form-grid">
        <div class="field">
          <label>Quality From</label>
          <input v-model="dataQualityFilters.date_from" type="date" />
        </div>
        <div class="field">
          <label>Quality To</label>
          <input v-model="dataQualityFilters.date_to" type="date" />
        </div>
        <div class="actions">
          <button type="button" @click="loadDataQuality" :disabled="loading.dataQuality">
            Load Data Quality
          </button>
        </div>
      </div>
      <div v-if="responses.modelPerformance" class="response-card">
        <h3>Model Performance</h3>
        <pre>{{ responses.modelPerformance }}</pre>
      </div>
      <div v-if="responses.dataQuality" class="response-card">
        <h3>Data Quality</h3>
        <pre>{{ responses.dataQuality }}</pre>
      </div>
    </section>

    <section class="section">
      <h2>Webhook Diagnostics</h2>
      <form class="form-grid" @submit.prevent="loadWebhookStats">
        <div class="field">
          <label>Start Date</label>
          <input v-model="webhookFilters.start_date" type="datetime-local" />
        </div>
        <div class="field">
          <label>End Date</label>
          <input v-model="webhookFilters.end_date" type="datetime-local" />
        </div>
        <div class="actions">
          <button type="submit" :disabled="loading.webhookStats">Load Stats</button>
          <button type="button" class="secondary" @click="loadWebhookEvents" :disabled="loading.webhookEvents">
            List Events
          </button>
          <button type="button" class="secondary" @click="loadWebhookEventLogs" :disabled="loading.webhookEventLogs">
            Event Logs
          </button>
          <button type="button" class="secondary" @click="loadWebhookDeliveries" :disabled="loading.webhookDeliveries">
            Deliveries
          </button>
        </div>
      </form>
      <div v-if="responses.webhookStats" class="response-card">
        <h3>Webhook Stats</h3>
        <pre>{{ responses.webhookStats }}</pre>
      </div>
      <div v-if="responses.webhookEvents" class="response-card">
        <h3>Webhook Events</h3>
        <pre>{{ responses.webhookEvents }}</pre>
      </div>
      <div v-if="responses.webhookEventLogs" class="response-card">
        <h3>Event Logs</h3>
        <pre>{{ responses.webhookEventLogs }}</pre>
      </div>
      <div v-if="responses.webhookDeliveries" class="response-card">
        <h3>Deliveries</h3>
        <pre>{{ responses.webhookDeliveries }}</pre>
      </div>
    </section>
  </div>
</template>

<script setup>
import { reactive, ref } from 'vue'
import { analyticsApi, webhookApi } from '../services/api'

const authFilters = reactive({
  success: '',
  auth_method: '',
  date_from: '',
  date_to: ''
})

const alertFilters = reactive({
  severity: '',
  alert_type: '',
  resolved: ''
})

const metricsFilters = reactive({
  metric_name: '',
  metric_type: '',
  limit: 10
})

const behaviorFilters = reactive({
  risk_level: ''
})

const faceStatsFilters = reactive({
  date_from: '',
  date_to: ''
})

const modelFilters = reactive({
  environment: '',
  model: ''
})

const dataQualityFilters = reactive({
  date_from: '',
  date_to: ''
})

const webhookFilters = reactive({
  start_date: '',
  end_date: ''
})

const dashboardPeriod = ref('30_days')

const loading = reactive({
  authLogs: false,
  alerts: false,
  dashboard: false,
  statistics: false,
  systemMetrics: false,
  monitoring: false,
  userBehavior: false,
  faceStats: false,
  modelPerformance: false,
  dataQuality: false,
  webhookStats: false,
  webhookEvents: false,
  webhookEventLogs: false,
  webhookDeliveries: false
})

const responses = reactive({
  authLogs: '',
  alerts: '',
  dashboard: '',
  statistics: '',
  systemMetrics: '',
  monitoring: '',
  userBehavior: '',
  faceStats: '',
  modelPerformance: '',
  dataQuality: '',
  webhookStats: '',
  webhookEvents: '',
  webhookEventLogs: '',
  webhookDeliveries: ''
})

function stringify(data) {
  return JSON.stringify(data, null, 2)
}

function buildParams(filters) {
  return Object.fromEntries(
    Object.entries(filters).filter(([, value]) => value !== '' && value !== null && value !== undefined)
  )
}

async function loadAuthLogs() {
  loading.authLogs = true
  try {
    const response = await analyticsApi.authLogs(buildParams(authFilters))
    responses.authLogs = stringify(response.data)
  } catch (error) {
    responses.authLogs = stringify(error.response?.data || error.message)
  } finally {
    loading.authLogs = false
  }
}

async function loadSecurityAlerts() {
  loading.alerts = true
  try {
    const response = await analyticsApi.securityAlerts(buildParams(alertFilters))
    responses.alerts = stringify(response.data)
  } catch (error) {
    responses.alerts = stringify(error.response?.data || error.message)
  } finally {
    loading.alerts = false
  }
}

async function loadDashboard() {
  loading.dashboard = true
  try {
    const response = await analyticsApi.dashboard({ period: dashboardPeriod.value })
    responses.dashboard = stringify(response.data)
  } catch (error) {
    responses.dashboard = stringify(error.response?.data || error.message)
  } finally {
    loading.dashboard = false
  }
}

async function loadStatistics() {
  loading.statistics = true
  try {
    const response = await analyticsApi.statistics()
    responses.statistics = stringify(response.data)
  } catch (error) {
    responses.statistics = stringify(error.response?.data || error.message)
  } finally {
    loading.statistics = false
  }
}

async function loadSystemMetrics() {
  loading.systemMetrics = true
  try {
    const response = await analyticsApi.systemMetrics(buildParams(metricsFilters))
    responses.systemMetrics = stringify(response.data)
  } catch (error) {
    responses.systemMetrics = stringify(error.response?.data || error.message)
  } finally {
    loading.systemMetrics = false
  }
}

async function loadMonitoring() {
  loading.monitoring = true
  try {
    const response = await analyticsApi.monitoringOverview()
    responses.monitoring = stringify(response.data)
  } catch (error) {
    responses.monitoring = stringify(error.response?.data || error.message)
  } finally {
    loading.monitoring = false
  }
}

async function loadUserBehavior() {
  loading.userBehavior = true
  try {
    const response = await analyticsApi.userBehavior(buildParams(behaviorFilters))
    responses.userBehavior = stringify(response.data)
  } catch (error) {
    responses.userBehavior = stringify(error.response?.data || error.message)
  } finally {
    loading.userBehavior = false
  }
}

async function loadFaceStats() {
  loading.faceStats = true
  try {
    const response = await analyticsApi.faceRecognitionStats(buildParams(faceStatsFilters))
    responses.faceStats = stringify(response.data)
  } catch (error) {
    responses.faceStats = stringify(error.response?.data || error.message)
  } finally {
    loading.faceStats = false
  }
}

async function loadModelPerformance() {
  loading.modelPerformance = true
  try {
    const response = await analyticsApi.modelPerformance(buildParams(modelFilters))
    responses.modelPerformance = stringify(response.data)
  } catch (error) {
    responses.modelPerformance = stringify(error.response?.data || error.message)
  } finally {
    loading.modelPerformance = false
  }
}

async function loadDataQuality() {
  loading.dataQuality = true
  try {
    const response = await analyticsApi.dataQuality(buildParams(dataQualityFilters))
    responses.dataQuality = stringify(response.data)
  } catch (error) {
    responses.dataQuality = stringify(error.response?.data || error.message)
  } finally {
    loading.dataQuality = false
  }
}

async function loadWebhookStats() {
  loading.webhookStats = true
  try {
    const response = await webhookApi.stats(buildParams(webhookFilters))
    responses.webhookStats = stringify(response.data)
  } catch (error) {
    responses.webhookStats = stringify(error.response?.data || error.message)
  } finally {
    loading.webhookStats = false
  }
}

async function loadWebhookEvents() {
  loading.webhookEvents = true
  try {
    const response = await webhookApi.listEvents()
    responses.webhookEvents = stringify(response.data)
  } catch (error) {
    responses.webhookEvents = stringify(error.response?.data || error.message)
  } finally {
    loading.webhookEvents = false
  }
}

async function loadWebhookEventLogs() {
  loading.webhookEventLogs = true
  try {
    const response = await webhookApi.listEventLogs()
    responses.webhookEventLogs = stringify(response.data)
  } catch (error) {
    responses.webhookEventLogs = stringify(error.response?.data || error.message)
  } finally {
    loading.webhookEventLogs = false
  }
}

async function loadWebhookDeliveries() {
  loading.webhookDeliveries = true
  try {
    const response = await webhookApi.listDeliveries()
    responses.webhookDeliveries = stringify(response.data)
  } catch (error) {
    responses.webhookDeliveries = stringify(error.response?.data || error.message)
  } finally {
    loading.webhookDeliveries = false
  }
}
</script>
