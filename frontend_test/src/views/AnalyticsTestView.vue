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
  </div>
</template>

<script setup>
import { reactive, ref } from 'vue'
import { analyticsApi } from '../services/api'

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

const dashboardPeriod = ref('30_days')

const loading = reactive({
  authLogs: false,
  alerts: false,
  dashboard: false,
  statistics: false
})

const responses = reactive({
  authLogs: '',
  alerts: '',
  dashboard: '',
  statistics: ''
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
</script>
