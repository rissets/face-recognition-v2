<template>
  <div class="layout-grid">
    <section class="section">
      <h2>Registration</h2>
      <p>Create a new user using the registration endpoint.</p>
      <form class="form-grid" @submit.prevent="register">
        <div class="field">
          <label>Email</label>
          <input v-model="registerForm.email" type="email" required />
        </div>
        <div class="field">
          <label>Username</label>
          <input v-model="registerForm.username" required />
        </div>
        <div class="field">
          <label>First Name</label>
          <input v-model="registerForm.first_name" />
        </div>
        <div class="field">
          <label>Last Name</label>
          <input v-model="registerForm.last_name" />
        </div>
        <div class="field">
          <label>Phone Number</label>
          <input v-model="registerForm.phone_number" placeholder="+62812..." />
        </div>
        <div class="field">
          <label>Password</label>
          <input v-model="registerForm.password" type="password" required />
        </div>
        <div class="field">
          <label>Confirm Password</label>
          <input v-model="registerForm.password_confirm" type="password" required />
        </div>
        <div class="actions">
          <button type="submit" :disabled="loaders.register">Register</button>
          <span v-if="registerError" class="status-error">{{ registerError }}</span>
        </div>
      </form>
      <div v-if="registerResponse" class="response-card">
        <pre>{{ registerResponse }}</pre>
      </div>
    </section>

    <section class="section">
      <h2>Profile</h2>
      <div class="actions">
        <button type="button" @click="loadProfile">Load Profile</button>
      </div>
      <div v-if="profileData" class="response-card">
        <pre>{{ profileData }}</pre>
      </div>
      <h3>Update Profile</h3>
      <form class="form-grid" @submit.prevent="updateProfile">
        <div class="field">
          <label>First Name</label>
          <input v-model="updateForm.first_name" />
        </div>
        <div class="field">
          <label>Last Name</label>
          <input v-model="updateForm.last_name" />
        </div>
        <div class="field">
          <label>Phone Number</label>
          <input v-model="updateForm.phone_number" placeholder="+62812..." />
        </div>
        <div class="field">
          <label>Bio</label>
          <textarea v-model="updateForm.bio" rows="3"></textarea>
        </div>
        <div class="actions">
          <button type="submit" :disabled="loaders.updateProfile">Update</button>
          <span v-if="updateError" class="status-error">{{ updateError }}</span>
        </div>
      </form>
      <div v-if="updateResponse" class="response-card">
        <pre>{{ updateResponse }}</pre>
      </div>
    </section>

    <section class="section">
      <h2>Devices & History</h2>
      <div class="actions">
        <button type="button" @click="loadDevices">Device List</button>
        <button type="button" @click="loadAuthHistory">Auth History</button>
        <button type="button" @click="loadSecurityAlerts">Security Alerts</button>
      </div>
      <div v-if="devicesData" class="response-card">
        <h3>Devices</h3>
        <pre>{{ devicesData }}</pre>
      </div>
      <div v-if="historyData" class="response-card">
        <h3>Authentication History</h3>
        <pre>{{ historyData }}</pre>
      </div>
      <div v-if="alertsData" class="response-card">
        <h3>Security Alerts</h3>
        <pre>{{ alertsData }}</pre>
      </div>
    </section>
  </div>
</template>

<script setup>
import { reactive, ref } from 'vue'
import { useAuthStore } from '../stores/auth'
import { authApi } from '../services/api'

const authStore = useAuthStore()

const registerForm = reactive({
  email: '',
  username: '',
  first_name: '',
  last_name: '',
  phone_number: '',
  password: '',
  password_confirm: ''
})

const updateForm = reactive({
  first_name: '',
  last_name: '',
  phone_number: '',
  bio: ''
})

const loaders = reactive({
  register: false,
  updateProfile: false
})

const registerResponse = ref('')
const registerError = ref('')
const profileData = ref('')
const updateResponse = ref('')
const updateError = ref('')
const devicesData = ref('')
const historyData = ref('')
const alertsData = ref('')

function stringify(data) {
  return JSON.stringify(data, null, 2)
}

async function register() {
  loaders.register = true
  registerError.value = ''
  try {
    const payload = JSON.parse(JSON.stringify(registerForm))
    const response = await authApi.register(payload)
    registerResponse.value = stringify(response.data)
  } catch (error) {
    registerError.value = stringify(error.response?.data || error.message)
  } finally {
    loaders.register = false
  }
}

async function loadProfile() {
  try {
    const data = await authStore.fetchProfile()
    profileData.value = stringify(data)
    Object.assign(updateForm, {
      first_name: data.first_name || '',
      last_name: data.last_name || '',
      phone_number: data.phone_number || '',
      bio: data.bio || ''
    })
  } catch (error) {
    profileData.value = stringify(error.response?.data || error.message)
  }
}

async function updateProfile() {
  loaders.updateProfile = true
  updateError.value = ''
  try {
    const payload = JSON.parse(JSON.stringify(updateForm))
    const response = await authApi.updateProfile(payload)
    updateResponse.value = stringify(response.data)
    await loadProfile()
  } catch (error) {
    updateError.value = stringify(error.response?.data || error.message)
  } finally {
    loaders.updateProfile = false
  }
}

async function loadDevices() {
  try {
    const response = await authApi.userDevices()
    devicesData.value = stringify(response.data)
  } catch (error) {
    devicesData.value = stringify(error.response?.data || error.message)
  }
}

async function loadAuthHistory() {
  try {
    const response = await authApi.authHistory()
    historyData.value = stringify(response.data)
  } catch (error) {
    historyData.value = stringify(error.response?.data || error.message)
  }
}

async function loadSecurityAlerts() {
  try {
    const response = await authApi.securityAlerts()
    alertsData.value = stringify(response.data)
  } catch (error) {
    alertsData.value = stringify(error.response?.data || error.message)
  }
}
</script>
