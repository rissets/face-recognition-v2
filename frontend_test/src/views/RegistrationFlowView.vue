<template>
  <div class="layout-grid registration-flow">
    <section class="section">
      <header>
        <h1>Registrasi &amp; Enrollment Wajah</h1>
        <p>
          Ikuti tiga langkah berikut: daftar akun baru, masuk secara otomatis, dan lanjutkan enrollment wajah
          menggunakan kamera perangkat Anda.
        </p>
      </header>

      <div class="step-indicator">
        <span :class="{ active: step === 1, done: step > 1 }">1. Registrasi</span>
        <span :class="{ active: step === 2, done: step > 2 }">2. Login Otomatis</span>
        <span :class="{ active: step === 3 }">3. Enrollment</span>
      </div>
    </section>

    <section v-if="step === 1" class="section">
      <h2>Daftar Akun Baru</h2>
      <form class="form-grid" @submit.prevent="submitRegistration">
        <div class="field">
          <label>Email</label>
          <input v-model="registerForm.email" type="email" required placeholder="user@example.com" />
        </div>
        <div class="field">
          <label>Username</label>
          <input v-model="registerForm.username" required />
        </div>
        <div class="field">
          <label>Nama Depan</label>
          <input v-model="registerForm.first_name" required />
        </div>
        <div class="field">
          <label>Nama Belakang</label>
          <input v-model="registerForm.last_name" required />
        </div>
        <div class="field">
          <label>No. Telepon</label>
          <input v-model="registerForm.phone_number" placeholder="+62812..." />
        </div>
        <div class="field">
          <label>Password</label>
          <input v-model="registerForm.password" type="password" required minlength="8" />
        </div>
        <div class="field">
          <label>Konfirmasi Password</label>
          <input v-model="registerForm.password_confirm" type="password" required minlength="8" />
        </div>
        <div class="actions">
          <button type="submit" :disabled="loading.register">Daftar</button>
          <RouterLink class="btn secondary" to="/login">Sudah punya akun?</RouterLink>
        </div>
        <div v-if="errors.register" class="status-error">{{ errors.register }}</div>
        <div v-if="registerResponse" class="status-success">
          Registrasi berhasil untuk {{ registerResponse.email }}. Melanjutkan ke login otomatis...
        </div>
      </form>
    </section>

    <section v-else-if="step === 2" class="section">
      <h2>Login Otomatis</h2>
      <p>
        Mengautentikasi ke sistem menggunakan kredensial yang baru dibuat. Setelah berhasil, Anda akan diarahkan ke
        langkah enrollment.
      </p>
      <div class="card">
        <p v-if="loading.login">Memproses login...</p>
        <p v-else-if="errors.login" class="status-error">{{ errors.login }}</p>
        <p v-else>Login berhasil. Mengalihkan ke langkah enrollment...</p>
        <div class="actions" v-if="errors.login">
          <button type="button" @click="attemptLogin" :disabled="loading.login">Coba Lagi</button>
          <RouterLink class="btn secondary" to="/login">Login manual</RouterLink>
        </div>
      </div>
    </section>

    <section v-else class="section">
      <h2>Enrollment Wajah</h2>
      <div class="info-card success">
        <strong>Selamat datang, {{ authStore.profile?.full_name || authStore.profile?.email }}</strong>
        <span>
          Akun Anda sudah terautentikasi. Lanjutkan dengan proses enrollment untuk menyimpan sample wajah ke sistem.
        </span>
      </div>

      <div class="enrollment-layout">
        <div class="enrollment-instructions">
          <h3>Petunjuk</h3>
          <ul>
            <li>Pastikan kamera memiliki pencahayaan yang cukup.</li>
            <li>Klik <strong>Start</strong> lalu hadapkan wajah ke kamera.</li>
            <li>Kedipkan mata dan gerakkan kepala perlahan untuk memenuhi syarat liveness.</li>
            <li>Setelah selesai, Anda dapat melanjutkan ke dashboard atau melakukan autentikasi wajah.</li>
          </ul>
          <div class="actions">
            <RouterLink class="btn secondary" to="/dashboard">Ke Dashboard</RouterLink>
            <RouterLink class="btn" to="/face-login">Coba Face Login</RouterLink>
          </div>
        </div>
        <div class="enrollment-widget">
          <WebRTCEnrollment v-if="authStore.isAuthenticated" />
          <div v-else class="status-error">
            Sesi login kadaluwarsa. Silakan <RouterLink to="/login">login</RouterLink> kembali untuk melanjutkan.
          </div>
        </div>
      </div>
    </section>
  </div>
</template>

<script setup>
import { reactive, ref } from 'vue'
import { RouterLink } from 'vue-router'
import { authApi } from '../services/api'
import { useAuthStore } from '../stores/auth'
import WebRTCEnrollment from '../components/WebRTCEnrollment.vue'

const authStore = useAuthStore()

const step = ref(authStore.isAuthenticated ? 3 : 1)
const registerForm = reactive({
  email: '',
  username: '',
  first_name: '',
  last_name: '',
  phone_number: '',
  password: '',
  password_confirm: ''
})

const loading = reactive({
  register: false,
  login: false
})

const errors = reactive({
  register: '',
  login: ''
})

const registerResponse = ref(null)
const credentials = ref(null)

function stringifyError(error, fallback = 'Terjadi kesalahan. Silakan coba lagi.') {
  const data = error?.response?.data
  if (!data) return error?.message || fallback
  if (typeof data === 'string') return data
  if (data.detail) return data.detail
  if (data.error) return data.error
  if (Array.isArray(data) && data.length) return JSON.stringify(data[0])
  if (data.non_field_errors) return data.non_field_errors.join(', ')
  return JSON.stringify(data)
}

async function submitRegistration() {
  loading.register = true
  errors.register = ''
  errors.login = ''
  registerResponse.value = null

  try {
    const payload = JSON.parse(JSON.stringify(registerForm))
    const response = await authApi.register(payload)
    registerResponse.value = response.data
    credentials.value = {
      email: registerForm.email,
      password: registerForm.password
    }
    step.value = 2
    await attemptLogin()
  } catch (error) {
    errors.register = stringifyError(error, 'Registrasi gagal.')
  } finally {
    loading.register = false
  }
}

async function attemptLogin() {
  if (!credentials.value) return
  loading.login = true
  errors.login = ''

  try {
    await authStore.login({
      email: credentials.value.email,
      password: credentials.value.password,
      deviceInfo: {
        device_name: 'Registration Flow',
        device_id: 'registration-flow',
        device_type: 'web'
      }
    })
    step.value = 3
  } catch (error) {
    errors.login = stringifyError(error, 'Login otomatis gagal.')
  } finally {
    loading.login = false
  }
}
</script>

<style scoped>
.registration-flow .step-indicator {
  display: flex;
  gap: 1rem;
  font-weight: 600;
  margin-top: 1rem;
}

.registration-flow .step-indicator span {
  padding: 0.25rem 0.75rem;
  border-radius: 999px;
  background: var(--color-muted);
  color: var(--color-text-muted);
}

.registration-flow .step-indicator span.active {
  background: var(--color-primary);
  color: #fff;
}

.registration-flow .step-indicator span.done {
  background: var(--color-success);
  color: #fff;
}

.enrollment-layout {
  display: grid;
  gap: 1.5rem;
  grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
}

.enrollment-instructions ul {
  margin: 0.5rem 0 1.5rem;
  padding-left: 1.25rem;
}

.enrollment-widget {
  background: var(--color-surface);
  border-radius: 1rem;
  padding: 1rem;
  box-shadow: var(--shadow-small);
}

.info-card.success {
  border-left: 4px solid var(--color-success);
}
</style>
