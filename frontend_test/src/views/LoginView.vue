<template>
  <div class="layout-grid">
    <section class="section">
      <h1>Masuk ke FaceRec Tester</h1>
      <p>Gunakan email dan password akun Django Anda. Setelah login Anda dapat melakukan enrollment dan face login realtime.</p>
      <form class="form-grid" @submit.prevent="submit">
        <div class="field">
          <label for="email">Email</label>
          <input id="email" v-model="email" type="email" required autocomplete="username" placeholder="user@example.com" />
        </div>
        <div class="field">
          <label for="password">Password</label>
          <input id="password" v-model="password" type="password" required autocomplete="current-password" />
        </div>
        <div class="field">
          <label for="device-name">Nama Device (opsional)</label>
          <input id="device-name" v-model="deviceName" placeholder="API Tester" />
        </div>
        <div class="actions">
          <button type="submit" :disabled="authStore.loading">Masuk</button>
          <RouterLink class="btn secondary" to="/account" v-if="authStore.isAuthenticated">
            Kelola Akun
          </RouterLink>
          <RouterLink class="btn secondary" to="/register" v-else>
            Daftar Akun Baru
          </RouterLink>
        </div>
        <div v-if="errorMessage" class="status-error">{{ errorMessage }}</div>
      </form>
    </section>

    <section class="section">
      <h2>Login Via Face Recognition</h2>
      <p>Sudah pernah enrol wajah? Masuk langsung dengan kamera. Email bersifat opsional bila ingin mengunci ke akun tertentu.</p>
      <div class="face-login-panel">
        <div v-if="showFaceLogin" class="face-login-card">
          <FaceLoginInline :email="email" @cancel="closeFaceLogin" @completed="handleFaceLoginSuccess" />
        </div>
        <div v-else class="actions">
          <button type="button" class="btn secondary" @click="beginFaceLogin">Mulai Face Login</button>
          <RouterLink class="btn" to="/face-login">Halaman Face Login Lengkap</RouterLink>
        </div>
        <div v-if="faceLoginMessage" :class="['status-pill', faceLoginSuccess ? 'success' : 'warning']">
          {{ faceLoginMessage }}
        </div>
      </div>
    </section>

    <section class="section">
      <h2>Langkah Berikutnya</h2>
      <div class="grid-two">
        <RouterLink class="info-card" to="/enrollment" v-if="authStore.isAuthenticated">
          <strong>Enrollment Wajah</strong>
          <span>Mulai sesi streaming enrollment untuk mendaftarkan wajah Anda.</span>
        </RouterLink>
        <RouterLink class="info-card" to="/face-login" v-if="authStore.isAuthenticated">
          <strong>Login via Face Recognition</strong>
          <span>Uji autentikasi realtime dan lihat skor similarity langsung.</span>
        </RouterLink>
        <div class="info-card" v-if="authStore.profile">
          <strong>Profil Saat Ini</strong>
          <span class="mono">{{ formattedProfile }}</span>
        </div>
        <RouterLink class="info-card" to="/register">
          <strong>Belum punya akun?</strong>
          <span>Daftar dan langsung lanjut ke enrollment wajah Anda.</span>
        </RouterLink>
      </div>
    </section>
  </div>
</template>

<script setup>
import { computed, ref } from 'vue'
import { useRoute, useRouter, RouterLink } from 'vue-router'
import { useAuthStore } from '../stores/auth'
import FaceLoginInline from '../components/FaceLoginInline.vue'

const email = ref('')
const password = ref('')
const deviceName = ref('API Tester')
const showFaceLogin = ref(false)
const faceLoginMessage = ref('')
const faceLoginSuccess = ref(false)

const route = useRoute()
const router = useRouter()
const authStore = useAuthStore()

const errorMessage = computed(() => {
  const err = authStore.error
  if (!err) return ''
  if (typeof err === 'string') return err
  if (err.detail) return err.detail
  return JSON.stringify(err, null, 2)
})

const formattedProfile = computed(() =>
  JSON.stringify(authStore.profile, null, 2)
)

async function submit() {
  try {
    await authStore.login({
      email: email.value,
      password: password.value,
      deviceInfo: {
        device_name: deviceName.value
      }
    })
    const redirect = route.query.redirect
    router.push(typeof redirect === 'string' ? redirect : '/dashboard')
  } catch (error) {
    console.error('Login failed', error)
  }
}

function beginFaceLogin() {
  faceLoginSuccess.value = false
  faceLoginMessage.value = email.value
    ? 'Verifikasi wajah untuk akun ini. Kamera akan aktif dan lakukan kedipan mata.'
    : 'Mode identifikasi aktif. Kamera akan mencari wajah yang terdaftar tanpa email.'
  faceLoginSuccess.value = false
  showFaceLogin.value = true
}

function closeFaceLogin() {
  showFaceLogin.value = false
  faceLoginMessage.value = ''
}

async function handleFaceLoginSuccess(payload) {
  try {
    if (payload.accessToken && payload.refreshToken) {
      authStore.setTokens({ access: payload.accessToken, refresh: payload.refreshToken })
      await authStore.fetchProfile()
    }
    faceLoginSuccess.value = true
    faceLoginMessage.value = `Berhasil login sebagai ${payload.user?.email || 'pengguna'}`
    showFaceLogin.value = false
    router.push('/dashboard')
  } catch (error) {
    faceLoginSuccess.value = false
    faceLoginMessage.value = 'Token login tidak dapat disimpan. Silakan ulangi proses.'
    console.error('Face login token handling failed', error)
  }
}
</script>
