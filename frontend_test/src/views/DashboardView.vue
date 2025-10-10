<template>
  <div class="layout-grid">
    <section class="hero">
      <div class="hero-card">
        <h1>Third-Party Integration Dashboard</h1>
        <p>
          Pantau status client Anda, cek statistik pemakaian API, dan jalankan alur enrollment serta
          authentication sebagai pihak ketiga.
        </p>
        <div class="actions">
          <button class="btn" type="button" @click="goTo('/enrollment')">Mulai Enrollment</button>
          <button class="btn secondary" type="button" @click="goTo('/face-login')">
            Jalankan Face Authentication
          </button>
        </div>
      </div>
      <div class="section">
        <h2>Profil Client</h2>
        <div class="session-summary" v-if="client">
          <div class="session-tile">
            <span>Nama</span>
            <strong>{{ client.name }}</strong>
          </div>
          <div class="session-tile">
            <span>Tier</span>
            <strong class="mono">{{ client.tier }}</strong>
          </div>
          <div class="session-tile">
            <span>API Key</span>
            <strong class="mono">{{ client.api_key }}</strong>
          </div>
          <div class="session-tile">
            <span>Domain</span>
            <strong>{{ client.domain }}</strong>
          </div>
        </div>
        <div v-else class="status-warning">
          Tidak menemukan detail client. Coba muat ulang data.
        </div>
        <div class="actions">
          <button class="secondary" type="button" @click="reload">Muat Ulang Data Client</button>
        </div>
      </div>
    </section>

    <section class="section">
      <h2>Ringkasan Penggunaan</h2>
      <div v-if="statsLoading" class="status-pill">Mengambil data statistik…</div>
      <div v-else-if="statsError" class="status-error mono">{{ statsError }}</div>
      <div v-else class="grid-two">
        <div class="info-card">
          <strong>Total Pengguna</strong>
          <span>{{ stats.total_users }} pengguna terdaftar</span>
        </div>
        <div class="info-card">
          <strong>Pengguna Enrolled</strong>
          <span>{{ stats.enrolled_users }} sudah melakukan face enrollment</span>
        </div>
        <div class="info-card">
          <strong>Face Auth Aktif</strong>
          <span>{{ stats.active_face_auth }} pengguna dapat autentikasi wajah</span>
        </div>
        <div class="info-card">
          <strong>API Calls 24 Jam</strong>
          <span>{{ stats.api_calls_last_24h }} permintaan</span>
        </div>
        <div class="info-card">
          <strong>API Calls 7 Hari</strong>
          <span>{{ stats.api_calls_last_7d }} permintaan</span>
        </div>
        <div class="info-card">
          <strong>Total Enrollment</strong>
          <span>{{ stats.total_enrollments }} sesi tersimpan</span>
        </div>
        <div class="info-card">
          <strong>Auth Berhasil</strong>
          <span>{{ stats.successful_authentications }} kali sukses</span>
        </div>
        <div class="info-card">
          <strong>Auth Gagal</strong>
          <span>{{ stats.failed_authentications }} kali gagal</span>
        </div>
        <div class="info-card">
          <strong>Webhook Success Rate</strong>
          <span>{{ stats.webhook_success_rate }}%</span>
        </div>
      </div>
    </section>

    <section class="section">
      <h2>Menu Cepat</h2>
      <div class="grid-two">
        <button class="info-card" type="button" @click="goTo('/account')">
          <strong>Manajemen Akun</strong>
          <span>Kelola profil, update data pengguna, lihat device & security alerts.</span>
        </button>
        <button class="info-card" type="button" @click="goTo('/recognition-data')">
          <strong>Analisa Data Embedding</strong>
          <span>Periksa embedding, sesi enrollment, serta riwayat autentikasi.</span>
        </button>
        <button class="info-card" type="button" @click="goTo('/streaming')">
          <strong>Streaming & WebRTC</strong>
          <span>Monitoring sesi streaming, signaling, dan troubleshooting WebRTC.</span>
        </button>
        <button class="info-card" type="button" @click="goTo('/analytics')">
          <strong>Dashboard Analytics</strong>
          <span>Tarik data log autentikasi, security alert, dan statistik performa.</span>
        </button>
      </div>
    </section>
  </div>
</template>

<script setup>
import { computed, onMounted, ref, watch } from 'vue'
import { useRouter } from 'vue-router'
import { useAuthStore } from '../stores/auth'
import { clientApi } from '../services/api'

const router = useRouter()
const authStore = useAuthStore()

const client = computed(() => authStore.client)
const stats = ref({
  total_users: 0,
  enrolled_users: 0,
  active_face_auth: 0,
  total_enrollments: 0,
  successful_authentications: 0,
  failed_authentications: 0,
  api_calls_last_24h: 0,
  api_calls_last_7d: 0,
  webhook_success_rate: 0
})
const statsLoading = ref(false)
const statsError = ref('')

async function loadStats() {
  if (!authStore.client?.id) return
  statsLoading.value = true
  statsError.value = ''
  try {
    const response = await clientApi.stats(authStore.client.id)
    stats.value = response.data
  } catch (error) {
    statsError.value = JSON.stringify(error.response?.data || error.message, null, 2)
  } finally {
    statsLoading.value = false
  }
}

function reload() {
  authStore.refreshClientDetails()
  loadStats()
}

onMounted(() => {
  if (authStore.isAuthenticated) {
    loadStats()
  }
})

watch(
  () => authStore.client?.id,
  (val) => {
    if (val) {
      loadStats()
    }
  }
)

function goTo(path) {
  router.push(path)
}
</script>
