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
        <div class="session-summary" v-if="clientProfile">
          <div class="session-tile">
            <span>Nama</span>
            <strong>{{ clientProfile.name }}</strong>
          </div>
          <div class="session-tile">
            <span>Tier</span>
            <strong class="mono">{{ clientProfile.tier }}</strong>
          </div>
          <div class="session-tile">
            <span>Status</span>
            <strong>{{ statusLabel(clientProfile.status) }}</strong>
          </div>
          <div class="session-tile">
            <span>API Key</span>
            <strong class="mono">{{ clientProfile.api_key || '—' }}</strong>
          </div>
          <div class="session-tile">
            <span>Domain</span>
            <strong>{{ clientProfile.domain || '—' }}</strong>
          </div>
          <div class="session-tile">
            <span>Rate Limit</span>
            <strong>
              {{ formatNumber(rateLimits.per_hour) }} / jam · {{ formatNumber(rateLimits.per_day) }} / hari
            </strong>
          </div>
          <div class="session-tile">
            <span>Terakhir Aktivitas</span>
            <strong>{{ formatDate(clientOverview?.client?.last_activity) }}</strong>
          </div>
        </div>
        <div v-else class="status-warning">
          Tidak menemukan detail client. Coba muat ulang data.
        </div>
        <div class="actions">
          <button class="secondary" type="button" @click="reload" :disabled="overviewLoading">
            {{ overviewLoading ? 'Memuat…' : 'Muat Ulang Data Client' }}
          </button>
        </div>
        <div v-if="activeFeatureFlags.length" class="chip-list chip-list--muted">
          <span class="chip chip--muted" v-for="feature in activeFeatureFlags" :key="feature.key">
            {{ feature.label }}
          </span>
        </div>
      </div>
    </section>

    <section class="section">
      <div class="section-header">
        <div>
          <h2>Ringkasan Penggunaan</h2>
          <p class="section-subtitle">Angka terbaru dari aktivitas tenant Anda.</p>
        </div>
        <div class="actions">
          <button class="secondary" type="button" @click="reload" :disabled="overviewLoading">
            {{ overviewLoading ? 'Memuat…' : 'Segarkan' }}
          </button>
        </div>
      </div>
      <div v-if="overviewLoading" class="status-pill">Mengambil data statistik…</div>
      <div v-else-if="overviewError" class="status-error mono">{{ overviewError }}</div>
      <div v-else-if="analyticsCards.length" class="stat-grid">
        <div class="stat-card" v-for="card in analyticsCards" :key="card.label">
          <span class="stat-card__label">{{ card.label }}</span>
          <strong>{{ card.value }}</strong>
          <small>{{ card.caption }}</small>
        </div>
      </div>
      <p v-else class="status-warning">Statistik belum tersedia untuk client ini.</p>
    </section>

    <section class="section">
      <h2>Aktivitas Terkini</h2>
      <div class="layout-split">
        <div class="info-panel">
          <header class="info-panel__header">
            <h3>API Usage</h3>
            <span class="mono small">{{ formatNumber(usageMetrics.today) }} hari ini</span>
          </header>
          <div class="info-panel__body">
            <div class="summary-grid">
              <div>
                <span class="summary-label">Hari Ini</span>
                <strong>{{ formatNumber(usageMetrics.today) }}</strong>
              </div>
              <div>
                <span class="summary-label">7 Hari Terakhir</span>
                <strong>{{ formatNumber(usageMetrics.last_7_days) }}</strong>
              </div>
            </div>
            <table v-if="recentUsage.length" class="mini-table">
              <thead>
                <tr>
                  <th>Endpoint</th>
                  <th>Status</th>
                  <th>Waktu</th>
                </tr>
              </thead>
              <tbody>
                <tr v-for="entry in recentUsage" :key="entry.created_at + entry.endpoint">
                  <td>
                    <strong>{{ entry.endpoint }}</strong>
                    <div class="mono small">{{ entry.method }}</div>
                  </td>
                  <td>{{ entry.status_code }}</td>
                  <td>{{ formatDate(entry.created_at) }}</td>
                </tr>
              </tbody>
            </table>
            <p v-else class="status-warning">Belum ada permintaan tercatat.</p>
          </div>
        </div>
        <div class="info-panel">
          <header class="info-panel__header">
            <h3>Webhook & Domain</h3>
            <span class="mono small">{{ formatNumber(webhookSummary.success) }} sukses</span>
          </header>
          <div class="info-panel__body">
            <div class="summary-grid">
              <div>
                <span class="summary-label">Total Event</span>
                <strong>{{ formatNumber(webhookSummary.total) }}</strong>
              </div>
              <div>
                <span class="summary-label">Berhasil</span>
                <strong>{{ formatNumber(webhookSummary.success) }}</strong>
              </div>
              <div>
                <span class="summary-label">Gagal</span>
                <strong>{{ formatNumber(webhookSummary.failed) }}</strong>
              </div>
            </div>
            <div>
              <h4>Allowed Domains</h4>
              <ul v-if="allowedDomains.length" class="bullet-list">
                <li v-for="domain in allowedDomains" :key="domain">{{ domain }}</li>
              </ul>
              <p v-else class="status-warning">Belum ada domain yang didaftarkan.</p>
            </div>
            <div v-if="webhookEvents.length">
              <h4>Event Webhook</h4>
              <div class="chip-list chip-list--muted">
                <span class="chip chip--muted" v-for="event in webhookEvents" :key="event">
                  {{ event }}
                </span>
              </div>
            </div>
          </div>
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
        <!-- <button class="info-card" type="button" @click="goTo('/recognition-data')">
          <strong>Analisa Data Embedding</strong>
          <span>Periksa embedding, sesi enrollment, serta riwayat autentikasi.</span>
        </button> -->
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
import { coreApi } from '../services/api'

const router = useRouter()
const authStore = useAuthStore()

const overview = ref(null)
const overviewLoading = ref(false)
const overviewError = ref('')

const numberFormatter = new Intl.NumberFormat('id-ID')

const CLIENT_STATUS_LABELS = {
  active: 'Aktif',
  suspended: 'Disuspend',
  inactive: 'Nonaktif',
  trial: 'Percobaan'
}

const clientOverview = computed(() => overview.value)

const clientProfile = computed(() => {
  if (!authStore.client && !clientOverview.value?.client) {
    return null
  }
  return {
    ...authStore.client,
    ...clientOverview.value?.client
  }
})

const rateLimits = computed(() => clientOverview.value?.rate_limits || { per_hour: 0, per_day: 0 })

const analyticsCards = computed(() => {
  const analytics = clientOverview.value?.analytics
  if (!analytics) return []
  return [
    {
      label: 'Total Pengguna',
      value: formatNumber(analytics.total_users || 0),
      caption: 'Pengguna yang terdaftar'
    },
    {
      label: 'Sudah Enrollment',
      value: formatNumber(analytics.enrolled_users || 0),
      caption: 'Telah menyelesaikan face enrollment'
    },
    {
      label: 'Face Auth Aktif',
      value: formatNumber(analytics.active_face_auth || 0),
      caption: 'Pengguna siap autentikasi wajah'
    },
    {
      label: 'Total Enrollment',
      value: formatNumber(analytics.total_enrollments || 0),
      caption: 'Rekaman enrollment tersimpan'
    },
    {
      label: 'Autentikasi Berhasil',
      value: formatNumber(analytics.auth_success || 0),
      caption: 'Dalam 7 hari terakhir'
    },
    {
      label: 'Autentikasi Gagal',
      value: formatNumber(analytics.auth_failed || 0),
      caption: 'Dalam 7 hari terakhir'
    },
    {
      label: 'Sesi Aktif',
      value: formatNumber(analytics.active_sessions || 0),
      caption: 'Autentikasi yang sedang berjalan'
    }
  ]
})

const usageMetrics = computed(() => clientOverview.value?.usage || { today: 0, last_7_days: 0 })
const recentUsage = computed(() => clientOverview.value?.usage?.recent || [])

const webhookSummary = computed(() => clientOverview.value?.webhook?.summary || {
  total: 0,
  success: 0,
  failed: 0,
  retrying: 0
})
const webhookEvents = computed(() => clientOverview.value?.webhook?.events || [])
const allowedDomains = computed(() => clientOverview.value?.allowed_domains || [])

const activeFeatureFlags = computed(() => {
  const features = clientOverview.value?.features || {}
  return Object.entries(features)
    .filter(([, value]) => typeof value === 'boolean' && value)
    .map(([key]) => ({
      key,
      label: humanizeKey(key)
    }))
    .sort((a, b) => a.label.localeCompare(b.label))
})

function humanizeKey(key) {
  if (!key) return ''
  return key
    .replace(/_/g, ' ')
    .replace(/\b\w/g, (char) => char.toUpperCase())
}

function statusLabel(status) {
  return CLIENT_STATUS_LABELS[status] || status || '—'
}

function formatNumber(value) {
  if (value == null || Number.isNaN(value)) return '0'
  return numberFormatter.format(value)
}

function formatDate(value) {
  if (!value) return '—'
  try {
    return new Date(value).toLocaleString('id-ID', {
      dateStyle: 'medium',
      timeStyle: 'short'
    })
  } catch {
    return value
  }
}

function parseError(error) {
  const payload = error?.response?.data
  if (!payload) {
    return error?.message || 'Terjadi kesalahan'
  }
  if (typeof payload === 'string') {
    return payload
  }
  if (payload.detail) {
    return payload.detail
  }
  return JSON.stringify(payload, null, 2)
}

async function loadOverview() {
  if (!authStore.isAuthenticated) return
  overviewLoading.value = true
  overviewError.value = ''
  try {
    await authStore.refreshClientDetails()
    const response = await coreApi.clientInfo()
    overview.value = response.data
  } catch (error) {
    overviewError.value = parseError(error)
  } finally {
    overviewLoading.value = false
  }
}

function reload() {
  loadOverview()
}

function goTo(path) {
  router.push(path)
}

onMounted(() => {
  if (authStore.isAuthenticated) {
    loadOverview()
  }
})

watch(
  () => authStore.isAuthenticated,
  (isAuthenticated) => {
    if (isAuthenticated) {
      loadOverview()
    } else {
      overview.value = null
    }
  }
)
</script>
