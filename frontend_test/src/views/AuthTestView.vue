<template>
  <div class="layout-grid">
    <section class="section">
      <div class="section-header">
        <div>
          <h2>Informasi Client</h2>
          <p class="section-subtitle">
            Ikhtisar tenant, fitur yang aktif, serta aktivitas penggunaan API terkini.
          </p>
        </div>
        <div class="actions">
          <button
            type="button"
            class="secondary"
            @click="loadClientOverview"
            :disabled="overviewLoading"
          >
            {{ overviewLoading ? 'Memuat…' : 'Muat Ulang' }}
          </button>
          <button
            type="button"
            class="secondary"
            @click="loadUsers"
            :disabled="usersLoading"
          >
            {{ usersLoading ? 'Memuat Pengguna…' : 'Muat Daftar Pengguna' }}
          </button>
        </div>
      </div>

      <p v-if="overviewError" class="status-error mono">{{ overviewError }}</p>
      <div v-else-if="overviewLoading" class="status-pill">Memuat informasi client…</div>
      <div v-else-if="clientOverview" class="client-overview">
        <div class="client-identity">
          <div>
            <h3>{{ clientOverview.client.name }}</h3>
            <div class="mono small">{{ clientOverview.client.client_id }}</div>
          </div>
          <div class="client-tags">
            <span class="status-pill" :class="statusClass(clientOverview.client.status)">
              {{ statusLabel(clientOverview.client.status) }}
            </span>
            <span class="status-pill badge-tier">{{ clientOverview.client.tier }}</span>
          </div>
        </div>

        <div class="info-grid">
          <div class="info-card">
            <span class="info-card__label">Kontak Utama</span>
            <strong>{{ clientOverview.client.contact.name }}</strong>
            <div class="mono small">{{ clientOverview.client.contact.email }}</div>
          </div>
          <div class="info-card">
            <span class="info-card__label">Rate Limit</span>
            <strong>{{ formatNumber(clientOverview.rate_limits.per_hour) }} / jam</strong>
            <div class="mono small">{{ formatNumber(clientOverview.rate_limits.per_day) }} / hari</div>
          </div>
          <div class="info-card">
            <span class="info-card__label">Aktivitas</span>
            <strong>{{ formatDate(clientOverview.client.last_activity) }}</strong>
            <div class="mono small">
              Dibuat {{ formatDate(clientOverview.timestamps.created_at) }}
            </div>
          </div>
        </div>

        <div class="stat-grid">
          <div class="stat-card" v-for="card in analyticsCards" :key="card.label">
            <span class="stat-card__label">{{ card.label }}</span>
            <strong>{{ card.value }}</strong>
            <small>{{ card.caption }}</small>
          </div>
        </div>

        <div class="layout-split">
          <div class="info-panel">
            <header class="info-panel__header">
              <h3>Fitur Aktif</h3>
              <span class="mono small">{{ activeFeatureFlags.length }} fitur</span>
            </header>
            <div v-if="activeFeatureFlags.length" class="chip-list">
              <span class="chip" v-for="feature in activeFeatureFlags" :key="feature.key">
                {{ feature.label }}
              </span>
            </div>
            <p v-else class="status-warning">Belum ada fitur yang aktif.</p>
            <div v-if="featureLimits.length" class="feature-meta">
              <div class="feature-meta__item" v-for="item in featureLimits" :key="item.key">
                <span>{{ item.label }}</span>
                <strong>{{ item.value }}</strong>
              </div>
            </div>
          </div>

          <div class="info-panel">
            <header class="info-panel__header">
              <h3>Domain & Webhook</h3>
              <span class="mono small">{{ webhookSummary.total }} event</span>
            </header>
            <div class="info-panel__body">
              <div>
                <h4>Allowed Domains</h4>
                <ul v-if="allowedDomains.length" class="bullet-list">
                  <li v-for="domain in allowedDomains" :key="domain">{{ domain }}</li>
                </ul>
                <p v-else class="status-warning">Belum ada domain yang ditambahkan.</p>
              </div>
              <div class="webhook-summary">
                <div>
                  <span class="summary-label">Webhook URL</span>
                  <span class="mono small">
                    {{ clientOverview.webhook.url || 'Belum dikonfigurasi' }}
                  </span>
                </div>
                <div class="summary-grid">
                  <div>
                    <span class="summary-label">Berhasil</span>
                    <strong>{{ formatNumber(webhookSummary.success) }}</strong>
                  </div>
                  <div>
                    <span class="summary-label">Gagal</span>
                    <strong>{{ formatNumber(webhookSummary.failed) }}</strong>
                  </div>
                  <div>
                    <span class="summary-label">Retry</span>
                    <strong>{{ formatNumber(webhookSummary.retrying) }}</strong>
                  </div>
                </div>
              </div>
              <div v-if="webhookEvents.length" class="chip-list chip-list--muted">
                <span class="chip chip--muted" v-for="event in webhookEvents" :key="event">
                  {{ event }}
                </span>
              </div>
            </div>
          </div>
        </div>

        <div class="layout-split">
          <div class="info-panel">
            <header class="info-panel__header">
              <h3>Aktivitas API</h3>
              <span class="mono small">{{ formatNumber(clientOverview.usage.today) }} hari ini</span>
            </header>
            <div class="info-panel__body">
              <div class="summary-grid">
                <div>
                  <span class="summary-label">Hari Ini</span>
                  <strong>{{ formatNumber(clientOverview.usage.today) }}</strong>
                </div>
                <div>
                  <span class="summary-label">7 Hari Terakhir</span>
                  <strong>{{ formatNumber(clientOverview.usage.last_7_days) }}</strong>
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
              <p v-else class="status-warning">Belum ada aktivitas terbaru.</p>
            </div>
          </div>

          <div class="info-panel">
            <header class="info-panel__header">
              <h3>Insight Pengguna</h3>
              <span class="mono small">Progres enrollment dan face auth</span>
            </header>
            <ul class="insight-list">
              <li>
                <span>Pengguna belum enrollment</span>
                <strong>{{ formatNumber(outstandingEnrollments) }}</strong>
              </li>
              <li>
                <span>Face auth nonaktif</span>
                <strong>{{ formatNumber(faceAuthDisabled) }}</strong>
              </li>
              <li>
                <span>Sesi autentikasi aktif</span>
                <strong>{{ formatNumber(clientOverview.analytics.active_sessions || 0) }}</strong>
              </li>
            </ul>
          </div>
        </div>
      </div>
      <p v-else class="status-warning">Client belum dimuat. Klik "Muat Ulang" untuk mencoba lagi.</p>
    </section>

    <section class="section">
      <h2>Tambah Pengguna Client</h2>
      <p class="section-subtitle">Formulir sederhana tanpa perlu menyusun JSON manual.</p>
      <form class="form-grid form-grid--two-column" @submit.prevent="createUser">
        <div class="field">
          <label for="external-id">External User ID</label>
          <input
            id="external-id"
            v-model.trim="userForm.externalUserId"
            required
            placeholder="john.doe"
          />
        </div>
        <div class="field">
          <label for="full-name">Nama Lengkap</label>
          <input id="full-name" v-model.trim="userForm.fullName" placeholder="John Doe" />
        </div>
        <div class="field">
          <label for="email">Email</label>
          <input id="email" v-model.trim="userForm.email" type="email" placeholder="john@company.com" />
        </div>
        <div class="field">
          <label for="phone">Nomor Telepon</label>
          <input id="phone" v-model.trim="userForm.phone" placeholder="+62 812-1234-5678" />
        </div>
        <div class="field">
          <label for="role">Peran / Departemen</label>
          <input id="role" v-model.trim="userForm.role" placeholder="Operations" />
        </div>
        <div class="field field--textarea">
          <label for="notes">Catatan</label>
          <textarea
            id="notes"
            v-model.trim="userForm.notes"
            rows="3"
            placeholder="Catatan khusus atau preferensi pengguna"
          ></textarea>
        </div>
        <div class="field checkbox-field">
          <label>
            <input type="checkbox" v-model="userForm.faceAuthEnabled" />
            Aktifkan face authentication setelah dibuat
          </label>
        </div>
        <div class="actions">
          <button type="submit" :disabled="userSaving">
            {{ userSaving ? 'Menyimpan…' : 'Simpan Pengguna' }}
          </button>
          <span v-if="feedback.success" class="status-pill success">{{ feedback.success }}</span>
          <span v-if="feedback.error" class="status-error mono">{{ feedback.error }}</span>
        </div>
      </form>
    </section>

    <section class="section">
      <div class="section-header">
        <h2>Daftar Pengguna Terdaftar</h2>
        <div class="actions">
          <button type="button" class="secondary" @click="loadUsers" :disabled="usersLoading">
            {{ usersLoading ? 'Memuat…' : 'Muat Ulang' }}
          </button>
        </div>
      </div>

      <div v-if="usersLoading" class="status-pill">Memuat daftar pengguna…</div>
      <div v-else-if="!hasUsers" class="status-warning">Belum ada pengguna di client ini.</div>
      <div v-else class="table-responsive">
        <table class="data-table">
          <thead>
            <tr>
              <th>Pengguna</th>
              <th>Kontak</th>
              <th>Enrollment</th>
              <th>Face Auth</th>
              <th>Terakhir Aktivitas</th>
              <th>Aksi</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="user in users" :key="user.id">
              <td>
                <strong>{{ user.display_name }}</strong>
                <div class="mono small">{{ user.external_user_id }}</div>
              </td>
              <td>
                <div>{{ user.profile?.email || '—' }}</div>
                <div v-if="user.profile?.phone || user.profile?.phone_number" class="mono small">
                  {{ user.profile?.phone || user.profile?.phone_number }}
                </div>
              </td>
              <td>
                <span
                  class="status-pill"
                  :class="user.is_enrolled ? 'success' : 'warning'"
                >
                  {{ user.is_enrolled ? 'Selesai' : 'Belum' }}
                </span>
              </td>
              <td>
                <span
                  class="status-pill"
                  :class="user.face_auth_enabled ? 'success' : 'danger'"
                >
                  {{ user.face_auth_enabled ? 'Aktif' : 'Nonaktif' }}
                </span>
              </td>
              <td>{{ formatDate(user.last_recognition_at) }}</td>
              <td class="table-actions">
                <button
                  class="secondary"
                  type="button"
                  @click="toggleUser(user)"
                  :disabled="togglingUserId === user.id"
                >
                  {{
                    togglingUserId === user.id
                      ? 'Memproses…'
                      : user.face_auth_enabled
                        ? 'Nonaktifkan'
                        : 'Aktifkan'
                  }}
                </button>
                <button class="secondary" type="button" @click="loadEnrollments(user)">
                  Lihat Enrollment
                </button>
              </td>
            </tr>
          </tbody>
        </table>
      </div>
    </section>

    <section v-if="enrollments" class="section">
      <div class="section-header">
        <h2>
          Enrollment {{ enrollments.user.display_name || enrollments.user.external_user_id }}
        </h2>
        <div class="actions">
          <button class="secondary" type="button" @click="clearEnrollments">Tutup</button>
        </div>
      </div>
      <div v-if="enrollmentLoading" class="status-pill">Memuat data enrollment…</div>
      <div v-else class="response-card mono">
        <pre>{{ enrollmentsText }}</pre>
      </div>
    </section>
  </div>
</template>

<script setup>
import { computed, onMounted, reactive, ref, watch } from 'vue'
import { useAuthStore } from '../stores/auth'
import { clientUsersApi, coreApi } from '../services/api'

const authStore = useAuthStore()

const overview = ref(null)
const overviewLoading = ref(false)
const overviewError = ref('')

const users = ref([])
const usersLoading = ref(false)
const userSaving = ref(false)
const togglingUserId = ref(null)

const enrollments = ref(null)
const enrollmentsText = computed(() => JSON.stringify(enrollments.value?.data || {}, null, 2))
const enrollmentLoading = ref(false)

const feedback = reactive({
  success: '',
  error: ''
})

const userForm = reactive({
  externalUserId: '',
  fullName: '',
  email: '',
  phone: '',
  role: '',
  notes: '',
  faceAuthEnabled: true
})

const numberFormatter = new Intl.NumberFormat('id-ID')

const CLIENT_STATUS_LABELS = {
  active: 'Aktif',
  suspended: 'Disuspend',
  inactive: 'Nonaktif',
  trial: 'Percobaan'
}

const CLIENT_STATUS_CLASS = {
  active: 'success',
  suspended: 'danger',
  inactive: 'warning',
  trial: 'warning'
}

const clientOverview = computed(() => overview.value)

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
      caption: 'Pengguna menyelesaikan face enrollment'
    },
    {
      label: 'Face Auth Aktif',
      value: formatNumber(analytics.active_face_auth || 0),
      caption: 'Pengguna siap autentikasi wajah'
    },
    {
      label: 'Total Enrollment',
      value: formatNumber(analytics.total_enrollments || 0),
      caption: 'Rekaman enrollment yang tersimpan'
    },
    {
      label: 'Auth Berhasil',
      value: formatNumber(analytics.auth_success || 0),
      caption: 'Dalam 7 hari terakhir'
    },
    {
      label: 'Auth Gagal',
      value: formatNumber(analytics.auth_failed || 0),
      caption: 'Dalam 7 hari terakhir'
    }
  ]
})

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

const featureLimits = computed(() => {
  const features = clientOverview.value?.features || {}
  return Object.entries(features)
    .filter(([, value]) => typeof value === 'number')
    .map(([key, value]) => ({
      key,
      label: humanizeKey(key),
      value: formatNumber(value)
    }))
    .sort((a, b) => a.label.localeCompare(b.label))
})

const allowedDomains = computed(() => clientOverview.value?.allowed_domains || [])
const webhookSummary = computed(() => clientOverview.value?.webhook?.summary || {
  total: 0,
  success: 0,
  failed: 0,
  retrying: 0
})
const webhookEvents = computed(() => clientOverview.value?.webhook?.events || [])
const recentUsage = computed(() => clientOverview.value?.usage?.recent || [])

const outstandingEnrollments = computed(() => {
  const analytics = clientOverview.value?.analytics
  if (!analytics) return 0
  const diff = (analytics.total_users || 0) - (analytics.enrolled_users || 0)
  return diff > 0 ? diff : 0
})

const faceAuthDisabled = computed(() => {
  const analytics = clientOverview.value?.analytics
  if (!analytics) return 0
  const diff = (analytics.total_users || 0) - (analytics.active_face_auth || 0)
  return diff > 0 ? diff : 0
})

const hasUsers = computed(() => users.value.length > 0)

function humanizeKey(key) {
  if (!key) return ''
  return key
    .replace(/_/g, ' ')
    .replace(/\b\w/g, (char) => char.toUpperCase())
}

function statusLabel(status) {
  return CLIENT_STATUS_LABELS[status] || status
}

function statusClass(status) {
  return CLIENT_STATUS_CLASS[status] || 'warning'
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

function buildProfile() {
  const profile = {}
  if (userForm.fullName) {
    profile.display_name = userForm.fullName
    profile.full_name = userForm.fullName
    profile.name = userForm.fullName
  }
  if (userForm.email) {
    profile.email = userForm.email
  }
  if (userForm.phone) {
    profile.phone = userForm.phone
    profile.phone_number = userForm.phone
  }
  return profile
}

function buildMetadata() {
  const metadata = {}
  if (userForm.role) {
    metadata.role = userForm.role
  }
  if (userForm.notes) {
    metadata.notes = userForm.notes
  }
  return metadata
}

function resetUserForm() {
  userForm.externalUserId = ''
  userForm.fullName = ''
  userForm.email = ''
  userForm.phone = ''
  userForm.role = ''
  userForm.notes = ''
  userForm.faceAuthEnabled = true
}

async function loadClientOverview() {
  if (!authStore.isAuthenticated) return
  overviewLoading.value = true
  overviewError.value = ''
  try {
    await authStore.refreshClientDetails()
    const response = await coreApi.clientInfo()
    overview.value = response.data
  } catch (error) {
    overview.value = null
    overviewError.value = parseError(error)
  } finally {
    overviewLoading.value = false
  }
}

async function loadUsers() {
  if (!authStore.isAuthenticated) return
  usersLoading.value = true
  feedback.error = ''
  try {
    const response = await clientUsersApi.list({ ordering: '-created_at' })
    users.value = response.data
  } catch (error) {
    feedback.error = parseError(error)
  } finally {
    usersLoading.value = false
  }
}

async function createUser() {
  if (!authStore.isAuthenticated) return
  if (!userForm.externalUserId.trim()) {
    feedback.error = 'External User ID wajib diisi.'
    return
  }

  userSaving.value = true
  feedback.error = ''
  feedback.success = ''

  const payload = {
    external_user_id: userForm.externalUserId.trim(),
    face_auth_enabled: userForm.faceAuthEnabled
  }

  const profile = buildProfile()
  if (Object.keys(profile).length) {
    payload.profile = profile
  }

  const metadata = buildMetadata()
  if (Object.keys(metadata).length) {
    payload.metadata = metadata
  }

  try {
    await clientUsersApi.create(payload)
    feedback.success = 'Pengguna berhasil ditambahkan.'
    resetUserForm()
    await Promise.all([loadUsers(), loadClientOverview()])
  } catch (error) {
    feedback.error = parseError(error)
  } finally {
    userSaving.value = false
  }
}

async function toggleUser(user) {
  if (!authStore.isAuthenticated) return
  togglingUserId.value = user.id
  feedback.error = ''
  feedback.success = ''
  try {
    if (user.face_auth_enabled) {
      await clientUsersApi.deactivate(user.id)
      feedback.success = `Face authentication dinonaktifkan untuk ${user.display_name || user.external_user_id}.`
    } else {
      await clientUsersApi.activate(user.id)
      feedback.success = `Face authentication diaktifkan untuk ${user.display_name || user.external_user_id}.`
    }
    await Promise.all([loadUsers(), loadClientOverview()])
  } catch (error) {
    feedback.error = parseError(error)
  } finally {
    togglingUserId.value = null
  }
}

async function loadEnrollments(user) {
  if (!authStore.isAuthenticated) return
  enrollmentLoading.value = true
  enrollments.value = {
    user,
    data: {}
  }
  try {
    const response = await clientUsersApi.enrollments(user.id)
    enrollments.value = {
      user,
      data: response.data
    }
  } catch (error) {
    enrollments.value = {
      user,
      data: {
        error: parseError(error)
      }
    }
  } finally {
    enrollmentLoading.value = false
  }
}

function clearEnrollments() {
  enrollments.value = null
}

async function initialize() {
  if (!authStore.isAuthenticated) return
  await loadClientOverview()
  await loadUsers()
}

onMounted(() => {
  if (authStore.isAuthenticated) {
    initialize()
  }
})

watch(
  () => authStore.isAuthenticated,
  (isAuthenticated) => {
    if (isAuthenticated) {
      initialize()
    } else {
      overview.value = null
      users.value = []
    }
  }
)
</script>
