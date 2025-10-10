<template>
  <div class="layout-grid">
    <section class="section">
      <h2>Informasi Client</h2>
      <div v-if="client" class="response-card mono">
        <pre>{{ formattedClient }}</pre>
      </div>
      <p v-else class="status-warning">Client belum dimuat. Klik "Refresh" untuk mencoba lagi.</p>
      <div class="actions">
        <button type="button" class="secondary" @click="refreshClient">Refresh</button>
        <button type="button" class="secondary" @click="loadUsers">Muat Daftar Pengguna</button>
      </div>
    </section>

    <section class="section">
      <h2>Tambah Pengguna Client</h2>
      <form class="form-grid" @submit.prevent="createUser">
        <div class="field">
          <label for="external-id">External User ID</label>
          <input id="external-id" v-model="userForm.externalUserId" required placeholder="john.doe" />
        </div>
        <div class="field">
          <label for="profile-json">Profil (JSON)</label>
          <textarea
            id="profile-json"
            v-model="userForm.profile"
            rows="4"
            placeholder='{"email":"john@company.com","display_name":"John Doe"}'
          ></textarea>
        </div>
        <div class="actions">
          <button type="submit" :disabled="userLoading">Simpan Pengguna</button>
          <span v-if="userError" class="status-error mono">{{ userError }}</span>
        </div>
      </form>
    </section>

    <section class="section">
      <h2>Daftar Pengguna Terdaftar</h2>
      <div class="actions">
        <button type="button" class="secondary" @click="loadUsers" :disabled="userLoading">
          {{ userLoading ? 'Memuat…' : 'Muat Ulang' }}
        </button>
      </div>
      <div v-if="users.length === 0" class="status-warning">Belum ada pengguna di client ini.</div>
      <div v-else class="table-responsive">
        <table>
          <thead>
            <tr>
              <th>ID Eksternal</th>
              <th>Enrolled</th>
              <th>Face Auth</th>
              <th>Terakhir Recognize</th>
              <th>Aksi</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="user in users" :key="user.id">
              <td>
                <strong>{{ user.external_user_id }}</strong>
                <div class="mono small">{{ user.display_name }}</div>
              </td>
              <td>{{ user.is_enrolled ? 'Ya' : 'Belum' }}</td>
              <td>{{ user.face_auth_enabled ? 'Aktif' : 'Nonaktif' }}</td>
              <td>{{ formatDate(user.last_recognition_at) }}</td>
              <td>
                <button class="secondary" type="button" @click="toggleUser(user)">
                  {{ user.face_auth_enabled ? 'Nonaktifkan' : 'Aktifkan' }}
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
      <h2>Enrollment {{ enrollments.user?.external_user_id }}</h2>
      <div class="response-card mono">
        <pre>{{ enrollmentsText }}</pre>
      </div>
    </section>
  </div>
</template>

<script setup>
import { computed, reactive, ref } from 'vue'
import { useAuthStore } from '../stores/auth'
import { clientApi, clientUsersApi } from '../services/api'

const authStore = useAuthStore()

const userForm = reactive({
  externalUserId: '',
  profile: ''
})

const userLoading = ref(false)
const userError = ref('')
const users = ref([])
const enrollments = ref(null)

const client = computed(() => authStore.client)
const formattedClient = computed(() => JSON.stringify(client.value, null, 2))
const enrollmentsText = computed(() => JSON.stringify(enrollments.value?.data || {}, null, 2))

function formatDate(value) {
  if (!value) return '-'
  return new Date(value).toLocaleString()
}

async function refreshClient() {
  await authStore.refreshClientDetails()
}

async function loadUsers() {
  userLoading.value = true
  userError.value = ''
  try {
    const response = await clientUsersApi.list()
    users.value = response.data
  } catch (error) {
    userError.value = JSON.stringify(error.response?.data || error.message, null, 2)
  } finally {
    userLoading.value = false
  }
}

async function createUser() {
  userLoading.value = true
  userError.value = ''
  try {
    let profilePayload = {}
    if (userForm.profile.trim()) {
      profilePayload = JSON.parse(userForm.profile)
    }
    await clientUsersApi.create({
      external_user_id: userForm.externalUserId,
      profile: profilePayload
    })
    userForm.externalUserId = ''
    userForm.profile = ''
    await loadUsers()
  } catch (error) {
    userError.value = JSON.stringify(error.response?.data || error.message, null, 2)
  } finally {
    userLoading.value = false
  }
}

async function toggleUser(user) {
  userLoading.value = true
  userError.value = ''
  try {
    if (user.face_auth_enabled) {
      await clientUsersApi.deactivate(user.id)
    } else {
      await clientUsersApi.activate(user.id)
    }
    await loadUsers()
  } catch (error) {
    userError.value = JSON.stringify(error.response?.data || error.message, null, 2)
  } finally {
    userLoading.value = false
  }
}

async function loadEnrollments(user) {
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
        error: error.response?.data || error.message
      }
    }
  }
}

if (authStore.isAuthenticated) {
  loadUsers()
}
</script>
