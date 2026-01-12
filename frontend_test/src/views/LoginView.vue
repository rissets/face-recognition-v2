<template>
  <div class="layout-grid">
    <section class="section">
      <h1>Hubungkan Client Third-Party Anda</h1>
      <p>
        Masukkan <strong>API Key</strong> dan <strong>API Secret</strong> client untuk menguji seluruh alur
        third-party face recognition. Token JWT akan dibuat otomatis dan dipakai untuk setiap request berikutnya.
      </p>
      <form class="form-grid" @submit.prevent="connect">
        <div class="field">
          <label for="api-key">API Key</label>
          <input
            id="api-key"
            v-model="form.apiKey"
            placeholder="frapi_xxxxx"
            required
            autocomplete="off"
          />
        </div>
        <div class="field">
          <label for="api-secret">API Secret</label>
          <input
            id="api-secret"
            v-model="form.apiSecret"
            placeholder="Paste secret key dari dashboard client"
            required
            autocomplete="off"
          />
        </div>
        <div class="actions">
          <button type="submit" :disabled="authStore.loading">
            {{ authStore.loading ? 'Menghubungkan…' : 'Hubungkan' }}
          </button>
          <RouterLink v-if="authStore.isAuthenticated" class="btn secondary" to="/dashboard">
            Lihat Dashboard
          </RouterLink>
        </div>
        <p v-if="errorMessage" class="status-error mono">{{ errorMessage }}</p>
      </form>
    </section>

    <section class="section">
      <h2>Kenapa butuh API Key?</h2>
      <ul class="bullet-list">
        <li>Setiap client memiliki isolasi data sendiri (multi-tenant).</li>
        <li>API Secret dipakai untuk menandatangani JWT akses sesi third-party.</li>
        <li>Access token akan otomatis terpasang sebagai <code>JWT</code> dan <code>X-API-Key</code>.</li>
      </ul>
      <div class="info-card mono" v-if="authStore.client">
        <strong>Client aktif:</strong>
        <pre>{{ formattedClient }}</pre>
      </div>
    </section>
  </div>
</template>

<script setup>
import { computed, ref } from 'vue'
import { useRoute, useRouter, RouterLink } from 'vue-router'
import { useAuthStore } from '../stores/auth'

const form = ref({
  apiKey: '',
  apiSecret: ''
})

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

const formattedClient = computed(() => JSON.stringify(authStore.client, null, 2))

async function connect() {
  try {
    await authStore.connect({
      apiKey: form.value.apiKey,
      apiSecret: form.value.apiSecret
    })
    const redirect = route.query.redirect
    router.push(typeof redirect === 'string' ? redirect : '/dashboard')
  } catch (error) {
    console.error('Failed to connect client', error)
  }
}
</script>
