<template>
  <nav>
    <div class="nav-inner">
      <div class="nav-brand">
        <RouterLink to="/" class="brand-link">FaceRec Tester</RouterLink>
      </div>
      <div class="nav-links">
        <template v-if="!isAuthenticated">
          <RouterLink to="/login">Connect</RouterLink>
        </template>
        <template v-else>
          <RouterLink to="/dashboard">Overview</RouterLink>
          <RouterLink to="/enrollment">Enrollment</RouterLink>
          <RouterLink to="/face-login">Face Login</RouterLink>
          <RouterLink to="/recognition-data">Data</RouterLink>
          <RouterLink to="/analytics">Analytics</RouterLink>
          <RouterLink to="/streaming">Streaming</RouterLink>
          <RouterLink to="/webrtc/enrollment">WebRTC Enroll</RouterLink>
          <RouterLink to="/webrtc/auth">WebRTC Auth</RouterLink>
          <RouterLink to="/enhanced-auth" class="enhanced-link">Enhanced Demo</RouterLink>
          <RouterLink to="/account">Account</RouterLink>
          <RouterLink to="/system">System</RouterLink>
        </template>
      </div>

      <div class="nav-auth">
        <span v-if="isAuthenticated" class="badge">
          <span>Client</span>
          <strong>{{ authClientLabel }}</strong>
        </span>
        <button v-if="isAuthenticated" class="secondary" @click="logout" type="button">
          Logout
        </button>
      </div>
    </div>
  </nav>
</template>

<script setup>
import { computed } from 'vue'
import { RouterLink, useRouter } from 'vue-router'
import { useAuthStore } from '../stores/auth'

const router = useRouter()
const authStore = useAuthStore()

const isAuthenticated = computed(() => authStore.isAuthenticated)
const authClientLabel = computed(() => authStore.client?.name || 'Unknown Client')

function logout() {
  authStore.disconnect()
  router.push('/login')
}
</script>
