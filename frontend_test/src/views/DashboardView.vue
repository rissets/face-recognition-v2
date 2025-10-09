<template>
  <div class="layout-grid">
    <section class="hero">
      <div class="hero-card">
        <h1>Face Recognition Control Center</h1>
        <p>
          Jalankan seluruh alur face recognition layaknya aplikasi produksi: mulai dari login,
          enrollment wajah, autentikasi realtime, hingga monitoring streaming dan analytics.
        </p>
        <div class="actions">
          <button class="btn" type="button" @click="goTo('/enrollment')">Mulai Enrollment</button>
          <button class="btn secondary" type="button" @click="goTo('/face-login')">
            Login via Face Recognition
          </button>
        </div>
      </div>
      <div class="section">
        <h2>Langkah Cepat</h2>
        <ul class="step-list">
          <li>
            <div class="step-badge">1</div>
            <div>
              <strong>Masuk menggunakan email & password</strong>
              <p>Sesi JWT akan dipakai untuk akses seluruh endpoint yang butuh autentikasi.</p>
            </div>
          </li>
          <li>
            <div class="step-badge">2</div>
            <div>
              <strong>Lakukan enrollment wajah melalui kamera</strong>
              <p>Siapkan pencahayaan baik, jalankan sesi streaming, dan kumpulkan sampel otomatis.</p>
            </div>
          </li>
          <li>
            <div class="step-badge">3</div>
            <div>
              <strong>Validasi login via face recognition</strong>
              <p>Tes mode identification atau verification dan pantau skor similarity secara realtime.</p>
            </div>
          </li>
          <li>
            <div class="step-badge">4</div>
            <div>
              <strong>Analisa data & monitoring</strong>
              <p>Lihat histori, analytics, dan traffic streaming untuk memastikan semuanya stabil.</p>
            </div>
          </li>
        </ul>
      </div>
    </section>

    <section class="section">
      <h2>Status Akun</h2>
      <div class="session-summary">
        <div class="session-tile">
          <span>Email</span>
          <strong>{{ profile?.email || '-' }}</strong>
        </div>
        <div class="session-tile">
          <span>Enrollment</span>
          <strong>{{ profile?.face_enrolled ? 'Sudah' : 'Belum' }}</strong>
        </div>
        <div class="session-tile">
          <span>Progress Enrollment</span>
          <strong>{{ Math.round(profile?.enrollment_progress || 0) }}%</strong>
        </div>
        <div class="session-tile">
          <span>Face Auth Enable</span>
          <strong>{{ profile?.face_auth_enabled ? 'Aktif' : 'Nonaktif' }}</strong>
        </div>
      </div>
      <div class="pill-group">
        <span class="status-pill" :class="profile?.two_factor_enabled ? 'success' : 'warning'">
          2FA {{ profile?.two_factor_enabled ? 'Aktif' : 'Nonaktif' }}
        </span>
        <span class="status-pill" :class="profile?.is_verified ? 'success' : 'warning'">
          Email {{ profile?.is_verified ? 'Terverifikasi' : 'Belum Verifikasi' }}
        </span>
        <span class="status-pill" :class="profile?.face_enrolled ? 'success' : 'danger'">
          Face Enrollment {{ profile?.face_enrolled ? 'Siap' : 'Perlu Action' }}
        </span>
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
import { computed } from 'vue'
import { useRouter } from 'vue-router'
import { useAuthStore } from '../stores/auth'

const router = useRouter()
const authStore = useAuthStore()

const profile = computed(() => authStore.profile)

function goTo(path) {
  router.push(path)
}
</script>
