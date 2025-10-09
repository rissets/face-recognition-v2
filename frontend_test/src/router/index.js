import { createRouter, createWebHistory } from 'vue-router'
import { useAuthStore } from '../stores/auth'

const routes = [
  {
    path: '/',
    redirect: '/dashboard'
  },
  {
    path: '/webrtc/enrollment',
    component: () => import('../views/WebRTCEnrollmentView.vue'),
    meta: { requiresAuth: true }
  },
  {
    path: '/webrtc/auth',
    component: () => import('../views/WebRTCAuthorizationView.vue'),
    meta: { requiresAuth: true }
  },
  {
    path: '/enhanced-auth',
    component: () => import('../views/EnhancedAuthView.vue'),
    meta: { requiresAuth: false }
  },
  {
    path: '/login',
    component: () => import('../views/LoginView.vue')
  },
  {
    path: '/register',
    component: () => import('../views/RegistrationFlowView.vue')
  },
  {
    path: '/dashboard',
    component: () => import('../views/DashboardView.vue'),
    meta: { requiresAuth: true }
  },
  {
    path: '/account',
    component: () => import('../views/AuthTestView.vue'),
    meta: { requiresAuth: true }
  },
  {
    path: '/enrollment',
    component: () => import('../views/EnrollmentTestView.vue'),
    meta: { requiresAuth: true }
  },
  {
    path: '/face-login',
    component: () => import('../views/FaceAuthenticationView.vue'),
    meta: { requiresAuth: true }
  },
  {
    path: '/recognition-data',
    component: () => import('../views/RecognitionDataView.vue'),
    meta: { requiresAuth: true }
  },
  {
    path: '/streaming',
    component: () => import('../views/StreamingTestView.vue'),
    meta: { requiresAuth: true }
  },
  {
    path: '/analytics',
    component: () => import('../views/AnalyticsTestView.vue'),
    meta: { requiresAuth: true }
  },
  {
    path: '/system',
    component: () => import('../views/SystemStatusView.vue'),
    meta: { requiresAuth: true }
  }
]

const router = createRouter({
  history: createWebHistory(),
  routes
})

let isRestored = false

router.beforeEach((to, from, next) => {
  const auth = useAuthStore()

  if (!isRestored) {
    auth.initFromStorage()
    isRestored = true
  }

  if (to.meta.requiresAuth && !auth.isAuthenticated) {
    return next({
      path: '/login',
      query: to.fullPath ? { redirect: to.fullPath } : {}
    })
  }

  if (to.path === '/login' && auth.isAuthenticated) {
    return next('/dashboard')
  }

  next()
})

export default router
