<template>
  <div class="camera-shell" :class="{ 'is-active': isActive }">
    <video
      ref="videoRef"
      class="camera-video"
      autoplay
      playsinline
      muted
    ></video>
    <canvas ref="canvasRef" class="camera-canvas" aria-hidden="true"></canvas>
    <div class="camera-overlay">
      <slot name="overlay"></slot>
    </div>
  </div>
</template>

<script setup>
import { computed, onBeforeUnmount, ref } from 'vue'

const props = defineProps({
  facingMode: {
    type: String,
    default: 'user'
  },
  width: {
    type: Number,
    default: 640
  },
  height: {
    type: Number,
    default: 480
  }
})

const emit = defineEmits(['started', 'stopped', 'error'])

const videoRef = ref(null)
const canvasRef = ref(null)
const mediaStream = ref(null)
const active = ref(false)

const isActive = computed(() => active.value)

async function start() {
  if (active.value) {
    return
  }

  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: {
        facingMode: props.facingMode,
        width: { ideal: props.width },
        height: { ideal: props.height }
      },
      audio: false
    })

    mediaStream.value = stream

    if (videoRef.value) {
      videoRef.value.srcObject = stream
      await videoRef.value.play()
    }

    active.value = true
    emit('started')
  } catch (error) {
    emit('error', error)
    throw error
  }
}

function stop() {
  if (mediaStream.value) {
    mediaStream.value.getTracks().forEach((track) => track.stop())
    mediaStream.value = null
  }
  if (videoRef.value) {
    videoRef.value.srcObject = null
  }
  if (active.value) {
    emit('stopped')
  }
  active.value = false
}

function ensureCanvasSize(video) {
  const canvas = canvasRef.value
  if (!canvas) return

  const width = video.videoWidth || props.width
  const height = video.videoHeight || props.height

  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width
    canvas.height = height
  }
}

async function captureFrame({ format = 'image/jpeg', quality = 0.85 } = {}) {
  if (!active.value) {
    throw new Error('Camera is not active')
  }

  const video = videoRef.value
  const canvas = canvasRef.value

  if (!video || !canvas) {
    throw new Error('Camera elements are not ready')
  }

  ensureCanvasSize(video)

  const context = canvas.getContext('2d')
  if (!context) {
    throw new Error('Unable to access 2D context')
  }

  context.drawImage(video, 0, 0, canvas.width, canvas.height)
  return canvas.toDataURL(format, quality)
}

onBeforeUnmount(() => {
  stop()
})

defineExpose({
  start,
  stop,
  captureFrame,
  isActive
})
</script>
