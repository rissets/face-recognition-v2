import { ref } from 'vue'

export function useWebRTCSession({ onResult, onError, onFinal } = {}) {
  const wsRef = ref(null)
  const sessionToken = ref(null)
  const status = ref('idle')
  const lastResult = ref(null)
  const finalMessage = ref(null)
  const sessionOutcome = ref(null)
  const frames = ref(0)
  const embeddingsSaved = ref(0)
  const similarity = ref(0)
  const liveness = ref(0)

  function connect(token) {
    sessionToken.value = token
    const proto = window.location.protocol === 'https:' ? 'wss' : 'ws'
    const host = window.location.host
    const url = `${proto}://${host}/ws/face-recognition/${token}/`
    status.value = 'connecting'
    const ws = new WebSocket(url)
    wsRef.value = ws

    ws.onopen = () => {
      status.value = 'connected'
    }
    ws.onclose = () => {
      if (status.value !== 'error' && status.value !== 'completed') {
        status.value = 'closed'
      }
    }
    ws.onerror = (e) => {
      status.value = 'error'
      onError && onError(e)
    }
    ws.onmessage = (evt) => {
      try {
        const msg = JSON.parse(evt.data)
        if (msg.type === 'frame_result') {
          lastResult.value = msg.result
          frames.value++
          if (msg.result?.quality_score) {
            similarity.value = msg.result.similarity_score || similarity.value
          }
          if (msg.result?.liveness_data) {
            const blinks = msg.result.liveness_data.blinks_detected
            if (typeof blinks === 'number') {
              liveness.value = blinks
            }
          }
          if (msg.result?.embedding_saved) embeddingsSaved.value++
          onResult && onResult(msg.result, msg)
        } else if (msg.type === 'session_final') {
          finalMessage.value = msg
          sessionOutcome.value = msg?.result?.success ? 'success' : 'failed'
          if (typeof msg.frames_processed === 'number') {
            frames.value = msg.frames_processed
          }
          status.value = sessionOutcome.value === 'success' ? 'completed' : 'finished'
          onFinal && onFinal(msg)
        }
      } catch (err) {
        console.error('WS parse error', err)
      }
    }
  }

  function sendFrame(frameData) {
    if (!wsRef.value || wsRef.value.readyState !== 1) return
    wsRef.value.send(JSON.stringify({ type: 'frame_data', frame_data: frameData, timestamp: Date.now() }))
  }

  function close() {
    if (wsRef.value) {
      wsRef.value.close()
      wsRef.value = null
    }
    finalMessage.value = null
    sessionOutcome.value = null
    status.value = 'closed'
  }

  return {
    connect,
    sendFrame,
    close,
    status,
    lastResult,
    frames,
    embeddingsSaved,
    similarity,
    liveness,
    sessionToken,
    finalMessage,
    sessionOutcome
  }
}
