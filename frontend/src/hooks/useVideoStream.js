import { useEffect, useRef, useCallback } from 'react'

const WS_URL = `${window.location.protocol === 'https:' ? 'wss' : 'ws'}://${window.location.host}/ws/stream`

/**
 * Hook kết nối WebSocket /ws/stream và render JPEG frames lên canvas.
 *
 * @param {React.RefObject<HTMLCanvasElement>} canvasRef
 * @param {{ onAlert: (alert: object) => void }} options
 * @returns {{ connected: boolean }}
 */
export function useVideoStream(canvasRef, { onAlert, onFrame, onVideoEnded } = {}) {
  const wsRef = useRef(null)
  const connectedRef = useRef(false)
  const retryRef = useRef(null)

  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return

    const ws = new WebSocket(WS_URL)
    ws.binaryType = 'arraybuffer'
    wsRef.current = ws

    ws.onopen = () => {
      connectedRef.current = true
      if (retryRef.current) {
        clearTimeout(retryRef.current)
        retryRef.current = null
      }
    }

    ws.onmessage = (event) => {
      // Binary → JPEG frame
      if (event.data instanceof ArrayBuffer) {
        const blob = new Blob([event.data], { type: 'image/jpeg' })
        const url = URL.createObjectURL(blob)
        const img = new Image()
        img.onload = () => {
          const canvas = canvasRef.current
          if (!canvas) { URL.revokeObjectURL(url); return }
          const ctx = canvas.getContext('2d')
          // Fit canvas to image ratio
          if (canvas.width !== img.width || canvas.height !== img.height) {
            canvas.width = img.width
            canvas.height = img.height
          }
          ctx.drawImage(img, 0, 0)
          URL.revokeObjectURL(url)
          if (onFrame) onFrame()
        }
        img.src = url
        return
      }

      // Text → JSON alert
      try {
        const payload = JSON.parse(event.data)
        if (payload.type === 'violation' && onAlert) {
          onAlert(payload.data)
        } else if (payload.type === 'video_ended' && onVideoEnded) {
          onVideoEnded(payload.data)
        }
      } catch (_) { /* ignore */ }
    }

    ws.onclose = () => {
      connectedRef.current = false
      // Auto-reconnect sau 3s
      retryRef.current = setTimeout(connect, 3000)
    }

    ws.onerror = () => {
      ws.close()
    }
  }, [canvasRef, onAlert, onFrame, onVideoEnded])

  useEffect(() => {
    connect()
    return () => {
      if (retryRef.current) clearTimeout(retryRef.current)
      wsRef.current?.close()
    }
  }, [connect])
}
