import { useEffect, useRef } from 'react'

const BACKEND_HOST = `${window.location.hostname}:8000`
const WS_URL = `${window.location.protocol === 'https:' ? 'wss' : 'ws'}://${BACKEND_HOST}/ws/stream`

export function useVideoStream(canvasRef, { onAlert, onFrame, onVideoEnded } = {}) {
  const callbacksRef = useRef({ onAlert, onFrame, onVideoEnded })

  useEffect(() => {
    callbacksRef.current = { onAlert, onFrame, onVideoEnded }
  }, [onAlert, onFrame, onVideoEnded])

  useEffect(() => {
    let ws = null
    let retryTimer = null
    let rafId = null
    let latestFrame = null
    let drawing = false
    let active = true

    const drawBlobFallback = (blob) => {
      return new Promise((resolve, reject) => {
        const url = URL.createObjectURL(blob)
        const img = new Image()
        img.onload = () => {
          const canvas = canvasRef.current
          if (canvas) {
            const ctx = canvas.getContext('2d')
            if (canvas.width !== img.width || canvas.height !== img.height) {
              canvas.width = img.width
              canvas.height = img.height
            }
            ctx.drawImage(img, 0, 0)
          }
          URL.revokeObjectURL(url)
          resolve()
        }
        img.onerror = () => {
          URL.revokeObjectURL(url)
          reject(new Error('Cannot decode JPEG frame'))
        }
        img.src = url
      })
    }

    const renderLatestFrame = async () => {
      if (drawing || !active) return

      const frame = latestFrame
      latestFrame = null
      if (!frame) return

      drawing = true
      try {
        const blob = new Blob([frame], { type: 'image/jpeg' })
        const canvas = canvasRef.current

        if (canvas && 'createImageBitmap' in window) {
          const bitmap = await createImageBitmap(blob)
          const ctx = canvas.getContext('2d')
          if (canvas.width !== bitmap.width || canvas.height !== bitmap.height) {
            canvas.width = bitmap.width
            canvas.height = bitmap.height
          }
          ctx.drawImage(bitmap, 0, 0)
          bitmap.close?.()
        } else {
          await drawBlobFallback(blob)
        }

        callbacksRef.current.onFrame?.()
      } catch {
        // Drop malformed/stale frames instead of blocking the live stream.
      } finally {
        drawing = false
        if (active && latestFrame && rafId === null) {
          rafId = requestAnimationFrame(() => {
            rafId = null
            renderLatestFrame()
          })
        }
      }
    }

    const scheduleRender = () => {
      if (drawing || rafId !== null) return
      rafId = requestAnimationFrame(() => {
        rafId = null
        renderLatestFrame()
      })
    }

    const connect = () => {
      if (!active) return
      if (ws?.readyState === WebSocket.OPEN || ws?.readyState === WebSocket.CONNECTING) return

      ws = new WebSocket(WS_URL)
      ws.binaryType = 'arraybuffer'

      ws.onopen = () => {
        if (retryTimer) {
          clearTimeout(retryTimer)
          retryTimer = null
        }
      }

      ws.onmessage = (event) => {
        if (event.data instanceof ArrayBuffer) {
          latestFrame = event.data
          scheduleRender()
          return
        }

        try {
          const payload = JSON.parse(event.data)
          if (payload.type === 'violation') {
            callbacksRef.current.onAlert?.(payload.data)
          } else if (payload.type === 'video_ended') {
            callbacksRef.current.onVideoEnded?.(payload.data)
          }
        } catch {
          // Ignore non-JSON text messages.
        }
      }

      ws.onclose = () => {
        ws = null
        if (!active) return
        retryTimer = setTimeout(connect, 3000)
      }

      ws.onerror = () => {
        ws?.close()
      }
    }

    connect()

    return () => {
      active = false
      latestFrame = null
      if (retryTimer) clearTimeout(retryTimer)
      if (rafId !== null) cancelAnimationFrame(rafId)
      ws?.close()
      ws = null
    }
  }, [canvasRef])
}
