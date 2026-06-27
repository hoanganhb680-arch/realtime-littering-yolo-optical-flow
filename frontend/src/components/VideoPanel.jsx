import { useRef, useState } from 'react'
import { useVideoStream } from '../hooks/useVideoStream'
import { restartStream, stopStream } from '../api/violations'
import './VideoPanel.css'

export default function VideoPanel({ onAlert, onStreamRestart, latestViolation }) {
  const canvasRef = useRef(null)
  const [connected, setConnected] = useState(false)
  const [fps, setFps] = useState(0)
  
  // Trạng thái dừng video & chạy hết video
  const [videoEnded, setVideoEnded] = useState(false)
  const [videoEndedData, setVideoEndedData] = useState(null)
  const [restarting, setRestarting] = useState(false)
  const latestEvidenceUrl = latestViolation?.evidence_url ?? latestViolation?.evidenceUrl

  // FPS counter
  const fpsRef     = useRef({ count: 0, last: 0 })
  const handleAlert = (alert) => {
    if (onAlert) onAlert(alert)
  }

  const handleVideoEnded = (data) => {
    setVideoEnded(true)
    setVideoEndedData(data)
    setFps(0)
  }

  const handleStop = async () => {
    try {
      await stopStream()
      setConnected(false)
      setFps(0)
    } catch (err) {
      console.error("Lỗi khi dừng stream:", err)
    }
  }

  const handleRestart = async () => {
    setRestarting(true)
    setVideoEnded(false)
    try {
      await restartStream()
      onStreamRestart?.()
      setConnected(true)
    } catch (err) {
      console.error("Lỗi khi chạy lại stream:", err)
    } finally {
      setRestarting(false)
    }
  }

  // WebSocket → canvas
  useVideoStream(canvasRef, {
    onAlert: handleAlert,
    onFrame: () => {
      const now = performance.now()
      if (fpsRef.current.last === 0) {
        fpsRef.current.last = now
      }
      fpsRef.current.count++
      if (now - fpsRef.current.last >= 1000) {
        setFps(fpsRef.current.count)
        fpsRef.current.count = 0
        fpsRef.current.last  = now
      }
      setConnected(true)
    },
    onVideoEnded: handleVideoEnded
  })

  // Tính toán baseURL để tải video kết quả
  const backendBaseUrl = `${window.location.protocol}//${window.location.hostname}:8000`

  return (
    <div className="video-panel">
      {/* Header bar */}
      <div className="video-header">
        <div className="video-title-group">
          <span className={`conn-dot ${connected && !videoEnded ? 'conn-dot--live' : ''}`} />
          <span className="video-title">Video Feed</span>
        </div>
        
        <div className="video-meta">
          {/* Nút điều khiển stream trực tiếp */}
          <div className="video-controls">
            {connected && !videoEnded && (
              <button onClick={handleStop} className="control-btn control-btn--stop" title="Dừng camera/video stream">
                ⏹ Dừng Stream
              </button>
            )}
            {(!connected || videoEnded) && (
              <button onClick={handleRestart} disabled={restarting} className="control-btn control-btn--play">
                {restarting ? '⏳ Đang khởi động...' : '🔄 Khởi động lại'}
              </button>
            )}
          </div>

          {connected && !videoEnded && <span className="fps-badge">{fps} FPS</span>}
          <span className={`status-chip ${connected && !videoEnded ? 'status-chip--live' : 'status-chip--wait'}`}>
            {connected && !videoEnded ? '● LIVE' : videoEnded ? '◌ ENDED' : '◌ Chờ kết nối...'}
          </span>
        </div>
      </div>

      {/* Canvas & overlays */}
      <div className="canvas-wrapper">
        {!connected && !videoEnded && latestEvidenceUrl && (
          <div className="evidence-preview">
            <img src={latestEvidenceUrl} alt="Anh bang chung moi nhat" />
            <div className="evidence-preview-badge">Anh xac nhan moi nhat</div>
          </div>
        )}

        {!connected && !videoEnded && !latestEvidenceUrl && (
          <div className="canvas-placeholder">
            <div className="placeholder-icon">📡</div>
            <p className="placeholder-title">Đang kết nối tới camera...</p>
            <p className="placeholder-sub">WebSocket ws://127.0.0.1:8000/ws/stream</p>
            <div className="connecting-bars">
              {[...Array(5)].map((_, i) => (
                <div key={i} className="bar" style={{ animationDelay: `${i * 0.15}s` }} />
              ))}
            </div>
          </div>
        )}
        <canvas
          ref={canvasRef}
          className={`video-canvas ${connected ? 'video-canvas--visible' : ''}`}
        />

        {/* Glassmorphism Video Ended Overlay */}
        {videoEnded && (
          <div className="video-ended-overlay">
            <div className="overlay-content">
              <div className="overlay-icon">🎬</div>
              <h3 className="overlay-title">XỬ LÝ VIDEO HOÀN TẤT</h3>
              <p className="overlay-msg">
                Hệ thống đã phân tích hết tệp tin video và phát hiện tổng cộng{' '}
                <strong className="text-highlight">{videoEndedData?.total_violations ?? 0}</strong> trường hợp vi phạm vứt rác bừa bãi.
              </p>
              
              <div className="overlay-actions">
                <button className="action-btn action-btn--restart" onClick={handleRestart} disabled={restarting}>
                  {restarting ? '⏳ Đang tải...' : '🔄 Phân tích lại'}
                </button>
                <a 
                  href={`${backendBaseUrl}/api/v1/stream/video`} 
                  download="processed_video.mp4" 
                  className="action-btn action-btn--download"
                  title="Tải xuống tệp tin video đã vẽ hộp nhận diện vi phạm"
                >
                  📥 Tải Video Kết Quả
                </a>
                <button className="action-btn action-btn--close" onClick={() => setVideoEnded(false)}>
                  ❌ Đóng
                </button>
              </div>
            </div>
          </div>
        )}

        {/* Corner overlay labels */}
        {connected && !videoEnded && (
          <>
            <div className="corner corner-tl">AI Detection</div>
            <div className="corner corner-br">YOLO v8 | ByteTrack</div>
          </>
        )}
      </div>
    </div>
  )
}
