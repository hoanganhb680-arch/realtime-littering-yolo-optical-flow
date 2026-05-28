import { useState, useCallback } from 'react'
import VideoPanel  from '../components/VideoPanel'
import AlertFeed   from '../components/AlertFeed'
import StatsBar    from '../components/StatsBar'
import { useViolations } from '../hooks/useViolations'
import './Dashboard.css'

export default function Dashboard() {
  const { violations, total, loading, addAlert } = useViolations()

  // Danh sách cảnh báo real-time thời gian thực
  const [liveAlerts, setLiveAlerts] = useState([])

  const handleAlert = useCallback((alertData) => {
    // Thêm vào cơ sở dữ liệu và danh sách lịch sử
    addAlert(alertData)
    
    // Cập nhật danh sách real-time hiển thị ở sidebar
    setLiveAlerts(prev => {
      // Tìm xem đối tượng personId này đã xuất hiện trong danh sách real-time chưa
      const idx = prev.findIndex(a => a.personId === alertData.personId)
      
      if (idx > -1) {
        // Nếu hành vi mới có điểm số (%) cao hơn, cập nhật trực tiếp tại chỗ
        if (alertData.score > prev[idx].score) {
          const updated = [...prev]
          updated[idx] = {
            ...alertData,
            id: prev[idx].id, // Giữ nguyên ID gốc để AnimatePresence không re-render
            isNew: true
          }
          return updated
        }
        return prev
      } else {
        // Nếu là đối tượng mới vi phạm, thêm vào đầu và giới hạn tối đa 50 phần tử
        const alertId = alertData.id ?? alertData.trashId ?? Date.now()
        return [
          { ...alertData, id: alertId, isNew: true },
          ...prev.slice(0, 49),
        ]
      }
    })
  }, [addAlert])

  // Xóa sạch danh sách cảnh báo real-time (Dọn dẹp)
  const handleClearAlerts = useCallback(() => {
    setLiveAlerts([])
  }, [])

  // Xóa một cảnh báo đơn lẻ
  const handleDismissAlert = useCallback((id) => {
    setLiveAlerts(prev => prev.filter(alert => alert.id !== id))
  }, [])

  return (
    <main className="dashboard">
      {/* Stats row */}
      <StatsBar violations={violations} />

      {/* Main content: video + sidebar */}
      <div className="dashboard-grid">
        {/* Left: Video stream */}
        <div className="dashboard-video">
          <VideoPanel onAlert={handleAlert} />

          {/* Info bar */}
          <div className="video-info-bar">
            <span className="info-chip">
              📊 Tổng vi phạm đã ghi nhận: <strong>{total}</strong>
            </span>
            {loading && <span className="info-chip info-chip--muted">⏳ Đang tải...</span>}
          </div>
        </div>

        {/* Right: Real-time alert feed */}
        <div className="dashboard-sidebar">
          <AlertFeed 
            alerts={liveAlerts} 
            onClear={handleClearAlerts} 
            onDismiss={handleDismissAlert} 
          />
        </div>
      </div>
    </main>
  )
}
