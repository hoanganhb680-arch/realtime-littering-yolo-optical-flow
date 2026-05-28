import { useState, useEffect, useCallback } from 'react'
import { fetchViolations, clearAllViolations } from '../api/violations'

/**
 * Hook quản lý danh sách vi phạm:
 * - Load từ REST API khi mount
 * - Merge real-time alerts từ WebSocket
 *
 * @returns {{
 *   violations: Array,
 *   total: number,
 *   loading: boolean,
 *   error: string|null,
 *   addAlert: (alert: object) => void,
 *   refetch: () => void,
 *   clearHistory: () => Promise<void>,
 * }}
 */
export function useViolations() {
  const [violations, setViolations] = useState([])
  const [total,      setTotal]      = useState(0)
  const [loading,    setLoading]    = useState(true)
  const [error,      setError]      = useState(null)

  const load = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      const res = await fetchViolations(100, 0)
      setViolations(res.data ?? [])
      setTotal(res.total ?? 0)
    } catch (err) {
      setError('Không thể kết nối tới backend.')
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => { load() }, [load])

  /**
   * Được gọi khi WebSocket push alert mới.
   * Gộp các hành vi vi phạm có cùng personId, chỉ giữ lại hành vi có điểm đánh giá (%) cao nhất.
   */
  const addAlert = useCallback((alertData) => {
    setViolations(prev => {
      // Tìm xem personId này đã có vi phạm nào trong danh sách hiện tại chưa
      const existingIdx = prev.findIndex(v => v.person_id === alertData.personId)
      
      if (existingIdx > -1) {
        const existingItem = prev[existingIdx]
        
        // Nếu vi phạm mới có điểm số cao hơn, cập nhật trực tiếp tại chỗ
        if (alertData.score > existingItem.score) {
          const updated = [...prev]
          updated[existingIdx] = {
            ...existingItem,
            trash_id:       alertData.trashId,
            violation_type: alertData.violationType,
            score:          alertData.score,
            timestamp:      alertData.timestamp,
            evidence_url:   null, // Sẽ được tải lại qua DB sau 2s
            isNew:          true,
          }
          return updated
        }
        return prev // Giữ nguyên nếu điểm thấp hơn hoặc bằng
      } else {
        // Person ID mới hoàn toàn -> thêm mới vào đầu danh sách
        const newItem = {
          id:             Date.now(),
          person_id:      alertData.personId,
          trash_id:       alertData.trashId,
          violation_type: alertData.violationType,
          score:          alertData.score,
          timestamp:      alertData.timestamp,
          evidence_url:   null,
          isNew:          true,
        }
        setTotal(t => t + 1) // Chỉ tăng tổng số khi phát hiện người vi phạm mới
        return [newItem, ...prev]
      }
    })

    // Sau 2s, refresh từ DB để lấy URL ảnh thật đã được cập nhật đè ở MinIO
    setTimeout(load, 2000)
  }, [load])

  /**
   * Gọi API dọn dẹp sạch toàn bộ dữ liệu lịch sử vi phạm
   */
  const clearHistory = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      await clearAllViolations()
      setViolations([])
      setTotal(0)
    } catch (err) {
      setError('Không thể dọn dẹp lịch sử vi phạm.')
    } finally {
      setLoading(false)
    }
  }, [])

  return { violations, total, loading, error, addAlert, refetch: load, clearHistory }
}
