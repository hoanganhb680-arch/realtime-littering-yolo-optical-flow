import { useState, useEffect, useCallback } from 'react'
import { fetchViolations, clearAllViolations } from '../api/violations'

export function useViolations() {
  const [violations, setViolations] = useState([])
  const [total, setTotal] = useState(0)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  const load = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      const res = await fetchViolations(100, 0)
      setViolations(res.data ?? [])
      setTotal(res.total ?? 0)
    } catch {
      setError('Không thể kết nối tới backend.')
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    const timer = setTimeout(load, 0)
    return () => clearTimeout(timer)
  }, [load])

  const addAlert = useCallback((alertData) => {
    setViolations(prev => {
      const existingIdx = prev.findIndex(v => v.person_id === alertData.personId)

      if (existingIdx > -1) {
        const existingItem = prev[existingIdx]
        if (alertData.score > existingItem.score) {
          const updated = [...prev]
          updated[existingIdx] = {
            ...existingItem,
            trash_id: alertData.trashId,
            violation_type: alertData.violationType,
            score: alertData.score,
            timestamp: alertData.timestamp,
            evidence_url: null,
            evidence_video_url: null,
            isNew: true,
          }
          return updated
        }
        return prev
      }

      const newItem = {
        id: Date.now(),
        person_id: alertData.personId,
        trash_id: alertData.trashId,
        violation_type: alertData.violationType,
        score: alertData.score,
        timestamp: alertData.timestamp,
        evidence_url: null,
        evidence_video_url: null,
        isNew: true,
      }
      setTotal(t => t + 1)
      return [newItem, ...prev]
    })

    setTimeout(load, 2000)
  }, [load])

  const clearHistory = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      await clearAllViolations()
      setViolations([])
      setTotal(0)
    } catch {
      setError('Không thể dọn dẹp lịch sử vi phạm.')
    } finally {
      setLoading(false)
    }
  }, [])

  return { violations, total, loading, error, addAlert, refetch: load, clearHistory }
}
