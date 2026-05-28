import axios from 'axios'

const api = axios.create({
  baseURL: '/api/v1',
  timeout: 8000,
})

/**
 * Lấy danh sách vi phạm từ SQLite qua REST API
 * @param {number} limit
 * @param {number} offset
 * @returns {Promise<{total: number, data: Array}>}
 */
export async function fetchViolations(limit = 50, offset = 0) {
  const res = await api.get('/violations', { params: { limit, offset } })
  return res.data
}

/**
 * Gửi lệnh khởi động lại stream video
 */
export async function restartStream() {
  const res = await api.post('/stream/restart')
  return res.data
}

/**
 * Gửi lệnh dừng stream video
 */
export async function stopStream() {
  const res = await api.post('/stream/stop')
  return res.data
}

/**
 * Gửi lệnh dọn dẹp sạch toàn bộ dữ liệu lịch sử vi phạm
 */
export async function clearAllViolations() {
  const res = await api.post('/violations/clear')
  return res.data
}

export default api
