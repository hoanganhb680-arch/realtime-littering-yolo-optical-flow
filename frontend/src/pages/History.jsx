import { useState } from 'react'
import ViolationCard from '../components/ViolationCard'
import Lightbox      from '../components/Lightbox'
import { useViolations } from '../hooks/useViolations'
import './History.css'

const TYPE_FILTERS = ['Tất cả', 'Đột ngột', 'Đứng yên', 'Bỏ rác tại chỗ']

export default function History() {
  const { violations, total, loading, error, refetch, clearHistory } = useViolations()
  const [lightboxSrc, setLightboxSrc] = useState(null)
  const [filter, setFilter]           = useState('Tất cả')
  const [search, setSearch]           = useState('')

  const handleClearHistory = async () => {
    const isConfirmed = window.confirm(
      "⚠️ CẢNH BÁO NGUY HIỂM!\n\nBạn có chắc chắn muốn xóa toàn bộ lịch sử vi phạm không?\nHành động này sẽ xóa vĩnh viễn toàn bộ dữ liệu trong cơ sở dữ liệu SQLite, dọn dẹp các tệp tin ảnh bằng chứng lưu cục bộ và xóa sạch ảnh trên MinIO Storage!"
    )
    if (isConfirmed) {
      await clearHistory()
    }
  }

  const filtered = violations.filter(v => {
    const matchType   = filter === 'Tất cả' || v.violation_type === filter
    const matchSearch = search === '' ||
      String(v.person_id).includes(search) ||
      String(v.trash_id).includes(search)  ||
      v.timestamp?.includes(search)
    return matchType && matchSearch
  })

  return (
    <main className="history-page">
      {/* Page header */}
      <div className="history-header">
        <div>
          <h1 className="history-title">Lịch Sử Vi Phạm</h1>
          <p className="history-sub">
            Tổng cộng <strong>{total}</strong> vi phạm đã được ghi nhận
          </p>
        </div>
        <div className="history-header-actions">
          {violations.length > 0 && (
            <button className="btn btn-danger" onClick={handleClearHistory} disabled={loading}>
              🧹 Dọn dẹp lịch sử
            </button>
          )}
          <button className="btn btn-primary" onClick={refetch} disabled={loading}>
            {loading ? '⏳ Đang tải...' : '↻ Làm mới'}
          </button>
        </div>
      </div>

      {/* Filters */}
      <div className="history-controls">
        <div className="filter-tabs">
          {TYPE_FILTERS.map(f => (
            <button
              key={f}
              className={`filter-tab ${filter === f ? 'filter-tab--active' : ''}`}
              onClick={() => setFilter(f)}
            >
              {f}
            </button>
          ))}
        </div>
        <input
          className="search-input"
          type="text"
          placeholder="🔍  Tìm kiếm theo Person ID, Trash ID, thời gian..."
          value={search}
          onChange={e => setSearch(e.target.value)}
        />
      </div>

      {/* Error state */}
      {error && (
        <div className="history-error">
          ⚠️ {error}
          <button className="btn btn-ghost" onClick={refetch}>Thử lại</button>
        </div>
      )}

      {/* Loading skeletons */}
      {loading && !violations.length && (
        <div className="vcard-grid">
          {[...Array(6)].map((_, i) => (
            <div key={i} className="vcard-skeleton">
              <div className="skeleton" style={{ height: 160 }} />
              <div style={{ padding: 16, display: 'flex', flexDirection: 'column', gap: 8 }}>
                <div className="skeleton" style={{ height: 18, width: '60%' }} />
                <div className="skeleton" style={{ height: 14, width: '80%' }} />
                <div className="skeleton" style={{ height: 14, width: '45%' }} />
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Violation cards grid */}
      {!loading && filtered.length === 0 && !error && (
        <div className="history-empty">
          <span>📂</span>
          <p>{search || filter !== 'Tất cả' ? 'Không tìm thấy vi phạm phù hợp' : 'Chưa có vi phạm nào được ghi nhận'}</p>
        </div>
      )}

      {filtered.length > 0 && (
        <div className="vcard-grid">
          {filtered.map(v => (
            <ViolationCard
              key={v.id}
              violation={v}
              onImageClick={setLightboxSrc}
            />
          ))}
        </div>
      )}

      {/* Lightbox */}
      <Lightbox src={lightboxSrc} onClose={() => setLightboxSrc(null)} />
    </main>
  )
}
