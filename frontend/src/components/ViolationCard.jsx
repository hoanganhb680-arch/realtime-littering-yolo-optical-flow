import './ViolationCard.css'

const TYPE_COLOR = {
  'Đột ngột':       'danger',
  'Đứng yên':       'warning',
  'Bỏ rác tại chỗ': 'accent',
}
const TYPE_ICON = {
  'Đột ngột':       '⚡',
  'Đứng yên':       '🕒',
  'Bỏ rác tại chỗ': '🗑️',
}

export default function ViolationCard({ violation, onImageClick }) {
  const {
    id, person_id, trash_id,
    violation_type, score, timestamp, evidence_url, created_at,
  } = violation

  const color = TYPE_COLOR[violation_type] ?? 'accent'
  const icon  = TYPE_ICON[violation_type]  ?? '🚨'

  return (
    <div className={`vcard vcard--${color}`}>
      {/* Evidence image */}
      <div
        className="vcard-img-wrap"
        onClick={() => evidence_url && onImageClick?.(evidence_url)}
        title={evidence_url ? 'Nhấp để phóng to' : 'Chưa có ảnh bằng chứng'}
      >
        {evidence_url ? (
          <img src={evidence_url} alt={`Vi phạm #${id}`} className="vcard-img" loading="lazy" />
        ) : (
          <div className="vcard-img-placeholder">📷</div>
        )}
        {evidence_url && <div className="vcard-img-overlay">🔍 Phóng to</div>}
      </div>

      {/* Body */}
      <div className="vcard-body">
        <div className="vcard-header">
          <span className={`badge badge-${color}`}>{icon} {violation_type}</span>
          <span className="vcard-id">#{id}</span>
        </div>

        <div className="vcard-meta">
          <div className="vcard-meta-item">
            <span className="meta-label">Người vi phạm</span>
            <span className="meta-value">Person #{person_id}</span>
          </div>
          <div className="vcard-meta-item">
            <span className="meta-label">Rác ID</span>
            <span className="meta-value">#{trash_id}</span>
          </div>
          <div className="vcard-meta-item">
            <span className="meta-label">Độ tin cậy</span>
            <span className={`meta-value meta-score meta-score--${color}`}>
              {((score ?? 0) * 100).toFixed(1)}%
            </span>
          </div>
          <div className="vcard-meta-item">
            <span className="meta-label">Thời gian</span>
            <span className="meta-value meta-mono">{timestamp}</span>
          </div>
        </div>

        {created_at && (
          <div className="vcard-footer">
            Ghi nhận lúc {new Date(created_at).toLocaleString('vi-VN')}
          </div>
        )}
      </div>
    </div>
  )
}
