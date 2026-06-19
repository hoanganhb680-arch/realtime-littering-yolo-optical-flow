import './ViolationCard.css'

const TYPE_COLOR = {
  'Đột ngột': 'danger',
  'Đứng yên': 'warning',
  'Bỏ rác tại chỗ': 'accent',
}

export default function ViolationCard({ violation, onImageClick }) {
  const {
    id, person_id, trash_id,
    violation_type, score, timestamp, evidence_url, evidence_video_url, created_at,
  } = violation

  const color = TYPE_COLOR[violation_type] ?? 'accent'
  const hasVideo = Boolean(evidence_video_url)

  return (
    <div className={`vcard vcard--${color}`}>
      <div className="vcard-media-stack">
        {hasVideo && (
          <div className="vcard-video-wrap">
            <video
              src={evidence_video_url}
              className="vcard-video"
              controls
              preload="metadata"
            />
          </div>
        )}

        <div
          className={`vcard-img-wrap ${hasVideo ? 'vcard-img-wrap--thumb' : ''}`}
          onClick={() => evidence_url && onImageClick?.(evidence_url)}
          title={evidence_url ? 'Nhấp để phóng to ảnh' : 'Chưa có ảnh bằng chứng'}
        >
          {evidence_url ? (
            <img src={evidence_url} alt={`Vi phạm #${id}`} className="vcard-img" loading="lazy" />
          ) : (
            <div className="vcard-img-placeholder">Chưa có ảnh</div>
          )}
          {evidence_url && <div className="vcard-img-overlay">Phóng to ảnh</div>}
        </div>
      </div>

      <div className="vcard-body">
        <div className="vcard-header">
          <span className={`badge badge-${color}`}>{violation_type}</span>
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
