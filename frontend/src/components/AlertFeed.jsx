import { motion, AnimatePresence } from 'framer-motion'
import './AlertFeed.css'

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

export default function AlertFeed({ alerts = [], onClear, onDismiss }) {
  return (
    <div className="alert-feed">
      <div className="alert-header">
        <span className="alert-title">Vi Phạm Real-time</span>
        <div className="alert-header-right">
          <span className="alert-count">{alerts.length}</span>
          {alerts.length > 0 && onClear && (
            <button onClick={onClear} className="clear-btn" title="Dọn dẹp danh sách cảnh báo">
              🧹 Dọn dẹp
            </button>
          )}
        </div>
      </div>

      <div className="alert-list">
        <AnimatePresence initial={false}>
          {alerts.length === 0 ? (
            <div className="alert-empty">
              <span>✅</span>
              <p>Chưa có vi phạm nào</p>
            </div>
          ) : (
            alerts.map((alert, idx) => {
              const color = TYPE_COLOR[alert.violationType] ?? 'accent'
              const icon  = TYPE_ICON[alert.violationType]  ?? '🚨'
              const itemId = alert.id ?? idx
              return (
                <motion.div
                  key={itemId}
                  className={`alert-item alert-item--${color}`}
                  initial={{ opacity: 0, x: 40, scale: 0.92 }}
                  animate={{ opacity: 1, x: 0,  scale: 1    }}
                  exit={   { opacity: 0, x: 40, scale: 0.88 }}
                  transition={{ type: 'spring', stiffness: 280, damping: 22 }}
                  layout
                >
                  <div className="alert-icon">{icon}</div>
                  <div className="alert-body">
                    <div className="alert-type">{alert.violationType}</div>
                    <div className="alert-meta">
                      <span>👤 P{alert.personId ?? alert.person_id}</span>
                      <span>🗑️ T{alert.trashId  ?? alert.trash_id}</span>
                      <span className="alert-score">
                        {((alert.score ?? 0) * 100).toFixed(0)}%
                      </span>
                    </div>
                    <div className="alert-time">{alert.timestamp}</div>
                  </div>
                  {onDismiss && (
                    <button 
                      onClick={(e) => { 
                        e.stopPropagation(); 
                        onDismiss(itemId); 
                      }} 
                      className="dismiss-btn" 
                      title="Xóa cảnh báo này"
                    >
                      ×
                    </button>
                  )}
                  {(alert.isNew) && <div className="new-pulse" />}
                </motion.div>
              )
            })
          )}
        </AnimatePresence>
      </div>
    </div>
  )
}
