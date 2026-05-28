import './StatsBar.css'

const STATS = [
  { label: 'Tổng Vi Phạm',  key: 'total',   icon: '🚨', color: 'danger'  },
  { label: 'Đột Ngột',      key: 'sudden',   icon: '⚡', color: 'danger'  },
  { label: 'Đứng Yên',      key: 'standing', icon: '🕒', color: 'warning' },
  { label: 'Bỏ Rác Tại Chỗ', key: 'onspot', icon: '🗑️', color: 'accent'  },
]

export default function StatsBar({ violations = [] }) {
  const counts = violations.reduce(
    (acc, v) => {
      acc.total++
      if (v.violation_type === 'Đột ngột')       acc.sudden++
      else if (v.violation_type === 'Đứng yên')   acc.standing++
      else if (v.violation_type === 'Bỏ rác tại chỗ') acc.onspot++
      return acc
    },
    { total: 0, sudden: 0, standing: 0, onspot: 0 }
  )

  return (
    <div className="stats-bar">
      {STATS.map(({ label, key, icon, color }) => (
        <div key={key} className={`stat-card stat-card--${color}`}>
          <div className="stat-icon">{icon}</div>
          <div className="stat-body">
            <div className="stat-value">{counts[key]}</div>
            <div className="stat-label">{label}</div>
          </div>
        </div>
      ))}
    </div>
  )
}
