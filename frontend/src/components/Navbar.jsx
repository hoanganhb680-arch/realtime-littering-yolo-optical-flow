import { NavLink } from 'react-router-dom'
import './Navbar.css'

export default function Navbar({ violationCount = 0 }) {
  return (
    <nav className="navbar">
      <div className="navbar-brand">
        <span className="navbar-icon">🎯</span>
        <span className="navbar-title">TrashGuard <span className="navbar-sub">AI Monitor</span></span>
      </div>

      <div className="navbar-links">
        <NavLink to="/"        className={({ isActive }) => isActive ? 'nav-link active' : 'nav-link'} end>
          Dashboard
        </NavLink>
        <NavLink to="/history" className={({ isActive }) => isActive ? 'nav-link active' : 'nav-link'}>
          Lịch sử
          {violationCount > 0 && (
            <span className="nav-badge">{violationCount > 99 ? '99+' : violationCount}</span>
          )}
        </NavLink>
      </div>

      <div className="navbar-status">
        <span className="live-dot" />
        <span className="status-text">LIVE</span>
      </div>
    </nav>
  )
}
