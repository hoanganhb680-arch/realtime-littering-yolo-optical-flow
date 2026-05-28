import { BrowserRouter, Routes, Route } from 'react-router-dom'
import Navbar    from './components/Navbar'
import Dashboard from './pages/Dashboard'
import History   from './pages/History'
import { useViolations } from './hooks/useViolations'
import './App.css'

export default function App() {
  const { total } = useViolations()

  return (
    <BrowserRouter>
      <Navbar violationCount={total} />
      <div className="page-content">
        <Routes>
          <Route path="/"        element={<Dashboard />} />
          <Route path="/history" element={<History />} />
        </Routes>
      </div>
    </BrowserRouter>
  )
}
