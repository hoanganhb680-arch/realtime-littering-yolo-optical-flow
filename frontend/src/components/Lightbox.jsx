import { useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import './Lightbox.css'

export default function Lightbox({ src, onClose }) {
  // Đóng khi bấm Escape
  useEffect(() => {
    const handler = (e) => { if (e.key === 'Escape') onClose?.() }
    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
  }, [onClose])

  return (
    <AnimatePresence>
      {src && (
        <motion.div
          className="lightbox-backdrop"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{    opacity: 0 }}
          transition={{ duration: 0.2 }}
          onClick={onClose}
        >
          <motion.div
            className="lightbox-content"
            initial={{ scale: 0.82, opacity: 0 }}
            animate={{ scale: 1,    opacity: 1 }}
            exit={{    scale: 0.82, opacity: 0 }}
            transition={{ type: 'spring', stiffness: 300, damping: 26 }}
            onClick={(e) => e.stopPropagation()}
          >
            <button className="lightbox-close" onClick={onClose} aria-label="Đóng">✕</button>
            <img src={src} alt="Ảnh bằng chứng vi phạm" className="lightbox-img" />
            <div className="lightbox-caption">
              Ảnh bằng chứng · Click ngoài hoặc Esc để đóng
            </div>
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  )
}
