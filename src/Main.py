# ==============================================================================
# ENTRY POINT
# ==============================================================================
from TrashViolationDetector import TrashViolationDetector
from Config import Config

if __name__ == "__main__":
    detector = TrashViolationDetector(cfg=Config())
    detector.run()
