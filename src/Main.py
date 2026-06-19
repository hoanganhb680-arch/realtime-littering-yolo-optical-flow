# ==============================================================================
# ENTRY POINT
# ==============================================================================
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from TrashViolationDetector import TrashViolationDetector
from Config import Config

if __name__ == "__main__":
    detector = TrashViolationDetector(cfg=Config())
    detector.run()
