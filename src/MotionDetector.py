# ==============================================================================
# MOTION DETECTOR — MOG2 background subtraction → danh sách tâm chuyển động
# ==============================================================================
import cv2
import numpy as np

from Config import Config


class MotionDetector:
    """
    Phát hiện vùng chuyển động trong frame bằng MOG2.
    Trả về danh sách (cx, cy, area) cho mỗi vùng vượt ngưỡng diện tích.
    """

    # Khởi tạo bộ trừ nền MOG2 và kernel morphology làm sạch mask.
    def __init__(
            self,
            history:   int = Config.MOG2_HISTORY,
            threshold: int = Config.MOG2_THRESHOLD,
            min_area:  int = Config.MOG2_MIN_AREA,
    ):
        self.min_area   = min_area
        self._subtractor = cv2.createBackgroundSubtractorMOG2(
            history=history,
            varThreshold=threshold,
            detectShadows=False,
        )
        self._kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    # Trả về tâm các vùng foreground/chuyển động đủ lớn trong frame.
    def get_alerts(self, frame: np.ndarray) -> list[tuple[int, int, float]]:
        """
        Áp MOG2 + morphology lên `frame`.
        Trả về list[(cx, cy, area)] — tâm & diện tích mỗi blob chuyển động.
        """
        fg = self._subtractor.apply(frame)
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN,  self._kernel)
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, self._kernel)

        contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        alerts = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.min_area:
                continue
            M = cv2.moments(cnt)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                alerts.append((cx, cy, area))
        return alerts
