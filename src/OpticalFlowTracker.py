# ==============================================================================
# OPTICAL FLOW TRACKER — Lucas-Kanade per-person velocity estimation
# ==============================================================================
import cv2
import numpy as np
from collections import deque

from Config import Config


class OpticalFlowTracker:
    """
    Theo dõi vector vận tốc của từng người bằng Pyramidal Lucas-Kanade.

    Attributes:
        flow_pts  : {person_id → np.ndarray shape (1,1,2)} — điểm tracking hiện tại.
        flow_vecs : {person_id → deque[(vx, vy)]}          — lịch sử vector vận tốc.
    """

    def __init__(
            self,
            lk_params:           dict = Config.LK_PARAMS,
            flow_history_frames: int  = Config.FLOW_HISTORY_FRAMES,
    ):
        self._lk_params           = lk_params
        self._flow_history_frames = flow_history_frames

        self.flow_pts:  dict[int, np.ndarray]     = {}
        self.flow_vecs: dict[int, deque]           = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(
            self,
            prev_gray:       np.ndarray | None,
            curr_gray:       np.ndarray,
            current_persons: dict[int, tuple[int, int]],
    ) -> dict[int, tuple[float, float]]:
        """
        Cập nhật optical flow cho tất cả người trong `current_persons`.

        Args:
            prev_gray:       Frame xám trước (None ở frame đầu tiên).
            curr_gray:       Frame xám hiện tại.
            current_persons: {person_id → (cx, cy)}.

        Returns:
            {person_id → (avg_vx, avg_vy)} cho những người tính được vector.
        """
        self._drop_stale(current_persons)

        if prev_gray is None:
            self._init_points(current_persons)
            return {}

        result: dict[int, tuple[float, float]] = {}
        for p_id, (cx, cy) in current_persons.items():
            pts_old = self.flow_pts.get(
                p_id,
                np.array([[cx, cy]], dtype=np.float32).reshape(-1, 1, 2),
            )
            pts_new, status, _ = cv2.calcOpticalFlowPyrLK(
                prev_gray, curr_gray, pts_old, None, **self._lk_params
            )
            if pts_new is None:
                continue

            good_new = pts_new[status == 1]
            good_old = pts_old[status == 1]
            if len(good_new) > 0:
                vecs   = good_new - good_old
                avg_vx = float(np.mean(vecs[:, 0]))
                avg_vy = float(np.mean(vecs[:, 1]))
                if p_id not in self.flow_vecs:
                    self.flow_vecs[p_id] = deque(maxlen=self._flow_history_frames)
                self.flow_vecs[p_id].append((avg_vx, avg_vy))
                result[p_id] = (avg_vx, avg_vy)

            # Cập nhật điểm gốc sang vị trí mới nhất của bbox
            self.flow_pts[p_id] = np.array([[cx, cy]], dtype=np.float32).reshape(-1, 1, 2)

        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _drop_stale(self, current_persons: dict) -> None:
        """Xoá tracking points của người đã rời khung hình."""
        stale = set(self.flow_pts) - set(current_persons)
        for sid in stale:
            self.flow_pts.pop(sid, None)
            self.flow_vecs.pop(sid, None)

    def _init_points(self, current_persons: dict) -> None:
        """Khởi tạo điểm tracking cho frame đầu tiên."""
        for p_id, (cx, cy) in current_persons.items():
            self.flow_pts[p_id]  = np.array([[cx, cy]], dtype=np.float32).reshape(-1, 1, 2)
            self.flow_vecs[p_id] = deque(maxlen=self._flow_history_frames)
