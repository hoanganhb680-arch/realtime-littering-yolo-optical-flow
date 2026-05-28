# ==============================================================================
# OWNERSHIP SCORER — Tính điểm "ai là chủ của rác này?"
# ==============================================================================
import math
import numpy as np
from collections import deque

from Config import Config


class OwnershipScorer:
    """
    Tổng hợp 4 tín hiệu để cho điểm xác suất một người là chủ nhân của vật rác:
        1. proximity_score  — khoảng cách gần nhất trong lịch sử quỹ đạo
        2. direction_score  — hướng di chuyển rời xa vị trí rác
        3. flow_score       — vector optical-flow chỉ ra hướng rời xa
        4. recency_score    — thời điểm người gần rác gần đây đến đâu
    """

    def __init__(
            self,
            trajectory_radius:  float = Config.TRAJECTORY_RADIUS,
            spawn_radius:        float = Config.SPAWN_RADIUS,
            history_frames:      int   = Config.HISTORY_FRAMES,
            min_score:           float = Config.MIN_SCORE,
            ambiguous_margin:    float = Config.AMBIGUOUS_MARGIN,
            mog2_boost_radius:   float = Config.MOG2_BOOST_RADIUS,
    ):
        self.trajectory_radius = trajectory_radius
        self.spawn_radius      = spawn_radius
        self.history_frames    = history_frames
        self.min_score         = min_score
        self.ambiguous_margin  = ambiguous_margin
        self.mog2_boost_radius = mog2_boost_radius

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute(
            self,
            trash_center:   tuple[int, int],
            p_id:           int,
            history:        deque,
            current_frame:  int,
            flow_vecs:      dict,
    ) -> float:
        """
        Tính điểm ownership [0.0, 1.0] cho một người cụ thể.

        Args:
            trash_center:  (tx, ty) tâm vật rác.
            p_id:          ID người cần tính.
            history:       deque[(cx, cy, frame_idx)] lịch sử vị trí người đó.
            current_frame: Index frame hiện tại.
            flow_vecs:     {p_id → deque[(vx, vy)]}.

        Returns:
            Điểm tổng hợp trong [0.0, 1.0].
        """
        if not history:
            return 0.0

        tx, ty    = trash_center
        positions = list(history)
        min_dist, closest_offset = self._nearest(positions, tx, ty, current_frame)

        if min_dist > self.trajectory_radius:
            return 0.0

        proximity_score = 1.0 - (min_dist / self.trajectory_radius)
        direction_score = self._direction_score(positions, tx, ty)
        flow_score      = self._flow_score(p_id, positions, tx, ty, flow_vecs)
        recency_score   = max(0.0, 1.0 - (closest_offset / self.history_frames))

        score = (
                0.40 * proximity_score
                + 0.30 * direction_score
                + 0.20 * flow_score
                + 0.10 * recency_score
        )
        return round(score, 4)

    def find_best_owner(
            self,
            trash_center:    tuple[int, int],
            current_persons: dict[int, tuple[int, int]],
            current_frame:   int,
            p_history:       dict,
            flow_vecs:       dict,
            mog2_alerts:     list | None = None,
    ) -> tuple[int | None, float, bool]:
        """
        Duyệt tất cả người trong scene, chọn người có điểm cao nhất.

        Returns:
            (best_owner_id | None, best_score, is_ambiguous)
        """
        mog2_near = mog2_alerts and any(
            math.hypot(mx - trash_center[0], my - trash_center[1]) < self.mog2_boost_radius
            for mx, my, _ in mog2_alerts
        )

        scores: dict[int, float] = {}
        for p_id, (cx, cy) in current_persons.items():
            if p_id in p_history and p_history[p_id]:
                s = self.compute(trash_center, p_id, p_history[p_id], current_frame, flow_vecs)
            else:
                dist = math.hypot(cx - trash_center[0], cy - trash_center[1])
                s = (1.0 - dist / self.spawn_radius) * 0.7 if dist < self.spawn_radius else 0.0

            if mog2_near and s > 0:
                s = min(1.0, s * 1.15)
            if s > 0:
                scores[p_id] = s

        if not scores:
            return None, 0.0, False

        sorted_s = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        best_id, best_s = sorted_s[0]

        if best_s < self.min_score:
            return None, best_s, False

        is_ambig = (
                len(sorted_s) >= 2
                and (best_s - sorted_s[1][1]) < self.ambiguous_margin
        )
        return best_id, best_s, is_ambig

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _nearest(positions, tx, ty, current_frame):
        min_dist, closest_offset = float("inf"), 0
        for cx, cy, fidx in positions:
            d = math.hypot(cx - tx, cy - ty)
            if d < min_dist:
                min_dist = d
                closest_offset = current_frame - fidx
        return min_dist, closest_offset

    @staticmethod
    def _direction_score(positions, tx, ty) -> float:
        if len(positions) < 3:
            return 0.0
        recent = positions[-1]
        mid    = positions[max(0, len(positions) // 2)]
        move   = np.array([recent[0] - mid[0], recent[1] - mid[1]], dtype=float)
        m_dist = np.linalg.norm(move)
        if m_dist <= 2:
            return 0.0
        move_n = move / m_dist
        away   = np.array([recent[0] - tx, recent[1] - ty], dtype=float)
        a_dist = np.linalg.norm(away)
        if a_dist <= 1:
            return 0.0
        return max(0.0, float(np.dot(move_n, away / a_dist)))

    @staticmethod
    def _flow_score(p_id, positions, tx, ty, flow_vecs) -> float:
        if p_id not in flow_vecs or not flow_vecs[p_id]:
            return 0.0
        recent_flows = list(flow_vecs[p_id])[-4:]
        avg_vx  = np.mean([v[0] for v in recent_flows])
        avg_vy  = np.mean([v[1] for v in recent_flows])
        flow_mag = math.hypot(avg_vx, avg_vy)
        if flow_mag <= 0.5:
            return 0.0
        flow_n = np.array([avg_vx, avg_vy]) / flow_mag
        latest = positions[-1]
        away   = np.array([latest[0] - tx, latest[1] - ty], dtype=float)
        a_mag  = np.linalg.norm(away)
        if a_mag <= 1:
            return 0.0
        return max(0.0, float(np.dot(flow_n, away / a_mag)))
