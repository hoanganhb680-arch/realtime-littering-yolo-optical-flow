# ==============================================================================
# OWNER / EVIDENCE HELPERS
# ==============================================================================
import cv2
import math
from collections import deque
import numpy as np


class OwnerResolutionMixin:
    """Resolve who owns a trash candidate and keep evidence frames for that owner."""

    def _owner_is_usable(self, owner_id) -> bool:
        if owner_id is None:
            return False
        min_seen = getattr(self.cfg, "MIN_OWNER_SEEN_FRAMES", 1)
        if self._person_seen_counts.get(owner_id, 0) < min_seen:
            return False
        return self._owner_has_motion(self._person_history.get(owner_id, deque()))

    def _encode_evidence_frame(self, annotated: np.ndarray) -> bytes | None:
        quality = int(getattr(self.cfg, "EVIDENCE_CLIP_JPEG_QUALITY", 65))
        ok, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, quality])
        return buf.tobytes() if ok else None

    def _remember_person_evidence(
            self,
            current_persons: dict[int, tuple[int, int]],
            annotated: np.ndarray,
    ) -> None:
        if not current_persons:
            return
        frame_jpg = self._encode_evidence_frame(annotated)
        if frame_jpg is None:
            return
        for person_id in current_persons:
            self._person_frame_jpg[person_id] = frame_jpg

    def _find_owner_for_trash(
            self,
            t_center: tuple[int, int],
            current_persons: dict[int, tuple[int, int]],
            mog2_alerts: list,
            frame_idx: int,
    ) -> tuple[int | None, float, bool]:
        owner_id, score, is_ambig = self._scorer.find_best_owner(
            t_center, current_persons, frame_idx,
            self._person_history, self._flow_tracker.flow_vecs, mog2_alerts,
        )
        if owner_id is not None and self._owner_is_usable(owner_id):
            return owner_id, score, is_ambig

        fallback_owner, fallback_score, fallback_ambig = self._fallback_recent_owner(t_center, frame_idx)
        if fallback_owner is not None:
            return fallback_owner, fallback_score, fallback_ambig
        return owner_id, score, is_ambig

    def _owner_evidence_frame(
            self,
            owner_id,
            owner_in_scene: bool,
            annotated: np.ndarray,
            fallback_jpg: bytes | None = None,
    ) -> bytes | None:
        if owner_id is not None and not owner_in_scene:
            return (
                self._person_frame_jpg.get(owner_id)
                or fallback_jpg
                or self._encode_evidence_frame(annotated)
            )
        return self._encode_evidence_frame(annotated) or fallback_jpg

    def _apply_owner_update(
            self,
            data: dict,
            owner_id,
            score: float,
            is_ambig: bool,
            current_persons: dict[int, tuple[int, int]],
            annotated: np.ndarray,
            frame_idx: int,
    ) -> None:
        owner_in_scene = owner_id is not None and owner_id in current_persons
        data.update({
            "owner_id": owner_id,
            "score": score,
            "is_ambiguous": is_ambig,
            "owner_seen_count": (
                self._person_seen_counts.get(owner_id, 0)
                if owner_id is not None else 0
            ),
            "owner_frame_jpg": self._owner_evidence_frame(
                owner_id, owner_in_scene, annotated, data.get("owner_frame_jpg")
            ),
        })
        if owner_id is None:
            return
        if owner_in_scene:
            data["owner_left_frame"] = None
            data["stationary_after_owner_gone"] = 0
        elif data.get("owner_left_frame") is None:
            data["owner_left_frame"] = frame_idx
            data["stationary_after_owner_gone"] = 1

    def _should_apply_owner_update(self, data: dict, owner_id, score: float, is_ambig: bool) -> bool:
        if owner_id is None:
            return False
        if data.get("owner_id") is None:
            return True
        if not self._owner_is_usable(data.get("owner_id")):
            return True
        if data.get("is_ambiguous") and not is_ambig:
            return True
        return score > data.get("score", 0.0)

    def _update_owner_presence(
            self,
            data: dict,
            current_persons: dict[int, tuple[int, int]],
            frame_idx: int,
            is_stationary_now: bool,
    ) -> tuple[int | None, bool]:
        owner_id = data.get("owner_id")
        owner_in_scene = owner_id is not None and owner_id in current_persons
        if owner_id is not None:
            data["owner_seen_count"] = max(
                data.get("owner_seen_count", 0),
                self._person_seen_counts.get(owner_id, 0),
            )
        if owner_id is not None and not owner_in_scene and data.get("owner_left_frame") is None:
            data["owner_left_frame"] = frame_idx
        if data.get("owner_left_frame") is not None and is_stationary_now:
            data["stationary_after_owner_gone"] += 1
        return owner_id, owner_in_scene

    def _save_ownerless_candidate(self, t_id, t_center, annotated, frame_idx) -> None:
        if not getattr(self.cfg, "SAVE_OWNERLESS_CANDIDATES", False):
            return
        cooldown = int(getattr(self.cfg, "OWNERLESS_CANDIDATE_COOLDOWN_FRAMES", 60) or 60)
        if frame_idx - self._last_ownerless_candidate_frame < cooldown:
            return
        self._last_ownerless_candidate_frame = frame_idx
        self._logger.save_candidate(
            t_id,
            t_center,
            annotated,
            frame_idx,
            clip_frames=self._get_evidence_clip_frames(annotated),
            reason="owner_none",
        )

    def _fallback_recent_owner(
            self,
            t_center: tuple[int, int],
            frame_idx: int,
    ) -> tuple[int | None, float, bool]:
        max_age = int(getattr(self.cfg, "RECENT_OWNER_GRACE_FRAMES", self.cfg.HISTORY_FRAMES))
        radius = float(getattr(self.cfg, "RECENT_OWNER_RADIUS", self.cfg.TRAJECTORY_RADIUS))
        candidates = []
        for pid, hist in self._person_history.items():
            if not hist or self._person_seen_counts.get(pid, 0) < self.cfg.MIN_OWNER_SEEN_FRAMES:
                continue
            if not self._owner_has_motion(hist):
                continue
            dist, nearest_frame = self._nearest_history_distance(t_center, hist, frame_idx, max_age)
            if nearest_frame is None:
                continue
            age = frame_idx - nearest_frame
            if dist > radius:
                continue
            distance_score = 1.0 - (dist / radius)
            recency_score = 1.0 - (age / max_age)
            score = round(max(self.cfg.MIN_SCORE, 0.70 * distance_score + 0.30 * recency_score), 4)
            candidates.append((pid, score, dist))
        if not candidates:
            return None, 0.0, False
        candidates.sort(key=lambda item: item[1], reverse=True)
        best_id, best_score, _ = candidates[0]
        is_ambig = len(candidates) >= 2 and (best_score - candidates[1][1]) < self.cfg.AMBIGUOUS_MARGIN
        return best_id, best_score, is_ambig

    def _owner_has_motion(self, hist: deque) -> bool:
        min_motion = float(getattr(self.cfg, "MIN_OWNER_MOTION_PX", 0) or 0)
        if min_motion <= 0 or len(hist) < 2:
            return True
        anchors = [self._history_entry_anchor(entry) for entry in hist]
        first_x, first_y = anchors[0]
        return any(math.hypot(px - first_x, py - first_y) >= min_motion for px, py in anchors[1:])
