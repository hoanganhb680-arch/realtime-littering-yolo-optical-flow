# ==============================================================================
# VIOLATION CONFIRMATION / DRAWING HELPERS
# ==============================================================================
import cv2
from collections import deque
import numpy as np


class ViolationConfirmationMixin:
    """Decide when a tracked trash becomes a violation and update the UI overlay."""

    def _try_confirm(self, t_id, t_center, data, annotated, current_persons, frame_idx):
        status = self._confirmation_status(t_center, data, annotated, current_persons, frame_idx)
        if status["violation_type"] is None:
            self._log_pending_reason(t_id, data, current_persons, frame_idx, status)
            return

        owner_id = data["owner_id"]
        score = data["score"]
        if owner_id is not None and not data["is_ambiguous"]:
            old_score = self._person_highest_score.get(owner_id)
            if old_score is not None and score <= old_score:
                return
            self._person_highest_score[owner_id] = score

        self._logger.confirm_and_log(
            t_id, t_center, data, status["violation_type"], annotated, current_persons, frame_idx,
            clip_frames=self._get_evidence_clip_frames(annotated),
        )
        self._push_alert({
            "type": "violation",
            "data": {
                "personId": data["owner_id"],
                "trashId": t_id,
                "violationType": status["violation_type"],
                "score": round(data["score"], 4),
                "timestamp": data["spawn_time"],
                "frame": frame_idx,
            },
        })

    def _confirmation_status(self, t_center, data, annotated, current_persons, frame_idx) -> dict:
        cfg = self.cfg
        owner_id = data.get("owner_id")
        owner_seen_count = self._person_seen_counts.get(owner_id, data.get("owner_seen_count", 0))
        frames_owner_gone = (
            frame_idx - data["owner_left_frame"]
            if data["owner_left_frame"] is not None
            else 0
        )
        owner_truly_gone = (
            data["owner_left_frame"] is not None
            and frames_owner_gone >= cfg.MIN_OWNER_GONE_FRAMES
        )
        owner_seen_enough = (
            owner_id is not None
            and owner_seen_count >= getattr(cfg, "MIN_OWNER_SEEN_FRAMES", 1)
            and self._owner_has_motion(self._person_history.get(owner_id, deque()))
        )
        ground_condition = self._is_ground_trash(t_center, annotated)
        common_ok = (
            data["score"] >= cfg.MIN_SCORE
            and owner_seen_enough
            and not data["is_ambiguous"]
            and ground_condition
        )
        frames_since_spawn = frame_idx - data["spawn_frame"]
        sudden_ok = (
            common_ok
            and owner_truly_gone
            and frames_since_spawn <= cfg.CONFIRM_FRAMES_SUDDEN + cfg.MIN_OWNER_GONE_FRAMES
            and data["confirm_ctr"] >= cfg.CONFIRM_FRAMES_SUDDEN
        )
        stationary_ok = (
            common_ok
            and owner_truly_gone
            and data["confirm_ctr"] >= cfg.CONFIRM_FRAMES
            and data["stationary_after_owner_gone"] >= cfg.STATIONARY_REQUIRED
        )
        proximity_ok = (
            common_ok
            and data["confirm_ctr"] >= cfg.CONFIRM_FRAMES
            and data["stationary_ctr"] >= cfg.STATIONARY_REQUIRED
            and data["owner_id"] in current_persons
        )
        violation_type = None
        if sudden_ok:
            violation_type = "\u0110\u1ed9t ng\u1ed9t"
        elif stationary_ok:
            violation_type = "\u0110\u1ee9ng y\u00ean"
        elif proximity_ok:
            violation_type = "B\u1ecf r\u00e1c t\u1ea1i ch\u1ed7"

        return {
            "violation_type": violation_type,
            "owner_id": owner_id,
            "owner_seen_count": owner_seen_count,
            "owner_seen_enough": owner_seen_enough,
            "owner_truly_gone": owner_truly_gone,
            "frames_owner_gone": frames_owner_gone,
            "ground_condition": ground_condition,
        }

    def _log_pending_reason(
            self,
            t_id: int,
            data: dict,
            current_persons: dict[int, tuple[int, int]],
            frame_idx: int,
            status: dict,
    ) -> None:
        interval = int(getattr(self.cfg, "PENDING_LOG_INTERVAL_FRAMES", 30) or 30)
        last_log = self._last_pending_reason_log.get(t_id, -interval)
        if frame_idx - last_log < interval:
            return
        self._last_pending_reason_log[t_id] = frame_idx

        owner_id = status["owner_id"]
        reasons = []
        if owner_id is None:
            reasons.append("owner_none")
        if data.get("is_ambiguous"):
            reasons.append("ambiguous_owner")
        if data.get("score", 0.0) < self.cfg.MIN_SCORE:
            reasons.append("score_low")
        if not status["owner_seen_enough"]:
            reasons.append("owner_not_usable")
        if not status["ground_condition"]:
            reasons.append("not_ground")
        if owner_id is not None and owner_id not in current_persons and not status["owner_truly_gone"]:
            reasons.append("waiting_owner_gone")
        if data.get("confirm_ctr", 0) < self.cfg.CONFIRM_FRAMES:
            reasons.append("confirm_frames")
        if data.get("stationary_after_owner_gone", 0) < self.cfg.STATIONARY_REQUIRED:
            reasons.append("stationary_after_gone")

        print(
            "[PENDING] "
            f"trash={t_id} owner={owner_id} score={data.get('score', 0.0):.4f} "
            f"ambig={data.get('is_ambiguous')} seen={status['owner_seen_count']} "
            f"in_scene={owner_id is not None and owner_id in current_persons} "
            f"gone_frames={status['frames_owner_gone']} confirm={data.get('confirm_ctr', 0)} "
            f"stationary={data.get('stationary_ctr', 0)} "
            f"after_gone={data.get('stationary_after_owner_gone', 0)} "
            f"ground={status['ground_condition']} reason={','.join(reasons) or 'waiting'}",
            flush=True,
        )

    def _is_ground_trash(self, t_center: tuple[int, int], annotated: np.ndarray) -> bool:
        min_y_ratio = float(getattr(self.cfg, "VIOLATION_GROUND_MIN_Y_RATIO", 0.0) or 0.0)
        if min_y_ratio <= 0:
            return True
        return t_center[1] >= int(annotated.shape[0] * min_y_ratio)

    def _cleanup_stale_trash(self, current_trashes: dict, frame_idx: int) -> None:
        stale = [
            tid for tid, tdata in self._trash_registry.items()
            if tid not in current_trashes
            and (frame_idx - tdata.get("last_seen_frame", tdata["spawn_frame"])) > self.cfg.STALE_FRAMES
        ]
        for tid in stale:
            del self._trash_registry[tid]

    def _draw_sticky_trash(self, annotated: np.ndarray, current_trashes: dict, frame_idx: int) -> None:
        max_missed = int(getattr(self.cfg, "DRAW_STALE_TRASH_FRAMES", 0) or 0)
        if max_missed <= 0:
            return
        for tid, data in self._trash_registry.items():
            if tid in current_trashes:
                continue
            last_seen = data.get("last_seen_frame", data.get("spawn_frame", frame_idx))
            missed = frame_idx - last_seen
            if missed <= 0 or missed > max_missed:
                continue
            cx, cy = data.get("last_pos", (None, None))
            if cx is None or cy is None:
                continue
            color = (0, 180, 255)
            cv2.circle(annotated, (int(cx), int(cy)), 18, color, 2)
            cv2.putText(
                annotated,
                f"id:{tid} trash hold",
                (max(0, int(cx) - 45), max(18, int(cy) - 22)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                1,
            )

    def _draw_hud(self, annotated: np.ndarray, frame_idx: int) -> None:
        cv2.putText(
            annotated,
            f"F:{frame_idx}  VI PHAM: {len(self._logger.violation_log)}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2,
        )
