# ==============================================================================
# TRASH LIFECYCLE HELPERS
# ==============================================================================
import math
import time
from collections import deque
import numpy as np

from owner_resolution import OwnerResolutionMixin
from violation_confirmation import ViolationConfirmationMixin


class TrashLifecycleMixin(OwnerResolutionMixin, ViolationConfirmationMixin):
    """Track trash candidates from first sighting until confirmation or cleanup."""

    # Xử lý toàn bộ rác trong frame: đăng ký mới, cập nhật cũ, recovery và confirm.
    def _process_trashes(
            self,
            current_trashes: dict,
            current_persons: dict,
            mog2_alerts: list,
            annotated: np.ndarray,
            frame_idx: int,
    ) -> None:
        for t_id, t_center in current_trashes.items():
            if t_id not in self._trash_registry:
                self._register_new_trash(
                    t_id, t_center, current_persons, mog2_alerts, annotated, frame_idx
                )
                continue

            data = self._trash_registry[t_id]
            data["last_seen_frame"] = frame_idx
            if data["status"] == "confirmed":
                continue

            self._update_trash_state(data, t_center, current_persons, mog2_alerts, annotated, frame_idx)
            self._try_confirm(t_id, t_center, data, annotated, current_persons, frame_idx)

        self._recover_recently_lost_trashes(
            current_trashes, current_persons, mog2_alerts, annotated, frame_idx
        )

    # Đăng ký rác mới xuất hiện và tính owner ban đầu cho rác đó.
    def _register_new_trash(self, t_id, t_center, current_persons, mog2_alerts, annotated, frame_idx):
        owner_id, score, is_ambig = self._find_owner_for_trash(
            t_center, current_persons, mog2_alerts, frame_idx
        )
        owner_in_scene = owner_id is not None and owner_id in current_persons
        self._trash_registry[t_id] = self._new_trash_record(
            owner_id, score, is_ambig, owner_in_scene, t_center, annotated, frame_idx
        )
        print(
            f"[TRACK] new_trash={t_id} owner={owner_id} score={score:.4f} "
            f"owner_seen={self._trash_registry[t_id]['owner_seen_count']} "
            f"owner_in_scene={owner_in_scene} frame={frame_idx}",
            flush=True,
        )
        if owner_id is None:
            self._save_ownerless_candidate(t_id, t_center, annotated, frame_idx)

    # Tạo record pending chứa owner, score, vị trí, counters và bằng chứng rác.
    def _new_trash_record(
            self,
            owner_id,
            score: float,
            is_ambig: bool,
            owner_in_scene: bool,
            t_center,
            annotated: np.ndarray,
            frame_idx: int,
    ) -> dict:
        owner_seen_count = self._person_seen_counts.get(owner_id, 0) if owner_id is not None else 0
        owner_left_frame = frame_idx if owner_id is not None and not owner_in_scene else None
        return {
            "owner_id": owner_id,
            "score": score,
            "is_ambiguous": is_ambig,
            "owner_seen_count": owner_seen_count,
            "spawn_frame": frame_idx,
            "spawn_time": time.strftime("%H:%M:%S"),
            "confirm_ctr": 1,
            "stationary_ctr": 0,
            "stationary_after_owner_gone": 1 if owner_left_frame is not None else 0,
            "owner_left_frame": owner_left_frame,
            "last_pos": t_center,
            "last_seen_frame": frame_idx,
            "seen_count": 1,
            "spawn_pos": t_center,
            "owner_frame_jpg": self._owner_evidence_frame(
                owner_id, owner_in_scene, annotated
            ),
            "status": "pending",
        }

    # Cập nhật rác đã tồn tại: confirm_ctr, stationary và owner hiện tại.
    def _update_trash_state(self, data, t_center, current_persons, mog2_alerts, annotated, frame_idx):
        data["confirm_ctr"] += 1
        data["seen_count"] = data.get("seen_count", 1) + 1

        is_stationary_now = self._mark_latest_trash_position(data, t_center)
        owner_id, owner_in_scene = self._update_owner_presence(
            data, current_persons, frame_idx, is_stationary_now
        )
        if self._should_reevaluate_owner(data, owner_id, owner_in_scene):
            self._reevaluate_owner(data, t_center, current_persons, mog2_alerts, annotated, frame_idx)

    # Cập nhật vị trí mới nhất của rác và đếm rác có đứng yên không.
    def _mark_latest_trash_position(self, data, t_center) -> bool:
        disp = math.hypot(t_center[0] - data["last_pos"][0], t_center[1] - data["last_pos"][1])
        is_stationary_now = disp < self.cfg.STATIONARY_PX
        data["stationary_ctr"] = data["stationary_ctr"] + 1 if is_stationary_now else 0
        data["last_pos"] = t_center
        return is_stationary_now

    # Quyết định có cần tính lại owner khi thông tin ban đầu chưa chắc không.
    def _should_reevaluate_owner(self, data: dict, owner_id, owner_in_scene: bool) -> bool:
        cfg = self.cfg
        reeval_frames = int(getattr(cfg, "OWNER_REEVAL_FRAMES", cfg.CONFIRM_FRAMES))
        return data["confirm_ctr"] <= reeval_frames and (
            data["owner_id"] is None
            or data["is_ambiguous"]
            or data["score"] < cfg.REEVAL_SCORE_THRESH
            or (owner_id is not None and not owner_in_scene)
            or not self._owner_is_usable(owner_id)
        )

    # Tính lại owner cho rác và áp dụng nếu kết quả tốt hơn.
    def _reevaluate_owner(self, data, t_center, current_persons, mog2_alerts, annotated, frame_idx):
        new_owner, new_score, new_ambig = self._find_owner_for_trash(
            t_center, current_persons, mog2_alerts, frame_idx
        )
        if self._should_apply_owner_update(data, new_owner, new_score, new_ambig):
            self._apply_owner_update(
                data, new_owner, new_score, new_ambig,
                current_persons, annotated, frame_idx
            )

    # Cứu rác vừa mất bbox bằng owner/MOG2/last_pos để lifecycle không bị đứt.
    def _recover_recently_lost_trashes(
            self,
            current_trashes: dict,
            current_persons: dict,
            mog2_alerts: list,
            annotated: np.ndarray,
            frame_idx: int,
    ) -> None:
        max_missed = int(getattr(self.cfg, "LOST_TRASH_RECOVERY_FRAMES", 0) or 0)
        if max_missed <= 0:
            return
        min_seen = int(getattr(self.cfg, "LOST_TRASH_MIN_SEEN", 1) or 1)
        min_ground_y = int(
            annotated.shape[0] * float(getattr(self.cfg, "LOST_TRASH_GROUND_MIN_Y_RATIO", 0.65))
        )

        for t_id, data in list(self._trash_registry.items()):
            if not self._can_recover_trash(t_id, data, current_trashes, frame_idx, max_missed, min_seen):
                continue

            recovered_pos = self._recover_lost_trash_position(
                data, current_persons, mog2_alerts, min_ground_y, annotated, frame_idx
            )
            if recovered_pos is None:
                continue

            is_stationary_now = self._mark_latest_trash_position(data, recovered_pos)
            data["confirm_ctr"] += 1
            self._update_owner_presence(data, current_persons, frame_idx, is_stationary_now)

            if self._lost_trash_needs_owner(data):
                self._reevaluate_owner(
                    data, recovered_pos, current_persons, mog2_alerts, annotated, frame_idx
                )
            self._try_confirm(t_id, recovered_pos, data, annotated, current_persons, frame_idx)

    # Kiểm tra rác có đủ điều kiện để thử phục hồi khi YOLO mất bbox không.
    def _can_recover_trash(
            self,
            t_id,
            data: dict,
            current_trashes: dict,
            frame_idx: int,
            max_missed: int,
            min_seen: int,
    ) -> bool:
        if t_id in current_trashes or data.get("status") == "confirmed":
            return False
        seen_count = data.get("seen_count", 0)
        if seen_count < min_seen and not self._can_recover_strong_single_seen_trash(data):
            return False
        last_seen = data.get("last_seen_frame", data.get("spawn_frame", frame_idx))
        missed = frame_idx - last_seen
        return 0 < missed <= max_missed

    # Cho phép cứu rác chỉ thấy 1 frame nếu owner rõ, score cao và không mơ hồ.
    def _can_recover_strong_single_seen_trash(self, data: dict) -> bool:
        return (
            data.get("seen_count", 0) >= 1
            and data.get("owner_id") is not None
            and not data.get("is_ambiguous")
            and data.get("score", 0.0) >= self.cfg.REEVAL_SCORE_THRESH
        )

    # Kiểm tra rác mất bbox có cần tính lại owner không.
    def _lost_trash_needs_owner(self, data: dict) -> bool:
        return (
            data.get("owner_id") is None
            or data.get("is_ambiguous")
            or data.get("score", 0.0) < self.cfg.REEVAL_SCORE_THRESH
            or not self._owner_is_usable(data.get("owner_id"))
        )

    # Tìm vị trí tạm cho rác bị mất bằng chân owner, MOG2 hoặc last_pos.
    def _recover_lost_trash_position(
            self,
            data: dict,
            current_persons: dict[int, tuple[int, int]],
            mog2_alerts: list,
            min_ground_y: int,
            annotated: np.ndarray,
            frame_idx: int,
    ) -> tuple[int, int] | None:
        last_pos = data.get("last_pos")
        if not last_pos:
            return None
        lx, ly = last_pos
        radius = float(getattr(self.cfg, "LOST_TRASH_FOOT_SNAP_RADIUS", 220))

        owner_pos = self._snap_lost_trash_to_owner(data, current_persons, min_ground_y, radius)
        if owner_pos is not None:
            return owner_pos

        alert_pos = self._snap_lost_trash_to_motion(lx, ly, mog2_alerts, min_ground_y, radius)
        if alert_pos is not None:
            return alert_pos

        last_pos = (int(lx), int(ly))
        if ly >= min_ground_y or self._is_ground_trash(
            last_pos, annotated, data.get("owner_id"), frame_idx
        ):
            return int(lx), int(ly)
        return None

    # Nếu owner còn trong scene, kéo vị trí rác mất về điểm chân owner gần last_pos.
    def _snap_lost_trash_to_owner(
            self,
            data: dict,
            current_persons: dict[int, tuple[int, int]],
            min_ground_y: int,
            radius: float,
    ) -> tuple[int, int] | None:
        owner_id = data.get("owner_id")
        if owner_id not in current_persons:
            return None
        lx, ly = data["last_pos"]
        hist = self._person_history.get(owner_id, deque())
        points = self._history_entry_points(hist[-1]) if hist else [current_persons[owner_id]]
        eligible = [
            point for point in points
            if point[1] >= min_ground_y and math.hypot(point[0] - lx, point[1] - ly) <= radius
        ]
        if not eligible:
            return None
        return min(eligible, key=lambda point: math.hypot(point[0] - lx, point[1] - ly))

    # Nếu có MOG2 motion gần rác cũ, dùng tâm motion làm vị trí phục hồi.
    @staticmethod
    def _snap_lost_trash_to_motion(
            lx,
            ly,
            mog2_alerts: list,
            min_ground_y: int,
            radius: float,
    ) -> tuple[int, int] | None:
        floor_alerts = [
            (mx, my, area)
            for mx, my, area in mog2_alerts
            if my >= min_ground_y and math.hypot(mx - lx, my - ly) <= radius
        ]
        if not floor_alerts:
            return None
        mx, my, _ = min(
            floor_alerts,
            key=lambda alert: math.hypot(alert[0] - lx, alert[1] - ly),
        )
        return int(mx), int(my)
