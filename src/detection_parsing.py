# ==============================================================================
# DETECTION PARSING / ID HELPERS
# ==============================================================================
import cv2
import math
from collections import deque
import numpy as np


class DetectionParsingMixin:
    """Turn model outputs and motion blobs into stable person/trash ids."""
    def _parse_detections(
            self, results, frame_idx: int
    ) -> tuple[dict[int, tuple[int, int]], dict[int, tuple[int, int]]]:
        current_persons: dict[int, tuple[int, int]] = {}
        current_trashes: dict[int, tuple[int, int]] = {}
        boxes = results[0].boxes
        if boxes is None:
            return current_persons, current_trashes
        ids = boxes.id.cpu().numpy() if boxes.id is not None else [None] * len(boxes)
        used_trash_ids: set[int] = set()
        for box, obj_id, cls, conf in zip(
                boxes.xyxy.cpu().numpy(),
                ids,
                boxes.cls.cpu().numpy(),
                boxes.conf.cpu().numpy(),
        ):
            x1, y1, x2, y2 = box
            cx  = int((x1 + x2) / 2)
            cy  = int((y1 + y2) / 2)
            if int(cls) == 0:   # person
                if not self._is_valid_person_box(box, conf, results):
                    continue
                # Dùng điểm neo gần chân vì rác đặt dưới đất thường gần chân,
                # không gần tâm thân người.
                anchor, points = self._person_points_from_box(box, results[0].orig_shape)
                oid = self._resolve_person_id(obj_id, anchor, frame_idx, current_persons)
                current_persons[oid] = anchor
                if oid not in self._person_history:
                    self._person_history[oid] = deque(maxlen=self.cfg.HISTORY_FRAMES)
                self._person_seen_counts[oid] = self._person_seen_counts.get(oid, 0) + 1
                self._person_history[oid].append(
                    self._make_person_history_entry(anchor, points, frame_idx)
                )
                if self._person_seen_counts[oid] <= 4 or frame_idx % 50 == 0:
                    print(
                        f"[PERSON] id={oid} conf={float(conf):.2f} seen={self._person_seen_counts[oid]} "
                        f"anchor={anchor} points={len(points)} frame={frame_idx}",
                        flush=True,
                    )
            elif int(cls) == 1:  # trash
                oid = self._resolve_trash_id(obj_id, (cx, cy), frame_idx, used_trash_ids)
                used_trash_ids.add(oid)
                current_trashes[oid] = (cx, cy)
        return current_persons, current_trashes

    def _augment_persons_from_motion(
            self,
            current_persons: dict[int, tuple[int, int]],
            current_trashes: dict[int, tuple[int, int]],
            mog2_alerts: list,
            frame_idx: int,
            frame_shape,
            annotated: np.ndarray,
    ) -> None:
        if not getattr(self.cfg, "MOTION_PERSON_FALLBACK", False):
            return
        if not mog2_alerts:
            return

        h, w = frame_shape[:2]
        min_area = float(getattr(self.cfg, "MOTION_PERSON_MIN_AREA", 1200))
        max_area = float(w * h * getattr(self.cfg, "MOTION_PERSON_MAX_AREA_RATIO", 0.45))
        min_y = int(h * float(getattr(self.cfg, "MOTION_PERSON_MIN_Y_RATIO", 0.10)))
        max_y = int(h * float(getattr(self.cfg, "MOTION_PERSON_MAX_Y_RATIO", 0.98)))
        max_items = int(getattr(self.cfg, "MOTION_PERSON_MAX_PER_FRAME", 2) or 2)
        foot_ratio = float(getattr(self.cfg, "MOTION_PERSON_FOOT_OFFSET_RATIO", 0.35))

        added = 0
        for mx, my, area in sorted(mog2_alerts, key=lambda item: item[2], reverse=True):
            if added >= max_items:
                break
            if area < min_area or area > max_area or my < min_y or my > max_y:
                continue
            if any(math.hypot(mx - px, my - py) <= 80 for px, py in current_persons.values()):
                continue
            if any(math.hypot(mx - tx, my - ty) <= 55 for tx, ty in current_trashes.values()) and area < min_area * 2:
                continue

            offset = int((area ** 0.5) * foot_ratio)
            anchor = (int(mx), min(h - 1, int(my + offset)))
            points = [
                (int(mx), int(my)),
                anchor,
                (int(mx), min(h - 1, int(my + offset * 2))),
            ]
            oid = self._resolve_person_id(None, anchor, frame_idx, current_persons)
            current_persons[oid] = anchor
            if oid not in self._person_history:
                self._person_history[oid] = deque(maxlen=self.cfg.HISTORY_FRAMES)
            self._person_seen_counts[oid] = self._person_seen_counts.get(oid, 0) + 1
            self._person_history[oid].append(
                self._make_person_history_entry(anchor, points, frame_idx)
            )

            cv2.circle(annotated, anchor, 10, (255, 180, 0), 2)
            cv2.putText(
                annotated,
                f"motion P{oid}",
                (anchor[0] + 8, max(18, anchor[1] - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 180, 0),
                1,
            )
            if self._person_seen_counts[oid] <= 4 or frame_idx % 50 == 0:
                print(
                    f"[MOTION_PERSON] id={oid} area={float(area):.0f} "
                    f"seen={self._person_seen_counts[oid]} anchor={anchor} frame={frame_idx}",
                    flush=True,
                )
            added += 1

    def _is_valid_person_box(self, box, conf, results) -> bool:
        if float(conf) < float(getattr(self.cfg, "LIVE_PERSON_CONF", 0.0)):
            return False
        x1, y1, x2, y2 = box
        h, w = results[0].orig_shape[:2]
        bw, bh = max(1.0, x2 - x1), max(1.0, y2 - y1)
        min_h = h * float(getattr(self.cfg, "LIVE_PERSON_MIN_HEIGHT_RATIO", 0.0))
        min_area = w * h * float(getattr(self.cfg, "LIVE_PERSON_MIN_AREA_RATIO", 0.0))
        if bh < min_h or (bw * bh) < min_area:
            return False
        return True

    def _person_points_from_box(self, box, frame_shape) -> tuple[tuple[int, int], list[tuple[int, int]]]:
        x1, y1, x2, y2 = [float(v) for v in box]
        h, w = frame_shape[:2]
        bw, bh = max(1.0, x2 - x1), max(1.0, y2 - y1)
        cx = (x1 + x2) / 2.0
        extend_down = float(getattr(self.cfg, "PERSON_BOX_EXTEND_DOWN_RATIO", 0.55))
        extend_side = float(getattr(self.cfg, "PERSON_BOX_EXTEND_SIDE_RATIO", 0.20))

        def clamp_point(px, py) -> tuple[int, int]:
            return (
                max(0, min(w - 1, int(px))),
                max(0, min(h - 1, int(py))),
            )

        lower = clamp_point(cx, y1 + bh * 0.88)
        raw_points = [
            lower,
            clamp_point(cx, y2),
            clamp_point(cx, y2 + bh * extend_down),
            clamp_point(cx, y1 + bh * 0.55),
            clamp_point(x1 - bw * extend_side, y2 + bh * extend_down * 0.45),
            clamp_point(x2 + bw * extend_side, y2 + bh * extend_down * 0.45),
        ]
        points = []
        for point in raw_points:
            if point not in points:
                points.append(point)
        return lower, points

    @staticmethod
    def _make_person_history_entry(
            anchor: tuple[int, int],
            points: list[tuple[int, int]],
            frame_idx: int,
    ) -> dict:
        return {
            "anchor": (int(anchor[0]), int(anchor[1])),
            "points": [(int(x), int(y)) for x, y in points],
            "frame": int(frame_idx),
        }

    @staticmethod
    def _history_entry_anchor(entry) -> tuple[int, int]:
        if isinstance(entry, dict):
            anchor = entry.get("anchor", (0, 0))
            return int(anchor[0]), int(anchor[1])
        return int(entry[0]), int(entry[1])

    @staticmethod
    def _history_entry_frame(entry) -> int:
        if isinstance(entry, dict):
            return int(entry.get("frame", 0))
        return int(entry[2]) if len(entry) >= 3 else 0

    @staticmethod
    def _history_entry_points(entry) -> list[tuple[int, int]]:
        if isinstance(entry, dict):
            points = entry.get("points") or [entry.get("anchor", (0, 0))]
        elif len(entry) >= 4 and entry[3]:
            points = entry[3]
        else:
            points = [DetectionParsingMixin._history_entry_anchor(entry)]
        return [(int(x), int(y)) for x, y in points]

    def _nearest_history_distance(
            self,
            center: tuple[int, int],
            hist: deque,
            frame_idx: int,
            max_age: int,
    ) -> tuple[float, int | None]:
        best_dist, best_frame = float("inf"), None
        for entry in hist:
            entry_frame = self._history_entry_frame(entry)
            age = frame_idx - entry_frame
            if age < 0 or age > max_age:
                continue
            for px, py in self._history_entry_points(entry):
                dist = math.hypot(center[0] - px, center[1] - py)
                if dist < best_dist:
                    best_dist, best_frame = dist, entry_frame
        return best_dist, best_frame

    def _resolve_person_id(
            self,
            raw_id,
            anchor: tuple[int, int],
            frame_idx: int,
            current_persons: dict[int, tuple[int, int]],
    ) -> int:
        match_radius = float(getattr(self.cfg, "PERSON_ID_MATCH_RADIUS", 160))
        best_id, best_dist = None, float("inf")
        for pid, hist in self._person_history.items():
            if pid in current_persons or not hist:
                continue
            last_x, last_y = self._history_entry_anchor(hist[-1])
            last_frame = self._history_entry_frame(hist[-1])
            if frame_idx - last_frame > self.cfg.HISTORY_FRAMES:
                continue
            dist = math.hypot(anchor[0] - last_x, anchor[1] - last_y)
            if dist < best_dist and dist <= match_radius:
                best_id, best_dist = pid, dist
        if best_id is not None:
            return best_id
        if raw_id is not None:
            oid = int(raw_id)
            if oid not in current_persons:
                return oid
        oid = self._next_synthetic_person_id
        self._next_synthetic_person_id += 1
        return oid

    def _resolve_trash_id(
            self,
            raw_id,
            center: tuple[int, int],
            frame_idx: int,
            used_ids: set[int],
    ) -> int:
        match_radius = float(getattr(self.cfg, "TRASH_ID_MATCH_RADIUS", 80))
        best_id, best_dist = None, float("inf")
        for tid, data in self._trash_registry.items():
            if tid in used_ids:
                continue
            last_seen = data.get("last_seen_frame", data.get("spawn_frame", frame_idx))
            if frame_idx - last_seen > self.cfg.STALE_FRAMES:
                continue
            last_pos = data.get("last_pos")
            if not last_pos:
                continue
            dist = math.hypot(center[0] - last_pos[0], center[1] - last_pos[1])
            if dist < best_dist and dist <= match_radius:
                best_id, best_dist = tid, dist
        if best_id is not None:
            return best_id

        if raw_id is not None:
            oid = int(raw_id)
            if oid not in used_ids:
                return oid

        oid = self._next_synthetic_trash_id
        self._next_synthetic_trash_id += 1
        return oid

    def _detect_floor_trash_candidates(
            self,
            model,
            frame: np.ndarray,
            annotated: np.ndarray,
            current_trashes: dict[int, tuple[int, int]],
            current_persons: dict[int, tuple[int, int]],
            mog2_alerts: list,
            frame_idx: int,
            floor_kwargs: dict | None,
    ) -> dict[int, tuple[int, int]]:
        if floor_kwargs is None:
            return {}
        interval = int(getattr(self.cfg, "LIVE_FLOOR_PASS_INTERVAL", 1) or 1)
        if interval > 1 and frame_idx % interval != 0:
            return {}

        h, w = frame.shape[:2]
        roi_top_ratio = float(getattr(self.cfg, "LIVE_FLOOR_ROI_TOP", 0.45))
        y0 = min(h - 1, max(0, int(h * roi_top_ratio)))
        roi = frame[y0:h, :]
        if roi.size == 0:
            return {}

        try:
            results = model.predict(roi, **floor_kwargs)
        except Exception:
            return {}

        extra: dict[int, tuple[int, int]] = {}
        used_ids = set(current_trashes)
        boxes = results[0].boxes if results else None
        if boxes is None:
            return extra

        for box, cls, conf in zip(
                boxes.xyxy.cpu().numpy(),
                boxes.cls.cpu().numpy(),
                boxes.conf.cpu().numpy(),
        ):
            if int(cls) != 1:
                continue
            x1, y1, x2, y2 = box
            bw, bh = max(1.0, x2 - x1), max(1.0, y2 - y1)
            area = bw * bh
            if area < 60 or area > (w * h * 0.12):
                continue
            cx = int((x1 + x2) / 2)
            cy = int(y0 + (y1 + y2) / 2)
            center = (cx, cy)
            if not self._floor_candidate_has_context(center, current_persons, mog2_alerts, frame_idx):
                continue
            if any(math.hypot(cx - tx, cy - ty) < 35 for tx, ty in current_trashes.values()):
                continue
            tid = self._resolve_trash_id(None, center, frame_idx, used_ids)
            used_ids.add(tid)
            extra[tid] = center

            gx1, gy1, gx2, gy2 = int(x1), int(y0 + y1), int(x2), int(y0 + y2)
            color = (0, 165, 255)
            cv2.rectangle(annotated, (gx1, gy1), (gx2, gy2), color, 2)
            cv2.putText(
                annotated,
                f"id:{tid} trash floor {float(conf):.2f}",
                (gx1, max(18, gy1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                1,
            )
        return extra

    def _floor_candidate_has_context(
            self,
            center: tuple[int, int],
            current_persons: dict[int, tuple[int, int]],
            mog2_alerts: list,
            frame_idx: int,
    ) -> bool:
        cx, cy = center
        near_person = any(
            math.hypot(cx - px, cy - py) <= self.cfg.SPAWN_RADIUS
            for px, py in current_persons.values()
        )
        if near_person:
            return True
        for hist in self._person_history.values():
            if not hist:
                continue
            dist, _ = self._nearest_history_distance(
                center, hist, frame_idx, self.cfg.HISTORY_FRAMES
            )
            if dist <= self.cfg.TRAJECTORY_RADIUS:
                return True
        return any(
            math.hypot(cx - mx, cy - my) <= 120
            for mx, my, _ in mog2_alerts
        )

    # ------------------------------------------------------------------
    # Trash processing
    # ------------------------------------------------------------------
