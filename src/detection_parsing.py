# ==============================================================================
# DETECTION PARSING / ID HELPERS
# ==============================================================================
import math
from collections import deque


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
        bh = max(1.0, y2 - y1)
        cx = (x1 + x2) / 2.0

        def clamp_point(px, py) -> tuple[int, int]:
            return (
                max(0, min(w - 1, int(px))),
                max(0, min(h - 1, int(py))),
            )

        lower = clamp_point(cx, y1 + bh * 0.88)
        foot = clamp_point(cx, y2)
        points = [lower] if foot == lower else [lower, foot]
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

