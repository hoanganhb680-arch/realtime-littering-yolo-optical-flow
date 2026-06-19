# ==============================================================================
# VIOLATION LOGGER — Xác nhận vi phạm, lưu ảnh bằng chứng, ghi log
# ==============================================================================
import os
import cv2
import numpy as np
from Config     import Config
from ApiSyncer import ApiSyncer
import db as _db

class ViolationLogger:
    """
    Chịu trách nhiệm:
        - Lưu ảnh bằng chứng (annotated frame).
        - Thêm overlay text trên frame hiển thị.
        - Ghi vào violation_log nội bộ.
        - Lưu vào SQLite DB.
        - Trigger ApiSyncer để đẩy lên Backend.
    """
    def __init__(
            self,
            output_dir: str       = Config.OUTPUT_DIR,
            syncer:     ApiSyncer | None = None,
    ):
        self.output_dir    = output_dir
        self._syncer       = syncer or ApiSyncer()
        self.violation_log: list[dict] = []
        os.makedirs(output_dir, exist_ok=True)
        _db.init_db()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def confirm_and_log(
            self,
            t_id:            int,
            t_center:        tuple[int, int],
            data:            dict,
            vtype:           str,
            annotated:       np.ndarray,
            current_persons: dict[int, tuple[int, int]],
            frame_idx:       int,
    ) -> None:
        """
        Đánh dấu vi phạm đã xác nhận, lưu bằng chứng, ghi log & đẩy API.
        Chỉnh sửa trực tiếp `annotated` (overlay text/box) và `data['status']`.
        """
        data["status"] = "confirmed"
        owner_id = data["owner_id"]

        if owner_id is not None and not data["is_ambiguous"]:
            self._save_evidence(
                t_id, t_center, data, vtype, annotated, current_persons, frame_idx,
                owner_id,
            )
        elif data["is_ambiguous"]:
            self._draw_ambiguous(t_center, annotated)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _save_evidence(
            self,
            t_id,
            t_center,
            data,
            vtype,
            annotated,
            current_persons,
            frame_idx,
            owner_id,
    ):
        evidence = self._decode_frame(data.get("owner_frame_jpg"))
        if evidence is None:
            evidence = annotated.copy()

        # Overlay trên ảnh bằng chứng
        cv2.putText(
            evidence, f"VI PHAM: Person_{owner_id}",
            (t_center[0] - 10, t_center[1] - 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2,
        )
        cv2.circle(evidence, t_center, 25, (0, 0, 255), 3)
        if owner_id in current_persons:
            px, py = current_persons[owner_id]
            cv2.rectangle(evidence, (px - 40, py - 80), (px + 40, py + 40), (0, 0, 255), 2)

        local_path = os.path.join(
            self.output_dir,
            f"violation_P{owner_id}_T{t_id}_F{frame_idx}.jpg",
        )
        cv2.imwrite(local_path, evidence)

        v_data = {
            "person_id" : owner_id,
            "trash_id"  : t_id,
            "score"     : data["score"],
            "frame"     : frame_idx,
            "time"      : data["spawn_time"],
            "type"      : vtype,
            "local_path": local_path,
        }
        self.violation_log.append(v_data)
        print(
            f"[DETECT] violation person={owner_id} trash={t_id} "
            f"score={data['score']:.4f} frame={frame_idx} file={local_path}",
            flush=True,
        )

        # Gửi API bất đồng bộ (upload ảnh lên MinIO + lưu vào DB qua POST endpoint)
        self._syncer.send(v_data, local_path)

        # Overlay trên frame video đang hiển thị
        cv2.putText(
            annotated,
            f"VI PHAM: P{owner_id} ({data['score']:.2f})",
            (t_center[0] - 10, t_center[1] - 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2,
        )
        cv2.circle(annotated, t_center, 20, (0, 0, 255), 2)

    def save_candidate(
            self,
            t_id: int,
            t_center: tuple[int, int],
            annotated: np.ndarray,
            frame_idx: int,
            reason: str = "owner_none",
    ) -> None:
        evidence = annotated.copy()
        cv2.putText(
            evidence,
            f"CANDIDATE: {reason}",
            (max(5, t_center[0] - 20), max(30, t_center[1] - 30)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 165, 255),
            2,
        )
        cv2.circle(evidence, t_center, 22, (0, 165, 255), 2)

        image_path = os.path.join(
            self.output_dir,
            f"candidate_T{t_id}_F{frame_idx}_{reason}.jpg",
        )
        cv2.imwrite(image_path, evidence)
        print(
            f"[CANDIDATE] trash={t_id} reason={reason} frame={frame_idx} "
            f"file={image_path}",
            flush=True,
        )

    @staticmethod
    def _decode_frame(jpg_bytes: bytes | None) -> np.ndarray | None:
        if not jpg_bytes:
            return None
        arr = np.frombuffer(jpg_bytes, dtype=np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)

    @staticmethod
    def _draw_ambiguous(t_center, annotated):
        cv2.putText(
            annotated, "KHONG RO NGUOI",
            (t_center[0] - 10, t_center[1] - 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 165, 255), 2,
        )
