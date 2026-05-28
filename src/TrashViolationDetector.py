# ==============================================================================
# TRASH VIOLATION DETECTOR — Pipeline chính tích hợp tất cả các module
# ==============================================================================
import cv2
import math
import time
import subprocess
import os
import numpy as np
from collections import deque
from ultralytics import YOLO
from Config               import Config
from MotionDetector      import MotionDetector
from OpticalFlowTracker import OpticalFlowTracker
from OwnershipScorer     import OwnershipScorer
from ViolationLogger     import ViolationLogger
from ApiSyncer          import ApiSyncer
# Import stream queues để push frame + alert lên WebSocket
try:
    from stream_router import frame_queue, alert_queue
    _STREAM_ENABLED = True
except ImportError:
    _STREAM_ENABLED = False
class TrashViolationDetector:
    """
    Điều phối toàn bộ pipeline:
        1. Đọc video / camera → YOLO tracking
        2. MOG2 motion alert
        3. Optical flow cho từng người
        4. Tính điểm ownership khi rác mới xuất hiện
        5. Re-evaluate & xác nhận vi phạm (3 loại)
        6. Ghi video output + cleanup trash cũ
        7. Push JPEG frame + alert lên WebSocket queue

    VIDEO_SOURCE hỗ trợ:
        - int (0, 1, 2...): Webcam USB
        - "rtsp://...":     Camera IP qua RTSP
        - "video/xxx.mp4":  File video
    """
    def __init__(self, cfg: Config = Config()):
        self.cfg = cfg
        # Sub-modules
        self._motion_detector  = MotionDetector()
        self._flow_tracker     = OpticalFlowTracker()
        self._scorer           = OwnershipScorer()
        self._logger           = ViolationLogger(syncer=ApiSyncer())
        # State
        self._person_history: dict[int, deque]           = {}
        self._trash_registry: dict[int, dict]            = {}
        self.stopped                                     = False
        self._person_highest_score: dict[int, float]     = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def run(self) -> None:
        cfg = self.cfg
        self._validate_paths()
        is_live = cfg.IS_LIVE
        print(f"🚀 Khởi tạo model... nguồn: {cfg.VIDEO_SOURCE}")
        model = YOLO(cfg.MODEL_PATH)
        
        # Sử dụng ThreadedCamera cho camera trực tuyến để loại bỏ hoàn toàn độ trễ tích lũy buffer
        if is_live:
            from ThreadedCamera import ThreadedCamera
            cap = ThreadedCamera(cfg.VIDEO_SOURCE, cfg.CAMERA_BUFFER).start()
        else:
            cap = cv2.VideoCapture(cfg.VIDEO_SOURCE)

        if not cap.isOpened():
            raise RuntimeError(f"❌ Không mở được nguồn video: {cfg.VIDEO_SOURCE}")

        out_vid, fps = self._init_writer(cap)
        prev_gray, frame_idx = None, 0
        while cap.isOpened() and not self.stopped:
            t_start = time.time()  # Bắt đầu tính thời gian xử lý frame này
            ret, frame = cap.read()
            if not ret:
                if is_live:
                    print("⚠️  Mất kết nối camera, thử lại...")
                    time.sleep(0.5)
                    continue
                break
            frame_idx += 1
            curr_gray    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            mog2_alerts  = self._motion_detector.get_alerts(frame)
            
            # Tối ưu hóa 2 chế độ:
            # - IP Camera: dùng imgsz=320 để YOLO chạy siêu nhanh, giảm độ trễ truyền lên Web
            # - File video: dùng mặc định (640px) để nhận diện chính xác nhất vật thể nhỏ
            yolo_imgsz = 320 if is_live else 640
            results = model.track(frame, imgsz=yolo_imgsz, persist=True, tracker="bytetrack.yaml", verbose=False)
            
            annotated    = results[0].plot()
            current_persons, current_trashes = self._parse_detections(results, frame_idx)
            self._flow_tracker.update(prev_gray, curr_gray, current_persons)
            self._process_trashes(current_trashes, current_persons, mog2_alerts, annotated, frame_idx)
            self._cleanup_stale_trash(current_trashes, frame_idx)
            self._draw_hud(annotated, frame_idx)
            out_vid.write(annotated)

            # Push frame lên WebSocket (non-blocking)
            self._push_frame(annotated)

            prev_gray = curr_gray.copy()
            
            # Điều tiết FPS cho chế độ File video (5-7 FPS) để CPU nhẹ nhàng và xem tốc độ tự nhiên
            if not is_live:
                elapsed = time.time() - t_start
                wait_time = (1.0 / cfg.FILE_MODE_FPS) - elapsed
                if wait_time > 0:
                    time.sleep(wait_time)
                    
            if frame_idx % 50 == 0:
                print(f"  [{frame_idx} frames] xử lý xong...")
        cap.release()
        out_vid.release()
        print(f"\n✅ Xử lý hoàn tất. Ghi raw video tại: {cfg.LOCAL_VIDEO_RAW}")
        if not is_live:
            self._convert_h264()
        
        # Đẩy alert video_ended để FE biết video đã chạy hết
        self._push_alert({
            "type": "video_ended",
            "data": {
                "message": "Video đã xử lý xong!",
                "video_raw": cfg.LOCAL_VIDEO_RAW,
                "video_h264": cfg.LOCAL_VIDEO_H264,
                "total_violations": len(self._logger.violation_log)
            }
        })

    # ------------------------------------------------------------------
    # WebSocket stream helpers
    # ------------------------------------------------------------------

    def _push_frame(self, annotated: np.ndarray) -> None:
        """Encode frame thành JPEG và đẩy vào queue (bỏ qua nếu queue đầy)."""
        if not _STREAM_ENABLED:
            return
        ret, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if ret:
            try:
                frame_queue.put_nowait(buf.tobytes())
            except Exception:
                pass  # Queue đầy → bỏ frame này

    @staticmethod
    def _push_alert(alert_payload: dict) -> None:
        """Đẩy alert JSON vào queue."""
        if not _STREAM_ENABLED:
            return
        try:
            alert_queue.put_nowait(alert_payload)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------
    def _validate_paths(self) -> None:
        # Chỉ kiểm tra model; nguồn video có thể là camera (int) hoặc camera IP (RTSP/HTTP) nên không check tồn tại file cục bộ
        if not os.path.exists(self.cfg.MODEL_PATH):
            raise FileNotFoundError(f"❌ Không tìm thấy model tại {self.cfg.MODEL_PATH}")
        if isinstance(self.cfg.VIDEO_SOURCE, str) and not (self.cfg.VIDEO_SOURCE.startswith("rtsp") or self.cfg.VIDEO_SOURCE.startswith("http")):
            if not os.path.exists(self.cfg.VIDEO_SOURCE):
                raise FileNotFoundError(f"❌ Không tìm thấy video tại {self.cfg.VIDEO_SOURCE}")

    def _init_writer(self, cap) -> tuple[cv2.VideoWriter, float]:
        w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        writer = cv2.VideoWriter(
            self.cfg.LOCAL_VIDEO_RAW,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (w, h),
        )
        return writer, fps

    # ------------------------------------------------------------------
    # Per-frame detection parsing
    # ------------------------------------------------------------------
    def _parse_detections(
            self, results, frame_idx: int
    ) -> tuple[dict[int, tuple[int, int]], dict[int, tuple[int, int]]]:
        current_persons: dict[int, tuple[int, int]] = {}
        current_trashes: dict[int, tuple[int, int]] = {}
        boxes = results[0].boxes
        if boxes is None or boxes.id is None:
            return current_persons, current_trashes
        for box, obj_id, cls in zip(
                boxes.xyxy.cpu().numpy(),
                boxes.id.cpu().numpy(),
                boxes.cls.cpu().numpy(),
        ):
            cx  = int((box[0] + box[2]) / 2)
            cy  = int((box[1] + box[3]) / 2)
            oid = int(obj_id)
            if int(cls) == 0:   # person
                current_persons[oid] = (cx, cy)
                if oid not in self._person_history:
                    self._person_history[oid] = deque(maxlen=self.cfg.HISTORY_FRAMES)
                self._person_history[oid].append((cx, cy, frame_idx))
            elif int(cls) == 1:  # trash
                current_trashes[oid] = (cx, cy)
        return current_persons, current_trashes

    # ------------------------------------------------------------------
    # Trash processing
    # ------------------------------------------------------------------
    def _process_trashes(
            self,
            current_trashes: dict,
            current_persons: dict,
            mog2_alerts:     list,
            annotated:       np.ndarray,
            frame_idx:       int,
    ) -> None:
        for t_id, t_center in current_trashes.items():
            if t_id not in self._trash_registry:
                self._register_new_trash(t_id, t_center, current_persons, mog2_alerts, frame_idx)
                continue
            data = self._trash_registry[t_id]
            data["last_seen_frame"] = frame_idx
            if data["status"] == "confirmed":
                continue
            self._update_trash_state(data, t_center, current_persons, mog2_alerts, frame_idx)
            self._try_confirm(t_id, t_center, data, annotated, current_persons, frame_idx)

    def _register_new_trash(self, t_id, t_center, current_persons, mog2_alerts, frame_idx):
        owner_id, score, is_ambig = self._scorer.find_best_owner(
            t_center, current_persons, frame_idx,
            self._person_history, self._flow_tracker.flow_vecs, mog2_alerts,
        )
        self._trash_registry[t_id] = {
            "owner_id"                   : owner_id,
            "score"                      : score,
            "is_ambiguous"               : is_ambig,
            "spawn_frame"                : frame_idx,
            "spawn_time"                 : time.strftime("%H:%M:%S"),
            "confirm_ctr"                : 1,
            "stationary_ctr"             : 0,
            "stationary_after_owner_gone": 0,
            "owner_left_frame"           : None,
            "last_pos"                   : t_center,
            "last_seen_frame"            : frame_idx,
            "status"                     : "pending",
        }

    def _update_trash_state(self, data, t_center, current_persons, mog2_alerts, frame_idx):
        cfg = self.cfg
        data["confirm_ctr"] += 1
        disp              = math.hypot(t_center[0] - data["last_pos"][0], t_center[1] - data["last_pos"][1])
        is_stationary_now = disp < cfg.STATIONARY_PX
        data["stationary_ctr"] = data["stationary_ctr"] + 1 if is_stationary_now else 0
        data["last_pos"]       = t_center
        owner_id      = data["owner_id"]
        owner_in_scene = owner_id is not None and owner_id in current_persons
        if not owner_in_scene and data["owner_left_frame"] is None and owner_id:
            data["owner_left_frame"] = frame_idx
        if data["owner_left_frame"] is not None and is_stationary_now:
            data["stationary_after_owner_gone"] += 1
        if data["confirm_ctr"] <= cfg.CONFIRM_FRAMES and (
                data["owner_id"] is None
                or data["is_ambiguous"]
                or data["score"] < cfg.REEVAL_SCORE_THRESH
                or (owner_id and not owner_in_scene)
        ):
            new_owner, new_score, new_ambig = self._scorer.find_best_owner(
                t_center, current_persons, frame_idx,
                self._person_history, self._flow_tracker.flow_vecs, mog2_alerts,
            )
            if new_score > data["score"]:
                data.update({"owner_id": new_owner, "score": new_score, "is_ambiguous": new_ambig})
                if new_owner and new_owner in current_persons:
                    data["owner_left_frame"]            = None
                    data["stationary_after_owner_gone"] = 0

    def _try_confirm(self, t_id, t_center, data, annotated, current_persons, frame_idx):
        cfg = self.cfg
        frames_owner_gone = (
            frame_idx - data["owner_left_frame"]
            if data["owner_left_frame"] is not None
            else 0
        )
        owner_truly_gone = (
                data["owner_left_frame"] is not None
                and frames_owner_gone >= cfg.MIN_OWNER_GONE_FRAMES
        )
        frames_since_spawn = frame_idx - data["spawn_frame"]
        sudden_condition = (
                frames_since_spawn <= cfg.CONFIRM_FRAMES_SUDDEN + cfg.MIN_OWNER_GONE_FRAMES
                and data["confirm_ctr"] >= cfg.CONFIRM_FRAMES_SUDDEN
                and data["score"] >= cfg.MIN_SCORE
                and owner_truly_gone
        )
        stationary_condition = (
                data["confirm_ctr"] >= cfg.CONFIRM_FRAMES
                and data["stationary_after_owner_gone"] >= cfg.STATIONARY_REQUIRED
                and owner_truly_gone
        )
        proximity_confirm = (
                data["confirm_ctr"] >= cfg.CONFIRM_FRAMES
                and data["stationary_ctr"] >= cfg.STATIONARY_REQUIRED
                and data["owner_id"] is not None
                and not data["is_ambiguous"]
                and data["score"] >= cfg.MIN_SCORE
                and data["owner_id"] in current_persons
        )
        if sudden_condition:
            vtype = "Đột ngột"
        elif stationary_condition:
            vtype = "Đứng yên"
        elif proximity_confirm:
            vtype = "Bỏ rác tại chỗ"
        else:
            return

        owner_id = data["owner_id"]
        score = data["score"]

        # Gộp các hành vi của cùng 1 person_id trong 1 run: chỉ giữ lại hành vi có điểm đánh giá cao nhất
        if owner_id is not None and not data["is_ambiguous"]:
            if owner_id in self._person_highest_score:
                if score <= self._person_highest_score[owner_id]:
                    return  # Bỏ qua vì hành vi mới không rõ nét bằng hành vi đã lưu
                else:
                    self._person_highest_score[owner_id] = score
            else:
                self._person_highest_score[owner_id] = score

        self._logger.confirm_and_log(
            t_id, t_center, data, vtype, annotated, current_persons, frame_idx
        )

        # Push alert lên WebSocket
        self._push_alert({
            "type": "violation",
            "data": {
                "personId"     : data["owner_id"],
                "trashId"      : t_id,
                "violationType": vtype,
                "score"        : round(data["score"], 4),
                "timestamp"    : data["spawn_time"],
                "frame"        : frame_idx,
            },
        })

    # ------------------------------------------------------------------
    # Cleanup & HUD
    # ------------------------------------------------------------------
    def _cleanup_stale_trash(self, current_trashes: dict, frame_idx: int) -> None:
        stale = [
            tid for tid, tdata in self._trash_registry.items()
            if tid not in current_trashes
               and (frame_idx - tdata.get("last_seen_frame", tdata["spawn_frame"])) > self.cfg.STALE_FRAMES
        ]
        for tid in stale:
            del self._trash_registry[tid]

    def _draw_hud(self, annotated: np.ndarray, frame_idx: int) -> None:
        cv2.putText(
            annotated,
            f"F:{frame_idx}  VI PHAM: {len(self._logger.violation_log)}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2,
        )

    # ------------------------------------------------------------------
    # Post-processing
    # ------------------------------------------------------------------
    def _convert_h264(self) -> None:
        try:
            print("🔄 Đang thử convert sang H.264 qua FFmpeg...")
            subprocess.run(
                [
                    "ffmpeg", "-y", "-i", self.cfg.LOCAL_VIDEO_RAW,
                    "-vcodec", "libx264", "-crf", "23",
                    "-preset", "fast", "-pix_fmt", "yuv420p",
                    self.cfg.LOCAL_VIDEO_H264,
                ],
                check=True,
                capture_output=True,
            )
            print(f"✅ Convert H.264 thành công: {self.cfg.LOCAL_VIDEO_H264}")
        except Exception:
            print("⚠️  Không thể convert bằng FFmpeg. Bỏ qua bước này.")
