# ==============================================================================
# TRASH VIOLATION DETECTOR
# ==============================================================================
import cv2
import time
from collections import deque
from ultralytics import YOLO

from Config import Config
from MotionDetector import MotionDetector
from OpticalFlowTracker import OpticalFlowTracker
from OwnershipScorer import OwnershipScorer
from ViolationLogger import ViolationLogger
from ApiSyncer import ApiSyncer
from detector_io import DetectorIoMixin
from detection_parsing import DetectionParsingMixin
from trash_lifecycle import TrashLifecycleMixin


class TrashViolationDetector(DetectorIoMixin, DetectionParsingMixin, TrashLifecycleMixin):
    """Coordinate camera/video input, detection, ownership scoring, and logging."""

    # Khởi tạo các module con và bộ nhớ theo dõi người/rác.
    def __init__(self, cfg=None):
        self.cfg = cfg or Config()
        self._motion_detector = MotionDetector()
        self._flow_tracker = OpticalFlowTracker()
        self._scorer = OwnershipScorer()
        self._logger = ViolationLogger(syncer=ApiSyncer())

        self._person_history: dict[int, deque] = {}
        self._person_seen_counts: dict[int, int] = {}
        self._person_frame_jpg: dict[int, bytes] = {}
        self._trash_registry: dict[int, dict] = {}
        self._person_highest_score: dict[int, float] = {}

        self.stopped = False
        self._last_live_process_at = 0.0
        self._last_stream_push_at = 0.0
        self._next_synthetic_person_id = 200000
        self._next_synthetic_trash_id = 100000
        self._last_ownerless_candidate_frame = -10**9
        self._last_pending_reason_log: dict[int, int] = {}
        self._last_camera_wait_log_at = 0.0

    # Vòng lặp chính: load YOLO, đọc frame, xử lý và xuất kết quả.
    def run(self) -> None:
        cfg = self.cfg
        self._validate_paths()
        is_live = cfg.IS_LIVE

        print(f"[DETECTOR] Starting model, source={cfg.VIDEO_SOURCE}")
        model = YOLO(cfg.MODEL_PATH)
        track_kwargs = self._build_track_kwargs(is_live)

        cap = self._open_capture(is_live)
        out_vid, source_fps = self._init_writer(cap)
        file_frame_step = self._file_frame_step(source_fps, is_live)
        prev_gray, frame_idx, raw_frame_idx = None, 0, 0
        last_live_seq = -1

        while not self.stopped and (is_live or cap.isOpened()):
            if is_live:
                self._pace_live_loop()

            t_start = time.monotonic()
            ret, frame = cap.read()
            if not ret:
                if is_live:
                    self._log_camera_wait(cfg.VIDEO_SOURCE)
                    time.sleep(0.5)
                    continue
                break
            raw_frame_idx += 1

            if is_live:
                seq = getattr(cap, "seq", None)
                if seq is not None and seq == last_live_seq:
                    time.sleep(0.005)
                    continue
                last_live_seq = seq
            elif self._skip_file_frame(raw_frame_idx, file_frame_step):
                continue

            frame = self._prepare_frame(frame)
            frame_idx += 1
            curr_gray, annotated = self._process_frame(
                model, frame, frame_idx, prev_gray, track_kwargs
            )

            if out_vid is not None:
                out_vid.write(annotated)
            self._push_frame(annotated)
            prev_gray = curr_gray.copy()

            self._pace_file_mode(is_live, t_start)
            if frame_idx % 50 == 0:
                print(f"  [{frame_idx} frames] processed")

        self._finish_run(cap, out_vid, is_live)

    # Mở nguồn đầu vào: camera live hoặc file video.
    def _open_capture(self, is_live: bool):
        if is_live:
            from ThreadedCamera import ThreadedCamera
            cap = ThreadedCamera(self.cfg.VIDEO_SOURCE, self.cfg.CAMERA_BUFFER).start()
        else:
            cap = cv2.VideoCapture(self.cfg.VIDEO_SOURCE)

        if is_live and not cap.isOpened():
            print(f"[LIVE] Camera is not ready; detector will keep reconnecting: {self.cfg.VIDEO_SOURCE}")
        if not is_live and not cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {self.cfg.VIDEO_SOURCE}")
        return cap

    # Tính bước nhảy frame để file mode chạy gần FPS mục tiêu.
    def _file_frame_step(self, source_fps: float, is_live: bool) -> int:
        if is_live:
            return 1
        explicit_stride = int(getattr(self.cfg, "FILE_FRAME_STRIDE", 0) or 0)
        if explicit_stride > 0:
            return explicit_stride
        target_fps = float(getattr(self.cfg, "FILE_MODE_FPS", source_fps) or source_fps)
        if source_fps <= 0 or target_fps <= 0:
            return 1
        return max(1, round(source_fps / target_fps))

    # Kiểm tra frame file hiện tại có cần bỏ qua để giảm tải không.
    @staticmethod
    def _skip_file_frame(raw_frame_idx: int, file_frame_step: int) -> bool:
        return (raw_frame_idx - 1) % max(1, file_frame_step) != 0

    # Xử lý một frame: MOG2, YOLO+ByteTrack, parse, flow và lifecycle.
    def _process_frame(
            self,
            model,
            frame,
            frame_idx: int,
            prev_gray,
            track_kwargs: dict,
    ):
        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mog2_alerts = self._motion_detector.get_alerts(frame)
        results = model.track(frame, **track_kwargs)
        annotated = results[0].plot()
        current_persons, current_trashes = self._parse_detections(results, frame_idx)

        self._remember_person_evidence(current_persons, annotated)

        self._flow_tracker.update(prev_gray, curr_gray, current_persons)
        self._process_trashes(current_trashes, current_persons, mog2_alerts, annotated, frame_idx)
        self._cleanup_stale_trash(current_trashes, frame_idx)
        self._draw_sticky_trash(annotated, current_trashes, frame_idx)
        self._draw_hud(annotated, frame_idx)
        return curr_gray, annotated

    # Giữ nhịp file mode theo FILE_MODE_FPS để xem output ổn định.
    def _pace_file_mode(self, is_live: bool, started_at: float) -> None:
        if is_live:
            return
        wait_time = (1.0 / self.cfg.FILE_MODE_FPS) - (time.monotonic() - started_at)
        if wait_time > 0:
            time.sleep(wait_time)

    # Dọn tài nguyên và gửi thông báo khi video xử lý xong.
    def _finish_run(self, cap, out_vid, is_live: bool) -> None:
        cap.release()
        if out_vid is not None:
            out_vid.release()
            print(f"\nDone. Raw video saved to: {self.cfg.LOCAL_VIDEO_RAW}")
        else:
            print("\nLive stream stopped. Output recording was disabled.")

        if not is_live:
            self._push_alert({
                "type": "video_ended",
                "data": {
                    "message": "Video \u0111\u00e3 x\u1eed l\u00fd xong!",
                    "video_raw": self.cfg.LOCAL_VIDEO_RAW,
                    "total_violations": len(self._logger.violation_log),
                },
            })
