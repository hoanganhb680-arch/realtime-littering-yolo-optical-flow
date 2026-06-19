# ==============================================================================
# CONFIG — Toàn bộ hằng số & đường dẫn tập trung tại đây
# ==============================================================================
import cv2
import os
from pathlib import Path

# Thư mục gốc dự án (DA_TGMT/) — luôn đúng dù chạy từ đâu
_ROOT = Path(__file__).parent.parent.resolve()


class Config:
    # ══════════════════════════════════════════════════════════════════════
    # 🎬  NGUỒN VIDEO — chỉ cần đổi SOURCE_MODE là xong
    #     "file"      → đọc file MP4 local
    #     "ip_camera" → kết nối camera điện thoại qua WiFi (HTTP MJPEG)
    # ══════════════════════════════════════════════════════════════════════
    SOURCE_MODE = "ip_camera"           # ← ĐỔI Ở ĐÂY: "file" | "ip_camera"

    # --- Cấu hình điều chỉnh tốc độ & tối ưu ---
    FILE_MODE_FPS = 6                   # Giới hạn FPS khi chạy file MP4 (5-7 FPS) để hiển thị đều & giảm tải CPU
    FILE_FRAME_STRIDE = 0               # 0 = tự tính theo FPS gốc; >0 = lấy mỗi N frame

    # --- Chế độ 1: File video MP4 ---
    VIDEO_FILE   = str(_ROOT / "video" / "di_bo_17.mp4")
    # --- Chế độ 2: IP Camera từ điện thoại (qua WiFi cùng mạng) ---
    # App Android gợi ý: "IP Webcam" (com.pas.webcam) trên CH Play
    #   → Mở app → cuộn xuống → "Start server" → xem IP hiển thị
    # ┌─────────────────────────────────────────────────────────┐
    # │  Giao thức     URL mẫu (dùng test_camera.py để kiểm)   │
    # │  "mjpeg"  →  http://IP:8080/video       (mặc định)     │
    # │  "rtsp"   →  rtsp://IP:8080/h264_ulaw.sdp (ổn định hơn)│
    # └─────────────────────────────────────────────────────────┘
    IP_CAM_HOST     = "192.168.0.108"# ← IP điện thoại (xem trong app)
    IP_CAM_PORT     = 8080             # Port mặc định IP Webcam
    IP_CAM_PROTOCOL = "mjpeg"          # "mjpeg" hoặc "rtsp"
    # Xác thực (để trống nếu app không đặt password)
    IP_CAM_USER  = ""
    IP_CAM_PASS  = ""
    # Buffer size — camera live dùng 1 để giảm độ trễ
    CAMERA_BUFFER = 1
    LIVE_FORCE_PORTRAIT = True         # IP Webcam is set to portrait; keep YOLO/FE frames vertical
    LIVE_ROTATE = "clockwise"          # Fallback only if IP Webcam still emits a landscape frame
    LIVE_PROCESS_MAX_SIDE = 960        # Portrait stream is capped to <= 960px on the long side for YOLO
    LIVE_TARGET_FPS = 6.0              # CPU YOLO is stable around 5-6 FPS
    LIVE_YOLO_IMGSZ = 512              # Main full-frame pass keeps CPU latency acceptable
    LIVE_YOLO_CONF = 0.12              # Slightly lower than default for hand-held trash
    LIVE_YOLO_IOU = 0.50
    YOLO_DEVICE = "auto"               # "auto" | "cpu" | "0" ...
    YOLO_HALF = True                   # Half precision only when CUDA is available
    STREAM_MAX_HEIGHT = 720            # Portrait preview sent to FE, capped to reduce browser decode lag
    STREAM_JPEG_QUALITY = 50           # Match IP Webcam quality 50 profile
    STREAM_TARGET_FPS = 6.0
    EVIDENCE_JPEG_QUALITY = 65
    MODEL_PATH       = str(_ROOT / "weights" / "best.pt")
    # --- Đường dẫn & API ---
    OUTPUT_DIR       = str(_ROOT / "violations")
    LOCAL_VIDEO_RAW  = str(_ROOT / "output_raw.mp4")
    FASTAPI_URL      = "http://127.0.0.1:8000/api/v1/violations"
    ENABLE_API_SYNC  = True
    # --- Tracking & Xác nhận vi phạm ---
    HISTORY_FRAMES        = 90
    CONFIRM_FRAMES        = 12
    CONFIRM_FRAMES_SUDDEN = 8
    OWNER_REEVAL_FRAMES   = 90
    STATIONARY_PX         = 16
    STATIONARY_REQUIRED   = 10
    SPAWN_RADIUS          = 320
    TRAJECTORY_RADIUS     = 380
    MIN_SCORE             = 0.12
    AMBIGUOUS_MARGIN      = 0.12
    STALE_FRAMES          = 90
    REEVAL_SCORE_THRESH   = 0.25
    MIN_OWNER_GONE_FRAMES = 6
    MIN_OWNER_SEEN_FRAMES = 2
    VIOLATION_GROUND_MIN_Y_RATIO = 0.58
    LIVE_PERSON_CONF = 0.14
    LIVE_PERSON_MIN_HEIGHT_RATIO = 0.12
    LIVE_PERSON_MIN_AREA_RATIO = 0.012
    MIN_OWNER_MOTION_PX = 5
    PERSON_ID_MATCH_RADIUS = 180
    RECENT_OWNER_GRACE_FRAMES = 90
    RECENT_OWNER_RADIUS = 520
    TRASH_ID_MATCH_RADIUS = 90
    DRAW_STALE_TRASH_FRAMES = 18
    LOST_TRASH_RECOVERY_FRAMES = 60
    LOST_TRASH_MIN_SEEN = 2
    LOST_TRASH_FOOT_SNAP_RADIUS = 220
    LOST_TRASH_GROUND_MIN_Y_RATIO = 0.58
    # --- MOG2 Background Subtraction ---
    MOG2_HISTORY      = 200
    MOG2_THRESHOLD    = 40
    MOG2_MIN_AREA     = 300
    MOG2_BOOST_RADIUS = 50
    SAVE_OWNERLESS_CANDIDATES = True
    OWNERLESS_CANDIDATE_COOLDOWN_FRAMES = 60
    PENDING_LOG_INTERVAL_FRAMES = 30
    # --- Optical Flow ---
    FLOW_HISTORY_FRAMES = 8
    LK_PARAMS = dict(
        winSize  = (15, 15),
        maxLevel = 2,
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
    )
    # ──────────────────────────────────────────────────────────────────────
    # Tự động build VIDEO_SOURCE từ SOURCE_MODE khi khởi tạo
    # ──────────────────────────────────────────────────────────────────────
    def __init__(self):
        self.SOURCE_MODE = os.getenv("DETECTOR_SOURCE_MODE", self.SOURCE_MODE).strip().lower()
        self.VIDEO_FILE = os.getenv("DETECTOR_VIDEO_FILE", self.VIDEO_FILE)
        self.IP_CAM_HOST = os.getenv("DETECTOR_IP_CAM_HOST", self.IP_CAM_HOST)
        self.IP_CAM_PORT = int(os.getenv("DETECTOR_IP_CAM_PORT", self.IP_CAM_PORT))
        self.IP_CAM_PROTOCOL = os.getenv("DETECTOR_IP_CAM_PROTOCOL", self.IP_CAM_PROTOCOL)

        if self.SOURCE_MODE == "file":
            self.VIDEO_SOURCE = self.VIDEO_FILE
            self.IS_LIVE      = False
        elif self.SOURCE_MODE == "ip_camera":
            # Build URL có xác thực nếu cần
            auth = ""
            if self.IP_CAM_USER:
                auth = f"{self.IP_CAM_USER}:{self.IP_CAM_PASS}@"
            protocol = str(self.IP_CAM_PROTOCOL).lower()
            if protocol == "rtsp":
                self.VIDEO_SOURCE = (
                    f"rtsp://{auth}{self.IP_CAM_HOST}:{self.IP_CAM_PORT}/h264_ulaw.sdp"
                )
            else:
                self.VIDEO_SOURCE = (
                    f"http://{auth}{self.IP_CAM_HOST}:{self.IP_CAM_PORT}/video"
                )
            self.IS_LIVE = True
        else:
            raise ValueError(
                f"SOURCE_MODE không hợp lệ: '{self.SOURCE_MODE}'. "
                f"Chọn 'file' hoặc 'ip_camera'."
            )
        print(f"📹 Nguồn video  : [{self.SOURCE_MODE.upper()}] {self.VIDEO_SOURCE}")
        print(f"🔴 Chế độ live  : {self.IS_LIVE}")
class Settings:
    # --- FastAPI ---
    APP_TITLE = "Hệ thống giám sát vứt rác - Backend API"
    HOST      = "127.0.0.1"
    PORT      = 8000
    # --- MinIO ---
    MINIO_URL        = "127.0.0.1:9000"
    MINIO_ACCESS_KEY = "minioadmin"
    MINIO_SECRET_KEY = "minioadmin"
    MINIO_SECURE     = False
    BUCKET_NAME      = "violations"
settings = Settings()
