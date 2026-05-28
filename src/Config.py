# ==============================================================================
# CONFIG — Toàn bộ hằng số & đường dẫn tập trung tại đây
# ==============================================================================
import cv2
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
    IP_CAM_HOST     = "192.168.31.115"# ← IP điện thoại (xem trong app)
    IP_CAM_PORT     = 8080             # Port mặc định IP Webcam
    IP_CAM_PROTOCOL = "mjpeg"          # "mjpeg" hoặc "rtsp"
    # Xác thực (để trống nếu app không đặt password)
    IP_CAM_USER  = ""
    IP_CAM_PASS  = ""
    # Buffer size — camera live dùng 1 để giảm độ trễ
    CAMERA_BUFFER = 1
    MODEL_PATH       = str(_ROOT / "weights" / "best.pt")
    # --- Đường dẫn & API ---
    OUTPUT_DIR       = str(_ROOT / "violations")
    LOCAL_VIDEO_RAW  = str(_ROOT / "output_raw.mp4")
    LOCAL_VIDEO_H264 = str(_ROOT / "output_h264.mp4")
    FASTAPI_URL      = "http://127.0.0.1:8000/api/v1/violations"
    ENABLE_API_SYNC  = True
    # --- Tracking & Xác nhận vi phạm ---
    HISTORY_FRAMES        = 15
    CONFIRM_FRAMES        = 8
    CONFIRM_FRAMES_SUDDEN = 4
    STATIONARY_PX         = 6
    STATIONARY_REQUIRED   = 4
    SPAWN_RADIUS          = 220
    TRAJECTORY_RADIUS     = 220
    MIN_SCORE             = 0.15
    AMBIGUOUS_MARGIN      = 0.15
    STALE_FRAMES          = 30
    REEVAL_SCORE_THRESH   = 0.30
    MIN_OWNER_GONE_FRAMES = 5
    # --- MOG2 Background Subtraction ---
    MOG2_HISTORY      = 200
    MOG2_THRESHOLD    = 40
    MOG2_MIN_AREA     = 300
    MOG2_BOOST_RADIUS = 50
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
        if self.SOURCE_MODE == "file":
            self.VIDEO_SOURCE = self.VIDEO_FILE
            self.IS_LIVE      = False
        elif self.SOURCE_MODE == "ip_camera":
            # Build URL có xác thực nếu cần
            auth = ""
            if self.IP_CAM_USER:
                auth = f"{self.IP_CAM_USER}:{self.IP_CAM_PASS}@"
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
