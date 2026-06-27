
import cv2
import os
from pathlib import Path

# Thư mục gốc dự án (DA_TGMT/) — luôn đúng dù chạy từ đâu
_ROOT = Path(__file__).parent.parent.resolve()


def _load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_env_file(_ROOT / ".env")


class Config:
  

    SOURCE_MODE = "file"               

    #  Cấu hình điều chỉnh tốc độ & tối ưu 
    FILE_MODE_FPS = 6
    FILE_FRAME_STRIDE = 0               # 0 = tự tính theo FPS gốc; >0 = lấy mỗi N frame

    #  video MP4
    VIDEO_FILE   = str(_ROOT / "video" / "di_bo_17.mp4")
    # IP Camera 
    
    IP_CAM_HOST     = "192.168.0.103"
    IP_CAM_PORT     = 8080             
    IP_CAM_PROTOCOL = "mjpeg"          # "mjpeg" hoặc "rtsp"
    # Xác thực 
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
    HISTORY_FRAMES        = 90    # Lưu lịch sử vị trí người trong 90 frame (~15s ở 6 FPS) để xét ai từng đi gần rác.
    CONFIRM_FRAMES        = 12    # Rác phải tồn tại đủ 12 frame (~2s) trước khi xét xác nhận vi phạm thường.
    CONFIRM_FRAMES_SUDDEN = 8     # Ngưỡng ngắn hơn cho hành vi bỏ rác nhanh/rời đi đột ngột (~1.33s).
    OWNER_REEVAL_FRAMES   = 90    # Cho phép tính lại owner trong 90 frame đầu (~15s) nếu điểm ban đầu chưa chắc.
    STATIONARY_PX         = 16    # Nếu tâm rác lệch dưới 16 px giữa 2 lần cập nhật thì coi như vẫn đứng yên.
    STATIONARY_REQUIRED   = 10    # Rác cần đứng yên đủ 10 frame (~1.67s) để tránh nhầm vật đang di chuyển.
    SPAWN_RADIUS          = 320   # Khi rác mới xuất hiện, ưu tiên xét người gần rác trong bán kính 320 px.
    TRAJECTORY_RADIUS     = 380   # Bán kính kiểm tra lịch sử quỹ đạo người từng đi gần vị trí rác.
    MIN_SCORE             = 0.12  # Điểm owner tối thiểu; thấp hơn ngưỡng này thì chưa đủ tin để gán người vi phạm.
    AMBIGUOUS_MARGIN      = 0.12  # Nếu điểm top 1 và top 2 chênh dưới 0.12 thì coi là mơ hồ, chưa chắc ai là chủ.
    STALE_FRAMES          = 90    # Xóa rác khỏi bộ nhớ nếu mất track quá 90 frame (~15s).
    REEVAL_SCORE_THRESH   = 0.25  # Nếu score dưới 0.25 thì owner chưa chắc, có thể tính lại owner.
    MIN_OWNER_GONE_FRAMES = 6     # Owner phải rời khung hình ít nhất 6 frame (~1s) mới coi là đã rời đi thật.
    MIN_OWNER_SEEN_FRAMES = 2     # Người phải xuất hiện ít nhất 2 frame để tránh nhận diện nhầm chỉ trong 1 frame.
    VIOLATION_GROUND_MIN_Y_RATIO = 0.58  # Chỉ xác nhận rác ở vùng thấp của ảnh, gần mặt đất.
    GROUND_OWNER_FOOT_MAX_DISTANCE = 180  # Fallback cho góc camera cao: rác phải gần quỹ đạo chân owner.
    GROUND_OWNER_FOOT_MAX_VERTICAL_RATIO = 0.17  # Chênh lệch dọc tối đa giữa rác và chân owner theo chiều cao ảnh.
    LIVE_PERSON_CONF = 0.14       # Ngưỡng confidence người khi realtime; thấp để không bỏ sót trong cảnh mờ/rung.
    LIVE_PERSON_MIN_HEIGHT_RATIO = 0.12  # Bỏ bbox người quá thấp so với chiều cao ảnh vì thường là nhận diện sai.
    LIVE_PERSON_MIN_AREA_RATIO = 0.012   # Bỏ bbox người quá nhỏ so với frame để lọc nhiễu ở xa/nền.
    MIN_OWNER_MOTION_PX = 5       # Owner cần dịch chuyển tối thiểu 5 px để chứng minh có chuyển động thật.
    PERSON_ID_MATCH_RADIUS = 180  # Ghép lại ID người giữa các frame nếu điểm neo gần nhau trong 180 px.
    RECENT_OWNER_GRACE_FRAMES = 90  # Vẫn xét người vừa rời khung hình trong 90 frame gần nhất (~15s).
    RECENT_OWNER_RADIUS = 520     # Bán kính fallback lớn hơn để tìm owner gần đây khi tracking chính chưa chắc.
    TRASH_ID_MATCH_RADIUS = 90    # Ghép ID rác giữa các frame nếu vị trí rác gần nhau trong 90 px.
    DRAW_STALE_TRASH_FRAMES = 18  # Vẫn vẽ rác đã mất track tối đa 18 frame (~3s) để hình ảnh đỡ nhấp nháy.
    LOST_TRASH_RECOVERY_FRAMES = 60  # Cho phép phục hồi rác bị mất track trong 60 frame (~10s).
    LOST_TRASH_MIN_SEEN = 2       # Chỉ phục hồi rác từng được thấy ít nhất 2 frame để tránh phục hồi nhiễu.
    LOST_TRASH_FOOT_SNAP_RADIUS = 220  # Khi mất rác, tìm lại quanh vùng gần chân owner trong bán kính 220 px.
    LOST_TRASH_GROUND_MIN_Y_RATIO = 0.58  # Chỉ phục hồi rác ở vùng thấp gần mặt đất.

    # --- MOG2 Background Subtraction ---
    MOG2_HISTORY      = 200  # Số frame MOG2 dùng để học nền (~33s ở 6 FPS), giúp nền ổn định hơn.
    MOG2_THRESHOLD    = 40   # Ngưỡng khác biệt pixel so với nền; càng thấp càng nhạy nhưng dễ nhiễu.
    MOG2_MIN_AREA     = 300  # Bỏ vùng chuyển động nhỏ hơn 300 px để lọc nhiễu ánh sáng/bóng nhỏ.
    MOG2_BOOST_RADIUS = 50   # Nếu có chuyển động gần rác trong bán kính 50 px thì tăng nhẹ điểm owner.
    SAVE_OWNERLESS_CANDIDATES = True  # Lưu ảnh candidate khi thấy rác nhưng chưa xác định được owner để debug.
    OWNERLESS_CANDIDATE_COOLDOWN_FRAMES = 60  # Giới hạn lưu candidate mỗi 60 frame (~10s), tránh spam ảnh.
    PENDING_LOG_INTERVAL_FRAMES = 30  # In log pending mỗi 30 frame (~5s), tránh log quá dày.

    # --- Optical Flow ---
    FLOW_HISTORY_FRAMES = 8  # Lưu 8 vector optical flow gần nhất (~1.33s) để lấy hướng di chuyển ngắn hạn.
    LK_PARAMS = dict(
        winSize  = (15, 15),  # Cửa sổ tìm kiếm 15x15 quanh điểm cũ; phù hợp chuyển động nhỏ-vừa giữa 2 frame.
        maxLevel = 2,         # Dùng 2 tầng pyramid để theo dõi được chuyển động lớn hơn.
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),  # Dừng sau 10 vòng lặp hoặc sai số < 0.03.
    )
   
    #  build VIDEO_SOURCE từ SOURCE_MODE khi khởi tạo

    def __init__(self):
        self.SOURCE_MODE = os.getenv("DETECTOR_SOURCE_MODE", self.SOURCE_MODE).strip().lower()
        self.VIDEO_FILE = os.getenv("DETECTOR_VIDEO_FILE", self.VIDEO_FILE)
        self.MODEL_PATH = os.getenv("DETECTOR_MODEL_PATH", self.MODEL_PATH)
        self.OUTPUT_DIR = os.getenv("DETECTOR_OUTPUT_DIR", self.OUTPUT_DIR)
        self.IP_CAM_HOST = os.getenv("DETECTOR_IP_CAM_HOST", self.IP_CAM_HOST)
        self.IP_CAM_PORT = int(os.getenv("DETECTOR_IP_CAM_PORT", self.IP_CAM_PORT))
        self.IP_CAM_PROTOCOL = os.getenv("DETECTOR_IP_CAM_PROTOCOL", self.IP_CAM_PROTOCOL)
        self.IP_CAM_USER = os.getenv("DETECTOR_IP_CAM_USER", self.IP_CAM_USER)
        self.IP_CAM_PASS = os.getenv("DETECTOR_IP_CAM_PASS", self.IP_CAM_PASS)

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
    HOST      = os.getenv("APP_HOST", "127.0.0.1")
    PORT      = int(os.getenv("APP_PORT", "8000"))
    # --- MinIO ---
    MINIO_URL        = os.getenv("MINIO_URL", "127.0.0.1:9000")
    MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
    MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minioadmin")
    MINIO_SECURE     = os.getenv("MINIO_SECURE", "false").lower() == "true"
    BUCKET_NAME      = os.getenv("MINIO_BUCKET_NAME", "violations")
settings = Settings()
