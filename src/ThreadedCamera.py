# ==============================================================================
# THREADED CAMERA — Luồng đọc camera phụ để triệt tiêu độ trễ OpenCV buffer
# ==============================================================================
import cv2
import time
import threading

class ThreadedCamera:
    """
    Đọc camera liên tục trong một daemon thread riêng biệt để dọn sạch
    buffer của OpenCV (vốn bị tích lũy độ trễ trên luồng RTSP/HTTP live).
    Hàm read() sẽ luôn trả về frame mới nhất ngay lập tức.
    """
    def __init__(self, source, buffer_size=1):
        self.source = source
        self.cap = cv2.VideoCapture(source)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, buffer_size)
        self.grabbed, self.frame = self.cap.read()
        self.started = False
        self.read_lock = threading.Lock()

    def start(self):
        if self.started:
            return self
        self.started = True
        self.thread = threading.Thread(target=self._update, args=(), daemon=True, name="CamReaderThread")
        self.thread.start()
        print(f"📹 [ThreadedCamera] Luồng đọc camera đã khởi động cho nguồn: {self.source}")
        return self

    def _update(self):
        while self.started:
            if not self.cap.isOpened():
                break
            grabbed, frame = self.cap.read()
            if grabbed:
                with self.read_lock:
                    self.grabbed = grabbed
                    self.frame = frame
            else:
                # Nếu mất kết nối, tạm nghỉ rồi thử lại
                time.sleep(0.01)

    def read(self) -> tuple[bool, cv2.Mat]:
        with self.read_lock:
            # Sao chép frame để tránh tranh chấp bộ nhớ giữa các luồng
            frame_copy = self.frame.copy() if self.frame is not None else None
            return self.grabbed, frame_copy

    def release(self) -> None:
        self.started = False
        if hasattr(self, "thread"):
            self.thread.join(timeout=1.0)
        self.cap.release()
        print("📹 [ThreadedCamera] Đã giải phóng camera.")

    def isOpened(self) -> bool:
        return self.cap.isOpened()

    def get(self, propId) -> float:
        """Cho phép truy cập các thuộc tính OpenCV của camera."""
        return self.cap.get(propId)
