import os
import threading
import time
from typing import Any

import cv2


class ThreadedCamera:
    """
    Continuously drains a live camera stream in a daemon thread.

    The detector can then sample the freshest decoded frame instead of waiting
    behind OpenCV/FFMPEG network buffers.
    """

    def __init__(self, source: Any, buffer_size: int = 1):
        self.source = source
        self.buffer_size = buffer_size
        self.cap = None
        self.grabbed = False
        self.frame = None
        self.frame_seq = 0
        self.last_frame_at = 0.0
        self.started = False
        self.read_lock = threading.Lock()

        self.cap = self._open_capture()
        self._read_initial_frame()

    def start(self):
        if self.started:
            return self
        self.started = True
        self.thread = threading.Thread(target=self._update, daemon=True, name="CamReaderThread")
        self.thread.start()
        print(f"[ThreadedCamera] Started low-latency reader for: {self.source}")
        return self

    def _open_capture(self):
        if isinstance(self.source, str) and self.source.startswith(("http://", "https://", "rtsp://")):
            os.environ.setdefault(
                "OPENCV_FFMPEG_CAPTURE_OPTIONS",
                "fflags;nobuffer|flags;low_delay|rtsp_transport;tcp|max_delay;0",
            )
            cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)
        else:
            cap = cv2.VideoCapture(self.source)

        cap.set(cv2.CAP_PROP_BUFFERSIZE, self.buffer_size)
        open_timeout = getattr(cv2, "CAP_PROP_OPEN_TIMEOUT_MSEC", None)
        read_timeout = getattr(cv2, "CAP_PROP_READ_TIMEOUT_MSEC", None)
        if open_timeout is not None:
            cap.set(open_timeout, 5000)
        if read_timeout is not None:
            cap.set(read_timeout, 5000)
        return cap

    def _read_initial_frame(self) -> None:
        if not self.cap or not self.cap.isOpened():
            return
        grabbed, frame = self.cap.read()
        if grabbed and frame is not None:
            self._store_frame(frame)

    def _update(self) -> None:
        while self.started:
            if not self.cap or not self.cap.isOpened():
                self._mark_disconnected()
                self._reconnect()
                continue

            grabbed, frame = self.cap.read()
            if grabbed and frame is not None:
                self._store_frame(frame)
            else:
                self._mark_disconnected()
                time.sleep(0.02)

    def _store_frame(self, frame) -> None:
        with self.read_lock:
            self.grabbed = True
            self.frame = frame
            self.frame_seq += 1
            self.last_frame_at = time.monotonic()

    def _mark_disconnected(self) -> None:
        with self.read_lock:
            self.grabbed = False

    def _reconnect(self) -> None:
        time.sleep(0.5)
        try:
            if self.cap:
                self.cap.release()
        except Exception:
            pass
        self.cap = self._open_capture()
        self._read_initial_frame()

    def read(self):
        with self.read_lock:
            frame_copy = self.frame.copy() if self.frame is not None else None
            return self.grabbed and frame_copy is not None, frame_copy

    @property
    def seq(self) -> int:
        with self.read_lock:
            return self.frame_seq

    def release(self) -> None:
        self.started = False
        if hasattr(self, "thread"):
            self.thread.join(timeout=1.0)
        if self.cap:
            self.cap.release()
        print("[ThreadedCamera] Released camera.")

    def isOpened(self) -> bool:
        return bool(self.cap and self.cap.isOpened())

    def get(self, prop_id) -> float:
        if not self.cap:
            return 0.0
        return self.cap.get(prop_id)
