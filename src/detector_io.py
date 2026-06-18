# ==============================================================================
# DETECTOR IO / RUNTIME HELPERS
# ==============================================================================
import cv2
import os
import subprocess
import time
import numpy as np

# Import stream queues de push frame + alert len WebSocket
try:
    from stream_router import frame_queue, alert_queue, has_stream_clients
    _STREAM_ENABLED = True
except ImportError:
    _STREAM_ENABLED = False
    has_stream_clients = None


class DetectorIoMixin:
    """Camera/video runtime, streaming, evidence buffering, and output helpers."""
    def _build_track_kwargs(self, is_live: bool) -> dict:
        cfg = self.cfg
        kwargs = {
            "imgsz": cfg.LIVE_YOLO_IMGSZ if is_live else 640,
            "conf": cfg.LIVE_YOLO_CONF if is_live else 0.25,
            "iou": cfg.LIVE_YOLO_IOU,
            "persist": True,
            "tracker": "bytetrack.yaml",
            "verbose": False,
        }
        device, half = self._resolve_yolo_runtime()
        if device is not None:
            kwargs["device"] = device
        if half:
            kwargs["half"] = True
        return kwargs

    def _build_floor_trash_kwargs(self, is_live: bool) -> dict | None:
        if not is_live or not getattr(self.cfg, "LIVE_FLOOR_TRASH_PASS", False):
            return None
        kwargs = {
            "imgsz": getattr(self.cfg, "LIVE_FLOOR_YOLO_IMGSZ", 640),
            "conf": getattr(self.cfg, "LIVE_FLOOR_TRASH_CONF", 0.05),
            "iou": self.cfg.LIVE_YOLO_IOU,
            "classes": [1],
            "verbose": False,
        }
        device, half = self._resolve_yolo_runtime()
        if device is not None:
            kwargs["device"] = device
        if half:
            kwargs["half"] = True
        return kwargs

    def _resolve_yolo_runtime(self) -> tuple[str | int | None, bool]:
        device_cfg = str(getattr(self.cfg, "YOLO_DEVICE", "auto")).strip().lower()
        half_cfg = bool(getattr(self.cfg, "YOLO_HALF", True))
        if device_cfg != "auto":
            return device_cfg, half_cfg and device_cfg != "cpu"
        try:
            import torch
            if torch.cuda.is_available():
                return 0, half_cfg
        except Exception:
            pass
        return "cpu", False

    def _pace_live_loop(self) -> None:
        target_fps = float(getattr(self.cfg, "LIVE_TARGET_FPS", 0) or 0)
        if target_fps <= 0:
            return
        now = time.monotonic()
        min_interval = 1.0 / target_fps
        elapsed = now - self._last_live_process_at
        if self._last_live_process_at > 0 and elapsed < min_interval:
            time.sleep(min_interval - elapsed)
            now = time.monotonic()
        self._last_live_process_at = now

    def _log_camera_wait(self, source) -> None:
        now = time.monotonic()
        if now - self._last_camera_wait_log_at < 5.0:
            return
        self._last_camera_wait_log_at = now
        print(f"[LIVE] Waiting for camera/reconnect: {source}", flush=True)

    def _prepare_frame(self, frame: np.ndarray) -> np.ndarray:
        frame = self._orient_frame(frame)
        if self.cfg.IS_LIVE:
            frame = self._resize_for_processing(frame)
        return frame

    def _resize_for_processing(self, frame: np.ndarray) -> np.ndarray:
        max_side = int(getattr(self.cfg, "LIVE_PROCESS_MAX_SIDE", 0) or 0)
        if max_side <= 0:
            return frame
        h, w = frame.shape[:2]
        side = max(h, w)
        if side <= max_side:
            return frame
        scale = max_side / side
        new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
        return cv2.resize(frame, new_size, interpolation=cv2.INTER_AREA)

    # ------------------------------------------------------------------
    # WebSocket stream helpers
    # ------------------------------------------------------------------

    def _push_frame(self, annotated: np.ndarray) -> None:
        """Encode frame thành JPEG và đẩy vào queue (bỏ qua nếu queue đầy)."""
        if not _STREAM_ENABLED:
            return
        if has_stream_clients is not None and not has_stream_clients():
            return
        target_fps = float(getattr(self.cfg, "STREAM_TARGET_FPS", 0) or 0)
        now = time.monotonic()
        if target_fps > 0 and self._last_stream_push_at > 0:
            if now - self._last_stream_push_at < (1.0 / target_fps):
                return
        self._last_stream_push_at = now
        stream_frame = self._resize_for_stream(annotated)
        quality = getattr(self.cfg, "STREAM_JPEG_QUALITY", 70)
        ret, buf = cv2.imencode(".jpg", stream_frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if ret:
            try:
                while not frame_queue.empty():
                    frame_queue.get_nowait()
                frame_queue.put_nowait(buf.tobytes())
            except Exception:
                pass  # Queue đầy → bỏ frame này

    def _resize_for_stream(self, frame: np.ndarray) -> np.ndarray:
        max_h = getattr(self.cfg, "STREAM_MAX_HEIGHT", 0)
        if not max_h or frame.shape[0] <= max_h:
            return frame
        scale = max_h / frame.shape[0]
        new_w = max(1, int(frame.shape[1] * scale))
        return cv2.resize(frame, (new_w, max_h), interpolation=cv2.INTER_AREA)

    def _remember_evidence_frame(self, annotated: np.ndarray) -> None:
        clip_seconds = float(getattr(self.cfg, "EVIDENCE_CLIP_SECONDS", 0) or 0)
        if clip_seconds <= 0:
            return
        frame = self._resize_for_evidence_clip(annotated)
        quality = int(getattr(self.cfg, "EVIDENCE_CLIP_JPEG_QUALITY", 65))
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if not ok:
            return
        now = time.monotonic()
        self._evidence_clip_buffer.append((now, buf.tobytes()))
        self._trim_evidence_clip_buffer(now)

    def _get_evidence_clip_frames(self, annotated: np.ndarray) -> list[tuple[float, bytes]]:
        now = time.monotonic()
        self._trim_evidence_clip_buffer(now)
        frames = list(self._evidence_clip_buffer)
        frame = self._resize_for_evidence_clip(annotated)
        quality = int(getattr(self.cfg, "EVIDENCE_CLIP_JPEG_QUALITY", 65))
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if ok:
            frames.append((now, buf.tobytes()))
        return frames

    def _trim_evidence_clip_buffer(self, now: float) -> None:
        clip_seconds = float(getattr(self.cfg, "EVIDENCE_CLIP_SECONDS", 45))
        max_fps = float(getattr(self.cfg, "EVIDENCE_CLIP_MAX_FPS", 6) or 6)
        max_frames = max(1, int(clip_seconds * max_fps) + 2)
        cutoff = now - clip_seconds
        while self._evidence_clip_buffer and (
            self._evidence_clip_buffer[0][0] < cutoff
            or len(self._evidence_clip_buffer) > max_frames
        ):
            self._evidence_clip_buffer.popleft()

    def _resize_for_evidence_clip(self, frame: np.ndarray) -> np.ndarray:
        max_h = int(getattr(self.cfg, "EVIDENCE_CLIP_MAX_HEIGHT", 0) or 0)
        if max_h <= 0 or frame.shape[0] <= max_h:
            return frame
        scale = max_h / frame.shape[0]
        new_w = max(1, int(frame.shape[1] * scale))
        return cv2.resize(frame, (new_w, max_h), interpolation=cv2.INTER_AREA)

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

    def _init_writer(self, cap) -> tuple[cv2.VideoWriter | None, float]:
        w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        if self.cfg.IS_LIVE and not getattr(self.cfg, "LIVE_SAVE_OUTPUT", False):
            return None, fps
        writer_fps = fps
        if not self.cfg.IS_LIVE:
            writer_fps = float(getattr(self.cfg, "FILE_MODE_FPS", fps) or fps)
        if self._should_swap_live_size(w, h):
            w, h = h, w
        if self.cfg.IS_LIVE:
            w, h = self._scaled_dimensions(w, h)
        writer = cv2.VideoWriter(
            self.cfg.LOCAL_VIDEO_RAW,
            cv2.VideoWriter_fourcc(*"mp4v"),
            writer_fps,
            (w, h),
        )
        return writer, fps

    def _scaled_dimensions(self, w: int, h: int) -> tuple[int, int]:
        max_side = int(getattr(self.cfg, "LIVE_PROCESS_MAX_SIDE", 0) or 0)
        if max_side <= 0:
            return w, h
        side = max(w, h)
        if side <= max_side:
            return w, h
        scale = max_side / side
        return max(1, int(w * scale)), max(1, int(h * scale))

    def _orient_frame(self, frame: np.ndarray) -> np.ndarray:
        if not self.cfg.IS_LIVE:
            return frame
        if getattr(self.cfg, "LIVE_FORCE_PORTRAIT", False):
            h, w = frame.shape[:2]
            if h >= w:
                return frame
            rotate = getattr(self.cfg, "LIVE_ROTATE", "clockwise")
            if rotate == "counterclockwise":
                return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
            return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        rotate = getattr(self.cfg, "LIVE_ROTATE", "none")
        if rotate == "clockwise":
            return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        if rotate == "counterclockwise":
            return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        if rotate == "180":
            return cv2.rotate(frame, cv2.ROTATE_180)
        return frame

    def _should_swap_live_size(self, w: int, h: int) -> bool:
        if not self.cfg.IS_LIVE:
            return False
        if getattr(self.cfg, "LIVE_FORCE_PORTRAIT", False):
            return w > h
        return getattr(self.cfg, "LIVE_ROTATE", "none") in {"clockwise", "counterclockwise"}

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
