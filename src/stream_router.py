# ==============================================================================
# STREAM ROUTER — WebSocket endpoint để push frame JPEG + alert vi phạm
# ==============================================================================
import asyncio
import json
import os
import threading
from queue import Empty, Queue
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import FileResponse

router = APIRouter(tags=["stream"])
WS_SEND_TIMEOUT = 0.25
# ------------------------------------------------------------------
# Connection Manager — quản lý tất cả FE client đang kết nối
# ------------------------------------------------------------------
class ConnectionManager:
    def __init__(self) -> None:
        self._clients: list[WebSocket] = []
    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        self._clients.append(ws)
    def disconnect(self, ws: WebSocket) -> None:
        if ws in self._clients:
            self._clients.remove(ws)
    async def broadcast_bytes(self, data: bytes) -> None:
        """Gửi frame JPEG binary tới tất cả client."""
        dead: list[WebSocket] = []
        for ws in list(self._clients):
            try:
                await asyncio.wait_for(ws.send_bytes(data), timeout=WS_SEND_TIMEOUT)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws)
    async def broadcast_json(self, payload: dict[str, Any]) -> None:
        """Gửi JSON event (alert vi phạm mới) tới tất cả client."""
        text = json.dumps(payload, ensure_ascii=False)
        dead: list[WebSocket] = []
        for ws in list(self._clients):
            try:
                await ws.send_text(text)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws)
    @property
    def active_count(self) -> int:
        return len(self._clients)
# Singleton dùng chung toàn app
manager = ConnectionManager()


def has_stream_clients() -> bool:
    return manager.active_count > 0
# Queue để TrashViolationDetector (luồng sync) đẩy frame vào
# Main event loop sẽ drain queue này
frame_queue: Queue[bytes] = Queue(maxsize=1)
alert_queue: Queue[dict] = Queue(maxsize=50)

# ------------------------------------------------------------------
# Background Detector Thread Management
# ------------------------------------------------------------------
detector_thread: threading.Thread = None
detector_instance: Any = None

def start_detector_thread() -> bool:
    """Khởi động luồng detector chạy nền."""
    global detector_thread, detector_instance
    if detector_thread and detector_thread.is_alive():
        print("⚠️ Detector thread đang chạy rồi.")
        return False
        
    try:
        from TrashViolationDetector import TrashViolationDetector
        from Config import Config
        from violation_router import reset_session_violations
        
        # Reset lại bộ nhớ session các vi phạm ở backend
        reset_session_violations()
        
        detector_instance = TrashViolationDetector(cfg=Config())
        
        def _run():
            try:
                detector_instance.run()
            except Exception as exc:
                print(f"[Detector] ⚠️ Lỗi luồng: {exc}")
                
        detector_thread = threading.Thread(target=_run, daemon=True, name="DetectorThread")
        detector_thread.start()
        print("🎥 Detector thread khởi chạy nền thành công.")
        return True
    except Exception as e:
        print(f"[Detector] ⚠️ Không thể khởi động detector: {e}")
        return False

def stop_detector_thread() -> bool:
    """Gửi tín hiệu dừng luồng detector."""
    global detector_instance
    if detector_instance and not detector_instance.stopped:
        detector_instance.stopped = True
        print("🛑 Đã gửi tín hiệu dừng tới Detector.")
        return True
    return False

# ------------------------------------------------------------------
# Background task — drain queues và broadcast
# ------------------------------------------------------------------
async def _queue_broadcaster() -> None:
    """Chạy song song với uvicorn, liên tục relay frame + alert."""
    while True:
        # Drain frame queue
        while not frame_queue.empty():
            try:
                frame_bytes = frame_queue.get_nowait()
                await manager.broadcast_bytes(frame_bytes)
            except Empty:
                break
        # Drain alert queue
        while not alert_queue.empty():
            try:
                alert = alert_queue.get_nowait()
                await manager.broadcast_json(alert)
            except Empty:
                break
        await asyncio.sleep(0.01)   # ~100 Hz poll
# ------------------------------------------------------------------
# WebSocket & REST Endpoints
# ------------------------------------------------------------------
@router.websocket("/ws/stream")
async def websocket_stream(ws: WebSocket) -> None:
    """
    FE kết nối WS /ws/stream để nhận:
      - binary message  → JPEG frame bytes
      - text   message  → JSON alert {"type": "violation", "data": {...}}
    """
    await manager.connect(ws)
    try:
        while True:
            # Giữ kết nối; FE không cần gửi gì
            await ws.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(ws)

@router.post("/api/v1/stream/restart")
async def restart_stream():
    """Dừng detector cũ (nếu có) và khởi chạy detector mới."""
    global detector_thread, detector_instance
    # Gửi tín hiệu dừng
    stop_detector_thread()
    
    # Chờ tối đa 1.5 giây để luồng cũ kết thúc và giải phóng camera/file video
    retry = 15
    while detector_thread and detector_thread.is_alive() and retry > 0:
        await asyncio.sleep(0.1)
        retry -= 1
        
    # Khởi động detector mới
    success = start_detector_thread()
    if success:
        return {"status": "success", "message": "Đã khởi động lại camera stream."}
    else:
        raise HTTPException(status_code=500, detail="Không thể khởi động lại camera stream.")

@router.post("/api/v1/stream/stop")
async def stop_stream_api():
    """Dừng stream hiện tại."""
    stopped = stop_detector_thread()
    if stopped:
        return {"status": "success", "message": "Đã dừng stream."}
    return {"status": "warning", "message": "Stream không chạy hoặc đã dừng từ trước."}

@router.get("/api/v1/stream/video")
async def get_output_video():
    """Tải file video kết quả sau khi xử lý xong."""
    from Config import Config
    cfg = Config()
    if os.path.exists(cfg.LOCAL_VIDEO_RAW):
        return FileResponse(cfg.LOCAL_VIDEO_RAW, media_type="video/mp4", filename="processed_video_raw.mp4")
    raise HTTPException(
        status_code=404,
        detail="Chưa có video kết quả. Vui lòng đợi video chạy hết để hoàn tất kết xuất."
    )
