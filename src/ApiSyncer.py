# ==============================================================================
# API SYNCER — Đẩy vi phạm lên FastAPI theo luồng nền (background thread)
# ==============================================================================
import os
import threading
import requests

from Config import Config


class ApiSyncer:
    """
    Gửi bất đồng bộ thông tin vi phạm + ảnh bằng chứng lên FastAPI.
    Mỗi lần gọi send() sẽ tạo một daemon thread riêng, không block luồng chính.
    """

    def __init__(self, url: str = Config.FASTAPI_URL, enabled: bool = Config.ENABLE_API_SYNC):
        self.url     = url
        self.enabled = enabled

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def send(self, v_data: dict, image_path: str) -> None:
        """Khởi thread gửi dữ liệu; trả về ngay lập tức."""
        if not self.enabled:
            return
        thread = threading.Thread(
            target=self._post,
            args=(v_data, image_path),
            daemon=True,
        )
        thread.start()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _post(self, v_data: dict, image_path: str) -> None:
        payload = {
            "personId"     : v_data["person_id"],
            "trashId"      : v_data["trash_id"],
            "score"        : v_data["score"],
            "violationType": v_data["type"],
            "timestamp"    : v_data["time"],
        }
        try:
            with open(image_path, "rb") as f:
                files = {
                    "evidenceImage": (os.path.basename(image_path), f, "image/jpeg")
                }
                res = requests.post(self.url, data=payload, files=files, timeout=15)
            if res.status_code in (200, 201):
                print(f"\n[API] ✅ Đã đồng bộ vi phạm T{v_data['trash_id']} lên Backend")
            else:
                print(f"\n[API] ❌ Lỗi BE: {res.status_code} - {res.text}")
        except Exception as exc:
            print(f"\n[API] ⚠️  Không thể kết nối FastAPI: {exc}")
