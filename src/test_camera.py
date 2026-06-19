"""
Script chẩn đoán kết nối IP Camera từ điện thoại.
Chạy: python test_camera.py
"""
import cv2
import urllib.request
import socket
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ══════════════════════════════════════════════
# ← ĐỔI 2 DÒNG NÀY TRƯỚC KHI CHẠY
IP   = "192.168.0.108"   # IP điện thoại (xem trong app IP Webcam)
PORT = 8080             # Port (mặc định 8080)
# ══════════════════════════════════════════════

URLS = {
    "MJPEG stream" : f"http://{IP}:{PORT}/video",
    "RTSP stream"  : f"rtsp://{IP}:{PORT}/h264_ulaw.sdp",
    "JPEG snapshot": f"http://{IP}:{PORT}/shot.jpg",
}

def check_network():
    print(f"\n{'='*55}")
    print(f" Bước 1 — Ping điện thoại ({IP}:{PORT})")
    print(f"{'='*55}")
    try:
        sock = socket.create_connection((IP, PORT), timeout=3)
        sock.close()
        print(f"  ✅ Kết nối TCP tới {IP}:{PORT} thành công")
        return True
    except OSError as e:
        print(f"  ❌ Không reach được {IP}:{PORT}")
        print(f"     Lỗi: {e}")
        print()
        print("  Kiểm tra:")
        print("  • Điện thoại và máy tính có cùng mạng WiFi không?")
        print("  • App IP Webcam đã nhấn 'Start server' chưa?")
        print("  • IP trong app có khớp không? (xem màn hình app)")
        return False

def check_http_snapshot():
    print(f"\n{'='*55}")
    print(f" Bước 2 — Thử tải ảnh snapshot")
    print(f"{'='*55}")
    url = f"http://{IP}:{PORT}/shot.jpg"
    try:
        with urllib.request.urlopen(url, timeout=4) as r:
            data = r.read()
        print(f"  ✅ HTTP snapshot OK — {len(data)} bytes")
        return True
    except Exception as e:
        print(f"  ❌ Lỗi HTTP: {e}")
        return False

def check_opencv(url_label, url):
    print(f"\n  → OpenCV thử: {url_label}")
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000)
    opened = cap.isOpened()
    if opened:
        ret, frame = cap.read()
        if ret and frame is not None:
            h, w = frame.shape[:2]
            print(f"     ✅ Đọc frame thành công! Kích thước: {w}x{h}")
            cap.release()
            return True
        else:
            print(f"     ⚠️  Mở được nhưng không đọc được frame")
    else:
        print(f"     ❌ OpenCV không mở được URL này")
    cap.release()
    return False

def check_all_opencv():
    print(f"\n{'='*55}")
    print(f" Bước 3 — OpenCV thử từng URL")
    print(f"{'='*55}")
    for label, url in URLS.items():
        if check_opencv(label, url):
            print(f"\n  🎉 URL hoạt động: {url}")
            print(f"  → Dùng URL này trong Config.py")
            return url
    return None

def print_summary(working_url):
    print(f"\n{'='*55}")
    print(f" KẾT QUẢ")
    print(f"{'='*55}")
    if working_url:
        print(f"  ✅ Camera kết nối thành công!")
        print(f"  URL: {working_url}")
        print()
        print(f"  Nếu muốn dùng RTSP → trong Config.py đặt IP_CAM_PROTOCOL = \"rtsp\"")
        print(f"  Nếu dùng MJPEG hiện tại → giữ IP_CAM_PROTOCOL = \"mjpeg\"")
    else:
        print("  ❌ Không kết nối được. Nguyên nhân phổ biến:")
        print()
        print("  1. Khác mạng WiFi")
        print("     → Cả 2 thiết bị phải dùng cùng 1 router")
        print()
        print("  2. IP sai")
        print("     → Mở app IP Webcam → xem IP hiển thị → điền vào đây")
        print()
        print("  3. Firewall Windows chặn")
        print("     → Mở Windows Defender Firewall")
        print("     → Cho phép Python/uvicorn qua Private network")
        print()
        print("  4. App IP Webcam chưa Start")
        print("     → Mở app → cuộn xuống → nhấn 'Start server'")
        print()
        print("  5. Dùng app khác")
        print("     → Thử 'DroidCam' (port 4747) hoặc 'Alfred'")

if __name__ == "__main__":
    print(f"\n🔍 Test kết nối IP Camera: {IP}:{PORT}")
    reachable = check_network()
    if not reachable:
        print_summary(None)
        sys.exit(1)

    check_http_snapshot()
    working = check_all_opencv()
    print_summary(working)
