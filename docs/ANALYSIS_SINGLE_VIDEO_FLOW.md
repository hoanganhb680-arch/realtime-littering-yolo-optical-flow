# 📊 PHÂN TÍCH CHI TIẾT: 1 LƯỢT XỬ LÝ VIDEO INPUT

## 🎯 Tổng quan kiến trúc

**Ý tưởng chính:** Hệ thống có **2 phần chạy song song**:
- **Phần 1 (Backend):** Detector thread xử lý video, phát hiện vi phạm
- **Phần 2 (Frontend):** WebSocket broadcaster đẩy kết quả tới client liên tục

```
╔════════════════════════════════════════════════════════════════╗
║                     FastAPI Server (server.py)                ║
║  - Chạy trên port 8000                                        ║
║  - Tiếp nhận kết nối từ Frontend (React + Vite)              ║
║  - Quản lý REST API endpoints                                ║
╚════════════════════════════════════════════════════════════════╝
    ↓ On Startup (khi server khởi chạy)
    ├─ [Step 1] _db.init_db()
    │            ↳ Tạo bảng SQLite lưu violation logs
    │
    ├─ [Step 2] asyncio.create_task(_queue_broadcaster())
    │            ↳ Chạy 1 async task chuyên pull frame & alert từ queues
    │            ↳ Broadcast tới tất cả FE clients qua WebSocket
    │            ↳ Poll mỗi 10ms (100 Hz)
    │
    └─ [Step 3] start_detector_thread()
                 ↳ Tạo 1 thread riêng (daemon) chạy TrashViolationDetector.run()
                 ↳ Thread này hoạt động 24/7, xử lý video frame-by-frame
                 ↳ Không block event loop của FastAPI
                 ↳ Push kết quả vào frame_queue & alert_queue

┌─────────────────────────────────────────┐
│  Detector Thread (chạy nền, daemon=True) │
│  File: TrashViolationDetector.run()      │
└─────────────────────────────────────────┘
    ↓ (song song với FastAPI event loop)
    ├─ Đọc video frame: cap.read() → numpy array
    ├─ Chạy YOLO detect: model.track() 
    ├─ Phát hiện motion: MOG2.apply()
    ├─ Track optical flow: cv2.calcOpticalFlowPyrLK()
    ├─ Tính ownership score: OwnershipScorer.compute()
    ├─ Log vi phạm: ViolationLogger.confirm_and_log()
    └─ Push kết quả:
        ├─ frame_queue.put(frame_jpg_bytes)  ← JPEG output
        └─ alert_queue.put(violation_dict)   ← Violation event

┌──────────────────────────────────────┐
│  WebSocket Broadcaster (async task)  │
│  File: stream_router.py              │
└──────────────────────────────────────┘
    ↓ Chạy liên tục (async while True loop)
    ├─ Kiểm frame_queue có frame mới?
    │  ├─ YES: Lấy frame_bytes
    │  └─ Gửi tới tất cả connected clients: manager.broadcast_bytes()
    │
    └─ Kiểm alert_queue có alert mới?
       ├─ YES: Lấy violation_dict
       └─ Gửi tới tất cả connected clients: manager.broadcast_json()

┌──────────────────────────────────┐
│  Frontend Clients (React)         │
│  WebSocket: /ws/stream           │
└──────────────────────────────────┘
    ↓ Nhận từ broadcaster
    ├─ Nhận JPEG frame_bytes
    │  └─ Render trên canvas/img
    │
    └─ Nhận JSON alert
       └─ Popup thông báo vi phạm
```

**💡 Chìa khóa:** Thread detector xử lý độc lập, đẩy kết quả vào queue.
Broadcaster async lấy từ queue, phân phát cho FE. Không chặn nhau!

---

## 🔄 CHI TIẾT: Xử lý 1 frame video

### **Bước 1: KHỞI ĐỘNG HỆ THỐNG**

**File: `server.py` → `on_startup()` event**

```python
@app.on_event("startup")
async def on_startup() -> None:
    # ← Khi server khởi chạy, chạy 3 bước này:
    
    _db.init_db()
    # Bước 1: Tạo bảng SQLite
    # - Bảng: violations (lưu mỗi lần phát hiện rác)
    # - Bảng: evidence_clips (lưu video clip bằng chứng)
    
    asyncio.create_task(_queue_broadcaster())
    # Bước 2: Khởi chạy async task "broadcaster"
    # - Chạy song song với FastAPI event loop
    # - Mỗi 10ms:
    #   ├─ Check frame_queue có frame mới?
    #   │  ↳ YES → broadcast_bytes() tới FE
    #   └─ Check alert_queue có vi phạm mới?
    #      ↳ YES → broadcast_json() tới FE (popup)
    
    start_detector_thread()
    # Bước 3: Khởi chạy detector thread
    # - Tạo thread daemon (sẽ dừng khi app dừng)
    # - Gọi TrashViolationDetector().run()
    # - Xử lý video frame-by-frame chạy liên tục
```

**📊 Kết quả:**
- ✅ Database SQLite sẵn sàng
- ✅ Broadcaster task chạy (poll queue mỗi 10ms)
- ✅ Detector thread chạy (xử lý video liên tục)

---

### **Bước 2: KHỞI TẠO DETECTOR**

**File: `TrashViolationDetector.py` → `run()` method**

```python
def run(self) -> None:
    # ═══════════════════════════════════════════════════════
    # PHASE 1: SETUP & INITIALIZATION
    # ═══════════════════════════════════════════════════════
    
    # 1. Load YOLO model từ weights
    model = YOLO(cfg.MODEL_PATH)  
    # ← Tải pretrained YOLOv8 từ file
    # ← Dùng GPU nếu có sẵn (nhanh gấp 10x)
    # ← Model đã được train để detect:
    #   - Class 0: Person (người)
    #   - Class 1: Trash (rác)
    
    # 2. Mở video source
    cap = self._open_capture(is_live)
    # ← Nếu is_live=True:
    #   └─ Mở ThreadedCamera (camera stream thực thời)
    # ← Nếu is_live=False:
    #   └─ Mở file video từ disk (cv2.VideoCapture)
    
    # 3. Tạo output writer
    out_vid, source_fps = self._init_writer(cap)
    # ← Tạo file output video (dạng .avi)
    # ← source_fps: FPS của video gốc (VD: 30 FPS)
    
    # 4. Tính frame stride (bỏ qua frame để giảm FPS)
    file_frame_step = self._file_frame_step(source_fps, is_live)
    # ← Ví dụ: video gốc 30fps, muốn output 15fps
    #   → frame_step = 2 (xử lý frame 1, 3, 5, 7, ...)
    # ← Giúp tiết kiệm thời gian, GPU memory
    
    # 5. Khởi tạo biến tạm
    prev_gray = None
    # ← Sẽ chứa frame xám của frame trước (t-1)
    # ← Dùng để tính optical flow (vận tốc) ở frame hiện tại (t)
    frame_idx = 0  # Đếm frame xử lý thành công
    raw_frame_idx = 0  # Đếm tất cả frame (kể cả bỏ qua)
```

**📊 Kết quả sau bước setup:**
```
✅ YOLO model: sẵn sàng detect người & rác
✅ Video capture: sẵn sàng đọc frame
✅ Output writer: sẵn sàng ghi file
✅ frame_step = 2 (ví dụ 30fps → 15fps)
```

---

### **Bước 3: VÒ LẶP CHỦ (Main Loop) — Frame-by-Frame**

**File: `TrashViolationDetector.py` → `run()` main loop**

```python
# ═══════════════════════════════════════════════════════
# PHASE 2: MAIN LOOP - Xử lý từng frame
# ═══════════════════════════════════════════════════════

while not self.stopped and (is_live or cap.isOpened()):
    # ← Chạy liên tục cho đến khi:
    #   ├─ self.stopped == True (người dùng stop), HOẶC
    #   └─ cap.isOpened() == False (video kết thúc)
    
    # ─────────────────────────────────────────
    # 3.1: ĐỌC FRAME TỪ CAMERA/VIDEO
    # ─────────────────────────────────────────
    ret, frame = cap.read()
    # ← cap.read() trả về:
    #   ├─ ret: bool (True nếu đọc thành công, False nếu hết video)
    #   └─ frame: numpy array shape (height, width, 3) - ảnh BGR
    # ← VD: frame = array shape (1080, 1920, 3), dtype uint8
    
    if not ret:
        # ← Không đọc được frame (hết video hoặc lỗi camera)
        if is_live:
            self._log_camera_wait(cfg.VIDEO_SOURCE)
            time.sleep(0.5)  # Chờ 0.5s để camera sẵn sàng
            continue  # Retry
        break  # Video kết thúc, thoát loop
    
    raw_frame_idx += 1
    # ← Đếm tất cả frame (kể cả frame bỏ qua)
    # ← VD: raw_frame_idx = 1, 2, 3, 4, 5, ...
    
    # ─────────────────────────────────────────
    # 3.2: BỎ QUA FRAME NẾU CẦN (Frame Stride)
    # ─────────────────────────────────────────
    if self._skip_file_frame(raw_frame_idx, file_frame_step):
        continue
    # ← Nếu file_frame_step=2:
    #   ├─ Frame 1: KHÔNG bỏ (1-1) % 2 = 0 ✓
    #   ├─ Frame 2: BỎ (2-1) % 2 = 1 ✗
    #   ├─ Frame 3: KHÔNG bỏ (3-1) % 2 = 0 ✓
    #   ├─ Frame 4: BỎ (4-1) % 2 = 1 ✗
    #   └─ Frame 5: KHÔNG bỏ (5-1) % 2 = 0 ✓
    # ← Giúp giảm FPS: 30fps → 15fps
    
    # ─────────────────────────────────────────
    # 3.3: CHUẨN BỊ FRAME
    # ─────────────────────────────────────────
    frame = self._prepare_frame(frame)
    # ← Resize frame về kích thước chuẩn (VD: 640x480)
    # ← Normalize pixel values (VD: 0-255 → 0-1)
    # ← Giảm thời gian YOLO inference
    
    frame_idx += 1
    # ← Đếm frame xử lý thành công
    # ← VD: frame_idx = 1, 2, 3, 4, 5, ...
    
    # ─────────────────────────────────────────
    # 3.4: ← CORE PROCESSING PIPELINE ←
    # ─────────────────────────────────────────
    curr_gray, annotated = self._process_frame(
        model, frame, frame_idx, prev_gray, track_kwargs, floor_kwargs
    )
    # ← HÀM QUAN TRỌNG NHẤT! Xử lý toàn bộ detection, tracking, scoring
    # ← Input:
    #   ├─ model: YOLO
    #   ├─ frame: ảnh BGR đã chuẩn bị
    #   ├─ frame_idx: số frame hiện tại
    #   ├─ prev_gray: ảnh xám frame trước (để tính optical flow)
    #   ├─ track_kwargs: config YOLO tracking
    #   └─ floor_kwargs: config detect rác trên đất
    # ← Output:
    #   ├─ curr_gray: ảnh xám frame hiện tại
    #   └─ annotated: ảnh đã vẽ bounding boxes + text
    #      (sẽ được save video + push FE)
    
    # ─────────────────────────────────────────
    # 3.5: LƯU VIDEO OUTPUT
    # ─────────────────────────────────────────
    if out_vid is not None:
        out_vid.write(annotated)
    # ← Ghi frame annotated vào file video output
    # ← File path: cfg.LOCAL_VIDEO_RAW (VD: "output/raw.avi")
    # ← Mỗi frame ghi mất ~2-5ms (I/O)
    
    # ─────────────────────────────────────────
    # 3.6: PUSH FRAME QFra WEBSOCKET
    # ─────────────────────────────────────────
    self._push_frame(annotated)
    # ← Chuyển ảnh annotated sang JPEG bytes
    # ← Push vào frame_queue
    # ← Broadcaster sẽ pull + send tới FE
    # ← FE render trên canvas
    
    # ─────────────────────────────────────────
    # 3.7: LƯU FRAME XÁM CHO FRAME TIẾP THEO
    # ─────────────────────────────────────────
    prev_gray = curr_gray.copy()
    # ← Frame xám hiện tại → sẽ là "prev_gray" ở frame tiếp theo
    # ← Dùng để tính optical flow (t-1 → t)
    
    # ─────────────────────────────────────────
    # 3.8: LOG & SPEED CONTROL
    # ─────────────────────────────────────────
    self._pace_file_mode(is_live, t_start)
    # ← Nếu file mode: thêm delay để FPS phù hợp
    # ← VD: target 15fps → delay ~66ms/frame
    
    if frame_idx % 50 == 0:
        print(f"  [{frame_idx} frames] processed")
    # ← In log mỗi 50 frame (VD: "  [50 frames] processed")
```

**📊 Kết quả mỗi lần loop:**
```
Frame 1: ✅ Đọc, bỏ qua (frame_step=2)
Frame 2: ✅ Đọc, xử lý → frame_idx=1, push FE
Frame 3: ✅ Đọc, bỏ qua
Frame 4: ✅ Đọc, xử lý → frame_idx=2, push FE
...
(Cứ sau 2 frame đọc, xử lý 1 frame)
```

---

### **Bước 4: CORE PROCESSING PIPELINE** ⭐⭐⭐

**File: `TrashViolationDetector._process_frame()` — ĐÂY LÀ TRÁI TIM!**

```python
def _process_frame(self, model, frame, frame_idx, prev_gray, track_kwargs, floor_kwargs):
    # ═══════════════════════════════════════════════════════
    # PHASE 3A: CONVERT TO GRAYSCALE
    # ═══════════════════════════════════════════════════════
    
    curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # ← Chuyển frame BGR → Grayscale (1 channel thay vì 3)
    # ← Dùng cho optical flow tracking
    # ← VD: frame (1080, 1920, 3) → curr_gray (1080, 1920)
    
    # ═══════════════════════════════════════════════════════
    # PHASE 3B: MOTION DETECTION (MOG2)
    # ═══════════════════════════════════════════════════════
    
    mog2_alerts = self._motion_detector.get_alerts(frame)
    # ← Phát hiện vùng chuyển động bằng MOG2 (Mixture of Gaussians)
    # ← Input: frame BGR gốc
    # ← Output: [(cx1, cy1, area1), (cx2, cy2, area2), ...]
    #   ├─ cx, cy: tọa độ tâm vùng chuyển động
    #   └─ area: diện tích vùng (pixels²)
    # ← VD: [(640, 720, 15000), (250, 650, 8000)]
    #       (2 vùng chuyển động được phát hiện)
```

**💡 MOG2 là gì?**
```
MOG2 = Mixture of Gaussians v2
- Là thuật toán detect nền (background subtraction)
- Ý tưởng: "Pixel nào khác biệt so với nền → là foreground"
- Áp dụng:
  1. Gaussian Mixture Model học nền từ 500 frame đầu
  2. Mỗi frame mới: so sánh pixel với model nền
  3. Pixel khác biệt → marking as foreground (white)
  4. Apply morphology (open/close) để loại noise
  5. Find contours → tính tâm & diện tích

VD: Đoạn video tĩnh (nền là tường)
Frame 1-500: MOG2 "học" nền = tường
Frame 501: Người bước vào → pixels người khác biệt so với nền
          → MOG2 tô trắng vùng người
          → Find contours → 1 blob chuyển động
```

```python
    # ═══════════════════════════════════════════════════════
    # PHASE 3C: YOLO DETECTION & TRACKING
    # ═══════════════════════════════════════════════════════
    
    results = model.track(frame, **track_kwargs)
    # ← Chạy YOLO detection + tracking trên frame
    # ← model.track():
    #   ├─ Detect: Tìm bounding box của mỗi object
    #   ├─ Classify: Dán nhãn class (0=person, 1=trash)
    #   ├─ Track: Gán track ID (nếu có người/rác ở frame trước)
    #   └─ Return: results[] list (1 result per image)
    # ← track_kwargs:
    #   ├─ persist=True: Giữ track ID khi object tạm mất (occlusion)
    #   ├─ conf=0.5: Chỉ lấy detection có confidence > 0.5
    #   ├─ iou=0.5: NMS threshold (bỏ duplicate boxes gần nhau)
    #   └─ max_det=300: Tối đa 300 objects/frame
    
    annotated = results[0].plot()
    # ← Vẽ bounding boxes + track IDs lên frame
    # ← VD: □ Person_101 (xanh)
    #       □ Trash_005 (đỏ)
    # ← Frame có boxes → sẵn sàng push FE
    
    # YOLO results[0] chứa:
    # results[0].boxes.xyxy  = [[x1, y1, x2, y2], ...]  bounding boxes
    # results[0].boxes.id    = [101, 102, 5, 6, ...]    track IDs
    # results[0].boxes.cls   = [0, 0, 1, 1, ...]        classes (0=person, 1=trash)
    # results[0].boxes.conf  = [0.92, 0.87, 0.95, ...]  confidence
```

**💡 YOLO Tracking là gì?**
```
YOLO track:
- Frame t-1 (trước):
  □ Person 101 tại (640, 720)
  □ Trash 005 tại (500, 750)

- Frame t (hiện tại):
  □ Bounding box tại (642, 718) → YOLO tính: "Gần nhất tới Person 101"
                                 → Gán lại ID=101
  □ Bounding box tại (499, 752) → YOLO tính: "Gần nhất tới Trash 005"
                                 → Gán lại ID=005

→ Track ID ổn định qua các frame!
```

```python
    # ═══════════════════════════════════════════════════════
    # PHASE 3D: PARSE DETECTIONS THÀNH STABLE IDs
    # ═══════════════════════════════════════════════════════
    
    current_persons, current_trashes = self._parse_detections(results, frame_idx)
    # ← Parse YOLO output → structured format
    # ← Output:
    #   ├─ current_persons:  {1001: (640, 720), 1002: (320, 680), ...}
    #   │                     {person_id: (cx, cy), ...}
    #   │                     ← cy = foot anchor (chân, không tâm thân)
    #   │                     ← VÌ rác thường gần chân, không gần tâm
    #   └─ current_trashes:  {101: (500, 750), 102: (300, 600), ...}
    #                        {trash_id: (cx, cy), ...}
    #                        ← (cx, cy) = tâm rác
    
    self._augment_persons_from_motion(
        current_persons, current_trashes, mog2_alerts, frame_idx, frame.shape, annotated
    )
    # ← Nếu YOLO miss người (VD: người bé, bị che khuất)
    # ← Dùng MOG2 motion detection để bổ sung
    # ← VD: YOLO detect 5 người, MOG2 phát hiện thêm 2 người
    # ← Thêm những motion blob vào current_persons với synthetic ID
    # ← VD: current_persons = {1001, 1002, 1003, 1004, 1005, 200001, 200002}
    #       (5 từ YOLO + 2 synthetic từ MOG2)
```

**💡 Tại sao dùng motion fallback?**
```
Tình cảnh thực tế:
- YOLO có khi detect miss (VD: người mặc màu xám giống tường)
- MOG2 vẫn bắt được chuyển động
- Kết hợp cả 2:
  ├─ YOLO: Chính xác (70-90% confidence)
  └─ MOG2: Coverage tốt hơn (catch miss case)
  → Hiệu suất tổng thể cao hơn
```

```python
    # ═══════════════════════════════════════════════════════
    # PHASE 3E: REMEMBER PERSON EVIDENCE
    # ═══════════════════════════════════════════════════════
    
    self._remember_person_evidence(current_persons, annotated)
    # ← Lưu ảnh frame hiện tại cho mỗi người
    # ← Dùng làm evidence sau (nếu person đó là owner của rác)
    # ← VD: _person_frame_jpg[1001] = (frame_idx, jpeg_bytes)
    
    # ═══════════════════════════════════════════════════════
    # PHASE 3F: OPTICAL FLOW TRACKING
    # ═══════════════════════════════════════════════════════
    
    self._flow_tracker.update(prev_gray, curr_gray, current_persons)
    # ← Tính vector vận tốc (velocity) của mỗi người
    # ← Input:
    #   ├─ prev_gray: frame xám của frame trước (t-1)
    #   ├─ curr_gray: frame xám hiện tại (t)
    #   └─ current_persons: {1001: (640, 720), ...}
    # ← Output (lưu trong self._flow_tracker.flow_vecs):
    #   {1001: [(vx1, vy1), (vx2, vy2), ...], ...}
    #   ← Danh sách vector vận tốc qua các frame
    #   ← VD: [(2.5, -1.3), (2.1, -0.8), (1.9, -0.5), ...]
    #   ← Đơn vị: pixel/frame
    #   ← Ý nghĩa: người 1001 đang di chuyển theo hướng (2.5px phải, 1.3px lên)
```

**💡 Optical Flow là gì?**
```
Optical Flow = Phát hiện chuyển động của pixels
- Lucas-Kanade algorithm:
  1. Xác định feature points (góc, edge) trên prev_gray
  2. Track những points đó trên curr_gray
  3. Tính vector (vx, vy) mỗi point đã di chuyển
  4. Trung bình hóa → vận tốc trung bình person

VD: Người đi từ trái sang phải
  prev_gray: □ (tại x=640)
  curr_gray: □ (tại x=642)
  → vector = (2, 0) = di chuyển 2px sang phải
```

```python
    # ═══════════════════════════════════════════════════════
    # PHASE 3G: FLOOR TRASH DETECTION
    # ═══════════════════════════════════════════════════════
    
    current_trashes.update(
        self._detect_floor_trash_candidates(
            model, frame, annotated, current_trashes, current_persons,
            mog2_alerts, frame_idx, floor_kwargs
        )
    )
    # ← Detect rác trên mặt đất (special case)
    # ← VD: Một cái túi nhựa nằm dưới đất (YOLO có thể miss)
    # ← Dùng image segmentation / contour analysis
    # ← Thêm vào current_trashes: {103: (450, 800), ...}
    
    # ═══════════════════════════════════════════════════════
    # PHASE 3H: PROCESS TRASHES (Core Logic!)
    # ═══════════════════════════════════════════════════════
    
    self._process_trashes(
        current_trashes, current_persons, mog2_alerts, annotated, frame_idx
    )
    # ← ← ← ĐÂY LÀ LOGIC PHÁT HIỆN VI PHẠM CHÍNH! ← ← ←
    # ← For mỗi rác trong current_trashes:
    #   1. Kiểm: rác này lần đầu thấy? → Tạo entry trong trash_registry
    #   2. Tính: ownership score từ mỗi người
    #   3. Chọn: người có score cao nhất làm owner
    #   4. Nếu score > MIN_SCORE → CONFIRMED VI PHẠM!
    #   5. Log: lưu ảnh, DB, push alert
    # ← Chi tiết xem Bước 6 & 7 dưới
    
    # ═══════════════════════════════════════════════════════
    # PHASE 3I: CLEANUP & VISUALIZATION
    # ═══════════════════════════════════════════════════════
    
    self._cleanup_stale_trash(current_trashes, frame_idx)
    # ← Xóa rác đã rời scene (quá lâu không thấy)
    
    self._draw_sticky_trash(annotated, current_trashes, frame_idx)
    # ← Vẽ bounding box rác (sticky = vẫn vẽ ngay cả nếu YOLO mất detect)
    
    self._draw_hud(annotated, frame_idx)
    # ← Vẽ text HUD: frame count, fps, etc.
    
    self._remember_evidence_frame(annotated)
    # ← Lưu frame annotated vào buffer (dùng làm evidence clip)
    
    return curr_gray, annotated
    # ← Trả lại:
    #   ├─ curr_gray: frame xám (dùng làm prev_gray ở frame tiếp theo)
    #   └─ annotated: frame đã vẽ (dùng để lưu video + push FE)
```

#### **4.1: Chuyển đổi frame → xám (grayscale)**
```python
curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
```
- Dùng cho optical flow tracking (Lucas-Kanade)

#### **4.2: Phát hiện chuyển động (MOG2)**
```python
mog2_alerts = self._motion_detector.get_alerts(frame)
```

**File: `MotionDetector.py`**
```python
def get_alerts(self, frame):
    fg = self._subtractor.apply(frame)  # Background subtraction
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN,  self._kernel)   # Loại noise
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, self._kernel)   # Fill holes
    
    contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    alerts = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        M = cv2.moments(cnt)
        cx = int(M['m10'] / M['m00'])  # Tâm X
        cy = int(M['m01'] / M['m00'])  # Tâm Y
        alerts.append((cx, cy, area))
    
    return alerts  # [(cx, cy, area), ...]
```

**Kết quả:** Danh sách vùng chuyển động: `[(x1, y1, area1), (x2, y2, area2), ...]`

#### **4.3: YOLO Detection & Tracking**
```python
results = model.track(frame, **track_kwargs)
annotated = results[0].plot()
```

**Track_kwargs** (từ `_build_track_kwargs`):
```python
track_kwargs = {
    "persist": True,
    "conf": cfg.CONF,
    "iou": cfg.IOU,
    "max_det": cfg.MAX_DET,
}
```

**YOLO output** (`results[0]`):
- `boxes.xyxy`: `[[x1, y1, x2, y2], ...]` (bounding boxes)
- `boxes.id`: `[123, 124, 125, ...]` (track IDs)
- `boxes.cls`: `[0, 1, 0, 1, ...]` (class: 0=person, 1=trash)
- `boxes.conf`: `[0.92, 0.87, ...]` (confidence)

**Kết quả:** Frame với bounding boxes đã được vẽ

#### **4.4: Parse Detections thành ID ổn định**

**File: `detection_parsing.py` → `_parse_detections()`**

```python
def _parse_detections(self, results, frame_idx):
    current_persons = {}  # {person_id: (cx, cy), ...}
    current_trashes = {}  # {trash_id: (cx, cy), ...}
    
    boxes = results[0].boxes
    ids = boxes.id.cpu().numpy() if boxes.id else [None] * len(boxes)
    
    for box, obj_id, cls, conf in zip(...):
        x1, y1, x2, y2 = box
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)
        
        if int(cls) == 0:  # Person
            # Dùng điểm gần chân (foot anchor) thay vì tâm
            anchor, points = self._person_points_from_box(box, results[0].orig_shape)
            
            # Resolve ID (nếu YOLO mất track, dùng spatial proximity)
            oid = self._resolve_person_id(obj_id, anchor, frame_idx, current_persons)
            current_persons[oid] = anchor
            
            # Lưu lịch sử
            if oid not in self._person_history:
                self._person_history[oid] = deque(maxlen=HISTORY_FRAMES)
            self._person_history[oid].append({
                'anchor': anchor,
                'points': points,
                'frame_idx': frame_idx,
            })
            
        elif int(cls) == 1:  # Trash
            oid = self._resolve_trash_id(obj_id, (cx, cy), frame_idx, used_trash_ids)
            current_trashes[oid] = (cx, cy)
    
    return current_persons, current_trashes
```

**Kết quả:**
- `current_persons`: `{1001: (640, 720), 1002: (320, 680), ...}`
- `current_trashes`: `{101: (500, 750), 102: (300, 600), ...}`

#### **4.5: Augment Persons từ Motion (Fallback)**

Nếu YOLO miss người nhỏ/bị che, dùng MOG2 motion blobs:

**File: `detection_parsing.py` → `_augment_persons_from_motion()`**

```python
if getattr(cfg, "MOTION_PERSON_FALLBACK", False):
    for mx, my, area in sorted(mog2_alerts, key=lambda x: x[2], reverse=True):
        # Check constraints
        if area < min_area or area > max_area or my < min_y or my > max_y:
            continue
        # Check không quá gần người đã detect
        if any(hypot(mx - px, my - py) <= 80 for px, py in current_persons.values()):
            continue
        
        # Thêm synthetic person ID
        anchor = (int(mx), min(h - 1, int(my + offset)))
        oid = self._resolve_person_id(None, anchor, frame_idx, current_persons)
        current_persons[oid] = anchor
```

**Kết quả:** Bổ sung thêm người từ motion detection nếu cần

#### **4.6: Optical Flow Tracking (Vận tốc người)**

**File: `OpticalFlowTracker.py` → `update()`**

```python
def update(self, prev_gray, curr_gray, current_persons):
    for p_id, (cx, cy) in current_persons.items():
        pts_old = self.flow_pts.get(p_id, ...)
        
        # Lucas-Kanade optical flow
        pts_new, status, _ = cv2.calcOpticalFlowPyrLK(
            prev_gray, curr_gray, pts_old, None, **self._lk_params
        )
        
        good_new = pts_new[status == 1]  # Điểm track thành công
        good_old = pts_old[status == 1]
        
        # Tính vận tốc trung bình
        vecs = good_new - good_old  # (vx, vy)
        avg_vx = np.mean(vecs[:, 0])
        avg_vy = np.mean(vecs[:, 1])
        
        # Lưu lịch sử vận tốc
        self.flow_vecs[p_id].append((avg_vx, avg_vy))
```

**Kết quả:** `flow_vecs = {1001: [(1.2, 0.5), (-0.3, 0.8), ...], ...}`

#### **4.7: Floor Trash Detection (Rác trên đất)**

Một bước detect riêng cho rác đặt trên mặt đất (không phải YOLO detect):

```python
current_trashes.update(
    self._detect_floor_trash_candidates(
        model, frame, annotated, current_trashes, current_persons,
        mog2_alerts, frame_idx, floor_kwargs
    )
)
```

#### **4.8: Process Trashes (Xác định owner & vi phạm)**

**File: `trash_lifecycle.py` → `_process_trashes()`**

```python
def _process_trashes(self, current_trashes, current_persons, mog2_alerts, annotated, frame_idx):
    for t_id, t_center in current_trashes.items():
        if t_id not in self._trash_registry:
            # Trash mới → tạo entry
            self._trash_registry[t_id] = {
                'first_seen': frame_idx,
                'owner_id': None,
                'owner_score': 0.0,
                'status': 'tracking',
            }
        
        trash_data = self._trash_registry[t_id]
        
        # Tính ownership score cho mỗi người
        best_owner_id, best_score, is_ambiguous = self._scorer.find_best_owner(
            trash_center,
            current_persons,
            frame_idx,
            self._person_history,
            self._flow_tracker.flow_vecs,
            mog2_alerts,
        )
        
        trash_data['owner_id'] = best_owner_id
        trash_data['owner_score'] = best_score
        trash_data['is_ambiguous'] = is_ambiguous
        
        # Confirm violation
        if best_owner_id is not None and best_score >= cfg.MIN_SCORE:
            self._logger.confirm_and_log(
                t_id, t_center, trash_data, 
                'trash_violation', annotated, current_persons, frame_idx
            )
```

---

### **Bước 5: OWNERSHIP SCORING** ⭐

**File: `OwnershipScorer.py` → `compute()` & `find_best_owner()`**

**4 tín hiệu được kết hợp:**

```python
def compute(self, trash_center, p_id, history, current_frame, flow_vecs):
    tx, ty = trash_center
    positions = list(history)  # [(cx, cy, frame_idx), ...]
    
    # 1️⃣ PROXIMITY SCORE (Khoảng cách)
    min_dist, _ = self._nearest(positions, tx, ty, current_frame)
    if min_dist > self.trajectory_radius:
        return 0.0  # Quá xa
    proximity_score = 1.0 - (min_dist / self.trajectory_radius)
    
    # 2️⃣ DIRECTION SCORE (Hướng di chuyển)
    direction_score = self._direction_score(positions, tx, ty)
    # ← Kiểm: người có rời xa rác không?
    # ← Nếu có → điểm cao
    
    # 3️⃣ FLOW SCORE (Optical flow vector)
    flow_score = self._flow_score(p_id, positions, tx, ty, flow_vecs)
    # ← Vector vận tốc chỉ ra hướng rời xa rác?
    
    # 4️⃣ RECENCY SCORE (Gần đây)
    recency_score = 1.0 - (closest_offset / self.history_frames)
    # ← Người gần rác bao lâu rồi?
    
    # Tổng hợp (weighted sum)
    score = (
        0.40 * proximity_score +
        0.30 * direction_score +
        0.20 * flow_score +
        0.10 * recency_score
    )
    
    return round(score, 4)  # [0.0, 1.0]
```

**Ví dụ cụ thể:**
```
Rác ID 101 tại (500, 750)
  ├─ Person 1001: score = 0.85 (gần nhất, chuyển động rời xa)
  ├─ Person 1002: score = 0.42 (cách xa, không rời xa)
  └─ Person 1003: score = 0.15 (rất cách xa)

→ Best owner = Person 1001 (score 0.85 > MIN_SCORE=0.5)
```

---

### **Bước 6: VIOLATION CONFIRMATION & LOGGING**

**File: `ViolationLogger.py` → `confirm_and_log()`**

```python
def confirm_and_log(self, t_id, t_center, data, vtype, annotated, 
                    current_persons, frame_idx, clip_frames=None):
    data["status"] = "confirmed"
    owner_id = data["owner_id"]
    
    if owner_id is not None and not data["is_ambiguous"]:
        # 1️⃣ Lưu ảnh bằng chứng
        evidence = self._decode_frame(data.get("owner_frame_jpg")) or annotated.copy()
        
        cv2.putText(evidence, f"VI PHAM: Person_{owner_id}", ...)
        cv2.circle(evidence, t_center, 25, (0, 0, 255), 3)
        
        local_path = f"violation_P{owner_id}_T{t_id}_F{frame_idx}.jpg"
        cv2.imwrite(local_path, evidence)
        
        # 2️⃣ Lưu vào SQLite
        v_data = {
            "person_id": owner_id,
            "trash_id": t_id,
            "violation_type": vtype,
            "owner_score": data["owner_score"],
            "is_ambiguous": data["is_ambiguous"],
            "evidence_path": local_path,
            "frame_idx": frame_idx,
        }
        _db.insert_violation(v_data)
        self.violation_log.append(v_data)
        
        # 3️⃣ Push qua API
        self._syncer.queue_violation(v_data)
```

---

### **Bước 7: FRAME OUTPUT**

#### **7.1: Lưu video output**
```python
if out_vid is not None:
    out_vid.write(annotated)  # Ghi frame đã annotate vào file
```

#### **7.2: Push frame qua WebSocket**
```python
def _push_frame(self, annotated):
    if not has_stream_clients():
        return
    
    _, frame_jpg = cv2.imencode('.jpg', annotated)
    frame_queue.put(frame_jpg.tobytes(), timeout=0.1)
```

**Stream router broadcaster (`_queue_broadcaster`):**
```python
async def _queue_broadcaster():
    while True:
        while not frame_queue.empty():
            frame_bytes = frame_queue.get_nowait()
            await manager.broadcast_bytes(frame_bytes)  # → tất cả FE clients
        
        while not alert_queue.empty():
            alert = alert_queue.get_nowait()
            await manager.broadcast_json(alert)  # → tất cả FE clients
        
        await asyncio.sleep(0.01)  # ~100Hz poll
```

---

## 📊 FLOW DIAGRAM - Một frame đi qua hệ thống

```
RAW FRAME (từ video)
    ↓
[cv2.cvtColor] → GRAYSCALE (cho optical flow)
    ↓
[MOG2] → Motion detection alerts: [(x, y, area), ...]
    ↓
[YOLO] → Detect: boxes, IDs, classes
    ├─ Class 0 (Person) → Parse ra person IDs
    ├─ Class 1 (Trash) → Parse ra trash IDs
    └─ Results + annotated frame
    ↓
[Augment từ Motion] → Thêm person nếu MOG2 bắt được nhưng YOLO miss
    ↓
[Optical Flow] → Tính vận tốc mỗi person
    ├─ prev_gray × curr_gray → Lucas-Kanade
    ├─ flow_pts: điểm tracking
    └─ flow_vecs: vector vận tốc [(vx, vy), ...]
    ↓
[Floor Trash Detect] → Detect rác trên mặt đất
    ↓
[Process Trashes]
    ├─ For each trash:
    │  ├─ [Ownership Scoring] → 4 factors: proximity, direction, flow, recency
    │  └─ [Confirm Violation] → Log, save evidence, push API
    └─ Update trash registry
    ↓
[Draw Visualizations] → Bounding boxes, text, HUD
    ↓
[Output]
    ├─ Write to video file (annotated)
    ├─ Push JPEG to WebSocket queue (frame_queue)
    └─ Push alert to WebSocket queue (alert_queue)
    ↓
[WebSocket Broadcaster] (async)
    ├─ Broadcast frame_bytes → all FE clients
    └─ Broadcast alert JSON → all FE clients
    ↓
[FE] Hiển thị frame + popup alert
```

---

## 🔑 Key Data Structures

### **Current Frame State**
```python
current_persons = {
    1001: (640, 720),  # person_id → foot anchor
    1002: (320, 680),
}

current_trashes = {
    101: (500, 750),   # trash_id → center
    102: (300, 600),
}

mog2_alerts = [
    (640, 720, 15000),  # (cx, cy, area)
    (250, 650, 8000),
]
```

### **Persistent State**
```python
_person_history[1001] = deque([
    {'anchor': (640, 720), 'points': [...], 'frame_idx': 100},
    {'anchor': (642, 718), 'points': [...], 'frame_idx': 101},
    # ... up to HISTORY_FRAMES
])

_trash_registry[101] = {
    'first_seen': 95,
    'owner_id': 1001,
    'owner_score': 0.85,
    'is_ambiguous': False,
    'status': 'confirmed',
}

flow_vecs[1001] = deque([
    (1.2, 0.5),    # (vx, vy) pixel/frame
    (-0.3, 0.8),
    # ...
])
```

---

## 📍 Configuration Parameters (Config.py)

```python
# Model & Detection
CONF = 0.5                      # YOLO confidence threshold
IOU = 0.5                       # IOU threshold for NMS

# Motion Detection (MOG2)
MOG2_HISTORY = 500
MOG2_THRESHOLD = 16
MOG2_MIN_AREA = 500             # Diện tích tối thiểu

# Ownership Scoring
TRAJECTORY_RADIUS = 150         # Vùng gần rác (pixels)
SPAWN_RADIUS = 50
MIN_SCORE = 0.5                 # Ngưỡng xác nhận vi phạm
AMBIGUOUS_MARGIN = 0.1

# Optical Flow
LK_PARAMS = {
    'winSize': (15, 15),
    'maxLevel': 2,
    'criteria': (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
}
FLOW_HISTORY_FRAMES = 30

# History
HISTORY_FRAMES = 300            # Lữu lịch 10 giây @ 30fps

# Output
FILE_MODE_FPS = 15              # FPS output video (file mode)
LOCAL_VIDEO_RAW = "output/raw.avi"
```

---

## 🎬 Timeline: 30 FPS video

| Frame | Time | Event |
|-------|------|-------|
| 1-10  | 0-0.33s | Motion warmup (MOG2 learning) |
| 11-50 | 0.37-1.67s | Người vào scene |
| 51-100 | 1.7-3.33s | Người đặt rác |
| 101-150 | 3.37-5s | Ownership scoring (4 signals accumulate) |
| 151-200 | 5-6.67s | Confidence cao → **VIOLATION CONFIRMED** |
| 201+ | 6.7+ | Evidence lưu, alert push FE |

---

## 🚀 Performance Notes

- **1 frame:** ~50-100ms (YOLO inference slowest)
- **MOG2:** ~5ms
- **Optical Flow:** ~10-15ms
- **Ownership Scoring:** ~2-5ms per trash
- **Total throughput:** ~10-15 FPS real-time (bottleneck YOLO)

---

## ❓ Frequently Asked Questions

**Q: Tại sao cần lưu gray frame (prev_gray)?**
A: Dùng cho optical flow Lucas-Kanade để track vận tốc người (t-1 → t)

**Q: MOG2 vs YOLO - khi nào dùng cái nào?**
A: YOLO chính; MOG2 dùng để detect người YOLO miss + boost ownership score nếu motion nearby trash

**Q: Làm sao xác định "ai chủ nhân rác"?**
A: 4 factors: proximity (gần nhất?), direction (rời xa?), flow (vận tốc chỉ ra đó?), recency (gần đây bao lâu?)

**Q: WebSocket queue maxsize = 1 có nghĩa gì?**
A: Chỉ giữ frame mới nhất, bỏ frame cũ (real-time, không backlog)

---

*Generated: 2026-06-18*
