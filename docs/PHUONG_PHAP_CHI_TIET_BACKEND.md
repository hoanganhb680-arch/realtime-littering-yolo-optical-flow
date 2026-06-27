# PHƯƠNG PHÁP XỬ LÝ BACKEND - GIẢI THÍCH CHI TIẾT NGUYÊN LÝ, CƠ SỞ LÝ THUYẾT VÀ CODE

Tài liệu này chỉ tập trung vào **phần phương pháp xử lý chính ở backend**, tức là phần quyết định hệ thống phát hiện, theo dõi, suy luận owner và xác nhận hành vi vứt rác. Phần frontend, API phụ, CSS, giao diện hiển thị không đưa vào phần này để tránh lẫn với phương pháp thuật toán.

Các file thuộc phần phương pháp chính:

| STT | File | Vai trò ngắn gọn |
|---:|---|---|
| 1 | `src/Config.py` | Cấu hình model, nguồn video/camera, FPS, ngưỡng xác nhận vi phạm. |
| 2 | `src/TrashViolationDetector.py` | Pipeline chính: đọc frame, chạy YOLO, tracking, xử lý rác, xác nhận vi phạm. |
| 3 | `src/detector_io.py` | Xử lý đọc/ghi video, resize, giới hạn FPS, stream frame, buffer bằng chứng. |
| 4 | `src/detection_parsing.py` | Parse kết quả YOLO, tách `person` và `trash`, gán ID. |
| 5 | `src/MotionDetector.py` | Phát hiện vùng chuyển động bằng MOG2. |
| 6 | `src/OpticalFlowTracker.py` | Tính hướng chuyển động người bằng optical flow. |
| 7 | `src/OwnershipScorer.py` | Tính điểm người có khả năng là chủ của rác. |
| 8 | `src/owner_resolution.py` | Chọn owner cuối cùng cho vật rác. |
| 9 | `src/trash_lifecycle.py` | Quản lý trạng thái rác: mới xuất hiện, đang theo dõi, mất track, recover. |
| 10 | `src/violation_confirmation.py` | Xác nhận vi phạm và phân loại: `Đột ngột`, `Đứng yên`, `Bỏ rác tại chỗ`. |

---

## 1. Nguyên lý tổng quát của phương pháp

### 1.1. Vì sao không thể phát hiện vi phạm bằng một frame đơn lẻ?

Bài toán phát hiện hành vi vứt rác không giống bài toán phát hiện vật thể thông thường. Nếu chỉ nhìn một ảnh đơn, hệ thống chỉ biết trong ảnh có người và có vật giống rác. Tuy nhiên, từ một ảnh đơn hệ thống chưa thể kết luận:

- Vật rác đó có vừa mới xuất hiện hay đã nằm sẵn từ trước.
- Người nào là người tạo ra vật rác.
- Người đó có tương tác với rác hay chỉ đi ngang qua.
- Rác có nằm yên sau khi người rời đi hay vẫn đang chuyển động.
- Detection rác có phải nhiễu tạm thời, bóng đổ, túi xách, chân người hoặc vật thể khác hay không.

Vì vậy hệ thống dùng phương pháp **suy luận theo chuỗi thời gian**. Video được xem như một chuỗi frame:

```text
Video = frame_1, frame_2, frame_3, ..., frame_n
```

Ở mỗi frame, hệ thống phát hiện người và rác. Sau đó nó theo dõi ID của các đối tượng qua nhiều frame, lưu lịch sử vị trí, tính hướng chuyển động và chỉ xác nhận vi phạm khi các điều kiện theo thời gian đủ tin cậy.

Nói cách khác, phương pháp của hệ thống không phải:

```text
Có người + có rác trong 1 frame => vi phạm
```

mà là:

```text
Rác xuất hiện gần lịch sử chuyển động của một người
+ người đó đủ tin cậy
+ rác tồn tại đủ lâu
+ rác đứng yên đủ lâu
+ owner rời đi hoặc hành vi phù hợp
+ không ambiguous
=> xác nhận vi phạm
```

### 1.2. Các tầng xử lý chính

Pipeline phương pháp gồm 6 tầng:

```text
Tầng 1: Đọc video/camera và chuẩn hóa FPS
Tầng 2: Phát hiện person/trash bằng YOLO
Tầng 3: Tracking ID bằng ByteTrack và matching bổ sung
Tầng 4: Bổ sung tín hiệu chuyển động bằng MOG2 và Optical Flow
Tầng 5: Tính điểm owner cho từng người
Tầng 6: Xác nhận và phân loại vi phạm theo điều kiện thời gian/không gian
```

Mỗi tầng được tách ra thành các file riêng để dễ đọc, dễ kiểm soát logic và dễ giải thích trong báo cáo.

### 1.3. Quy đổi frame sang thời gian

Vì hệ thống xử lý video theo frame, mọi ngưỡng như `CONFIRM_FRAMES`, `HISTORY_FRAMES`, `STATIONARY_REQUIRED` đều phải hiểu theo FPS xử lý.

Công thức:

```text
thời gian (giây) = số frame / FPS xử lý
```

Trong hệ thống, FPS xử lý mục tiêu là khoảng 6 FPS:

```text
6 frame  ≈ 1 giây
12 frame ≈ 2 giây
90 frame ≈ 15 giây
```

Vì vậy:

| Tham số | Giá trị | Ý nghĩa thời gian tại 6 FPS |
|---|---:|---:|
| `CONFIRM_FRAMES` | 12 | Rác cần tồn tại khoảng 2 giây trước khi xác nhận thông thường. |
| `CONFIRM_FRAMES_SUDDEN` | 8 | Tình huống đột ngột cần khoảng 1.33 giây. |
| `MIN_OWNER_GONE_FRAMES` | 6 | Owner phải rời đi tối thiểu khoảng 1 giây. |
| `STATIONARY_REQUIRED` | 10 | Rác phải đứng yên khoảng 1.67 giây. |
| `HISTORY_FRAMES` | 90 | Lưu lịch sử người khoảng 15 giây. |

Cơ sở chọn 6 FPS:

- YOLO là bước nặng nhất, nếu xử lý 25-30 FPS trên CPU sẽ dễ bị trễ.
- 6 FPS vẫn đủ quan sát hành vi người đi, rác xuất hiện và rác đứng yên.
- Các ngưỡng thời gian như 1 giây, 2 giây, 15 giây dễ quy đổi sang frame.
- Real-time ưu tiên độ trễ thấp hơn độ mượt tuyệt đối.

---

## 2. Luồng dữ liệu tổng quát

Luồng xử lý của các file phương pháp có thể mô tả như sau:

```text
Config.py
    -> khai báo model, nguồn video/camera, FPS, ngưỡng

TrashViolationDetector.py
    -> nạp YOLO từ weights/best.pt
    -> mở video file hoặc camera
    -> đọc từng frame
    -> gọi detector_io.py để tiền xử lý frame
    -> chạy YOLO + ByteTrack
    -> gọi detection_parsing.py để tách person/trash
    -> gọi MotionDetector.py để lấy vùng chuyển động
    -> gọi OpticalFlowTracker.py để lấy vector chuyển động
    -> gọi trash_lifecycle.py để quản lý rác
        -> owner_resolution.py chọn owner
            -> OwnershipScorer.py tính điểm owner
        -> violation_confirmation.py xác nhận loại vi phạm
```

Đầu vào chính khi chạy thật:

```text
weights/best.pt
video/di_bo_17.mp4 hoặc camera IP
```

Đầu ra của phần phương pháp:

```text
frame annotated
trạng thái rác
owner_id
violation_type
score
ảnh/video bằng chứng
alert vi phạm
```

---

## 3. `src/Config.py` - Cấu hình và cơ sở chọn tham số

### 3.1. Vai trò của file

`Config.py` là file cấu hình trung tâm. Tất cả thông số ảnh hưởng đến thuật toán đều được gom vào đây, gồm:

- Nguồn video/camera.
- Đường dẫn model YOLO.
- FPS xử lý.
- Kích thước ảnh đưa vào YOLO.
- Ngưỡng confidence, IoU.
- Ngưỡng tracking người/rác.
- Ngưỡng xác nhận vi phạm.
- Thông số MOG2.
- Thông số optical flow.
- Thông số lưu bằng chứng.

Việc đặt toàn bộ config trong một file giúp hệ thống có tính nhất quán. Khi báo cáo hoặc đánh giá mô hình, ta có thể chỉ ra rõ mọi quyết định thuật toán đều dựa trên các tham số cụ thể, không nằm rải rác trong nhiều file.

### 3.2. Nguyên lý cấu hình nguồn video

Hệ thống hỗ trợ hai chế độ:

```python
SOURCE_MODE = "ip_camera"  # "file" hoặc "ip_camera"
```

Nếu chạy video file:

```python
VIDEO_FILE = video/di_bo_17.mp4
```

Nếu chạy camera IP:

```python
IP_CAM_HOST = "192.168.0.108"
IP_CAM_PORT = 8080
IP_CAM_PROTOCOL = "mjpeg"
```

Trong `__init__()`, file này tự dựng `VIDEO_SOURCE`:

```text
SOURCE_MODE = file      -> VIDEO_SOURCE = VIDEO_FILE
SOURCE_MODE = ip_camera -> VIDEO_SOURCE = http://IP:PORT/video hoặc rtsp://...
```

Cơ sở lý thuyết:

- Video file dùng để kiểm thử ổn định, có thể chạy lại nhiều lần với cùng dữ liệu.
- Camera IP dùng cho bài toán real-time, dữ liệu đến liên tục và có độ trễ mạng.
- Hai chế độ này vẫn đi qua cùng một pipeline, giúp thuật toán không bị tách đôi.

### 3.3. Cấu hình model

```python
MODEL_PATH = weights/best.pt
```

`weights/best.pt` là model YOLO đã huấn luyện. Khi chạy, hệ thống không đọc toàn bộ dataset nữa. Dataset chỉ dùng ở giai đoạn train. Runtime chỉ cần model `.pt` và nguồn video/camera.

Ý nghĩa:

- Model đã học đặc trưng của 2 class: `person`, `trash`.
- Backend nạp model một lần khi detector bắt đầu.
- Mỗi frame sau đó được đưa qua model để nhận bounding box.

### 3.4. Cấu hình FPS

Các thông số:

```python
FILE_MODE_FPS = 6
LIVE_TARGET_FPS = 6.0
STREAM_TARGET_FPS = 6.0
```

Cơ sở lý thuyết:

- FPS cao giúp video mượt hơn nhưng chi phí tính toán tăng tuyến tính.
- Nếu xử lý 30 FPS, số lần YOLO chạy mỗi giây gấp 5 lần so với 6 FPS.
- Với CPU, 30 FPS dễ gây backlog. Backlog làm hệ thống phân tích frame cũ, không còn real-time.
- 6 FPS là mức cân bằng: vẫn đủ thông tin hành vi, nhưng giảm tải đáng kể.

Ví dụ video gốc 30 FPS:

```text
source_fps = 30
target_fps = 6
frame_step = round(30 / 6) = 5
```

Nghĩa là:

```text
lấy 1 frame, bỏ 4 frame
```

### 3.5. Cấu hình YOLO

Các thông số live:

```python
LIVE_YOLO_IMGSZ = 512
LIVE_YOLO_CONF = 0.12
LIVE_YOLO_IOU = 0.50
```

Giải thích:

- `imgsz`: kích thước ảnh đưa vào YOLO. Ảnh lớn hơn có thể chính xác hơn nhưng chậm hơn.
- `conf`: ngưỡng confidence. Ngưỡng thấp giúp bắt rác nhỏ tốt hơn, nhưng có thể tăng false positive.
- `iou`: ngưỡng Intersection over Union cho NMS, dùng để loại box trùng nhau.

Vì rác thường nhỏ, dễ bị che và confidence thấp, hệ thống dùng `LIVE_YOLO_CONF = 0.12`. Tuy nhiên để tránh báo sai, hệ thống không xác nhận ngay theo một frame, mà yêu cầu thêm điều kiện thời gian trong `violation_confirmation.py`.

### 3.6. Cấu hình ngưỡng thời gian và tracking

Các thông số quan trọng:

```python
HISTORY_FRAMES = 90
CONFIRM_FRAMES = 12
CONFIRM_FRAMES_SUDDEN = 8
OWNER_REEVAL_FRAMES = 90
STATIONARY_REQUIRED = 10
STALE_FRAMES = 90
```

Ý nghĩa:

- `HISTORY_FRAMES`: lưu lịch sử vị trí người để truy ngược owner.
- `CONFIRM_FRAMES`: số frame rác phải tồn tại trước khi xác nhận.
- `CONFIRM_FRAMES_SUDDEN`: ngưỡng nhanh hơn cho tình huống đột ngột.
- `OWNER_REEVAL_FRAMES`: khoảng thời gian cho phép đánh giá lại owner nếu ban đầu chưa chắc.
- `STATIONARY_REQUIRED`: số frame rác phải đứng yên.
- `STALE_FRAMES`: xóa rác nếu mất quá lâu.

Cơ sở:

- Rác nhỏ có thể mất detection tạm thời, nên cần giữ trạng thái nhiều frame.
- Owner có thể rời khỏi khung hình trước khi rác được phát hiện rõ, nên cần lịch sử người.
- False positive thường xuất hiện ít frame, nên yêu cầu tồn tại nhiều frame giúp giảm báo sai.

### 3.7. Cấu hình ngưỡng không gian

Các thông số:

```python
STATIONARY_PX = 16
SPAWN_RADIUS = 320
TRAJECTORY_RADIUS = 380
RECENT_OWNER_RADIUS = 520
TRASH_ID_MATCH_RADIUS = 90
PERSON_ID_MATCH_RADIUS = 180
VIOLATION_GROUND_MIN_Y_RATIO = 0.58
```

Giải thích:

- `STATIONARY_PX = 16`: cho phép bbox rác dao động nhẹ nhưng vẫn coi là đứng yên.
- `SPAWN_RADIUS = 320`: bán kính tìm owner khi rác mới xuất hiện.
- `TRAJECTORY_RADIUS = 380`: bán kính xét lịch sử quỹ đạo người.
- `RECENT_OWNER_RADIUS = 520`: bán kính fallback khi owner đã rời đi.
- `VIOLATION_GROUND_MIN_Y_RATIO = 0.58`: chỉ xét rác ở vùng dưới ảnh, gần mặt đất.

Cơ sở:

- Bounding box của vật nhỏ thường jitter vài pixel.
- Camera đặt nghiêng khiến khoảng cách ảnh không đúng hoàn toàn với khoảng cách thật.
- Rác bị vứt thường nằm ở phần dưới khung hình, không phải ở trên cao.

---

## 4. `src/TrashViolationDetector.py` - Pipeline xử lý chính

### 4.1. Vai trò của file

`TrashViolationDetector.py` là file điều phối toàn bộ quá trình xử lý. File này không nên chứa quá nhiều thuật toán chi tiết, mà đóng vai trò gọi các module chuyên trách.

Luồng chính trong `run()`:

```text
1. Đọc config
2. Kiểm tra đường dẫn model/video
3. Nạp YOLO
4. Tạo tham số tracking
5. Mở video/camera
6. Tính FPS và frame stride
7. Lặp qua từng frame
8. Tiền xử lý frame
9. Chạy detect + tracking
10. Xử lý rác/owner/vi phạm
11. Stream frame và lưu output
```

### 4.2. Nạp model YOLO

Trong code:

```python
model = YOLO(cfg.MODEL_PATH)
```

Nguyên lý:

- YOLO là mô hình object detection một giai đoạn.
- Mỗi frame được đưa vào model.
- Model trả về bounding box, class, confidence.
- Kết hợp `model.track()` với ByteTrack để có ID qua nhiều frame.

Tại runtime, hệ thống chỉ dùng `weights/best.pt`. Dataset train không được đọc lại.

### 4.3. Mở nguồn video/camera

Code gọi:

```python
cap = self._open_capture(is_live)
```

Nếu `is_live = False`:

```python
cap = cv2.VideoCapture(self.cfg.VIDEO_SOURCE)
```

Nếu `is_live = True`:

```python
cap = ThreadedCamera(self.cfg.VIDEO_SOURCE, self.cfg.CAMERA_BUFFER).start()
```

Cơ sở:

- Video file có thể đọc tuần tự bằng OpenCV.
- Camera real-time cần thread riêng để tránh bị trễ do buffer mạng.

### 4.4. Tính stride cho video file

Code:

```python
file_frame_step = self._file_frame_step(source_fps, is_live)
```

Nếu không phải live và `FILE_FRAME_STRIDE = 0`, hệ thống tự tính:

```python
round(source_fps / FILE_MODE_FPS)
```

Ví dụ:

```text
30 FPS / 6 FPS = 5
```

Ý nghĩa:

- Video gốc vẫn có thể là 25-30 FPS.
- Hệ thống chỉ xử lý khoảng 6 FPS.
- Các ngưỡng frame giữ đúng ý nghĩa thời gian.

### 4.5. Vòng lặp xử lý frame

Trong mỗi vòng lặp:

```text
ret, frame = cap.read()
frame = self._prepare_frame(frame)
curr_gray, annotated = self._process_frame(...)
self._push_frame(annotated)
```

Ý nghĩa:

- `cap.read()` lấy frame mới.
- `_prepare_frame()` xoay/resize nếu cần.
- `_process_frame()` chạy toàn bộ logic AI.
- `_push_frame()` gửi frame đã vẽ lên frontend.

### 4.6. `_process_frame()` là lõi xử lý

Luồng trong `_process_frame()`:

```text
1. Chuyển frame sang grayscale
2. Chạy MOG2 để lấy motion alerts
3. Chạy YOLO + ByteTrack
4. Parse person/trash
5. Bổ sung person từ motion nếu YOLO bỏ sót
6. Lưu frame bằng chứng người
7. Chạy floor-pass để bắt rác nhỏ vùng mặt đất
8. Cập nhật optical flow
9. Xử lý vòng đời rác
10. Dọn rác stale
11. Vẽ HUD
12. Lưu frame vào evidence buffer
```

Đây là lý do `TrashViolationDetector.py` được xem là file pipeline chính.

---

## 5. `src/detector_io.py` - I/O, tối ưu tốc độ và bằng chứng

### 5.1. Vai trò của file

`detector_io.py` gom các hàm không trực tiếp quyết định vi phạm nhưng rất quan trọng để hệ thống chạy ổn định:

- Tạo tham số YOLO.
- Chọn CPU/GPU.
- Giới hạn FPS.
- Resize frame xử lý.
- Encode JPEG để stream.
- Lưu buffer clip bằng chứng.
- Ghi video kết quả.

### 5.2. Cơ sở lý thuyết: tối ưu real-time

Trong xử lý real-time, có hai yêu cầu đối nghịch:

```text
Muốn ảnh lớn, FPS cao -> chính xác/mượt hơn
Muốn ảnh nhỏ, FPS thấp -> nhanh hơn, ít trễ hơn
```

Hệ thống chọn hướng cân bằng:

```text
xử lý khoảng 6 FPS
resize cạnh dài live <= 960
stream preview <= 720p
JPEG quality = 50
```

Mục tiêu là không để detector chạy sau camera quá xa.

### 5.3. Tạo tham số YOLO/ByteTrack

Trong `_build_track_kwargs()`:

```python
kwargs = {
    "imgsz": cfg.LIVE_YOLO_IMGSZ if is_live else 640,
    "conf": cfg.LIVE_YOLO_CONF if is_live else 0.25,
    "iou": cfg.LIVE_YOLO_IOU,
    "persist": True,
    "tracker": "bytetrack.yaml",
    "verbose": False,
}
```

Giải thích:

- `persist=True`: giữ tracker qua nhiều frame.
- `tracker="bytetrack.yaml"`: dùng ByteTrack.
- Live mode dùng `imgsz=512`, `conf=0.12` để tăng tốc và tăng khả năng bắt rác nhỏ.
- File mode dùng `imgsz=640`, `conf=0.25` vì video file ổn định hơn và không gấp về latency như live.

### 5.4. Floor-pass cho rác nhỏ

Trong `_build_floor_trash_kwargs()`:

```python
classes = [1]
conf = LIVE_FLOOR_TRASH_CONF
imgsz = LIVE_FLOOR_YOLO_IMGSZ
```

Ý nghĩa:

- Chỉ detect class `trash`.
- Chạy trên vùng dưới ảnh.
- Confidence thấp hơn để không bỏ sót rác nhỏ.

Cơ sở:

- Rác thường nằm ở mặt đất.
- Rác nhỏ có confidence thấp trong full-frame.
- Chạy thêm một lượt chỉ ở ROI giúp tăng recall mà không tăng tải quá nhiều.

### 5.5. Giới hạn FPS live

`_pace_live_loop()` đảm bảo khoảng cách giữa hai lần xử lý không nhỏ hơn:

```text
1 / LIVE_TARGET_FPS
```

Với `LIVE_TARGET_FPS = 6`:

```text
mỗi frame xử lý cách nhau khoảng 0.167 giây
```

Điều này tránh việc hệ thống cố xử lý quá nhanh khi không cần thiết, đồng thời ổn định thời gian giữa các frame.

### 5.6. Resize frame xử lý

`_resize_for_processing()` giới hạn cạnh dài:

```python
LIVE_PROCESS_MAX_SIDE = 960
```

Cơ sở:

- Độ phức tạp xử lý ảnh phụ thuộc vào số pixel.
- Resize giảm số pixel, YOLO chạy nhanh hơn.
- Cạnh dài 960 vẫn đủ để thấy người/rác trong camera gần.

### 5.7. Stream frame lên frontend

`_push_frame()` encode ảnh thành JPEG:

```python
cv2.imencode(".jpg", stream_frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
```

Hệ thống dùng queue size 1 ở `stream_router.py`, nên nếu frame cũ chưa gửi kịp thì frame mới sẽ thay thế. Đây là nguyên lý thường dùng trong live stream: ưu tiên frame mới nhất thay vì giữ tất cả frame.

### 5.8. Evidence buffer

`_remember_evidence_frame()` lưu các frame gần đây vào buffer:

```python
self._evidence_clip_buffer.append((now, buf.tobytes()))
```

`_trim_evidence_clip_buffer()` giới hạn buffer theo:

```python
EVIDENCE_CLIP_SECONDS
EVIDENCE_CLIP_MAX_FPS
```

Ý nghĩa:

- Khi vi phạm xảy ra, hệ thống có thể tạo clip chứa cả trước và sau thời điểm xác nhận.
- Không cần lưu toàn bộ video dài.
- Buffer giới hạn giúp tiết kiệm RAM.

---

## 6. `src/detection_parsing.py` - Parse detection, gán ID và tạo dữ liệu logic

### 6.1. Vai trò của file

YOLO trả về dữ liệu dạng model output. File này chuyển output đó thành dữ liệu dễ dùng cho suy luận hành vi:

```python
current_persons: dict[int, tuple[int, int]]
current_trashes: dict[int, tuple[int, int]]
```

Trong đó:

- Key là ID.
- Value là điểm đại diện trên ảnh.

### 6.2. Parse class `person` và `trash`

Trong `_parse_detections()`:

```python
if int(cls) == 0:
    # person
elif int(cls) == 1:
    # trash
```

Ý nghĩa:

- Class 0 là người.
- Class 1 là rác.

YOLO chỉ phát hiện bbox. Nhưng hệ thống cần điểm đại diện để tính khoảng cách. Vì vậy code tính:

```python
cx = int((x1 + x2) / 2)
cy = int((y1 + y2) / 2)
```

Với rác, tâm bbox là hợp lý vì rác là vật nhỏ.

### 6.3. Vì sao người dùng điểm gần chân?

Với người, file không dùng tâm bbox đơn giản. Nó gọi:

```python
anchor, points = self._person_points_from_box(box, results[0].orig_shape)
```

Cơ sở:

- Hành vi vứt rác diễn ra gần mặt đất.
- Rác thường nằm gần chân hoặc vùng thấp của người.
- Tâm bbox người nằm ở thân trên, không phản ánh vị trí tương tác với rác.
- Nếu dùng tâm người, khoảng cách người-rác bị phóng đại, nhất là khi người đứng thẳng.

Vì vậy điểm anchor chính là vùng gần chân:

```text
lower point gần y2 của bbox
```

Ngoài anchor, file còn tạo nhiều điểm phụ quanh phần dưới bbox để tăng khả năng match với rác trong camera góc nghiêng.

### 6.4. Lọc bbox người không hợp lệ

`_is_valid_person_box()` lọc người dựa trên:

```python
LIVE_PERSON_CONF
LIVE_PERSON_MIN_HEIGHT_RATIO
LIVE_PERSON_MIN_AREA_RATIO
```

Cơ sở:

- Detection người quá nhỏ hoặc confidence quá thấp dễ là nhiễu.
- Person false positive có thể làm owner scoring sai.
- Cần lọc người trước khi đưa vào lịch sử.

### 6.5. Lưu lịch sử người

Khi có person hợp lệ:

```python
self._person_history[oid].append(
    self._make_person_history_entry(anchor, points, frame_idx)
)
```

Lịch sử gồm:

```text
anchor
points
frame index
```

Lịch sử này dùng cho:

- Tính owner theo khoảng cách quá khứ.
- Tìm người vừa đi gần rác.
- Tính recency.
- Recover owner khi người đã rời khung hình.

### 6.6. Gán lại ID người khi ByteTrack mất ID

ByteTrack có thể mất ID trong các trường hợp:

- Người bị che khuất.
- Camera rung.
- Bounding box mất trong vài frame.
- Detection confidence thay đổi.

`_resolve_person_id()` match người mới với lịch sử cũ nếu khoảng cách nhỏ hơn:

```python
PERSON_ID_MATCH_RADIUS
```

Nguyên lý:

```text
Nếu điểm chân người mới gần điểm chân cuối cùng của một ID cũ
và ID đó chưa quá stale
=> tiếp tục dùng ID cũ
```

Nếu không match được và raw ID từ tracker không có, hệ thống tạo synthetic ID:

```python
self._next_synthetic_person_id
```

### 6.7. Gán lại ID rác

`_resolve_trash_id()` tương tự, nhưng dùng:

```python
TRASH_ID_MATCH_RADIUS
```

Rác nhỏ thường jitter hoặc mất tracking ID. Nếu rác ở gần vị trí cũ, hệ thống vẫn coi là cùng một rác.

### 6.8. Bổ sung người từ motion

`_augment_persons_from_motion()` dùng MOG2 alerts để tạo person fallback.

Cơ sở:

- YOLO đôi khi bỏ sót người trong camera live.
- Nhưng MOG2 vẫn phát hiện vùng chuyển động lớn.
- Nếu vùng motion đủ lớn, nằm trong vùng hợp lý và không trùng với person/trash hiện có, hệ thống tạo một person giả.

Điều này giúp không mất toàn bộ owner history khi YOLO bị miss vài frame.

### 6.9. Floor-pass phát hiện rác vùng mặt đất

`_detect_floor_trash_candidates()` lấy ROI dưới ảnh:

```python
y0 = int(h * LIVE_FLOOR_ROI_TOP)
roi = frame[y0:h, :]
```

Sau đó chỉ detect class rác trong ROI.

Cơ sở:

- Rác cần xét thường nằm trên mặt đất.
- Vùng dưới ảnh chứa thông tin sàn/đường.
- Chạy thêm detection trên ROI giúp rác nhỏ được phóng tương đối lớn hơn trong đầu vào YOLO.

File cũng yêu cầu context:

```text
gần người hiện tại
hoặc gần lịch sử người
hoặc gần motion alert
```

Điều này tránh nhận mọi vật nhỏ dưới đất là rác vi phạm.

---

## 7. `src/MotionDetector.py` - Phát hiện chuyển động bằng MOG2

### 7.1. Vai trò của file

`MotionDetector.py` phát hiện các vùng chuyển động trong frame. Kết quả không phải vi phạm trực tiếp mà là tín hiệu phụ giúp:

- Bổ sung người khi YOLO bỏ sót.
- Tăng độ tin cậy owner nếu có motion gần rác.
- Hỗ trợ recover rác mất track.

### 7.2. Cơ sở lý thuyết MOG2

MOG2 là viết tắt của Gaussian Mixture-based Background/Foreground Segmentation Algorithm. Ý tưởng:

- Mỗi pixel được mô hình hóa bằng một hỗn hợp Gaussian theo thời gian.
- Các giá trị pixel xuất hiện thường xuyên được coi là nền.
- Các giá trị mới khác nền được coi là foreground.

Trong video giám sát, foreground thường là người hoặc vật đang chuyển động.

### 7.3. Code khởi tạo

```python
cv2.createBackgroundSubtractorMOG2(
    history=history,
    varThreshold=threshold,
    detectShadows=False,
)
```

Ý nghĩa:

- `history`: số frame dùng để học nền.
- `varThreshold`: ngưỡng khác biệt để pixel được coi là foreground.
- `detectShadows=False`: tắt detect bóng để giảm nhiễu trong bài toán này.

### 7.4. Morphology lọc nhiễu

Sau khi lấy foreground mask, file dùng:

```python
cv2.morphologyEx(fg, cv2.MORPH_OPEN, kernel)
cv2.morphologyEx(fg, cv2.MORPH_CLOSE, kernel)
```

Giải thích:

- Opening giúp loại các điểm nhiễu nhỏ.
- Closing giúp lấp lỗ nhỏ trong vùng chuyển động.
- Kernel ellipse 5x5 phù hợp với vùng chuyển động mềm như người.

### 7.5. Tìm contour và lọc diện tích

File tìm contour:

```python
contours, _ = cv2.findContours(...)
```

Sau đó lọc:

```python
if area < self.min_area:
    continue
```

Lý do:

- Nhiễu camera, ánh sáng, bóng nhỏ có thể tạo vùng foreground nhỏ.
- Chỉ giữ vùng chuyển động đủ lớn giúp giảm false positive.

Kết quả trả về:

```python
(cx, cy, area)
```

Đây là vị trí và kích thước vùng chuyển động.

---

## 8. `src/OpticalFlowTracker.py` - Tính hướng chuyển động bằng Lucas-Kanade

### 8.1. Vai trò của file

`OpticalFlowTracker.py` tính vector chuyển động của từng người qua hai frame liên tiếp. Vector này là một phần trong điểm owner.

### 8.2. Cơ sở lý thuyết Optical Flow

Optical flow dựa trên giả định brightness constancy:

```text
độ sáng của một điểm ảnh gần như không đổi giữa hai frame gần nhau
```

Nếu một điểm ảnh di chuyển từ vị trí `(x, y)` sang `(x + dx, y + dy)`, optical flow ước lượng vector:

```text
(vx, vy) = (dx, dy)
```

Trong hệ thống, điểm cần theo dõi là điểm đại diện của người.

### 8.3. Lucas-Kanade

File dùng:

```python
cv2.calcOpticalFlowPyrLK(...)
```

Lucas-Kanade giả định chuyển động trong một cửa sổ nhỏ là gần như đồng nhất. Vì vậy nó tìm vector dịch chuyển tốt nhất cho nhóm pixel trong cửa sổ.

Tham số trong config:

```python
winSize = (15, 15)
maxLevel = 2
criteria = (...)
```

Giải thích:

- `winSize=(15,15)`: cửa sổ đủ lớn để bắt chuyển động cục bộ nhưng không quá rộng.
- `maxLevel=2`: dùng pyramid nhiều mức, giúp theo dõi chuyển động lớn hơn giữa hai frame.
- `criteria`: điều kiện dừng lặp tối ưu.

### 8.4. Lưu vector theo từng người

File lưu:

```python
flow_vecs[p_id] = deque(maxlen=FLOW_HISTORY_FRAMES)
```

Với `FLOW_HISTORY_FRAMES = 8`, ở 6 FPS tương đương khoảng 1.33 giây. Đây là khoảng đủ để biết hướng chuyển động gần đây của người.

### 8.5. Vì sao optical flow hữu ích cho phát hiện vứt rác?

Nếu một người đứng gần vị trí rác, sau đó vector chuyển động của người hướng ra xa rác, thì hành vi đó phù hợp với mẫu:

```text
người bỏ rác -> rời khỏi vị trí rác
```

Vì vậy `flow_score` trong `OwnershipScorer.py` tăng điểm cho người có hướng optical flow rời xa rác.

---

## 9. `src/OwnershipScorer.py` - Tính điểm owner

### 9.1. Vai trò của file

File này trả lời câu hỏi:

```text
Trong các người từng xuất hiện, ai có khả năng là người tạo ra vật rác?
```

Đây là bước rất quan trọng vì phát hiện rác chưa đủ để kết luận vi phạm. Cần gắn rác với một người cụ thể.

### 9.2. Vì sao cần scoring nhiều tín hiệu?

Nếu chỉ dùng khoảng cách frame hiện tại:

- Người đã rời đi sẽ không được chọn.
- Người đứng gần nhưng không liên quan có thể bị chọn sai.
- Rác bị phát hiện muộn sẽ mất owner thật.

Nếu chỉ dùng hướng di chuyển:

- Người đi ngang qua có thể trông giống như đang rời xa rác.
- Optical flow có thể nhiễu.

Vì vậy file kết hợp nhiều tín hiệu.

### 9.3. Công thức điểm owner

Trong code:

```python
score = (
    0.40 * proximity_score
    + 0.30 * direction_score
    + 0.20 * flow_score
    + 0.10 * recency_score
)
```

Ý nghĩa:

| Thành phần | Trọng số | Cơ sở |
|---|---:|---|
| `proximity_score` | 0.40 | Người từng gần rác là tín hiệu mạnh nhất. |
| `direction_score` | 0.30 | Người rời xa rác phù hợp hành vi bỏ rác. |
| `flow_score` | 0.20 | Optical flow bổ sung hướng chuyển động ngắn hạn. |
| `recency_score` | 0.10 | Người vừa gần rác gần đây đáng tin hơn người đã đi qua lâu. |

### 9.4. `proximity_score`

File tìm khoảng cách nhỏ nhất giữa rác và lịch sử vị trí người:

```text
min_dist = khoảng cách gần nhất giữa person history và trash center
```

Sau đó:

```python
proximity_score = 1.0 - (min_dist / trajectory_radius)
```

Nếu người càng gần rác, `min_dist` càng nhỏ, score càng cao.

Nếu `min_dist > TRAJECTORY_RADIUS`, score bằng 0 vì người quá xa để coi là liên quan.

### 9.5. `direction_score`

File lấy vị trí gần đây và vị trí giữa lịch sử để tạo vector chuyển động:

```text
move = recent_position - mid_position
```

Sau đó so sánh với vector từ rác tới người hiện tại:

```text
away = recent_position - trash_position
```

Nếu hai vector cùng hướng, nghĩa là người đang đi xa khỏi rác.

Code dùng cosine similarity:

```text
dot(move_normalized, away_normalized)
```

Giá trị càng gần 1 thì hướng càng giống nhau.

### 9.6. `flow_score`

`flow_score` dùng vector optical flow trung bình gần đây:

```text
avg_flow = mean(last flow vectors)
```

Sau đó so sánh hướng flow với hướng rời xa rác. Nếu flow chỉ ra rằng người đang di chuyển ra xa rác, score tăng.

### 9.7. `recency_score`

Nếu người gần rác ở frame rất cũ, khả năng liên quan thấp. Nếu người gần rác ngay trước khi rác xuất hiện, khả năng liên quan cao.

Code:

```python
recency_score = 1.0 - (closest_offset / HISTORY_FRAMES)
```

Trong đó `closest_offset` là khoảng cách frame giữa thời điểm hiện tại và frame người gần rác nhất.

### 9.8. Ambiguous owner

Nếu hai người có score gần nhau:

```python
best_score - second_score < AMBIGUOUS_MARGIN
```

hệ thống đánh dấu ambiguous.

Cơ sở:

- Khi nhiều người đứng gần nhau, gán sai owner là lỗi nghiêm trọng.
- Tốt hơn là không xác nhận hoặc chờ thêm frame thay vì kết luận vội.

---

## 10. `src/owner_resolution.py` - Chọn owner cuối cùng và lưu bằng chứng owner

### 10.1. Vai trò của file

`OwnershipScorer.py` chỉ tính điểm. `owner_resolution.py` quyết định điểm đó có đủ đáng tin để dùng hay không.

Nó làm 4 việc chính:

```text
1. Kiểm tra owner có usable không
2. Gọi scorer để tìm owner tốt nhất
3. Dùng fallback recent-owner nếu owner hiện tại không đủ
4. Lưu frame bằng chứng của owner
```

### 10.2. Owner usable là gì?

Trong `_owner_is_usable()`:

```text
owner phải có ID
owner phải xuất hiện đủ frame
owner phải có chuyển động tối thiểu
```

Các thông số:

```python
MIN_OWNER_SEEN_FRAMES = 2
MIN_OWNER_MOTION_PX = 5
```

Cơ sở:

- Detection 1 frame có thể là nhiễu.
- Người thật trong video thường có chuyển động.
- Nếu owner không có chuyển động, có thể là false positive hoặc vật bị nhận nhầm.

### 10.3. Fallback recent-owner

Trường hợp thường gặp:

```text
người bỏ rác
-> rác bị người che
-> người bước đi
-> rác mới hiện rõ
```

Nếu chỉ tìm owner trong frame hiện tại, hệ thống sẽ bỏ lỡ người thật. Vì vậy `_fallback_recent_owner()` tìm trong lịch sử người gần vị trí rác.

Score fallback:

```python
score = 0.70 * distance_score + 0.30 * recency_score
```

Giải thích:

- `distance_score` quan trọng hơn vì owner phải từng gần vị trí rác.
- `recency_score` bổ sung yếu tố thời gian.

### 10.4. Cập nhật owner

`_apply_owner_update()` cập nhật:

```text
owner_id
score
is_ambiguous
owner_seen_count
owner_frame_jpg
```

Nếu owner đang còn trong scene, `owner_left_frame = None`. Nếu owner đã rời scene, hệ thống ghi lại frame owner rời đi.

### 10.5. Lưu frame bằng chứng owner

`_remember_person_evidence()` lưu frame JPEG gần nhất của mỗi người.

Cơ sở:

- Khi vi phạm được xác nhận, owner có thể đã rời khỏi khung hình.
- Nếu không lưu trước, ảnh bằng chứng chỉ thấy rác mà không thấy người.
- Lưu frame owner giúp bằng chứng rõ hơn.

---

## 11. `src/trash_lifecycle.py` - Quản lý vòng đời rác

### 11.1. Vai trò của file

File này quản lý từng vật rác từ lúc mới xuất hiện đến khi:

```text
được xác nhận vi phạm
hoặc bị xóa vì stale
hoặc được recover sau khi mất detection
```

Trạng thái rác được lưu trong:

```python
self._trash_registry
```

### 11.2. Vì sao cần vòng đời rác?

Rác là vật nhỏ và khó phát hiện ổn định. Nếu hệ thống chỉ xử lý từng frame độc lập:

- Rác mất detection 1 frame sẽ bị coi là vật mới.
- Vi phạm có thể bị báo trùng.
- Không biết rác đã tồn tại bao lâu.
- Không biết rác có đứng yên hay không.

Vì vậy cần lưu trạng thái rác qua thời gian.

### 11.3. Khi rác mới xuất hiện

`_register_new_trash()` gọi:

```python
owner_id, score, is_ambig = self._find_owner_for_trash(...)
```

Sau đó tạo record:

```text
owner_id
score
is_ambiguous
spawn_frame
spawn_time
confirm_ctr
stationary_ctr
last_pos
status = pending
```

Ý nghĩa:

- `spawn_frame`: frame rác xuất hiện.
- `confirm_ctr`: số frame rác đã được theo dõi.
- `stationary_ctr`: số frame rác gần như đứng yên.
- `owner_id`: người liên quan.
- `score`: độ tin cậy owner.
- `status`: pending hoặc confirmed.

### 11.4. Cập nhật rác qua mỗi frame

Khi rác đã có trong registry:

```python
data["confirm_ctr"] += 1
data["seen_count"] += 1
```

Sau đó kiểm tra rác có đứng yên:

```python
disp = distance(current_pos, last_pos)
is_stationary_now = disp < STATIONARY_PX
```

Nếu đứng yên:

```python
stationary_ctr += 1
```

Nếu di chuyển:

```python
stationary_ctr = 0
```

### 11.5. Cơ sở của `STATIONARY_PX`

`STATIONARY_PX = 16` không có nghĩa là rác được phép di chuyển thật 16 pixel. Nó chủ yếu dùng để hấp thụ jitter của bbox.

Rác nhỏ thường có bbox dao động do:

- YOLO không bắt chính xác cùng một biên hộp ở mỗi frame.
- Camera rung nhẹ.
- Motion blur.
- Rác bị che một phần bởi chân người.

Nếu đặt ngưỡng quá nhỏ, rác thật sự đứng yên cũng bị coi là di chuyển. Nếu đặt quá lớn, vật đang di chuyển có thể bị coi là đứng yên. Vì vậy 16 pixel là mức cân bằng với frame live đã resize cạnh dài khoảng 960 pixel.

### 11.6. Đánh giá lại owner

Trong các frame đầu, owner có thể chưa rõ. File này cho phép reevaluate owner nếu:

```text
owner_id chưa có
hoặc ambiguous
hoặc score thấp
hoặc owner không usable
hoặc owner đã rời scene
```

Cơ sở:

- Frame đầu rác mới xuất hiện có thể bị che.
- Người có thể chưa được YOLO detect đủ ổn định.
- Cần chờ thêm lịch sử để chọn owner chính xác hơn.

### 11.7. Recover rác mất track

`_recover_recently_lost_trashes()` xử lý trường hợp rác bị YOLO bỏ sót tạm thời.

Điều kiện:

```python
LOST_TRASH_RECOVERY_FRAMES = 60
LOST_TRASH_MIN_SEEN = 2
```

Ở 6 FPS:

```text
60 frame ≈ 10 giây
```

Nếu rác từng được thấy ít nhất 2 frame và mất không quá 10 giây, hệ thống thử recover.

Cách recover:

```text
1. Snap về điểm gần chân owner nếu hợp lý
2. Snap về motion alert gần mặt đất
3. Giữ vị trí cuối nếu vẫn nằm vùng mặt đất
```

Cơ sở:

- Rác nhỏ dễ mất detection.
- Không nên xóa rác ngay khi mất 1-2 frame.
- Nhưng cũng không giữ mãi, vì sẽ tạo false positive.

---

## 12. `src/violation_confirmation.py` - Xác nhận và phân loại vi phạm

### 12.1. Vai trò của file

Đây là file đưa ra quyết định cuối cùng:

```text
rác pending có trở thành vi phạm hay không?
nếu có thì thuộc loại nào?
```

Nó không phát hiện object, không tracking, không tính owner. Nó chỉ dùng trạng thái đã có để xác nhận hành vi.

### 12.2. Điều kiện chung trước khi xác nhận

Trong `_confirmation_status()`, hệ thống tạo điều kiện chung:

```text
common_ok =
    score >= MIN_SCORE
    owner_seen_enough
    not ambiguous
    ground_condition
```

Giải thích:

- `score >= MIN_SCORE`: owner phải có liên hệ tối thiểu với rác.
- `owner_seen_enough`: owner phải xuất hiện đủ frame, tránh detection ảo.
- `not ambiguous`: không có nhiều người điểm gần nhau.
- `ground_condition`: rác phải nằm ở vùng mặt đất.

### 12.3. Điều kiện mặt đất

`_is_ground_trash()` kiểm tra:

```python
t_center[1] >= frame_height * VIOLATION_GROUND_MIN_Y_RATIO
```

Với:

```python
VIOLATION_GROUND_MIN_Y_RATIO = 0.58
```

Ý nghĩa:

- Chỉ xét rác ở khoảng 42% vùng dưới ảnh.
- Tránh nhầm vật trên tay, túi xách, đồ trên người là rác đã vứt xuống đất.

Cơ sở:

- Hành vi vứt rác tạo ra vật nằm trên nền/sàn/đường.
- Trong ảnh camera, mặt đất thường nằm ở phần dưới khung hình.

### 12.4. Owner rời đi

Owner được coi là thật sự rời đi nếu:

```python
frames_owner_gone >= MIN_OWNER_GONE_FRAMES
```

Với:

```python
MIN_OWNER_GONE_FRAMES = 6
```

Ở 6 FPS:

```text
6 frame ≈ 1 giây
```

Lý do:

- Nếu owner mất khỏi detection 1-2 frame, có thể chỉ là YOLO miss.
- Chờ khoảng 1 giây giúp chắc hơn rằng owner đã rời khỏi vùng rác hoặc không còn được detect ổn định.

### 12.5. Loại vi phạm `Đột ngột`

Điều kiện:

```text
common_ok
owner_truly_gone
frames_since_spawn <= CONFIRM_FRAMES_SUDDEN + MIN_OWNER_GONE_FRAMES
confirm_ctr >= CONFIRM_FRAMES_SUDDEN
```

Ý nghĩa:

- Rác xuất hiện và được xác nhận khá nhanh.
- Owner đã rời đi đủ tối thiểu.
- Dùng cho tình huống người thả/vứt rác nhanh rồi đi.

Với 6 FPS:

```text
CONFIRM_FRAMES_SUDDEN = 8  -> 1.33 giây
MIN_OWNER_GONE_FRAMES = 6  -> 1 giây
tổng cửa sổ khoảng 2.33 giây
```

Cơ sở:

- Hành vi vứt rác nhanh không nên phải chờ quá lâu.
- Nhưng vẫn cần ít nhất hơn 1 giây để tránh detection chớp nhoáng.

### 12.6. Loại vi phạm `Đứng yên`

Điều kiện:

```text
common_ok
owner_truly_gone
confirm_ctr >= CONFIRM_FRAMES
stationary_after_owner_gone >= STATIONARY_REQUIRED
```

Ý nghĩa:

- Owner đã rời đi.
- Rác tồn tại đủ lâu.
- Rác đứng yên đủ lâu sau khi owner rời đi.

Với 6 FPS:

```text
CONFIRM_FRAMES = 12      -> 2 giây
STATIONARY_REQUIRED = 10 -> 1.67 giây
```

Cơ sở:

- Rác thật sau khi bị bỏ xuống thường nằm yên.
- Vật đang được mang theo hoặc bóng/chân người thường không đứng yên ổn định.
- Điều kiện này giảm false positive rất mạnh.

### 12.7. Loại vi phạm `Bỏ rác tại chỗ`

Điều kiện:

```text
common_ok
confirm_ctr >= CONFIRM_FRAMES
stationary_ctr >= STATIONARY_REQUIRED
owner_id vẫn còn trong current_persons
```

Ý nghĩa:

- Owner vẫn còn trong khung hình.
- Rác đã nằm yên đủ lâu.
- Phù hợp tình huống người bỏ rác xuống rồi đứng lại, chưa rời khỏi camera.

Cơ sở:

- Không phải mọi hành vi vứt rác đều kết thúc bằng việc người đi khỏi khung hình.
- Có người bỏ rác rồi đứng gần đó, nói chuyện hoặc đi chậm.

### 12.8. Ghi log pending

Nếu chưa đủ điều kiện, `_log_pending_reason()` ghi lý do:

```text
owner_none
ambiguous_owner
score_low
owner_not_usable
not_ground
waiting_owner_gone
confirm_frames
stationary_after_gone
```

Ý nghĩa:

- Giúp debug vì sao rác chưa được xác nhận.
- Giúp tinh chỉnh config dựa trên lý do thực tế.
- Giúp báo cáo giải thích logic không xác nhận vội.

---

## 13. Kết luận phần phương pháp

Phương pháp của hệ thống là kết hợp **object detection**, **multi-object tracking**, **motion analysis**, **temporal reasoning** và **rule-based confirmation**.

Từng file có vai trò rõ:

```text
Config.py
    -> định nghĩa thông số và cơ sở thời gian/không gian

TrashViolationDetector.py
    -> điều phối pipeline xử lý từng frame

detector_io.py
    -> tối ưu I/O, FPS, stream và evidence

detection_parsing.py
    -> chuyển YOLO output thành person/trash có ID và lịch sử

MotionDetector.py
    -> phát hiện chuyển động bằng MOG2

OpticalFlowTracker.py
    -> tính hướng chuyển động bằng Lucas-Kanade

OwnershipScorer.py
    -> tính điểm owner bằng nhiều tín hiệu

owner_resolution.py
    -> chọn owner cuối cùng và fallback theo lịch sử

trash_lifecycle.py
    -> quản lý trạng thái rác qua thời gian

violation_confirmation.py
    -> xác nhận và phân loại vi phạm
```

Điểm quan trọng nhất là hệ thống không kết luận từ một frame đơn lẻ. Nó yêu cầu rác tồn tại đủ lâu, đứng yên đủ lâu, owner đủ tin cậy, không ambiguous và thỏa mãn điều kiện mặt đất. Nhờ đó, hệ thống giảm được báo sai do detection chớp nhoáng, vật nhỏ nhiễu, camera rung hoặc người đi ngang qua không liên quan.

