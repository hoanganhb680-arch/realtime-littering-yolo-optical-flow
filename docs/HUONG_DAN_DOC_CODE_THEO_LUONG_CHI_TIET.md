# Hướng dẫn đọc code theo luồng cực chi tiết

Tài liệu này dùng để học vấn đáp và đọc code theo đúng luồng chạy thật của hệ thống. Mục tiêu không phải học thuộc từng dòng, mà là hiểu:

- Frame video đi qua hàm nào trước, hàm nào sau.
- YOLO + ByteTrack tạo dữ liệu gì.
- Dữ liệu đó được lưu vào biến nào.
- MOG2, Optical Flow hỗ trợ ở đâu.
- Khi nào rác được tạo record, khi nào được gán owner, khi nào xác nhận vi phạm.

Luồng tổng quát:

```text
Config
  ↓
TrashViolationDetector.__init__()
  ↓
TrashViolationDetector.run()
  ↓
detector_io._build_track_kwargs()
detector_io._open_capture()
detector_io._init_writer()
  ↓
while đọc từng frame:
    _skip_file_frame()
    detector_io._prepare_frame()
    TrashViolationDetector._process_frame()
        ↓
        MotionDetector.get_alerts()
        YOLO model.track()
        detection_parsing._parse_detections()
        owner_resolution._remember_person_evidence()
        OpticalFlowTracker.update()
        trash_lifecycle._process_trashes()
            ↓
            owner_resolution._find_owner_for_trash()
                ↓
                OwnershipScorer.find_best_owner()
                OwnershipScorer.compute()
            ↓
            violation_confirmation._try_confirm()
                ↓
                violation_confirmation._confirmation_status()
                ViolationLogger.confirm_and_log()
```

---

# 1. `src/Config.py`

## Vai trò file

`Config.py` là nơi chứa toàn bộ cấu hình hệ thống: đường dẫn model, nguồn video, FPS xử lý, ngưỡng YOLO, tham số MOG2, tham số Optical Flow, tham số tracking/lifecycle và điều kiện xác nhận vi phạm.

Nếu thầy hỏi “các thông số nằm ở đâu?”, trả lời:

```text
Các ngưỡng chính được tập trung trong Config.py để dễ chỉnh và giải thích, tránh hard-code rải rác.
```

## Biến nguồn video/model

### `SOURCE_MODE`

```python
SOURCE_MODE = "file"
```

Ý nghĩa:

- `"file"`: đọc video local.
- `"ip_camera"`: đọc camera điện thoại/IP camera.

### `VIDEO_FILE`

```python
VIDEO_FILE = str(_ROOT / "video" / "di_bo_17.mp4")
```

Đường dẫn video mặc định khi chạy file mode.

### `MODEL_PATH`

```python
MODEL_PATH = str(_ROOT / "weights" / "best.pt")
```

Đường dẫn model YOLO đã train/fine-tune. Code load file này một lần ở đầu `run()`.

## Biến tốc độ xử lý

### `FILE_MODE_FPS = 6`

File video được xử lý khoảng 6 FPS.

Lý do:

- YOLO chạy CPU/GPU tốn tài nguyên.
- Video gốc có thể 25-30 FPS.
- Không cần xử lý mọi frame để suy luận hành vi.

Nếu video gốc 30 FPS, target 6 FPS:

```text
30 / 6 = 5
→ lấy khoảng mỗi 5 frame
```

## Biến lịch sử/tracking

### `HISTORY_FRAMES = 90`

Lưu lịch sử người trong 90 processed frames.

Ở 6 FPS:

```text
90 / 6 = 15 giây
```

Dùng cho:

- Tìm người từng đi gần rác.
- Tính `proximity_score`.
- Tính `direction_score`.
- Fallback owner nếu người vừa rời khung.
- Ground fallback: rác gần quỹ đạo chân owner.

### `PERSON_ID_MATCH_RADIUS = 180`

Bán kính fallback để nối ID người khi ByteTrack không trả raw ID.

Sau fix hiện tại:

```text
Nếu ByteTrack có raw_id → dùng raw_id.
Nếu raw_id None → mới dùng bán kính 180 px để cứu ID.
```

Vì vậy 180 px không còn chạy trước ByteTrack nữa.

### `TRASH_ID_MATCH_RADIUS = 90`

Bán kính nối ID rác theo vị trí cũ trong `_trash_registry`.

Rác nhỏ, dễ mất ID hơn người, nên code ưu tiên nối rác bằng vị trí `last_pos`.

## Biến scoring

### `MIN_SCORE = 0.12`

Ngưỡng tối thiểu để một người được xem là candidate owner.

Quan trọng:

```text
0.12 là cửa vào candidate, không phải cửa kết án.
```

Sau đó vẫn cần:

- owner không mơ hồ;
- owner xuất hiện đủ frame;
- owner có chuyển động thật;
- rác hợp lý ở mặt đất/gần chân owner;
- rác tồn tại/đứng yên đủ frame.

### `AMBIGUOUS_MARGIN = 0.12`

Nếu top 1 và top 2 score chênh dưới 0.12:

```text
top1_score - top2_score < 0.12
→ owner mơ hồ
→ chưa xác nhận chắc
```

## Biến xác nhận vi phạm

### `CONFIRM_FRAMES = 12`

Rác cần tồn tại/theo dõi đủ 12 frame cho xác nhận thường.

Ở 6 FPS:

```text
12 / 6 = 2 giây
```

### `CONFIRM_FRAMES_SUDDEN = 8`

Ngưỡng ngắn hơn cho hành vi “Đột ngột”.

Ở 6 FPS:

```text
8 / 6 ≈ 1.33 giây
```

### `MIN_OWNER_GONE_FRAMES = 6`

Owner phải rời khung ít nhất 6 frame mới coi là rời thật.

Ở 6 FPS:

```text
6 / 6 = 1 giây
```

### `STATIONARY_PX = 16`

Nếu tâm rác lệch dưới 16 px giữa hai lần cập nhật:

```text
coi là rác đứng yên
```

### `STATIONARY_REQUIRED = 10`

Rác cần đứng yên đủ 10 frame.

Ở 6 FPS:

```text
10 / 6 ≈ 1.67 giây
```

## Biến ground

### `VIOLATION_GROUND_MIN_Y_RATIO = 0.58`

Rác được coi là vùng mặt đất nếu:

```text
trash_y >= 0.58 * frame_height
```

Nhưng vì góc camera có thể cao, code có fallback:

### `GROUND_OWNER_FOOT_MAX_DISTANCE = 180`

Nếu rác không đủ thấp theo `y`, vẫn có thể hợp lệ nếu gần quỹ đạo chân owner trong 180 px.

### `GROUND_OWNER_FOOT_MAX_VERTICAL_RATIO = 0.17`

Chênh lệch dọc tối đa giữa rác và chân owner:

```text
abs(trash_y - foot_y) <= 0.17 * frame_height
```

## Biến MOG2

### `MOG2_HISTORY = 200`

MOG2 học nền trong 200 frame.

### `MOG2_THRESHOLD = 40`

Ngưỡng khác biệt pixel so với nền.

### `MOG2_MIN_AREA = 300`

Contour chuyển động nhỏ hơn 300 px bị bỏ.

### `MOG2_BOOST_RADIUS = 50`

Nếu có MOG2 motion cách tâm rác dưới 50 px:

```text
score owner *= 1.15
```

## Biến Optical Flow

### `FLOW_HISTORY_FRAMES = 8`

Lưu 8 vector flow gần nhất.

### `LK_PARAMS`

```python
winSize=(15, 15)
maxLevel=2
criteria=(COUNT/EPS, 10, 0.03)
```

Ý nghĩa:

- `winSize=15x15`: LK dùng cửa sổ quanh điểm để giải vector.
- `maxLevel=2`: dùng pyramid mức 0, 1, 2.
- `criteria`: dừng sau 10 vòng hoặc sai số nhỏ hơn 0.03.

## Hàm `__init__()`

```python
def __init__(self):
```

Vai trò:

- Đọc biến môi trường nếu có.
- Dựng `VIDEO_SOURCE`.
- Gán `IS_LIVE`.

Luồng:

```text
Nếu SOURCE_MODE == "file":
    VIDEO_SOURCE = VIDEO_FILE
    IS_LIVE = False

Nếu SOURCE_MODE == "ip_camera":
    VIDEO_SOURCE = URL camera
    IS_LIVE = True
```

---

# 2. `src/TrashViolationDetector.py`

## Vai trò file

Đây là file điều phối chính. Nó không tự làm hết thuật toán, mà gọi các module con đúng thứ tự.

Nói ngắn:

```text
TrashViolationDetector.py là xương sống pipeline.
```

## Class `TrashViolationDetector`

Kế thừa:

```python
class TrashViolationDetector(
    DetectorIoMixin,
    DetectionParsingMixin,
    TrashLifecycleMixin
)
```

Nghĩa là class này dùng được hàm từ:

- `detector_io.py`;
- `detection_parsing.py`;
- `trash_lifecycle.py`;
- và gián tiếp từ `owner_resolution.py`, `violation_confirmation.py`.

## Hàm `__init__(self, cfg=None)`

### Vai trò

Khởi tạo module con và bộ nhớ trạng thái.

### Biến module con

```python
self._motion_detector = MotionDetector()
```

MOG2, tạo `mog2_alerts`.

```python
self._flow_tracker = OpticalFlowTracker()
```

Lucas-Kanade Optical Flow, lưu `flow_vecs`.

```python
self._scorer = OwnershipScorer()
```

Tính điểm owner.

```python
self._logger = ViolationLogger(...)
```

Lưu ảnh/log/database khi confirmed.

### Bộ nhớ hệ thống

```python
self._person_history = {}
```

Lưu lịch sử điểm chân người.

Ví dụ:

```python
_person_history[6] = deque([
    {"anchor": (500, 545), "points": [(500,545),(500,580)], "frame": 40},
    ...
])
```

```python
self._person_seen_counts = {}
```

Đếm người xuất hiện bao nhiêu frame.

```python
self._trash_registry = {}
```

Lưu vòng đời rác qua nhiều frame.

```python
self._person_highest_score = {}
```

Tránh ghi nhiều vi phạm trùng cho cùng người nếu score mới không cao hơn.

```python
self._next_synthetic_person_id = 200000
self._next_synthetic_trash_id = 100000
```

ID giả khi ByteTrack không có ID.

## Hàm `run()`

### Vai trò

Vòng lặp chính của chương trình.

### Các bước đầu

```python
self._validate_paths()
```

Kiểm tra model/video tồn tại.

```python
model = YOLO(cfg.MODEL_PATH)
```

Load YOLO từ `weights/best.pt`.

```python
track_kwargs = self._build_track_kwargs(is_live)
```

Lấy tham số YOLO/ByteTrack.

```python
cap = self._open_capture(is_live)
```

Mở video/camera.

```python
out_vid, source_fps = self._init_writer(cap)
```

Tạo file video output nếu là file mode.

### Vòng lặp frame

```python
while not self.stopped and (is_live or cap.isOpened()):
```

Trong mỗi vòng:

```python
ret, frame = cap.read()
```

Đọc frame.

```python
elif self._skip_file_frame(raw_frame_idx, file_frame_step):
    continue
```

Bỏ bớt frame để chạy khoảng 6 FPS.

```python
frame = self._prepare_frame(frame)
```

Xoay/resize nếu live.

```python
curr_gray, annotated = self._process_frame(...)
```

Xử lý frame chính.

```python
out_vid.write(annotated)
```

Ghi video output.

```python
self._push_frame(annotated)
```

Đẩy frame lên frontend.

```python
prev_gray = curr_gray.copy()
```

Lưu frame xám để Optical Flow frame sau dùng.

## Hàm `_open_capture(is_live)`

### Vai trò

Mở nguồn video.

Nếu live:

```python
ThreadedCamera(...)
```

Nếu file:

```python
cv2.VideoCapture(...)
```

## Hàm `_file_frame_step(source_fps, is_live)`

### Vai trò

Tính lấy mỗi bao nhiêu frame.

Ví dụ:

```text
source_fps = 30
FILE_MODE_FPS = 6
step = round(30/6) = 5
```

## Hàm `_skip_file_frame(raw_frame_idx, file_frame_step)`

### Vai trò

Quyết định frame có bị skip không.

```python
return (raw_frame_idx - 1) % file_frame_step != 0
```

Nếu `file_frame_step=5`:

```text
lấy frame 1, 6, 11, 16...
bỏ các frame còn lại
```

## Hàm `_process_frame(...)`

### Vai trò

Đây là tim của pipeline cho một frame.

Code chính:

```python
curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
mog2_alerts = self._motion_detector.get_alerts(frame)
results = model.track(frame, **track_kwargs)
annotated = results[0].plot()
current_persons, current_trashes = self._parse_detections(results, frame_idx)
self._remember_person_evidence(current_persons, annotated)
self._flow_tracker.update(prev_gray, curr_gray, current_persons)
self._process_trashes(current_trashes, current_persons, mog2_alerts, annotated, frame_idx)
self._cleanup_stale_trash(current_trashes, frame_idx)
self._draw_sticky_trash(annotated, current_trashes, frame_idx)
self._draw_hud(annotated, frame_idx)
```

### Dữ liệu đi qua

Input:

```text
frame màu BGR
prev_gray
YOLO model
track_kwargs
frame_idx
```

Output:

```text
curr_gray
annotated frame
```

Tác dụng phụ:

- cập nhật `_person_history`;
- cập nhật `_flow_tracker.flow_vecs`;
- cập nhật `_trash_registry`;
- có thể lưu ảnh vi phạm.

---

# 3. `src/detector_io.py`

## Vai trò file

Chuẩn bị input/output runtime:

- tham số YOLO/ByteTrack;
- chọn CPU/GPU;
- resize/xoay frame live;
- stream frame lên frontend;
- ghi video output.

## Hàm `_build_track_kwargs(is_live)`

### Vai trò

Tạo dictionary truyền vào:

```python
model.track(frame, **track_kwargs)
```

### Output chính

```python
kwargs = {
    "imgsz": 640 hoặc 512,
    "conf": 0.25 hoặc 0.12,
    "iou": 0.50,
    "persist": True,
    "tracker": "bytetrack.yaml",
    "verbose": False,
}
```

### `imgsz`

File mode:

```text
640
```

Live mode:

```text
512
```

Lý do:

- file mode ưu tiên chính xác;
- live mode ưu tiên giảm lag.

### `conf`

File mode:

```text
0.25
```

Live mode:

```text
0.12
```

Live thấp hơn vì camera rung/mờ dễ bỏ sót.

### `iou`

```text
0.50
```

Đây là IoU cho NMS của YOLO, không phải ByteTrack matching threshold.

### `persist=True`

Rất quan trọng.

```text
Giữ tracker state qua nhiều frame.
```

Nếu không persist, tracking ID dễ reset.

### `tracker="bytetrack.yaml"`

Chọn ByteTrack làm tracker.

## Hàm `_resolve_yolo_runtime()`

### Vai trò

Chọn device chạy YOLO.

Luồng:

```text
Nếu YOLO_DEVICE != "auto":
    dùng device đó
Nếu auto và torch.cuda.is_available():
    dùng GPU 0
Ngược lại:
    dùng CPU
```

### `half`

Nếu GPU CUDA:

```text
có thể dùng FP16 để nhanh hơn
```

Nếu CPU:

```text
không dùng half
```

## Hàm `_prepare_frame(frame)`

### Vai trò

Chuẩn bị frame trước khi đưa vào `_process_frame`.

Luồng:

```text
frame = _orient_frame(frame)
nếu live:
    frame = _resize_for_processing(frame)
```

File mode giữ nguyên frame.

## Hàm `_resize_for_processing(frame)`

### Vai trò

Resize frame live nếu cạnh dài vượt `LIVE_PROCESS_MAX_SIDE`.

Mục tiêu:

- giảm latency;
- giảm tải CPU/GPU;
- realtime ổn định hơn.

## Hàm `_push_frame(annotated)`

### Vai trò

Đẩy frame đã vẽ bbox lên frontend.

Luồng:

```text
annotated
→ resize_for_stream
→ JPEG encode
→ frame_queue
```

Không ảnh hưởng thuật toán.

## Hàm `_validate_paths()`

Kiểm tra:

- model tồn tại;
- video local tồn tại nếu nguồn không phải URL.

## Hàm `_init_writer(cap)`

Tạo `cv2.VideoWriter` để lưu video annotated.

Live mode không ghi output video.

## Hàm `_orient_frame(frame)`

Xoay frame live nếu camera trả landscape trong khi hệ thống muốn portrait.

---

# 4. `src/detection_parsing.py`

## Vai trò file

Đây là file biến output thô YOLO/ByteTrack thành dữ liệu hệ thống dùng được.

YOLO/ByteTrack output:

```text
bbox xyxy
class
confidence
raw track id
```

Hệ thống cần:

```text
current_persons = {person_id: điểm gần chân}
current_trashes = {trash_id: tâm rác}
_person_history = lịch sử người qua frame
```

## Hàm `_parse_detections(results, frame_idx)`

### Input

```text
results từ model.track()
frame_idx hiện tại
```

### Output

```python
current_persons, current_trashes
```

### Luồng bên trong

```python
boxes = results[0].boxes
ids = boxes.id.cpu().numpy() nếu có
```

Mỗi bbox có:

```text
box = [x1, y1, x2, y2]
obj_id = raw ByteTrack ID
cls = class
conf = confidence
```

Nếu `cls == 0`:

```text
person
```

Nếu `cls == 1`:

```text
trash
```

## Nhánh person

Code gọi:

```python
_is_valid_person_box()
_person_points_from_box()
_resolve_person_id()
```

Sau đó lưu:

```python
current_persons[oid] = anchor
self._person_history[oid].append(...)
self._person_seen_counts[oid] += 1
```

## Hàm `_is_valid_person_box(box, conf, results)`

### Vai trò

Lọc người nhiễu.

Điều kiện:

```text
confidence đủ
chiều cao bbox đủ
diện tích bbox đủ
```

Biến dùng:

```python
LIVE_PERSON_CONF = 0.14
LIVE_PERSON_MIN_HEIGHT_RATIO = 0.12
LIVE_PERSON_MIN_AREA_RATIO = 0.012
```

Nếu không đạt:

```python
return False
```

Người đó không được lưu history.

## Hàm `_person_points_from_box(box, frame_shape)`

### Vai trò

Tính điểm đại diện của người.

YOLO bbox:

```text
x1, y1, x2, y2
```

Tính:

```python
bh = y2 - y1
cx = (x1 + x2) / 2
lower = (cx, y1 + 0.88 * bh)
foot = (cx, y2)
```

Output:

```python
anchor = lower
points = [lower, foot]
```

Vì sao:

```text
rác nằm dưới đất → vị trí chân người liên quan hơn tâm thân người.
```

## Hàm `clamp_point(px, py)`

Hàm con bên trong `_person_points_from_box`.

Vai trò:

```text
ép điểm không vượt ra ngoài frame.
```

Ví dụ:

```text
x < 0 → x = 0
x > w-1 → x = w-1
```

## Hàm `_make_person_history_entry(anchor, points, frame_idx)`

Tạo entry lưu vào `_person_history`.

Output dạng:

```python
{
    "anchor": (x, y),
    "points": [(lower_x, lower_y), (foot_x, foot_y)],
    "frame": frame_idx,
}
```

## Hàm `_resolve_person_id(raw_id, anchor, frame_idx, current_persons)`

### Vai trò

Chọn final person ID.

Luồng sau fix:

```text
1. Nếu ByteTrack có raw_id:
       dùng raw_id

2. Nếu raw_id None:
       tìm person_history gần anchor nhất trong bán kính PERSON_ID_MATCH_RADIUS

3. Nếu không tìm được:
       tạo synthetic ID 200000+
```

### Vì sao ưu tiên raw ID?

Vì `results[0].plot()` vẽ ID từ ByteTrack. Nếu code nội bộ đổi ID khác, ảnh và log lệch nhau.

Case video 366:

```text
bbox vẽ id:11
code cũ đổi thành P5
→ sai
```

Sau fix:

```text
raw_id 11 → person_id 11
```

## Nhánh trash

Code lấy tâm bbox:

```python
cx = int((x1 + x2) / 2)
cy = int((y1 + y2) / 2)
```

Sau đó:

```python
oid = _resolve_trash_id(...)
current_trashes[oid] = (cx, cy)
```

## Hàm `_resolve_trash_id(raw_id, center, frame_idx, used_ids)`

### Vai trò

Chọn final trash ID.

Luồng:

```text
1. Tìm rác cũ trong _trash_registry có last_pos gần center <= TRASH_ID_MATCH_RADIUS
2. Nếu có, dùng ID cũ
3. Nếu không, dùng raw ByteTrack ID nếu có
4. Nếu vẫn không, tạo synthetic ID 100000+
```

Vì sao rác ưu tiên registry:

```text
rác nhỏ, dễ mất/đổi ByteTrack ID; nhưng rác thường đứng yên nên vị trí cũ đáng tin.
```

## Hàm `_nearest_history_distance(center, hist, frame_idx, max_age)`

### Vai trò

Tìm khoảng cách gần nhất từ rác tới lịch sử điểm chân người.

Input:

```text
center = tâm rác
hist = lịch sử người
frame_idx = frame hiện tại
max_age = số frame tối đa xét
```

Output:

```text
best_dist, best_frame
```

Dùng trong fallback owner.

---

# 5. `src/MotionDetector.py`

## Vai trò file

Chạy MOG2 để phát hiện foreground/chuyển động.

## Class `MotionDetector`

### Hàm `__init__()`

Tạo:

```python
cv2.createBackgroundSubtractorMOG2(
    history=200,
    varThreshold=40,
    detectShadows=False
)
```

Tạo kernel morphology:

```python
cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
```

## Hàm `get_alerts(frame)`

### Input

```text
frame màu BGR
```

### Output

```python
[(cx, cy, area), ...]
```

### Luồng

```text
fg = MOG2.apply(frame)
fg = morphology open
fg = morphology close
contours = findContours(fg)
lọc contour area < MOG2_MIN_AREA
tính centroid bằng moments
append (cx, cy, area)
```

### Dùng ở đâu?

`mog2_alerts` đi vào:

1. `OwnershipScorer.find_best_owner()` để boost score.
2. `trash_lifecycle._snap_lost_trash_to_motion()` để phục hồi rác mất bbox.

---

# 6. `src/OpticalFlowTracker.py`

## Vai trò file

Tính vector Optical Flow Lucas-Kanade cho từng người.

## Biến chính

### `flow_pts`

```python
self.flow_pts = {}
```

Lưu điểm đang tracking của từng person.

Ví dụ:

```python
flow_pts[6] = np.array([[[551, 545]]])
```

### `flow_vecs`

```python
self.flow_vecs = {}
```

Lưu lịch sử vector flow.

Ví dụ:

```python
flow_vecs[6] = deque([(5.2, -1.1), (6.0, 0.3), ...])
```

## Hàm `update(prev_gray, curr_gray, current_persons)`

### Input

```text
prev_gray = frame xám trước
curr_gray = frame xám hiện tại
current_persons = {person_id: anchor}
```

### Luồng

```text
1. _drop_stale(current_persons)
2. Nếu prev_gray None:
       _init_points(current_persons)
       return
3. Với mỗi person:
       lấy pts_old
       gọi cv2.calcOpticalFlowPyrLK()
       lấy pts_new
       vec = pts_new - pts_old
       lưu vào flow_vecs[person_id]
       reset flow_pts[person_id] = anchor mới từ YOLO
```

### Vì sao reset về anchor YOLO?

Để tránh điểm LK trôi quá xa khỏi người.

Optical Flow chỉ là tín hiệu chuyển động ngắn hạn, không phải tracker chính.

## Hàm `_drop_stale(current_persons)`

Xóa flow của người không còn trong frame.

Nếu person 6 không còn trong `current_persons`:

```python
flow_pts.pop(6)
flow_vecs.pop(6)
```

## Hàm `_init_points(current_persons)`

Frame đầu không có `prev_gray`, nên chỉ khởi tạo điểm tracking cho từng người.

---

# 7. `src/OwnershipScorer.py`

## Vai trò file

Tính điểm “người này có khả năng là chủ rác không”.

## Hàm `compute(trash_center, p_id, history, current_frame, flow_vecs)`

### Input

```text
trash_center = tâm rác
p_id = ID người
history = lịch sử điểm chân người đó
current_frame = frame hiện tại
flow_vecs = vector optical flow theo person
```

### Output

```text
score trong [0, 1]
```

### Công thức

```text
score = 0.40 * proximity_score
      + 0.30 * direction_score
      + 0.20 * flow_score
      + 0.10 * recency_score
```

## Thành phần `proximity_score`

Từ `_nearest()`:

```text
min_dist = khoảng cách gần nhất từ rác tới lịch sử điểm chân người
proximity = 1 - min_dist / TRAJECTORY_RADIUS
```

Nếu:

```text
min_dist > TRAJECTORY_RADIUS = 380
```

thì score = 0.

## Thành phần `direction_score`

Từ `_direction_score()`.

Lấy:

```text
mid = vị trí giữa lịch sử
recent = vị trí mới nhất
move = recent - mid
away = recent - trash
```

Sau đó:

```text
direction = max(0, dot(unit_move, unit_away))
```

Ý nghĩa:

```text
người có đang rời xa vị trí rác theo quỹ đạo tracking không?
```

## Thành phần `flow_score`

Từ `_flow_score()`.

Lấy 4 vector optical flow gần nhất:

```python
recent_flows = flow_vecs[p_id][-4:]
```

Trung bình:

```text
avg_flow = (avg_vx, avg_vy)
```

So với vector:

```text
away = latest_person_position - trash_center
```

Tính:

```text
flow_score = max(0, dot(unit_flow, unit_away))
```

Ý nghĩa:

```text
pixel motion gần đây có cho thấy người đang rời xa rác không?
```

## Thành phần `recency_score`

```text
recency = 1 - closest_offset / HISTORY_FRAMES
```

Nếu người gần rác mới gần đây:

```text
recency cao
```

Nếu người gần rác lâu rồi:

```text
recency thấp
```

## Hàm `find_best_owner(...)`

### Vai trò

Duyệt tất cả candidate người và chọn owner tốt nhất.

Candidate gồm:

```text
current_persons
+
recent_history_ids
```

Nghĩa là cả người đang trong frame và người vừa rời frame vẫn được xét.

### MOG2 boost

Nếu có MOG2 motion gần rác:

```text
distance(mog2_alert, trash_center) < 50
```

thì:

```text
score *= 1.15
```

### Ambiguous

Sau khi sort score:

```text
best_s - second_s < AMBIGUOUS_MARGIN
→ is_ambiguous = True
```

---

# 8. `src/owner_resolution.py`

## Vai trò file

Chọn owner khả dụng và quản lý ảnh bằng chứng owner.

## Hàm `_find_owner_for_trash(...)`

### Vai trò

Tìm owner cho một rác.

Luồng:

```text
1. gọi OwnershipScorer.find_best_owner()
2. nếu owner usable:
       trả owner
3. nếu không:
       gọi _fallback_recent_owner()
4. nếu fallback có:
       trả fallback
5. ngược lại trả kết quả ban đầu
```

## Hàm `_owner_is_usable(owner_id)`

Owner usable khi:

```text
owner_id không None
person_seen_counts[owner_id] >= MIN_OWNER_SEEN_FRAMES
owner có chuyển động thật >= MIN_OWNER_MOTION_PX
```

Tác dụng:

```text
tránh lấy detection người chỉ xuất hiện 1 frame hoặc nhiễu đứng yên.
```

## Hàm `_remember_person_evidence(current_persons, annotated)`

Lưu JPEG frame hiện tại cho từng người đang thấy.

Mục đích:

```text
nếu owner rời khung rồi mới confirm, vẫn còn ảnh owner lúc xuất hiện.
```

## Hàm `_apply_owner_update(...)`

Ghi owner mới vào record rác:

```python
data["owner_id"] = owner_id
data["score"] = score
data["is_ambiguous"] = is_ambig
data["owner_seen_count"] = ...
data["owner_frame_jpg"] = ...
```

Nếu owner không còn trong frame:

```python
data["owner_left_frame"] = frame_idx
```

## Hàm `_update_owner_presence(...)`

Cập nhật:

```text
owner đang trong scene không?
owner_left_frame đã set chưa?
stationary_after_owner_gone có tăng không?
```

## Hàm `_fallback_recent_owner(t_center, frame_idx)`

Tìm người vừa rời khung nhưng lịch sử gần rác.

Dùng:

```text
RECENT_OWNER_GRACE_FRAMES = 90
RECENT_OWNER_RADIUS = 520
```

---

# 9. `src/trash_lifecycle.py`

## Vai trò file

Quản lý vòng đời rác:

```text
rác mới
rác đang pending
rác được thấy lại
rác mất bbox
rác được phục hồi
rác confirmed
```

## Hàm `_process_trashes(...)`

### Input

```text
current_trashes
current_persons
mog2_alerts
annotated
frame_idx
```

### Luồng

```text
for mỗi rác đang thấy:
    nếu rác chưa có trong registry:
        _register_new_trash()
    ngược lại:
        _update_trash_state()
        _try_confirm()

sau đó:
    _recover_recently_lost_trashes()
```

## Hàm `_register_new_trash(...)`

Khi YOLO thấy rác mới:

```text
1. _find_owner_for_trash()
2. _new_trash_record()
3. lưu vào _trash_registry
```

Log:

```text
[TRACK] new_trash=... owner=... score=...
```

## Hàm `_new_trash_record(...)`

Tạo record:

```python
{
    "owner_id": owner_id,
    "score": score,
    "is_ambiguous": is_ambig,
    "spawn_frame": frame_idx,
    "confirm_ctr": 1,
    "stationary_ctr": 0,
    "stationary_after_owner_gone": ...,
    "owner_left_frame": ...,
    "last_pos": t_center,
    "last_seen_frame": frame_idx,
    "seen_count": 1,
    "spawn_pos": t_center,
    "status": "pending",
}
```

## Hàm `_update_trash_state(...)`

Khi rác đã có trong registry và frame hiện tại vẫn thấy rác:

```text
confirm_ctr += 1
seen_count += 1
_mark_latest_trash_position()
_update_owner_presence()
nếu cần thì _reevaluate_owner()
```

## Hàm `_mark_latest_trash_position(data, t_center)`

Tính displacement:

```text
disp = distance(t_center, last_pos)
```

Nếu:

```text
disp < STATIONARY_PX = 16
```

thì:

```text
stationary_ctr += 1
```

Ngược lại:

```text
stationary_ctr = 0
```

Sau đó:

```python
data["last_pos"] = t_center
```

## Hàm `_should_reevaluate_owner(...)`

Tính lại owner nếu còn trong `OWNER_REEVAL_FRAMES` và:

```text
owner None
hoặc owner ambiguous
hoặc score thấp
hoặc owner rời scene
hoặc owner không usable
```

## Hàm `_recover_recently_lost_trashes(...)`

### Vai trò

Cứu rác khi YOLO mất bbox.

Điều kiện:

```text
rác không có trong current_trashes
rác chưa confirmed
rác chưa mất quá LOST_TRASH_RECOVERY_FRAMES = 60
rác đủ seen_count hoặc là candidate mạnh 1-frame
```

Nếu có thể recover:

```text
_recover_lost_trash_position()
_mark_latest_trash_position()
confirm_ctr += 1
_update_owner_presence()
_try_confirm()
```

## Hàm `_can_recover_strong_single_seen_trash(data)`

Cho phép cứu rác chỉ thấy 1 frame nếu:

```text
seen_count >= 1
owner_id != None
not ambiguous
score >= REEVAL_SCORE_THRESH = 0.25
```

Đây là fix giúp video đi xe.

## Hàm `_recover_lost_trash_position(...)`

Thử 3 cách theo thứ tự:

```text
1. _snap_lost_trash_to_owner()
2. _snap_lost_trash_to_motion()
3. giữ last_pos nếu hợp lý ground/gần chân owner
```

## Hàm `_snap_lost_trash_to_owner(...)`

Nếu owner còn trong frame:

```text
lấy điểm chân owner mới nhất
nếu điểm đó gần last_pos rác <= 220 px
và đủ vùng mặt đất
→ dùng điểm đó làm recovered_pos
```

## Hàm `_snap_lost_trash_to_motion(...)`

Nếu có MOG2 motion:

```text
motion center gần last_pos rác <= 220 px
và y >= min_ground_y
→ dùng motion center làm recovered_pos
```

---

# 10. `src/violation_confirmation.py`

## Vai trò file

Quyết định rác pending đã đủ điều kiện thành vi phạm chưa.

## Hàm `_try_confirm(...)`

Luồng:

```text
status = _confirmation_status()
nếu status["violation_type"] None:
    _log_pending_reason()
    return
nếu owner rõ:
    ViolationLogger.confirm_and_log()
    push alert frontend
```

## Hàm `_confirmation_status(...)`

### Điều kiện chung

```python
common_ok = (
    data["score"] >= MIN_SCORE
    and owner_seen_enough
    and not data["is_ambiguous"]
    and ground_condition
)
```

### `owner_seen_enough`

Owner phải:

```text
owner_id != None
seen_count >= MIN_OWNER_SEEN_FRAMES = 2
owner có motion thật >= MIN_OWNER_MOTION_PX = 5
```

### `ground_condition`

Gọi:

```python
_is_ground_trash()
```

Rác hợp lệ nếu:

```text
trash_y >= 0.58 * frame_height
```

hoặc:

```text
rác gần quỹ đạo chân owner <= 180 px
và chênh dọc <= 0.17 * frame_height
```

## Ba kiểu xác nhận

### `sudden_ok` — Đột ngột

```text
common_ok
owner đã rời khung >= 6 frame
frames_since_spawn <= 8 + 6 = 14
confirm_ctr >= 8
```

### `stationary_ok` — Đứng yên

```text
common_ok
owner đã rời khung
confirm_ctr >= 12
stationary_after_owner_gone >= 10
```

### `proximity_ok` — Bỏ rác tại chỗ

```text
common_ok
owner vẫn trong current_persons
confirm_ctr >= 12
stationary_ctr >= 10
```

## Hàm `_log_pending_reason(...)`

In lý do chưa confirm:

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

Đây là hàm debug cực hữu ích khi video không detect.

## Hàm `_cleanup_stale_trash(...)`

Xóa rác khỏi registry nếu mất quá `STALE_FRAMES`.

## Hàm `_draw_sticky_trash(...)`

Vẽ tạm rác vừa mất bbox vài frame để UI không nhấp nháy.

## Hàm `_draw_hud(...)`

Vẽ:

```text
F:{frame_idx} VI PHAM:{count}
```

---

# 11. `src/ViolationLogger.py`

## Vai trò file

Lưu bằng chứng sau khi đã confirmed.

## Hàm `confirm_and_log(...)`

Nếu:

```text
owner_id != None
not ambiguous
```

thì:

```text
_save_evidence()
```

Nếu ambiguous:

```text
_draw_ambiguous()
```

## Hàm `_save_evidence(...)`

Làm:

```text
1. Lấy ảnh owner_frame_jpg nếu có
2. Vẽ chữ VI PHAM: Person_X
3. Vẽ vòng tròn quanh rác
4. Lưu file jpg
5. Append vào violation_log
6. Gửi API sync
```

Tên file:

```text
violation_P{owner_id}_T{trash_id}_F{frame_idx}.jpg
```

Ví dụ:

```text
violation_P6_T27_F70.jpg
```

## Hàm `save_candidate(...)`

Lưu ảnh debug khi có rác nhưng chưa tìm được owner.

## Hàm `_decode_frame(...)`

Biến JPEG bytes trong RAM thành ảnh OpenCV.

---

# 12. Năm biến/dictionary quan trọng nhất

Nếu chỉ nhớ 5 biến, nhớ mấy cái này:

## `current_persons`

Tồn tại trong 1 frame.

```python
{person_id: foot_anchor}
```

Ví dụ:

```python
{6: (551, 545)}
```

## `current_trashes`

Tồn tại trong 1 frame.

```python
{trash_id: trash_center}
```

Ví dụ:

```python
{27: (410, 492)}
```

## `_person_history`

Sống qua nhiều frame.

```python
{person_id: deque(history_entries)}
```

Dùng cho:

- proximity;
- direction;
- fallback owner;
- ground fallback.

## `_flow_tracker.flow_vecs`

Sống qua nhiều frame nhưng reset khi person rời scene.

```python
{person_id: deque([(vx, vy), ...])}
```

Dùng cho:

- flow_score.

## `_trash_registry`

Sống qua nhiều frame.

```python
{trash_id: trash_record}
```

Dùng cho:

- lifecycle rác;
- confirm counter;
- stationary counter;
- owner;
- recovery.

---

# 13. Một luồng cụ thể: video đi xe

```text
Frame 28-59:
    YOLO thấy người đi xe P6
    _parse_detections lưu _person_history[6]
    OpticalFlowTracker lưu flow_vecs[6]

Frame 60:
    YOLO thấy trash T27
    current_trashes[27] = (410, 492)
    _register_new_trash()
    _find_owner_for_trash()
    OwnershipScorer chọn P6, score ≈ 0.79
    _trash_registry[27] = pending record

Frame 61-69:
    YOLO mất bbox rác
    _recover_recently_lost_trashes()
    vì rác 1-frame nhưng owner rõ, score cao, không mơ hồ
    code giữ/recover last_pos
    confirm_ctr tăng

Frame 70:
    _try_confirm()
    sudden_ok đạt
    ViolationLogger lưu violation_P6_T27_F70.jpg
```

---

# 14. Một luồng cụ thể: video 366

```text
YOLO/ByteTrack vẽ người vi phạm raw_id = 11
_resolve_person_id() dùng raw_id 11 trước
_person_history[11] được lưu

Rác T25 xuất hiện
OwnershipScorer tính P11 là owner
_trash_registry[T25].owner_id = 11

Đủ điều kiện confirm
ViolationLogger lưu violation_P11_T25
```

Lỗi cũ:

```text
_resolve_person_id() ưu tiên match lịch sử trước raw_id
raw_id 11 bị kéo nhầm về P5
```

Fix:

```text
raw_id có thì dùng trước
fallback khoảng cách chỉ dùng khi raw_id None
```

---

# 15. Cách tự đọc code khi mở IDE

Đọc theo thứ tự này:

```text
1. TrashViolationDetector.run()
2. TrashViolationDetector._process_frame()
3. detector_io._build_track_kwargs()
4. detection_parsing._parse_detections()
5. MotionDetector.get_alerts()
6. OpticalFlowTracker.update()
7. trash_lifecycle._process_trashes()
8. owner_resolution._find_owner_for_trash()
9. OwnershipScorer.compute()
10. violation_confirmation._confirmation_status()
11. ViolationLogger._save_evidence()
```

Mỗi lần đọc một hàm, hỏi 4 câu:

```text
Input của hàm là gì?
Output của hàm là gì?
Hàm này sửa biến global/object nào?
Hàm này gọi hàm nào tiếp theo?
```

---

# 16. Câu nói vấn đáp tổng kết

Nếu thầy hỏi “code chạy từ đầu đến cuối thế nào?”, nói:

```text
Hệ thống đọc video theo từng frame trong TrashViolationDetector.run().
Mỗi frame đi vào _process_frame().
Ở đây MOG2 tạo danh sách vùng chuyển động, YOLOv8s phát hiện person/trash, ByteTrack gán raw ID.
detection_parsing chuyển bbox thành điểm chân người, tâm rác, current_persons/current_trashes và cập nhật person_history.
OpticalFlowTracker lấy vector chuyển động ngắn hạn cho từng người.
trash_lifecycle quản lý rác mới/cũ/mất bbox; khi rác mới xuất hiện thì owner_resolution gọi OwnershipScorer để tính owner.
OwnershipScorer dùng proximity, direction, optical flow và recency để tính điểm.
violation_confirmation kiểm tra score, ambiguous, owner_seen, ground_condition, confirm_ctr, stationary và owner_left.
Nếu đủ điều kiện, ViolationLogger lưu ảnh bằng chứng, log và gửi API.
```

Câu chốt:

```text
YOLO/ByteTrack tạo dữ liệu thô.
detection_parsing biến thành ID + vị trí.
person_history/flow_vecs/trash_registry lưu thông tin qua thời gian.
OwnershipScorer và Confirmation mới kết luận vi phạm.
```
