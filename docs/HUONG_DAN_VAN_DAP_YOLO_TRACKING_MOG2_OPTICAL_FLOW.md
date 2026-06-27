# HƯỚNG DẪN VẤN ĐÁP: YOLO - BYTETRACK - MOG2 - OPTICAL FLOW

Tài liệu này đối chiếu trực tiếp:

- Code hiện tại trong `src/`.
- Trọng số thật `weights/best.pt`.
- Slide `ThuyetTrinh-6.pptx`.
- Báo cáo `Báo cáo nhóm 6_68CS1.pdf`.

Phạm vi chính là tầng quan sát và mã hóa thông tin trước khi tính Ownership Score:

```text
Video/frame
    -> tiền xử lý
    -> MOG2
    -> YOLOv8s
    -> ByteTrack
    -> ổn định ID bổ sung
    -> lịch sử vị trí người/rác
    -> Lucas-Kanade Optical Flow
    -> dữ liệu đầu vào cho Ownership Score
```

---

# 1. Kết luận kiểm tra code với báo cáo và slide

## 1.1. Những nội dung đang đúng

Các mô tả sau trong báo cáo và slide phù hợp với code:

1. YOLO phát hiện hai lớp `person` và `trash`.
2. ByteTrack được gọi thông qua `model.track(..., tracker="bytetrack.yaml")`.
3. Tracking được duy trì qua nhiều lần gọi nhờ `persist=True`.
4. Người được biểu diễn bằng điểm gần chân, rác được biểu diễn bằng tâm bounding box.
5. Hệ thống có tầng nối lại ID dựa trên khoảng cách sau ByteTrack.
6. MOG2 không thay thế YOLO; nó chỉ cung cấp tín hiệu foreground/chuyển động.
7. MOG2 được dùng để tăng nhẹ owner score khi có chuyển động gần rác.
8. MOG2 còn tham gia phục hồi vị trí rác khi YOLO mất detection ngắn hạn.
9. Optical Flow dùng Pyramidal Lucas-Kanade.
10. Flow được lưu riêng cho từng `person_id`.
11. Flow score tăng khi vector chuyển động cùng hướng rời xa rác.
12. Ownership Score dùng bốn thành phần với trọng số 40%-30%-20%-10%.

## 1.2. Những nội dung cần sửa cách nói hoặc bổ sung khi vấn đáp

### Điểm 1: Model phải được gọi đúng là YOLOv8s

Slide và báo cáo chủ yếu viết chung là “YOLO”. Trọng số thật cho thấy:

```text
Model gốc: yolov8s.pt
Task: detect
Số lớp: 2
Class 0: person
Class 1: trash
Input train: 640
Backbone: Conv + C2f + SPPF
Đầu ra đa tỉ lệ: P3/8, P4/16, P5/32
Số tham số: khoảng 11,14 triệu
```

Khi thầy hỏi “nhóm dùng YOLO nào?”, phải trả lời:

> Nhóm fine-tune YOLOv8s pretrained cho bài toán detection hai lớp person và trash, không phải dùng YOLO chung chung hoặc model COCO nguyên bản.

### Điểm 2: Slide 7 không nên nói “MOG2 là chuyển động của nền”

Cách nói đúng:

> MOG2 xây dựng mô hình thống kê của nền tại từng pixel, sau đó phân loại những pixel không phù hợp mô hình nền thành foreground. Foreground thường tương ứng vùng chuyển động hoặc vùng cảnh vừa thay đổi.

MOG2 không đo “nền đang chuyển động”. Nó mô hình hóa nền để tìm phần khác nền.

### Điểm 3: Ví dụ phép tính ở slide 10 bị sai

Slide ghi:

```text
P = 0.80
D = 0.60
F = 0.50
R = 0.70
Score = 0.65
```

Kết quả đúng:

```text
Score
= 0.40 x 0.80
+ 0.30 x 0.60
+ 0.20 x 0.50
+ 0.10 x 0.70
= 0.32 + 0.18 + 0.10 + 0.07
= 0.67
```

Nếu thầy tính lại, phải thừa nhận slide bị lỗi số học và trả lời kết quả đúng là `0.67`.

### Điểm 4: `maxLevel=2` không phải chỉ có hai mức ảnh

OpenCV định nghĩa `maxLevel` theo chỉ số bắt đầu từ 0:

```text
maxLevel = 0 -> chỉ level 0
maxLevel = 1 -> level 0 và 1
maxLevel = 2 -> level 0, 1 và 2
```

Vì vậy code đang dùng ba mức pyramid, dù comment trong `Config.py` ghi “2 tầng”.

### Điểm 5: Báo cáo chưa cập nhật điều kiện mặt đất mới

Báo cáo mô tả:

```text
trash_y >= 0.58 x image_height
```

Code hiện tại còn có fallback:

```text
Nếu không qua ngưỡng 58%,
rác vẫn có thể được xem là dưới đất
nếu nằm đủ gần quỹ đạo chân của owner.
```

Hai tham số mới:

```text
GROUND_OWNER_FOOT_MAX_DISTANCE = 180 px
GROUND_OWNER_FOOT_MAX_VERTICAL_RATIO = 0.17
```

Nếu bị hỏi, nói rằng điều kiện 58% là điều kiện chính; khoảng cách tới chân owner là fallback cho camera góc cao.

### Điểm 6: ByteTrack hai tầng nhưng file mode gần như không tận dụng tầng confidence thấp

ByteTrack mặc định trong môi trường dự án:

```text
track_high_thresh = 0.25
track_low_thresh  = 0.10
new_track_thresh  = 0.25
track_buffer      = 30
match_thresh      = 0.80
fuse_score        = True
```

Trong code:

```text
File video: YOLO conf = 0.25
Camera live: YOLO conf = 0.12
```

Hệ quả:

- File mode đã loại detection dưới 0.25 trước khi ByteTrack xử lý.
- Vì vậy ByteTrack hầu như không nhận được nhóm detection 0.10-0.25 cho lượt ghép thứ hai.
- Live mode cho phép detection từ 0.12, nên ByteTrack có thể dùng vùng 0.12-0.25 ở lượt thứ hai.

Nếu thầy hỏi “nhóm có thật sự tận dụng low-confidence association của ByteTrack không?”, câu trả lời trung thực:

> Có rõ hơn ở live mode. Ở file mode, `conf=0.25` trùng `track_high_thresh`, nên lợi thế của lượt low-confidence bị hạn chế.

### Điểm 7: Optical Flow trong code là một phiên bản rất nhẹ

Code không:

- Tìm nhiều corner bằng Shi-Tomasi.
- Theo dõi toàn bộ người.
- Tạo dense optical-flow field.

Code chỉ đặt một điểm theo dõi gần chân cho mỗi người:

```python
flow_pts[person_id] = [[cx, cy]]
```

Sau đó dùng Lucas-Kanade tìm vị trí tương ứng của điểm đó ở frame tiếp theo.

Cách diễn đạt đúng:

> Hệ thống dùng sparse pyramidal Lucas-Kanade trên một điểm neo gần chân của mỗi person, nhằm lấy tín hiệu hướng chuyển động nhẹ, chứ Optical Flow không đảm nhiệm việc giữ ID.

### Điểm 8: Flow bị xóa ngay khi người không còn ở frame hiện tại

Trong `_drop_stale()`:

```python
stale = set(flow_pts) - set(current_persons)
flow_pts.pop(person_id)
flow_vecs.pop(person_id)
```

Vì vậy `FLOW_HISTORY_FRAMES=8` không có nghĩa flow luôn được giữ 8 frame sau khi người rời ảnh. Nó chỉ giữ tối đa 8 vector trong lúc người vẫn đang được nhận diện liên tục.

### Điểm 9: Tầng matching khoảng cách có thể ghi đè cách dùng ID thô của ByteTrack

Trong `_resolve_person_id()`, code tìm một lịch sử gần nhất trước. Chỉ khi không tìm được mới dùng `raw_id` của ByteTrack.

Ưu điểm:

- Nối lại ID khi ByteTrack đổi ID.

Rủi ro:

- Hai người đứng gần hoặc cắt nhau có thể bị gộp nhầm.
- Đây là heuristic theo vị trí, không có ReID đặc trưng ngoại hình.

### Điểm 10: Số liệu dataset “45 video” chưa được chứng minh hoàn toàn bởi dữ liệu đang có

Các thư mục dataset hiện tại chứa:

```text
data_cu1_clean:       33 source video ID, 1.824 ảnh
data_cu1_clean_v2:    37 source video ID, 1.986 ảnh
data_cu1_clean_v3_bg: 37 source video ID, 2.401 ảnh
```

Có thể nhóm từng quay khoảng 45 video thô rồi loại một số video, nhưng repository hiện không đủ bằng chứng để khẳng định chính xác. Trước khi vấn đáp, cả nhóm nên thống nhất:

- 45 là số video thô đã quay; hay
- 37 là số video thực sự đi vào phiên bản dataset cuối.

### Điểm 11: Báo cáo có phép quy đổi dataset chưa khớp

Báo cáo viết video dài 15-20 giây và trích 5 frame/giây.

Theo phép tính:

```text
15 x 5 = 75 frame
20 x 5 = 100 frame
```

Vì vậy khoảng hợp lý là 75-100 frame/video, không phải 100-130 frame/video nếu mọi video chỉ dài 15-20 giây.

### Điểm 12: Báo cáo chưa nêu metrics thật của model

Checkpoint lưu các chỉ số:

```text
Precision:   0.87198
Recall:      0.73907
mAP@0.50:    0.80467
mAP@0.50:95: 0.44909
```

Model được cấu hình train tối đa 100 epoch, kết quả có 97 epoch. Giá trị mAP50-95 tốt nhất xuất hiện ở epoch 77; sau đó không cải thiện trong khoảng patience nên quá trình dừng ở epoch 97.

---

# 2. Phải hiểu “tầng trước Ownership Score” là gì?

Ownership Score không nhận trực tiếp ảnh.

Nó nhận dữ liệu đã được các tầng trước biến đổi:

```text
Ảnh màu BGR
    |
    +--> MOG2
    |       -> [(motion_x, motion_y, area), ...]
    |
    +--> YOLOv8s
            -> bounding boxes
            -> class
            -> confidence
                    |
                    v
               ByteTrack
                    -> track ID
                    |
                    v
              Detection Parsing
                    -> current_persons
                    -> current_trashes
                    -> person_history
                    |
                    v
             Lucas-Kanade Flow
                    -> flow_vecs

Ownership Score nhận:

- trash_center
- current_persons
- person_history
- flow_vecs
- mog2_alerts
- current_frame
```

Đây có thể gọi là tầng “trích xuất và mã hóa tín hiệu quan sát”.

---

# 3. Luồng chính xác khi xử lý một frame

## Bước 1: Đọc frame

`cv2.VideoCapture` hoặc `ThreadedCamera` trả về một ảnh BGR:

```python
ret, frame = cap.read()
```

Nếu là file 30 FPS và mục tiêu 6 FPS:

```text
file_frame_step = round(30 / 6) = 5
```

Hệ thống xử lý frame gốc:

```text
1, 6, 11, 16, 21, ...
```

Các frame ở giữa bị bỏ.

## Bước 2: Chuẩn hóa frame

File video:

- Không xoay.
- Không resize thủ công trước YOLO.
- YOLO tự letterbox về `imgsz=640`.

Live camera:

- Có thể xoay dọc.
- Cạnh dài được giới hạn tối đa 960 pixel.
- YOLO chạy với `imgsz=512`.

## Bước 3: Chuyển grayscale

```python
curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
```

Grayscale được dùng cho Lucas-Kanade vì Optical Flow dựa vào độ sáng/cường độ ảnh, không cần ba kênh màu.

## Bước 4: Chạy MOG2

```python
mog2_alerts = motion_detector.get_alerts(frame)
```

Kết quả ví dụ:

```python
[
    (315, 640, 1250.0),
    (510, 900, 820.0)
]
```

Mỗi phần tử gồm:

```text
cx, cy: tâm vùng foreground
area: diện tích contour
```

MOG2 chưa biết đó là người, xe, rác, bóng hay nhiễu.

## Bước 5: Chạy YOLO + ByteTrack

```python
results = model.track(frame, **track_kwargs)
```

File mode tương đương:

```python
model.track(
    frame,
    imgsz=640,
    conf=0.25,
    iou=0.50,
    persist=True,
    tracker="bytetrack.yaml",
    device="cpu",
    half=False,
)
```

YOLO tạo detection, ByteTrack gắn track ID.

Mỗi detection có:

```text
xyxy = [x1, y1, x2, y2]
cls  = class index
conf = confidence
id   = tracking ID hoặc None
```

## Bước 6: Tách person và trash

Code đọc:

```python
boxes.xyxy
boxes.cls
boxes.conf
boxes.id
```

Nếu `cls == 0`, đối tượng là người.

Nếu `cls == 1`, đối tượng là rác.

Kết quả:

```python
current_persons = {
    person_id: (anchor_x, anchor_y)
}

current_trashes = {
    trash_id: (center_x, center_y)
}
```

## Bước 7: Lọc person box

Người bị loại nếu:

```text
confidence < 0.14
hoặc bbox height < 12% chiều cao frame
hoặc bbox area < 1.2% diện tích frame
```

Lưu ý: hàm này chạy cả file và live dù tên biến có chữ `LIVE`.

Trong file mode, YOLO đã dùng `conf=0.25`, nên điều kiện `conf>=0.14` gần như thừa. Hai điều kiện kích thước vẫn có tác dụng.

## Bước 8: Chọn điểm đại diện của người

Với bbox:

```text
(x1, y1) --------
|                |
|     person     |
|                |
-------- (x2,y2)
```

Code tính:

```python
cx = (x1 + x2) / 2
lower = (cx, y1 + 0.88 x bbox_height)
foot = (cx, y2)
```

`lower` là anchor chính.

`lower` và `foot` cùng được lưu trong lịch sử để tính khoảng cách tới rác.

## Bước 9: Ổn định ID bổ sung

Person:

1. So sánh anchor mới với vị trí cuối các lịch sử cũ.
2. Chỉ xét lịch sử không quá 90 frame.
3. Nếu khoảng cách <=180 pixel, tái sử dụng ID cũ.
4. Nếu không khớp, dùng ByteTrack ID.
5. Nếu ByteTrack không có ID, tạo synthetic ID từ 200000.

Trash:

1. So sánh tâm rác với `last_pos` trong `trash_registry`.
2. Nếu khoảng cách <=90 pixel, dùng ID rác cũ.
3. Nếu không, dùng ByteTrack ID.
4. Nếu không có, tạo synthetic ID từ 100000.

## Bước 10: Cập nhật person history

Ví dụ:

```python
person_history[2].append({
    "anchor": (320, 700),
    "points": [(320, 700), (320, 725)],
    "frame": 41,
})
```

Deque chỉ giữ tối đa 90 phần tử gần nhất.

## Bước 11: Cập nhật Lucas-Kanade Flow

```python
flow_tracker.update(
    prev_gray,
    curr_gray,
    current_persons,
)
```

Với mỗi person:

```text
Điểm cũ: p = (x, y)
Tìm điểm mới: p' = (x+u, y+v)
Flow vector: (u, v) = p' - p
```

Vector được thêm vào:

```python
flow_vecs[person_id]
```

Mỗi person giữ tối đa 8 vector.

## Bước 12: Dữ liệu được chuyển tới lifecycle và scorer

Nếu có rác:

```python
process_trashes(
    current_trashes,
    current_persons,
    mog2_alerts,
    annotated,
    frame_idx,
)
```

Khi cần tìm owner:

```python
find_best_owner(
    trash_center,
    current_persons,
    current_frame,
    person_history,
    flow_vecs,
    mog2_alerts,
)
```

---

# 4. YOLOv8s: lý thuyết phải nắm

## 4.1. Object detection là gì?

Object detection giải đồng thời hai bài toán:

1. Localization: vật nằm ở đâu?
2. Classification: vật thuộc lớp nào?

Đầu ra:

```text
Bounding box
Class label
Confidence score
```

Khác classification:

- Classification chỉ nói ảnh có gì.
- Detection nói có vật gì và nằm ở đâu.

Khác segmentation:

- Detection biểu diễn vật bằng hình chữ nhật.
- Segmentation dự đoán mask theo pixel.

## 4.2. Vì sao gọi là YOLO?

Ý tưởng lịch sử của YOLO là xử lý toàn ảnh bằng một mạng duy nhất thay vì:

1. Sinh proposal vùng.
2. Chạy classifier riêng trên từng proposal.

YOLO biến object detection thành bài toán dự đoán trực tiếp từ ảnh tới box và class.

Không nên nói “YOLO chỉ thực hiện đúng một phép toán”. “One look” nghĩa là một lần forward qua mạng cho toàn ảnh.

## 4.3. YOLOv8s trong dự án

Checkpoint cho thấy model gốc là `yolov8s.pt`.

Chữ `s` là small:

- Lớn hơn `n` (nano).
- Nhẹ hơn `m`, `l`, `x`.
- Cân bằng tốc độ và độ chính xác.

Kiến trúc checkpoint:

```text
Input
 -> Backbone: Conv, C2f
 -> SPPF
 -> Neck/head: Upsample, Concat, C2f
 -> Detect tại 3 scale: stride 8, 16, 32
```

### Backbone

Backbone trích xuất feature từ ảnh:

- Lớp đầu học cạnh, màu, texture.
- Lớp sâu học bộ phận và cấu trúc phức tạp hơn.

### C2f

C2f giúp:

- Chia và nối luồng feature.
- Tăng khả năng truyền gradient.
- Tái sử dụng feature.
- Cân bằng chi phí tính toán và khả năng biểu diễn.

Không cần thuộc từng convolution bên trong nếu thầy không yêu cầu, nhưng phải biết C2f là block đặc trưng của YOLOv8.

### SPPF

SPPF là Spatial Pyramid Pooling Fast.

Mục đích:

- Mở rộng receptive field.
- Kết hợp ngữ cảnh ở nhiều phạm vi.
- Giúp model hiểu vật trong bối cảnh rộng hơn.

### Multi-scale detection

Model dự đoán tại:

```text
P3 / stride 8:  feature map độ phân giải cao, hữu ích cho vật nhỏ.
P4 / stride 16: vật vừa.
P5 / stride 32: vật lớn.
```

Rác nhỏ phụ thuộc mạnh vào nhánh P3, nhưng YOLOv8 chuẩn không có P2/4. Đây là một lý do vật rất nhỏ có thể khó nhận diện.

## 4.4. Input 640 nghĩa là gì?

`imgsz=640` không nhất thiết bóp méo ảnh thành đúng 640x640.

Ultralytics thường:

1. Resize giữ tỉ lệ.
2. Padding để kích thước phù hợp stride.
3. Chuẩn hóa pixel.
4. Đưa tensor vào model.

File mode dùng 640, đúng với kích thước train.

Live dùng 512 để giảm độ trễ CPU, đổi lại vật nhỏ có ít pixel hơn và dễ bị mất.

## 4.5. Confidence là gì?

Confidence là điểm tin cậy của detection.

Không nên phát biểu:

> Confidence 0.8 nghĩa chắc chắn 80% đó là rác.

Cách nói an toàn:

> Confidence là điểm tin cậy do model sinh ra để xếp hạng và lọc detection; nó không nhất thiết là xác suất đã được hiệu chuẩn.

Trong dự án:

```text
File: conf >= 0.25
Live: conf >= 0.12
```

Hạ confidence:

- Tăng recall.
- Giảm bỏ sót rác nhỏ.
- Nhưng tăng false positive.

Tăng confidence:

- Detection sạch hơn.
- Nhưng dễ bỏ sót rác.

## 4.6. IoU là gì?

Với hai box A và B:

```text
IoU = Area(A giao B) / Area(A hợp B)
```

Khoảng:

```text
0: không giao nhau
1: trùng hoàn toàn
```

Trong `_build_track_kwargs()`:

```text
iou = 0.50
```

Đây chủ yếu là ngưỡng NMS của detection, không phải `match_thresh=0.8` của ByteTrack.

Phải phân biệt:

```text
YOLO iou=0.50:
lọc các box dự đoán trùng nhau trong cùng frame.

ByteTrack match_thresh=0.80:
ngưỡng chi phí/association dùng để ghép detection với track.
```

## 4.7. NMS là gì?

Model có thể dự đoán nhiều box quanh cùng một vật.

NMS:

1. Chọn box có confidence cao nhất.
2. Loại box khác có IoU quá cao với box đã chọn.
3. Lặp lại với các box còn lại.

Mục tiêu: một vật không bị trả về nhiều box trùng nhau.

## 4.8. Kết quả training thật

```text
Pretrained model: YOLOv8s
Classes: person, trash
Optimizer: AdamW
Input size: 640
Configured epochs: 100
Actual history: 97 epochs
Best epoch theo mAP50-95: 77
Precision: 0.872
Recall: 0.739
mAP50: 0.805
mAP50-95: 0.449
```

### Precision

Trong các detection model dự đoán ra, bao nhiêu detection là đúng:

```text
Precision = TP / (TP + FP)
```

### Recall

Trong các vật thật, model tìm được bao nhiêu:

```text
Recall = TP / (TP + FN)
```

### AP và mAP

AP là diện tích dưới đường Precision-Recall của một lớp.

mAP là trung bình AP của các lớp.

`mAP50` dùng IoU 0.5.

`mAP50-95` trung bình nhiều ngưỡng IoU từ 0.5 đến 0.95, nghiêm ngặt hơn nên thường thấp hơn.

## 4.9. Hạn chế YOLO trong bài toán này

1. Rác nhỏ, ít pixel.
2. Rác bị tay hoặc người che.
3. Rác giống nền.
4. Camera xa hoặc rung.
5. Input live 512 làm mất chi tiết.
6. Sampling 6 FPS có thể bỏ qua các frame model nhận diện tốt.
7. Model chỉ nhận diện vật, không hiểu động tác “thả” hoặc “ném”.

---

# 5. Tracking và ByteTrack

## 5.1. Vì sao detection chưa đủ?

YOLO chạy riêng từng frame sẽ trả về:

```text
Frame 1: person box
Frame 2: person box
```

Nhưng YOLO thuần không đảm bảo hai box đó là cùng một người.

Tracking thêm identity:

```text
Frame 1: Person ID 3
Frame 2: Person ID 3
Frame 3: Person ID 3
```

Nhờ đó mới lưu được quỹ đạo.

## 5.2. ByteTrack là tracking-by-detection

ByteTrack không tự nhìn ảnh để nhận diện person/trash như YOLO.

Nó nhận đầu vào là các detection box:

```text
[x1, y1, x2, y2, confidence, class]
```

Sau đó liên kết box qua thời gian.

## 5.3. Các thành phần lý thuyết

### Kalman Filter

Kalman Filter dự đoán trạng thái track ở frame tiếp theo.

Trạng thái thường chứa:

- Vị trí box.
- Kích thước/aspect ratio.
- Vận tốc.

Chu trình:

```text
Predict:
Dùng trạng thái cũ dự đoán box mới.

Update:
Dùng detection thực tế sửa lại dự đoán.
```

### IoU/cost matching

Track dự đoán và detection mới được so sánh.

Nếu box gần/trùng nhau, chi phí ghép thấp hơn.

### Linear assignment

Khi có nhiều track và nhiều detection, cần tìm cách ghép tổng thể tốt nhất, không thể mỗi track tự chọn detection gần nhất độc lập.

ByteTrack dùng bài toán linear assignment, thường được giải bằng thuật toán Hungarian hoặc biến thể tương đương.

## 5.4. Ý tưởng cốt lõi hai lượt của ByteTrack

### Lượt 1: detection confidence cao

```text
score >= track_high_thresh
```

Ghép các detection đáng tin với track đang hoạt động.

### Lượt 2: detection confidence thấp

Những track chưa ghép được thử lại với:

```text
track_low_thresh < score < track_high_thresh
```

Mục tiêu:

- Khi vật bị che một phần, confidence có thể giảm.
- Detection thấp vẫn có thể là vật thật.
- Dùng quan hệ với track cũ để cứu detection thật và bỏ nhiễu.

Đây là ý nghĩa của “associating almost every detection box”.

## 5.5. Thông số ByteTrack thật

```text
track_high_thresh = 0.25
track_low_thresh = 0.10
new_track_thresh = 0.25
track_buffer = 30
match_thresh = 0.80
fuse_score = True
```

### `track_high_thresh`

Ngưỡng detection cho lượt ghép đầu.

### `track_low_thresh`

Ngưỡng thấp nhất cho lượt ghép cứu track.

### `new_track_thresh`

Detection phải đủ điểm để tạo track mới.

### `track_buffer`

Track bị mất detection chưa bị xóa ngay. Nó được giữ tối đa một số frame để chờ xuất hiện lại.

Ở khoảng 6 FPS:

```text
30 frame ~ 5 giây
```

Nhưng đây là quy đổi gần đúng theo processed frames.

### `match_thresh`

Ngưỡng cho bài toán ghép track-detection.

Không được nhầm với YOLO NMS IoU.

### `fuse_score`

Kết hợp confidence detection với thông tin khoảng cách/IoU khi tạo cost matching.

## 5.6. `persist=True` làm gì?

Code gọi `model.track()` riêng cho từng frame.

Nếu không persist, tracker có thể bị khởi tạo lại và mất lịch sử.

`persist=True` yêu cầu Ultralytics tái sử dụng tracker đã gắn với predictor, giúp ID nối qua các lần gọi.

## 5.7. ByteTrack có dùng ngoại hình không?

ByteTrack chuẩn trong cấu hình này:

- Không dùng ReID embedding.
- Không nhận biết áo, màu quần, khuôn mặt.
- Chủ yếu dựa Kalman motion và IoU.
- Không có camera motion compensation.

Do đó dễ đổi ID khi:

- Hai người che nhau.
- Hai người đi cắt nhau.
- Camera rung mạnh.
- Vật biến mất lâu.

## 5.8. Tầng stable ID của nhóm

Sau ByteTrack, nhóm thêm heuristic khoảng cách.

Đây không phải ByteTrack nguyên bản.

Cách trả lời:

> ByteTrack tạo raw track ID. Sau đó code của nhóm có lớp ổn định ID bằng khoảng cách giữa anchor hiện tại và vị trí cuối trong lịch sử, nhằm nối lại các đoạn track bị đứt.

## 5.9. Tại sao dùng ByteTrack và Optical Flow cùng lúc?

Hai phần có nhiệm vụ khác nhau:

```text
ByteTrack:
Giữ identity và bounding box qua nhiều frame.

Optical Flow:
Ước lượng hướng dịch chuyển cục bộ gần đây của điểm neo người.
```

Optical Flow không thay ByteTrack.

ByteTrack không cung cấp trực tiếp `flow_score` theo công thức của nhóm.

---

# 6. MOG2

## 6.1. Background subtraction là gì?

Mục tiêu:

```text
Frame hiện tại - mô hình nền = foreground mask
```

Không phải phép trừ pixel đơn giản, vì nền có:

- Nhiễu camera.
- Lá cây rung.
- Ánh sáng thay đổi.
- Bóng.
- Vật nền dao động.

MOG2 dùng mô hình thống kê thích nghi.

## 6.2. Vì sao dùng Gaussian Mixture?

Tại một pixel cố định, giá trị màu theo thời gian có thể có nhiều trạng thái.

Ví dụ pixel ở vị trí tán cây:

- Trạng thái lá sáng.
- Trạng thái lá tối.
- Trạng thái khoảng trời phía sau.

Một Gaussian duy nhất không đủ. MOG2 mô hình pixel bằng hỗn hợp nhiều Gaussian:

```text
P(X_t) = tổng w_k x Gaussian(X_t; mean_k, covariance_k)
```

Trong đó:

- `w_k`: mức phổ biến của trạng thái.
- `mean_k`: màu trung bình.
- `variance/covariance_k`: độ dao động.

Các thành phần ổn định, xuất hiện thường xuyên được xem là background.

Pixel mới không khớp background được đánh dấu foreground.

## 6.3. Ngưỡng Mahalanobis

MOG2 không chỉ dùng khoảng cách màu Euclidean.

Nó xét khoảng cách đã chuẩn hóa theo variance:

```text
distance^2 ~ (x - mean)^T covariance^-1 (x - mean)
```

OpenCV gọi `varThreshold` là ngưỡng trên squared Mahalanobis distance để quyết định pixel có được mô hình nền giải thích hay không.

Code:

```text
MOG2_THRESHOLD = 40
```

Ngưỡng nhỏ:

- Nhạy hơn.
- Nhiều foreground.
- Dễ nhiễu.

Ngưỡng lớn:

- Ít nhạy hơn.
- Nền ổn định hơn.
- Có thể bỏ chuyển động nhẹ.

## 6.4. `history=200`

OpenCV định nghĩa đây là số frame gần đây ảnh hưởng tới mô hình nền.

Ở 6 FPS:

```text
200 / 6 ~ 33,3 giây
```

Nhưng không nên gọi đây là cửa sổ cứng chính xác 33 giây. Mô hình cập nhật thích nghi và `learningRate=-1` để OpenCV tự chọn tốc độ học.

## 6.5. `detectShadows=False`

Code tắt shadow detection:

```python
detectShadows=False
```

Ưu điểm:

- Mask đơn giản chỉ foreground/background.
- Không phải xử lý giá trị mask riêng cho shadow.

Nhược điểm:

- Bóng chuyển động có thể bị coi là foreground.

## 6.6. Morphological opening

Code:

```python
fg = morphologyEx(fg, MORPH_OPEN, kernel)
```

Opening:

```text
Erosion -> Dilation
```

Tác dụng:

- Xóa chấm foreground nhỏ.
- Giảm nhiễu muối.
- Tách kết nối mảnh không mong muốn.

## 6.7. Morphological closing

Code:

```python
fg = morphologyEx(fg, MORPH_CLOSE, kernel)
```

Closing:

```text
Dilation -> Erosion
```

Tác dụng:

- Lấp lỗ nhỏ trong blob.
- Nối các vùng foreground gần nhau.

Kernel là ellipse 5x5.

## 6.8. Contour và centroid

Sau mask:

```python
findContours(...)
```

Contour có diện tích dưới 300 pixel bị loại.

Centroid được tính bằng moment:

```text
cx = M10 / M00
cy = M01 / M00
```

Đầu ra:

```python
(cx, cy, area)
```

## 6.9. MOG2 được dùng ở đâu?

### Dùng để boost score

Nếu có tâm motion trong vòng 50 pixel quanh rác:

```python
score = min(1.0, score x 1.15)
```

MOG2 không tự chọn owner. Nó chỉ tăng 15% score đã có.

### Dùng để phục hồi rác mất track

Nếu YOLO từng thấy rác ít nhất hai lần nhưng sau đó mất:

1. Tìm vị trí gần chân owner.
2. Nếu không có, tìm motion blob gần last position.
3. Nếu phù hợp vùng đất và bán kính, dùng nó làm vị trí phục hồi tạm.

Đây là heuristic, không phải MOG2 đã nhận diện được rác.

## 6.10. Điểm yếu MOG2

1. Camera rung làm gần như cả ảnh thành foreground.
2. Thay đổi ánh sáng đột ngột.
3. Bóng chuyển động.
4. Lá cây, mưa, màn hình nhấp nháy.
5. Vật đứng yên lâu có thể bị hấp thụ vào background.
6. Frame đầu tiên mô hình nền chưa ổn định.

---

# 7. Lucas-Kanade Optical Flow

## 7.1. Optical Flow là gì?

Optical Flow là trường vector biểu diễn chuyển động biểu kiến của cấu trúc ảnh giữa hai frame.

Nó không nhất thiết bằng vận tốc vật lý thật vì:

- Camera có thể chuyển động.
- Vật có thể thay đổi kích thước do tiến gần camera.
- Ánh sáng thay đổi.
- Chiếu phối cảnh làm biến dạng chuyển động.

## 7.2. Giả thiết brightness constancy

Một điểm vật thể giữ cường độ gần như không đổi:

```text
I(x, y, t) = I(x + u, y + v, t + dt)
```

Trong đó:

- `(x,y)` là vị trí cũ.
- `(u,v)` là dịch chuyển.

Khai triển Taylor bậc một:

```text
Ix*u + Iy*v + It = 0
```

Đây là phương trình Optical Flow constraint.

## 7.3. Vì sao một pixel không đủ?

Một phương trình:

```text
Ix*u + Iy*v = -It
```

nhưng có hai ẩn `u`, `v`.

Đây là aperture problem.

Lucas-Kanade giả sử các pixel trong một cửa sổ nhỏ có cùng chuyển động.

Với nhiều pixel:

```text
A [u v]^T = b
```

Giải least squares:

```text
[u v]^T = (A^T A)^-1 A^T b
```

Muốn giải ổn định, cửa sổ phải có gradient theo ít nhất hai hướng rõ. Góc/corner tốt hơn vùng phẳng hoặc chỉ có một cạnh thẳng.

## 7.4. Vì sao dùng pyramid?

Lucas-Kanade cơ bản dựa trên giả thiết chuyển động nhỏ.

Nếu vật di chuyển nhiều pixel:

1. Thu nhỏ ảnh ở level cao.
2. Chuyển động lớn ở ảnh gốc trở thành chuyển động nhỏ.
3. Ước lượng ở ảnh thô.
4. Phóng và tinh chỉnh dần xuống ảnh gốc.

Đây là coarse-to-fine.

Code:

```text
maxLevel = 2
```

Tức dùng level 0, 1, 2.

## 7.5. `winSize=(15,15)`

Đây là cửa sổ tìm kiếm tại mỗi pyramid level.

Cửa sổ lớn:

- Chịu nhiễu tốt hơn.
- Theo dõi chuyển động lớn tốt hơn.
- Nhưng có thể trộn nhiều vật/chuyển động.
- Chậm hơn.

Cửa sổ nhỏ:

- Chính xác cục bộ.
- Nhanh.
- Dễ mất điểm khi chuyển động lớn hoặc texture yếu.

## 7.6. Termination criteria

```python
(COUNT | EPS, 10, 0.03)
```

Thuật toán dừng khi:

- Đủ 10 vòng lặp; hoặc
- Thay đổi nhỏ hơn epsilon 0.03.

## 7.7. `status` và `err`

OpenCV trả:

```text
nextPts
status
err
```

`status=1`: tìm được flow cho điểm.

`status=0`: không tìm được.

Code dùng `status`, nhưng bỏ qua `err`.

Do đó code chưa lọc thêm theo độ lỗi tracking.

## 7.8. Code đang theo dõi điểm nào?

Điểm neo gần chân của person:

```python
(cx, y1 + 0.88 x bbox_height)
```

Frame đầu:

```python
flow_pts[pid] = anchor
```

Frame sau:

```python
pts_new = calcOpticalFlowPyrLK(prev_gray, curr_gray, pts_old)
vector = pts_new - pts_old
```

Sau đó code đặt lại:

```python
flow_pts[pid] = current YOLO anchor
```

Ý nghĩa:

- LK chỉ đo dịch chuyển cục bộ quanh anchor giữa hai frame.
- YOLO anchor hiệu chỉnh lại điểm gốc mỗi frame, giảm drift tích lũy.
- Nhưng vector có thể nhiễu nếu chân nằm trên vùng ít texture hoặc bị che.

## 7.9. Vì sao code dùng `avg_vx`, `avg_vy` dù chỉ có một điểm?

Hàm OpenCV hỗ trợ nhiều điểm nên code viết tổng quát:

```python
vecs = good_new - good_old
avg_vx = mean(vecs[:,0])
avg_vy = mean(vecs[:,1])
```

Hiện mỗi person thường chỉ có một điểm, nên mean bằng chính vector đó.

## 7.10. Flow score được tính thế nào?

Lấy tối đa bốn vector mới nhất:

```python
recent_flows = flow_vecs[pid][-4:]
```

Tính vector trung bình:

```text
flow = (avg_vx, avg_vy)
```

Tạo vector từ rác tới người:

```text
away = latest_person_position - trash_position
```

Chuẩn hóa hai vector rồi tính dot product:

```text
flow_score = max(0, flow_unit dot away_unit)
```

Ý nghĩa:

```text
1.0: di chuyển gần như thẳng ra xa rác
0.0: vuông góc, đứng yên hoặc đi về phía rác
```

Nếu dot product âm, code cắt về 0.

## 7.11. Hạn chế Optical Flow hiện tại

1. Chỉ một điểm mỗi person.
2. Không chọn corner tốt.
3. Không kiểm tra `err`.
4. Không bù chuyển động camera.
5. Xóa flow ngay khi person mất detection.
6. Sampling 6 FPS làm chuyển động giữa hai processed frame lớn hơn.
7. Điểm chân có thể nằm trên nền thay vì trên người.

Vì vậy flow chỉ được đặt trọng số 20%, không phải căn cứ duy nhất.

---

# 8. Phân biệt những khái niệm thầy dễ hỏi lẫn

## YOLO và ByteTrack

```text
YOLO: phát hiện vật trong từng frame.
ByteTrack: nối detection qua nhiều frame và gán ID.
```

## ByteTrack và stable ID heuristic

```text
ByteTrack: tracker chuẩn từ Ultralytics.
Stable ID: code nhóm tự nối lại bằng khoảng cách.
```

## ByteTrack và Optical Flow

```text
ByteTrack: identity và trajectory bbox.
Optical Flow: hướng chuyển động cục bộ để tạo flow_score.
```

## Direction score và Flow score

```text
Direction:
Dùng lịch sử anchor, lấy dịch chuyển từ giữa lịch sử tới vị trí mới nhất.

Flow:
Dùng biến đổi cường độ ảnh giữa các frame qua Lucas-Kanade.
```

Hai tín hiệu có liên quan nhưng không giống nhau.

## IoU NMS và IoU tracking

```text
NMS IoU:
Loại box trùng trong một frame.

Tracking association:
Ghép box frame hiện tại với track dự đoán từ frame trước.
```

## MOG2 và Optical Flow

```text
MOG2:
Cho biết vùng nào khác nền, không tạo identity.

Optical Flow:
Ước lượng vector dịch chuyển của điểm.
```

## Detection confidence và Ownership Score

```text
Detection confidence:
YOLO tin box/class tới mức nào.

Ownership Score:
Luật của hệ thống đánh giá một người liên quan tới rác tới mức nào.
```

Ownership Score không phải confidence của YOLO.

---

# 9. Một ví dụ đầy đủ trước khi tính điểm owner

Giả sử frame 40:

```text
YOLO:
Person raw ID 5:
box = [200, 300, 320, 650]
conf = 0.84

Trash raw ID 11:
box = [300, 610, 340, 650]
conf = 0.62
```

## Person anchor

```text
cx = (200+320)/2 = 260
height = 650-300 = 350
anchor_y = 300 + 0.88x350 = 608
foot_y = 650
```

Lưu:

```python
current_persons[5] = (260, 608)

person_history[5].append({
    "anchor": (260,608),
    "points": [(260,608),(260,650)],
    "frame": 40
})
```

## Trash center

```text
cx = (300+340)/2 = 320
cy = (610+650)/2 = 630
```

Lưu:

```python
current_trashes[11] = (320,630)
```

## MOG2

Giả sử có blob:

```python
(325, 625, 900)
```

Khoảng cách tới trash:

```text
sqrt((325-320)^2 + (625-630)^2)
~7.1 pixel
```

Nhỏ hơn `MOG2_BOOST_RADIUS=50`, nên sau này score ứng viên có thể được nhân 1.15.

## Optical Flow

Frame trước anchor khoảng:

```text
(250, 605)
```

Lucas-Kanade tìm điểm mới:

```text
(258, 607)
```

Flow:

```text
(8,2)
```

Vector được lưu:

```python
flow_vecs[5].append((8,2))
```

Sau tất cả bước trên, Ownership Score mới nhận:

```python
trash_center=(320,630)
current_persons={5:(260,608), ...}
person_history={...}
flow_vecs={5:[..., (8,2)]}
mog2_alerts=[(325,625,900), ...]
current_frame=40
```

---

# 10. Bộ câu hỏi vấn đáp và câu trả lời mẫu

## Câu 1: YOLO trong đề tài làm nhiệm vụ gì?

> YOLOv8s thực hiện object detection hai lớp person và trash trong từng frame. Nó trả bounding box, class và confidence. YOLO không tự kết luận hành vi và cũng không tự xác định người vi phạm.

## Câu 2: Nhóm dùng phiên bản YOLO nào?

> Trọng số được fine-tune từ YOLOv8s pretrained, input train 640, hai lớp person và trash.

## Câu 3: Tại sao chọn YOLOv8s?

> Bản small cân bằng tốc độ và độ chính xác. Nano nhanh hơn nhưng khả năng biểu diễn yếu hơn; các bản medium, large nặng hơn, không phù hợp mục tiêu chạy CPU khoảng 6 FPS.

## Câu 4: YOLO khác classification thế nào?

> Classification chỉ dự đoán nhãn cho toàn ảnh. Detection dự đoán cả nhãn và vị trí bằng bounding box, đồng thời có thể phát hiện nhiều vật trong một ảnh.

## Câu 5: Confidence có phải xác suất chính xác tuyệt đối không?

> Không. Nó là điểm tin cậy dùng để xếp hạng và lọc detection, không nhất thiết là xác suất đã hiệu chuẩn.

## Câu 6: IoU là gì?

> Là tỉ lệ diện tích giao trên diện tích hợp của hai bounding box, nằm từ 0 đến 1.

## Câu 7: NMS làm gì?

> NMS loại các box dự đoán trùng nhau quanh cùng một vật, ưu tiên box confidence cao.

## Câu 8: `iou=0.5` trong code có phải ngưỡng ByteTrack không?

> Không. Đây là IoU dùng trong post-processing/NMS của YOLO. ByteTrack có `match_thresh=0.8` riêng trong YAML.

## Câu 9: YOLO có gán ID không?

> YOLO detect thuần không duy trì ID qua thời gian. Trong code, `model.track()` kết hợp detector YOLO với ByteTrack, vì vậy kết quả `boxes.id` đến từ tracker.

## Câu 10: ByteTrack hoạt động theo nguyên lý nào?

> Đây là tracking-by-detection. Nó dùng Kalman Filter dự đoán track, sau đó ghép detection bằng chi phí dựa trên vị trí/IoU. Nó ghép hai lượt: detection điểm cao trước, rồi dùng detection điểm thấp để cứu các track chưa khớp.

## Câu 11: Tại sao ByteTrack dùng detection confidence thấp?

> Vật thật khi che khuất có thể bị giảm confidence. Nếu bỏ toàn bộ box thấp, trajectory dễ đứt. ByteTrack dùng quan hệ với track đã tồn tại để phân biệt box thấp nào có khả năng là vật thật.

## Câu 12: Dự án có tận dụng đầy đủ lượt thấp của ByteTrack không?

> Live mode có vì YOLO conf là 0.12, thấp hơn high threshold 0.25. File mode conf là 0.25 nên gần như không truyền nhóm box 0.10-0.25 vào tracker.

## Câu 13: `persist=True` để làm gì?

> Giữ tracker qua các lần gọi `model.track(frame)` liên tiếp. Nếu tracker bị reset mỗi frame thì ID không thể nối theo thời gian.

## Câu 14: ByteTrack có dùng nhận dạng ngoại hình không?

> Không trong cấu hình hiện tại. Nó không dùng ReID; chủ yếu dùng Kalman motion và IoU. Vì vậy có thể đổi ID khi người che nhau hoặc camera rung.

## Câu 15: Tại sao code vẫn cần matching khoảng cách sau ByteTrack?

> Để nối lại lịch sử khi tracker bị đổi hoặc mất ID ngắn hạn. Đây là heuristic bổ sung của nhóm, không phải thành phần nguyên bản của ByteTrack.

## Câu 16: Nhược điểm matching khoảng cách là gì?

> Có thể gộp nhầm hai người nếu họ ở gần nhau hoặc đi cắt nhau, vì không dùng đặc trưng ngoại hình.

## Câu 17: Tại sao dùng điểm gần chân?

> Rác bị bỏ lại thường liên quan tới vị trí tiếp xúc mặt đất gần chân. Tâm bbox người có thể ở ngực/bụng và làm sai khoảng cách người-rác.

## Câu 18: MOG2 là gì?

> Là phương pháp background subtraction dựa trên Gaussian Mixture Model thích nghi theo từng pixel. Nó mô hình hóa nền và tạo foreground mask cho các pixel không phù hợp nền.

## Câu 19: Tại sao một pixel cần nhiều Gaussian?

> Một vị trí nền có thể có nhiều trạng thái hợp lệ, ví dụ lá cây sáng/tối hoặc màn hình thay đổi. Mixture model biểu diễn được nền đa trạng thái tốt hơn một Gaussian.

## Câu 20: `varThreshold=40` có ý nghĩa gì?

> Đây là ngưỡng trên squared Mahalanobis distance để xét pixel có được background model giải thích không. Giảm ngưỡng làm nhạy hơn nhưng tăng nhiễu.

## Câu 21: `history=200` có phải lưu đúng 200 frame ảnh không?

> Nó là số frame ảnh hưởng đến mô hình nền, không nhất thiết lưu nguyên 200 ảnh. Ở 6 FPS có thể hiểu xấp xỉ 33 giây ảnh hưởng.

## Câu 22: Opening và closing dùng để làm gì?

> Opening xóa nhiễu foreground nhỏ. Closing lấp lỗ và nối các vùng foreground gần nhau.

## Câu 23: MOG2 có nhận diện được rác không?

> Không. Nó chỉ phát hiện vùng khác nền. Class person/trash vẫn do YOLO quyết định.

## Câu 24: MOG2 tham gia owner score thế nào?

> Nếu có motion blob trong vòng 50 pixel quanh rác, score ứng viên dương được tăng tối đa 15%.

## Câu 25: MOG2 có nhược điểm gì?

> Nhạy với camera rung, thay đổi ánh sáng, bóng, lá cây và giai đoạn đầu chưa học nền ổn định.

## Câu 26: Optical Flow là gì?

> Là chuyển động biểu kiến của cấu trúc ảnh giữa hai frame, biểu diễn bằng vector dịch chuyển.

## Câu 27: Giả thiết chính của Lucas-Kanade?

> Brightness constancy, chuyển động nhỏ và chuyển động gần như đồng nhất trong một cửa sổ cục bộ.

## Câu 28: Phương trình Optical Flow constraint?

```text
Ix*u + Iy*v + It = 0
```

## Câu 29: Vì sao một pixel không đủ tìm u và v?

> Vì có một phương trình nhưng hai ẩn. Lucas-Kanade lấy nhiều pixel trong cửa sổ và giải least squares.

## Câu 30: Aperture problem là gì?

> Qua một cửa sổ chỉ nhìn thấy cạnh thẳng, ta thường chỉ xác định rõ thành phần chuyển động vuông góc cạnh, không xác định đầy đủ chuyển động hai chiều.

## Câu 31: Vì sao corner tốt cho Optical Flow?

> Corner có gradient theo hai hướng, làm ma trận phương trình đủ thông tin và nghiệm ổn định hơn vùng phẳng hoặc một cạnh.

## Câu 32: Pyramid giải quyết gì?

> Nó hỗ trợ chuyển động lớn bằng cách ước lượng từ ảnh thu nhỏ rồi tinh chỉnh dần ở độ phân giải cao.

## Câu 33: `maxLevel=2` có bao nhiêu level?

> Ba: level 0, 1 và 2.

## Câu 34: Code dùng sparse hay dense Optical Flow?

> Sparse Optical Flow, vì chỉ tính cho một tập điểm rời rạc; cụ thể hiện tại là một điểm neo mỗi person.

## Câu 35: Flow có giữ ID không?

> Không. ID do ByteTrack và stable matching đảm nhiệm. Flow chỉ tạo vector hướng chuyển động.

## Câu 36: Tại sao flow chỉ chiếm 20%?

> Flow có thể nhiễu do texture yếu, che khuất, rung camera và chỉ theo dõi một điểm. Nó là tín hiệu bổ sung chứ không đủ để kết luận owner.

## Câu 37: Direction score và flow score khác nhau thế nào?

> Direction dùng quỹ đạo tọa độ anchor đã lưu. Flow dùng thay đổi cường độ ảnh giữa các frame qua Lucas-Kanade.

## Câu 38: Nếu camera rung thì chuyện gì xảy ra?

> MOG2 có thể tạo foreground toàn ảnh, Optical Flow chứa chuyển động của camera, ByteTrack dễ đổi ID vì ByteTrack không có camera motion compensation. Đây là hạn chế của cấu hình hiện tại.

## Câu 39: Tại sao không dùng MOG2 thay YOLO để tiết kiệm?

> MOG2 không biết class. Nó không phân biệt người, rác, bóng hay lá cây. YOLO cung cấp ngữ nghĩa, MOG2 chỉ cung cấp tín hiệu chuyển động.

## Câu 40: Tại sao không chọn người gần rác nhất?

> Người vứt rác có thể đã rời đi, còn người khác vừa đi tới đứng gần rác. Vì vậy cần lịch sử, hướng chuyển động, flow và recency.

## Câu 41: Tại sao xử lý 6 FPS?

> Để giảm tải CPU và giữ luồng hiển thị ổn định. Đổi lại, hành động nhanh hoặc detection chỉ xuất hiện ở vài frame có thể bị bỏ sót.

## Câu 42: Nếu tăng FPS thì các ngưỡng frame có còn đúng không?

> Không. Nếu tăng FPS xử lý mà giữ nguyên CONFIRM_FRAMES và STATIONARY_REQUIRED, thời gian thực tương ứng sẽ ngắn hơn. Các ngưỡng phải được quy đổi lại theo FPS.

## Câu 43: Model tốt tới mức nào?

> Checkpoint có precision khoảng 0.872, recall 0.739, mAP50 0.805 và mAP50-95 0.449 trên validation lúc train. Đây là metrics detection, không phải độ chính xác cuối của hành vi.

## Câu 44: Tại sao accuracy hành vi không bằng mAP YOLO?

> Vì hành vi còn phụ thuộc tracking, ID, lifecycle, ground condition, owner score và rule xác nhận. Detection tốt chưa đảm bảo gán thủ phạm đúng.

## Câu 45: Hệ thống có thực sự nhận diện động tác ném rác không?

> Không trực tiếp. Hệ thống suy luận từ rác xuất hiện, quỹ đạo người, chuyển động, thời gian tồn tại và trạng thái đứng yên. Muốn hiểu động tác tay rõ hơn cần pose estimation hoặc action recognition.

---

# 11. Bài trình bày miệng khoảng 3 phút

> Phần em phụ trách là tầng quan sát trước khi tính Ownership Score. Đầu tiên, video được lấy mẫu khoảng 6 FPS để phù hợp tốc độ CPU. Với mỗi frame, hệ thống chạy song song theo nghĩa logic hai nhánh. Nhánh thứ nhất là MOG2, dùng Gaussian Mixture Model thích nghi theo từng pixel để xây dựng nền và tạo foreground mask. Sau morphology open-close và lọc contour dưới 300 pixel, nhánh này trả về tâm các vùng chuyển động. MOG2 không nhận diện được người hay rác; nó chỉ là tín hiệu phụ.
>
> Nhánh thứ hai dùng model YOLOv8s đã fine-tune hai lớp person và trash. YOLO trả bounding box, class và confidence. Trong `model.track`, kết quả detection được đưa qua ByteTrack để duy trì ID. ByteTrack là tracking-by-detection, dùng Kalman Filter dự đoán vị trí track và ghép detection theo hai lượt: confidence cao trước, confidence thấp sau để cứu các vật bị che khuất. Cấu hình hiện tại dùng high threshold 0.25, low threshold 0.1 và track buffer 30 frame.
>
> Sau ByteTrack, nhóm có thêm matching khoảng cách để nối ID khi tracker bị đứt. Với người, hệ thống không dùng tâm cơ thể mà dùng anchor ở 88% chiều cao bbox, gần vị trí chân. Anchor và điểm đáy bbox được lưu trong lịch sử tối đa 90 processed frame. Với rác, hệ thống dùng tâm bounding box.
>
> Tiếp theo hệ thống chuyển frame sang grayscale và dùng Pyramidal Lucas-Kanade để ước lượng vector chuyển động của một điểm neo cho mỗi người. Phương pháp dựa trên giả thiết độ sáng gần như không đổi và chuyển động cục bộ nhỏ. Code dùng cửa sổ 15x15 và maxLevel 2, tức ba mức pyramid. Vector được lưu tối đa 8 frame và chỉ là tín hiệu hướng; nó không đảm nhiệm giữ ID.
>
> Cuối tầng này, Ownership Score nhận tâm rác, vị trí và lịch sử người, vector Optical Flow và các tâm chuyển động MOG2. Nhờ vậy tầng tính điểm không làm việc trực tiếp trên ảnh mà trên các đặc trưng quan sát đã được mã hóa.

---

# 12. Các câu tuyệt đối không nên trả lời sai

Không nói:

> YOLO phát hiện thủ phạm.

Phải nói:

> YOLO chỉ phát hiện person và trash.

Không nói:

> ByteTrack nhận diện người.

Phải nói:

> ByteTrack liên kết detection và duy trì ID.

Không nói:

> MOG2 nhận diện rác.

Phải nói:

> MOG2 phát hiện foreground so với background model.

Không nói:

> Optical Flow giữ ID.

Phải nói:

> Optical Flow cung cấp vector chuyển động cục bộ.

Không nói:

> Confidence là xác suất đúng tuyệt đối.

Phải nói:

> Confidence là điểm tin cậy dùng để lọc/xếp hạng.

Không nói:

> `maxLevel=2` là hai ảnh pyramid.

Phải nói:

> Có level 0, 1, 2.

Không nói:

> Score ví dụ trên slide là 0.65.

Phải nói:

> Tính đúng là 0.67.

---

# 13. Nguồn lý thuyết chính

- YOLO gốc: https://arxiv.org/abs/1506.02640
- Ultralytics object detection: https://docs.ultralytics.com/tasks/detect/
- Ultralytics tracking: https://docs.ultralytics.com/modes/track/
- Kiến trúc YOLOv8 chính thức: https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/models/v8/yolov8.yaml
- ByteTrack paper: https://arxiv.org/abs/2110.06864
- OpenCV MOG2: https://docs.opencv.org/4.x/d7/d7b/classcv_1_1BackgroundSubtractorMOG2.html
- OpenCV Pyramidal Lucas-Kanade: https://docs.opencv.org/4.x/dc/d6b/group__video__track.html
- Lucas-Kanade 1981: https://publications.ri.cmu.edu/storage/publications/pub_files/pub3/lucas_bruce_d_1981_1/lucas_bruce_d_1981_1.pdf

