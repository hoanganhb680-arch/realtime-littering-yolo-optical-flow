# BÁO CÁO HỆ THỐNG PHÁT HIỆN HÀNH VI VỨT RÁC BỪA BÃI

## 1. Tên đề tài

**Xây dựng hệ thống giám sát và phát hiện hành vi vứt rác bừa bãi từ video và camera thời gian thực bằng YOLO, ByteTrack và xử lý ảnh.**

## 2. Tóm tắt

Đề tài xây dựng một hệ thống giám sát có khả năng nhận diện người, nhận diện rác, theo dõi vị trí của các đối tượng theo thời gian và xác định hành vi vứt rác bừa bãi. Hệ thống hỗ trợ hai nguồn đầu vào: video có sẵn và camera IP thời gian thực. Backend sử dụng Python, OpenCV, YOLO và FastAPI; frontend sử dụng React để hiển thị luồng camera, cảnh báo vi phạm và lịch sử vi phạm.

Điểm chính của hệ thống không chỉ là phát hiện vật thể rác trong từng frame riêng lẻ, mà còn theo dõi lịch sử chuyển động của người và rác. Khi một vật thể rác mới xuất hiện, hệ thống tính điểm liên hệ giữa rác và từng người trong khung hình thông qua khoảng cách, hướng di chuyển, optical flow và độ gần về thời gian. Sau đó hệ thống chỉ xác nhận vi phạm khi rác đủ ổn định, người liên quan đủ tin cậy, vị trí rác nằm gần mặt đất và các điều kiện thời gian được thỏa mãn.

## 3. Mục tiêu đề tài

Mục tiêu tổng quát là xây dựng một hệ thống tự động phát hiện, cảnh báo và lưu bằng chứng đối với hành vi vứt rác bừa bãi.

Các mục tiêu cụ thể:

- Nhận diện người và rác trong video bằng mô hình YOLO đã huấn luyện.
- Theo dõi ID của người và rác qua nhiều frame bằng ByteTrack và cơ chế matching bổ sung.
- Xác định người có khả năng là chủ thể tạo ra vật rác dựa trên lịch sử vị trí và chuyển động.
- Phân loại một số tình huống vi phạm: bỏ rác đột ngột, rác đứng yên sau khi chủ thể rời đi, bỏ rác tại chỗ.
- Lưu ảnh/video bằng chứng và ghi nhận dữ liệu vào SQLite/MinIO.
- Hiển thị kết quả lên giao diện web theo thời gian thực qua WebSocket.
- Hỗ trợ cả video file và camera IP real-time trong cùng một pipeline.

## 4. Phạm vi hệ thống

Hệ thống hiện tại tập trung vào bài toán giám sát một luồng video/camera tại một thời điểm. Nguồn đầu vào có thể là:

- **Video file**: mặc định là `video/di_bo_17.mp4` khi chạy ở chế độ file.
- **Camera IP**: mặc định lấy stream MJPEG từ `http://192.168.0.108:8080/video`.

Mô hình nhận diện đang sử dụng là:

```text
weights/best.pt
```

Mô hình này được load trong `Config.py` thông qua:

```python
MODEL_PATH = str(_ROOT / "weights" / "best.pt")
```

## 5. Cơ sở lý thuyết

### 5.1. Video số và xử lý theo frame

Video số là một chuỗi các ảnh tĩnh liên tiếp gọi là frame. Nếu video có tốc độ 30 FPS, nghĩa là trong 1 giây có 30 frame. Khi xử lý video bằng OpenCV, hệ thống không cần tách video thành các ảnh lưu trên ổ đĩa. Thay vào đó, `cv2.VideoCapture` đọc từng frame trực tiếp trong RAM:

```text
Video/Camera -> frame_1 -> frame_2 -> ... -> frame_n
```

Đối với bài toán vứt rác, hành vi thường diễn ra trong khoảng vài giây, không cần xử lý đủ 25-30 FPS như video gốc. Nếu xử lý quá nhiều frame, hệ thống sẽ chậm và dễ bị trễ trong chế độ real-time. Vì vậy hệ thống chuẩn hóa tốc độ xử lý về khoảng 6 FPS.

Quan hệ giữa số frame và thời gian:

```text
Thời gian (giây) = Số frame / FPS xử lý
Số frame = Thời gian mong muốn x FPS xử lý
```

Ví dụ với `FPS xử lý = 6`:

```text
12 frame ≈ 2 giây
90 frame ≈ 15 giây
60 frame ≈ 10 giây
```

Đây là cơ sở quan trọng để chọn các tham số như `CONFIRM_FRAMES`, `HISTORY_FRAMES`, `STALE_FRAMES`.

### 5.2. Cơ sở chọn ngưỡng thời gian theo FPS

Trong bài toán phát hiện hành vi, tham số quan trọng không phải chỉ là số frame, mà là **khoảng thời gian hành vi tồn tại trong thực tế**. Vì chương trình xử lý theo frame, nên các ngưỡng thời gian phải được chuyển đổi sang frame bằng công thức:

```text
Số frame cần kiểm tra = Thời gian quan sát mong muốn x FPS xử lý
```

Ví dụ hệ thống xử lý ở 6 FPS:

```text
1 giây  = 6 frame
2 giây  = 12 frame
10 giây = 60 frame
15 giây = 90 frame
```

Do đó, khi đặt `CONFIRM_FRAMES = 12`, ý nghĩa thực tế không phải là "12 frame" một cách tùy ý, mà là **rác phải tồn tại ổn định khoảng 2 giây** trước khi được xác nhận. Cách đặt này giúp hệ thống tránh kết luận sai từ các detection chớp nhoáng.

#### 5.2.1. Vì sao chọn FPS xử lý khoảng 6 FPS?

Camera hoặc video gốc thường có 25-30 FPS, nhưng bài toán vứt rác không yêu cầu phân tích chuyển động rất nhanh. Hành vi vứt rác thường diễn ra theo đơn vị **giây**, gồm các giai đoạn:

```text
người đi tới / đứng gần -> rác xuất hiện -> rác nằm lại -> người rời đi
```

Các giai đoạn này thường kéo dài từ khoảng 1 đến vài giây, không phải vài chục mili-giây. Vì vậy xử lý 6 FPS vẫn đủ quan sát diễn biến hành vi:

```text
6 FPS nghĩa là cứ khoảng 0.167 giây có 1 frame xử lý
```

Khoảng 0.167 giây/frame đủ mịn để thấy người di chuyển, rác xuất hiện và rác nằm lại. Đồng thời, 6 FPS giúp giảm tải vì mỗi frame phải chạy nhiều bước nặng:

- YOLO detection.
- ByteTrack tracking.
- MOG2 motion detection.
- Optical flow.
- Vẽ frame.
- Encode JPEG.
- Gửi WebSocket.
- Lưu evidence nếu có vi phạm.

Nếu xử lý 25-30 FPS, hệ thống có thể mượt hơn nhưng chi phí tính toán tăng khoảng 4-5 lần. Khi backend không xử lý kịp, luồng real-time bị trễ, tức là hệ thống đang xem frame cũ thay vì frame mới. Với bài toán giám sát, **độ trễ thấp quan trọng hơn độ mượt tuyệt đối**.

#### 5.2.2. Vì sao không xác nhận rác ngay khi vừa phát hiện?

YOLO có thể tạo false positive trong một vài frame do:

- Bóng đổ hoặc vùng sáng giống rác.
- Chân người, túi xách, vật nhỏ bị nhận nhầm là rác.
- Bounding box nhấp nháy khi vật nhỏ hoặc bị che khuất.
- Motion blur khi camera rung.

Nếu chỉ cần thấy rác 1 frame là kết luận vi phạm, hệ thống sẽ rất dễ báo sai. Vì vậy hệ thống dùng nguyên tắc **temporal persistence**: một đối tượng chỉ đáng tin hơn nếu nó tồn tại qua nhiều frame liên tiếp hoặc gần liên tiếp.

Với `CONFIRM_FRAMES = 12` ở 6 FPS:

```text
12 / 6 = 2 giây
```

Tức là rác phải được theo dõi khoảng 2 giây. Trong thực tế, rác bị vứt xuống đất sẽ nằm lại trong nhiều giây, còn nhiễu detection thường chỉ xuất hiện thoáng qua. Đây là lý do chọn ngưỡng khoảng 2 giây.

#### 5.2.3. Vì sao rác phải đứng yên khoảng 1.67 giây?

Thông số:

```python
STATIONARY_REQUIRED = 10
```

Với 6 FPS:

```text
10 / 6 ≈ 1.67 giây
```

Rác sau khi bị bỏ xuống thường có đặc điểm là **nằm lại tại một vị trí gần như cố định**. Ngược lại, các vật thể không phải rác hoặc nhiễu thường có một trong các hiện tượng:

- Di chuyển theo người.
- Chỉ xuất hiện ở 1-2 frame rồi biến mất.
- Bounding box thay đổi mạnh.
- Là một phần của cơ thể người hoặc đồ vật đang được mang theo.

Vì vậy hệ thống yêu cầu rác đứng yên khoảng 1.67 giây để phân biệt:

```text
rác thật nằm lại trên sàn
```

với:

```text
vật thể đang chuyển động / detection nhầm / bbox nhấp nháy
```

Khoảng 1.67 giây được chọn vì đủ dài để lọc nhiễu ngắn hạn, nhưng không quá dài khiến cảnh báo bị chậm. Nếu đặt 0.5 giây, hệ thống dễ báo nhầm. Nếu đặt 5 giây, hệ thống phản hồi chậm và có thể bỏ lỡ đoạn bằng chứng quan trọng.

#### 5.2.4. Vì sao owner phải rời đi tối thiểu khoảng 1 giây?

Thông số:

```python
MIN_OWNER_GONE_FRAMES = 6
```

Với 6 FPS:

```text
6 / 6 = 1 giây
```

Trong hành vi vứt rác, việc người rời khỏi vị trí rác là dấu hiệu quan trọng. Tuy nhiên, ở từng frame đơn lẻ, người có thể bị YOLO mất detection tạm thời do che khuất hoặc nhòe ảnh. Nếu hệ thống coi owner "đã rời đi" ngay khi mất detection 1 frame, sẽ dễ báo sai.

Do đó cần đợi owner vắng mặt ít nhất khoảng 1 giây. Ngưỡng này có ý nghĩa:

- Đủ dài để bỏ qua mất detection tạm thời.
- Đủ ngắn để không làm cảnh báo chậm quá nhiều.
- Phù hợp với hành vi thực tế: sau khi bỏ rác, người thường bắt đầu rời khỏi vùng rác trong khoảng vài giây.

#### 5.2.5. Vì sao lưu lịch sử người 15 giây?

Thông số:

```python
HISTORY_FRAMES = 90
```

Với 6 FPS:

```text
90 / 6 = 15 giây
```

Rác có thể không được YOLO phát hiện đúng tại thời điểm người vừa thả xuống. Có thể sau vài frame hoặc vài giây, rác mới được nhận diện rõ do:

- Người che rác lúc mới bỏ.
- Rác nhỏ, confidence ban đầu thấp.
- Camera rung hoặc motion blur.
- Rác chỉ rõ khi người bước ra.

Vì vậy hệ thống cần lưu lịch sử vị trí người trong một khoảng trước đó. 15 giây đủ để truy ngược người từng đi qua/đứng gần vị trí rác, nhưng không quá dài để gán nhầm một người đã đi qua từ rất lâu.

#### 5.2.6. Nguyên tắc chung khi thay đổi FPS

Các thông số dạng frame phải thay đổi theo FPS. Nếu tăng FPS xử lý mà giữ nguyên số frame, thời gian quan sát thực tế sẽ bị ngắn lại.

Ví dụ:

```text
CONFIRM_FRAMES = 12
```

- Ở 6 FPS: 12 frame = 2 giây.
- Ở 12 FPS: 12 frame = 1 giây.
- Ở 30 FPS: 12 frame = 0.4 giây.

Vì vậy nếu muốn giữ ý nghĩa "xác nhận sau khoảng 2 giây", cần tính:

```text
CONFIRM_FRAMES = 2 x FPS xử lý
```

Ví dụ:

| FPS xử lý | `CONFIRM_FRAMES` nên dùng để giữ 2 giây |
|---:|---:|
| 6 FPS | 12 |
| 10 FPS | 20 |
| 12 FPS | 24 |
| 30 FPS | 60 |

Do đó, các giá trị trong config hiện tại được hiểu theo giả định hệ thống xử lý quanh **6 FPS**.

### 5.3. Object Detection bằng YOLO

YOLO là mô hình phát hiện vật thể một giai đoạn. Thay vì tách riêng bước đề xuất vùng và phân loại, YOLO dự đoán trực tiếp bounding box, class và confidence score từ ảnh đầu vào. Trong hệ thống này, YOLO được dùng để phát hiện hai lớp chính:

- `person`: người.
- `trash`: rác.

Đầu ra của YOLO gồm:

- Bounding box: tọa độ vùng chứa đối tượng.
- Class ID: loại đối tượng.
- Confidence: độ tin cậy của dự đoán.

Trong code, YOLO được gọi tại `TrashViolationDetector._process_frame()`:

```python
results = model.track(frame, **track_kwargs)
```

Với chế độ live, tham số YOLO chính là:

```python
LIVE_YOLO_IMGSZ = 512
LIVE_YOLO_CONF = 0.12
LIVE_YOLO_IOU = 0.50
```

Với video file, hệ thống dùng `imgsz = 640` và `conf = 0.25`.

### 5.4. Tracking bằng ByteTrack

Phát hiện vật thể ở từng frame riêng lẻ chưa đủ để kết luận hành vi. Hệ thống cần biết người nào xuất hiện ở frame trước và frame sau có phải cùng một người hay không. Vì vậy mô hình được chạy với:

```python
persist=True
tracker="bytetrack.yaml"
```

ByteTrack giúp duy trì ID đối tượng qua nhiều frame. Khi YOLO/ByteTrack mất ID hoặc ID không ổn định, hệ thống bổ sung cơ chế matching dựa trên khoảng cách vị trí lịch sử trong `detection_parsing.py`.

### 5.5. Background subtraction bằng MOG2

MOG2 là phương pháp tách nền động. Nó mô hình hóa nền của video và phát hiện vùng có thay đổi so với nền. Trong hệ thống này, MOG2 không thay thế YOLO, mà đóng vai trò hỗ trợ:

- Phát hiện vùng có chuyển động gần rác.
- Bổ sung người giả lập từ motion blob khi YOLO bỏ sót người.
- Tăng điểm owner nếu có chuyển động gần vị trí rác.

Thông số MOG2 hiện tại:

```python
MOG2_HISTORY = 200
MOG2_THRESHOLD = 40
MOG2_MIN_AREA = 300
```

Ý nghĩa:

- `MOG2_HISTORY = 200`: số frame dùng để học nền. Với 6 FPS, 200 frame tương đương khoảng 33 giây. Khoảng này đủ dài để nền ổn định, nhưng vẫn thích nghi nếu ánh sáng/cảnh thay đổi chậm.
- `MOG2_THRESHOLD = 40`: ngưỡng sai khác để coi một pixel là foreground. Giá trị vừa phải, tránh quá nhạy với nhiễu nhỏ.
- `MOG2_MIN_AREA = 300`: bỏ qua vùng chuyển động quá nhỏ, giúp lọc nhiễu do camera, ánh sáng hoặc chi tiết nền.

### 5.6. Optical flow Lucas-Kanade

Optical flow dùng để ước lượng hướng chuyển động của điểm ảnh giữa hai frame liên tiếp. Hệ thống dùng Pyramidal Lucas-Kanade thông qua OpenCV:

```python
cv2.calcOpticalFlowPyrLK(...)
```

Trong hệ thống, optical flow được dùng để tính vector chuyển động của từng người. Nếu người di chuyển ra xa vị trí rác sau khi rác xuất hiện, đó là dấu hiệu tăng khả năng người đó là chủ thể vứt rác.

Thông số Lucas-Kanade:

```python
winSize = (15, 15)
maxLevel = 2
criteria = (EPS | COUNT, 10, 0.03)
FLOW_HISTORY_FRAMES = 8
```

Ý nghĩa:

- `winSize = (15,15)`: cửa sổ tìm kiếm cục bộ đủ lớn để bắt chuyển động vừa phải giữa hai frame.
- `maxLevel = 2`: dùng pyramid nhiều mức, giúp theo dõi chuyển động ở các tỉ lệ khác nhau.
- `FLOW_HISTORY_FRAMES = 8`: lưu 8 vector gần nhất, với 6 FPS tương đương khoảng 1.33 giây. Khoảng này đủ để lấy xu hướng chuyển động gần đây, không bị kéo dài bởi chuyển động quá cũ.

## 6. Kiến trúc hệ thống

### 6.1. Tổng quan pipeline

Pipeline xử lý chính:

```text
Video file / IP camera
        |
        v
Đọc frame bằng OpenCV / ThreadedCamera
        |
        v
Tiền xử lý frame: rotate, resize, giới hạn FPS
        |
        v
YOLO + ByteTrack phát hiện người/rác
        |
        v
Parse detection, gán ID, lưu lịch sử người
        |
        v
MOG2 + Optical Flow hỗ trợ chuyển động
        |
        v
Tạo hoặc cập nhật trạng thái rác
        |
        v
Tính điểm owner cho từng người
        |
        v
Kiểm tra điều kiện xác nhận vi phạm
        |
        v
Lưu bằng chứng + gửi WebSocket + ghi DB
```

### 6.2. Các module chính

| Module | Vai trò |
|---|---|
| `TrashViolationDetector.py` | Điều phối pipeline: mở nguồn, đọc frame, gọi detect, cập nhật tracker, kết thúc run. |
| `detector_io.py` | Xử lý I/O: cấu hình YOLO, resize, stream WebSocket, evidence buffer, ghi video output. |
| `detection_parsing.py` | Parse kết quả YOLO, gán ID người/rác, tạo điểm neo gần chân người. |
| `trash_lifecycle.py` | Quản lý vòng đời rác: rác mới, rác đang theo dõi, rác mất track, recover rác. |
| `owner_resolution.py` | Xác định người liên quan đến rác và lưu frame bằng chứng của người. |
| `violation_confirmation.py` | Kiểm tra điều kiện xác nhận vi phạm, log lý do pending, vẽ HUD. |
| `OwnershipScorer.py` | Tính điểm owner dựa trên khoảng cách, hướng di chuyển, optical flow và độ gần thời gian. |
| `MotionDetector.py` | Phát hiện chuyển động bằng MOG2. |
| `OpticalFlowTracker.py` | Tính vector chuyển động người bằng Lucas-Kanade optical flow. |
| `ViolationLogger.py` | Lưu ảnh/video bằng chứng, ghi log và gửi API. |
| `stream_router.py` | WebSocket streaming, start/stop/restart detector thread. |
| `violation_router.py` | REST API lưu và lấy lịch sử vi phạm. |
| `db.py` | SQLite storage. |

### 6.3. Phương pháp triển khai theo file và nguồn dữ liệu

Phần phương pháp của hệ thống được mô tả theo hai nhóm: nhóm file được đọc khi hệ thống chạy thực tế và nhóm file dữ liệu dùng trong giai đoạn huấn luyện mô hình. Cách tách này quan trọng vì khi demo hoặc chạy real-time, backend không đọc toàn bộ dataset ảnh/label nữa; backend chỉ đọc model đã huấn luyện, nguồn video/camera và các file cấu hình.

#### 6.3.1. Các nguồn dữ liệu hệ thống đọc khi chạy

| Nhóm | File/đường dẫn | Đọc từ đâu | Mục đích |
|---|---|---|---|
| Model nhận diện | `weights/best.pt` | `Config.MODEL_PATH`, sau đó được `YOLO(cfg.MODEL_PATH)` trong `TrashViolationDetector.py` nạp vào bộ nhớ | Chứa trọng số mô hình YOLO đã huấn luyện để nhận diện `person` và `trash`. Đây là file model đang chạy thật. |
| Video file | `video/di_bo_17.mp4` | `Config.VIDEO_FILE`, được `cv2.VideoCapture(cfg.VIDEO_SOURCE)` đọc khi `SOURCE_MODE="file"` | Là video kiểm thử chính. Hệ thống đọc từng frame, sample về khoảng 6 FPS rồi đưa vào pipeline phát hiện. |
| Camera IP | `http://192.168.0.108:8080/video` | `Config.__init__()` tự ghép từ `IP_CAM_HOST`, `IP_CAM_PORT`, `IP_CAM_PROTOCOL` khi `SOURCE_MODE="ip_camera"` | Là nguồn real-time từ điện thoại/camera IP. `ThreadedCamera.py` đọc liên tục frame mới nhất để giảm độ trễ. |
| Ảnh/video bằng chứng | `violations/` | `Config.OUTPUT_DIR`, được `ViolationLogger.py` ghi file sau khi xác nhận vi phạm | Lưu ảnh bằng chứng `.jpg` và clip bằng chứng `.mp4`. Đây là dữ liệu đầu ra của hệ thống. |
| Cơ sở dữ liệu lịch sử | `violations.db` | `db.py` tạo tại thư mục gốc dự án | Lưu bản ghi vi phạm gồm person ID, trash ID, loại vi phạm, score, thời gian và URL bằng chứng. |
| Frontend live stream | `/ws/stream` | `stream_router.py` gửi frame JPEG qua WebSocket, frontend nhận trong `useVideoStream.js` | Hiển thị video đã vẽ bbox/HUD theo thời gian thực trên dashboard. |
| Frontend lịch sử | `/api/v1/violations` | `violation_router.py` đọc SQLite qua `db.py`, frontend gọi bằng `fetchViolations()` | Hiển thị danh sách vi phạm ở trang lịch sử. |

#### 6.3.2. Dataset dùng để huấn luyện model

Dataset không được đọc trực tiếp trong lúc chạy detector. Dataset được dùng ở giai đoạn trước đó để huấn luyện ra file `weights/best.pt`.

| File/thư mục | Vai trò |
|---|---|
| `data_cu1_clean_v2/data.yaml` | File khai báo dataset YOLO. Trong file này có `path: E:/TGMTTTT/data_cu1_clean_v2`, `train: images/train`, `val: images/val`, và class `0: person`, `1: trash`. |
| `data_cu1_clean_v2/images/train` | Ảnh train dùng để mô hình học đặc trưng của người và rác. |
| `data_cu1_clean_v2/images/val` | Ảnh validation dùng để đánh giá mô hình trong quá trình huấn luyện. |
| `data_cu1_clean_v2/labels/train` | Nhãn YOLO cho tập train, mỗi file `.txt` chứa class và bounding box chuẩn hóa. |
| `data_cu1_clean_v2/labels/val` | Nhãn YOLO cho tập validation. |
| `data_cu1_clean_v2/train.txt`, `data_cu1_clean_v2/val.txt` | Danh sách ảnh train/val, hỗ trợ kiểm soát tập dữ liệu. |
| `data_cu1_clean_v2/clean_report.json`, `clean_report.txt` | Báo cáo làm sạch dữ liệu, dùng để kiểm tra dữ liệu lỗi, nhãn thiếu hoặc ảnh không hợp lệ. |

Vì vậy, trong báo cáo có thể viết rõ: dữ liệu huấn luyện được đọc từ `data_cu1_clean_v2`, còn dữ liệu chạy thực tế được đọc từ `weights/best.pt` và nguồn video/camera trong `Config.py`.

#### 6.3.3. Giải thích từng file backend trong phương pháp

| File | Vai trò chi tiết trong phương pháp |
|---|---|
| `start_backend.ps1` | Script khởi chạy backend. File này đặt `PYTHONPATH` trỏ tới `.venv312/Lib/site-packages` và thư mục `src`, sau đó chạy `uvicorn server:app --host 127.0.0.1 --port 8000`. Đây là điểm bắt đầu khi chạy hệ thống theo kiểu web backend. |
| `src/server.py` | Entry point của FastAPI. File này tạo app, bật CORS cho frontend Vite, gắn `violation_router` và `stream_router`. Khi startup, server gọi `db.init_db()`, tạo task `_queue_broadcaster()` để phát frame qua WebSocket và gọi `start_detector_thread()` để detector chạy nền. |
| `src/Main.py` | Entry point đơn giản nếu chỉ muốn chạy detector trực tiếp không qua FastAPI. File này tạo `TrashViolationDetector(cfg=Config())` rồi gọi `detector.run()`. |
| `src/Config.py` | File cấu hình trung tâm. Tại đây khai báo nguồn video, camera IP, model path, FPS, ngưỡng YOLO, ngưỡng tracking, ngưỡng xác nhận vi phạm, thông số MOG2, optical flow, stream và evidence. `Config.__init__()` còn đọc biến môi trường như `DETECTOR_SOURCE_MODE`, `DETECTOR_VIDEO_FILE`, `DETECTOR_IP_CAM_HOST` để đổi chế độ chạy mà không cần sửa code. |
| `src/TrashViolationDetector.py` | File điều phối pipeline chính. `run()` kiểm tra đường dẫn model/video, nạp YOLO từ `weights/best.pt`, mở nguồn video/camera, tính stride cho video file, đọc từng frame, gọi `_process_frame()`, ghi video output nếu cần và đẩy frame lên frontend. Đây là file nối toàn bộ các module xử lý lại với nhau. |
| `src/detector_io.py` | Nhóm hàm I/O và runtime. File này tạo tham số chạy YOLO/ByteTrack, giới hạn FPS live, xoay/resize frame, encode JPEG để stream, giữ buffer clip bằng chứng 45 giây, ghi video output và convert H.264 nếu có FFmpeg. Nó giúp tách phần đọc/ghi/stream ra khỏi logic phát hiện vi phạm. |
| `src/detection_parsing.py` | Chuyển output YOLO thành dữ liệu có thể suy luận. File này đọc bounding box, class, confidence và ID từ ByteTrack; class `0` được coi là người, class `1` là rác. Với người, hệ thống lấy điểm neo gần chân thay vì tâm bbox vì hành vi vứt rác thường xảy ra gần mặt đất. File này cũng xử lý mất ID tạm thời, tạo ID giả khi cần, thêm người từ motion blob và chạy floor-pass để bắt rác nhỏ ở vùng dưới ảnh. |
| `src/trash_lifecycle.py` | Quản lý vòng đời của từng vật rác. Khi rác mới xuất hiện, file này tạo bản ghi trong `_trash_registry`, tìm owner ban đầu, lưu vị trí xuất hiện và trạng thái pending. Khi rác tiếp tục được thấy, file này tăng `confirm_ctr`, kiểm tra rác có đứng yên không, cập nhật owner, đánh giá lại owner nếu score thấp hoặc ambiguous. Nếu rác mất detection, nó thử recover dựa trên vị trí cuối, chân owner hoặc motion gần mặt đất. |
| `src/owner_resolution.py` | Xác định người có khả năng liên quan đến rác. File này gọi `OwnershipScorer`, kiểm tra owner có đủ số frame xuất hiện và có chuyển động thật hay không. Nếu owner hiện tại không đủ tin cậy, file này dùng fallback recent-owner: tìm người từng đi gần vị trí rác trong lịch sử gần đây. Nó cũng lưu frame bằng chứng của owner để khi người rời khỏi khung hình vẫn có ảnh chứng minh. |
| `src/OwnershipScorer.py` | Tính điểm owner. Điểm được tổng hợp từ 4 tín hiệu: khoảng cách gần rác trong lịch sử, hướng người di chuyển rời xa rác, vector optical flow và độ gần về thời gian. Công thức dùng trọng số `0.40 proximity + 0.30 direction + 0.20 flow + 0.10 recency`. File này cũng đánh dấu ambiguous nếu hai người có điểm quá gần nhau. |
| `src/MotionDetector.py` | Phát hiện vùng chuyển động bằng background subtraction MOG2. Kết quả trả về là danh sách `(cx, cy, area)` của các blob chuyển động đủ lớn. Tín hiệu này dùng để hỗ trợ owner scoring, tạo người fallback khi YOLO bỏ sót người và hỗ trợ recover rác mất track. |
| `src/OpticalFlowTracker.py` | Tính vector chuyển động của từng người bằng Pyramidal Lucas-Kanade optical flow. File này so sánh frame xám trước và frame xám hiện tại, lưu vector vận tốc gần đây cho từng person ID. Vector này giúp biết người đang di chuyển theo hướng nào so với vị trí rác. |
| `src/violation_confirmation.py` | File quyết định khi nào rác pending trở thành vi phạm thật. Nó kiểm tra các điều kiện chung: owner có score đủ, owner không ambiguous, owner xuất hiện đủ frame, rác nằm vùng mặt đất. Sau đó phân loại thành `Đột ngột`, `Đứng yên`, hoặc `Bỏ rác tại chỗ` dựa trên số frame xác nhận, trạng thái owner rời đi và số frame rác đứng yên. |
| `src/ViolationLogger.py` | Lưu bằng chứng sau khi vi phạm được xác nhận. File này lấy frame bằng chứng, vẽ overlay người/rác, lưu ảnh vào `violations/`, tạo clip `.mp4` từ evidence buffer, ghi log nội bộ và gọi `ApiSyncer` để gửi dữ liệu lên FastAPI. |
| `src/ApiSyncer.py` | Gửi vi phạm lên backend bằng background thread. File này POST `personId`, `trashId`, `score`, `violationType`, `timestamp`, ảnh bằng chứng và video bằng chứng tới `/api/v1/violations`. Chạy bất đồng bộ để không làm chậm luồng detector. |
| `src/violation_router.py` | REST API xử lý dữ liệu vi phạm. Endpoint `POST /api/v1/violations` nhận dữ liệu từ detector, upload ảnh/video lên MinIO hoặc lưu local fallback, rồi ghi vào SQLite. Endpoint `GET /api/v1/violations` trả lịch sử cho frontend. Endpoint `POST /api/v1/violations/clear` xóa lịch sử và bằng chứng. |
| `src/db.py` | Lớp lưu trữ SQLite. File này tạo bảng `violations`, thêm bản ghi mới, cập nhật bản ghi, đọc danh sách vi phạm, đếm tổng số và xóa toàn bộ lịch sử. Database nằm ở `violations.db` trong thư mục gốc dự án. |
| `src/minio_storage.py` | Quản lý lưu trữ bằng chứng. Nếu MinIO chạy được, file này upload ảnh/video vào bucket `violations` và trả URL public. Nếu MinIO offline, nó lưu file vào thư mục `violations/` và trả URL local `/api/v1/evidence/{filename}`. |
| `src/stream_router.py` | Quản lý WebSocket và detector thread. File này có `frame_queue` kích thước 1 để luôn gửi frame mới nhất, `alert_queue` để gửi cảnh báo JSON, endpoint `/ws/stream` cho frontend nhận video, và API `/api/v1/stream/restart`, `/api/v1/stream/stop`, `/api/v1/stream/video`. |
| `src/ThreadedCamera.py` | Đọc camera real-time bằng thread riêng. Mục tiêu là luôn lấy frame mới nhất từ camera/IP stream, tránh detector bị chậm do buffer mạng. File này cũng tự reconnect nếu camera mất kết nối. |
| `src/test_camera.py` | File kiểm tra camera, dùng để test URL camera IP, snapshot HTTP và OpenCV trước khi chạy detector thật. Đây là file hỗ trợ debug, không thuộc pipeline chính. |

#### 6.3.4. Giải thích từng file frontend trong phương pháp hiển thị

| File | Vai trò chi tiết |
|---|---|
| `frontend/vite.config.js` | Cấu hình Vite dev server port `5173` và proxy `/api` sang `http://127.0.0.1:8000`, proxy `/ws` sang `ws://127.0.0.1:8000`. Nhờ vậy frontend gọi API cùng origin mà không bị lỗi CORS khi phát triển. |
| `frontend/src/App.jsx` | Khai báo router chính gồm dashboard `/` và lịch sử `/history`, đồng thời truyền số lượng vi phạm lên navbar. |
| `frontend/src/hooks/useVideoStream.js` | Kết nối WebSocket `/ws/stream`. Nếu nhận binary message thì coi đó là JPEG frame và vẽ lên canvas. Nếu nhận JSON text thì phân biệt alert vi phạm hoặc sự kiện video đã xử lý xong. File này là cầu nối real-time giữa backend và giao diện. |
| `frontend/src/components/VideoPanel.jsx` | Hiển thị canvas video, trạng thái LIVE/ENDED, FPS frontend đo được, nút dừng stream, nút khởi động lại và nút tải video kết quả khi chạy file. |
| `frontend/src/hooks/useViolations.js` | Quản lý state lịch sử vi phạm ở frontend. File này gọi API lấy dữ liệu ban đầu, thêm alert mới khi WebSocket báo vi phạm, cập nhật bản ghi nếu cùng person có score cao hơn và gọi lại API sau 2 giây để lấy URL bằng chứng thật. |
| `frontend/src/api/violations.js` | Định nghĩa các hàm gọi REST API: lấy lịch sử vi phạm, restart stream, stop stream và xóa toàn bộ lịch sử. |
| `frontend/src/pages/Dashboard.jsx` | Trang giám sát chính. Trang này hiển thị video real-time, thống kê và danh sách cảnh báo tức thời. Khi `VideoPanel` nhận alert, `Dashboard` đưa alert vào `AlertFeed` và cập nhật lịch sử. |
| `frontend/src/components/AlertFeed.jsx` | Hiển thị danh sách cảnh báo vi phạm real-time ở sidebar, gồm loại vi phạm, person ID, trash ID, score và thời gian. |
| `frontend/src/pages/History.jsx` | Trang lịch sử. Trang này đọc dữ liệu từ `useViolations()`, hỗ trợ lọc theo loại vi phạm, tìm theo person/trash/time, làm mới dữ liệu và xóa lịch sử. |
| `frontend/src/components/ViolationCard.jsx` | Hiển thị từng vi phạm trong lịch sử, gồm ảnh bằng chứng, video clip nếu có, loại vi phạm, score, person ID, trash ID và thời gian ghi nhận. |

#### 6.3.5. Luồng đọc và xử lý dữ liệu theo thứ tự thực thi

```text
start_backend.ps1
    -> src/server.py
        -> src/Config.py
        -> src/stream_router.py khởi động detector thread
            -> src/TrashViolationDetector.py
                -> đọc weights/best.pt
                -> đọc video/di_bo_17.mp4 hoặc camera IP
                -> detector_io.py tiền xử lý frame, resize, giới hạn FPS
                -> YOLO + ByteTrack phát hiện person/trash
                -> detection_parsing.py parse bbox, gán ID, lưu lịch sử người
                -> MotionDetector.py + OpticalFlowTracker.py bổ sung chuyển động
                -> trash_lifecycle.py quản lý trạng thái rác
                -> owner_resolution.py + OwnershipScorer.py xác định owner
                -> violation_confirmation.py xác nhận loại vi phạm
                -> ViolationLogger.py lưu bằng chứng
                -> ApiSyncer.py gửi POST /api/v1/violations
        -> violation_router.py lưu MinIO/local + SQLite
        -> stream_router.py gửi frame/alert qua WebSocket
frontend
    -> useVideoStream.js nhận frame/alert
    -> Dashboard.jsx hiển thị live
    -> History.jsx đọc /api/v1/violations để hiển thị lịch sử
```

Tóm lại, phần phương pháp không nên trình bày như một thuật toán đơn lẻ. Nên trình bày theo chuỗi: nguồn dữ liệu vào, nạp mô hình, xử lý từng frame, phát hiện đối tượng, theo dõi ID, suy luận owner, xác nhận vi phạm, lưu bằng chứng và hiển thị lên frontend. Cách trình bày này giúp người đọc hiểu rõ file nào chịu trách nhiệm cho từng bước.

## 7. Xử lý nguồn video và FPS

### 7.1. Chế độ video file

Khi chạy file, hệ thống dùng:

```python
SOURCE_MODE = "file"
VIDEO_FILE = video/di_bo_17.mp4
FILE_MODE_FPS = 6
FILE_FRAME_STRIDE = 0
```

`FILE_FRAME_STRIDE = 0` nghĩa là hệ thống tự tính số frame cần bỏ qua dựa vào FPS gốc của video:

```python
file_frame_step = round(source_fps / FILE_MODE_FPS)
```

Ví dụ:

| FPS gốc | `FILE_MODE_FPS` | Frame step | Ý nghĩa |
|---:|---:|---:|---|
| 30 FPS | 6 FPS | 5 | Lấy 1 frame, bỏ 4 frame |
| 25 FPS | 6 FPS | 4 | Lấy 1 frame, bỏ 3 frame |

Việc sample này có hai mục đích:

1. Giảm tải tính toán cho YOLO.
2. Đảm bảo các tham số dạng frame có ý nghĩa thời gian ổn định.

Nếu không sample, video 30 FPS sẽ khiến `CONFIRM_FRAMES = 12` chỉ tương đương 0.4 giây. Điều đó quá ngắn để xác nhận hành vi vứt rác. Khi sample về 6 FPS, `CONFIRM_FRAMES = 12` tương đương khoảng 2 giây, hợp lý hơn cho hành vi cần quan sát theo thời gian.

### 7.2. Chế độ real-time camera

Khi chạy camera IP, hệ thống dùng:

```python
SOURCE_MODE = "ip_camera"
LIVE_TARGET_FPS = 6.0
CAMERA_BUFFER = 1
```

`CAMERA_BUFFER = 1` giúp giảm độ trễ. Camera chỉ giữ frame mới nhất thay vì tích lũy nhiều frame cũ. Đây là nguyên tắc quan trọng trong hệ thống real-time: ưu tiên hiển thị và xử lý frame mới hơn là xử lý hết frame cũ.

`LIVE_TARGET_FPS = 6.0` được chọn vì pipeline có YOLO, floor-pass, optical flow, encode JPEG và WebSocket. Với CPU hoặc GPU phổ thông, 5-6 FPS là mức cân bằng giữa độ trễ và độ ổn định. Bài toán vứt rác là hành vi chậm theo giây, không yêu cầu 30 FPS như các bài toán thể thao hoặc tracking tốc độ cao.

### 7.3. Quy đổi thông số frame sang thời gian

Với FPS xử lý mục tiêu là 6 FPS:

| Tham số | Giá trị | Thời gian tương đương | Ý nghĩa |
|---|---:|---:|---|
| `CONFIRM_FRAMES` | 12 | 2.0 giây | Rác phải tồn tại đủ lâu trước khi xác nhận. |
| `CONFIRM_FRAMES_SUDDEN` | 8 | 1.33 giây | Xác nhận nhanh hơn với tình huống rác xuất hiện đột ngột. |
| `MIN_OWNER_GONE_FRAMES` | 6 | 1.0 giây | Owner phải rời khỏi vùng ít nhất 1 giây. |
| `STATIONARY_REQUIRED` | 10 | 1.67 giây | Rác phải đứng yên đủ lâu để tránh nhiễu. |
| `HISTORY_FRAMES` | 90 | 15 giây | Lưu lịch sử vị trí người trong 15 giây. |
| `OWNER_REEVAL_FRAMES` | 90 | 15 giây | Cho phép cập nhật lại owner trong 15 giây đầu. |
| `STALE_FRAMES` | 90 | 15 giây | Xóa rác khỏi registry nếu mất quá lâu. |
| `LOST_TRASH_RECOVERY_FRAMES` | 60 | 10 giây | Cho phép recover rác bị mất track trong 10 giây. |
| `DRAW_STALE_TRASH_FRAMES` | 18 | 3 giây | Giữ hiển thị rác cũ trong vài giây. |
| `PENDING_LOG_INTERVAL_FRAMES` | 30 | 5 giây | Log lý do pending mỗi 5 giây. |
| `OWNERLESS_CANDIDATE_COOLDOWN_FRAMES` | 60 | 10 giây | Tránh lưu quá nhiều candidate liên tiếp. |
| `FLOW_HISTORY_FRAMES` | 8 | 1.33 giây | Lưu vector chuyển động gần nhất. |

Nếu thay đổi FPS xử lý, các tham số theo frame nên được scale theo công thức:

```text
Giá trị frame mới = Thời gian mong muốn x FPS mới
```

Ví dụ muốn giữ `CONFIRM_FRAMES` tương đương 2 giây khi tăng lên 10 FPS:

```text
CONFIRM_FRAMES = 2 x 10 = 20
```

## 8. Phát hiện người và rác

### 8.1. Phát hiện bằng YOLO

Trong mỗi frame, YOLO trả về danh sách bounding box. Hệ thống phân loại:

```python
class 0 -> person
class 1 -> trash
```

Đối với người, hệ thống không dùng tâm bounding box làm điểm đại diện. Thay vào đó dùng điểm neo gần chân:

```python
anchor = điểm gần chân người
```

Lý do:

- Rác thường nằm trên mặt đất.
- Vị trí liên hệ giữa người và rác thường gần chân hoặc vùng dưới cơ thể hơn là tâm thân người.
- Trong camera góc nghiêng, tâm bounding box người có thể nằm cao hơn nhiều so với vị trí tương tác với rác.

Ngoài anchor chính, hệ thống còn tạo nhiều điểm phụ quanh chân và vùng mở rộng dưới bounding box để tăng khả năng liên hệ người-rác.

### 8.2. Lọc người không hợp lệ

Thông số:

```python
LIVE_PERSON_CONF = 0.14
LIVE_PERSON_MIN_HEIGHT_RATIO = 0.12
LIVE_PERSON_MIN_AREA_RATIO = 0.012
```

Cơ sở chọn:

- Confidence người ở live được để thấp hơn mặc định vì camera IP có thể rung, mờ, ánh sáng kém.
- Tuy nhiên, để tránh nhận nhầm vật nhỏ là người, hệ thống yêu cầu bounding box người phải đủ cao và đủ diện tích.
- `LIVE_PERSON_MIN_HEIGHT_RATIO = 0.12` nghĩa là chiều cao bbox người phải tối thiểu 12% chiều cao frame. Đây là ngưỡng lọc các detection rất nhỏ ở xa hoặc nhiễu.
- `LIVE_PERSON_MIN_AREA_RATIO = 0.012` nghĩa là diện tích bbox người phải đủ lớn so với frame, tránh nhận nhầm các vùng nhỏ.

### 8.3. Phát hiện rác dưới sàn bằng floor-pass

Rác thường nhỏ, nằm ở vùng dưới ảnh và dễ bị YOLO bỏ sót ở full-frame. Vì vậy live mode có thêm floor-pass:

```python
LIVE_FLOOR_TRASH_PASS = True
LIVE_FLOOR_PASS_INTERVAL = 4
LIVE_FLOOR_ROI_TOP = 0.40
LIVE_FLOOR_YOLO_IMGSZ = 640
LIVE_FLOOR_TRASH_CONF = 0.10
```

Cơ sở chọn:

- `LIVE_FLOOR_ROI_TOP = 0.40`: chỉ lấy vùng từ 40% chiều cao ảnh trở xuống. Đây là vùng có khả năng chứa mặt đất/rác nhiều hơn.
- `LIVE_FLOOR_YOLO_IMGSZ = 640`: tăng kích thước inference cho ROI giúp rác nhỏ rõ hơn.
- `LIVE_FLOOR_TRASH_CONF = 0.10`: confidence thấp để không bỏ sót rác nhỏ. Việc giảm threshold được bù lại bằng các điều kiện thời gian, owner, mặt đất và stationary.
- `LIVE_FLOOR_PASS_INTERVAL = 4`: floor-pass chạy mỗi 4 frame xử lý, tức khoảng 0.67 giây ở 6 FPS. Rác không di chuyển nhanh, nên không cần chạy floor-pass từng frame; cách này giảm tải CPU.

## 9. Theo dõi ID người và rác

### 9.1. ID người

Hệ thống dùng ByteTrack ID nếu có. Nếu ByteTrack không trả ID ổn định, hệ thống tự khớp ID dựa trên khoảng cách đến vị trí lịch sử:

```python
PERSON_ID_MATCH_RADIUS = 180
```

Cơ sở chọn:

- Người có thể di chuyển đáng kể giữa các frame, nhất là khi camera chỉ xử lý 6 FPS.
- Bounding box người cũng có thể dao động do YOLO.
- Bán kính 180 pixel đủ để nối ID khi người di chuyển vừa phải, nhưng không quá lớn để nhập nhầm hai người đứng xa nhau.

### 9.2. ID rác

Rác thường nhỏ và gần như đứng yên, nên bán kính matching nhỏ hơn người:

```python
TRASH_ID_MATCH_RADIUS = 90
```

Cơ sở chọn:

- Rác sau khi xuất hiện thường nằm cố định trên mặt đất.
- YOLO có thể dao động bbox vài pixel đến vài chục pixel.
- 90 pixel đủ để giữ ID trong trường hợp bbox rác jitter hoặc rác bị detect lệch, nhưng vẫn hạn chế nhập nhầm hai vật rác khác nhau.

## 10. Tính điểm owner

### 10.1. Mục đích

Khi một vật rác xuất hiện, hệ thống cần xác định người nào có khả năng liên quan nhất. Đây là bài toán suy luận theo thời gian, không thể chỉ dựa vào một frame đơn lẻ.

Hệ thống tính điểm ownership trong `OwnershipScorer.py` theo công thức:

```text
score = 0.40 * proximity_score
      + 0.30 * direction_score
      + 0.20 * flow_score
      + 0.10 * recency_score
```

### 10.2. Proximity score

`proximity_score` dựa vào khoảng cách nhỏ nhất giữa rác và lịch sử vị trí người:

```text
proximity_score = 1 - min_distance / TRAJECTORY_RADIUS
```

Thông số:

```python
TRAJECTORY_RADIUS = 380
```

Cơ sở chọn:

- Rác có thể được phát hiện sau khi người đã bước ra xa vài frame.
- Người tương tác với rác thường từng đi qua hoặc đứng gần vị trí rác.
- 380 pixel là vùng đủ rộng để bao phủ sai lệch do góc camera, vị trí chân, bbox và độ trễ phát hiện rác.

Trọng số proximity là 0.40 vì khoảng cách gần là dấu hiệu mạnh nhất trong bài toán owner.

### 10.3. Direction score

`direction_score` đánh giá xem người có xu hướng di chuyển ra xa vị trí rác hay không. Về mặt toán học, hệ thống dùng cosine similarity giữa vector chuyển động của người và vector từ rác đến vị trí gần đây của người.

Nếu người di chuyển ra xa rác:

```text
direction_score cao
```

Nếu người đi về phía rác hoặc đứng yên:

```text
direction_score thấp
```

Trọng số direction là 0.30 vì hành vi bỏ rác thường đi kèm với việc người rời khỏi vị trí rác sau khi rác xuất hiện.

### 10.4. Flow score

`flow_score` sử dụng optical flow để xác nhận hướng chuyển động cục bộ của người. Đây là tín hiệu bổ sung cho direction score. Direction score dựa trên lịch sử vị trí bbox/anchor; flow score dựa trên chuyển động điểm ảnh giữa frame.

Trọng số flow là 0.20 vì optical flow hữu ích nhưng có thể nhiễu khi:

- Camera rung.
- Người bị che khuất.
- Ánh sáng thay đổi.
- YOLO cập nhật anchor không ổn định.

### 10.5. Recency score

`recency_score` đánh giá thời điểm người gần rác gần đây đến mức nào:

```text
recency_score = 1 - closest_offset / HISTORY_FRAMES
```

Nếu người vừa ở gần rác trong vài frame gần nhất, điểm cao. Nếu người đi qua quá lâu rồi, điểm giảm.

Trọng số recency là 0.10 vì nó chỉ là tín hiệu phụ, tránh việc một người đi qua rất lâu trước đó vẫn bị gán nhầm.

### 10.6. Ngưỡng score và ambiguous

Thông số:

```python
MIN_SCORE = 0.12
AMBIGUOUS_MARGIN = 0.12
REEVAL_SCORE_THRESH = 0.25
```

Cơ sở chọn:

- `MIN_SCORE = 0.12`: điểm tối thiểu để một người được coi là owner tiềm năng. Giá trị này không quá cao vì score tổng hợp có nhiều thành phần có thể bằng 0, ví dụ optical flow không ổn định hoặc direction chưa rõ. Tuy nhiên hệ thống không xác nhận chỉ dựa vào score; còn cần điều kiện owner đủ tin cậy, rác đứng yên, nằm trên mặt đất và không ambiguous.
- `AMBIGUOUS_MARGIN = 0.12`: nếu hai người có điểm gần nhau trong khoảng 0.12, hệ thống coi là mơ hồ. Điều này giảm nguy cơ kết luận sai khi có nhiều người đứng gần rác.
- `REEVAL_SCORE_THRESH = 0.25`: nếu score ban đầu thấp hơn 0.25, hệ thống tiếp tục đánh giá lại owner trong các frame tiếp theo. Lý do là rác có thể mới xuất hiện, detection chưa ổn định hoặc người vừa rời khỏi khung.

## 11. Điều kiện xác nhận vi phạm

Hệ thống không ghi nhận vi phạm ngay khi thấy rác. Một vật thể rác chỉ được xác nhận nếu thỏa mãn các điều kiện thời gian, không gian và owner.

### 11.1. Điều kiện chung

Trong `violation_confirmation.py`, điều kiện chung là:

```text
score >= MIN_SCORE
owner_seen_enough = True
is_ambiguous = False
ground_condition = True
```

Ý nghĩa:

- Owner phải có điểm đủ lớn.
- Người đó phải được nhìn thấy đủ số frame.
- Không được có nhiều owner cạnh tranh gần điểm nhau.
- Rác phải nằm ở vùng mặt đất.

Thông số:

```python
MIN_OWNER_SEEN_FRAMES = 2
MIN_OWNER_MOTION_PX = 5
VIOLATION_GROUND_MIN_Y_RATIO = 0.58
```

Cơ sở chọn:

- `MIN_OWNER_SEEN_FRAMES = 2`: cần ít nhất 2 frame để tránh gán owner cho detection chỉ xuất hiện nhất thời.
- `MIN_OWNER_MOTION_PX = 5`: owner phải có chuyển động tối thiểu 5 pixel trong lịch sử. Điều này tránh trường hợp một bbox nhiễu hoặc vật thể đứng yên bị coi là người liên quan.
- `VIOLATION_GROUND_MIN_Y_RATIO = 0.58`: rác phải nằm ở 58% chiều cao frame trở xuống. Với camera giám sát, mặt đất thường nằm ở nửa dưới khung hình; ngưỡng này giúp loại bỏ vật thể giống rác nhưng nằm trên cao hoặc nền phía xa.

### 11.2. Vi phạm loại "Đột ngột"

Điều kiện:

```text
frames_since_spawn <= CONFIRM_FRAMES_SUDDEN + MIN_OWNER_GONE_FRAMES
confirm_ctr >= CONFIRM_FRAMES_SUDDEN
owner_truly_gone = True
common_ok = True
```

Thông số:

```python
CONFIRM_FRAMES_SUDDEN = 8
MIN_OWNER_GONE_FRAMES = 6
```

Ở 6 FPS:

```text
CONFIRM_FRAMES_SUDDEN = 8 frame ≈ 1.33 giây
MIN_OWNER_GONE_FRAMES = 6 frame ≈ 1 giây
```

Cơ sở chọn:

- Hành vi "đột ngột" là rác xuất hiện trong thời gian ngắn sau khi người rời đi.
- Cần chờ ít nhất 8 frame, tương đương khoảng 1.33 giây, để bảo đảm vật thể rác không phải detection chớp nhoáng. Một detection nhầm thường chỉ xuất hiện rất ngắn, còn rác thật sau khi bị bỏ xuống sẽ tiếp tục nằm lại.
- Cần owner rời đi ít nhất 6 frame, tương đương khoảng 1 giây, để tránh trường hợp YOLO mất người tạm thời trong 1-2 frame do che khuất hoặc motion blur.
- Tổng cửa sổ `CONFIRM_FRAMES_SUDDEN + MIN_OWNER_GONE_FRAMES = 14 frame`, tương đương khoảng 2.33 giây. Khoảng này phù hợp với tình huống người vừa bỏ rác rồi rời đi nhanh: hệ thống vẫn xác nhận sớm nhưng không kết luận ngay tức thì.

### 11.3. Vi phạm loại "Đứng yên"

Điều kiện:

```text
confirm_ctr >= CONFIRM_FRAMES
stationary_after_owner_gone >= STATIONARY_REQUIRED
owner_truly_gone = True
common_ok = True
```

Thông số:

```python
CONFIRM_FRAMES = 12
STATIONARY_REQUIRED = 10
```

Ở 6 FPS:

```text
CONFIRM_FRAMES = 12 frame ≈ 2 giây
STATIONARY_REQUIRED = 10 frame ≈ 1.67 giây
```

Cơ sở chọn:

- Rác phải tồn tại đủ lâu và đứng yên sau khi người rời đi. Đây là dấu hiệu hành vi quan trọng: vật bị vứt sẽ nằm lại, còn vật đang được người cầm/mang theo sẽ tiếp tục chuyển động.
- `CONFIRM_FRAMES = 12` tương đương khoảng 2 giây. Khoảng này đủ để lọc detection chớp nhoáng nhưng vẫn cho cảnh báo kịp thời.
- `STATIONARY_REQUIRED = 10` tương đương khoảng 1.67 giây. Nếu một vật thể nằm gần như cố định trong thời gian này, khả năng cao đó là rác thật trên mặt đất thay vì một vùng nhiễu.
- Hai điều kiện này kết hợp với `owner_truly_gone` để tránh nhầm trường hợp người vẫn đang thao tác với vật hoặc vật chỉ xuất hiện tạm thời cạnh chân người.

### 11.4. Vi phạm loại "Bỏ rác tại chỗ"

Điều kiện:

```text
confirm_ctr >= CONFIRM_FRAMES
stationary_ctr >= STATIONARY_REQUIRED
owner_id vẫn còn trong current_persons
common_ok = True
```

Loại này dùng cho tình huống người vẫn còn trong khung hình hoặc gần vị trí rác nhưng vật rác đã nằm yên đủ lâu.

Cơ sở chọn:

- Không phải mọi hành vi vứt rác đều có người rời khỏi khung hình ngay. Có trường hợp người đứng lại, nói chuyện hoặc đi chậm sau khi bỏ rác.
- Vì vậy hệ thống vẫn cho phép xác nhận nếu rác đã đủ `CONFIRM_FRAMES`, đủ `STATIONARY_REQUIRED`, owner vẫn có mặt và các điều kiện chung thỏa mãn.
- Điều kiện `data["owner_id"] in current_persons` giúp phân biệt loại này với trường hợp owner đã rời đi, vốn được xử lý bởi nhóm "Đột ngột" hoặc "Đứng yên".

## 12. Thông số không gian và cơ sở lựa chọn

### 12.1. STATIONARY_PX = 16

```python
STATIONARY_PX = 16
```

Thông số này xác định rác có được xem là đứng yên hay không. Nếu khoảng dịch chuyển giữa frame hiện tại và frame trước nhỏ hơn 16 pixel, rác được coi là stationary.

Cơ sở chọn:

- Bounding box của vật nhỏ như rác thường bị jitter do YOLO, dù vật không thật sự di chuyển.
- Ngưỡng quá nhỏ sẽ làm rác đứng yên bị xem là đang di chuyển.
- Ngưỡng quá lớn sẽ coi vật đang di chuyển là đứng yên.
- Với frame live đã giới hạn cạnh dài khoảng 960 pixel, 16 pixel tương đương khoảng 1.7% cạnh dài, đủ để hấp thụ nhiễu detection nhưng vẫn nhỏ so với chuyển động thực.

### 12.2. SPAWN_RADIUS = 320

```python
SPAWN_RADIUS = 320
```

Thông số này dùng khi rác mới xuất hiện và cần tìm người gần đó. Rác thường xuất hiện ở gần chân/người vừa tương tác. Tuy nhiên do camera góc nghiêng, người có thể đứng lệch khỏi rác khá xa trên ảnh. Vì vậy bán kính 320 pixel cho phép bao phủ vùng tương tác hợp lý.

### 12.3. TRAJECTORY_RADIUS = 380

```python
TRAJECTORY_RADIUS = 380
```

Thông số này dùng trong tính điểm owner theo lịch sử quỹ đạo. Nó lớn hơn `SPAWN_RADIUS` vì người có thể đã rời khỏi vị trí rác khi rác được detect rõ. 380 pixel cho phép truy vết người từng đi qua gần rác trong vài giây trước đó.

### 12.4. RECENT_OWNER_RADIUS = 520

```python
RECENT_OWNER_RADIUS = 520
```

Đây là bán kính fallback cho trường hợp YOLO không tìm được owner trực tiếp nhưng có người vừa xuất hiện gần vị trí rác trong lịch sử. Bán kính này lớn hơn vì fallback cần bù cho mất detection hoặc mất track, nhưng vẫn bị giới hạn bởi recency và điều kiện owner có chuyển động.

### 12.5. LOST_TRASH_FOOT_SNAP_RADIUS = 220

```python
LOST_TRASH_FOOT_SNAP_RADIUS = 220
```

Khi rác bị mất track tạm thời, hệ thống cố gắng recover vị trí rác dựa vào điểm gần chân owner hoặc motion alert gần mặt đất. 220 pixel là bán kính vừa đủ để snap rác về vùng gần chân người mà không kéo rác sang đối tượng quá xa.

## 13. Xử lý rác mất track

YOLO có thể mất detection rác trong một số frame do:

- Rác quá nhỏ.
- Rác bị chân người che.
- Ánh sáng hoặc camera rung.
- Bounding box confidence giảm.

Vì vậy hệ thống không xóa rác ngay khi mất detection. Thay vào đó:

```python
LOST_TRASH_RECOVERY_FRAMES = 60
LOST_TRASH_MIN_SEEN = 2
```

Ở 6 FPS, `LOST_TRASH_RECOVERY_FRAMES = 60` tương đương 10 giây. Nếu rác đã được thấy ít nhất 2 frame, hệ thống cho phép recover trong 10 giây bằng:

- Vị trí chân owner.
- Motion blob gần mặt đất.
- Vị trí cuối cùng nếu vẫn nằm vùng mặt đất.

Cơ sở chọn:

- Không nên xóa rác quá nhanh vì vật nhỏ dễ mất detection.
- Nhưng cũng không nên giữ quá lâu vì có thể tạo false positive với vật thể đã biến mất.
- 10 giây là khoảng cân bằng cho video/camera giám sát một hành vi ngắn.

## 14. Lưu bằng chứng và truyền dữ liệu

### 14.1. Lưu ảnh bằng chứng

Khi vi phạm được xác nhận, `ViolationLogger` lưu ảnh annotated frame vào:

```text
violations/
```

Ảnh bằng chứng có overlay:

- ID người.
- ID rác.
- Vòng tròn tại vị trí rác.
- Bounding box/vị trí owner nếu còn trong frame.

### 14.2. Lưu video clip bằng chứng

Hệ thống có buffer evidence clip:

```python
EVIDENCE_CLIP_SECONDS = 45
EVIDENCE_CLIP_MAX_FPS = 6.0
EVIDENCE_CLIP_MAX_HEIGHT = 720
EVIDENCE_CLIP_JPEG_QUALITY = 65
```

Cơ sở chọn:

- 45 giây đủ để chứa giai đoạn trước, trong và sau khi vi phạm xảy ra.
- Giới hạn 6 FPS để tránh file clip quá nặng.
- Giới hạn chiều cao 720 pixel để giảm dung lượng nhưng vẫn đủ quan sát.
- JPEG quality 65 cân bằng giữa chất lượng bằng chứng và dung lượng.

### 14.3. Streaming lên frontend

Backend gửi frame qua WebSocket dưới dạng JPEG binary:

```python
frame_queue = Queue(maxsize=1)
```

Queue frame có kích thước 1 vì real-time cần frame mới nhất. Nếu frontend hoặc network chậm, frame cũ bị bỏ qua thay vì xếp hàng. Đây là thiết kế đúng cho live video vì giảm độ trễ.

Thông số stream:

```python
STREAM_MAX_HEIGHT = 720
STREAM_JPEG_QUALITY = 50
STREAM_TARGET_FPS = 6.0
```

Cơ sở chọn:

- 720p đủ hiển thị trên dashboard.
- Quality 50 giảm băng thông và thời gian decode browser.
- 6 FPS khớp tốc độ xử lý detector.

### 14.4. Lưu lịch sử vi phạm

Thông tin vi phạm được lưu vào SQLite:

```text
person_id
trash_id
violation_type
score
timestamp
evidence_url
evidence_video_url
created_at
```

Ngoài SQLite, hệ thống upload ảnh/video lên MinIO thông qua API. Backend còn có cơ chế giữ vi phạm có score cao nhất cho cùng `person_id` trong một session, tránh ghi nhiều bản ghi trùng người.

## 15. Bảng cấu hình chính

| Nhóm | Tham số | Giá trị | Cơ sở lựa chọn |
|---|---:|---:|---|
| Nguồn | `SOURCE_MODE` | `ip_camera` | Mặc định chạy camera thời gian thực. Có thể đổi bằng env. |
| Video file | `VIDEO_FILE` | `video/di_bo_17.mp4` | File kiểm thử chính. |
| Video file | `FILE_MODE_FPS` | 6 | Chuẩn hóa thời gian xử lý, giảm tải YOLO. |
| Video file | `FILE_FRAME_STRIDE` | 0 | Tự tính stride từ FPS gốc. |
| Camera | `CAMERA_BUFFER` | 1 | Giảm độ trễ live, ưu tiên frame mới. |
| Camera | `LIVE_TARGET_FPS` | 6.0 | Cân bằng giữa latency và năng lực YOLO. |
| Tiền xử lý | `LIVE_PROCESS_MAX_SIDE` | 960 | Giảm kích thước frame để tăng tốc nhưng vẫn đủ chi tiết. |
| YOLO live | `LIVE_YOLO_IMGSZ` | 512 | Tối ưu tốc độ full-frame. |
| YOLO live | `LIVE_YOLO_CONF` | 0.12 | Tăng recall cho vật nhỏ/camera mờ, dùng temporal filter bù false positive. |
| YOLO file | `conf` | 0.25 | Video file ổn định hơn, dùng confidence mặc định cao hơn. |
| Floor-pass | `LIVE_FLOOR_TRASH_PASS` | True | Bắt rác nhỏ ở vùng mặt đất tốt hơn. |
| Floor-pass | `LIVE_FLOOR_PASS_INTERVAL` | 4 | Chạy mỗi 0.67 giây ở 6 FPS để giảm tải. |
| Floor-pass | `LIVE_FLOOR_ROI_TOP` | 0.40 | Chỉ xét 60% vùng dưới ảnh. |
| Floor-pass | `LIVE_FLOOR_TRASH_CONF` | 0.10 | Không bỏ sót rác nhỏ, xác nhận bằng điều kiện thời gian. |
| Tracking | `HISTORY_FRAMES` | 90 | Lưu lịch sử 15 giây. |
| Tracking | `PERSON_ID_MATCH_RADIUS` | 180 | Bù mất ID người giữa các frame. |
| Tracking | `TRASH_ID_MATCH_RADIUS` | 90 | Bù jitter bbox rác. |
| Owner | `MIN_SCORE` | 0.12 | Ngưỡng owner thấp nhưng có điều kiện xác nhận bổ sung. |
| Owner | `AMBIGUOUS_MARGIN` | 0.12 | Loại trường hợp nhiều người điểm gần nhau. |
| Owner | `REEVAL_SCORE_THRESH` | 0.25 | Đánh giá lại nếu score chưa đủ ổn định. |
| Owner | `MIN_OWNER_SEEN_FRAMES` | 2 | Tránh dùng detection nhất thời. |
| Owner | `MIN_OWNER_MOTION_PX` | 5 | Owner phải có chuyển động tối thiểu. |
| Vi phạm | `CONFIRM_FRAMES` | 12 | Rác tồn tại tối thiểu 2 giây. |
| Vi phạm | `CONFIRM_FRAMES_SUDDEN` | 8 | Xác nhận nhanh tình huống đột ngột sau 1.33 giây. |
| Vi phạm | `MIN_OWNER_GONE_FRAMES` | 6 | Owner rời đi tối thiểu 1 giây. |
| Vi phạm | `STATIONARY_REQUIRED` | 10 | Rác đứng yên 1.67 giây. |
| Vi phạm | `STATIONARY_PX` | 16 | Cho phép jitter bbox nhỏ nhưng không nhận vật đang di chuyển. |
| Vi phạm | `VIOLATION_GROUND_MIN_Y_RATIO` | 0.58 | Chỉ xét rác ở nửa dưới/mặt đất. |
| Recover | `STALE_FRAMES` | 90 | Xóa rác nếu mất quá 15 giây. |
| Recover | `LOST_TRASH_RECOVERY_FRAMES` | 60 | Recover rác mất track trong 10 giây. |
| Recover | `LOST_TRASH_MIN_SEEN` | 2 | Chỉ recover nếu rác từng xuất hiện đủ tin cậy. |
| Evidence | `EVIDENCE_CLIP_SECONDS` | 45 | Lưu bối cảnh trước/sau vi phạm. |
| Evidence | `EVIDENCE_CLIP_MAX_HEIGHT` | 720 | Giảm dung lượng bằng chứng. |
| Stream | `STREAM_MAX_HEIGHT` | 720 | Gửi preview đủ xem, giảm băng thông. |
| Stream | `STREAM_JPEG_QUALITY` | 50 | Tối ưu truyền WebSocket. |

## 16. Kết quả đầu ra của hệ thống

Hệ thống tạo ra các đầu ra sau:

1. **Frame realtime trên dashboard**: hiển thị video/camera đã vẽ bbox, ID, HUD.
2. **Alert vi phạm**: gửi qua WebSocket khi có vi phạm.
3. **Ảnh bằng chứng**: lưu trong thư mục `violations/`.
4. **Video clip bằng chứng**: lưu cùng thư mục nếu đủ frame trong buffer.
5. **Bản ghi SQLite**: lưu lịch sử vi phạm.
6. **Video kết quả**: với video file, sau khi xử lý xong có thể tải `output_h264.mp4` hoặc `output_raw.mp4`.

## 17. Đánh giá ưu điểm

- Hệ thống không chỉ nhận diện rác đơn lẻ, mà có suy luận theo thời gian.
- Có cơ chế gán owner dựa trên nhiều tín hiệu: khoảng cách, hướng di chuyển, optical flow, recency.
- Hỗ trợ cả video file và camera real-time.
- Có cơ chế giảm độ trễ bằng queue kích thước 1, camera buffer 1 và FPS target 6.
- Có lưu ảnh/video bằng chứng và lịch sử vi phạm.
- Có cơ chế tránh ghi trùng vi phạm cho cùng một người trong một session.
- Các tham số frame được quy đổi dựa trên FPS xử lý, giúp hệ thống có cơ sở thời gian rõ ràng.

## 18. Hạn chế

- Nếu rác quá nhỏ hoặc bị che khuất lâu, YOLO có thể bỏ sót.
- Nếu nhiều người đứng gần nhau, hệ thống có thể đánh dấu ambiguous và không xác nhận.
- Các ngưỡng pixel phụ thuộc vào góc camera và kích thước frame sau resize.
- Camera rung hoặc ánh sáng thay đổi mạnh có thể làm MOG2 và optical flow nhiễu.
- `LIVE_YOLO_CONF` thấp giúp tăng recall nhưng có thể tăng false positive, nên cần điều kiện thời gian để bù.
- Hệ thống hiện xử lý một camera chính, chưa tối ưu cho nhiều camera đồng thời.

## 19. Hướng phát triển

- Huấn luyện thêm dữ liệu ở nhiều góc camera, nhiều loại rác và nhiều điều kiện ánh sáng.
- Đánh giá định lượng bằng Precision, Recall, F1-score và mAP cho mô hình YOLO.
- Tối ưu inference bằng ONNX/TensorRT hoặc GPU để tăng FPS.
- Bổ sung pose estimation để xác định hành động tay/người vứt rác rõ hơn.
- Bổ sung multi-camera tracking.
- Tự động hiệu chỉnh vùng mặt đất thay vì dùng ngưỡng `VIOLATION_GROUND_MIN_Y_RATIO` cố định.
- Thêm giao diện cấu hình thông số trực tiếp trên frontend.

## 20. Kết luận

Đề tài đã xây dựng được pipeline phát hiện hành vi vứt rác bừa bãi từ video và camera thời gian thực. Hệ thống sử dụng YOLO để phát hiện người/rác, ByteTrack để duy trì ID, MOG2 và optical flow để bổ sung thông tin chuyển động, sau đó dùng thuật toán scoring để xác định chủ thể liên quan đến rác.

Điểm quan trọng của hệ thống là chuyển bài toán từ phát hiện vật thể đơn lẻ sang suy luận hành vi theo thời gian. Các thông số như `CONFIRM_FRAMES`, `HISTORY_FRAMES`, `STATIONARY_REQUIRED` được chọn dựa trên FPS xử lý mục tiêu 6 FPS, từ đó quy đổi sang các khoảng thời gian có ý nghĩa thực tế. Nhờ vậy, hệ thống giảm được cảnh báo sai do detection chớp nhoáng và có khả năng ghi nhận bằng chứng rõ ràng hơn.

## 21. Tài liệu tham khảo gợi ý

Các tài liệu có thể đưa vào phần tham khảo chính thức khi hoàn thiện báo cáo:

1. Joseph Redmon et al., "You Only Look Once: Unified, Real-Time Object Detection".
2. Ultralytics YOLO documentation.
3. Yifu Zhang et al., "ByteTrack: Multi-Object Tracking by Associating Every Detection Box".
4. Zivkovic, "Improved Adaptive Gaussian Mixture Model for Background Subtraction".
5. Lucas and Kanade, "An Iterative Image Registration Technique with an Application to Stereo Vision".
6. OpenCV documentation: VideoCapture, MOG2, Optical Flow, VideoWriter.
7. FastAPI documentation: WebSocket and REST API.
8. React documentation: component state, hooks, rendering.

## 22. Phụ lục: cách chạy hệ thống

### 22.1. Chạy video file

```powershell
cd E:\TGMTTTT
$env:DETECTOR_SOURCE_MODE="file"
$env:DETECTOR_VIDEO_FILE="E:\TGMTTTT\video\di_bo_17.mp4"
.\start_backend.ps1
```

### 22.2. Chạy camera real-time

```powershell
cd E:\TGMTTTT
$env:DETECTOR_SOURCE_MODE="ip_camera"
$env:DETECTOR_IP_CAM_HOST="192.168.0.108"
$env:DETECTOR_IP_CAM_PORT="8080"
$env:DETECTOR_IP_CAM_PROTOCOL="mjpeg"
.\start_backend.ps1
```

### 22.3. Chạy frontend khi máy chưa nhận npm

```powershell
cd E:\TGMTTTT\frontend
& "C:\Users\hoang\.cache\codex-runtimes\codex-primary-runtime\dependencies\node\bin\node.exe" ".\node_modules\vite\bin\vite.js"
```
