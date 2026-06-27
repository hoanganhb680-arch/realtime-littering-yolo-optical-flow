# GIẢI THÍCH CHI TIẾT `src/detection_parsing.py` THEO TỪNG DÒNG

File gốc: `E:\TGMTTTT\src\detection_parsing.py`

Mục đích chung của file: nhận kết quả thô từ YOLO/ByteTrack và biến thành dữ liệu có thể dùng cho suy luận hành vi, gồm người hiện tại, rác hiện tại, ID ổn định, lịch sử vị trí người, người fallback từ motion và rác fallback ở vùng mặt đất.

---

## Dòng 1-8: Header và import thư viện

| Dòng | Giải thích |
|---:|---|
| 1-3 | Comment trang trí, báo rằng file này xử lý parsing detection và các helper liên quan tới ID. Không ảnh hưởng runtime. |
| 4 | `import cv2`: dùng OpenCV để vẽ hình lên frame, cụ thể dùng `cv2.circle`, `cv2.putText`, `cv2.rectangle`. |
| 5 | `import math`: dùng `math.hypot()` để tính khoảng cách Euclid giữa hai điểm `(x, y)`. |
| 6 | `from collections import deque`: dùng `deque` để lưu lịch sử vị trí người với độ dài giới hạn. |
| 7 | `import numpy as np`: dùng type hint `np.ndarray` cho ảnh/frame. |
| 8 | Dòng trống, chỉ để tách import với class. |

---

## Dòng 10-11: Khai báo class mixin

| Dòng | Giải thích |
|---:|---|
| 10 | `class DetectionParsingMixin:` khai báo một mixin. Mixin nghĩa là class này không chạy độc lập, mà được `TrashViolationDetector` kế thừa để có thêm các hàm parse detection. |
| 11 | Docstring: file này biến output model và motion blob thành ID người/rác ổn định. Đây chính là vai trò tổng quát của class. |

---

## Dòng 12-55: Hàm `_parse_detections()`

### Vai trò

Hàm này là hàm chính của file. Nó nhận output từ YOLO/ByteTrack và trả về:

```python
current_persons = {person_id: (x, y)}
current_trashes = {trash_id: (x, y)}
```

Trong đó `(x, y)` là điểm đại diện trong frame hiện tại.

| Dòng | Giải thích |
|---:|---|
| 12 | Bắt đầu khai báo hàm `_parse_detections`. Dấu `_` nghĩa là hàm nội bộ, chỉ dùng trong pipeline detector. |
| 13 | Tham số `results` là kết quả YOLO trả về; `frame_idx` là số thứ tự frame hiện tại. |
| 14 | Type hint: hàm trả về 2 dictionary. Dictionary thứ nhất cho người, dictionary thứ hai cho rác. Key là `int` ID, value là tuple tọa độ `(x, y)`. |
| 15 | Tạo dictionary rỗng `current_persons` để lưu người được thấy ở frame này. |
| 16 | Tạo dictionary rỗng `current_trashes` để lưu rác được thấy ở frame này. |
| 17 | Lấy danh sách bounding box từ `results[0].boxes`. YOLO trả về list result, frame hiện tại dùng phần tử đầu. |
| 18 | Nếu không có box nào thì không có object được detect. |
| 19 | Trả về 2 dict rỗng nếu không có detection. |
| 20 | Lấy ID tracking từ ByteTrack. Nếu `boxes.id` có tồn tại thì chuyển tensor sang numpy; nếu không có thì tạo list toàn `None`. |
| 21 | `used_trash_ids` lưu các ID rác đã dùng trong frame này để tránh hai bbox rác dùng trùng một ID. |
| 22 | Bắt đầu vòng lặp qua từng object YOLO detect được. |
| 23 | `boxes.xyxy.cpu().numpy()` lấy tọa độ bbox dạng `[x1, y1, x2, y2]`. `.cpu().numpy()` chuyển từ tensor sang numpy để xử lý Python. |
| 24 | `ids` là ID tracking tương ứng mỗi bbox. |
| 25 | `boxes.cls.cpu().numpy()` lấy class ID của bbox. Trong hệ này class `0` là `person`, class `1` là `trash`. |
| 26 | `boxes.conf.cpu().numpy()` lấy confidence của bbox. |
| 27 | Kết thúc phần `zip(...)`. Mỗi vòng lặp sẽ có `box`, `obj_id`, `cls`, `conf`. |
| 28 | Tách bbox thành `x1, y1, x2, y2`. Đây là góc trái trên và góc phải dưới của bounding box. |
| 29 | Tính hoành độ tâm bbox: trung bình của `x1` và `x2`. |
| 30 | Tính tung độ tâm bbox: trung bình của `y1` và `y2`. |
| 31 | Nếu class là `0` thì object này là người. |
| 32 | Gọi `_is_valid_person_box()` để lọc bbox người không đáng tin. |
| 33 | Nếu bbox người không hợp lệ thì bỏ qua object này, không lưu vào lịch sử. |
| 34-35 | Comment giải thích: hệ thống dùng điểm neo gần chân vì rác thường nằm dưới đất, gần chân hơn tâm thân người. |
| 36 | Gọi `_person_points_from_box()` để lấy `anchor` gần chân người và danh sách `points` phụ quanh vùng dưới bbox. |
| 37 | Gọi `_resolve_person_id()` để lấy ID người ổn định. Hàm này ưu tiên match với lịch sử cũ, sau đó mới dùng ID tracker hoặc tạo ID giả. |
| 38 | Lưu người hiện tại vào `current_persons` với key là `oid`, value là `anchor`. |
| 39 | Nếu người này chưa từng có trong `_person_history` thì cần tạo lịch sử mới. |
| 40 | Tạo `deque(maxlen=HISTORY_FRAMES)` để lưu lịch sử người. Khi quá dài, phần tử cũ tự bị bỏ. |
| 41 | Tăng số lần nhìn thấy người này. Nếu chưa có thì mặc định 0 rồi cộng 1. |
| 42 | Bắt đầu append một entry mới vào lịch sử người. |
| 43 | `_make_person_history_entry()` đóng gói `anchor`, `points`, `frame_idx` thành dict. |
| 44 | Kết thúc append vào history. |
| 45 | Điều kiện log debug: in thông tin vài lần đầu khi thấy người hoặc mỗi 50 frame. |
| 46 | Bắt đầu câu lệnh `print`. |
| 47 | In ID, confidence và số lần đã thấy người. |
| 48 | In anchor, số điểm phụ và frame hiện tại. |
| 49 | `flush=True` để log ra console ngay, không bị buffer. |
| 50 | Kết thúc `print`. |
| 51 | Nếu class là `1` thì object này là rác. |
| 52 | Gọi `_resolve_trash_id()` để lấy ID rác ổn định. Rác dùng tâm bbox `(cx, cy)`. |
| 53 | Đánh dấu ID rác này đã được dùng trong frame hiện tại. |
| 54 | Lưu rác hiện tại vào `current_trashes`. |
| 55 | Sau khi duyệt hết box, trả về danh sách người và rác hiện tại. |

### Tóm tắt đoạn 12-55

```text
YOLO raw boxes
-> tách class person/trash
-> lọc person không hợp lệ
-> lấy điểm chân người
-> gán ID ổn định
-> lưu lịch sử người
-> trả current_persons và current_trashes
```

---

## Dòng 57-122: Hàm `_augment_persons_from_motion()`

### Vai trò

Hàm này bổ sung người từ vùng chuyển động MOG2 khi YOLO bỏ sót người. Đây là fallback, không phải detection chính.

| Dòng | Giải thích |
|---:|---|
| 57 | Bắt đầu hàm `_augment_persons_from_motion`. |
| 58-64 | Khai báo tham số hàm. Hàm nhận người hiện tại, rác hiện tại, motion alerts, frame index, kích thước frame và frame đã annotate. |
| 65 | Type hint `-> None`: hàm sửa trực tiếp `current_persons`, `_person_history`, `annotated`, không trả dữ liệu mới. |
| 66 | Kiểm tra config `MOTION_PERSON_FALLBACK`. Nếu tắt thì không thêm người từ motion. |
| 67 | Thoát hàm nếu fallback motion bị tắt. |
| 68 | Nếu không có motion alert nào thì không có gì để bổ sung. |
| 69 | Thoát hàm nếu danh sách motion rỗng. |
| 71 | Lấy chiều cao `h` và chiều rộng `w` của frame. |
| 72 | Lấy ngưỡng diện tích motion nhỏ nhất. Blob nhỏ hơn ngưỡng này được coi là nhiễu. |
| 73 | Tính diện tích motion lớn nhất cho phép theo tỉ lệ frame. Blob quá lớn có thể là thay đổi nền, camera rung, vùng sáng lớn. |
| 74 | Tính tung độ nhỏ nhất cho motion person. Blob quá cao có thể không phải người trên mặt đất. |
| 75 | Tính tung độ lớn nhất cho motion person. |
| 76 | Số motion person tối đa được thêm trong một frame. Giới hạn để tránh thêm quá nhiều object giả. |
| 77 | `foot_ratio` dùng để dịch anchor từ tâm blob xuống phía dưới, gần chân hơn. |
| 79 | Biến đếm số person fallback đã thêm. |
| 80 | Duyệt các motion blob, sắp xếp theo diện tích giảm dần. Blob lớn hơn được xét trước vì khả năng là người cao hơn. |
| 81 | Nếu đã thêm đủ số lượng tối đa thì dừng. |
| 82 | `break` thoát vòng lặp. |
| 83 | Lọc blob theo diện tích và vị trí y. Nếu quá nhỏ, quá lớn, quá cao hoặc quá thấp thì bỏ. |
| 84 | Bỏ blob không hợp lệ. |
| 85 | Nếu motion blob nằm gần person YOLO đã detect thì bỏ, tránh thêm trùng người. |
| 86 | Bỏ blob trùng person hiện có. |
| 87 | Nếu motion blob nằm rất gần rác và diện tích chưa đủ lớn thì bỏ, tránh nhầm chuyển động nhỏ quanh rác thành người. |
| 88 | Bỏ blob nghi là rác/nhỏ nhiễu. |
| 90 | Tính offset từ diện tích blob. `area ** 0.5` xấp xỉ kích thước cạnh của blob; nhân `foot_ratio` để dịch xuống vùng chân. |
| 91 | Tạo anchor cho motion person: x là `mx`, y là `my + offset`, nhưng không vượt quá đáy frame. |
| 92 | Bắt đầu tạo danh sách điểm phụ cho motion person. |
| 93 | Điểm đầu là tâm motion blob. |
| 94 | Điểm thứ hai là anchor gần chân. |
| 95 | Điểm thứ ba thấp hơn anchor để bao phủ vùng chân/sàn. |
| 96 | Kết thúc danh sách points. |
| 97 | Gọi `_resolve_person_id()` với `raw_id=None` vì motion blob không có ID từ ByteTrack. |
| 98 | Thêm person fallback vào `current_persons`. |
| 99 | Nếu ID này chưa có history thì tạo mới. |
| 100 | Tạo `deque` lưu lịch sử cho person fallback. |
| 101 | Tăng số lần thấy person fallback. |
| 102 | Append entry mới vào lịch sử. |
| 103 | Entry gồm anchor, points, frame hiện tại. |
| 104 | Kết thúc append. |
| 106 | Vẽ vòng tròn lên frame tại anchor để debug/hiển thị motion person. |
| 107 | Bắt đầu vẽ text lên frame. |
| 108 | Frame cần vẽ là `annotated`. |
| 109 | Text hiển thị dạng `motion P{id}`. |
| 110 | Vị trí đặt text: lệch phải và nằm trên anchor một chút. |
| 111 | Font OpenCV. |
| 112 | Cỡ chữ 0.45. |
| 113 | Màu chữ cam nhạt `(255, 180, 0)` theo BGR. |
| 114 | Độ dày chữ 1. |
| 115 | Kết thúc `cv2.putText`. |
| 116 | Điều kiện in log debug tương tự person thường: vài lần đầu hoặc mỗi 50 frame. |
| 117 | Bắt đầu print. |
| 118 | In ID motion person và diện tích blob. |
| 119 | In số lần thấy, anchor và frame. |
| 120 | Flush log ngay. |
| 121 | Kết thúc print. |
| 122 | Tăng số lượng motion person đã thêm. |

### Tóm tắt đoạn 57-122

```text
MOG2 motion blobs
-> lọc blob giống người
-> tránh trùng person/rác
-> tạo synthetic person
-> lưu history
-> vẽ debug lên frame
```

---

## Dòng 124-134: Hàm `_is_valid_person_box()`

| Dòng | Giải thích |
|---:|---|
| 124 | Khai báo hàm kiểm tra bbox người có hợp lệ không. |
| 125 | Nếu confidence của bbox nhỏ hơn `LIVE_PERSON_CONF` thì không đáng tin. |
| 126 | Trả `False`, bỏ bbox này. |
| 127 | Tách bbox thành 4 tọa độ. |
| 128 | Lấy chiều cao, chiều rộng ảnh gốc từ YOLO result. |
| 129 | Tính chiều rộng bbox `bw` và chiều cao bbox `bh`; dùng `max(1.0, ...)` để tránh giá trị 0. |
| 130 | Tính chiều cao tối thiểu của person theo tỉ lệ ảnh. |
| 131 | Tính diện tích tối thiểu của person theo tỉ lệ diện tích frame. |
| 132 | Nếu bbox thấp hơn ngưỡng hoặc diện tích nhỏ hơn ngưỡng thì coi là person không hợp lệ. |
| 133 | Trả `False`. |
| 134 | Nếu vượt qua mọi điều kiện thì trả `True`. |

### Ý nghĩa

Hàm này ngăn false positive person đi vào hệ thống. Nếu một vật nhỏ bị YOLO nhầm thành người, nó có thể làm sai owner scoring. Vì vậy cần lọc confidence, chiều cao và diện tích.

---

## Dòng 136-163: Hàm `_person_points_from_box()`

### Vai trò

Từ bbox người, hàm này tạo:

```python
anchor = điểm chính gần chân người
points = nhiều điểm phụ quanh vùng dưới bbox
```

| Dòng | Giải thích |
|---:|---|
| 136 | Khai báo hàm tính điểm đại diện người từ bbox. |
| 137 | Ép 4 tọa độ bbox sang float để tính toán chính xác. |
| 138 | Lấy chiều cao và chiều rộng frame. |
| 139 | Tính chiều rộng và chiều cao bbox, đảm bảo ít nhất 1 pixel. |
| 140 | Tính tâm x của người. |
| 141 | Lấy tỉ lệ mở rộng xuống dưới bbox từ config. |
| 142 | Lấy tỉ lệ mở rộng sang hai bên từ config. |
| 144 | Khai báo hàm con `clamp_point()` để ép tọa độ không vượt biên frame. |
| 145 | Bắt đầu return tuple tọa độ đã clamp. |
| 146 | Clamp x trong khoảng `[0, w - 1]`. |
| 147 | Clamp y trong khoảng `[0, h - 1]`. |
| 148 | Kết thúc return của `clamp_point`. |
| 150 | Tạo `lower`: điểm chính ở khoảng 88% chiều cao bbox, tức gần chân hơn tâm người. |
| 151 | Bắt đầu tạo `raw_points`. |
| 152 | Điểm 1 là `lower`, điểm anchor chính. |
| 153 | Điểm 2 là tâm đáy bbox `(cx, y2)`. |
| 154 | Điểm 3 là điểm dưới đáy bbox, bù cho trường hợp chân/rác nằm thấp hơn bbox. |
| 155 | Điểm 4 là điểm giữa thân dưới, hỗ trợ match khi bbox/camera lệch. |
| 156 | Điểm 5 lệch trái và hơi dưới bbox, bù cho rác ở cạnh chân trái. |
| 157 | Điểm 6 lệch phải và hơi dưới bbox, bù cho rác ở cạnh chân phải. |
| 158 | Kết thúc list `raw_points`. |
| 159 | Tạo list rỗng `points` để loại điểm trùng. |
| 160 | Duyệt từng điểm trong `raw_points`. |
| 161 | Nếu điểm chưa có trong `points` thì giữ lại. |
| 162 | Thêm điểm không trùng. |
| 163 | Trả về `lower` làm anchor chính và `points` làm danh sách điểm phụ. |

### Vì sao cần nhiều điểm?

Camera thường nhìn nghiêng, rác có thể nằm thấp hơn chân bbox hoặc lệch trái/phải so với tâm người. Nếu chỉ dùng một điểm tâm người, khoảng cách người-rác dễ sai. Nhiều điểm vùng chân giúp owner scoring ổn định hơn.

---

## Dòng 165-175: Hàm `_make_person_history_entry()`

| Dòng | Giải thích |
|---:|---|
| 165 | `@staticmethod`: hàm không dùng `self`, chỉ đóng gói dữ liệu. |
| 166 | Khai báo hàm tạo entry lịch sử người. |
| 167 | Tham số `anchor`: điểm chính của người. |
| 168 | Tham số `points`: các điểm phụ quanh người. |
| 169 | Tham số `frame_idx`: frame hiện tại. |
| 170 | Type hint trả về dict. |
| 171 | Bắt đầu trả về dictionary. |
| 172 | Lưu anchor dạng tuple int. |
| 173 | Lưu toàn bộ points, ép từng tọa độ sang int. |
| 174 | Lưu frame index dạng int. |
| 175 | Kết thúc dictionary. |

### Ý nghĩa

Mỗi entry history lưu đủ thông tin để sau này tính:

- Người từng ở đâu.
- Các điểm quanh chân người ở đâu.
- Người gần rác tại frame nào.

---

## Dòng 177-198: Các hàm đọc entry lịch sử

Ba hàm này giúp đọc lịch sử theo một format thống nhất. Code có hỗ trợ cả entry kiểu dict mới và tuple/list kiểu cũ.

| Dòng | Giải thích |
|---:|---|
| 177 | `@staticmethod`: hàm không cần `self`. |
| 178 | Khai báo `_history_entry_anchor()`, lấy anchor từ entry. |
| 179 | Nếu entry là dict, dùng format mới. |
| 180 | Lấy field `"anchor"`, nếu thiếu thì mặc định `(0, 0)`. |
| 181 | Trả anchor dạng int. |
| 182 | Nếu entry không phải dict, coi entry là tuple/list cũ và lấy `entry[0], entry[1]`. |
| 184 | `@staticmethod`. |
| 185 | Khai báo `_history_entry_frame()`, lấy frame index từ entry. |
| 186 | Nếu entry là dict. |
| 187 | Lấy field `"frame"`, thiếu thì mặc định 0. |
| 188 | Nếu entry kiểu cũ, lấy `entry[2]` nếu có đủ phần tử, không thì trả 0. |
| 190 | `@staticmethod`. |
| 191 | Khai báo `_history_entry_points()`, lấy danh sách points từ entry. |
| 192 | Nếu entry là dict. |
| 193 | Lấy `"points"`, nếu không có thì dùng anchor làm list 1 điểm. |
| 194 | Nếu entry kiểu cũ có phần tử thứ 4 và phần tử này không rỗng. |
| 195 | Dùng phần tử thứ 4 làm points. |
| 196 | Nếu không có points. |
| 197 | Dùng anchor làm điểm duy nhất. |
| 198 | Ép tất cả points về list tuple int rồi trả về. |

### Ý nghĩa

Các helper này làm code phía sau không cần quan tâm entry history đang là dict hay tuple. Nó tăng tính tương thích khi code từng thay đổi format lưu history.

---

## Dòng 200-217: Hàm `_nearest_history_distance()`

### Vai trò

Tìm khoảng cách gần nhất giữa một điểm `center` và lịch sử một người.

| Dòng | Giải thích |
|---:|---|
| 200 | Khai báo hàm tính khoảng cách gần nhất tới history. |
| 201-205 | Tham số gồm `center`, `hist`, `frame_idx`, `max_age`. |
| 206 | Hàm trả về `(best_dist, best_frame)`: khoảng cách gần nhất và frame xảy ra khoảng cách đó. |
| 207 | Khởi tạo khoảng cách tốt nhất là vô cực, frame tốt nhất là `None`. |
| 208 | Duyệt từng entry trong lịch sử. |
| 209 | Lấy frame index của entry. |
| 210 | Tính tuổi của entry: frame hiện tại trừ frame entry. |
| 211 | Nếu entry ở tương lai hoặc quá cũ thì bỏ qua. |
| 212 | Bỏ entry không hợp lệ. |
| 213 | Duyệt mọi điểm phụ của người trong entry. |
| 214 | Tính khoảng cách Euclid từ `center` tới point `(px, py)`. |
| 215 | Nếu khoảng cách này nhỏ hơn best hiện tại. |
| 216 | Cập nhật khoảng cách tốt nhất và frame tốt nhất. |
| 217 | Trả kết quả. |

### Dùng ở đâu?

Hàm này dùng để kiểm tra rác floor-pass có gần lịch sử người không và trong owner fallback. Nó trả lời câu hỏi: “vật này có từng nằm gần người nào trong khoảng thời gian gần đây không?”

---

## Dòng 219-246: Hàm `_resolve_person_id()`

### Vai trò

Giữ ID người ổn định khi ByteTrack bị mất hoặc đổi ID.

| Dòng | Giải thích |
|---:|---|
| 219 | Khai báo hàm resolve ID người. |
| 220-224 | Tham số gồm raw ID từ tracker, anchor người, frame hiện tại, danh sách người đã có trong frame này. |
| 225 | Hàm trả về một ID kiểu int. |
| 226 | Lấy bán kính match person từ config. Nếu thiếu thì mặc định 160. |
| 227 | Khởi tạo ID tốt nhất và khoảng cách tốt nhất. |
| 228 | Duyệt tất cả người đã có history. |
| 229 | Nếu ID này đã được dùng trong frame hiện tại hoặc history rỗng thì bỏ qua. |
| 230 | Tiếp tục vòng lặp. |
| 231 | Lấy anchor cuối cùng của người cũ. |
| 232 | Lấy frame cuối cùng người cũ xuất hiện. |
| 233 | Nếu người cũ quá lâu không xuất hiện thì không match nữa. |
| 234 | Bỏ qua người quá cũ. |
| 235 | Tính khoảng cách từ anchor mới tới anchor cuối của người cũ. |
| 236 | Nếu khoảng cách nhỏ nhất hiện tại và nằm trong bán kính match. |
| 237 | Cập nhật best ID. |
| 238 | Nếu tìm được best ID từ lịch sử. |
| 239 | Trả về ID cũ, nghĩa là coi detection mới là cùng người cũ. |
| 240 | Nếu không match history nhưng raw ID từ ByteTrack có tồn tại. |
| 241 | Ép raw ID sang int. |
| 242 | Nếu raw ID chưa được dùng trong frame này. |
| 243 | Trả raw ID. |
| 244 | Nếu cả history và raw ID đều không dùng được, tạo synthetic ID mới. |
| 245 | Tăng bộ đếm synthetic person ID để lần sau không trùng. |
| 246 | Trả synthetic ID. |

### Thứ tự ưu tiên

```text
1. Match với lịch sử gần nhất
2. Dùng raw ID từ ByteTrack
3. Tạo synthetic ID mới
```

Lý do ưu tiên history trước: nếu ByteTrack đổi ID nhưng vị trí vẫn gần người cũ, hệ thống vẫn giữ ID cũ để owner history không bị đứt.

---

## Dòng 248-279: Hàm `_resolve_trash_id()`

### Vai trò

Giữ ID rác ổn định khi rác nhỏ, bbox jitter hoặc ByteTrack đổi ID.

| Dòng | Giải thích |
|---:|---|
| 248 | Khai báo hàm resolve ID rác. |
| 249-253 | Tham số gồm raw ID, tâm rác, frame hiện tại, tập ID đã dùng trong frame này. |
| 254 | Hàm trả về ID int. |
| 255 | Lấy bán kính match rác từ config, mặc định 80 nếu thiếu. |
| 256 | Khởi tạo ID tốt nhất và khoảng cách tốt nhất. |
| 257 | Duyệt registry các rác đang được hệ thống theo dõi. |
| 258 | Nếu ID rác này đã dùng trong frame hiện tại thì bỏ qua. |
| 259 | Tiếp tục vòng lặp. |
| 260 | Lấy frame rác được thấy gần nhất. Nếu không có thì dùng `spawn_frame`, thiếu nữa thì dùng frame hiện tại. |
| 261 | Nếu rác quá lâu không thấy, quá `STALE_FRAMES`, thì không match. |
| 262 | Bỏ qua rác stale. |
| 263 | Lấy vị trí cuối cùng của rác. |
| 264 | Nếu không có vị trí cuối thì bỏ. |
| 265 | Tiếp tục vòng lặp. |
| 266 | Tính khoảng cách từ tâm rác mới tới vị trí cuối của rác cũ. |
| 267 | Nếu khoảng cách tốt hơn và nằm trong bán kính match. |
| 268 | Cập nhật best ID. |
| 269 | Nếu tìm được best ID. |
| 270 | Trả ID rác cũ. |
| 272 | Nếu không match registry nhưng raw ID từ tracker có tồn tại. |
| 273 | Ép raw ID sang int. |
| 274 | Nếu raw ID chưa được dùng. |
| 275 | Trả raw ID. |
| 277 | Nếu không có ID nào dùng được, tạo synthetic trash ID. |
| 278 | Tăng bộ đếm synthetic trash ID. |
| 279 | Trả synthetic trash ID. |

### Ý nghĩa

Cùng một vật rác không bị tách thành nhiều ID khác nhau chỉ vì bbox hơi rung hoặc tracker đổi ID. Điều này rất quan trọng vì `trash_lifecycle.py` theo dõi trạng thái rác theo ID.

---

## Dòng 281-351: Hàm `_detect_floor_trash_candidates()`

### Vai trò

Đây là floor-pass: detect thêm rác nhỏ ở vùng dưới ảnh. Nó không thay YOLO chính, mà bổ sung khi full-frame detect bỏ sót rác.

| Dòng | Giải thích |
|---:|---|
| 281 | Khai báo hàm detect rác ở vùng mặt đất. |
| 282-290 | Tham số gồm model YOLO, frame gốc, frame annotate, rác/người hiện tại, motion alerts, frame index và tham số floor pass. |
| 291 | Hàm trả về dict rác bổ sung `{trash_id: center}`. |
| 292 | Nếu `floor_kwargs` là `None`, nghĩa là floor-pass không bật. |
| 293 | Trả về dict rỗng. |
| 294 | Lấy interval floor-pass từ config. |
| 295 | Nếu interval > 1 và frame hiện tại không đúng chu kỳ thì không chạy. |
| 296 | Trả rỗng để tiết kiệm xử lý. |
| 298 | Lấy chiều cao, chiều rộng frame. |
| 299 | Lấy tỉ lệ bắt đầu ROI từ config. |
| 300 | Tính `y0`, tức dòng bắt đầu vùng dưới ảnh. Clamp để không vượt frame. |
| 301 | Cắt ROI từ `y0` đến cuối ảnh. |
| 302 | Nếu ROI rỗng thì không xử lý. |
| 303 | Trả rỗng. |
| 305 | Bắt đầu try vì model predict có thể lỗi. |
| 306 | Chạy YOLO `predict` trên ROI với floor kwargs. |
| 307 | Nếu lỗi bất kỳ. |
| 308 | Trả rỗng, không làm sập pipeline chính. |
| 310 | Tạo dict `extra` để lưu rác floor-pass tìm được. |
| 311 | `used_ids` bắt đầu bằng ID các rác đã detect ở full-frame, tránh trùng. |
| 312 | Lấy boxes từ kết quả floor predict. |
| 313 | Nếu không có box. |
| 314 | Trả `extra` rỗng. |
| 316 | Bắt đầu duyệt từng box floor-pass. |
| 317 | Lấy bbox xyxy. |
| 318 | Lấy class. |
| 319 | Lấy confidence. |
| 320 | Kết thúc zip. |
| 321 | Nếu class không phải `1` tức không phải rác. |
| 322 | Bỏ qua. |
| 323 | Tách bbox trong tọa độ ROI. |
| 324 | Tính width/height bbox, đảm bảo ít nhất 1. |
| 325 | Tính diện tích bbox. |
| 326 | Nếu rác quá nhỏ hoặc quá lớn bất thường thì bỏ. Quá nhỏ dễ là nhiễu; quá lớn không giống rác nhỏ dưới sàn. |
| 327 | Bỏ box không hợp lệ. |
| 328 | Tính x tâm bbox trong ROI. Vì ROI giữ nguyên chiều ngang nên x không cần cộng offset. |
| 329 | Tính y tâm bbox trong frame gốc bằng cách cộng `y0`. |
| 330 | Tạo center rác trong tọa độ frame gốc. |
| 331 | Kiểm tra rác floor-pass có context liên quan người/motion không. |
| 332 | Nếu không có context thì bỏ. |
| 333 | Nếu rác này quá gần rác full-frame đã có thì bỏ để tránh trùng. |
| 334 | Bỏ trùng. |
| 335 | Resolve ID rác cho candidate mới. Raw ID là `None` vì predict ROI không dùng tracking ID. |
| 336 | Đánh dấu ID đã dùng. |
| 337 | Lưu rác candidate vào `extra`. |
| 339 | Chuyển bbox ROI sang bbox tọa độ frame gốc để vẽ. `gy1`, `gy2` cộng `y0`. |
| 340 | Chọn màu cam BGR để vẽ floor-trash. |
| 341 | Vẽ hình chữ nhật quanh rác floor-pass lên frame annotated. |
| 342 | Bắt đầu vẽ text. |
| 343 | Frame annotate. |
| 344 | Text gồm ID rác, chữ `trash floor`, confidence. |
| 345 | Vị trí text trên bbox. |
| 346 | Font OpenCV. |
| 347 | Cỡ chữ. |
| 348 | Màu chữ. |
| 349 | Độ dày chữ. |
| 350 | Kết thúc `putText`. |
| 351 | Trả về các rác floor-pass bổ sung. |

### Ý nghĩa

Floor-pass giúp tăng khả năng bắt rác nhỏ dưới đất. Nhưng nó không nhận bừa mọi vật dưới đất, vì dòng 331 bắt buộc candidate phải có context gần người, lịch sử người hoặc motion.

---

## Dòng 353-378: Hàm `_floor_candidate_has_context()`

### Vai trò

Kiểm tra rác floor-pass có liên quan đến hành vi không.

| Dòng | Giải thích |
|---:|---|
| 353 | Khai báo hàm kiểm tra context cho floor candidate. |
| 354-358 | Tham số gồm tâm candidate, người hiện tại, motion alerts, frame index. |
| 359 | Hàm trả về bool. |
| 360 | Tách tâm candidate thành `cx, cy`. |
| 361 | Bắt đầu kiểm tra candidate có gần người hiện tại không. |
| 362 | Tính khoảng cách từ candidate tới từng person hiện tại, so với `SPAWN_RADIUS`. |
| 363 | Duyệt các vị trí người hiện tại. |
| 364 | Kết thúc `any(...)`. Nếu có ít nhất một người gần candidate thì `near_person=True`. |
| 365 | Nếu gần người hiện tại. |
| 366 | Trả `True`, candidate có context. |
| 367 | Nếu không gần người hiện tại, duyệt lịch sử tất cả người. |
| 368 | Nếu history rỗng. |
| 369 | Bỏ qua. |
| 370 | Gọi `_nearest_history_distance()` để xem candidate từng gần lịch sử người không. |
| 371 | Truyền center, history, frame hiện tại và `HISTORY_FRAMES`. |
| 372 | Kết thúc lời gọi. |
| 373 | Nếu khoảng cách gần nhất tới history người nhỏ hơn `TRAJECTORY_RADIUS`. |
| 374 | Trả `True`, candidate có liên hệ với quỹ đạo người. |
| 375 | Nếu không gần người hiện tại/lịch sử, kiểm tra gần motion alerts không. |
| 376 | Tính khoảng cách candidate tới từng motion blob, ngưỡng 120 pixel. |
| 377 | Duyệt `mx, my, _` trong motion alerts. Dấu `_` bỏ qua area. |
| 378 | Trả kết quả `any(...)`. Nếu gần ít nhất một motion blob thì True, không thì False. |

### Ý nghĩa

Hàm này là lớp chống false positive cho floor-pass. Một vật nhỏ dưới đất chỉ được nhận là candidate rác nếu có ngữ cảnh hành vi: gần người, gần quỹ đạo người hoặc gần chuyển động.

---

## Dòng 380-382: Comment cuối file

| Dòng | Giải thích |
|---:|---|
| 380-382 | Comment phân cách `Trash processing`. Trong phiên bản hiện tại, phần xử lý rác thật đã được tách sang `trash_lifecycle.py`, nên ở file này chỉ còn comment cuối. |

---

## Tổng kết toàn file

`detection_parsing.py` là cầu nối giữa YOLO raw output và logic hành vi.

Nó làm toàn bộ chuỗi sau:

```text
1. Nhận boxes từ YOLO/ByteTrack
2. Tách người và rác theo class
3. Lọc bbox người không đáng tin
4. Chuyển bbox người thành điểm gần chân
5. Gán ID người ổn định
6. Lưu lịch sử vị trí người
7. Gán ID rác ổn định
8. Bổ sung người từ motion nếu YOLO miss
9. Bổ sung rác nhỏ ở vùng mặt đất bằng floor-pass
10. Chỉ nhận floor candidate nếu có context gần người/motion/history
```

Nếu thiếu file này, các module sau sẽ không có dữ liệu sạch để chạy:

- `OwnershipScorer.py` không có lịch sử người để tính owner.
- `owner_resolution.py` không biết người nào từng gần rác.
- `trash_lifecycle.py` không có ID rác ổn định để quản lý vòng đời.
- `violation_confirmation.py` không có trạng thái rác đáng tin để xác nhận vi phạm.

