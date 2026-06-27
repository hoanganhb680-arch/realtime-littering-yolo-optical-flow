# LỜI THUYẾT TRÌNH TỪNG SLIDE - NHÓM 6

Tài liệu bám theo 15 trang trong `ThuyetTrinh-6.pptx`.

Thời lượng dự kiến: khoảng 13-16 phút.

Quy ước:

- Nội dung trong `[ngoặc vuông]` là hướng dẫn thao tác, không đọc thành tiếng.
- Không cần học thuộc từng chữ. Chỉ cần nhớ ý chính và kể lại như đang giải thích cho một người chưa biết đề tài.
- Khi nói thuật ngữ tiếng Anh, nói chậm: YOLO, ByteTrack, MOG2, Optical Flow, Ownership Score.

## Hai chỗ cần lưu ý trước khi trình bày

1. Ở slide 7, không nên nói “MOG2 phát hiện chuyển động của nền”. Nói chính xác là: MOG2 mô hình hóa nền và tìm vùng foreground, tức vùng đang khác với nền.
2. Ở slide 10, ví dụ `P=0,8; D=0,6; F=0,5; R=0,7` cho kết quả đúng là:

```text
0,4 x 0,8 + 0,3 x 0,6 + 0,2 x 0,5 + 0,1 x 0,7 = 0,67
```

Nếu trên slide đang ghi `0,65`, nên sửa thành `0,67` trước khi thuyết trình.

---

# Slide 1 - Trang tiêu đề

Thời lượng: 35-45 giây.

## Lời nói

> Em xin kính chào thầy và các bạn. Nhóm em là nhóm 6, hôm nay nhóm xin trình bày đồ án “Hệ thống phát hiện hành vi vứt rác bừa bãi bằng thị giác máy tính”.
>
> Bài toán của nhóm không chỉ là nhìn vào một ảnh rồi phát hiện có người và có rác. Mục tiêu khó hơn là theo dõi diễn biến trong nhiều frame để xác định vật rác có thực sự bị bỏ lại hay không, và người nào có khả năng liên quan nhất.
>
> Để thực hiện việc đó, hệ thống kết hợp YOLOv8s để phát hiện người và rác, ByteTrack để duy trì ID, MOG2 và Optical Flow để bổ sung thông tin chuyển động, sau đó dùng Trash Lifecycle và Ownership Score để suy luận hành vi.
>
> Trong phần trình bày, nhóm sẽ đi từ bài toán, dữ liệu, nguyên lý từng phương pháp, cho đến luồng triển khai và kết quả kiểm thử.

## Câu chuyển trang

> Trước hết, nhóm xin trình bày lý do vì sao bài toán này không thể giải quyết chỉ bằng một frame ảnh.

---

# Slide 2 - Bối cảnh và yêu cầu bài toán

Thời lượng: 55-70 giây.

## Lời nói

> Trong thực tế, việc giám sát vứt rác bằng con người rất khó duy trì liên tục, đặc biệt khi có nhiều camera hoặc phải quan sát trong thời gian dài.
>
> Nếu chỉ dùng một ảnh đơn lẻ, hệ thống có thể nhìn thấy một người đứng cạnh một túi rác, nhưng chưa thể biết người đó vừa bỏ túi xuống, đang cầm túi lên, hay chỉ tình cờ đi ngang qua. Vì vậy, một detection đơn lẻ chưa đủ để kết luận hành vi.
>
> Ngoài ra, rác thường có kích thước nhỏ, dễ bị chân người che khuất, có màu gần giống nền hoặc bị mờ khi camera rung. Điều này làm cho kết quả nhận diện giữa các frame có thể không ổn định.
>
> Từ các khó khăn đó, hệ thống cần đáp ứng ba yêu cầu chính. Thứ nhất là theo dõi người và rác qua nhiều frame. Thứ hai là kiểm tra rác có thực sự nằm lại đủ lâu hay không. Thứ ba là gán người liên quan dựa trên lịch sử và chuyển động, thay vì chỉ chọn người đứng gần rác nhất ở thời điểm hiện tại.

## Nhấn mạnh

> Ý quan trọng nhất của đề tài là: nhóm phân tích một quá trình theo thời gian, không kết luận từ một ảnh.

## Câu chuyển trang

> Để mô hình có thể nhận diện được người và các loại rác trong bối cảnh thực tế, nhóm đã xây dựng bộ dữ liệu riêng.

---

# Slide 3 - Dataset

Thời lượng: 50-65 giây.

## Lời nói

> Dữ liệu của nhóm được thu thập bằng cách tự quay các tình huống gần với môi trường giám sát thực tế như vỉa hè, lòng đường, khu vực cầu thang và khu vực tập kết rác.
>
> Nhóm thu thập khoảng 45 video thô. Sau bước sàng lọc, trích frame và làm sạch dữ liệu, các ảnh phù hợp mới được đưa vào tập huấn luyện và kiểm định.
>
> Các video có cả tình huống người đi bộ và người đi xe máy vứt rác. Loại rác gồm túi bóng, chai nhựa, vỏ bánh và một số vật nhỏ khác. Việc đưa nhiều góc quay và nhiều kích thước rác vào dữ liệu giúp mô hình học được các bối cảnh đa dạng hơn.
>
> Trong dữ liệu, nhóm gán nhãn hai lớp chính là `person` và `trash`. Mỗi nhãn gồm class và bounding box bao quanh đối tượng. Sau đó dữ liệu được chia thành tập train và validation để huấn luyện và đánh giá mô hình.

## Nếu chỉ vào hình

> [Chỉ lần lượt ba ảnh.] Ba ảnh này minh họa các bối cảnh khác nhau: góc cầu thang, đường có xe máy và khu vực có người đi bộ.

## Câu chuyển trang

> Sau khi có dữ liệu, nhóm định nghĩa toàn bộ bài toán thành một chuỗi xử lý theo thời gian như sau.

---

# Slide 4 - Định nghĩa bài toán theo chuỗi thời gian

Thời lượng: 65-80 giây.

## Lời nói

> Toàn bộ hệ thống có thể hiểu bằng bốn giai đoạn.
>
> Giai đoạn thứ nhất là quan sát frame. Hệ thống đọc lần lượt từng frame từ video hoặc camera và điều chỉnh tốc độ xử lý để phù hợp khả năng của máy.
>
> Giai đoạn thứ hai là nhận diện và giữ ID. YOLO phát hiện bounding box của người và rác, còn ByteTrack liên kết các detection giữa nhiều frame để biết đối tượng nào vẫn là cùng một người hoặc cùng một vật rác.
>
> Giai đoạn thứ ba là theo dõi vòng đời rác. Khi rác mới xuất hiện, hệ thống chưa báo vi phạm mà đặt trạng thái là `pending`. Sau đó hệ thống đếm số frame rác tồn tại và kiểm tra vị trí rác có ổn định hay không.
>
> Cuối cùng là xác nhận vi phạm. Hệ thống chỉ cảnh báo khi rác tồn tại đủ lâu, đứng yên, nằm ở khu vực phù hợp và owner được gán với điểm đủ tin cậy.

## Nhấn mạnh

> Vì vậy, YOLO chỉ cung cấp quan sát ban đầu. Phần hiểu hành vi nằm ở tracking, lịch sử frame và các điều kiện suy luận phía sau.

## Câu chuyển trang

> Để tạo được các dữ liệu quan sát đó, nhóm kết hợp bốn nhóm phương pháp chính.

---

# Slide 5 - Tổng quan bốn nhóm phương pháp

Thời lượng: 55-70 giây.

## Lời nói

> Phương pháp thứ nhất là YOLO Object Detection. Trong dự án, nhóm sử dụng YOLOv8s đã fine-tune cho hai lớp person và trash. YOLO trả về bounding box, class và confidence trong từng frame.
>
> Phương pháp thứ hai là ByteTrack kết hợp matching ID. ByteTrack duy trì ID qua thời gian. Ngoài ra, code của nhóm còn có một lớp nối lại ID bằng khoảng cách khi tracker bị mất hoặc đổi ID ngắn hạn.
>
> Phương pháp thứ ba là MOG2. MOG2 xây dựng mô hình nền và tìm các vùng foreground, tức những vùng đang khác so với nền. Nó không biết vùng đó là người hay rác mà chỉ cung cấp tín hiệu cảnh có thay đổi.
>
> Phương pháp thứ tư là Lucas-Kanade Optical Flow. Phương pháp này ước lượng vector dịch chuyển gần đây của điểm neo trên mỗi người, từ đó hỗ trợ xác định người đang di chuyển theo hướng nào.
>
> Bốn phương pháp này chưa trực tiếp kết luận vi phạm. Chúng tạo dữ liệu đầu vào cho Trash Lifecycle và Ownership Score.

## Câu chuyển trang

> Tiếp theo, nhóm xin đi chi tiết vào quá trình YOLO phát hiện đối tượng và ByteTrack duy trì ID.

---

# Slide 6 - Nhận diện và theo dõi đối tượng

Thời lượng: 80-100 giây.

## Lời nói

> Khi chương trình bắt đầu, hệ thống load file `weights/best.pt` một lần. Đây là checkpoint YOLOv8s đã được huấn luyện trước bằng dữ liệu của nhóm. File này chứa cấu trúc và các trọng số đã học để nhận diện hai lớp person và trash.
>
> Sau đó, video được đọc từng frame. Mỗi frame được YOLO tiền xử lý, đưa qua backbone, neck và detection head để tạo các bounding box. Kết quả của mỗi box gồm tọa độ bốn cạnh, class và confidence.
>
> Tuy nhiên, YOLO chỉ nhận diện trong từng frame. Để biết người ở frame sau có phải người ở frame trước hay không, hệ thống sử dụng ByteTrack. ByteTrack dùng Kalman Filter để dự đoán vị trí của track cũ, rồi so sánh bbox dự đoán với bbox mới của YOLO để quyết định có giữ ID hay không.
>
> Nếu một người bị che ngắn hạn, track chưa bị xóa ngay mà được giữ ở trạng thái `lost` trong buffer. Khi người xuất hiện lại ở vị trí phù hợp, ByteTrack có thể khôi phục ID cũ.
>
> Sau ByteTrack, code nhóm còn kiểm tra khoảng cách với lịch sử. Với người, nếu điểm gần chân mới nằm trong bán kính 180 pixel so với lịch sử cũ thì hệ thống có thể dùng lại ID cũ. Với rác, ngưỡng tương ứng là 90 pixel.

## Chỉ vào ba hình

> [Chỉ ba frame.] Ba hình này cho thấy cùng một đối tượng được giữ ID khi di chuyển qua nhiều frame, thay vì mỗi frame lại bị coi là một người mới.

## Câu chuyển trang

> Sau khi đã có đối tượng và ID, hệ thống tiếp tục trích xuất thông tin chuyển động bằng MOG2 và Optical Flow.

---

# Slide 7 - Phân tích chuyển động hỗ trợ suy luận

Thời lượng: 90-110 giây.

## Lời nói

> Ở phần bên trái là MOG2. MOG2 không nhận diện rác mà xây dựng mô hình thống kê của nền tại từng pixel. Nếu giá trị pixel hiện tại không phù hợp với mô hình nền, pixel đó được đánh dấu là foreground.
>
> Sau đó mask foreground được làm sạch bằng phép opening để xóa nhiễu nhỏ và closing để lấp các lỗ trong vùng chuyển động. Code tìm contour, loại các vùng có diện tích dưới 300 pixel và lấy tâm của những vùng còn lại.
>
> Trong hệ thống hiện tại, MOG2 có hai vai trò hỗ trợ. Thứ nhất, khi hệ thống đang tìm owner và có foreground nằm trong bán kính 50 pixel quanh tâm rác, điểm owner được tăng nhẹ 15 phần trăm. Tuy nhiên tín hiệu này chỉ cho biết khu vực rác vừa có biến động, không xác định chuyển động thuộc người nào.
>
> Vai trò thứ hai là hỗ trợ khi YOLO tạm mất một vật rác đã từng được phát hiện. Code có thể tìm foreground gần vị trí cuối của rác để ước lượng vị trí tạm. Đây chỉ là heuristic nên có thể bị nhiễu bởi chân người hoặc vật khác.
>
> Ở phần bên phải là Optical Flow Lucas-Kanade. Code theo dõi một điểm neo gần chân của từng người giữa hai frame liên tiếp. Kết quả là vector cho biết điểm đó đang dịch chuyển theo hướng nào. Nếu vector có xu hướng rời xa vị trí rác thì `flow_score` sẽ cao hơn.

## Chỉ vào hình vector

> [Chỉ mũi tên.] Trong minh họa, người di chuyển theo hướng ra xa rác nên vector chuyển động củng cố khả năng người đó có liên quan.

## Câu chuyển trang

> Sau khi có detection, ID và chuyển động, hệ thống bắt đầu quản lý vòng đời của từng vật rác.

---

# Slide 8 - Vòng đời đối tượng rác

Thời lượng: 60-75 giây.

## Lời nói

> Khi một rác mới được YOLO phát hiện, hệ thống tạo một bản ghi trong `trash_registry`. Bản ghi này lưu ID rác, vị trí xuất hiện, vị trí cuối, owner ứng viên, điểm owner và trạng thái hiện tại.
>
> Ban đầu rác có trạng thái `pending`, nghĩa là mới chỉ là ứng viên và chưa đủ cơ sở kết luận vi phạm.
>
> Mỗi lần rác tiếp tục được nhìn thấy hoặc được phục hồi tạm thời, bộ đếm `confirm_ctr` tăng lên. Bộ đếm này cho biết rác đã được theo dõi đủ lâu hay chưa.
>
> Đồng thời, hệ thống so sánh tâm rác giữa hai frame. Nếu tâm lệch dưới 16 pixel thì được coi là đứng yên trong frame đó và `stationary_ctr` tăng. Nếu rác dịch chuyển nhiều hơn, bộ đếm đứng yên được đặt lại.
>
> Chỉ khi rác tồn tại đủ số frame, đứng yên đủ lâu và các điều kiện owner đều đạt, trạng thái mới chuyển từ `pending` sang `confirmed`.

## Nhấn mạnh

> Hai khái niệm phải phân biệt là “tồn tại đủ lâu” và “đứng yên đủ lâu”. Một chiếc túi đang được người cầm có thể tồn tại nhiều frame nhưng vẫn di chuyển, nên chưa được xem là rác bị bỏ lại.

## Câu chuyển trang

> Tiếp theo là phần hệ thống đánh giá người nào có khả năng liên quan nhất đến vật rác.

---

# Slide 9 - Gán chủ thể bằng Ownership Score

Thời lượng: 65-80 giây.

## Lời nói

> Hệ thống không chọn đơn giản người đang đứng gần rác nhất. Lý do là người thực sự bỏ rác có thể đã rời đi, trong khi một người khác vừa đi tới và tình cờ đứng gần rác.
>
> Vì vậy nhóm xây dựng Ownership Score từ bốn tín hiệu theo lịch sử.
>
> Thành phần thứ nhất là Proximity, chiếm 40 phần trăm, đo xem người đó đã từng đi gần vị trí rác tới mức nào.
>
> Thành phần thứ hai là Direction, chiếm 30 phần trăm, kiểm tra quỹ đạo của người có xu hướng rời xa rác hay không.
>
> Thành phần thứ ba là Flow, chiếm 20 phần trăm, sử dụng vector Optical Flow gần đây để củng cố hướng chuyển động.
>
> Thành phần cuối là Recency, chiếm 10 phần trăm, ưu tiên người ở gần rác sát thời điểm rác xuất hiện hơn.
>
> Sau khi tính điểm cho từng người, hệ thống chọn người có điểm cao nhất, nhưng chỉ sử dụng kết quả nếu điểm vượt ngưỡng và không quá gần điểm của người đứng thứ hai.

## Câu chuyển trang

> Ở trang tiếp theo, nhóm trình bày cụ thể cách tính từng thành phần trong công thức này.

---

# Slide 10 - Tính điểm owner bằng bốn tín hiệu

Thời lượng: 85-105 giây.

## Lời nói

> Trước hết là Proximity. Hệ thống không chỉ đo khoảng cách ở frame hiện tại mà duyệt lịch sử điểm chân của từng người trong 90 frame gần nhất. Người từng đi gần vị trí rác hơn sẽ có điểm proximity cao hơn.
>
> Direction được tính từ quỹ đạo tọa độ. Code so sánh hướng dịch chuyển của người với vector hướng từ rác ra người. Nếu hai hướng gần cùng chiều, nghĩa là người đang rời xa rác, thì điểm direction tăng.
>
> Flow cũng đánh giá hướng rời xa nhưng dựa trên thay đổi ảnh do Lucas-Kanade tính được, thay vì chỉ dựa trên tọa độ bounding box.
>
> Recency giảm theo số frame đã trôi qua kể từ lần người ở gần rác nhất. Người vừa đi qua vị trí rác có điểm cao hơn người đã đi qua từ lâu.
>
> Bốn thành phần được tổng hợp theo công thức: `0,4P + 0,3D + 0,2F + 0,1R`.
>
> Ví dụ, nếu P bằng 0,8; D bằng 0,6; F bằng 0,5 và R bằng 0,7 thì điểm tổng đúng là 0,67.
>
> Sau khi tính xong, nếu điểm nhỏ hơn `MIN_SCORE` thì hệ thống chưa gán owner. Nếu điểm người thứ nhất và thứ hai chênh nhau dưới `AMBIGUOUS_MARGIN`, hệ thống đánh dấu mơ hồ để tránh gán sai.

## Nhấn mạnh

> Ownership Score là điểm luật do nhóm thiết kế, không phải xác suất thống kê và cũng không phải confidence của YOLO.

## Câu chuyển trang

> Khi đã có owner ứng viên, hệ thống kết hợp với trạng thái rác để phân loại ba trường hợp vi phạm.

---

# Slide 11 - Điều kiện xác nhận và phân loại vi phạm

Thời lượng: 75-90 giây.

## Lời nói

> Trước khi xét ba loại hành vi, hệ thống yêu cầu một số điều kiện chung: owner score phải đủ ngưỡng, owner đã xuất hiện đủ số frame, owner có chuyển động thật, không bị mơ hồ và rác phải nằm ở khu vực mặt đất.
>
> Trường hợp thứ nhất là “Đột ngột”. Trường hợp này xảy ra khi owner rời khỏi khung hình nhanh sau khi rác xuất hiện. Hệ thống dùng ngưỡng xác nhận ngắn hơn để bắt hành vi bỏ vật rồi đi nhanh.
>
> Trường hợp thứ hai là “Đứng yên”. Owner đã rời khỏi ảnh và rác tiếp tục đứng yên đủ lâu sau thời điểm đó. Điều này cho thấy vật không còn đi theo người mà đã nằm lại.
>
> Trường hợp thứ ba là “Bỏ rác tại chỗ”. Owner vẫn còn trong khung hình, nhưng rác đã tồn tại ít nhất 12 processed frame và đứng yên ít nhất 10 frame.
>
> Khi một trong ba nhánh đạt đủ điều kiện, trạng thái rác chuyển sang `confirmed`, hệ thống lưu ảnh bằng chứng và gửi cảnh báo.

## Câu chuyển trang

> Tuy nhiên, trong video thực tế có nhiều trường hợp khó như che khuất, nhiều người đứng gần hoặc rác bị mất detection. Vì vậy nhóm bổ sung một số cơ chế giảm báo nhầm.

---

# Slide 12 - Cơ chế giảm báo nhầm

Thời lượng: 75-95 giây.

## Lời nói

> Trường hợp thứ nhất là owner mơ hồ. Nếu hai người có điểm quá gần nhau, hệ thống không nên cố chọn ngay một người vì dễ gán sai. Code tiếp tục theo dõi và có thể đánh giá lại owner trong các frame sau.
>
> Trường hợp thứ hai là rác mất detection ngắn hạn. Nếu YOLO đã thấy rác ít nhất hai frame nhưng sau đó tạm mất, hệ thống vẫn giữ bản ghi rác. Code có thể tìm vị trí gần chân owner, tìm foreground MOG2 gần vị trí cuối hoặc giữ nguyên vị trí cuối nếu nó nằm ở vùng mặt đất.
>
> Đây là cơ chế hỗ trợ chứ không phải detection thật. Chẳng hạn vùng MOG2 có thể là chân người khác, nên nếu bán kính phục hồi quá rộng thì có nguy cơ làm lệch tâm rác. Vì vậy khi giải thích kết quả, nhóm xem đây là heuristic chống mất dấu chứ không khẳng định MOG2 đã nhận diện được rác.
>
> Trường hợp thứ ba là ràng buộc mặt đất. Điều kiện này giúp tránh xem vật đang ở trên tay hoặc trong túi là rác bị bỏ lại. Code chính dùng ngưỡng theo chiều cao ảnh và có thêm fallback theo khoảng cách tới quỹ đạo chân owner cho các góc camera cao.

## Câu chuyển trang

> Các thuật toán trên được đóng gói trong một hệ thống gồm nguồn video, AI backend, API, lưu trữ và giao diện.

---

# Slide 13 - Kiến trúc triển khai hệ thống

Thời lượng: 55-70 giây.

## Lời nói

> Kiến trúc hệ thống gồm năm phần.
>
> Phần đầu vào là file MP4 hoặc camera IP sử dụng MJPEG hay RTSP. Hệ thống hiện xử lý mục tiêu khoảng 6 FPS để cân bằng độ trễ và tải CPU.
>
> Phần AI Pipeline thực hiện toàn bộ quá trình YOLO, ByteTrack, MOG2, Optical Flow và suy luận vi phạm.
>
> Khi vi phạm được xác nhận, backend FastAPI tiếp nhận dữ liệu, điều phối luồng video và cảnh báo realtime. WebSocket được sử dụng để đẩy frame và alert lên giao diện.
>
> Thông tin vi phạm được lưu trong SQLite. Ảnh hoặc video bằng chứng có thể được lưu cục bộ hoặc đồng bộ lên MinIO tùy cấu hình.
>
> Cuối cùng, frontend React hiển thị video đã chú thích, số lượng vi phạm, danh sách cảnh báo và ảnh bằng chứng.

## Nhấn mạnh

> Việc nhận diện và suy luận được thực hiện ở backend. Frontend chủ yếu nhận kết quả để hiển thị, không trực tiếp chạy mô hình AI.

## Câu chuyển trang

> Sau khi hoàn thiện pipeline, nhóm tiến hành kiểm thử trên nhiều tình huống khác nhau.

---

# Slide 14 - Thực nghiệm và đánh giá

Thời lượng: 65-85 giây.

## Lời nói

> Nhóm xây dựng các kịch bản kiểm thử gồm: người chỉ đi ngang nhưng không bỏ rác, người bỏ rác rồi rời đi, người vẫn đứng gần rác, rác nhỏ hoặc mất detection ngắn hạn và trường hợp có nhiều người ở gần.
>
> Việc đánh giá được thực hiện theo từng tầng. Ở tầng detection, nhóm kiểm tra YOLO có nhận diện đúng person và trash hay không. Ở tầng tracking, nhóm kiểm tra ID có ổn định qua nhiều frame hay không. Ở tầng suy luận, nhóm xem owner score có hợp lý, có tránh gán khi mơ hồ và cảnh báo có được lưu đúng không.
>
> Với checkpoint hiện tại, chỉ số validation của YOLO đạt precision khoảng 0,872; recall khoảng 0,739; mAP50 khoảng 0,805 và mAP50 đến 95 khoảng 0,449.
>
> Trong lần chạy lại bộ sáu video kiểm thử gần nhất ở cấu hình khoảng 6 FPS, hệ thống xác nhận được năm video. Video xe máy vẫn khó vì vật rác nhỏ và detection xuất hiện ngắt quãng. Điều này cho thấy giới hạn hiện tại chủ yếu nằm ở chất lượng detection vật nhỏ và việc lấy mẫu frame.

## Chỉ vào giao diện

> [Chỉ ảnh dashboard.] Trên giao diện, người dùng có thể xem video đã đánh box, loại vi phạm, owner score và ảnh bằng chứng.

## Câu chuyển trang

> Từ quá trình xây dựng và kiểm thử, nhóm rút ra các kết luận và hướng phát triển như sau.

---

# Slide 15 - Kết luận và hướng phát triển

Thời lượng: 50-65 giây.

## Lời nói

> Qua đồ án, nhóm đã xây dựng được một pipeline hoàn chỉnh từ video đầu vào đến cảnh báo vi phạm. Điểm chính của hệ thống là không phụ thuộc vào một detection đơn lẻ mà kết hợp nhận diện, tracking, lịch sử vị trí, chuyển động và trạng thái đứng yên của rác.
>
> YOLOv8s đóng vai trò là mắt của hệ thống, ByteTrack giúp duy trì ID, MOG2 và Optical Flow bổ sung tín hiệu chuyển động, còn Trash Lifecycle và Ownership Score thực hiện phần suy luận theo thời gian.
>
> Tuy nhiên, hệ thống vẫn còn hạn chế khi rác quá nhỏ, camera rung, nhiều người đi sát nhau hoặc góc camera không phù hợp. Các heuristic phục hồi rác cũng có thể bị nhiễu bởi chân người hoặc vật khác.
>
> Trong tương lai, nhóm có thể mở rộng dataset, thiết lập ROI riêng cho từng camera, thử segmentation để xác định rác chính xác hơn, dùng ReID để giữ ID tốt hơn và bổ sung pose estimation hoặc action recognition để hiểu rõ động tác tay.
>
> Phần trình bày của nhóm đến đây là kết thúc. Nhóm em xin cảm ơn thầy và các bạn đã lắng nghe, và nhóm xin sẵn sàng trả lời câu hỏi.

---

# Bản mở đầu cực ngắn nếu bị giới hạn thời gian

> Em xin kính chào thầy và các bạn. Nhóm em trình bày hệ thống phát hiện hành vi vứt rác từ video. Điểm chính của đề tài là không kết luận từ một frame mà theo dõi người và rác qua thời gian. YOLOv8s phát hiện đối tượng, ByteTrack giữ ID, MOG2 và Optical Flow cung cấp tín hiệu chuyển động, sau đó hệ thống dùng Ownership Score và Trash Lifecycle để xác nhận vi phạm.

# Bản kết thúc cực ngắn

> Tóm lại, hệ thống đã kết hợp được detection, tracking và suy luận theo thời gian để phát hiện hành vi vứt rác và lưu bằng chứng. Hạn chế chính vẫn là rác nhỏ, che khuất và ID trong cảnh đông người. Nhóm em xin cảm ơn thầy và các bạn đã lắng nghe.

# Mẹo trình bày để người nghe dễ hiểu

1. Không đọc nguyên văn chữ trên slide. Slide là từ khóa; lời nói phải giải thích quan hệ nguyên nhân - kết quả.
2. Mỗi slide chỉ cần người nghe nhớ một ý:

```text
Slide 2: Một frame không đủ.
Slide 4: Hệ thống suy luận theo thời gian.
Slide 5: Mỗi thuật toán có một nhiệm vụ.
Slide 6: YOLO phát hiện, ByteTrack giữ ID.
Slide 7: MOG2 tìm thay đổi, Flow tìm hướng.
Slide 8: Rác phải tồn tại và đứng yên.
Slide 9-10: Không chọn người gần nhất; dùng bốn tín hiệu.
Slide 11: Đủ điều kiện mới xác nhận.
```

3. Khi chỉ hình, hãy nói trước rồi mới chỉ:

> “Ở hình bên phải, chúng ta có thể thấy...”

4. Khi nói công thức, không đọc quá nhanh. Hãy nói ý nghĩa trước, trọng số sau.
5. Nếu quên lời, quay về câu cốt lõi:

> “YOLO chỉ phát hiện vật thể; hành vi được suy luận từ nhiều frame.”

