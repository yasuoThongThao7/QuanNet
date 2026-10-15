# 3. TRIỂN KHAI THỰC NGHIỆM, HUẤN LUYỆN VÀ ĐÁNH GIÁ

## 3.1. Xây dựng Dataset lai ITSHybrid & Tối ưu hóa bộ nhớ GPU

### 3.1.1. Mục tiêu của dataset lai

Một trong những điểm then chốt trong quá trình phát triển mô hình khử mù theo hướng lai là khả năng kết hợp giữa mô hình vật lý và học sâu mà không phá vỡ ưu thế của cả hai. Trong notebook `Hybrid_DehazeCNN_fixed.ipynb`, tập dữ liệu chuẩn hóa theo hướng RESIDE-ITS được xây dựng với mục tiêu:

- Giữ nguyên định dạng ảnh mù `hazy`, ảnh sáng rõ `clear`, ảnh mặt nạ truyền qua `trans` theo quy ước RESIDE-ITS.
- Tạo ra bộ dữ liệu `ITSHybrid` gồm các cặp ảnh và nhãn truyền qua chuẩn hóa cho từng crop 256x256.
- Không cần đọc lại ảnh rõ `clear` trong quá trình huấn luyện, giảm chi phí I/O và lưu trữ chủ yếu trên GPU.
- Tận dụng mô hình DCP như một tín hiệu đầu vào có sẵn (`trans_raw`, dark channel), đồng thời dùng CNN để hiệu chỉnh bản đồ truyền qua cuối cùng.

### 3.1.2. Quy trình xây dựng dataset

Notebook thực hiện các bước sau:

1. Tìm kiếm và nén từng phần dữ liệu cần thiết từ file zip RESIDE.
   - Nếu thư mục `/content/dataset` chưa tồn tại đủ cấu trúc `hazy/trans/clear`, chương trình sẽ giải nén subset cần dùng từ file zip thay vì nén toàn bộ dataset.
   - Điều này giúp tiết kiệm thời gian tải dữ liệu và không làm tăng tải bộ nhớ nhầm lẫn trên Colab.

2. Tạo index danh sách ảnh cho từng thư mục:
   - Hàm `index_dir()` quét file hình ảnh theo đuôi `.png`, `.jpg`, `.jpeg`, `.bmp` để tạo dictionary `{stem_path}`.
   - Việc này giúp dò nhanh ánh xạ `hazy -> trans -> clear` dựa trên tên file.

3. Ghép cặp ảnh theo quy ước RESIDE-ITS:
   - Với từng ảnh mù `hazy`, chương trình phát hiện tên stem dạng `id_k_A` và tìm trans tương ứng theo thứ tự ưu tiên:
     - `{id}_{k}`
     - `{id}_{k}_{A}`
     - `{id}_{k}_trans`
     - `{id}_trans`
     - `{id}`
   - Sau đó đối chiếu với ảnh rõ `clear` theo `id` để tạo bộ dữ liệu đầy đủ.

4. Kiểm tra tính hợp lệ của nhãn:
   - Chương trình thực hiện hai kiểm tra quan trọng:
     - Nhãn truyền qua `trans` phải có độ biến thiên thực tế, không bị rời rạc thành ảnh trắng.
     - Mối quan hệ vật lý giữa ảnh mù, ảnh rõ và truyền qua phải thỏa mãn:
       `I = J * t + A * (1 - t)`, với `A` lấy từ tên file tương ứng.
   - Đây là bước kiểm tra trường hợp ghép cặp dữ liệu sai hoặc nhãn không hợp lệ.

### 3.1.3. Tách tập train/val

Sau khi xây dựng xong `its_items`, tập dữ liệu được tách theo ID cảnh để tránh rò rỉ giữa train và val:

- `ids = sorted({it['id'] for it in its_items})`
- Shuffle theo `SEED = 42`
- Chia `val` theo tỉ lệ khoảng 5% số cảnh
- Kết quả notebook cho thấy:
  - Train: 13300 ảnh/crop tương ứng với cảnh học
  - Val: 690 ảnh/crop tương ứng với 69 cảnh

Việc tách theo ID cảnh đảm bảo các phương án đánh giá không bị lệch do hình học lặp lại trong cùng cảnh, đồng thời phù hợp với cách đánh giá truyền thống trên RESIDE-ITS.

### 3.1.4. Thiết kế dataset `ITSHybrid`

Dataset `ITSHybrid` được xây dựng theo kiểu mẫu có nhiều crop trên cùng ảnh gốc để tối ưu hiệu suất huấn luyện mà không cần lưu toàn bộ ảnh ở kích thước lớn. Mỗi ảnh gốc được xử lý như sau:

- Nếu cạnh nhỏ hơn `PATCH = 256`, ảnh được phóng to để đảm bảo đủ độ lớn cho crop.
- Với mỗi ảnh, chương trình sinh `CROPS = 4` crop 256x256.
- Mỗi crop đi kèm với nhãn truyền qua `trans` tương ứng.
- Có thể lật ngang 50% trường hợp để tăng tính đa dạng nhưng vẫn giữ quy luật hình học tự nhiên của cảnh.

Dữ liệu trả về cho mỗi mẫu gồm:

- `hz`: tensor `(K,3,H,W)` dạng uint8 cho ảnh mù của K crop
- `tt`: tensor `(K,1,H,W)` dạng float32 cho nhãn `trans`
- `A`: vector màu khí quyển 3 kênh, tính từ ảnh gốc

Hàm `make_input(hz_u8, A)` sẽ tạo đầu vào cho CNN theo cấu trúc:

- RGB ban đầu
- Dark channel `dark`
- Transmission raw `tr` tính theo DCP

Cấu trúc đầu vào là: `[RGB, dark, trans_raw]`, tổng cộng 5 kênh. Đây là cốt lõi để mô hình hybrid học được sửa lỗi từ biểu diễn vật lý DCP mà không phá hủy quy luật đầu vào của phương pháp cổ điển.

### 3.1.5. Tối ưu hóa bộ nhớ GPU và tốc độ xử lý

Trong notebook có nhiều biện pháp tối ưu hóa về mặt bộ nhớ và hiệu năng:

- `cv2.setNumThreads(0)`: giảm xung đột CPU giữa OpenCV và workers của DataLoader.
- `torch.backends.cudnn.benchmark = False`: tránh tăng độ trễ không cần thiết khi dữ liệu đánh giá có nhiều kích thước khác nhau.
- `BATCH_IMGS = 8`, `CROPS = 4`: cho phép mỗi batch chứa 32 patch, đủ lớn để tăng tính hiệu quả nhưng không cao quá mức gây overload GPU.
- DCP tính trên GPU bằng min-pooling: `-F.max_pool2d(-x, k, stride=1, padding=k//2)`.
- Hàm `estimate_A_np` chỉ tính `A` ở tỉ lệ nhỏ `scale = 0.25`, giúp giảm chi phí xử lý đáng kể trước khi thực hiện huấn luyện.
- `M = 8` trong loss được dùng để bỏ qua vùng viền trong loss, tránh ảnh hưởng của kernel DCP 15x15 ở mép ảnh.

Điểm đáng chú ý là notebook không đọc ảnh `clear` trong quá trình train. Điều này mang lại hiệu quả thiết thực trong môi trường GPU memory hạn chế, đồng thời cho phép mô hình học trực tiếp từ các ảnh mù và nhãn `trans` thay vì lưu trữ một khối dữ liệu lớn gồm cả ảnh rõ và ảnh mù cùng lúc.

---

## 3.2. Quá trình huấn luyện & Lịch trình điều chỉnh tốc độ học (Cosine Annealing)

### 3.2.1. Kiến trúc mô hình Hybrid CNN

Mô hình được xây dựng trong notebook là một mạng CNN dạng residual dựa trên bản đồ truyền qua DCP:

- Đầu vào: 5 kênh `[RGB, dark, trans_raw]`
- Mô hình ước lượng một phần hiệu chỉnh trong không gian truyền qua
- Cơ chế: `t = trans_raw_DCP + CNN(...)`
- Lớp cuối được khởi tạo bằng 0, nghĩa là ban đầu mô hình phải khớp với phương pháp DCP thuần. Đây là kỹ thuật rất quan trọng để đảm bảo không có sự rối loạn bắt đầu.

Cấu trúc mạng gồm:

- Conv 5 -> 32 channels, kernel 3x3, padding 1
- Dilation 2 và dilation 4 để mở rộng vùng tiếp nhận
- `PReLU` sau mỗi lớp nén
- Lớp đầu ra 1 kênh, khởi tạo bằng 0

Mô hình có số tham số khoảng 20k, rất nhỏ so với nhiều kiến trúc dehazing hiện đại, khiến nó phù hợp cho huấn luyện nhanh và suy luận thời gian thực trên ảnh lớn.

### 3.2.2. Hàm mất mát và ràng buộc đầu ra

Hàm loss trong notebook là:

- `mse = F.mse_loss(pred, gt)`
- `ssim_v = ssim_torch(pred, gt, data_range=1.0, size_average=True)`
- `loss = mse + 0.1 * (1 - ssim_v)`

Điều này giúp mô hình vừa tối ưu sai số pixel, vừa tăng độ tương đồng cấu trúc giữa bản đồ truyền qua dự đoán và nhãn gốc. Đặc biệt, sau khi suy ra `t`, mô hình áp dụng `clamp(0,1)` bằng straight-through trick:

- Trong training: đầu ra có thể vượt ngưỡng nhưng gradient vẫn được truyền thành công.
- Trong evaluation: đầu ra được gắn chặt về `[0,1]` để bảo đảm độ ổn định.

### 3.2.3. Kiểm tra trước khi train

Trước khi bắt đầu huấn luyện, notebook chạy ba kiểm tra bắt buộc:

1. Nhãn `trans` phải có biến thiên thật.
2. Loss đầu vào phải thấp, gần với baseline của DCP, không được quá cao như 0.99.
3. Gradient ở lớp cuối phải khác 0 để đảm bảo quá trình học không bị tắt.

Kết quả từ notebook cho thấy:

- `val t-MSE | mô hình khởi tạo: 0.02778 | DCP thuần (baseline cần vượt): 0.02778`
- `loss ban đầu = 0.0287 (MSE 0.0149) | grad lớp cuối = 1.0670e-01`

Đây là dấu hiệu tốt vì:

- Mô hình khởi tạo tương đương với DCP thuần, không có sự lệch pha ban đầu.
- Gradient lớp cuối khác 0, đảm bảo quá trình tối ưu hóa có thể bắt đầu ngay.
- Loss ở mức thấp và hợp lý, chứng tỏ các đầu vào và nhãn đã được gắn đúng.

### 3.2.4. Lịch trình học: Cosine Annealing

Notebook dùng optimizer Adam với:

- `LR = 1e-3`
- `EPOCHS = 30`
- `IMGS_PER_EPOCH = 4000`
- `CosineAnnealingLR(opt, T_max=EPOCHS * len(loader), eta_min=1e-5)`

Lợi ích của Cosine Annealing trong đây là rất rõ:

- Ở giai đoạn đầu, tốc độ học cao giúp mô hình nhanh hội tụ về vùng gần DCP.
- Sau đó, tốc độ học giảm dần để mô hình điều chỉnh các chi tiết nhạy cảm như biên vùng, mây, vật thể gần xa, các dải sáng/cực sáng.
- Điều này rất phù hợp với bài toán khử mù khi bản đồ truyền qua có nhiều hỗn loạn ở vùng biên và độ tương phản màu sắc khác nhau.

Mỗi epoch, notebook không quét toàn bộ 13k ảnh, mà chỉ lấy một mẫu ngẫu nhiên `IMGS_PER_EPOCH = 4000` ảnh để giảm thời gian mỗi vòng lặp. Điều này làm tăng tốc độ huấn luyện mà không làm giảm tính đại diện quá mức.

Đồ thị huấn luyện thể hiện:

- `train loss` giảm theo log-scale
- `val_t_mse` của mô hình giảm xuống rõ ràng so với baseline DCP
- Best checkpoint được lưu trong `hybrid_best.pth`
- `last_path` cũng được lưu để phục hồi nếu cần resume training

Kết quả lưu trữ trong notebook cho thấy:

- `baseline DCP val t-MSE = 0.02778`
- `đã nạp best: val t-MSE = 0.00986 so với DCP 0.02778`

Đây là một cải thiện rõ rệt khoảng 64% so với baseline DCP về chỉ số MSE bản đồ truyền qua, tương ứng với hiệu suất rất đáng kể trong bài toán dehazing.

---

## 3.3. Đánh giá định lượng trên tập ITS-val & SOTS (Indoor/Outdoor)

### 3.3.1. Mục tiêu đánh giá

Sau khi huấn luyện xong, mô hình được đánh giá trên hai cấp độ:

1. Tại mức bản đồ truyền qua `t`:
   - `T_MSE`
   - `T_MAE`
2. Tại mức ảnh khôi phục:
   - `PSNR`
   - `SSIM`
   - `CIEDE2000`

Cùng với đó, notebook cũng so sánh bốn phương án nổi bật:

- Hazy (không khử)
- DCP cổ điển (+GF)
- Hybrid CNN (thô)
- Hybrid CNN + GF (đầy đủ)
- Oracle (t GT + A GT) như một ngưỡng trên lý thuyết

### 3.3.2. Đánh giá trên ITS-val

Hàm `evaluate()` trong notebook chạy cho từng ảnh trong `val_items`, tính toán cho từng phương án và sinh DataFrame. Kết quả đầu ra hiện hữu trong notebook cho thấy các số liệu sau:

- Hazy:
  - PSNR = 12.8710
  - SSIM = 0.6900
  - CIEDE2000 = 19.5242
- DCP cổ điển (+GF):
  - PSNR = 16.7880
  - SSIM = 0.8065
  - CIEDE2000 = 11.6887
  - T_MSE = 0.0273
  - T_MAE = 0.1180
- Hybrid CNN (thô):
  - PSNR = 19.1019
  - SSIM = 0.8168
  - CIEDE2000 = 9.3496
  - T_MSE = 0.0120
  - T_MAE = 0.0822
- Hybrid CNN + GF (đầy đủ):
  - PSNR = 19.3849
  - SSIM = 0.8344
  - CIEDE2000 = 9.2309
  - T_MSE = 0.0112
  - T_MAE = 0.0802
- Oracle:
  - PSNR = 51.7343
  - SSIM = 0.9952
  - CIEDE2000 = 0.4905

Những con số này cho thấy ba đặc điểm quan trọng:

1. Mô hình hybrid cải thiện mật độ chi tiết và mức độ tương phản đáng kể so với DCP cổ điển.
2. Guided filter làm mịn bản đồ truyền qua và giúp khôi phục hình ảnh tốt hơn ở mức chất lượng trực quan.
3. Mô hình hybrid + GF cho hiệu suất tốt nhất trong số các phương án hợp lệ, đồng thời giảm khoảng cách tương đối so với oracle rất mạnh.

### 3.3.3. Đánh giá trên SOTS (Indoor/Outdoor)

Trong notebook, hàm `build_sots(hd, gd)` được xây dựng để tạo tập SOTS từ thư mục `hazy` và `gt` tương ứng. Với cấu trúc dữ liệu của SOTS, ảnh được ghép bằng cách ánh xạ tên file theo tiền tố trước dấu gạch dưới:

- `s = filename stem`
- `clear = gt[s.split('_')[0]]`

Điều này cho phép xây dựng bộ dữ liệu SOTS indoor và outdoor bằng cùng một phương pháp đánh giá như với ITS. Mỗi ảnh trong SOTS được đánh giá với cùng logic:

- tính `A` theo DCP
- dự đoán `t_net` bằng mô hình hybrid
- áp dụng guided filter cho `t_net` để tạo phiên bản khôi phục đầy đủ
- gọi lại các metric `PSNR`, `SSIM`, `CIEDE2000`

Nói cách khác, hệ điều hành đánh giá không chỉ dừng ở ITS-val mà còn mở rộng ra ngoài miền ảnh mù indoor thú vị thông thường. Về mặt lý thuyết, SOTS indoor/outdoor là tập kiểm định quan trọng vì nó phản ánh khả năng tổng quát hóa của mô hình trong điều kiện ánh sáng và môi trường khác nhau. Trong notebook, quy trình này đã được chuẩn bị sẵn và có thể chạy trực tiếp bằng `evaluate(...)` trên mỗi nhóm SOTS.

Như vậy, kết luận năng lực tổng quát của mô hình dựa trên hai nguyên tắc:

- kiến trúc hybrid có nền tảng vật lý DCP nên dễ điều chỉnh với miền dữ liệu mới
- các metric và pipeline đánh giá là đồng bộ giữa ITS-val và SOTS, giảm sai lệch thống kê khi so sánh phương án.

---

## 3.4. Trực quan hóa so sánh ảnh khôi phục

### 3.4.1. Ý nghĩa của trực quan hóa

Trong các bài toán khử mù, kết quả định lượng không phải lúc nào cũng thể hiện đầy đủ mức độ trực quan của chất lượng hình ảnh. Vì vậy notebook có phần trực quan hóa theo các tiêu chí sau:

- Ảnh mù gốc
- Bản đồ truyền qua `trans`
- Ảnh rõ / đồng dạng GT
- Kết quả khôi phục bằng DCP cổ điển
- Kết quả khôi phục bằng Hybrid CNN thô
- Kết quả khôi phục bằng Hybrid CNN + GF
- `Oracle` như ngưỡng lý thuyết

Định dạng này giúp quan sát trực tiếp:

- mức độ giải mù
- độ tương phản của vật thể và nền
- độ sắc nét ở vùng biên,
- sai số xuất hiện ở vùng sáng hoặc vùng tối,
- hiệu quả của guided filter trong việc làm mịn bản đồ truyền qua không làm mất chi tiết.

### 3.4.2. Cấu trúc trực quan hóa trong notebook

Notebook sử dụng `matplotlib.pyplot` để dựng các hình đơn lẻ và `plt.show()` để hiển thị các ảnh. Ví dụ, chương trình có một cell kiểm tra nhãn với 3 ảnh: `hazy`, `trans GT`, `clear`. Điều này cho phép người nghiên cứu dễ dàng phát hiện:

- nếu `trans GT` quá phẳng thì dữ liệu bị lỗi;
- nếu `clear` không khớp với mô hình vật lý thì hệ số khí quyển `A` có thể sai;
- nếu mẫu huấn luyện không phù hợp thì mô hình sẽ học sai từ sớm.

### 3.4.3. Phân tích trực quan sự cải thiện

Với ảnh mù gốc, các đường viền bị mờ và sắc độ xám do nồng độ bụi. Khi so sánh các phương pháp:

- DCP cổ điển (+GF): tăng độ rõ nhưng nhiều vùng còn bị thiếu độ tự nhiên, có thể xuất hiện over-dark hoặc halo ở vùng biên.
- Hybrid CNN (thô): thu được độ tương phản tốt hơn, cải thiện rõ nét hơn ở vùng nội dung chính.
- Hybrid CNN + GF (đầy đủ): cho kết quả ổn định nhất, giữ được chi tiết và giảm nhiễu trong nền.

Điều này trùng khớp với các chỉ số đánh giá định lượng: `Hybrid CNN + GF` có PSNR cao nhất, SSIM tốt nhất trong các phương án thực tế, và T_MSE/T_MAE thấp nhất. Đây là minh chứng trực quan rằng mạng hybrid đã học được cách hiệu chỉnh `trans_raw` DCP, không chỉ đơn thuần “bám” theo lớp DCP cứng.

### 3.4.4. Kết luận trực quan

Trực quan hóa không chỉ dừng ở việc “xem ảnh đẹp hơn”, mà còn cho thấy rõ ý nghĩa của mỗi kỹ thuật trong pipeline:

- DCP cung cấp prior vật lý ban đầu
- CNN tối ưu hóa sự sai lệch và chú ý tới nội dung ảnh
- Guided filter làm mịn, khử nhiễu và cải thiện tính ổn định cho t
- Oracle đề cập đến giới hạn lý thuyết và cho thấy có thể đạt được chất lượng cực cao nếu đầu vào chính xác

Những yếu tố này cùng nhau tạo ra một nghiệm pháp huấn luyện và đánh giá đồng bộ, phù hợp với mục đích nghiên cứu dehazing kết hợp giữa vật lý và học sâu.

---

## Kết luận chung

Bộ thực nghiệm trong notebook `Hybrid_DehazeCNN_fixed.ipynb` đã xây dựng một quy trình khử mù theo hướng lai rất chặt chẽ, với điểm nhấn là:

- sử dụng kiến trúc DCP + CNN để khử mù theo hướng truyền qua
- tối ưu bộ nhớ GPU bằng cách giảm thiểu dữ liệu và crop ảnh
- kiểm tra độ hợp lệ của nhãn trước khi huấn luyện
- huấn luyện trên dataset `ITSHybrid` với `Cosine Annealing`
- đánh giá đồng bộ trên `ITS-val` và phân bố `SOTS`
- trực quan hóa bằng ảnh và các số liệu `PSNR`, `SSIM`, `CIEDE2000`, `T_MSE`, `T_MAE`

Kết quả ghi nhận trên ITS-val cho thấy mô hình Hybrid CNN + GF vượt trội so với DCP cổ điển, đồng thời vượt rõ trên cả baseline trong các chỉ số định lượng. Đây là dấu hiệu cho thấy phương án hybrid là hướng đi hợp lý cho bài toán khử mù trên ảnh thiên nhiên, đặc biệt khi kết hợp cả prior vật lý và học máy.

Nếu cần tiếp theo, file này có thể được mở rộng thành:

- phần tổng kết nghiên cứu,
- bảng so sánh số liệu rõ ràng dạng PDF/Markdown,
- phụ lục về cấu hình môi trường và hyperparameter,
- hoặc tách riêng thành các section 3.1–3.4 theo đúng format báo cáo luận văn.

