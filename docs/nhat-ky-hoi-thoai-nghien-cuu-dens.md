# Nhật ký hội thoại nghiên cứu phân loại cảm xúc bằng EEG

> Ngày cập nhật: 2026-08-13  
> Phạm vi: Dự án cá nhân phân loại cảm xúc từ EEG, tập dữ liệu DENS/OpenNeuro và pipeline tiền xử lý P1–P10.  
> Lưu ý: Đây là biên bản có cấu trúc được tổng hợp từ cuộc hội thoại dài. Một số lượt hội thoại cũ đã được hệ thống rút gọn nên tài liệu không phải transcript nguyên văn 100%, nhưng giữ lại các quyết định, kết quả và quy tắc quan trọng.

## 1. Mục tiêu dự án

Xây dựng một đồ án nghiên cứu cá nhân có thể đưa vào CV cho vị trí AI Engineer, với mục tiêu:

- Phân loại cảm xúc dựa trên tín hiệu EEG.
- So sánh nhiều chế độ huấn luyện và đánh giá.
- Xây dựng pipeline từ dữ liệu thô, kiểm soát chất lượng, tiền xử lý, gán nhãn đến huấn luyện.
- Tránh data leakage và đánh giá đúng khả năng tổng quát hóa giữa các subject.
- Công bố trung thực giới hạn của một nghiên cứu có cohort nhỏ.

Ba chế độ huấn luyện dự kiến:

1. **Subject-dependent:** train và test trên cùng subject, nhưng split theo trial.
2. **Cross-subject LOSO:** giữ toàn bộ một subject làm test trong mỗi fold.
3. **Transfer learning/personalization:** pretrain trên các subject khác, fine-tune bằng một lượng nhỏ trial của subject đích.

## 2. Quá trình lựa chọn dữ liệu

### DEAP

- DEAP là dataset phổ biến và đáng tin cậy nhưng bản chính thức yêu cầu đăng ký và tuân thủ EULA.
- Với dự án cá nhân không có giảng viên/người chịu trách nhiệm của trường ký, việc lấy dữ liệu chính thức tương đối khó.
- Không nên dùng bản được chia sẻ lại trái phép hoặc không xác minh được nguồn gốc chỉ để tiện tải.

### EEG Dataset for Emotion Classification Using Low-Cost and High-End Equipment

- Có khoảng 20 người tham gia, raw EEG, dữ liệu đã preprocessing và handcrafted features.
- Có dữ liệu từ BrainVision và Emotiv EPOC+.
- Dataset có DOI và giấy phép CC BY 4.0.
- Khoảng 20 subject đủ cho proof-of-concept nhưng còn hạn chế nếu muốn chứng minh khả năng tổng quát hóa mạnh.
- File `EEG-BCI.rar` chứa nhiều nhánh dữ liệu; raw EEG nằm trong các file EEGLAB/EDF liên quan chứ không phải mọi folder đều cần cho bài toán hiện tại.

### DENS trên OpenNeuro

- DENS được chọn làm nguồn dữ liệu chính để xây dựng pipeline.
- Dữ liệu được quản lý theo snapshot/version trên OpenNeuro.
- Pipeline được thử trước trên một subject, sau đó mới tổng hợp thành rule dùng cho cohort.

## 3. Nguyên tắc làm việc đã thống nhất

- Làm việc trong project sạch tại:

  ```text
  T:\emotion-classification-by-EEGnet-
  ```

- Thực hiện theo checkpoint; sau mỗi phần phải báo kết quả để người dùng duyệt trước khi tiếp tục.
- Không thay đổi code/CSV chính trong các phân tích khảo sát riêng, ví dụ khảo sát trial 11.
- Không xóa EEG vật lý chỉ vì QC không đạt; trạng thái sử dụng được điều khiển bằng manifest/curation table.
- Không lấy kết quả bad-channel hoặc ICA của subject 3 áp dụng trực tiếp sang subject khác.
- Mỗi subject phải tự phát hiện bad channel, tự fit ICA và tự đánh giá IC.

## 4. Manifest

Manifest được xây dựng trước preprocessing nhằm lưu quan hệ giữa:

- Subject.
- Trial và event.
- File EEG nguồn.
- Self-report valence/arousal.
- Thời gian bắt đầu/kết thúc.
- Trạng thái hợp lệ của dữ liệu và lý do loại nếu có.

Manifest phải chạy trên toàn cohort, nhưng các quyết định QC chi tiết được fit riêng cho từng subject.

## 5. Pipeline preprocessing P1–P10

Các checkpoint đã được thảo luận và triển khai lần lượt:

- **P1–P2:** kiểm tra cấu trúc dữ liệu, sampling rate, channel, thời lượng, NaN/Inf và lọc tín hiệu cơ bản.
- **P3:** xử lý/lập danh sách channel lỗi và chuẩn bị dữ liệu cho các bước tiếp theo.
- **P4:** nội suy bad channel được xác định cho từng subject; không dùng danh sách cố định của subject 3 cho subject khác.
- **P5:** thiết lập và đánh giá bước ICA.
- **P6:** shortlist IC nghi ngờ dựa trên chỉ số/đặc trưng artifact.
- **P7–P8:** áp dụng quyết định IC đã duyệt và đánh giá lại tín hiệu sau ICA.
- **P9:** chia window, tính QC theo window và phân loại `pass`, `review`, `reject`.
- **P10:** curation cuối, nhãn, dataset chính/ablation và chuẩn bị split cho mô hình.

Tín hiệu dưới 1 Hz đã được loại theo cấu hình lọc đã duyệt. Việc lựa chọn cutoff cần được ghi rõ trong cấu hình và báo cáo để tái lập thí nghiệm.

## 6. Quy tắc QC và curation đã chốt

### Xử lý window

- `pass`: đưa vào dataset chính.
- `review`: không dùng trong lần train chính; giữ để chạy ablation `pass + review`.
- `reject`: loại khỏi tất cả train/validation/test.
- Không xóa file EEG vật lý.
- Sử dụng các trường như `include_primary = false` và ghi rõ lý do loại.

### Quy tắc loại toàn trial

Loại trial khỏi dataset chính nếu thỏa ít nhất một điều kiện:

- `pass_fraction < 30%`.
- `reject_fraction >= 20%`.
- Có NaN/Inf, sai số kênh hoặc sai thời lượng không thể phục hồi.

Các tỷ lệ được tính trên toàn bộ window 4 giây, overlap 50%. Không loại toàn trial chỉ vì một vài window reject cục bộ.

### Nhãn chính fixed-5

Ground truth sử dụng self-report valence/arousal, không sử dụng tên video.

Nhãn binary:

- `score > 5`: High.
- `score < 5`: Low.
- `score = 5`: Ambiguous, không dùng trong binary task.

Nhãn bốn quadrant:

- `HVHA`: valence > 5 và arousal > 5.
- `HVLA`: valence > 5 và arousal < 5.
- `LVHA`: valence < 5 và arousal > 5.
- `LVLA`: valence < 5 và arousal < 5.
- Nếu một score bằng đúng 5, sample không tham gia bài toán bốn lớp.

### Subject-relative

- Fixed-5 là nhãn chính vì có ý nghĩa thống nhất giữa các subject.
- Subject-relative là ablation phụ.
- Có thể dùng median valence/arousal của subject.
- Trong đánh giá mô hình, threshold/median phải tính chỉ từ trial training của từng fold để tránh leakage từ test set.

## 7. Kết quả subject 3

### Trial 11

- Pass: 24.1%.
- Reject: 20.7%.
- Trial 11 đồng thời vi phạm cả hai ngưỡng loại toàn trial.
- Đặt `exclude_primary = true`.
- Vẫn giữ dữ liệu để phân tích và ablation.

Trial 2, 9 và 10 được giữ; chỉ loại các window không đạt.

### Dataset chính subject 3 sau curation

- Còn 10 trial.
- Tổng cộng 290 window ngoài trial 11.
- 209 `pass`: dùng trong dataset chính.
- 74 `review`: chỉ dùng trong ablation.
- 7 `reject`: loại.
- 29 window của trial 11 không dùng trong thí nghiệm chính.

Các window overlap không phải quan sát độc lập. Split phải theo trial, không split ngẫu nhiên theo window.

Phân bố quadrant trong primary dataset từng ghi nhận:

- 85 `HVHA`.
- 124 `LVHA`.
- Không có `HVLA`/`LVLA` vì arousal của 10 trial đều lớn hơn 5.

Vì vậy, với subject 3, bài toán binary valence có thể dùng High/Low; binary arousal không có đủ hai lớp trong primary dataset.

## 8. Áp dụng pipeline sang cohort

Khi áp dụng rule từ subject 3 sang các subject khác:

- Dùng chung quy trình, threshold QC cấp trial/window và schema đầu ra đã chốt.
- Bad channel được phát hiện riêng trên dữ liệu từng subject.
- ICA được fit riêng cho từng subject.
- IC bị loại phải được quyết định dựa trên kết quả của chính subject đó.
- Không sao chép danh sách channel hoặc IC của subject 3.

Kết quả cohort trước khi kiểm tra OpenNeuro v1.0.6:

- 38 subject có cặp `.set/.fdt` trong snapshot được kiểm tra.
- Chỉ 8 subject có kích thước `.set/.fdt` nhất quán:
  - `sub-mit003`
  - `sub-mit004`
  - `sub-mit061`
  - `sub-mit076`
  - `sub-mit117`
  - `sub-mit121`
  - `sub-mit123`
  - `sub-mitb2017007`
- 30 subject có `.fdt` bị thiếu/truncated so với shape khai báo trong `.set`.
- Cohort 8 subject đã curation có 60 primary trial và 1,213 primary pass window.

## 9. Đối chiếu DENS OpenNeuro v1.0.6

Snapshot được lưu riêng tại:

```text
T:\emotion-classification-by-EEGnet-\data\raw\ds003751_v1.0.6
```

Snapshot xác nhận:

- Tag: `1.0.6`.
- Commit: `5bca1418a5be2daa3fb5fdc5fc5c5556d1659b1c`.
- DOI: `10.18112/openneuro.ds003751.v1.0.6`.

Đối chiếu v1.0.2 và v1.0.6:

- 76 file EEG đã tồn tại ở v1.0.2 giữ nguyên ID và kích thước trong v1.0.6.
- 30 subject lỗi cũ không được sửa hoặc thay file.
- v1.0.6 bổ sung `.set/.fdt` cho `sub-mit081` và `sub-mit108`.

### sub-mit081

- `.set` yêu cầu `.fdt` có 258,690,432 byte.
- `.fdt` được khai báo chỉ có 69,204,747 byte.
- Tỷ lệ khoảng 26.75%.
- Kết luận: vẫn truncated, không sử dụng được.

### sub-mit108

- 132 kênh.
- Sampling rate 250 Hz.
- 367,325 time points, khoảng 1,469 giây.
- `.fdt` có 193,947,600 byte, khớp tuyệt đối với shape trong `.set`.
- MD5: `d0d6f455547673f1383630d54fb5112a`.
- Quét toàn bộ 48,486,900 giá trị float32: 0 NaN/Inf.
- Có 114 event và self-report của 11 trial.
- Kết luận: file được khôi phục đầy đủ về mặt cấu trúc và có thể đưa sang P1–P9.

Tổng kết v1.0.6:

- 40 subject có EEG.
- 9 subject có `.set/.fdt` nhất quán.
- 31 subject lỗi, gồm 30 subject lỗi cũ và `sub-mit081`.
- `sub-mit108` là subject hợp lệ mới; chưa được tính là đã curation cho đến khi hoàn thành P1–P9.

Các báo cáo liên quan:

- `reports/dens_v102_v106_eeg_manifest_comparison.csv`
- `reports/dens_eeglab_integrity_v1.0.6_verified.csv`

## 10. Đánh giá việc train trên 9 subject

Train trên 9 subject đủ cho một **pilot study/proof-of-concept** và một đồ án CV tốt, nhưng không đủ để tuyên bố mô hình tổng quát trên quần thể lớn hoặc sẵn sàng triển khai thực tế.

Nguyên tắc đánh giá:

- Không xem các window overlap là các mẫu độc lập.
- Không để window của cùng trial xuất hiện ở cả train và test.
- Subject-dependent phải split theo trial.
- Cross-subject nên dùng LOSO với 9 fold.
- Mọi normalization, feature selection và subject-relative threshold phải fit trên training fold.
- Báo cáo macro-F1, balanced accuracy, confusion matrix và kết quả riêng từng subject.
- Báo cáo mean/median cùng độ phân tán hoặc confidence interval.

Mô hình ưu tiên với cohort nhỏ:

- Band power/Riemannian covariance + LDA hoặc SVM làm baseline.
- EEGNet nhỏ với regularization mạnh.
- ShallowConvNet nếu cần baseline deep learning bổ sung.
- Transfer learning/personalization.
- Ablation `pass-only` so với `pass + review`.

Không nên bắt đầu bằng Transformer hoặc CNN lớn huấn luyện từ đầu trên cohort này.

## 11. Trạng thái hiện tại và bước tiếp theo

Trạng thái:

- Subject 3 đã hoàn tất curation theo rule đã chốt.
- 8 subject cũ đã chạy pipeline cohort ở mức đã báo cáo.
- Snapshot DENS v1.0.6 đã được đối chiếu.
- `sub-mit108` đã xác nhận integrity nhưng chưa chạy P1–P9.
- Dữ liệu v1.0.6 được lưu riêng, không ghi đè dữ liệu cũ.
- Chưa tự động đưa `sub-mit108` vào CSV/pipeline chính.

Bước tiếp theo được đề xuất:

1. Chạy `sub-mit108` qua P1–P2 và báo kết quả.
2. Tiếp tục từng checkpoint P3–P9 sau khi được duyệt.
3. Cập nhật cohort và phân bố nhãn sau curation.
4. Chốt split subject-dependent, LOSO và transfer learning.
5. Xây baseline cổ điển trước EEGNet.
6. Chạy ablation QC và so sánh ba chế độ huấn luyện.

## 12. Git

Commit đã được ghi nhận cho P9 và bộ luật curation:

```text
d56b5ad67528bd06ca5b1f72a863a6b06d6b7722
feat: hoàn thành P9 và chốt bộ luật curation EEG
```

Các thay đổi cohort và báo cáo v1.0.6 sau đó chưa được mặc định commit; cần được xem xét và duyệt trước khi tạo commit tiếp theo.
