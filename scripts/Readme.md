# Emotion Classification from EEG

Dự án nghiên cứu phân loại cảm xúc từ tín hiệu EEG sử dụng tập dữ liệu
**DENS (OpenNeuro `ds003751`, snapshot `v1.0.2`)**.

Mục tiêu của giai đoạn hiện tại là xây dựng một pipeline tiền xử lý có thể tái
lập, kiểm soát chất lượng rõ ràng và không gây data leakage trước khi so sánh
các mô hình machine learning truyền thống với EEGNet.

> Trạng thái hiện tại: dữ liệu đã hoàn tất preprocessing và curation P1–P10.
> Dataset chưa được đóng gói thành tensor/DataLoader và chưa tạo split chính
> thức cho quá trình huấn luyện.

## Phạm vi dữ liệu

Dự án khóa dữ liệu theo snapshot **DENS v1.0.2**. Dữ liệu từ v1.0.6 không được
sử dụng trong pipeline hoặc trong các báo cáo modeling chính.

Quá trình kiểm tra quan hệ giữa file EEGLAB `.set` và `.fdt` cho thấy:

- 38 subject có cặp file EEG được kiểm tra.
- 8 subject có kích thước `.set/.fdt` nhất quán.
- 30 subject có `.fdt` thiếu hoặc truncated và bị loại khỏi cohort.
- Không cắt, zero-padding hoặc tạo dữ liệu giả để phục hồi các file lỗi.

Danh sách 8 subject có raw EEG hợp lệ:

```text
sub-mit003
sub-mit004
sub-mit061
sub-mit076
sub-mit117
sub-mit121
sub-mit123
sub-mitb2017007
```

Sau QC và curation:

- 7 subject còn ít nhất một primary trial.
- 6 subject còn ít nhất hai primary trial.
- 60 primary trial.
- 1,213 primary window `pass`.
- `sub-mit117` chỉ còn 1 primary trial.
- `sub-mitb2017007` không còn primary trial và không tham gia tập train chính.

## Nguyên tắc thiết kế pipeline

Pipeline tuân theo các nguyên tắc sau:

1. Không chỉnh sửa hoặc xóa raw EEG.
2. Mọi quyết định QC được lưu trong CSV/JSON để có thể audit.
3. Bad channel và ICA được fit riêng cho từng subject.
4. Không sao chép bad-channel list hoặc IC index từ subject này sang subject
   khác.
5. QC được thực hiện trong đơn vị vật lý trước normalization.
6. Window overlap không được phép đi qua các split khác nhau.
7. Split subject-dependent phải theo trial; cross-subject phải theo subject.
8. Normalization và subject-relative threshold chỉ được fit trên training fold.

Các rule dùng chung được lưu tại
[`configs/dens_rules.json`](configs/dens_rules.json).

## Tổng quan pipeline

| Giai đoạn | Mục tiêu | Script chính |
|---|---|---|
| Materialize | Tải file annex đã chọn và kiểm tra MD5 | `download_dens_raw.py` |
| Integrity | Kiểm tra `.set/.fdt` có đủ dữ liệu | `audit_dens_file_integrity.py` |
| Manifest | Ghép EEG, event và self-report theo trial | `build_dens_manifest.py` |
| P1–P2 | Audit cấu trúc và chất lượng raw EEG | `audit_dens_raw_quality.py` |
| P3 | So sánh cấu hình filter | `compare_dens_filters.py` |
| P4 | Phát hiện bad channel riêng từng subject | `detect_dens_bad_channels.py` |
| P5 | Kiểm chứng nội suy và average reference | `validate_dens_interpolation_reference.py` |
| P6 | Fit ICA và đề xuất IC artifact | `propose_dens_ica_components.py` |
| P6b | Ablation trước khi quyết định loại IC | `ablate_dens_ica_subject.py` |
| P7–P8 | Làm sạch từng trial và QC sau xử lý | `process_and_qc_dens_trials.py` |
| P9 | Chia window và gán `pass/review/reject` | `qc_dens_windows.py` |
| P10 | Loại trial, gán nhãn và tạo primary dataset | `curate_dens_subject.py` |
| Cohort | Chạy pipeline đã duyệt cho nhiều subject | `run_dens_cohort.py` |
| Summary | Tổng hợp QC và phân bố nhãn toàn cohort | `summarize_dens_cohort.py` |
| Modeling audit | Kiểm tra dữ liệu sẵn sàng cho modeling | `audit_dens_modeling_readiness.py` |

## Chi tiết quá trình phân tích và xử lý

### 1. Materialize dữ liệu và kiểm tra integrity của EEGLAB

[`scripts/download_dens_raw.py`](scripts/download_dens_raw.py) materialize các
file `.set/.fdt` được chọn từ OpenNeuro và xác minh kích thước cùng MD5 dựa trên
git-annex pointer.

Script này không quyết định version dataset. Repository dữ liệu phải được
checkout đúng snapshot v1.0.2 trước khi materialize; không dùng URL S3 mặc định
để suy luận version. Sau khi tải, mọi file vẫn phải qua integrity audit.

[`scripts/audit_dens_file_integrity.py`](scripts/audit_dens_file_integrity.py)
đọc metadata trong `.set` và tính kích thước `.fdt` kỳ vọng:

```text
expected_bytes = nbchan × pnts × trials × 4
```

Hệ số `4` tương ứng với mỗi mẫu `float32`. Subject chỉ được đưa vào pipeline
khi kích thước khai báo và kích thước `.fdt` khớp tuyệt đối.

Kết quả được ghi vào:

```text
reports/dens_eeglab_integrity.csv
```

### 2. Khảo sát cấu trúc subject và xây dựng manifest

[`scripts/audit_dens_subject.py`](scripts/audit_dens_subject.py) dùng để khảo
sát nhanh sampling rate, channel, event và self-report của một subject.

[`scripts/build_dens_manifest.py`](scripts/build_dens_manifest.py) ghép:

- EEG recording.
- Event onset và duration.
- Stimulus/trial.
- Self-report valence và arousal.
- Trạng thái match và validation issue.

Các event onset/duration của DENS được chuyển từ sample index sang giây bằng
sampling rate của recording. Manifest không tạo nhãn binary ở bước này.

Đầu ra chính:

```text
data/manifest/dens_trials.csv
data/manifest/dens_subjects.csv
reports/manifest_summary.json
```

### 3. P1–P2: audit raw EEG

[`scripts/audit_dens_raw_quality.py`](scripts/audit_dens_raw_quality.py) kiểm
tra raw EEG theo chunk để tránh phải preload toàn bộ recording.

Các kiểm tra bao gồm:

- Sampling rate, số channel và thời lượng.
- NaN/Inf.
- Biên độ và peak-to-peak.
- Flat channel.
- Variance outlier.
- Global correlation.
- Spatial-neighbor correlation.
- Line-noise và power spectral density.

Bước này chỉ phân tích và tạo báo cáo, không ghi đè raw EEG.

### 4. P3: lựa chọn filter

[`scripts/compare_dens_filters.py`](scripts/compare_dens_filters.py) so sánh
các cấu hình filter trên cùng dữ liệu và xuất PSD/metric để duyệt trước khi
chốt cấu hình.

Cấu hình được chọn cho pipeline:

- Notch filter: 50 Hz.
- Band-pass filter: 0.5–45 Hz.
- Padding trước/sau trial: 15 giây để hạn chế edge artifact.
- Zero-phase filtering.

P3 không lưu EEG đã filter; nó chỉ tạo bảng so sánh và biểu đồ.

### 5. P4: phát hiện bad channel

[`scripts/detect_dens_bad_channels.py`](scripts/detect_dens_bad_channels.py)
đánh giá từng channel trên từng trial bằng bốn nhóm metric độc lập:

- Variance outlier.
- Peak-to-peak outlier.
- Global correlation.
- Spatial-neighbor correlation.

Rule tổng hợp:

- `bad`: bị flag ở ít nhất 50% valid trial và bởi ít nhất hai metric độc lập.
- `review`: bị flag ở ít nhất 25% valid trial hoặc bởi ít nhất hai metric.
- `pass`: các trường hợp còn lại.

Quyết định của mỗi subject được lưu riêng trong
`p4_bad_channel_decisions.csv`.

### 6. P5: nội suy và average reference

[`scripts/validate_dens_interpolation_reference.py`](scripts/validate_dens_interpolation_reference.py)
kiểm chứng việc nội suy bad channel trước khi áp dụng chính thức.

Phép kiểm tra leave-one-good-channel-out giả lập mất một channel tốt, nội suy
lại channel đó và so sánh với tín hiệu gốc. Sau nội suy, EEG được chuyển sang
average reference; bad channel không đóng góp trực tiếp vào reference trước khi
được phục hồi.

Bad-channel list vẫn là kết quả riêng của từng subject.

### 7. P6–P6b: ICA và ablation

[`scripts/propose_dens_ica_components.py`](scripts/propose_dens_ica_components.py)
fit ICA riêng cho từng subject trên dữ liệu:

- High-pass 1 Hz khi fit ICA.
- Low-pass 45 Hz.
- Notch 50 Hz.
- Các bad channel đã được nội suy.
- Average reference.
- PCA giữ 99% phương sai.
- Random seed cố định để tái lập.

IC chỉ được đưa vào shortlist khi có bằng chứng artifact như tương quan ECG,
EMG hoặc đặc trưng high-frequency/kurtosis bất thường. Kurtosis đơn lẻ không
được dùng làm lý do loại IC.

[`scripts/ablate_dens_ica_subject.py`](scripts/ablate_dens_ica_subject.py) so
sánh ba chiến lược:

- Không loại IC.
- Chỉ loại IC có bằng chứng sinh lý.
- Loại toàn bộ IC candidate.

Quyết định tự động bảo thủ chỉ chấp nhận loại IC khi:

- Giảm auxiliary leakage ít nhất 10%.
- Relative RMS change không vượt 20%.
- Retention của delta/theta/alpha/beta/gamma nằm trong khoảng 0.75–1.25.

[`scripts/compare_dens_ica_removal.py`](scripts/compare_dens_ica_removal.py)
là công cụ chẩn đoán bổ sung để trực quan hóa ablation ICA khi cần review thủ
công.

### 8. P7–P8: tạo cleaned trial và QC sau xử lý

[`scripts/process_and_qc_dens_trials.py`](scripts/process_and_qc_dens_trials.py)
áp dụng pipeline đã duyệt theo thứ tự:

```text
raw trial
→ notch 50 Hz
→ band-pass 0.5–45 Hz
→ nội suy bad channel của subject
→ average reference
→ loại IC đã được duyệt của subject
→ post-cleaning QC
→ lưu FIF theo trial
```

Mỗi trial được lưu riêng tại:

```text
data/processed/<subject>/trials/<subject>_trial-XX_eeg.fif
```

QC sau xử lý kiểm tra biên độ, tỷ lệ mẫu vượt 200 µV, flat channel, variance
outlier, residual 50 Hz và tương quan ECG còn lại.

### 9. P9: window-level QC

[`scripts/qc_dens_windows.py`](scripts/qc_dens_windows.py) chia mỗi trial thành:

- Window dài 4 giây.
- Overlap 50%, tương đương stride 2 giây.

Mỗi window được gán một trạng thái:

- `pass`: sử dụng trong primary dataset.
- `review`: chỉ giữ cho ablation `pass + review`.
- `reject`: loại khỏi mọi train/validation/test set.

Ngưỡng chính:

| Trạng thái | Điều kiện |
|---|---|
| Reject | NaN/Inf |
| Reject | `max_abs > 1000 µV` |
| Reject | Tỷ lệ mẫu vượt 200 µV lớn hơn 1% |
| Review | `max_abs > 500 µV` |
| Review | Tỷ lệ mẫu vượt 200 µV lớn hơn 0.1% |
| Review | Có hơn 5 variance-outlier channel |

### 10. P10: curation và gán nhãn

[`scripts/curate_dens_subject.py`](scripts/curate_dens_subject.py) không xóa
file EEG. Script chỉ tạo các cột quyết định như `include_primary`,
`exclude_primary` và lý do loại.

Một trial bị loại khỏi primary dataset khi thỏa ít nhất một điều kiện:

- `pass_fraction < 30%`.
- `reject_fraction >= 20%`.
- Có lỗi integrity không thể phục hồi như NaN/Inf, sai số channel hoặc quá ngắn.

Self-report valence/arousal được dùng làm ground truth; tên video không được
dùng làm nhãn.

Nhãn fixed-5:

- `score > 5`: High.
- `score < 5`: Low.
- `score = 5`: Ambiguous và bị loại khỏi binary task tương ứng.

Nhãn bốn quadrant:

- `HVHA`: valence > 5 và arousal > 5.
- `HVLA`: valence > 5 và arousal < 5.
- `LVHA`: valence < 5 và arousal > 5.
- `LVLA`: valence < 5 và arousal < 5.

Subject-relative label chỉ là ablation. Threshold của nó phải được tính từ
training trial bên trong từng fold, không materialize trước trên toàn subject.

## Chạy pipeline

Ví dụ dưới đây dùng PowerShell và virtual environment của project:

```powershell
$PYTHON = ".\.venv\python.exe"
```

### Kiểm tra dữ liệu và tạo manifest

```powershell
& $PYTHON scripts\audit_dens_file_integrity.py `
  --dataset-root data\raw\ds003751

& $PYTHON scripts\build_dens_manifest.py `
  --dataset-root data\raw\ds003751
```

### Chạy từng checkpoint cho một subject

```powershell
$SUBJECT = "sub-mit003"
$DATASET = "data\raw\ds003751"
$MANIFEST = "data\manifest\dens_trials.csv"
$ICA = "reports\qc\$SUBJECT\p6_ica_solution-ica.fif"

& $PYTHON scripts\audit_dens_raw_quality.py `
  --dataset-root $DATASET --manifest $MANIFEST --subject $SUBJECT

& $PYTHON scripts\compare_dens_filters.py `
  --dataset-root $DATASET --manifest $MANIFEST --subject $SUBJECT

& $PYTHON scripts\detect_dens_bad_channels.py `
  --dataset-root $DATASET --manifest $MANIFEST --subject $SUBJECT

& $PYTHON scripts\validate_dens_interpolation_reference.py `
  --dataset-root $DATASET --manifest $MANIFEST --subject $SUBJECT

& $PYTHON scripts\propose_dens_ica_components.py `
  --dataset-root $DATASET --manifest $MANIFEST --subject $SUBJECT

& $PYTHON scripts\ablate_dens_ica_subject.py `
  --dataset-root $DATASET --manifest $MANIFEST `
  --ica $ICA --subject $SUBJECT

& $PYTHON scripts\process_and_qc_dens_trials.py `
  --dataset-root $DATASET --manifest $MANIFEST `
  --ica $ICA --subject $SUBJECT

& $PYTHON scripts\qc_dens_windows.py --subject $SUBJECT
& $PYTHON scripts\curate_dens_subject.py --subject $SUBJECT
```

### Chạy cohort theo cấu hình đã duyệt

`run_dens_cohort.py` chạy từ P4 đến P10, hỗ trợ resume và bỏ qua checkpoint đã
có output. Dùng `--force` khi thực sự muốn chạy lại.

```powershell
& $PYTHON scripts\run_dens_cohort.py `
  --subjects sub-mit004 sub-mit061 sub-mit076 `
             sub-mit117 sub-mit121 sub-mit123 sub-mitb2017007 `
  --available-only
```

Tổng hợp cohort và audit modeling-readiness:

```powershell
& $PYTHON scripts\summarize_dens_cohort.py
& $PYTHON scripts\audit_dens_modeling_readiness.py
```

## Phân bố nhãn primary hiện tại

| Task | Eligible trial | Primary window | Phân bố trial |
|---|---:|---:|---|
| Valence binary | 57 | 1,155 | 24 High / 33 Low |
| Arousal binary | 60 | 1,213 | 44 High / 16 Low |
| Four-quadrant | 57 | 1,155 | HVHA 20 / LVHA 23 / LVLA 10 / HVLA 4 |

Valence binary là task chính được đề xuất vì phân bố ở cấp trial cân bằng hơn.
Arousal binary và four-quadrant được giữ làm thí nghiệm phụ.

## Trạng thái trước khi huấn luyện

Pipeline preprocessing đã hoàn tất, nhưng còn các bước modeling sau:

1. Tạo split train/validation/test chống leakage.
2. Kiểm tra từng fold có đủ lớp.
3. Tạo Dataset/DataLoader từ các cleaned trial FIF.
4. Fit normalization chỉ trên training partition.
5. Trích xuất band-power hoặc Riemannian feature cho baseline truyền thống.
6. Huấn luyện và so sánh subject-dependent, LOSO và personalization.
7. Chạy ablation `pass-only` với `pass + review`.

Không được split ngẫu nhiên theo window vì các window overlap và các window của
cùng một trial không phải những quan sát độc lập.

## Cấu trúc thư mục

```text
configs/
  dens_rules.json             Rule preprocessing, QC, label và split
data/
  raw/                        DENS raw EEG, không đưa lên Git
  manifest/                   Mapping subject/trial/event/self-report
  processed/                  Cleaned FIF theo trial, không đưa lên Git
reports/
  qc/<subject>/               CSV, JSON, ICA và figure theo checkpoint
scripts/                      Toàn bộ pipeline phân tích và xử lý
```

Raw data, processed EEG, báo cáo QC và tài liệu nội bộ được kiểm soát bằng
`.gitignore`; repository chỉ lưu code và cấu hình cần thiết để tái lập pipeline.

## Giới hạn nghiên cứu

Đây là một pilot study với cohort nhỏ. Kết quả có thể dùng để đánh giá pipeline,
so sánh mô hình và nghiên cứu personalization, nhưng chưa đủ để tuyên bố khả
năng tổng quát hóa trên quần thể lớn hoặc khả năng triển khai lâm sàng.
