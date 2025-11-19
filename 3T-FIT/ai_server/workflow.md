# Workflow Training Model AI Server

Tài liệu này mô tả quy trình huấn luyện (training) cho mô hình Multi-Task Learning (MTL) được sử dụng trong dự án.

## 1. Tổng quan về Model

Model sử dụng kiến trúc **Multi-Task Learning (MTL)** để giải quyết đồng thời hai bài toán:
1.  **Gợi ý bài tập (Classification):** Dự đoán danh sách các bài tập phù hợp (Multi-label classification).
2.  **Dự đoán thông số tập luyện (Regression):** Dự đoán số sets, reps và khối lượng tạ (load_kg) phù hợp.

### Đặc điểm kỹ thuật:
*   **Backbone:** Mạng Neural Network thuần (MLP) với các lớp Linear, ReLU và Dropout.
*   **Loss Function:**
    *   Classification: `BCEWithLogitsLoss` với `pos_weight` (để xử lý mất cân bằng dữ liệu) hoặc `Focal Loss`.
    *   Regression: `SmoothL1Loss` (Huber Loss).
    *   Tổng hợp: `Loss = 1.0 * cls_loss + 0.25 * reg_loss`.
*   **Optimizer:** AdamW.

## 2. Chuẩn bị dữ liệu

Dữ liệu đầu vào là file Excel (`.xlsx`) chứa thông tin người dùng và lịch sử tập luyện.

*   **Đường dẫn mặc định:** `data/merged_omni_health_dataset.xlsx`
*   **Các cột quan trọng:**
    *   Features: `age`, `height_cm`, `weight_kg`, `gender`, `experience_level`, ...
    *   Labels (Bài tập): `exercise_name` (được xử lý thành multi-hot vector).
    *   Targets (Thông số): `sets/reps/weight/timeresteachset` (chuỗi dạng `sets | reps | weight`).

## 3. Cài đặt môi trường

Đảm bảo đã cài đặt các thư viện cần thiết:

```bash
pip install torch pandas numpy scikit-learn joblib openpyxl
```

## 4. Thực hiện Training

Chạy script `train_mtl_multilabel_weighted.py` để bắt đầu huấn luyện.

### Lệnh cơ bản:

```bash
python artifacts_unified/src/train_mtl_multilabel_weighted.py
```

### Các tham số tùy chỉnh (Arguments):

| Tham số | Mặc định | Mô tả |
| :--- | :--- | :--- |
| `--excel_path` | `data/merged_omni_health_dataset.xlsx` | Đường dẫn đến file dữ liệu huấn luyện. |
| `--artifacts` | `artifacts_omni_mlbce` | Thư mục lưu kết quả (model, preprocessor). |
| `--epochs` | `80` | Số vòng lặp huấn luyện. |
| `--batch_size` | `128` | Kích thước batch. |
| `--lr` | `0.001` | Learning rate. |
| `--use_focal` | `False` | Thêm cờ này nếu muốn dùng Focal Loss thay vì BCE thường. |
| `--load_cap_kg` | `200.0` | Giới hạn khối lượng tạ tối đa để chuẩn hóa. |

### Ví dụ chạy với Focal Loss và 100 epochs:

```bash
python artifacts_unified/src/train_mtl_multilabel_weighted.py --epochs 100 --use_focal --artifacts my_new_model
```

## 5. Kết quả đầu ra (Artifacts)

Sau khi training xong, các file sau sẽ được tạo ra trong thư mục `artifacts` (hoặc thư mục bạn chỉ định):

1.  **`best.pt`**: File trọng số của model (PyTorch state dict) tại epoch có độ chính xác (Precision@5) cao nhất trên tập validation.
2.  **`preprocessor.joblib`**: Pipeline xử lý dữ liệu (StandardScaler, OneHotEncoder) đã được fit. Cần dùng file này để xử lý dữ liệu mới khi inference.
3.  **`meta.json`**: File metadata chứa thông tin cấu hình, danh sách các cột bài tập, và các tham số scale cho regression (min/max của sets, reps, load).
4.  **`BEST.txt`**: Ghi lại kết quả tốt nhất đạt được.

## 6. Sử dụng Model (Inference)

Để sử dụng model dự đoán:
1.  Load `preprocessor.joblib` để biến đổi dữ liệu đầu vào của user.
2.  Khởi tạo class `MTLNet` với `in_dim` và `num_exercises` lấy từ `meta.json` hoặc `best.pt`.
3.  Load weights từ `best.pt`.
4.  Đưa dữ liệu qua model để lấy logits (bài tập) và regression output (sets/reps/load).
5.  Giải mã output:
    *   Logits -> Sigmoid -> Top K bài tập có xác suất cao nhất.
    *   Regression output -> Inverse transform (nhân với range + min) để ra số thực tế.
