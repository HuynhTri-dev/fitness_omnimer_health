# Phân Tích Chiến Lược: Tối Ưu Hóa Dữ Liệu & Huấn Luyện Mô Hình Gợi Ý Tập Luyện (Fitness Recommendation)

## 1. Tổng Quan Chiến Lược

**Đề xuất:** Thay vì huấn luyện mô hình AI dự đoán trực tiếp các thông số chi tiết (`Sets`, `Reps`, `Weight`, `Rest`) gây nhiễu và khó hội tụ, chúng ta sẽ:

1.  **Gộp (Encode):** Chuyển đổi dữ liệu lịch sử phức tạp thành một chỉ số đại diện duy nhất: **Estimated 1RM (Sức mạnh tối đa ước tính)** hoặc **Intensity Score**.
2.  **Dự đoán (Predict):** Mô hình AI chỉ dự đoán "Khả năng/Sức mạnh của người dùng trong ngày hôm nay".
3.  **Giải nén (Decode):** Sử dụng thuật toán Rule-based kết hợp với Mục tiêu (Goal) và Feedback (SePA) để sinh ra bài tập chi tiết.

**Đánh giá:** Đây là chiến lược **Highly Recommended** vì nó giảm chiều dữ liệu (Dimensionality Reduction), giúp mô hình học nhanh hơn và đảm bảo đầu ra luôn tuân thủ các nguyên tắc sinh học (tránh trường hợp AI đưa ra mức tạ quá nặng với số rep quá cao).

---

## 2. Quy Trình Kỹ Thuật Chi Tiết

### Giai đoạn 1: Feature Engineering (Tiền xử lý)

Dữ liệu hiện tại trong file `merged_omni_health_dataset` cột `sets/reps/weight...` đang ở dạng chuỗi (string). Cần xử lý như sau:

**Bước 1: Parse dữ liệu thô**
Tách chuỗi `"12x40x2 | 8x50x3"` thành danh sách các sets tập luyện.

**Bước 2: Chuẩn hóa về 1RM (One Rep Max)**
Sử dụng công thức Epley hoặc Brzycki để tính sức mạnh tối đa lý thuyết cho bài tập đó trong ngày.

> **Công thức:** $1RM_{est} = Weight \times (1 + \frac{Reps}{30})$

_Ví dụ:_ User tập Bench Press `10 reps x 60kg`.
$$1RM = 60 \times (1 + \frac{10}{30}) = 80kg$$
=> **Input cho Model:** `80` (Thay vì vector `[10, 60]`).

### Giai đoạn 2: Kiến trúc Mô hình (Training Phase)

[cite_start]Áp dụng kiến trúc **Sequence Prediction** tương tự như nghiên cứu **P3FitRec**[cite: 167].

- **Input:**
  - Chuỗi 1RM quá khứ (Sequence of past 1RMs).
  - User Profile (Gender, Experience Level) từ `README.md`.
  - Cảm giác (Mood, Fatigue, Effort) từ dataset.
- **Model Core:** LSTM hoặc GRU (để học xu hướng tăng tiến - Progressive Overload).
- **Output:** **Predicted Daily 1RM** (Khả năng sức mạnh dự đoán cho ngày hôm nay).

### Giai đoạn 3: Inference & Decoding (Sinh bài tập)

Khi Model dự đoán hôm nay User có khả năng đẩy **1RM = 82kg** cho bài Chest Press. Hệ thống sẽ dùng **Mục tiêu (Goal)** trong `README.md` để tính ngược ra Sets/Reps.

**Logic Rule-based:**

| Mục tiêu (Goal) | %1RM (Cường độ) | Số Reps (Target) | Tính toán Sets/Weight                              | Rest Time  |
| :-------------- | :-------------- | :--------------- | :------------------------------------------------- | :--------- |
| **Strength**    | 85% - 95%       | 3 - 5 reps       | Weight = $82 \times 0.90 = 73.8kg$ <br> Sets = 4-5 | 3-5 mins   |
| **Hypertrophy** | 70% - 80%       | 8 - 12 reps      | Weight = $82 \times 0.75 = 61.5kg$ <br> Sets = 3-4 | 1-2 mins   |
| **Endurance**   | 50% - 60%       | 15+ reps         | Weight = $82 \times 0.55 = 45kg$ <br> Sets = 2-3   | 30-60 secs |

---

## 3. Tích hợp Yếu tố "Sức khỏe hàng ngày" (SePA Integration)

[cite_start]Nghiên cứu **SePA** [cite: 906, 1197] nhấn mạnh việc dự đoán trạng thái (Stress, Soreness) _trước_ khi đưa ra lời khuyên. Chúng ta sẽ tích hợp vào bước Decoding:

**Công thức điều chỉnh (Auto-Regulation):**
$$Final\_Weight = Calculated\_Weight \times Readiness\_Factor$$

- **Kịch bản 1:** User khỏe mạnh (`Fatigue` thấp, `Mood` tốt).
  - _Readiness Factor_ = 1.05 (Tăng 5% để thử thách - Progressive Overload).
- **Kịch bản 2:** User mệt mỏi (`Fatigue` cao) hoặc `injury_or_pain_notes` có cảnh báo.
  - _Readiness Factor_ = 0.8 (Giảm 20% tải trọng để Recovery).

---

## 4. Xử lý Bài tập Cardio (Chạy bộ/Đạp xe)

Với các bài như `Treadmill Walking` (ID 9, 45...), không thể dùng 1RM. Hãy sử dụng **Pace (Tốc độ)** hoặc **Heart Rate Zone** làm chỉ số cường độ chung.

- **Input:** Avg Speed (km/h) & Avg Heart Rate.
- **Model Output:** Predicted Sustainable Pace (Tốc độ duy trì dự kiến).
- **Decoding:**
  - _Fat Loss:_ Thời gian dài, Pace thấp (Zone 2 HR).
  - _Cardio/HIIT:_ Thời gian ngắn, Pace cao (Zone 4-5 HR).

---

## 5. Đánh giá Ưu/Nhược điểm

### ✅ Ưu điểm (Pros)

1.  **Tính nhất quán (Consistency):** Loại bỏ hoàn toàn việc AI "bịa" ra các mức tạ vô lý (VD: 100kg x 50 reps).
2.  **Dễ giải thích (Explainability):** Dễ dàng giải thích cho user: "Hôm nay tạ nhẹ hơn vì hệ thống phát hiện bạn đang mệt/stress" (Theo tư tưởng SePA).
3.  **Linh hoạt (Flexibility):** Một model dự đoán sức mạnh có thể dùng chung cho cả mục tiêu Tăng cơ và Tăng sức mạnh, chỉ khác nhau ở bước chia bài.
4.  **Dữ liệu:** Tận dụng tốt cột `avg_hr`, `max_hr` và `fatigue` trong dataset hiện có.

### ⚠️ Nhược điểm & Giải pháp

- **Rủi ro:** Công thức 1RM có thể sai lệch với người mới tập hoặc bài tập bodyweight (như Plank).
  - _Giải pháp:_ Với Bodyweight, dùng RPE (Cảm nhận nỗ lực) làm chỉ số cường độ thay thế.
- **Lạnh (Cold Start):** Người mới chưa có lịch sử để tính 1RM.
  - _Giải pháp:_ Dùng User Profile (Age, Weight, Gender) để gán mức 1RM khởi điểm dựa trên trung bình cộng (theo P3FitRec giai đoạn Cold-start).

---

## 6. Kết luận & Next Steps

Việc tổng hợp về 1 trường cường độ (Estimated 1RM) là **bước đi chiến lược đúng đắn**. Nó biến bài toán từ "Dự đoán chuỗi phức tạp" thành "Dự đoán năng lực người dùng + Thuật toán chia bài".

**Việc cần làm ngay (Action Plan):**

1.  Viết script Python parse cột `sets/reps...` trong file Excel.
2.  Tạo cột mới `estimated_1rm` sử dụng công thức Epley.
3.  Tách tập dữ liệu: Cardio riêng (dùng Speed/HR), Strength riêng (dùng 1RM).
4.  Xây dựng bảng Rule-based mapping (Goal $\to$ Reps range) dựa trên file `README.md`.
