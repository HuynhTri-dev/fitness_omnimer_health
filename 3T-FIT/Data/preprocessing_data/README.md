# Gym Member Exercise Tracking Dataset

Dữ liệu này chứa thông tin theo dõi quá trình tập luyện của các thành viên phòng gym, bao gồm hồ sơ người dùng, chi tiết bài tập và các chỉ số cường độ đã được tính toán. File này được tạo ra từ `merged_omni_health_dataset.xlsx` sau quá trình làm sạch và feature engineering.

## Thông tin File

- **Tên file:** `own_gym_member_exercise_tracking.xlsx`
- **Đường dẫn:** `3T-FIT/Data/preprocessing_data/`

## Từ điển Dữ liệu (Data Dictionary)

Bảng dưới đây mô tả chi tiết các cột trong dataset:

| Tên Cột                  | Mô tả                        | Kiểu dữ liệu | Ghi chú                                                                                                                                                        |
| :----------------------- | :--------------------------- | :----------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `user_health_profile_id` | ID hồ sơ sức khỏe người dùng | String/Int   | Khóa ngoại liên kết với hồ sơ sức khỏe                                                                                                                         |
| `workout_id`             | ID phiên tập luyện           | String/Int   | Định danh duy nhất cho mỗi buổi tập                                                                                                                            |
| `user_id`                | ID người dùng                | String/Int   | Định danh người dùng                                                                                                                                           |
| `exercise_name`          | Tên bài tập                  | String       | Ví dụ: Bench Press, Treadmill Running                                                                                                                          |
| `duration_min`           | Thời gian thực hiện bài tập  | Float        | Đơn vị: Phút                                                                                                                                                   |
| `avg_hr`                 | Nhịp tim trung bình          | Int          | BPM (Beats Per Minute)                                                                                                                                         |
| `max_hr`                 | Nhịp tim tối đa              | Int          | BPM                                                                                                                                                            |
| `calories`               | Lượng calo tiêu thụ          | Float        | kcal                                                                                                                                                           |
| `suitability_x`          | Chỉ số phù hợp (1)           | String/Float | Dữ liệu từ nguồn merge đầu tiên                                                                                                                                |
| `age`                    | Tuổi                         | Int          |                                                                                                                                                                |
| `height_m`               | Chiều cao                    | Float        | Đơn vị: Mét                                                                                                                                                    |
| `weight_kg`              | Cân nặng                     | Float        | Đơn vị: Kilogram                                                                                                                                               |
| `bmi`                    | Chỉ số khối cơ thể           | Float        | $BMI = Weight / Height^2$                                                                                                                                      |
| `fat_percentage`         | Tỷ lệ mỡ cơ thể              | Float        | %                                                                                                                                                              |
| `resting_heartrate`      | Nhịp tim nghỉ ngơi           | Int          | BPM                                                                                                                                                            |
| `experience_level`       | Mức độ kinh nghiệm           | Int          | Đã map sang số:<br>1: Beginner<br>2: Intermediate<br>3: Advanced<br>4: Expert                                                                                  |
| `workout_frequency`      | Tần suất tập luyện           | Int          | Số ngày tập luyện trong tuần                                                                                                                                   |
| `health_status`          | Tình trạng sức khỏe          | String       | Các vấn đề sức khỏe hoặc "None"                                                                                                                                |
| `workout_type`           | Loại hình tập luyện          | String       | Loại bài tập người dùng muốn thực hiện (Renamed từ `category_type_want_todo`)                                                                                  |
| `location`               | Địa điểm tập luyện           | String       | Gym, Home, Outdoor                                                                                                                                             |
| `suitability_y`          | Chỉ số phù hợp (2)           | String/Float | Dữ liệu từ nguồn merge thứ hai                                                                                                                                 |
| `gender`                 | Giới tính                    | String       | Male / Female                                                                                                                                                  |
| `session_duration`       | Tổng thời lượng phiên tập    | Float        | Đơn vị: **Giờ**. Được chuyển đổi từ `total_duration_min`                                                                                                       |
| `unified_intensity`      | Chỉ số cường độ thống nhất   | Float        | Chỉ số tổng hợp đại diện cho cường độ:<br>- **Strength**: Estimated 1RM (kg)<br>- **Cardio (có distance)**: Tốc độ (km/h)<br>- **Khác**: Intensity Score (1-4) |

## Các bước Xử lý Dữ liệu (Preprocessing Steps)

Dữ liệu đã trải qua các bước xử lý sau bằng script `preprocessing_own_dataset.py`:

1.  **Làm sạch cơ bản**: Loại bỏ các cột rỗng và các dòng dữ liệu chưa hoàn thành (`done=0`).
2.  **Loại bỏ cột thừa**: Xóa các cột không phục vụ cho bài toán gợi ý (checkup_date, mood, recovery, v.v.).
3.  **Sắp xếp**: Đưa các cột ID (`user_health_profile_id`, `workout_id`, `user_id`) lên đầu để dễ quản lý.
4.  **Chuyển đổi đơn vị**: `total_duration_min` (phút) $\to$ `session_duration` (giờ).
5.  **Mã hóa dữ liệu (Encoding)**:
    - `experience_level`: Beginner $\to$ 1, Intermediate $\to$ 2, ...
    - `intensity`: Low $\to$ 1, Medium $\to$ 2, ...
6.  **Feature Engineering (`unified_intensity`)**:
    - Tự động phát hiện loại bài tập để tính toán chỉ số cường độ phù hợp nhất.
    - Sử dụng công thức Epley cho bài tập tạ để tính 1RM.
    - Sử dụng công thức $Distance / Time$ cho bài tập Cardio.
7.  **Đổi tên cột**: Chuẩn hóa tên cột `category_type_want_todo` thành `workout_type`.
