# Code Review: Health Profile Module

Dựa trên tiêu chuẩn kiến trúc đã định nghĩa trong `README.md`, dưới đây là các vấn đề và đề xuất cải thiện cho module `Health Profile`.

## 1. Vi phạm quy tắc Clean Architecture

### ❌ Business Logic nằm trong UI (Presentation Layer)

- **Vấn đề**: File `personal_health_profile_screen.dart` chứa hàm `_calculateMetrics` thực hiện tính toán BMI, BMR, Body Fat, Muscle Mass.
- **Tại sao sai**: Logic tính toán chỉ số sức khỏe là **Business Logic**, không phải UI Logic. Việc đặt nó trong Widget làm giảm khả năng test và tái sử dụng.
- **Giải pháp**:
  - Chuyển các hàm tính toán này vào `HealthProfile` entity (trong `domain/entities/`) hoặc tạo một `HealthCalculator` service/helper trong `domain/`.
  - UI chỉ nên gọi hàm này để hiển thị kết quả.

### ❌ Model không kế thừa Entity

- **Vấn đề**: Class `HealthProfileModel` (trong `data/models/`) khai báo lại toàn bộ các trường dữ liệu thay vì kế thừa từ `HealthProfile` (Entity).
- **Quy định README**: "Tạo `ProductModel` trong `data/models/` (extends Entity, thêm fromJson/toJson)".
- **Giải pháp**: Sửa `HealthProfileModel` để `extends HealthProfile`. Điều này giúp đồng bộ dữ liệu và giảm lặp code.

### ❌ Thiếu xử lý lỗi tại Repository

- **Vấn đề**: `HealthProfileRepositoryImpl` gọi trực tiếp `remoteDataSource` mà không có block `try-catch` để bắt lỗi từ API.
- **Quy định README**: "Transform API errors to domain-specific errors" và "Sử dụng... cơ chế try-catch tại Repository".
- **Hậu quả**: Các lỗi từ thư viện `Dio` hoặc `Network` (như `DioException`) đang bị ném thẳng ra Bloc/UI mà không được chuẩn hóa thành Domain Exception/Failure.
- **Giải pháp**: Bao bọc các gọi API trong `try-catch` tại Repository, bắt `DioException` và ném ra custom exception của Domain (ví dụ `ServerException`, `ConnectionException`).

## 2. Các vấn đề khác

### ⚠️ Hardcoded Strings

- **Vấn đề**: Các chuỗi văn bản hiển thị (Label, Title, Error Message) đang được hardcode trực tiếp trong file UI (ví dụ: 'Height (cm)', 'Weight (kg)').
- **Đề xuất**: Nên tách ra file Constants hoặc Localization để dễ quản lý và đa ngôn ngữ hóa.

### ⚠️ Duplicate Code trong Bloc

- **Vấn đề**: Các hàm `_on...` trong `HealthProfileBloc` lặp lại cấu trúc `try-catch` và `emit(HealthProfileError(...))` rất nhiều lần.
- **Đề xuất**: Có thể refactor để giảm boilerplate code hoặc sử dụng một hàm helper để xử lý gọi UseCase.

## 3. Kế hoạch sửa lỗi (Action Plan)

1.  **Refactor Entity & Model**:
    - Cập nhật `HealthProfileModel` extends `HealthProfile`.
2.  **Move Logic**:
    - Di chuyển logic tính toán từ `PersonalHealthProfileScreen` sang `HealthProfile` entity.
3.  **Update Repository**:
    - Thêm `try-catch` và mapping lỗi trong `HealthProfileRepositoryImpl`.
