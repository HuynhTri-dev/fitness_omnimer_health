# More Screen (Màn hình Mở rộng)

Thư mục này chứa mã nguồn cho màn hình **More** (Settings & Utilities) của ứng dụng OmniMer Health.

## 📋 Tổng Quan Chức Năng

Màn hình này đóng vai trò là trung tâm quản lý tài khoản, cài đặt ứng dụng và các tiện ích mở rộng.

### 1. 👤 Profile (Hồ sơ cá nhân)

Quản lý thông tin định danh và chỉ số cơ thể.

- **Edit Profile:** Cập nhật Avatar, Tên hiển thị, Email, SĐT.
- **Body Metrics:** Cập nhật Cân nặng, Chiều cao, Tuổi (Dữ liệu đầu vào quan trọng cho AI Model).
- **Change Password:** Đổi mật khẩu.
- **Logout:** Đăng xuất.

### 2. 🔒 Quản lý Quyền Riêng Tư & LOD (Privacy)

Kiểm soát dữ liệu và chia sẻ dữ liệu dưới dạng Linked Open Data.

- **Data Visibility:** Bật/tắt quyền chia sẻ cho từng loại dữ liệu (Steps, Heart Rate, v.v.).
- **Anonymize Data:** Tùy chọn ẩn danh tính khi chia sẻ dữ liệu public.
- **LOD Endpoint:** Cung cấp mã QR/Link để chia sẻ dữ liệu chuẩn RDF/JSON-LD cho nghiên cứu/bác sĩ.

### 3. ⌚ Kết Nối Thiết Bị (Device Connectivity)

Đồng bộ dữ liệu từ thiết bị đeo thông minh.

- **HealthKit (iOS):** Đồng bộ Steps, Heart Rate, Sleep từ Apple Watch/iPhone.
- **Health Connect (Android):** Đồng bộ từ Samsung Watch và các thiết bị Android Wear khác.
- **Sync Status:** Hiển thị trạng thái kết nối và lần đồng bộ cuối.

### 4. 🎨 Cài Đặt & Giao Diện

- **Theme:** Light Mode / Dark Mode / System Default.
- **Language:** Tiếng Việt / English.
- **Notifications:** Cài đặt nhắc nhở tập luyện.

### 5. 💳 Pay for Health (Premium)

Các gói dịch vụ nâng cao.

- **Premium Features:** AI Recommendation chuyên sâu, Báo cáo PDF, No Ads.
- **Payment:** Tích hợp cổng thanh toán (VNPay/Momo/In-App Purchase).

### 6. ⭐ Tiện Ích Khác

- **Rate Us:** Đánh giá ứng dụng trên Store.
- **Feedback:** Gửi góp ý/báo lỗi về Admin Dashboard.
- **About Us:** Thông tin phiên bản, điều khoản sử dụng, chính sách bảo mật.

---

## 🏗️ Cấu Trúc UI Gợi Ý

```
MoreScreen
├── Header (User Info)
├── Section: Account (Profile, Premium)
├── Section: Health Data (Connect Devices, Privacy/LOD, Export)
├── Section: Settings (Theme, Language, Noti)
├── Section: Support (Rate, Feedback, About)
└── Footer (Logout, Version)
```
