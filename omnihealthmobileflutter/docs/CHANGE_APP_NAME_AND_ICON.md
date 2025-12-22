# Hướng dẫn đổi Tên và Icon ứng dụng

## 1. Đổi tên ứng dụng (Display Name)

### Android

Đã được tự động đổi thành **"OmniMer Health"**.
File cấu hình: `android/app/src/main/AndroidManifest.xml`

```xml
<application
    android:label="OmniMer Health"
    ...
>
```

### iOS

Bạn cần mở file `ios/Runner/Info.plist` và sửa giá trị của `CFBundleDisplayName`.
(File này đang bị gitignore nên tôi không thể tự sửa trực tiếp).

```xml
<key>CFBundleDisplayName</key>
<string>OmniMer Health</string>
```

---

## 2. Đổi Icon ứng dụng

Tôi đã cấu hình sẵn thư viện `flutter_launcher_icons` trong `pubspec.yaml`.

### Bước 1: Chuẩn bị ảnh

Tạo một file ảnh icon (PNG, không nền trong suốt nếu được, kích thước tốt nhất là 1024x1024 pixel).
Đặt tên file là `app_icon.png`.
Lưu vào thư mục: `d:\dacn_omnimer_health\omnihealthmobileflutter\assets\images\logo\app_icon.png`

### Bước 2: Chạy lệnh tạo icon

Mở terminal tại thư mục `omnihealthmobileflutter` và chạy lần lượt 2 lệnh sau:

```bash
flutter pub get
dart run flutter_launcher_icons
```

Lệnh này sẽ tự động tạo ra các icon kích thước khác nhau cho cả Android và iOS.

### Lưu ý

- Nếu bạn muốn dùng ảnh khác, hãy đặt ảnh đó vào và cập nhật đường dẫn `image_path` trong file `pubspec.yaml`.
- Với Android, icon mới sẽ thay thế icon con robot mặc định.
- Bạn có thể cần gỡ app và cài lại (`flutter run`) để thấy thay đổi về icon và tên.
