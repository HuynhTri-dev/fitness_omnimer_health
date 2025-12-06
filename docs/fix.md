## 2. Dockerfile cho Flutter (Mobile Web) chưa hoàn thiện

**File:** `omnihealthmobileflutter/Dockerfile`

**Lỗi:**
Dockerfile hiện tại chỉ dừng lại ở bước build (`RUN flutter build web --release`). Không có stage để chạy ứng dụng (web server). Khi chạy container này, nó sẽ tắt ngay lập tức sau khi build xong.

**Cách khắc phục:**
Cần thêm stage Nginx để serve các file tĩnh đã build.
_Ví dụ:_

```dockerfile
# ... (phần build giữ nguyên)

# Stage 2: Serve
FROM nginx:alpine
COPY --from=builder /app/build/web /usr/share/nginx/html
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
```

## 4. Thiếu Service trong Docker Compose

**File:** `docker-compose.yml`

**Vấn đề:**
File này chưa định nghĩa service cho `adminpage` và `omnihealthmobileflutter`. Nếu bạn muốn chạy toàn bộ stack bằng 1 lệnh `docker-compose up`, cần thêm 2 service này vào.

**Gợi ý cấu hình:**

```yaml
admin_page:
  container_name: omnimer_admin_page
  build:
    context: ./adminpage
    dockerfile: Dockerfile
  ports:
    - "3000:80"
  networks:
    - omnimer_network

mobile_web_app:
  container_name: omnimer_mobile_web
  build:
    context: ./omnihealthmobileflutter
    dockerfile: Dockerfile
  ports:
    - "3001:80"
  networks:
    - omnimer_network
```

## 5. Dockerfile Backend: Kiểm tra port

**File:** `omnimer_health_server/Dockerfile`
**Lỗi nhẹ:** `EXPOSE 8000` khớp với `PORT=8000` trong `docker-compose.yml`. Tuy nhiên, cần đảm bảo code trong `src/server.ts` thực sự listen theo biến môi trường `PORT` hoặc mặc định là 8000. (Đã kiểm tra sơ bộ là khớp).
