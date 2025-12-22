# 🚀 Hướng dẫn chạy OmniMer Health Servers

## 📌 Tổng quan

Dự án `dacn_omnimer_health` bao gồm 2 server chính:

1. **🤖 AI Server** (`3T-FIT/ai_server`) - Python/FastAPI - Port 8888
2. **🌐 Backend Server** (`omnimer_health_server`) - Node.js/TypeScript - Port 8000

---

## 🤖 1. AI Server (Python/FastAPI)

### Thông tin cơ bản:

- **Framework**: FastAPI + Uvicorn
- **Port**: 8888
- **Model**: Two-Branch Neural Network (Intensity + Suitability Prediction)

### Cách chạy:

#### ✅ Option 1: Chạy bằng script (Khuyến nghị cho Windows)

```cmd
cd d:\dacn_omnimer_health\3T-FIT\ai_server
start_server.bat
```

#### ✅ Option 2: Chạy thủ công

```cmd
cd d:\dacn_omnimer_health\3T-FIT
set PYTHONPATH=d:\dacn_omnimer_health\3T-FIT\ai_server\app
uvicorn ai_server.app.main:app --host 0.0.0.0 --port 8888 --reload
```

#### ✅ Option 3: Docker (Production)

```cmd
cd d:\dacn_omnimer_health
docker-compose up ai_service
```

### Endpoints:

- **Health Check**: http://localhost:8888/health
- **Model Info**: http://localhost:8888/model/info
- **API Docs (Swagger)**: http://localhost:8888/docs
- **Recommend API**: http://localhost:8888/v4/recommend

### ⚠️ Lưu ý về Model Files:

Server sẽ chạy được nhưng endpoint `/v4/recommend` cần các file model tại:

```
d:\dacn_omnimer_health\3T-FIT\ai_server\model\src\v4\personal_model_v4\
├── model_weights.pth
├── feature_scaler.pkl
└── model_metadata.json
```

Nếu chưa có model, xem `ai_server/model/README.md` để biết cách train.

---

## 🌐 2. Backend Server (Node.js/TypeScript)

### Thông tin cơ bản:

- **Framework**: Express.js + TypeScript
- **Port**: 8000
- **Database**: MongoDB, Redis
- **Features**: User Management, Workout Tracking, Exercise Management, GraphDB Integration

### Yêu cầu:

- Node.js 24+
- MongoDB (local hoặc cloud)
- Redis (local hoặc cloud)

### Cách chạy:

#### ✅ Option 1: Development Mode (Khuyến nghị)

```cmd
cd d:\dacn_omnimer_health\omnimer_health_server

# Lần đầu: Install dependencies
npm install

# Copy .env.example và cấu hình
copy .env.example .env
# Sau đó sửa file .env với các thông tin kết nối thực tế

# Chạy dev mode (hot reload)
npm run dev
```

#### ✅ Option 2: Docker (Production)

```cmd
cd d:\dacn_omnimer_health
docker-compose up backend
```

#### ✅ Option 3: Docker Development Mode

```cmd
cd d:\dacn_omnimer_health
docker-compose -f docker-compose.yml -f docker-compose.dev.yml up backend
```

### Endpoints:

- **Server**: http://localhost:8000
- **API Routes**: Xem file `API_ROUTES.md`

### 📋 Environment Variables (`.env`):

Các biến quan trọng cần cấu hình:

```env
# Server
PORT=8000
NODE_ENV=development

# Database
MONGO_URI=mongodb://localhost:27017/omnimerhealth
# Hoặc MongoDB Atlas:
# MONGO_URI=mongodb+srv://username:password@cluster.mongodb.net/

# Redis
REDIS_HOST=127.0.0.1
REDIS_PORT=6379
REDIS_USERNAME=admin
REDIS_PASSWORD=health123

# AI Service
AI_API=http://localhost:8888
# Docker: AI_API=http://omnimer_health_ai:8888

# GraphDB
GRAPHDB_URL=http://localhost:7200
GRAPHDB_REPO=omnimer_health_lod

# JWT Secrets
ACCESS_TOKEN_SECRET=your_secret_here
REFRESH_TOKEN_SECRET=your_secret_here
```

Xem `.env.example` để biết đầy đủ các biến cần thiết.

---

## 🐳 3. Chạy toàn bộ hệ thống với Docker Compose

### Production Mode:

```cmd
cd d:\dacn_omnimer_health
docker-compose up
```

Sẽ khởi động tất cả services:

- ✅ **backend** (Node.js) - Port 8000
- ✅ **ai_service** (Python/FastAPI) - Port 8888
- ✅ **graphdb** (GraphDB) - Port 7200
- ✅ **admin_page** (React/Vite) - Port 5137

### Development Mode (Hot Reload):

```cmd
cd d:\dacn_omnimer_health
docker-compose -f docker-compose.yml -f docker-compose.dev.yml up
```

### Chạy từng service riêng lẻ:

```cmd
# Chỉ backend
docker-compose up backend

# Chỉ AI service
docker-compose up ai_service

# Backend + AI + GraphDB
docker-compose up backend ai_service graphdb
```

---

## 🔍 Troubleshooting

### AI Server:

**Lỗi: `ModuleNotFoundError: No module named 'api'`**

- ✅ **Giải pháp**: Chạy từ thư mục `3T-FIT` (không phải `ai_server`)
- ✅ Set `PYTHONPATH` đúng (đã được fix trong `start_server.bat`)

**Warning: `Model v4 loading failed`**

- ⚠️ **Nguyên nhân**: Chưa có file model
- ✅ **Giải pháp**: Server vẫn chạy được, nhưng cần train model hoặc copy model files vào thư mục chỉ định

### Backend Server:

**Lỗi: Cannot connect to MongoDB**

- ✅ Kiểm tra `MONGO_URI` trong `.env`
- ✅ Đảm bảo MongoDB đang chạy (local hoặc cloud accessible)

**Lỗi: Cannot connect to Redis**

- ✅ Kiểm tra Redis đang chạy: `redis-cli ping`
- ✅ Cài Redis trên Windows qua WSL hoặc dùng Docker

**Lỗi: AI Service not responding**

- ✅ Đảm bảo AI Server đang chạy ở port 8888
- ✅ Kiểm tra `AI_API` trong `.env` của backend

---

## 📊 Kiểm tra Services

### AI Server Health Check:

```cmd
curl http://localhost:8888/health
# Response: {"status":"healthy","message":"OmniMer Health Recommendation API is running"}
```

### Backend Health Check:

```cmd
curl http://localhost:8000/health
# hoặc endpoint tùy theo API design
```

### Docker Services Status:

```cmd
docker-compose ps
```

---

## 📚 Tài liệu tham khảo:

- **AI Server**: `3T-FIT/README.md` - Chi tiết kiến trúc model
- **Backend API**: `omnimer_health_server/API_ROUTES.md` - API documentation
- **Model Training**: `3T-FIT/ai_server/model/README.md` - Hướng dẫn train model

---

## 🎯 Quick Start (Recommended Steps):

1. **Start AI Server first:**

   ```cmd
   cd d:\dacn_omnimer_health\3T-FIT\ai_server
   start_server.bat
   ```

2. **Configure Backend `.env`:**

   ```cmd
   cd d:\dacn_omnimer_health\omnimer_health_server
   copy .env.example .env
   # Edit .env with your configurations
   ```

3. **Start Backend Server:**

   ```cmd
   npm run dev
   ```

4. **Verify both servers are running:**
   - AI: http://localhost:8888/docs
   - Backend: http://localhost:8000

---

Chúc bạn deployment thành công! 🚀
