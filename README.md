# OmniMer Health

A Personal Health Management & AI-driven Fitness Recommendation System
(Built with Flutter, Node.js TypeScript, Python DNN Model)

---

## 1. Overview

**OmniMer Health** là nền tảng quản lý sức khỏe cá nhân và gợi ý bài tập thông minh dựa trên dữ liệu người dùng, cảm biến thiết bị đồng hồ thông minh và mô hình AI.
Hệ thống bao gồm ứng dụng di động (Flutter), backend (Node.js TypeScript), và mô hình AI dự đoán – khuyến nghị (Python DNN).

Mục tiêu:

- Thu thập dữ liệu sức khỏe từ cảm biến & nhật ký cá nhân
- Phân tích sức khỏe theo thời gian thực
- Gợi ý bài tập phù hợp với thể trạng
- Xây dựng lộ trình luyện tập cá nhân hóa
- Theo dõi giấc ngủ, stress, calories, heart rate

---

## 2. System Architecture

```
           ┌──────────────────────────┐
           │       Flutter App        │
           │  (Mobile Frontend)       │
           └─────────────┬────────────┘
                         │
            REST API / WebSocket
                         │
┌────────────────────────▼────────────────────────┐
│            Node.js (TypeScript) Backend         │
│ - Auth, User Profile                            │
│ - Health Data API                               │
│ - Workout Database & Recommendation Service     │
│ - Integration: HealthKit, Google Fit            │
└─────────────┬───────────────────────────────────┘
              │
      AI Inference Service (Python)
              │
┌─────────────▼─────────────────────────────────────┐
│               DNN Recommendation Model             │
│ - Predict calories, heart-rate zones               │
│ - Recommend exercise intensity                     │
│ - Personalized workout plans                       │
└────────────────────────────────────────────────────┘
```

---

## 3. Key Features

### Health Tracking

- Thu thập bước chân, nhịp tim, calories, giấc ngủ
- Tự động đồng bộ từ HealthKit / Google Fit

### AI-powered Personalization

- Mô hình DNN dự đoán chỉ số thể lực
- Gợi ý bài tập phù hợp theo tuổi, BMI, lịch sử luyện tập
- Dự đoán lượng calories tiêu hao

### Workout Management

- Tập luyện theo mục tiêu (giảm cân, tăng cơ, cardio)
- Lộ trình training được AI gợi ý
- Danh mục bài tập đầy đủ và phân loại theo nhóm cơ

### Real-time Analytics

- Dashboard theo thời gian
- Stress monitoring
- HR Zone Visualization

### Secure User Data

- JWT authentication
- Data encryption
- Cloudflare + Rate-limit

---

## 4. Tech Stack

### Mobile

- **Flutter** (Dart)
- State Management: Bloc / Cubit
- Flutter Secure Storage / Shared Preference Storage
- HealthKit Integration

### Backend

- **Node.js (TypeScript)**
- Express.js
- MongoDB / PostgreSQL
- JWT + RBAC
- Swagger OpenAPI
- Cloudflare Workers

### AI Model

- **Python**
- PyTorch
- DNN regression & classification

---

## 5. Folder Structure

```
omnihealth/
│
├── omnimerhealthmobieflutter/                 # Flutter App
│   ├── lib/
│   ├── assets/
│   └── pubspec.yaml
│
├── omnimer_health_server/                # Node.js + TypeScript
│   ├── src/
│   │   ├── modules/
│   │   ├── controllers/
│   │   ├── services/
│   │   ├── models/
│   │   └── routes/
│   ├── prisma/ (optional)
│   └── tsconfig.json
│
└── 3T-FIT/               # Python DNN Model
    ├── notebooks/
    ├── dataset/
    ├── src/
    │   ├── train.py
    │   ├── model.py
    │   └── inference.py
    ├── requirements.txt
```

---

## 6. Getting Started

### Mobile (Flutter)

```
cd omnimerhealthmobieflutter
flutter pub get
flutter run
```

### Backend (Node.js)

```
cd omnimer_health_server
npm install
npm run dev
```

Cấu hình môi trường theo file .env.example

### AI Model (Python)

```
cd 3T-FIT
pip install -r requirements.txt
python src/inference.py
```

---

## 7. API Documentation (On Update)

Swagger UI:

```
GET /api/docs
```

---

## 8. AI Model Documentation

Model DNN bao gồm:

- Input: tuổi, giới tính, BMI, HR, thời lượng tập, cường độ
- Output:

  - Calories burned
  - HR zone
  - Exercise intensity recommendation

Training scripts:

```
python src/train.py
```

Inference example:

```
python src/inference.py --age 24 --height 175 --weight 70 --avg_hr 140
```

---

## 10. License

This project is licensed under a **Commercial Proprietary License**.
All rights reserved. Redistribution and unauthorized copying of this project is strictly prohibited.
