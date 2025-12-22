# 3T-FIT AI Server - Model Files

## ⚠️ Model Files Missing

Thư mục này cần chứa các file model đã được train để AI Server hoạt động đầy đủ.

### Required Files Structure:

```
ai_server/model/src/v4/personal_model_v4/
├── model_weights.pth         # PyTorch model weights
├── feature_scaler.pkl        # StandardScaler for input features
└── model_metadata.json       # Model configuration metadata
```

### File Descriptions:

1. **model_weights.pth**

   - PyTorch state dict của TwoBranchRecommendationModel
   - Chứa trọng số của cả 2 nhánh: Intensity Prediction & Suitability Prediction

2. **feature_scaler.pkl**
   - Sklearn StandardScaler đã được fit với training data
   - Bắt buộc phải có để chuẩn hóa input features
3. **model_metadata.json** (Optional)
   - Chứa thông tin: input_dim, architecture details, training metrics
   - Nếu không có, sẽ dùng default input_dim=28

### How to Train the Model:

Tham khảo README.md ở thư mục gốc (`3T-FIT/README.md`) để:

1. Chuẩn bị training data từ MongoDB (WatchLog, Workout, Exercise)
2. Chạy training pipeline
3. Lưu model weights và scaler vào thư mục này

### For Development/Testing:

Server sẽ vẫn khởi động được nhưng endpoint `/v4/recommend` sẽ không hoạt động khi thiếu model files.

Bạn có thể test các endpoint khác:

- `GET /health` - Health check
- `GET /model/info` - Model information
- `GET /docs` - Swagger API documentation
