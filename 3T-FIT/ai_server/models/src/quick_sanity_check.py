# quick_sanity_check.py
import json, joblib, torch, pandas as pd, numpy as np
from torch import nn

ARTIFACTS = "artifacts_omni_excel"
PREPROC = f"{ARTIFACTS}/preprocessor.joblib"
META    = f"{ARTIFACTS}/meta.json"
MODEL   = f"{ARTIFACTS}/best.pt"
EXCEL   = "data/merged_omni_health_dataset.xlsx"

# Load artifacts
with open(META, "r", encoding="utf-8") as f:
    meta = json.load(f)
pre = joblib.load(PREPROC)
ckpt = torch.load(MODEL, map_location="cpu")
in_dim, num_ex, ex_cols = ckpt["in_dim"], ckpt["num_ex"], ckpt["ex_cols"]

# Model
class MTLNet(nn.Module):
    def __init__(self, in_dim, num_exercises, hidden=256):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(0.1),
        )
        self.head_cls = nn.Linear(hidden, num_exercises)
        self.head_reg = nn.Sequential(nn.Linear(hidden,128), nn.ReLU(), nn.Linear(128,3))
    def forward(self, x):
        h = self.trunk(x)
        return self.head_cls(h), self.head_reg(h)

model = MTLNet(in_dim, num_ex).eval()
model.load_state_dict(ckpt["state_dict"])

# Lấy đúng schema features đã dùng khi train
feat_used = meta["features_used"]

# Đọc 1-3 dòng từ file Excel train và suy luận
df_raw = pd.read_excel(EXCEL).head(3)
df = df_raw.copy()
for c in feat_used:
    if c not in df.columns: df[c] = np.nan
df = df[feat_used]

# Ép kiểu theo preprocessor (numeric/categorical)
num_cols, cat_cols = [], []
for name, pipe, cols in pre.transformers_:
    if name == "num": num_cols = list(cols)
    if name == "cat": cat_cols = list(cols)
for c in num_cols:
    if c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce")
for c in cat_cols:
    if c in df.columns: df[c] = df[c].astype("object")

X = pre.transform(df)
if hasattr(X, "toarray"): X = X.toarray()
X = X.astype("float32")
with torch.no_grad():
    logits, reg = model(torch.from_numpy(X))
    probs = torch.sigmoid(logits).numpy()

print("Prob stats per row:")
for i,row in enumerate(probs):
    print(f"Row {i}: min={row.min():.6f}, max={row.max():.6f}, mean={row.mean():.6f}")
    top_idx = np.argsort(-row)[:5]
    for j in top_idx:
        print(f"  - {ex_cols[j]} -> {row[j]:.6f}")
