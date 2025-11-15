"""
inference_unified.py (fixed)
- Load unified model + preprocessor
- Ép kiểu input theo schema của preprocessor (num/cat) trước khi transform
- Xuất JSON theo format: exercises[] + suitabilityScore + predictedAvgHR/PeakHR
"""

import json, joblib, numpy as np, pandas as pd, torch, torch.nn as nn

ART = "artifacts_unified"
PRE = f"{ART}/preprocessor.joblib"
META = f"{ART}/meta.json"
CKPT = f"{ART}/best.pt"

meta = json.load(open(META, "r", encoding="utf-8"))
pre  = joblib.load(PRE)
ckpt = torch.load(CKPT, map_location="cpu")

features = meta["features_used"]
ex_cols  = ckpt["ex_cols"]
sc       = meta["scales"]
in_dim   = ckpt["in_dim"]
num_ex   = ckpt["num_ex"]

# ----------------- model định nghĩa giống lúc train -----------------
class UnifiedMTL(nn.Module):
    def __init__(self, in_dim, num_ex, d=256, joint_d=256, drop=0.15):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, d), nn.ReLU(), nn.Dropout(drop),
            nn.Linear(d, d),     nn.ReLU(), nn.Dropout(drop)
        )
        self.ex_embed = nn.Parameter(torch.randn(num_ex, d) * 0.02)
        self.head_cls = nn.Sequential(nn.Linear(d*3, joint_d), nn.ReLU(), nn.Linear(joint_d, 1))
        self.head_reg = nn.Sequential(nn.Linear(d*3, joint_d), nn.ReLU(),
                                      nn.Linear(joint_d,128), nn.ReLU(), nn.Linear(128,8))
    def forward(self, x):
        B = x.size(0)
        h = self.encoder(x)               # [B,d]
        E = self.ex_embed                 # [C,d]
        C = E.size(0)
        h_exp = h.unsqueeze(1).expand(B, C, h.size(1))
        E_exp = E.unsqueeze(0).expand(B, C, E.size(1))
        joint = torch.cat([h_exp, E_exp, h_exp*E_exp], dim=2)  # [B,C,3d]
        logits = self.head_cls(joint).squeeze(-1)              # [B,C]
        reg = torch.sigmoid(self.head_reg(joint))              # [B,C,8]
        return logits, reg

model = UnifiedMTL(in_dim, num_ex).eval()
model.load_state_dict(ckpt["state_dict"])

# ----------------- helpers -----------------
def inv(v, lo, hi): return float(v)*(hi-lo)+lo
def cardio_like(name: str):
    s = name.lower()
    return any(k in s for k in ["run","jog","bike","cycle","row","swim","elliptical","walking","cycling"])

def cast_to_pre_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Ép kiểu DataFrame theo đúng num/cat columns của preprocessor trước khi transform."""
    # đảm bảo đủ cột & đúng thứ tự
    for c in features:
        if c not in df.columns:
            df[c] = np.nan
    df = df[features]

    # lấy danh sách cột num/cat từ preprocessor đã fit
    num_cols = [c for name, _, cols in pre.transformers_ if name == "num" for c in cols]
    cat_cols = [c for name, _, cols in pre.transformers_ if name == "cat" for c in cols]

    # ép numeric -> số (chuỗi như "male" trong cột numeric sẽ thành NaN)
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # ép categorical -> object (kể cả "male"/"female"...)
    for c in cat_cols:
        if c in df.columns:
            df[c] = df[c].astype("object")

    return df

def recommend(profile: dict, top_k: int = 5):
    # 1) Chuẩn hóa đầu vào
    Xdf = cast_to_pre_schema(pd.DataFrame([profile]))
    X_trans = pre.transform(Xdf)
    if hasattr(X_trans, "toarray"): X_trans = X_trans.toarray()
    X = torch.tensor(X_trans.astype("float32"))

    # 2) Suy luận
    with torch.no_grad():
        logits, reg = model(X)
        probs = torch.sigmoid(logits).numpy()[0]   # [C]
        reg   = reg.numpy()[0]                     # [C,8] đã scale [0..1]

    # 3) Lấy top-K bài & dựng JSON
    top_idx = np.argsort(-probs)[:top_k]
    plan = []
    avgHRs, peakHRs, prob_list = [], [], []

    for i in top_idx:
        name = ex_cols[i].replace("exercise_", "")
        y = reg[i]
        sets    = max(1, int(round(inv(y[0], *sc["sets"]))))
        reps    = int(round(inv(y[1], *sc["reps"])))
        kg      = max(0.0, inv(y[2], *sc["kg"]))
        km      = max(0.0, inv(y[3], *sc["km"]))
        minutes = max(0.0, inv(y[4], *sc["min"]))
        rest    = max(0.0, inv(y[5], *sc["minRest"]))
        avgHR   = int(round(inv(y[6], *sc["avgHR"])))
        peakHR  = int(round(inv(y[7], *sc["peakHR"])))

        if "max_weight_lifted_kg" in profile and profile["max_weight_lifted_kg"]:
            kg = min(kg, 0.9 * float(profile["max_weight_lifted_kg"]))

        sets_list = []
        if cardio_like(name):
            for _ in range(sets):
                sets_list.append({"reps": 0, "kg": 0.0, "km": round(km,2), "min": round(minutes,1), "minRest": round(rest,1)})
        else:
            for _ in range(sets):
                sets_list.append({"reps": int(reps), "kg": round(kg,1), "km": 0.0, "min": 0.0, "minRest": round(rest,1)})

        plan.append({"name": name, "sets": sets_list})
        avgHRs.append(avgHR); peakHRs.append(peakHR); prob_list.append(float(probs[i]))

    suitability = float(np.mean(prob_list)) if prob_list else 0.0
    return {
        "exercises": plan,
        "suitabilityScore": round(suitability, 4),
        "predictedAvgHR": int(np.mean(avgHRs)) if avgHRs else 0,
        "predictedPeakHR": int(np.max(peakHRs)) if peakHRs else 0
    }

# ----------------- demo -----------------
if __name__ == "__main__":
    sample = {
        "age": 21,  # bạn có thể cập nhật thực tế
        "height_cm": 158,
        "weight_kg": 73,
        "bmi": 29.6,
        "body_fat_percentage": 26.8,
        "whr": 0.97,
        "resting_hr": 74,
        "bp_systolic": 120,  # mặc định nếu chưa đo huyết áp
        "bp_diastolic": 80,
        "workout_frequency_per_week": 4,
        "gender": "male",
        "experience_level": "intermediate",
        "activity_level": 4, 
        "health_status": {
            "known_conditions": [],
            "pain_locations": [],
            "joint_issues": [],
            "injuries": [],
            "abnormalities": [],
            "notes": ""
        },
        "goal_type": "WeightLoss",
        "target_metric": {"weight_target": 65},  # ví dụ, có thể tuỳ biến
        "exercises": [
            {"exercise_name": "Đá bụng"}
        ],
    }

    out = recommend(sample, top_k=5)
    print(json.dumps(out, ensure_ascii=False, indent=2))
