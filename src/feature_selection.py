# preprocessing/feature_selection.py
#!/usr/bin/env python3
import io
import os
import sys
from pathlib import Path

import boto3
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import cfg  # noqa: E402

BUCKET = cfg.get("storage.bucket")
PREFIX = cfg.prefix("engineered")
LABEL_COL = cfg.get("data.label_col")

# Which dataset to rank features on. The runner sets FILTERED_INPUT_FILE to
# the train split; ranking on anything containing test rows leaks.
INPUT_FILE = os.environ.get("FILTERED_INPUT_FILE", cfg.get("data.files.train"))

INPUT_KEY = f"{PREFIX}/{INPUT_FILE}"
OUTPUT_KEY = cfg.s3_key("engineered", "selected_features")

FS_MODE = str(cfg.get("feature_selection.mode")).lower()   # 'cv' | 'quick'
FS_METHOD = str(cfg.get("feature_selection.method")).lower()
TOP_K = int(cfg.get("feature_selection.top_k"))
CV_FOLDS = int(cfg.get("feature_selection.cv_folds"))
SEED = int(cfg.get("feature_selection.seed"))

# Previously this was read as a float defaulting to 0, and "0 means use top_k"
# was the implicit switch. The method is now explicit, so the two settings no
# longer silently compete.
FS_CUM_IMPORTANCE = float(cfg.get("feature_selection.cum_importance"))

s3 = boto3.client("s3")

def s3_read_csv(bucket, key):
    obj = s3.get_object(Bucket=bucket, Key=key)
    return pd.read_csv(io.BytesIO(obj["Body"].read()))

def s3_put_text(bucket, key, text):
    s3.put_object(Bucket=bucket, Key=key, Body=text.encode("utf-8"))
    print(f"📤 Uploaded s3://{bucket}/{key}")

# --- Load data ---
df = s3_read_csv(BUCKET, INPUT_KEY).dropna(subset=[LABEL_COL])
y_raw = df[LABEL_COL]
mapping = {"NO":0,"No":0,"no":0,"0":0,"FALSE":0,"False":0,"false":0,
           "YES":1,"Yes":1,"yes":1,"1":1,"TRUE":1,"True":1,"true":1,
           "<30":1, ">30":1}
y = pd.to_numeric(y_raw.map(mapping) if y_raw.dtype == object else y_raw,
                  errors="coerce").astype("float32")
mask = y.notna()
y = y[mask].astype("int8")

X_all = df.loc[mask].drop(columns=[LABEL_COL])
num_cols = X_all.select_dtypes(include=["number", "bool"]).columns.tolist()
if not num_cols:
    raise ValueError("No numeric features found. Ensure upstream step encoded categoricals.")
X = X_all[num_cols].copy().astype("float32")

print(f"📦 Candidate features: {len(num_cols)} | FS_MODE={FS_MODE}")

def gain_from_model(model, cols):
    booster = model.get_booster()
    score_map = booster.get_score(importance_type="gain")
    return np.array([score_map.get(c, 0.0) for c in cols], dtype=np.float64)

# --- Train & collect importances ---
if FS_MODE == "quick":
    model = xgb.XGBClassifier(
        objective="binary:logistic", eval_metric="aucpr",
        random_state=42, n_estimators=400, max_depth=6,
        learning_rate=0.1, subsample=0.8, colsample_bytree=0.8,
        n_jobs=-1, tree_method="hist",
    )
    model.fit(X, y, verbose=False)
    gain = gain_from_model(model, num_cols)
else:
    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)
    gain = np.zeros(len(num_cols), dtype=np.float64)
    for tr, va in skf.split(X, y):
        model = xgb.XGBClassifier(
            objective="binary:logistic", eval_metric="aucpr",
            random_state=42, n_estimators=400, max_depth=6,
            learning_rate=0.1, subsample=0.8, colsample_bytree=0.8,
            n_jobs=-1, tree_method="hist",
        )
        model.fit(X.iloc[tr], y.iloc[tr], eval_set=[(X.iloc[va], y.iloc[va])], verbose=False)
        gain += gain_from_model(model, num_cols)
    gain /= float(CV_FOLDS)

# --- Ranking ---
rank = pd.DataFrame({"feature": num_cols, "gain_cv": gain})
rank = rank.sort_values("gain_cv", ascending=False).reset_index(drop=True)
rank["cum_importance"] = rank["gain_cv"].cumsum() / rank["gain_cv"].sum()

# --- Selection ---
if FS_METHOD == "cumulative_importance":
    sel = rank[rank["cum_importance"] <= FS_CUM_IMPORTANCE]
    method = f"cumulative_importance<={FS_CUM_IMPORTANCE}"
else:
    sel = rank.head(TOP_K)
    method = f"top_{TOP_K}"

sel_out = sel[["feature"]].rename(columns={"feature":"selected_features"})

# --- Save ---
s3_put_text(BUCKET, OUTPUT_KEY, sel_out.to_csv(index=False))
rank_key = f"{PREFIX}/feature_ranking_{pd.Timestamp.utcnow().strftime('%Y%m%d%H%M%S')}.csv"
s3_put_text(BUCKET, rank_key, rank.to_csv(index=False))

print(f"✅ Saved {len(sel_out)} features using {method} → s3://{BUCKET}/{OUTPUT_KEY}")
print(f"ℹ️ First 5: {sel_out['selected_features'].head(5).tolist()}")
print(f"ℹ️ Ranking snapshot: s3://{BUCKET}/{rank_key}")
