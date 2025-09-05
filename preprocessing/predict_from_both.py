#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, sys, io, json
from datetime import datetime
from typing import List, Optional, Tuple

import boto3
import pandas as pd
from botocore.exceptions import ClientError, BotoCoreError

# -------------------
# Config / Endpoints
# -------------------
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
BUCKET = os.getenv("BUCKET", "diabetes-directory")
PREFIX = os.getenv("PREFIX", "02_engineered")
OUTPUT_PREFIX = os.getenv("OUTPUT_PREFIX", "03_scored")

ENDPOINT_XGB = os.getenv("ENDPOINT_XGB", os.getenv("ENDPOINT", "diabetes-xgb-endpoint"))
ENDPOINT_NN  = os.getenv("ENDPOINT_NN", "diabetes-nn-endpoint")

# what to run: both | xgb | nn
RUN_MODE = os.getenv("RUN_MODE", "both").lower().strip()

# Optional explicit S3 keys for feature lists (bucket is BUCKET)
XGB_FEATURES_KEY = os.getenv("XGB_FEATURES_KEY")   # e.g. "02_engineered/model_feature_lists/diabetes-xgb-endpoint-features.txt"
NN_FEATURES_KEY  = os.getenv("NN_FEATURES_KEY")    # e.g. "02_engineered/model_feature_lists/diabetes-nn-endpoint-features.txt"

# Legacy fallback (kept for backward compatibility)
TEST_KEY = os.getenv("TEST_KEY", f"{PREFIX}/prepared_diabetes_test.csv")
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "500"))

sm = boto3.client("sagemaker", region_name=AWS_REGION)
rt = boto3.client("sagemaker-runtime", region_name=AWS_REGION)
s3 = boto3.client("s3", region_name=AWS_REGION)

# -------------------
# S3 & feature utils
# -------------------
def _s3_read_text(bucket: str, key: str) -> str:
    obj = s3.get_object(Bucket=bucket, Key=key)
    return obj["Body"].read().decode("utf-8")

def _s3_read_csv(bucket: str, key: str) -> pd.DataFrame:
    obj = s3.get_object(Bucket=bucket, Key=key)
    return pd.read_csv(io.BytesIO(obj["Body"].read()))

def _s3_write_csv(bucket: str, key: str, df: pd.DataFrame):
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    s3.put_object(Bucket=bucket, Key=key, Body=buf.getvalue().encode("utf-8"),
                  ContentType="text/csv; charset=utf-8")

def _try_model_tag_features_key(endpoint_name: str) -> Optional[str]:
    """Return s3 key from model tag FeaturesListS3Key if present, else None."""
    try:
        e = sm.describe_endpoint(EndpointName=endpoint_name)
        econf = sm.describe_endpoint_config(EndpointConfigName=e["EndpointConfigName"])
        model_names = [pv["ModelName"] for pv in econf.get("ProductionVariants", [])]
        if not model_names:
            return None
        m = sm.describe_model(ModelName=model_names[0])
        tags = sm.list_tags(ResourceArn=m["ModelArn"]).get("Tags", [])
        tag_map = {t["Key"]: t["Value"] for t in tags}
        return tag_map.get("FeaturesListS3Key")
    except Exception:
        return None

def _fallback_features_key(endpoint_name: str) -> str:
    # Conventional location
    fname = f"{endpoint_name}-features.txt"
    return f"{PREFIX}/model_feature_lists/{fname}"

def _load_features_from_csv_first_col(key: str) -> List[str]:
    txt = _s3_read_text(BUCKET, key)
    df = pd.read_csv(io.StringIO(txt), header=None)
    cols = df.iloc[:,0].astype(str).str.strip().tolist()
    if cols and cols[0].lower() in {"selected_features", "feature", "features"}:
        cols = cols[1:]
    cols = [c for c in cols if c]
    return cols

def _load_features_from_txt_lines(key: str) -> List[str]:
    txt = _s3_read_text(BUCKET, key)
    cols = [ln.strip() for ln in txt.splitlines() if ln.strip()]
    return cols

def _resolve_features(endpoint_name: str,
                      explicit_key: Optional[str]) -> Tuple[List[str], str]:
    """
    Resolve features list for an endpoint and return (features, source_key_description).
    Tries explicit_key, then model tag, then conventional text file, then selected_features.csv.
    """
    tried = []

    # 1) explicit key
    if explicit_key:
        tried.append(f"s3://{BUCKET}/{explicit_key}")
        try:
            if explicit_key.lower().endswith(".csv"):
                cols = _load_features_from_csv_first_col(explicit_key)
            else:
                cols = _load_features_from_txt_lines(explicit_key)
            if cols:
                return cols, f"s3://{BUCKET}/{explicit_key}"
        except Exception:
            pass

    # 2) model tag
    tag_key = _try_model_tag_features_key(endpoint_name)
    if tag_key:
        tried.append(f"s3://{BUCKET}/{tag_key}")
        try:
            if tag_key.lower().endswith(".csv"):
                cols = _load_features_from_csv_first_col(tag_key)
            else:
                cols = _load_features_from_txt_lines(tag_key)
            if cols:
                return cols, f"s3://{BUCKET}/{tag_key}"
        except Exception:
            pass

    # 3) conventional text file
    conv_key = _fallback_features_key(endpoint_name)
    tried.append(f"s3://{BUCKET}/{conv_key}")
    try:
        cols = _load_features_from_txt_lines(conv_key)
        if cols:
            return cols, f"s3://{BUCKET}/{conv_key}"
    except Exception:
        pass

    # 4) selected_features.csv fallback
    sel_key = f"{PREFIX}/selected_features.csv"
    tried.append(f"s3://{BUCKET}/{sel_key}")
    try:
        cols = _load_features_from_csv_first_col(sel_key)
        if cols:
            return cols, f"s3://{BUCKET}/{sel_key}"
    except Exception:
        pass

    raise SystemExit("❌ Could not resolve features list for "
                     f"{endpoint_name}. Tried:\n  - " + "\n  - ".join(tried))

def _align_frame(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        print(f"ℹ️ Adding {len(missing)} missing columns as 0.0 (e.g. {missing[:5]})")
        for c in missing:
            df[c] = 0.0
    # reorder and force float
    aligned = df[cols].astype(float)
    return aligned

# ---------------
# Invoke helpers
# ---------------
def _invoke_csv(endpoint: str, X: pd.DataFrame) -> str:
    buf = io.StringIO()
    X.to_csv(buf, header=False, index=False)
    body = buf.getvalue().encode("utf-8")
    res = rt.invoke_endpoint(EndpointName=endpoint, ContentType="text/csv", Body=body)
    return res["Body"].read().decode("utf-8").strip()

def _parse_probs_any(raw: str, n_rows: int) -> List[float]:
    # JSON: {"predictions":[[p],[p],...]} or {"predictions":[p,p,...]}
    try:
        j = json.loads(raw)
        preds = j.get("predictions", [])
        out = []
        for p in preds:
            if isinstance(p, list):
                out.append(float(p[0]) if p else float("nan"))
            else:
                out.append(float(p))
        if len(out) == n_rows:
            return out
    except Exception:
        pass
    # CSV-ish / lines
    flat = raw.replace("\r", "\n").replace("\n", ",")
    vals = [v for v in (x.strip() for x in flat.split(",")) if v]
    out = [float(v) for v in vals]
    if len(out) == n_rows:
        return out
    # one per line fallback
    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    try:
        out = [float(ln) for ln in lines]
    except Exception:
        out = []
    if len(out) == n_rows:
        return out
    raise SystemExit(f"❌ Could not parse predictions into {n_rows} floats (got {len(out)}). Sample: {raw[:200]}")

def _maybe_inservice(endpoint_name: str) -> bool:
    try:
        desc = sm.describe_endpoint(EndpointName=endpoint_name)
        st = desc.get("EndpointStatus")
        return st == "InService"
    except Exception:
        return False

# ---------------
# Main
# ---------------
def main():
    # Allow CLI overrides for input and label (so we can emit *_with_predictions)
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-key", default=os.getenv("INPUT_KEY"), help="S3 key for input CSV (e.g., 02_engineered/test.csv)")
    ap.add_argument("--label-col", default=os.getenv("LABEL_COL"), help="Name of ground-truth label column in input")
    ap.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    args = ap.parse_args()

    if RUN_MODE not in {"both", "xgb", "nn"}:
        raise SystemExit(f"❌ RUN_MODE must be one of both|xgb|nn (got {RUN_MODE})")

    # 1) Load data
    if args.input_key:
        input_key = args.input_key
        df = _s3_read_csv(BUCKET, input_key)
        n_rows = len(df)
        print(f"📥 Loading input CSV with labels: s3://{BUCKET}/{input_key}  (rows={n_rows})")
    else:
        # Backward-compatible fallback to TEST_KEY (labelless)
        input_key = TEST_KEY
        txt = _s3_read_text(BUCKET, TEST_KEY)
        df = pd.read_csv(io.StringIO(txt))
        n_rows = len(df)
        print(f"📥 Loading legacy test CSV: s3://{BUCKET}/{TEST_KEY}  (rows={n_rows})")

    results = {}
    feature_sources = {}

    # 2) XGB branch
    if RUN_MODE in {"both", "xgb"}:
        if not _maybe_inservice(ENDPOINT_XGB):
            raise SystemExit(f"❌ XGB endpoint {ENDPOINT_XGB} is not InService.")
        xgb_cols, xgb_src = _resolve_features(ENDPOINT_XGB, XGB_FEATURES_KEY)
        feature_sources["xgb"] = xgb_src
        print(f"📌 XGB serving schema: {len(xgb_cols)} features from {xgb_src}")
        X_xgb = _align_frame(df.copy(), xgb_cols)
        print(f"➡️  Invoking XGB ({ENDPOINT_XGB}) with {n_rows} rows…")
        if n_rows <= args.batch_size:
            xgb_raw = _invoke_csv(ENDPOINT_XGB, X_xgb)
            results["xgb_prob"] = _parse_probs_any(xgb_raw, n_rows)
        else:
            probs = []
            for i in range(0, n_rows, args.batch_size):
                chunk = X_xgb.iloc[i:i+args.batch_size]
                r = _invoke_csv(ENDPOINT_XGB, chunk)
                probs.extend(_parse_probs_any(r, len(chunk)))
            results["xgb_prob"] = probs

    # 3) NN branch
    if RUN_MODE in {"both", "nn"}:
        if not _maybe_inservice(ENDPOINT_NN):
            raise SystemExit(f"❌ NN endpoint {ENDPOINT_NN} is not InService.")
        nn_cols, nn_src = _resolve_features(ENDPOINT_NN, NN_FEATURES_KEY)
        feature_sources["nn"] = nn_src
        print(f"📌 NN  serving schema: {len(nn_cols)} features from {nn_src}")
        X_nn = _align_frame(df.copy(), nn_cols)
        print(f"➡️  Invoking NN  ({ENDPOINT_NN}) with {n_rows} rows…")
        if n_rows <= args.batch_size:
            nn_raw = _invoke_csv(ENDPOINT_NN, X_nn)
            results["nn_prob"] = _parse_probs_any(nn_raw, n_rows)
        else:
            probs = []
            for i in range(0, n_rows, args.batch_size):
                chunk = X_nn.iloc[i:i+args.batch_size]
                r = _invoke_csv(ENDPOINT_NN, chunk)
                probs.extend(_parse_probs_any(r, len(chunk)))
            results["nn_prob"] = probs

    # 4) Always emit the compact predictions file (backward compatibility)
    compact = pd.DataFrame(results)
    # add IDs if present
    id_cols = [c for c in df.columns if c.lower() in {"encounter_id", "patient_nbr", "id"}]
    if id_cols:
        compact = pd.concat([df[id_cols].reset_index(drop=True), compact], axis=1)

    ts = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    suffix = RUN_MODE
    compact_key = f"{OUTPUT_PREFIX}/predictions_{suffix}_{ts}.csv"
    _s3_write_csv(BUCKET, compact_key, compact)
    print(f"✅ Wrote compact predictions to s3://{BUCKET}/{compact_key}")

    # 5) If we have input with labels, also emit <basename>_with_predictions.csv
    wrote_with_predictions = False
    if args.input_key and args.label_col and (args.label_col in df.columns):
        out_full = df.copy()
        out_full["xgb_prob"] = compact["xgb_prob"] if "xgb_prob" in compact.columns else None
        out_full["nn_prob"]  = compact["nn_prob"]  if "nn_prob"  in compact.columns else None

        base = os.path.basename(args.input_key)
        base_no_ext = base[:-4] if base.lower().endswith(".csv") else base
        out_key = f"{OUTPUT_PREFIX}/{base_no_ext}_with_predictions.csv"
        _s3_write_csv(BUCKET, out_key, out_full)
        wrote_with_predictions = True
        print(f"✅ Wrote labeled predictions to s3://{BUCKET}/{out_key}")
    else:
        if args.input_key and args.label_col and args.label_col not in df.columns:
            print(f"⚠️ Label column '{args.label_col}' not found in input; skipped *_with_predictions.csv")

    # 6) Trace feature sources
    if feature_sources:
        for k, src in feature_sources.items():
            print(f"ℹ️ {k.upper()} features from: {src}")

    # 7) Final notice
    if wrote_with_predictions:
        print("🎯 Ready for evaluation: use the *_with_predictions.csv with --label-col.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"💥 predict_from_both error: {e}", file=sys.stderr)
        sys.exit(1)
