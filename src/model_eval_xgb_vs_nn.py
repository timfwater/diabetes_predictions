#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Headless model evaluation for XGB vs NN with S3 I/O + plots + HTML.

Examples:
  python model_eval_xgb_vs_nn.py \
    --bucket diabetes-directory \
    --pred-key 03_scored/predictions_both_20250905-102823.csv \
    --label-col readmitted \
    --out-prefix 04_eval \
    --thresholds "0.2,0.3,0.5" \
    --region us-east-1
"""
import argparse, os, io, re, json
from datetime import datetime
import numpy as np
import pandas as pd
import boto3
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_auc_score, roc_curve, average_precision_score, precision_recall_curve,
    confusion_matrix, precision_score, recall_score, f1_score, brier_score_loss,
    accuracy_score
)

# ---------- Helpers ----------
def s3_read_csv(s3c, bucket, key):
    obj = s3c.get_object(Bucket=bucket, Key=key)
    body = obj["Body"].read()
    return pd.read_csv(io.BytesIO(body))

def s3_write_bytes(s3c, bucket, key, data: bytes, content_type="application/octet-stream"):
    s3c.put_object(Bucket=bucket, Key=key, Body=data, ContentType=content_type)

def s3_write_text(s3c, bucket, key, text: str, content_type="text/plain; charset=utf-8"):
    s3_write_bytes(s3c, bucket, key, text.encode("utf-8"), content_type)

def coerce_label(y: pd.Series) -> pd.Series:
    mapping = {"NO":0,"No":0,"no":0,"0":0,"FALSE":0,"False":0,"false":0,
               "YES":1,"Yes":1,"yes":1,"1":1,"TRUE":1,"True":1,"true":1,
               "<30":1,">30":1}
    if y.dtype == object:
        y = y.map(mapping).fillna(y)
    y = pd.to_numeric(y, errors="coerce")
    return y

def guess_label_col(df: pd.DataFrame):
    candidates = [
        "label","target","readmitted","y","y_true","is_readmitted",
        "Readmitted","READMITTED"
    ]
    for c in candidates:
        if c in df.columns:
            return c
    # fallback: binary column
    for c in df.columns:
        s = pd.to_numeric(df[c], errors="coerce").dropna()
        vals = set(s.unique().tolist())
        if len(vals) <= 2 and vals.issubset({0,1}):
            return c
    return None

def looks_like_prob(series: pd.Series) -> bool:
    s = pd.to_numeric(series, errors="coerce").dropna()
    return (len(s) > 0) and (s.min() >= 0.0) and (s.max() <= 1.0)

def to_proba(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce").fillna(0.0)
    if s.min() >= 0 and s.max() <= 1:  # already prob
        return s.clip(0, 1)
    # assume logits → sigmoid
    return 1.0 / (1.0 + np.exp(-s))

def find_proba_col(df: pd.DataFrame, tag: str):
    patterns = [
        rf"^{tag}.*(proba|prob|score|pred_prob|pred_proba)$",
        rf"^(proba|prob|score)_{tag}$",
        rf"^{tag}.*(pred|output)$",
        rf"^{tag}.*$"  # last resort: any col containing tag
    ]
    # Pass 1: pattern match + prob-like values
    for p in patterns:
        for c in df.columns:
            if re.search(p, c, flags=re.IGNORECASE):
                if looks_like_prob(df[c]):
                    return c
    # Pass 2: any prob-like column with tag in name
    for c in df.columns:
        if re.search(tag, c, flags=re.IGNORECASE) and looks_like_prob(df[c]):
            return c
    # Pass 3: any prob-like column
    for c in df.columns:
        if looks_like_prob(df[c]):
            return c
    # Give up; caller can still pass through and we'll sigmoid later
    return None

def point_metrics(y_true, y_prob, thr):
    y_pred = (y_prob >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
    return dict(
        threshold=thr,
        accuracy=accuracy_score(y_true, y_pred),
        precision=precision_score(y_true, y_pred, zero_division=0),
        recall=recall_score(y_true, y_pred, zero_division=0),
        f1=f1_score(y_true, y_pred, zero_division=0),
        tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp),
    )

def plot_roc(y_true, y_prob, title, outpath):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC={auc:.3f}")
    plt.plot([0,1],[0,1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(outpath, dpi=140)
    plt.close()

def plot_pr(y_true, y_prob, title, outpath):
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    plt.figure()
    plt.plot(rec, prec, label=f"AP={ap:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(title)
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(outpath, dpi=140)
    plt.close()

def plot_confusion(y_true, y_prob, thr, title, outpath):
    y_pred = (y_prob >= thr).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0,1])
    plt.figure()
    plt.imshow(cm, interpolation="nearest")
    plt.title(f"{title} (thr={thr:.2f})")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.xticks([0,1],[0,1])
    plt.yticks([0,1],[0,1])
    for (i,j), v in np.ndenumerate(cm):
        plt.text(j, i, str(v), ha="center", va="center")
    plt.tight_layout()
    plt.savefig(outpath, dpi=140)
    plt.close()

# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--region", default="us-east-1")
    ap.add_argument("--bucket", required=True)
    ap.add_argument("--pred-key", required=True)
    ap.add_argument("--label-col", default=None, help="If omitted, auto-detects")
    ap.add_argument("--thresholds", default="0.2,0.3,0.5")
    ap.add_argument("--out-prefix", default="04_eval", help="S3 prefix for outputs")
    ap.add_argument("--local-outdir", default=None, help="Local output dir; default: ./eval_reports/<ts>")
    ap.add_argument("--confusion-thr", type=float, default=0.5, help="Threshold to use for confusion plots")
    args = ap.parse_args()

    s3 = boto3.client("s3", region_name=args.region)
    df = s3_read_csv(s3, args.bucket, args.pred_key)

    # Label
    label_col = args.label_col or guess_label_col(df)
    if not label_col or label_col not in df.columns:
        raise SystemExit("❌ Label column not found (auto-detect failed). Use --label-col.")
    y = coerce_label(df[label_col])
    if y.isna().any():
        raise SystemExit("❌ Label column contains non-binary/NaN values after coercion.")

    # Find model proba columns
    xgb_col = "xgb_prob" if "xgb_prob" in df.columns else None
    nn_col  = "nn_prob"  if "nn_prob"  in df.columns else None
    if xgb_col is None: xgb_col = find_proba_col(df, "xgb")
    if nn_col  is None: nn_col  = find_proba_col(df, "nn")
    if xgb_col is None and "xgb_pred" in df.columns: xgb_col = "xgb_pred"
    if nn_col  is None and "nn_pred"  in df.columns: nn_col  = "nn_pred"
    if xgb_col is None or nn_col is None:
        raise SystemExit(f"❌ Could not find both model columns. Found xgb={xgb_col}, nn={nn_col}.")

    pxgb = to_proba(df[xgb_col])
    pnn  = to_proba(df[nn_col])

    # Optional ensemble
    ensemble = (pxgb + pnn) / 2.0

    # Summary metrics (threshold independent)
    models = {
        "xgb": pxgb,
        "nn":  pnn,
        "ensemble_avg": ensemble
    }
    rows = []
    for name, prob in models.items():
        rows.append({
            "model": name,
            "roc_auc": float(roc_auc_score(y, prob)),
            "pr_auc":  float(average_precision_score(y, prob)),
            "brier":   float(brier_score_loss(y, prob)),
        })
    summary_df = pd.DataFrame(rows)

    # Point metrics at thresholds
    thresholds = [float(x) for x in args.thresholds.split(",") if x.strip()]
    points = []
    for name, prob in models.items():
        for thr in thresholds:
            pm = point_metrics(y, prob, thr)
            pm["model"] = name
            points.append(pm)
    points_df = pd.DataFrame(points)

    # Output paths (S3 + local)
    ts = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    local_outdir = args.local_outdir or os.path.join("eval_reports", ts)
    os.makedirs(local_outdir, exist_ok=True)

    # --- Save CSVs locally and to S3
    csv1_local = os.path.join(local_outdir, f"summary_{ts}.csv")
    csv2_local = os.path.join(local_outdir, f"point_metrics_{ts}.csv")
    summary_df.to_csv(csv1_local, index=False)
    points_df.to_csv(csv2_local, index=False)

    csv1_s3 = f"{args.out_prefix}/tables/summary_{ts}.csv"
    csv2_s3 = f"{args.out_prefix}/tables/point_metrics_{ts}.csv"
    s3_write_text(s3, args.bucket, csv1_s3, summary_df.to_csv(index=False))
    s3_write_text(s3, args.bucket, csv2_s3, points_df.to_csv(index=False))

    # --- Plots: ROC/PR + Confusion (per chosen thr)
    figs = []
    for name, prob in models.items():
        roc_local = os.path.join(local_outdir, f"roc_{name}.png")
        pr_local  = os.path.join(local_outdir, f"pr_{name}.png")
        plot_roc(y, prob, f"{name.upper()} ROC", roc_local)
        plot_pr(y, prob, f"{name.upper()} PR",  pr_local)
        figs += [("roc", name, roc_local), ("pr", name, pr_local)]

        cm_local = os.path.join(local_outdir, f"cm_{name}_thr{args.confusion_thr:.2f}.png")
        plot_confusion(y, prob, args.confusion_thr, f"{name.upper()} Confusion", cm_local)
        figs.append(("cm", name, cm_local))

    # Upload figs to S3
    for kind, name, path in figs:
        key = f"{args.out_prefix}/figs/{kind}_{name}_{ts}.png"
        with open(path, "rb") as f:
            s3_write_bytes(s3, args.bucket, key, f.read(), content_type="image/png")

    # --- HTML report
    pred_file = f"s3://{args.bucket}/{args.pred_key}"
    html = []
    html.append("<html><head><meta charset='utf-8'><title>Eval Report</title></head><body>")
    html.append("<h1>Evaluation Report — XGB vs NN</h1>")
    html.append(f"<p><b>Predictions:</b> {pred_file}</p>")
    html.append(f"<p><b>Label column:</b> {label_col}</p>")
    html.append(f"<p><b>XGB column:</b> {xgb_col} &nbsp;&nbsp; <b>NN column:</b> {nn_col}</p>")
    html.append("<h2>Summary metrics</h2>")
    html.append(summary_df.to_html(index=False, float_format=lambda x: f"{x:.4f}"))
    html.append("<h2>Point metrics</h2>")
    html.append(points_df.to_html(index=False, float_format=lambda x: f"{x:.4f}"))

    # image refs (S3)
    html.append("<h2>Curves & Confusion</h2><ul>")
    for kind, name, _ in figs:
        img_key = f"{args.out_prefix}/figs/{kind}_{name}_{ts}.png"
        img_url = f"https://{args.bucket}.s3.amazonaws.com/{img_key}"
        html.append(f"<li>{kind.upper()} — {name}:<br><img src='{img_url}' style='max-width:900px'></li>")
    html.append("</ul>")
    html.append("</body></html>")
    html_str = "\n".join(html)

    # save & upload
    html_local = os.path.join(local_outdir, f"report_{ts}.html")
    with open(html_local, "w", encoding="utf-8") as f:
        f.write(html_str)
    html_s3 = f"{args.out_prefix}/report_{ts}.html"
    s3_write_text(s3, args.bucket, html_s3, html_str, content_type="text/html; charset=utf-8")

    # Small JSON summary for programmatic checks
    summary_json = {
        "predictions": pred_file,
        "label_col": label_col,
        "xgb_col": xgb_col,
        "nn_col": nn_col,
        "thresholds": thresholds,
        "confusion_thr": args.confusion_thr,
        "summary": rows,
        "s3_outputs": {
            "summary_csv": f"s3://{args.bucket}/{csv1_s3}",
            "point_metrics_csv": f"s3://{args.bucket}/{csv2_s3}",
            "html_report": f"s3://{args.bucket}/{html_s3}",
            "figs_prefix": f"s3://{args.bucket}/{args.out_prefix}/figs/"
        }
    }
    json_local = os.path.join(local_outdir, f"summary_{ts}.json")
    with open(json_local, "w", encoding="utf-8") as f:
        json.dump(summary_json, f, indent=2)

    print("✅ Wrote:")
    print(f"  s3://{args.bucket}/{csv1_s3}")
    print(f"  s3://{args.bucket}/{csv2_s3}")
    print(f"  s3://{args.bucket}/{html_s3}")
    print(f"  (figures at s3://{args.bucket}/{args.out_prefix}/figs/)")
    print(f"📁 Local artifacts in: {local_outdir}")

if __name__ == "__main__":
    main()
