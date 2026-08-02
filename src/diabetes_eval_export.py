#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -------------------------
# Run:
#   cd diabetes_predictions
#   python diabetes_eval_export.py \
#       --data s3://diabetes-directory/03_scored/prepared_diabetes_test_selected_with_predictions.csv \
#       --out  s3://diabetes-directory/04_eval/ \
#       --topk 10
# -------------------------

import os
import io
import argparse
import numpy as np
import pandas as pd
import boto3

# Use a non-interactive backend (safe for terminals/servers)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from typing import Literal, Tuple, Optional
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    roc_curve,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    matthews_corrcoef,
    accuracy_score,
    log_loss,
    brier_score_loss,
)

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)

# =========
# Utilities
# =========

def read_csv_any(path: str) -> pd.DataFrame:
    """Read CSV from local or s3://. Tries pandas+s3fs, falls back to boto3."""
    if not path.startswith("s3://"):
        return pd.read_csv(path)
    # S3 path
    try:
        import s3fs  # if present, pandas can read s3:// directly
        return pd.read_csv(path)
    except Exception:
        # fallback to boto3
        s3 = boto3.client("s3")
        bucket, key = path[5:].split("/", 1)
        obj = s3.get_object(Bucket=bucket, Key=key)
        return pd.read_csv(io.BytesIO(obj["Body"].read()))

def _save_current_figure(path: str, dpi: int = 300):
    """Save the *current* Matplotlib figure to local path or s3://bucket/key.ext."""
    ext = os.path.splitext(path)[1].lower()
    if ext not in (".png", ".pdf"):
        raise ValueError("Only .png or .pdf supported")

    if path.startswith("s3://"):
        bucket, key = path[5:].split("/", 1)
        fmt = "png" if ext == ".png" else "pdf"
        content_type = "image/png" if fmt == "png" else "application/pdf"
        buf = io.BytesIO()
        plt.savefig(buf, format=fmt, dpi=dpi, bbox_inches="tight")
        plt.close()
        buf.seek(0)
        boto3.client("s3").put_object(Bucket=bucket, Key=key, Body=buf.getvalue(), ContentType=content_type)
        print(f"✅ Saved to s3://{bucket}/{key}")
    else:
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        plt.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close()
        print(f"✅ Saved {path}")

def df_to_image(df: pd.DataFrame, filename: str, title: str | None = None, dpi: int = 300):
    """Render a small DataFrame to a static image/pdf and save (local or S3)."""
    fig, ax = plt.subplots(figsize=(len(df.columns)*1.2, len(df)*0.5 + 1))
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=14, pad=10)
    tbl = ax.table(cellText=df.values, colLabels=df.columns, cellLoc="center", loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1.2, 1.2)
    plt.tight_layout()
    _save_current_figure(filename, dpi=dpi)

# ================
# Metrics helpers
# ================

def add_threshold_flags(
    df: pd.DataFrame,
    c1: str,
    X=None, *,
    step=0.05,
    start=0.05,
    end=1.0,
    colname_fmt="{col}_ge_{thr_pct:03d}pct",
    preserve_na=True,
    inplace=False
) -> pd.DataFrame:
    if c1 not in df.columns:
        raise KeyError(f"Column '{c1}' not in DataFrame.")
    out_df = df if inplace else df.copy()

    if X is None:
        X = [round(x, 2) for x in np.arange(start, end + 1e-9, step)]
    else:
        X = sorted({round(float(x), 2) for x in X})
    for t in X:
        if not (0.05 <= t <= 1.0):
            raise ValueError(f"Threshold {t} out of allowed range [0.05, 1.0].")

    vals = pd.to_numeric(out_df[c1], errors="coerce")
    for t in X:
        thr_pct = int(round(t * 100))
        colname = colname_fmt.format(col=c1, thr=t, thr_pct=thr_pct)
        comp = (vals >= t)
        if preserve_na:
            series = pd.Series(
                np.where(vals.isna(), pd.NA, comp.astype(int)),
                index=out_df.index,
                dtype="Int8"
            )
        else:
            series = comp.fillna(False).astype("int8")
        out_df.loc[:, colname] = series
    return out_df

def _coerce_binary(y: pd.Series) -> pd.Series:
    if y.dtype == object:
        mapping = {
            "NO":0,"No":0,"no":0,"0":0,"FALSE":0,"False":0,"false":0,
            "YES":1,"Yes":1,"yes":1,"1":1,"TRUE":1,"True":1,"true":1,
            "<30":1, ">30":1
        }
        y = y.map(mapping)
    y = pd.to_numeric(y, errors="coerce")
    y = y.map(lambda v: np.nan if pd.isna(v) else (1 if v >= 0.5 else 0))
    return y.astype("float")

def _pick_threshold(
    y_true: np.ndarray,
    scores: np.ndarray,
    strategy: Literal["youden","f1","fixed","balanced"] = "youden",
    fixed_value: float = 0.5
) -> float:
    if strategy == "fixed":
        return float(fixed_value)
    m = ~np.isnan(scores) & ~np.isnan(y_true)
    y = y_true[m]
    s = scores[m]
    if np.unique(s).size < 2:
        return float(np.median(s))
    if strategy == "youden":
        fpr, tpr, thr = roc_curve(y, s)
        j = tpr - fpr
        best = int(np.argmax(j))
        return float(thr[best])
    elif strategy == "f1":
        prec, rec, thr = precision_recall_curve(y, s)
        prec, rec = prec[:-1], rec[:-1]
        f1_vals = np.where((prec + rec) > 0, 2 * prec * rec / (prec + rec), 0.0)
        if f1_vals.size == 0:
            return float(np.median(s))
        return float(thr[int(np.argmax(f1_vals))])
    elif strategy == "balanced":
        prevalence = y.mean()
        uniq = np.unique(s)
        if uniq.size > 1000:
            qs = np.linspace(0, 1, 1001)
            uniq = np.quantile(s, qs)
        diffs = []
        for t in uniq:
            yhat = (s >= t).astype(int)
            diffs.append(abs(yhat.mean() - prevalence))
        return float(uniq[int(np.argmin(diffs))])
    return 0.5

def _confusion_counts(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[int,int,int,int]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
    return tp, tn, fp, fn

def metrics_vs_all(
    df: pd.DataFrame,
    c1: str,
    *,
    threshold_strategy: Literal["youden","f1","fixed","balanced"] = "youden",
    fixed_threshold: float = 0.5,
    min_unique_scores: int = 2,
    sort_by: str = "auc_roc",
    sort_desc: bool = True,
    invert_scores_if_auc_below_half: bool = True,
) -> pd.DataFrame:
    if c1 not in df.columns:
        raise KeyError(f"Target column '{c1}' not found in DataFrame.")
    y_series = _coerce_binary(df[c1])
    results = []

    for col in df.columns:
        if col == c1:
            continue
        x_raw = pd.to_numeric(df[col], errors="coerce")
        mask = (~y_series.isna()) & (~x_raw.isna())
        if not mask.any():
            continue
        y = y_series[mask].astype(int).values
        x = x_raw[mask].astype(float).values
        n = y.shape[0]
        if np.unique(y).size < 2:
            continue

        x_unique = np.unique(x)
        is_binary_pred = np.array_equal(np.sort(x_unique), np.array([0.0, 1.0])) or (
            x_unique.size <= 3 and set(np.round(x_unique, 6)).issubset({0.0, 1.0})
        )

        auc_roc = np.nan
        auc_pr = np.nan
        avg_prec = np.nan
        threshold_used: Optional[float] = None
        yhat = None

        try:
            if is_binary_pred:
                yhat = x.astype(int)
                if np.unique(x).size >= min_unique_scores:
                    auc_roc = roc_auc_score(y, x)
                    avg_prec = average_precision_score(y, x)
                    auc_pr = avg_prec
            else:
                if x_unique.size < min_unique_scores:
                    continue
                auc_roc = roc_auc_score(y, x)
                x_for_thr = x.copy()
                if invert_scores_if_auc_below_half and auc_roc < 0.5:
                    x_for_thr = -x_for_thr
                    auc_roc = 1.0 - auc_roc
                avg_prec = average_precision_score(y, x_for_thr)
                auc_pr = avg_prec
                threshold_used = _pick_threshold(
                    y, x_for_thr, strategy=threshold_strategy, fixed_value=fixed_threshold
                )
                yhat = (x_for_thr >= threshold_used).astype(int)

            if yhat is None:
                t = np.median(x)
                yhat = (x >= t).astype(int)
                threshold_used = t

            tp, tn, fp, fn = _confusion_counts(y, yhat)
            prec = precision_score(y, yhat, zero_division=0)
            rec = recall_score(y, yhat, zero_division=0)
            f1 = f1_score(y, yhat, zero_division=0)
            acc = accuracy_score(y, yhat)
            spec = tn / (tn + fp) if (tn + fp) > 0 else np.nan
            mcc = matthews_corrcoef(y, yhat) if (tp+tn+fp+fn) > 0 else np.nan
            prev = y.mean()
            ppr = yhat.mean()

            ll = np.nan
            bs = np.nan
            try:
                if not is_binary_pred:
                    xmin, xmax = np.nanmin(x), np.nanmax(x)
                    if xmax > xmin:
                        p = (x - xmin) / (xmax - xmin)
                        ll = log_loss(y, p, labels=[0,1])
                        bs = brier_score_loss(y, p)
                else:
                    if set(np.unique(x)).issubset({0.0,1.0}):
                        ll = log_loss(y, x, labels=[0,1])
                        bs = brier_score_loss(y, x)
            except Exception:
                pass

            results.append({
                "feature": col,
                "auc_roc": float(auc_roc) if not np.isnan(auc_roc) else np.nan,
                "auc_pr": float(auc_pr) if not np.isnan(auc_pr) else np.nan,
                "avg_precision": float(avg_prec) if not np.isnan(avg_prec) else np.nan,
                "f1": float(f1),
                "precision": float(prec),
                "recall": float(rec),
                "specificity": float(spec) if not np.isnan(spec) else np.nan,
                "accuracy": float(acc),
                "mcc": float(mcc) if not np.isnan(mcc) else np.nan,
                "log_loss": float(ll) if not np.isnan(ll) else np.nan,
                "brier": float(bs) if not np.isnan(bs) else np.nan,
                "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
                "threshold_used": float(threshold_used) if threshold_used is not None else np.nan,
                "pred_pos_rate": float(ppr),
                "prevalence": float(prev),
                "n": int(n),
            })
        except Exception:
            continue

    out = pd.DataFrame(results).set_index("feature")
    if (len(out) > 0) and (sort_by in out.columns):
        out = out.sort_values(sort_by, ascending=not sort_desc)
    return out

# ==================
# Plotting helpers
# ==================

def plot_cm_from_index(
    df: pd.DataFrame,
    index_value,
    *,
    tp_col: str = "tp",
    tn_col: str = "tn",
    fp_col: str = "fp",
    fn_col: str = "fn",
    labels = (0, 1),
    title: str | None = None,
    normalize: str | None = None,
    annot_fontsize: int = 12,
    linewidths: float = 1.0,
    savepath: str | None = None,
    dpi: int = 300
):
    if index_value not in df.index:
        raise KeyError(f"Index {index_value!r} not found in DataFrame index.")
    row = df.loc[index_value]
    tp, tn, fp, fn = int(row[tp_col]), int(row[tn_col]), int(row[fp_col]), int(row[fn_col])

    cm_counts = np.array([[tn, fp],
                          [fn, tp]], dtype=float)

    def _normalize_cm(cm, how):
        if how is None:
            return cm
        if how == "all":
            total = cm.sum()
            return cm / total if total > 0 else cm
        if how == "true":
            denom = cm.sum(axis=1, keepdims=True)
            return np.divide(cm, denom, out=np.zeros_like(cm), where=(denom != 0))
        if how == "pred":
            denom = cm.sum(axis=0, keepdims=True)
            return np.divide(cm, denom, out=np.zeros_like(cm), where=(denom != 0))
        raise ValueError("normalize must be one of {None,'all','true','pred'}")

    cm = _normalize_cm(cm_counts.copy(), normalize)

    cell_labels = np.array([
        ["True Negatives", "False Positives"],
        ["False Negatives", "True Positives"]
    ])

    ann = np.empty_like(cell_labels, dtype=object)
    for i in range(2):
        for j in range(2):
            if normalize is None:
                ann[i, j] = f"{cell_labels[i,j]}\n{int(cm_counts[i,j]):,}"
            else:
                ann[i, j] = f"{cell_labels[i,j]}\n{int(cm_counts[i,j]):,}\n({cm[i,j]*100:.1f}%)"

    plt.figure(figsize=(6, 5))
    ax = sns.heatmap(
        cm,
        annot=ann, fmt="",
        cmap="Blues", cbar=False,
        xticklabels=labels, yticklabels=labels,
        linewidths=linewidths, linecolor="gray", square=True,
        annot_kws={"fontsize": annot_fontsize}
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual", fontsize=12)
    ax.set_title(title or f"Confusion Matrix — index={index_value}", fontsize=14)
    plt.tight_layout()

    if savepath:
        _save_current_figure(savepath, dpi=dpi)
    else:
        plt.show()

def plot_histograms(
    dataframe: pd.DataFrame,
    columns,
    hist_figsize=(8, 6),
    save_dir: str | None = None,
    dpi: int = 300,
    file_ext: str = ".png",
    on_missing: str = "error",
    bins: int = 1000,
):
    if file_ext not in (".png", ".pdf"):
        raise ValueError("file_ext must be '.png' or '.pdf'")

    available = set(dataframe.columns)
    missing = [c for c in columns if c not in available]
    if missing:
        msg = f"Columns not found in DataFrame: {missing}\nAvailable sample: {sorted(list(available))[:10]}..."
        if on_missing == "error":
            raise KeyError(msg)
        else:
            print("⚠️  " + msg)
            columns = [c for c in columns if c in available]
            if not columns:
                print("No valid columns left to plot; exiting.")
                return

    for column in columns:
        series = pd.to_numeric(dataframe[column], errors="coerce").dropna()
        if series.empty:
            print(f"⚠️  Column '{column}' has no numeric values; skipping.")
            continue

        plt.figure(figsize=hist_figsize)
        plt.hist(series, bins=bins, density=True, alpha=0.7, label=column)
        plt.xlabel(f"Predicted Values for {column}")
        plt.ylabel("Frequency")
        plt.title(f"Distribution of Prediction Values from Model — {column}")
        mean_value = float(series.mean())
        plt.axvline(mean_value, color="red", linestyle="dashed", linewidth=2,
                    label=f"Mean: {mean_value:.3f}")
        plt.legend()

        if save_dir is None:
            plt.show()
        else:
            out_name = f"hist_{column}{file_ext}"
            target = out_name if not save_dir else (
                f"{save_dir.rstrip('/')}/{out_name}" if save_dir.startswith("s3://")
                else os.path.join(save_dir, out_name)
            )
            _save_current_figure(target, dpi=dpi)
        print(f"The mean predictive value for {column} is: {mean_value:.3f}")

# =========
# Main
# =========

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="s3://diabetes-directory/03_scored/prepared_diabetes_test_selected_with_predictions.csv",
                   help="Input CSV (local path or s3://bucket/key)")
    p.add_argument("--out",  default="s3://diabetes-directory/04_eval/",
                   help="Output dir (local path or s3://bucket/prefix/)")
    p.add_argument("--label-col", default="readmitted")
    p.add_argument("--xgb-col",   default="xgb_prob")
    p.add_argument("--nn-col",    default="nn_prob")
    p.add_argument("--thr-min", type=float, default=0.05)
    p.add_argument("--thr-max", type=float, default=0.95)
    p.add_argument("--thr-step", type=float, default=0.05)
    p.add_argument("--topk", type=int, default=10)
    args = p.parse_args()

    print("✅ Config loaded.")
    df_temp = read_csv_any(args.data)

    # Base frame + ensemble
    cols = [args.label_col, args.xgb_col, args.nn_col]
    df = df_temp.loc[:, cols].copy()
    df.loc[:, "ensemble_prob"] = (pd.to_numeric(df[args.xgb_col], errors="coerce") +
                                  pd.to_numeric(df[args.nn_col], errors="coerce")) / 2.0

    # Threshold flags
    df2 = add_threshold_flags(
            add_threshold_flags(
                add_threshold_flags(df, args.xgb_col, start=args.thr_min, end=args.thr_max, step=args.thr_step),
                args.nn_col, start=args.thr_min, end=args.thr_max, step=args.thr_step
            ),
            "ensemble_prob", start=args.thr_min, end=args.thr_max, step=args.thr_step
        )

    # Metrics
    metrics_table = metrics_vs_all(
        df2, args.label_col,
        threshold_strategy="youden",
        fixed_threshold=0.5,
        invert_scores_if_auc_below_half=True,
    )

    # Cost model
    cost_of_management_program = 500
    success_rate_of_intervention = 0.5
    cost_of_readmission = 15000

    metrics_table["total_cost_of_implimenting_program"] = (metrics_table["tp"]+metrics_table["fp"]) * cost_of_management_program
    metrics_table["per_patient_outlay"] = round(metrics_table["total_cost_of_implimenting_program"]/len(df2), 2)
    metrics_table["readmissions_prevented"] = round(success_rate_of_intervention * metrics_table["tp"])
    metrics_table["savings_from_readmissions_prevention"] = round(metrics_table["readmissions_prevented"] * cost_of_readmission, 2)
    metrics_table["per_patient_savings_from_readmissions_prevention"] = round(metrics_table["savings_from_readmissions_prevention"]/len(metrics_table), 2)
    metrics_table["net_savings_to_hosptial_system"] = metrics_table["savings_from_readmissions_prevention"] - metrics_table["total_cost_of_implimenting_program"]
    metrics_table["net_per_patient_savings"] = round(metrics_table["net_savings_to_hosptial_system"]/len(df2), 2)

    # ------------- Exports -------------
    out_dir = args.out.rstrip("/")

    # Tables
    dx1 = metrics_table.sort_values("auc_roc", ascending=False).head(args.topk)
    df_to_image(dx1, f"{out_dir}/AUC_table.png", title="Greatest AUC")

    dx2 = metrics_table.sort_values("net_per_patient_savings", ascending=False).head(args.topk)
    df_to_image(dx2, f"{out_dir}/Savings_table.pdf", title="Greatest Savings")

    dx3 = metrics_table.sort_values(by=["fn", "net_per_patient_savings"], ascending=[True, False]).head(args.topk)
    df_to_image(dx3, f"{out_dir}/FN_then_Savings_table.png", title="Top 10 (FN asc, Savings desc)")

    # Confusion matrices (pick indices that exist in metrics_table)
    for idx_name, fname in [
        (args.xgb_col, "xgb_confusion.png"),
        (f"{args.xgb_col}_ge_035pct", "AUC_confusion.png"),
        ("ensemble_prob_ge_035pct", "FN_confusion.png"),
    ]:
        if idx_name in metrics_table.index:
            plot_cm_from_index(metrics_table, idx_name, savepath=f"{out_dir}/{fname}")
        else:
            print(f"ℹ️  Skipping CM for '{idx_name}' (not found in metrics_table.index).")

    # Histograms
    plot_histograms(df2, [args.xgb_col, args.nn_col, "ensemble_prob"],
                    save_dir=f"{out_dir}/histograms", file_ext=".png")

    print("✅ Done.")

if __name__ == "__main__":
    main()
