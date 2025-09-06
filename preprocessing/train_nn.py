#!/usr/bin/env python3
import argparse, os, json, sys
from typing import Tuple, List, Optional, Dict

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler

np.set_printoptions(threshold=10, edgeitems=3, suppress=True)

# -----------------------------
# Utilities
# -----------------------------
def list_csvs(dir_path: str) -> List[str]:
    if not os.path.isdir(dir_path):
        return []
    return sorted([os.path.join(dir_path, f) for f in os.listdir(dir_path) if f.endswith(".csv")])

def load_concat_csvs(paths: List[str]) -> pd.DataFrame:
    frames = []
    for p in paths:
        try:
            frames.append(pd.read_csv(p))
        except Exception as e:
            print(f"⚠️  Failed to read {p}: {e}", file=sys.stderr)
    if not frames:
        raise RuntimeError("❌ No readable CSV files found.")
    return pd.concat(frames, axis=0, ignore_index=True)

def coerce_xy(df: pd.DataFrame, label_col: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    if label_col not in df.columns:
        raise ValueError(f"❌ label column '{label_col}' not found. Available: {list(df.columns)[:10]}...")
    feature_cols = [c for c in df.columns if c != label_col]
    X = df[feature_cols].apply(pd.to_numeric, errors="coerce")
    y = df[label_col]
    # normalize label encodings to {0,1}
    mapping = {
        "NO":0,"No":0,"no":0,"0":0,"FALSE":0,"False":0,"false":0,
        "YES":1,"Yes":1,"yes":1,"1":1,"TRUE":1,"True":1,"true":1,
        "<30":1,">30":1
    }
    if y.dtype == object:
        y = y.map(mapping)
    y = pd.to_numeric(y, errors="coerce")
    good = ~y.isna()
    X = X.loc[good]
    y = y.loc[good]
    col_means = X.mean(axis=0, skipna=True).fillna(0.0)
    X = X.fillna(col_means)
    return X.to_numpy(dtype=np.float32), y.to_numpy(dtype=np.float32), feature_cols

def compute_class_weight_auto(y: np.ndarray) -> Optional[Dict[int, float]]:
    vals, counts = np.unique(y, return_counts=True)
    if len(vals) != 2:
        return None
    total = counts.sum()
    return {int(v): total / (2.0 * float(c)) for v, c in zip(vals, counts)}

def build_model(input_dim: int,
                hidden_dim: int = 128,
                hidden_layers: int = 2,
                dropout: float = 0.3,
                lr: float = 1e-3,
                l2_reg: float = 0.0) -> keras.Model:
    reg = keras.regularizers.l2(l2_reg) if l2_reg and l2_reg > 0 else None
    inputs = keras.Input(shape=(input_dim,), name="features")
    x = inputs
    for _ in range(int(hidden_layers)):
        x = keras.layers.Dense(int(hidden_dim), activation="relu", kernel_regularizer=reg)(x)
        if dropout and dropout > 0:
            x = keras.layers.Dropout(float(dropout))(x)
    out = keras.layers.Dense(1, activation="sigmoid", name="logit")(x)
    model = keras.Model(inputs, out)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=float(lr)),
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.AUC(curve="PR", name="aucpr"),
        ],
    )
    return model

def make_normalizer(kind: str, X: np.ndarray):
    kind = (kind or "none").lower()
    if kind == "none":
        return lambda Z: Z, {"type": "none"}
    if kind == "standard":
        sc = StandardScaler().fit(X)
        return lambda Z: sc.transform(Z), {"type": "standard", "mean": sc.mean_.tolist(), "scale": sc.scale_.tolist()}
    if kind == "minmax":
        sc = MinMaxScaler().fit(X)
        return lambda Z: sc.transform(Z), {"type": "minmax", "min": sc.data_min_.tolist(), "max": sc.data_max_.tolist()}
    raise ValueError(f"unknown normalizer '{kind}'")

# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label-col", required=True)

    # training dynamics
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)

    # architecture
    ap.add_argument("--hidden-dim", type=int, default=128)
    ap.add_argument("--hidden-layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--l2", type=float, default=0.0)

    # normalization & randomness
    ap.add_argument("--normalization", choices=["none", "standard", "minmax"], default="standard")
    ap.add_argument("--standardize", type=int, default=None)  # new flag (1/0); if set overrides --normalization
    ap.add_argument("--seed", type=int, default=42)

    # class-imbalance (new flags, but keep backward compat)
    ap.add_argument("--use-class-weights", type=int, default=None)  # 1/0
    ap.add_argument("--class-weight-pos", type=float, default=None)

    # legacy compat flag from your older script
    ap.add_argument("--use-class-weight", type=str, default=None)  # "auto" | "off"

    # misc / SageMaker
    ap.add_argument("--aucpr-objective", type=int, default=1)  # guides which metric callbacks watch
    ap.add_argument("--model_dir", default=None)

    args, _ = ap.parse_known_args()

    print(f"🔢 TF={tf.__version__}")
    print(f"🧪 Args: {vars(args)}")

    # SageMaker channels/dirs
    train_dir = os.environ.get("SM_CHANNEL_TRAIN", "/opt/ml/input/data/train")
    val_dir   = os.environ.get("SM_CHANNEL_VALIDATION", "/opt/ml/input/data/validation")
    model_dir = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")
    output_dir = os.environ.get("SM_OUTPUT_DATA_DIR", "/opt/ml/output")

    tf.keras.utils.set_random_seed(args.seed)

    # Load CSVs
    train_csvs = list_csvs(train_dir)
    val_csvs   = list_csvs(val_dir)
    if not train_csvs or not val_csvs:
        raise RuntimeError(f"❌ No CSV found in channels. Train: {train_csvs}  Val: {val_csvs}")

    df_tr = load_concat_csvs(train_csvs)
    df_va = load_concat_csvs(val_csvs)

    Xtr, ytr, feat_cols = coerce_xy(df_tr, args.label_col)
    Xva, yva, _         = coerce_xy(df_va, args.label_col)

    # Normalization logic (supports both flags)
    if args.standardize is not None:
        norm = "standard" if args.standardize else "none"
    else:
        norm = args.normalization
    norm_fn, norm_meta = make_normalizer(norm, Xtr)
    Xtr = norm_fn(Xtr); Xva = norm_fn(Xva)

    # Build model
    model = build_model(
        input_dim=Xtr.shape[1],
        hidden_dim=args.hidden_dim,
        hidden_layers=args.hidden_layers,
        dropout=args.dropout,
        lr=args.lr,
        l2_reg=args.l2
    )

    # Class weights (support new + legacy flags)  *** fixed underscores ***
    class_weight = None
    if args.use_class_weights is not None:
        if args.use_class_weights:  # 1
            if args.class_weight_pos is not None:
                class_weight = {0: 1.0, 1: float(args.class_weight_pos)}
            else:
                cw = compute_class_weight_auto(ytr)
                if cw: class_weight = cw
    else:
        # legacy path
        if args.use_class_weight and args.use_class_weight.lower() == "auto":
            cw = compute_class_weight_auto(ytr)
            if cw: class_weight = cw

    if class_weight:
        print(f"🧮 Using class weights: {class_weight}")

    # Callbacks: early stop + LR schedule on PR AUC (or ROC AUC)  *** fixed underscores ***
    monitor_metric = "val_aucpr" if args.aucpr_objective else "val_auc"
    es = keras.callbacks.EarlyStopping(monitor=monitor_metric, patience=6, restore_best_weights=True, mode="max")
    rlrop = keras.callbacks.ReduceLROnPlateau(monitor=monitor_metric, factor=0.5, patience=3, mode="max", verbose=1)

    # Train
    hist = model.fit(
        Xtr, ytr,
        validation_data=(Xva, yva),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=[es, rlrop],
        class_weight=class_weight,
        verbose=2
    )

    # Manual metrics for redundancy
    preds = model.predict(Xva, batch_size=args.batch_size).ravel()
    preds = np.clip(preds, 0.0, 1.0)
    auc   = float(roc_auc_score(yva, preds))
    ap    = float(average_precision_score(yva, preds))

    # Emit for tuner regex (both names supported):
    print(f"validation:auc={auc:.6f}")
    print(f"validation-auc={auc:.6f}")
    print(f"validation:aucpr={ap:.6f}")
    print(f"validation-aucpr={ap:.6f}")

    # FINAL SAVE — SavedModel directory to avoid keras options clashes
    final_dir = os.path.join(model_dir, "savedmodel")
    os.makedirs(final_dir, exist_ok=True)
    model.save(final_dir)

    artifacts = {
        "validation_auc": float(auc),
        "validation_aucpr": float(ap),
        "feature_names": feat_cols,
        "normalization": norm_meta,
        "class_weight": class_weight,
        "hyperparams": {
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "lr": float(args.lr),
            "hidden_dim": int(args.hidden_dim),
            "hidden_layers": int(args.hidden_layers),
            "dropout": float(args.dropout),
            "l2": float(args.l2),
            "seed": int(args.seed),
            "normalization": norm,
        }
    }
    with open(os.path.join(model_dir, "artifacts.json"), "w") as f:
        json.dump(artifacts, f)
    print(f"✅ Saved model to {final_dir}")
    print("✅ Saved artifacts.json")

    try:
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "metrics.json"), "w") as f:
            json.dump({"validation_auc": float(auc), "validation_aucpr": float(ap)}, f)
    except Exception as e:
        print(f"⚠️ Could not write metrics to output dir: {e}", file=sys.stderr)

if __name__ == "__main__":
    main()
