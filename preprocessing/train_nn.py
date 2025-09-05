#!/usr/bin/env python3
import argparse, os, json, sys
from typing import Tuple, List, Optional, Dict

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import roc_auc_score

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
    y = pd.to_numeric(df[label_col], errors="coerce")
    good = ~y.isna()
    X = X.loc[good]; y = y.loc[good]
    col_means = X.mean(axis=0, skipna=True).fillna(0.0)
    X = X.fillna(col_means)
    return X.to_numpy(dtype=np.float32), y.to_numpy(dtype=np.float32), feature_cols

def compute_class_weight(y: np.ndarray) -> Optional[Dict[int, float]]:
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
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=float(lr)),
                  loss="binary_crossentropy")
    return model

def make_normalizer(kind: str, X: np.ndarray):
    kind = (kind or "none").lower()
    if kind == "none":
        return lambda Z: Z, {"type": "none"}
    if kind == "standard":
        mean = X.mean(axis=0)
        std = X.std(axis=0)
        std = np.where(std == 0, 1.0, std)
        return lambda Z: (Z - mean) / std, {"type": "standard", "mean": mean.tolist(), "std": std.tolist()}
    if kind == "minmax":
        mn = X.min(axis=0); mx = X.max(axis=0)
        rng = mx - mn; rng = np.where(rng == 0, 1.0, rng)
        return lambda Z: (Z - mn) / rng, {"type": "minmax", "min": mn.tolist(), "max": mx.tolist()}
    raise ValueError(f"unknown normalizer '{kind}'")

# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label-col", required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--hidden-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--l2", type=float, default=0.0)
    parser.add_argument("--normalization", choices=["none", "standard", "minmax"], default="standard")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use-class-weight", type=str, default="auto")  # "auto" | "off"
    parser.add_argument("--model_dir", default=None)  # injected by SageMaker; we’ll ignore
    args, _ = parser.parse_known_args()

    print(f"🔢 TF={tf.__version__}")
    print(f"🧪 Args: {vars(args)}")

    train_dir = os.environ.get("SM_CHANNEL_TRAIN", "/opt/ml/input/data/train")
    val_dir   = os.environ.get("SM_CHANNEL_VALIDATION", "/opt/ml/input/data/validation")
    model_dir = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")
    output_dir = os.environ.get("SM_OUTPUT_DATA_DIR", "/opt/ml/output")

    tf.keras.utils.set_random_seed(args.seed)

    train_csvs = list_csvs(train_dir)
    val_csvs   = list_csvs(val_dir)
    if not train_csvs or not val_csvs:
        raise RuntimeError(f"❌ No CSV found in channels. Train: {train_csvs}  Val: {val_csvs}")

    df_tr = load_concat_csvs(train_csvs)
    df_va = load_concat_csvs(val_csvs)

    Xtr, ytr, feat_cols = coerce_xy(df_tr, args.label_col)
    Xva, yva, _         = coerce_xy(df_va, args.label_col)

    norm_fn, norm_meta = make_normalizer(args.normalization, Xtr)
    Xtr = norm_fn(Xtr); Xva = norm_fn(Xva)

    model = build_model(
        input_dim=Xtr.shape[1],
        hidden_dim=args.hidden_dim,
        hidden_layers=args.hidden_layers,
        dropout=args.dropout,
        lr=args.lr,
        l2_reg=args.l2
    )

    class_weight = None
    if args.use_class_weight.lower() == "auto":
        cw = compute_class_weight(ytr)
        if cw:
            class_weight = cw
            print(f"🧮 Using class weights: {cw}")

    # Keep only “safe” callbacks (no saving during fit)
    es = keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
    rlrop = keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, verbose=1)

    model.fit(
        Xtr, ytr,
        validation_data=(Xva, yva),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=[es, rlrop],
        class_weight=class_weight,
        verbose=2
    )

    preds = model.predict(Xva, batch_size=args.batch_size).ravel()
    preds = np.clip(preds, 0.0, 1.0)
    auc = float(roc_auc_score(yva, preds))

    # Emit for tuner regex
    print(f"validation:auc={auc:.6f}")
    print(f"validation-auc={auc:.6f}")

    # FINAL SAVE — use SavedModel directory (no .keras suffix)
    final_dir = os.path.join(model_dir, "savedmodel")  # directory target
    os.makedirs(final_dir, exist_ok=True)
    model.save(final_dir)  # SavedModel supports options and avoids the native-keras/options clash

    artifacts = {
        "validation_auc": float(auc),
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
            "seed": int(args.seed)
        }
    }
    with open(os.path.join(model_dir, "artifacts.json"), "w") as f:
        json.dump(artifacts, f)
    print(f"✅ Saved model to {final_dir}")
    print("✅ Saved artifacts.json")

    try:
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "metrics.json"), "w") as f:
            json.dump({"validation_auc": float(auc)}, f)
    except Exception as e:
        print(f"⚠️ Could not write metrics to output dir: {e}", file=sys.stderr)

if __name__ == "__main__":
    main()
