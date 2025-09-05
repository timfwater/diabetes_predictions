# Used by SageMaker's Python service (script-mode).
# Robust to a variety of training artifact layouts.

import os, json, glob
import numpy as np
import tensorflow as tf
from tensorflow import keras

def _first_existing(*paths):
    for p in paths:
        if p and os.path.exists(p):
            return p
    return None

def _find_savedmodel(root):
    # Look for a SavedModel directory (has saved_model.pb)
    for cand in [
        root,
        os.path.join(root, "model"),
        os.path.join(root, "model", "1"),
        os.path.join(root, "output", "model"),
        os.path.join(root, "output", "model", "1"),
    ]:
        if os.path.exists(os.path.join(cand, "saved_model.pb")):
            return cand
    # Fallback: recursive search (cheap on small artifacts)
    for dirpath, dirnames, filenames in os.walk(root):
        if "saved_model.pb" in filenames:
            return dirpath
    return None

def _find_keras_file(root):
    pats = [
        os.path.join(root, "model.keras"),
        os.path.join(root, "model.h5"),
        os.path.join(root, "output", "model.keras"),
        os.path.join(root, "output", "model.h5"),
    ]
    for p in pats:
        if os.path.exists(p):
            return p
    # Any .keras/.h5 anywhere?
    hits = glob.glob(os.path.join(root, "**", "*.keras"), recursive=True)
    if hits:
        return hits[0]
    hits = glob.glob(os.path.join(root, "**", "*.h5"), recursive=True)
    return hits[0] if hits else None

def model_fn(model_dir: str):
    # Try native Keras format first
    kfile = _find_keras_file(model_dir)
    if kfile:
        return keras.models.load_model(kfile)

    # Try SavedModel
    smdir = _find_savedmodel(model_dir)
    if smdir:
        return tf.keras.models.load_model(smdir)

    # As a last resort, let Keras attempt to load the root (works if training saved there)
    try:
        return keras.models.load_model(model_dir)
    except Exception as e:
        raise RuntimeError(f"Could not locate a Keras or SavedModel artifact under {model_dir}: {e}")

def input_fn(request_body, content_type="text/csv"):
    if content_type == "text/csv":
        lines = [l for l in request_body.strip().splitlines() if l.strip()]
        rows = [list(map(float, ln.split(","))) for ln in lines]
        return np.asarray(rows, dtype=np.float32)
    if content_type == "application/json":
        obj = json.loads(request_body)
        # Support {"instances": [[...], ...]} or a raw [[...], ...]
        if isinstance(obj, dict) and "instances" in obj:
            obj = obj["instances"]
        return np.asarray(obj, dtype=np.float32)
    raise ValueError(f"Unsupported content type: {content_type}")

def predict_fn(input_data, model):
    # Works for tf.keras Models and tf.Module signatures
    preds = model(input_data, training=False) if callable(model) else model.predict(input_data, verbose=0)
    preds = np.array(preds).reshape((len(input_data), -1))
    # If single-output, flatten to [N]
    if preds.shape[1] == 1:
        preds = preds.ravel()
    # Clamp to [0,1] if these are probabilities
    preds = np.clip(preds, 0.0, 1.0)
    return preds

def output_fn(prediction, accept="text/csv"):
    arr = np.asarray(prediction)
    if accept == "text/csv":
        if arr.ndim == 1:
            return ",".join(f"{float(x):.6f}" for x in arr), "text/csv"
        # 2D -> one row per line
        lines = [";".join(f"{float(v):.6f}" for v in row) for row in arr]
        return "\n".join(lines), "text/csv"
    return json.dumps(arr.tolist()), "application/json"
