from preprocessing import data_engineering, feature_selection, model_tuning
import joblib
import os
import argparse
from sklearn.metrics import mean_squared_error
import math

INPUT_PATH = "Diabetes_Input.csv"

# SageMaker passes this environment variable during training jobs
model_output_dir = os.environ.get("SM_MODEL_DIR", "legacy_outputs")
MODEL_OUTPUT_PATH = os.path.join(model_output_dir, "model.joblib")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_depth", type=int, default=5)
    parser.add_argument("--eta", type=float, default=0.1)
    parser.add_argument("--gamma", type=float, default=0)
    parser.add_argument("--min_child_weight", type=int, default=1)
    parser.add_argument("--subsample", type=float, default=1.0)
    return parser.parse_args()

def main():
    args = parse_args()

    # Data loading and preprocessing
    df = data_engineering.clean_data(data_engineering.load_data(INPUT_PATH))
    df = feature_selection.select_features(df)

    # Train model
    model = model_tuning.train_model(
        df,
        max_depth=args.max_depth,
        eta=args.eta,
        gamma=args.gamma,
        min_child_weight=args.min_child_weight,
        subsample=args.subsample
    )

    # Save model
    joblib.dump(model, MODEL_OUTPUT_PATH)
    print("✅ Model saved to", MODEL_OUTPUT_PATH)

    # Compute and print RMSE
    X = df.drop("readmitted", axis=1)
    y = df["readmitted"]
    y_pred = model.predict(X)
    rmse = math.sqrt(mean_squared_error(y, y_pred))
    print(f"validation:rmse={rmse:.4f}")  # This line enables SageMaker to track the metric

if __name__ == "__main__":
    main()
