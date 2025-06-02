from preprocessing import data_engineering, feature_selection, model_tuning
import joblib
import os

INPUT_PATH = "legacy_outputs/Diabetes_Input.csv"

# SageMaker passes this environment variable during training jobs
model_output_dir = os.environ.get("SM_MODEL_DIR", "legacy_outputs")
MODEL_OUTPUT_PATH = os.path.join(model_output_dir, "model.joblib")

def main():
    print("🚀 Starting training pipeline")
    df = data_engineering.clean_data(data_engineering.load_data(INPUT_PATH))
    df = feature_selection.select_features(df)
    model = model_tuning.train_model(df)
    joblib.dump(model, MODEL_OUTPUT_PATH)
    print(f"✅ Model saved to {MODEL_OUTPUT_PATH}")

if __name__ == "__main__":
    main()
