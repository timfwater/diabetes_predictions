import logging
from preprocessing import data_engineering, feature_selection, model_tuning, evaluate_model

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- File Paths ---
RAW_DATA_PATH = "legacy_outputs/Diabetes_Input.csv"
ENGINEERED_DATA_PATH = "legacy_outputs/processed_output.csv"
FEATURE_SELECTED_DATA_PATH = "legacy_outputs/features_selected.csv"
MODEL_PATH = "legacy_outputs/best_model.joblib"


def run_pipeline():
    logger.info("🔁 Starting full pipeline execution")

    # Step 1: Data Engineering
    df_cleaned = data_engineering.clean_data(
        data_engineering.load_data(RAW_DATA_PATH)
    )
    df_cleaned.to_csv(ENGINEERED_DATA_PATH, index=False)
    logger.info(f"✅ Saved cleaned data to: {ENGINEERED_DATA_PATH}")

    # Step 2: Feature Selection
    df_selected = feature_selection.select_features(df_cleaned)
    df_selected.to_csv(FEATURE_SELECTED_DATA_PATH, index=False)
    logger.info(f"✅ Saved selected features to: {FEATURE_SELECTED_DATA_PATH}")

    # Step 3: Model Tuning (local test only for now)
    model_tuning.run_local_model()

    # Step 4: Evaluation
    evaluate_model.evaluate_model()

    logger.info("🏁 Pipeline execution complete.")


if __name__ == "__main__":
    run_pipeline()
