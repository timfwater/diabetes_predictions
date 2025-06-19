import os
import xgboost as xgb
import logging
import traceback

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    try:
        logger.info("🚀 Starting training script...")

        train_path = "/opt/ml/input/data/train/train_proto.data"
        logger.info(f"📂 Checking if training data exists: {train_path}")

        if not os.path.exists(train_path):
            logger.error("❌ File does not exist!")
        else:
            logger.info("✅ File exists.")

        try:
            with open(train_path, "rb") as f:
                logger.info("📖 File successfully opened.")
                logger.info(f"🔍 First 64 bytes: {f.read(64)}")
        except Exception as open_err:
            logger.error("❌ Failed to open file:")
            traceback.print_exc()

        try:
            dtrain = xgb.DMatrix(train_path)
            logger.info("✅ DMatrix loaded successfully.")
        except Exception as dmex:
            logger.error("💥 Failed to load DMatrix from protobuf.")
            traceback.print_exc()
            raise dmex  # This was previously the broken line

        # Train model
        params = {
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "max_depth": int(os.environ.get("max_depth", 5)),
            "eta": float(os.environ.get("eta", 0.2)),
            "min_child_weight": int(os.environ.get("min_child_weight", 1))
        }
        num_round = int(os.environ.get("num_round", 100))

        evals_result = {}
        bst = xgb.train(params, dtrain, num_boost_round=num_round,
                        evals=[(dtrain, "train")], evals_result=evals_result)

        auc_scores = evals_result.get("train", {}).get("auc", [])
        if auc_scores:
            logger.info(f"✅ Final training AUC: {auc_scores[-1]}")
            print(f"validation:auc\t{auc_scores[-1]}")

        model_dir = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, "xgboost-model")
        bst.save_model(model_path)
        logger.info(f"📦 Model saved to: {model_path}")

    except Exception as e:
        logger.error("💥 Unhandled exception during training!")
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main()
