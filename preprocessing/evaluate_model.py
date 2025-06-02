"""
This script implements the cleaned logic from `notebooks/evaluate_model.ipynb`.
For full exploratory justification and visualizations, refer to that notebook.
"""

import pandas as pd
import logging
import joblib
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def evaluate_model():
    logger.info("📊 Loading data and model for evaluation...")

    df = pd.read_csv("legacy_outputs/features_selected.csv")
    model = joblib.load("legacy_outputs/best_model.joblib")

    # 🔧 Ensure only numeric columns are used, as in training
    df = df.select_dtypes(include=["number", "bool"])
    
    X = df.drop("readmitted", axis=1)
    y = df["readmitted"]

    preds = model.predict(X)
    logger.info("✅ Evaluation report:")
    print(classification_report(y, preds))

    # Confusion matrix
    cm = confusion_matrix(y, preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
    disp.plot()
    plt.title("Confusion Matrix")
    plt.show()

    # Precision-Recall Curve
    probs = model.predict_proba(X)[:, 1]
    precision, recall, _ = precision_recall_curve(y, probs)
    avg_precision = average_precision_score(y, probs)

    plt.figure()
    plt.plot(recall, precision, label=f"Avg Precision = {avg_precision:.2f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    evaluate_model()

if __name__ == "__main__":
    main()
