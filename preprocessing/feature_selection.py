"""
This script implements the cleaned logic from `notebooks/feature_selection.ipynb`.
For full exploratory justification and visualizations, refer to that notebook.
"""

import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def select_features(df):
    logger.info(f"📊 Starting feature selection on shape: {df.shape}")

    # Drop columns with >50% missing values
    missing = df.isnull().mean()
    df = df.drop(columns=missing[missing > 0.5].index)
    logger.info(f"✅ Dropped high-missing columns. New shape: {df.shape}")

    # Drop columns with zero variance
    nunique = df.nunique()
    df = df.drop(columns=nunique[nunique == 1].index)
    logger.info(f"✅ Dropped zero-variance columns. New shape: {df.shape}")

    # (Optional) Drop any highly correlated features (placeholders for now)
    # corr = df.corr().abs()
    # upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    # to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
    # df = df.drop(columns=to_drop)
    # logger.info(f"✅ Dropped highly correlated features: {to_drop}")

    return df

def main():
    input_path = "legacy_outputs/processed_output.csv"
    output_path = "legacy_outputs/features_selected.csv"

    df = pd.read_csv(input_path)
    df = select_features(df)
    df.to_csv(output_path, index=False)

    logger.info(f"✅ Feature selection complete. Output saved to {output_path}")

if __name__ == "__main__":
    main()
