"""
WPFI - XGBoost Training & Evaluation
Trains XGBoost on the engineered feature table, using a TIME-BASED
split (not random shuffling) - train on earlier months, test on later
months the model never saw. This honestly mimics real forecasting,
where you never actually have future data during training.
"""
import os
import sys
import sqlite3
import numpy as np
import pandas as pd
import joblib
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from feature_engineering import build_training_table
DB_PATH = os.path.join(SCRIPT_DIR, "..", "data", "wpfi.db")
MODEL_PATH = os.path.join(SCRIPT_DIR, "xgboost_model.pkl")
FEATURE_COLS = ["well_id", "lag_1", "lag_2", "lag_3", "rolling_avg_3", "month_index"]
TARGET_COL = "target"
# Time-based split: train on rows predicting month 18 or earlier,
# test on rows predicting month 19 or later (these months' TARGETS
# are what we're splitting on - not the row's own month_index)
SPLIT_MONTH = 18
def train_and_evaluate():
    conn = sqlite3.connect(DB_PATH)
    data = build_training_table(conn)
    conn.close()
    # The target for a row at month_index M is month M+1's actual value.
    # So "target month" = month_index + 1.
    data["target_month"] = data["month_index"] + 1
    train = data[data["target_month"] <= SPLIT_MONTH]
    test = data[data["target_month"] > SPLIT_MONTH]
    X_train, y_train = train[FEATURE_COLS], train[TARGET_COL]
    X_test, y_test = test[FEATURE_COLS], test[TARGET_COL]
    model = XGBRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=4,
        random_state=42
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    mape = mean_absolute_percentage_error(y_test, predictions)
    return dict(
        model=model,
        train_size=len(train),
        test_size=len(test),
        mae=mae,
        mape=mape,
        test_df=test.assign(predicted=predictions)
    )
if __name__ == "__main__":
    result = train_and_evaluate()
    print(f"Training rows (target month <= {SPLIT_MONTH}): {result['train_size']}")
    print(f"Testing rows (target month > {SPLIT_MONTH}):  {result['test_size']}")
    print()
    print(f"Mean Absolute Error (MAE): {result['mae']:.2f} barrels")
    print(f"Mean Absolute Percentage Error (MAPE): {result['mape']:.2%}")
    print()
    print("Sample predictions vs actual (5 random test rows):")
    sample = result["test_df"].sample(5, random_state=1)
    print(sample[["well_id", "month_index", "target", "predicted"]].to_string(index=False))

    # Save the trained model so the monthly prediction step can load
    # it directly, without retraining.
    joblib.dump(result["model"], MODEL_PATH)
    print()
    print(f"Model saved to: {MODEL_PATH}")