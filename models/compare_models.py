"""
WPFI - Arps vs XGBoost Comparison

For EVERY well, both models are given only months 1-18 and asked to
predict months 19-24 (which we already know the real answer to, but
hid from both models). Whichever model's MAPE is lower for a given
well WINS - and that's the model whose forecast gets shown to the
user for that specific well.
"""

import os
import sys
import sqlite3
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from forecast import forecast_well
from train_xgboost import train_and_evaluate

DB_PATH = os.path.join(SCRIPT_DIR, "..", "data", "wpfi.db")
SPLIT_MONTH = 18


def mape(actual, predicted):
    actual, predicted = np.array(actual), np.array(predicted)
    return np.mean(np.abs((actual - predicted) / actual)) * 100


def compare_all_wells():
    conn = sqlite3.connect(DB_PATH)
    wells = pd.read_sql_query("SELECT well_id, well_name FROM wells", conn)

    # Get XGBoost's per-well test predictions (already computed on the
    # same held-out months 19-24 via the time-based split)
    xgb_result = train_and_evaluate()
    xgb_test = xgb_result["test_df"]

    comparison_rows = []

    for _, w in wells.iterrows():
        well_id = w["well_id"]
        df = pd.read_sql_query(
            f"SELECT * FROM production WHERE well_id = {well_id} ORDER BY month_index", conn
        )
        oil_full = df["oil_rate"].values
        water_full = df["water_rate"].values

        # ARPS: fit using ONLY months 1-18, forecast the next 6 (19-24)
        oil_train = oil_full[:SPLIT_MONTH]
        water_train = water_full[:SPLIT_MONTH]
        actual_test = oil_full[SPLIT_MONTH:]  # months 19-24, real values

        arps_result = forecast_well(oil_train, water_train, forecast_months=len(actual_test))
        arps_predicted = arps_result["forecast_values"]
        arps_mape = mape(actual_test, arps_predicted)

        # XGBOOST: pull this well's rows from the test set already computed
        well_xgb_rows = xgb_test[xgb_test["well_id"] == well_id]
        if len(well_xgb_rows) > 0:
            xgb_mape_this_well = mape(well_xgb_rows["target"], well_xgb_rows["predicted"])
        else:
            xgb_mape_this_well = np.nan

        winner = "Arps" if arps_mape < xgb_mape_this_well else "XGBoost"

        comparison_rows.append(dict(
            well_id=well_id, well_name=w["well_name"],
            arps_mape=arps_mape, xgboost_mape=xgb_mape_this_well,
            winner=winner
        ))

    conn.close()
    return pd.DataFrame(comparison_rows)


if __name__ == "__main__":
    results = compare_all_wells()

    print(results.to_string(index=False))
    print()
    print("Winner counts:")
    print(results["winner"].value_counts().to_string())