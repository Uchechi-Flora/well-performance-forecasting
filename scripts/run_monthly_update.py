
"""
WPFI - Monthly Automation (Main Script)

This is the script GitHub Actions will run once a month.

For every well:
1. Check whether the well has reached its economic limit.
2. Check which forecasting model was selected for the well.
3. Use Arps or XGBoost to predict the next month's oil rate.
4. Generate water and gas using the existing logic.
5. Insert the new month into the database.
6. Safely fall back to Arps if model selection is missing or invalid.

XGBoost is NOT retrained here. The already-trained xgboost_model.pkl
is loaded and used for prediction.
"""

import os
import sys
import sqlite3
import warnings
from datetime import date

import joblib
import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta

warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(SCRIPT_DIR, "..", "models")
sys.path.insert(0, MODELS_DIR)

from add_next_month import (
    generate_next_month_oil,
    generate_next_month_water,
    generate_next_month_gas,
)
from economic_limit import get_well_status

DB_PATH = os.path.join(SCRIPT_DIR, "..", "data", "wpfi.db")
XGBOOST_MODEL_PATH = os.path.join(MODELS_DIR, "xgboost_model.pkl")


# ---------------------------------------------------------------------------
# XGBOOST LIVE PREDICTION
# ---------------------------------------------------------------------------

def predict_next_month_xgboost(model, oil_hist):
    """
    Builds the same feature structure used during XGBoost training
    and predicts the next month's oil rate.

    Training defines, for a row at month_index M: lag_1/2/3 = oil at
    M-1/M-2/M-3, rolling_avg_3 = mean(M-1,M-2,M-3), target = oil at M+1.
    So to predict the next unseen month N, the row's own month_index
    must be M = N - 1, and the lags reach one month further back than
    "the 3 most recent known months."

    oil_hist is ordered oldest -> newest; oil_hist[-1] is month M
    (the most recent month we actually have data for, i.e. N - 1).
    """

    if len(oil_hist) < 4:
        raise ValueError(
            "XGBoost requires at least 4 months of oil history "
            "to construct the required lag features."
        )

    M = len(oil_hist) - 1  # the row's own month_index (N - 1)

    lag_1 = oil_hist[-2]
    lag_2 = oil_hist[-3]
    lag_3 = oil_hist[-4]

    feature_row = pd.DataFrame([{
        "well_id": None,  # filled in by caller
        "lag_1": lag_1,
        "lag_2": lag_2,
        "lag_3": lag_3,
        "rolling_avg_3": np.mean([lag_1, lag_2, lag_3]),
        "month_index": M,
    }])

    return feature_row
'''
def predict_next_month_xgboost(model, oil_hist):
    """
    Builds the same feature structure used during XGBoost training
    and predicts the next month's oil rate.

    Training features:
        well_id
        lag_1
        lag_2
        lag_3
        rolling_avg_3
        month_index

    For the next unseen month:
        lag_1 = most recent known month
        lag_2 = two months ago
        lag_3 = three months ago
        rolling_avg_3 = average of those three months
        month_index = most recent known month
    """

    if len(oil_hist) < 4:
        raise ValueError(
            "XGBoost requires at least 4 months of oil history "
            "to construct the required lag features."
        )

    latest_month = len(oil_hist)

    feature_row = pd.DataFrame([{
        "well_id": None,  # filled in by caller
        "lag_1": oil_hist[-1],
        "lag_2": oil_hist[-2],
        "lag_3": oil_hist[-3],
        "rolling_avg_3": np.mean(oil_hist[-3:]),
        "month_index": latest_month,
    }])

    return feature_row
'''

def predict_xgboost(model, well_id, oil_hist):
    """
    Predict the next month's oil rate using the saved XGBoost model.
    """

    features = predict_next_month_xgboost(model, oil_hist)

    features["well_id"] = well_id

    feature_cols = [
        "well_id",
        "lag_1",
        "lag_2",
        "lag_3",
        "rolling_avg_3",
        "month_index",
    ]

    prediction = model.predict(features[feature_cols])[0]

    # Production cannot be negative.
    return max(float(prediction), 0.0)


# ---------------------------------------------------------------------------
# MONTHLY AUTOMATION
# ---------------------------------------------------------------------------

def run_monthly_update():

    conn = sqlite3.connect(DB_PATH)

    wells = pd.read_sql_query(
        "SELECT well_id, well_name, start_date FROM wells",
        conn
    )

    # -----------------------------------------------------------------------
    # LOAD XGBOOST MODEL ONCE
    # -----------------------------------------------------------------------
    # We do NOT retrain XGBoost every month.
    # The model was trained separately and saved as xgboost_model.pkl.
    #
    # If the file cannot be loaded, we simply disable XGBoost for this
    # monthly cycle and safely fall back to Arps.
    # -----------------------------------------------------------------------

    xgboost_model = None

    if os.path.exists(XGBOOST_MODEL_PATH):
        try:
            xgboost_model = joblib.load(XGBOOST_MODEL_PATH)
            print("XGBoost model loaded successfully.")
        except Exception as e:
            print(
                f"WARNING: Could not load XGBoost model: {e}"
            )
            print("All wells will safely fall back to Arps this cycle.")
    else:
        print(
            "WARNING: xgboost_model.pkl not found."
        )
        print("All wells will safely fall back to Arps this cycle.")

    print()

    summary = []

    for _, w in wells.iterrows():

        well_id = int(w["well_id"])

        # -------------------------------------------------------------------
        # 1. ECONOMIC LIMIT CHECK
        # -------------------------------------------------------------------

        status_info = get_well_status(well_id, conn)

        if status_info["status"] == "Reached Economic Limit":

            summary.append(dict(
                well_name=w["well_name"],
                month_index=status_info["latest_month"],
                oil_rate=round(status_info["latest_rate"], 1),
                water_rate=None,
                model="N/A",
                event="SKIPPED - already reached economic limit"
            ))

            continue

        # -------------------------------------------------------------------
        # 2. GET WELL'S PRODUCTION HISTORY
        # -------------------------------------------------------------------

        df = pd.read_sql_query(
            """
            SELECT *
            FROM production
            WHERE well_id = ?
            ORDER BY month_index
            """,
            conn,
            params=(well_id,)
        )

        next_month_index = int(df["month_index"].max()) + 1

        start_date = date.fromisoformat(w["start_date"])

        next_date = start_date + relativedelta(
            months=next_month_index - 1
        )

        # -------------------------------------------------------------------
        # 3. REAL-DATE SAFEGUARD
        # -------------------------------------------------------------------

        if date.today() < next_date:

            summary.append(dict(
                well_name=w["well_name"],
                month_index=next_month_index,
                oil_rate=None,
                water_rate=None,
                model="N/A",
                event=(
                    f"SKIPPED - not due yet "
                    f"(next update on {next_date.isoformat()})"
                )
            ))

            continue

        oil_hist = df["oil_rate"].values
        water_hist = df["water_rate"].values
        gas_hist = df["gas_rate"].values

        # -------------------------------------------------------------------
        # 4. CHECK SELECTED MODEL
        # -------------------------------------------------------------------

        selection = pd.read_sql_query(
            """
            SELECT selected_model
            FROM model_selection
            WHERE well_id = ?
            """,
            conn,
            params=(well_id,)
        )

        # SAFE DEFAULT:
        # If there is no model-selection record, use Arps.
        selected_model = "arps"

        if not selection.empty:
            stored_model = str(
                selection.iloc[0]["selected_model"]
            ).strip().lower()

            if stored_model in ("arps", "xgboost"):
                selected_model = stored_model

        # If XGBoost was selected but the model isn't available,
        # safely fall back to Arps.
        if selected_model == "xgboost" and xgboost_model is None:
            selected_model = "arps"

        # -------------------------------------------------------------------
        # 5. PREDICT NEXT MONTH'S OIL
        # -------------------------------------------------------------------

        if selected_model == "xgboost":

            try:
                new_oil = predict_xgboost(
                    xgboost_model,
                    well_id,
                    oil_hist
                )

            except Exception as e:

                # XGBoost failure should NOT kill the entire monthly job.
                # Fall back to the existing Arps prediction.
                print(
                    f"WARNING: XGBoost prediction failed for "
                    f"{w['well_name']}: {e}"
                )
                print("Falling back to Arps for this well.")

                new_oil = generate_next_month_oil(
                    oil_hist,
                    water_hist
                )

                selected_model = "arps"

        else:

            new_oil = generate_next_month_oil(
                oil_hist,
                water_hist
            )

        # -------------------------------------------------------------------
        # 6. GENERATE WATER + GAS
        # -------------------------------------------------------------------
        # These functions don't care whether oil came from Arps or XGBoost.
        # They simply receive the predicted new oil rate.

        new_water, event_note = generate_next_month_water(
            oil_hist,
            water_hist,
            new_oil
        )

        new_gas = generate_next_month_gas(
            oil_hist,
            gas_hist,
            new_oil
        )

        # -------------------------------------------------------------------
        # 7. INSERT NEW MONTH
        # -------------------------------------------------------------------

        conn.execute(
            """
            INSERT OR IGNORE INTO production
            (well_id, month_index, date, oil_rate, gas_rate,
             water_rate, event_note)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                well_id,
                next_month_index,
                next_date.isoformat(),
                round(float(new_oil), 1),
                round(float(new_gas), 1),
                round(float(new_water), 1),
                event_note
            )
        )

        summary.append(dict(
            well_name=w["well_name"],
            month_index=next_month_index,
            oil_rate=round(float(new_oil), 1),
            water_rate=round(float(new_water), 1),
            model=selected_model.upper(),
            event=event_note
        ))

    conn.commit()
    conn.close()

    return pd.DataFrame(summary)


# ---------------------------------------------------------------------------
# SCRIPT ENTRY POINT
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    summary = run_monthly_update()

    added = summary[
        summary["event"] == ""
    ]

    skipped = summary[
        summary["event"] != ""
    ]

    print(
        f"Monthly update run complete: "
        f"{len(added)} well(s) updated, "
        f"{len(skipped)} well(s) skipped"
    )

    print()

    print(summary.to_string(index=False))

    events = summary[
        summary["event"] != ""
    ]

    if len(events) > 0:

        print()
        print("Notes this cycle:")

        print(
            events[
                ["well_name", "model", "event"]
            ].to_string(index=False)
        )
