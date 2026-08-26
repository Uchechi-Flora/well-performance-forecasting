"""
WPFI - Model Selection

Runs the model comparison ONCE and saves, per well, which model won.
This becomes a permanent record - the automation pipeline reads FROM
this table going forward, rather than re-running models every cycle.

NOTE: LSTM is intentionally NOT part of this ongoing comparison. It
was tested separately (see train_lstm.py / compare_models.py) and
never won a single well - that result is documented in the article.
Re-training a neural network here every time this script runs would
be wasted computation for a model already shown not to be selected.
Only Arps and XGBoost - the two models that DO get selected - are
compared here.
"""

import os
import sys
import sqlite3
from datetime import datetime, timezone
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from compare_models import compare_all_wells

DB_PATH = os.path.join(SCRIPT_DIR, "..", "data", "wpfi.db")


def run_model_selection():
    """
    Runs Arps vs XGBoost on every well and determines the winner per
    well by lowest MAPE.
    """
    results = compare_all_wells()  # already has arps_mape, xgboost_mape, winner

    results["selected_model"] = results["winner"]
    results["selected_mape"] = results[["arps_mape", "xgboost_mape"]].min(axis=1)

    return results


def save_model_selection(results_df):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS model_selection (
            well_id         INTEGER PRIMARY KEY,
            selected_model  TEXT NOT NULL,
            selected_mape   REAL NOT NULL,
            arps_mape       REAL NOT NULL,
            xgboost_mape    REAL NOT NULL,
            evaluated_at    TEXT NOT NULL,
            FOREIGN KEY (well_id) REFERENCES wells(well_id)
        )
    """)

    timestamp = datetime.now(timezone.utc).isoformat()
    for _, row in results_df.iterrows():
        conn.execute("""
            INSERT OR REPLACE INTO model_selection
            (well_id, selected_model, selected_mape, arps_mape, xgboost_mape, evaluated_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            int(row["well_id"]), row["selected_model"], row["selected_mape"],
            row["arps_mape"], row["xgboost_mape"], timestamp
        ))

    conn.commit()
    conn.close()


if __name__ == "__main__":
    results = run_model_selection()
    save_model_selection(results)

    print("Model selection saved to database (model_selection table)")
    print()
    print(results[["well_name", "arps_mape", "xgboost_mape", "selected_model"]]
          .round(2).to_string(index=False))
    print()
    print("Selected model counts (this is what the automation pipeline will use going forward):")
    print(results["selected_model"].value_counts().to_string())