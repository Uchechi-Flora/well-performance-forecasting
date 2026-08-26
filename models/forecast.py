"""
WPFI - Arps Forecasting

Takes the MOST RECENT segment from segmented_fit (the well's current
behavior) and projects it forward 12 months - genuinely predicting
months that haven't happened yet, using the same Arps formula, just
fed future time values instead of past ones.
"""

import os
import sys
import sqlite3
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from segmented_fit import segmented_fit
from arps_fit import arps_rate

DB_PATH = os.path.join(SCRIPT_DIR, "..", "data", "wpfi.db")
FORECAST_MONTHS = 12


def forecast_well(oil_rates, water_rates, forecast_months=FORECAST_MONTHS):
    """
    Runs segmented_fit to get the well's current (most recent) curve,
    then projects that curve forecast_months beyond the end of the
    known history.

    Returns the segmented_fit result PLUS a forecast array.
    """
    result = segmented_fit(oil_rates, water_rates)
    last_segment = result["segments"][-1]  # the most recent segment
    fit = last_segment["fit"]

    # How many months INTO this segment does the known history already
    # cover? We need this to know where the forecast should continue from.
    segment_length = last_segment["end_month"] - last_segment["start_month"] + 1

    # Future time steps: continue counting from where this segment's
    # own history left off
    future_t = np.arange(segment_length + 1, segment_length + 1 + forecast_months)
    forecast_values = arps_rate(future_t, fit["qi"], fit["Di"], max(fit["b"], 1e-6))

    # Build the actual future MONTH NUMBERS (continuing the well's
    # overall timeline, not the segment's private one) for readability
    total_history_length = len(oil_rates)
    future_month_numbers = np.arange(
        total_history_length + 1, total_history_length + 1 + forecast_months
    )

    result["forecast_months"] = future_month_numbers
    result["forecast_values"] = forecast_values
    return result


if __name__ == "__main__":
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(
        "SELECT * FROM production WHERE well_id = 13 ORDER BY month_index", conn
    )
    true_info = pd.read_sql_query(
        "SELECT well_name FROM wells WHERE well_id = 13", conn
    ).iloc[0]
    conn.close()

    result = forecast_well(df["oil_rate"].values, df["water_rate"].values)

    print(f"Well: {true_info['well_name']}")
    print(f"Forecasting {FORECAST_MONTHS} months beyond month {len(df)} "
          f"(using the most recent segment's curve)")
    print()
    print(f"{'Month':>6} {'Forecast Oil Rate':>18}")
    for m, v in zip(result["forecast_months"], result["forecast_values"]):
        print(f"{m:>6} {v:>18.1f}")