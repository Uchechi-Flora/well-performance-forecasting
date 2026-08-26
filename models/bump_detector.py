"""
WPFI - Event / Bump Detector

Scans a well's monthly oil production and flags months where something
unexpected happened - specifically:
  1. An unexpected RISE (production going up when it should be declining)
     -> signals a workover, recompletion, or similar operational event
  2. A sudden DROP toward zero
     -> signals a shut-in

This works from the numbers alone - it does NOT look at the event_note
column. That column is only used afterward, separately, to check how
well this detector actually performed.
"""

import os
import sqlite3
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(SCRIPT_DIR, "..", "data", "wpfi.db")

RISE_THRESHOLD = 0.12     # flag if production rises more than 12% month-over-month
SHUT_IN_THRESHOLD = 0.90  # flag if production drops more than 90% (near-zero)


def detect_events(oil_rates):
    """
    oil_rates: array of monthly oil rates for ONE well, in order.
    Returns a list of (month_index, event_type) for every flagged month.
    month_index is 1-based to match the rest of the project.
    """
    oil_rates = np.array(oil_rates)
    flagged = []

    for i in range(1, len(oil_rates)):
        prev = oil_rates[i - 1]
        curr = oil_rates[i]
        month_index = i + 1  # the CURRENT month is where we flag the change

        if prev <= 0:
            continue  # can't compute a meaningful % change from zero

        pct_change = (curr - prev) / prev

        if pct_change > RISE_THRESHOLD:
            flagged.append((month_index, "unexpected_rise"))
        elif pct_change < -SHUT_IN_THRESHOLD:
            flagged.append((month_index, "sudden_drop"))

    return flagged


if __name__ == "__main__":
    # Standalone test on WELL-13 (the workover well) - same well we used
    # to test the fitter, so we can directly compare
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(
        "SELECT * FROM production WHERE well_id = 13 ORDER BY month_index", conn
    )
    true_info = pd.read_sql_query(
        "SELECT well_name, complication_type, complication_month FROM wells WHERE well_id = 13", conn
    ).iloc[0]
    conn.close()

    detected = detect_events(df["oil_rate"].values)

    print(f"Well: {true_info['well_name']}")
    print(f"TRUE event (from generation, not shown to the detector): "
          f"{true_info['complication_type']} at month {true_info['complication_month']}")
    print(f"DETECTED events (found purely from the numbers): {detected}")
