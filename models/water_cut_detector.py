"""
WPFI - Water Cut Detector

Watches WATER CUT (the fraction of total liquid that is water) rather
than oil rate. Water cut normally creeps up slowly and steadily as a
well ages - this detector flags months where it jumps up much faster
than that normal, gradual pace, which signals water breakthrough.

This is a separate, second detector because bump_detector.py only
watches oil rate - and water breakthrough (in this project) does NOT
create a visible jump in oil rate, only in water rate. Two different
signals, two different detectors.
"""

import os
import sqlite3
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(SCRIPT_DIR, "..", "data", "wpfi.db")

# If water cut jumps by more than 3 percentage points in a single month,
# that's far beyond the normal ~1 point/month gradual aging trend
JUMP_THRESHOLD = 0.03


def detect_water_breakthrough(oil_rates, water_rates):
    """
    oil_rates, water_rates: arrays of monthly rates for ONE well, in order.
    Returns a list of (month_index, event_type) for every flagged month.
    """
    oil_rates = np.array(oil_rates)
    water_rates = np.array(water_rates)

    # Water cut = what fraction of TOTAL liquid (oil + water) is water
    total_liquid = oil_rates + water_rates
    water_cut = np.divide(
        water_rates, total_liquid,
        out=np.zeros_like(water_rates, dtype=float),
        where=total_liquid > 0
    )

    flagged = []
    for i in range(1, len(water_cut)):
        month_index = i + 1
        change = water_cut[i] - water_cut[i - 1]

        if change > JUMP_THRESHOLD:
            flagged.append((month_index, "water_cut_acceleration"))
            break  # only need the FIRST month this happens - that's our
                   # dividing line for segmentation, not an ongoing log

    return flagged


if __name__ == "__main__":
    conn = sqlite3.connect(DB_PATH)

    # Test on WELL-15, the water breakthrough well the bump detector missed
    df = pd.read_sql_query(
        "SELECT * FROM production WHERE well_id = 15 ORDER BY month_index", conn
    )
    true_info = pd.read_sql_query(
        "SELECT well_name, complication_type, complication_month FROM wells WHERE well_id = 15", conn
    ).iloc[0]
    conn.close()

    detected = detect_water_breakthrough(df["oil_rate"].values, df["water_rate"].values)

    print(f"Well: {true_info['well_name']}")
    print(f"TRUE event: {true_info['complication_type']} at month {true_info['complication_month']}")
    print(f"DETECTED events (found purely from water cut trend): {detected}")