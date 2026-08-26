"""
WPFI - Segmented Arps Fitting

Connects the two detectors (bump_detector, water_cut_detector) to the
Arps fitter (arps_fit). Instead of blindly fitting ONE curve across a
well's entire history (which we proved goes badly wrong across an
event boundary - see WELL-13's early test), this:

  1. Runs both detectors to find if/when something changed
  2. If nothing was flagged: fits one curve, same as before
  3. If something WAS flagged: splits the history at that month and
     fits a separate curve on each side

The "after" segment is what matters most for forecasting forward,
since it reflects the well's CURRENT behavior, not its outdated
pre-event behavior.
"""

import os
import sys
import sqlite3
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)  # so we can import our other model files

from arps_fit import fit_well
from bump_detector import detect_events
from water_cut_detector import detect_water_breakthrough

DB_PATH = os.path.join(SCRIPT_DIR, "..", "data", "wpfi.db")


def find_split_month(oil_rates, water_rates):
    """
    Runs both detectors and returns the EARLIEST flagged month, if any.
    Returns None if neither detector found anything.
    """
    bump_events = detect_events(oil_rates)
    water_events = detect_water_breakthrough(oil_rates, water_rates)

    all_events = bump_events + water_events
    if not all_events:
        return None, None

    # Sort by month, take the earliest one
    all_events.sort(key=lambda x: x[0])
    earliest_month, earliest_type = all_events[0]
    return earliest_month, earliest_type


def segmented_fit(oil_rates, water_rates):
    """
    Main function: fits either ONE curve (no event found) or TWO curves
    (split at the detected event) for a single well's history.

    Returns a dict describing what was done and the resulting fit(s).
    """
    split_month, event_type = find_split_month(oil_rates, water_rates)

    if split_month is None:
        # No event detected - fit one curve across everything, same as before
        result = fit_well(oil_rates)
        return dict(
            split_detected=False,
            split_month=None,
            event_type=None,
            segments=[dict(start_month=1, end_month=len(oil_rates), fit=result)]
        )

    # Event detected - split into "before" and "after"
    before = oil_rates[:split_month - 1]   # months 1 .. split_month-1

    # If the event is a SUDDEN DROP (e.g. shut-in), the months right at
    # and after the split may be genuine ZEROS - a real production gap,
    # not informative curve data. Arps cannot represent "flat zero,
    # then a jump back up" as a smooth decline, so we skip past any
    # leading zero-run and start the "after" segment fit from wherever
    # production actually RESUMES.
    after_start_index = split_month - 1
    while (after_start_index < len(oil_rates)
           and oil_rates[after_start_index] <= 0):
        after_start_index += 1

    after = oil_rates[after_start_index:]  # starts at resumed production, not the gap

    segments = []

    if len(before) >= 3:  # need at least a few points for a meaningful fit
        fit_before = fit_well(before)
        segments.append(dict(start_month=1, end_month=split_month - 1, fit=fit_before))

    if len(after) >= 3:
        fit_after = fit_well(after)
        segments.append(dict(
            start_month=after_start_index + 1, end_month=len(oil_rates), fit=fit_after
        ))
    else:
        # Not enough post-event data yet to fit a meaningful curve -
        # fall back to fitting the whole history rather than a
        # near-empty, unreliable segment
        fit_fallback = fit_well(oil_rates)
        segments.append(dict(start_month=1, end_month=len(oil_rates), fit=fit_fallback))

    return dict(
        split_detected=True,
        split_month=split_month,
        event_type=event_type,
        segments=segments
    )


if __name__ == "__main__":
    conn = sqlite3.connect(DB_PATH)

    # Test on WELL-13 - the same well whose single-curve fit went badly
    df = pd.read_sql_query(
        "SELECT * FROM production WHERE well_id = 13 ORDER BY month_index", conn
    )
    true_info = pd.read_sql_query(
        "SELECT well_name, complication_type, complication_month FROM wells WHERE well_id = 13", conn
    ).iloc[0]
    conn.close()

    result = segmented_fit(df["oil_rate"].values, df["water_rate"].values)

    print(f"Well: {true_info['well_name']}")
    print(f"TRUE event: {true_info['complication_type']} at month {true_info['complication_month']}")
    print(f"Split detected at month: {result['split_month']} (type: {result['event_type']})")
    print()
    for seg in result["segments"]:
        f = seg["fit"]
        print(f"Segment months {seg['start_month']}-{seg['end_month']}: "
              f"qi={f['qi']:.1f}  Di={f['Di']:.4f}  b={f['b']:.2f}")