"""
WPFI - Estimated Ultimate Recovery (EUR)

EUR = cumulative production so far + forecasted future production,
projected forward until the well's rate crosses its own economic
limit (reusing forecast.py and economic_limit.py - no new modeling,
just combining what's already built).
"""

import os
import sys
import sqlite3
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from forecast import forecast_well
from economic_limit import ECONOMIC_LIMIT_PCT

MAX_FORECAST_MONTHS = 240  # 20-year safety cap, so a very slow-declining
                            # (e.g. harmonic) well can't forecast forever


def estimate_eur(well_id, conn):
    well = pd.read_sql_query(f"SELECT qi_oil FROM wells WHERE well_id = {well_id}", conn).iloc[0]
    production = pd.read_sql_query(
        f"SELECT oil_rate, water_rate FROM production WHERE well_id = {well_id} ORDER BY month_index", conn
    )

    cumulative_to_date = production["oil_rate"].sum()
    economic_limit = well["qi_oil"] * ECONOMIC_LIMIT_PCT

    # Project far enough forward to be sure we cross the limit, then
    # only SUM the months that are still genuinely above it - the rest
    # of the projection (if any) is discarded, since the well would
    # have been shut in by then.
    result = forecast_well(
        production["oil_rate"].values, production["water_rate"].values,
        forecast_months=MAX_FORECAST_MONTHS
    )
    forecast_values = result["forecast_values"]

    above_limit = forecast_values[forecast_values > economic_limit]
    remaining_to_limit = above_limit.sum()
    months_to_limit = len(above_limit)

    hit_cap = months_to_limit == MAX_FORECAST_MONTHS  # never crossed within 20 years

    return dict(
        well_id=well_id,
        cumulative_to_date=round(cumulative_to_date, 1),
        remaining_to_limit=round(remaining_to_limit, 1),
        eur_total=round(cumulative_to_date + remaining_to_limit, 1),
        months_to_limit=months_to_limit,
        hit_cap=hit_cap
    )


if __name__ == "__main__":
    conn = sqlite3.connect(os.path.join(SCRIPT_DIR, "..", "data", "wpfi.db"))
    wells = pd.read_sql_query("SELECT well_id, well_name FROM wells", conn)

    rows = []
    for _, w in wells.iterrows():
        eur = estimate_eur(int(w["well_id"]), conn)
        eur["well_name"] = w["well_name"]
        rows.append(eur)
    conn.close()

    df = pd.DataFrame(rows)
    print(df[["well_name", "cumulative_to_date", "remaining_to_limit",
              "eur_total", "months_to_limit", "hit_cap"]].to_string(index=False))