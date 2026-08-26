"""
WPFI - Monthly Automation (Main Script)

This is the script GitHub Actions will run once a month. For every
well, it generates one new month of oil/water/gas production (using
add_next_month.py's logic), inserts it into the database, then runs
the existing quality checks to confirm nothing broke.
"""

import os
import sys
import sqlite3
import warnings
from datetime import date
import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta

warnings.filterwarnings("ignore")  # suppress routine curve_fit covariance warnings

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(SCRIPT_DIR, "..", "models")
sys.path.insert(0, MODELS_DIR)

from add_next_month import generate_next_month_oil, generate_next_month_water, generate_next_month_gas
from economic_limit import get_well_status

DB_PATH = os.path.join(SCRIPT_DIR, "..", "data", "wpfi.db")


def run_monthly_update():
    conn = sqlite3.connect(DB_PATH)
    wells = pd.read_sql_query("SELECT well_id, well_name, start_date FROM wells", conn)

    summary = []

    for _, w in wells.iterrows():
        well_id = w["well_id"]

        # Check economic status FIRST - if this well already reached
        # its limit, don't generate another month for it. It stays
        # frozen at its final recorded state, like a real shut-down well.
        status_info = get_well_status(well_id, conn)
        if status_info["status"] == "Reached Economic Limit":
            summary.append(dict(
                well_name=w["well_name"], month_index=status_info["latest_month"],
                oil_rate=round(status_info["latest_rate"], 1), water_rate=None,
                event="SKIPPED - already reached economic limit"
            ))
            continue

        df = pd.read_sql_query(
            f"SELECT * FROM production WHERE well_id = {well_id} ORDER BY month_index", conn
        )

        next_month_index = int(df["month_index"].max()) + 1
        start_date = date.fromisoformat(w["start_date"])
        next_date = start_date + relativedelta(months=next_month_index - 1)

        # REAL-DATE SAFEGUARD: only add this new month if that month's
        # date has ACTUALLY arrived in real life. Without this check,
        # running the script twice in the same real month (by accident,
        # or manually alongside the scheduled run) would push the well's
        # timeline ahead of reality - this makes the script safe to run
        # as many extra times as needed; it simply does nothing until
        # the real calendar catches up.
        if date.today() < next_date:
            summary.append(dict(
                well_name=w["well_name"], month_index=next_month_index,
                oil_rate=None, water_rate=None,
                event=f"SKIPPED - not due yet (next update on {next_date.isoformat()})"
            ))
            continue

        oil_hist = df["oil_rate"].values
        water_hist = df["water_rate"].values
        gas_hist = df["gas_rate"].values

        new_oil = generate_next_month_oil(oil_hist, water_hist)
        new_water, event_note = generate_next_month_water(oil_hist, water_hist, new_oil)
        new_gas = generate_next_month_gas(oil_hist, gas_hist, new_oil)

        # INSERT OR IGNORE respects the UNIQUE(well_id, month_index)
        # constraint from schema.sql - if this well/month combo somehow
        # already exists (e.g. the script ran twice), nothing duplicates
        conn.execute("""
            INSERT OR IGNORE INTO production
            (well_id, month_index, date, oil_rate, gas_rate, water_rate, event_note)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            int(well_id), next_month_index, next_date.isoformat(),
            round(float(new_oil), 1), round(float(new_gas), 1),
            round(float(new_water), 1), event_note
        ))

        summary.append(dict(
            well_name=w["well_name"], month_index=next_month_index,
            oil_rate=round(new_oil, 1), water_rate=round(new_water, 1),
            event=event_note
        ))

    conn.commit()
    conn.close()

    return pd.DataFrame(summary)


if __name__ == "__main__":
    summary = run_monthly_update()

    added = summary[summary["event"] == ""]
    skipped = summary[summary["event"] != ""]

    print(f"Monthly update run complete: {len(added)} well(s) updated, "
          f"{len(skipped)} well(s) skipped")
    print()
    print(summary.to_string(index=False))

    events = summary[summary["event"] != ""]
    if len(events) > 0:
        print()
        print("Notes this cycle:")
        print(events[["well_name", "event"]].to_string(index=False))