"""
WPFI - Economic Limit Status

Determines whether a well is still comfortably Active, Approaching its
economic limit, or has genuinely Reached it - based on comparing its
MOST RECENT production against a threshold set at 10% of that well's
OWN original starting rate (qi_oil from the wells table). Using each
well's own starting rate keeps this fair across very different-sized
wells, rather than one fixed number for everyone.
"""

import os
import sqlite3
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(SCRIPT_DIR, "..", "data", "wpfi.db")

ECONOMIC_LIMIT_PCT = 0.10       # 10% of the well's own starting rate
APPROACHING_BUFFER_PCT = 0.20   # within 20% above the limit counts as "Approaching"


def get_well_status(well_id, conn):
    """
    Returns a dict describing one well's current economic status.
    """
    well = pd.read_sql_query(
        f"SELECT well_name, qi_oil FROM wells WHERE well_id = {well_id}", conn
    ).iloc[0]

    production = pd.read_sql_query(
        f"SELECT month_index, oil_rate FROM production WHERE well_id = {well_id} "
        f"ORDER BY month_index", conn
    )

    initial_rate = well["qi_oil"]
    latest_rate = production["oil_rate"].iloc[-1]
    latest_month = production["month_index"].iloc[-1]

    economic_limit = initial_rate * ECONOMIC_LIMIT_PCT
    approaching_ceiling = economic_limit * (1 + APPROACHING_BUFFER_PCT)

    if latest_rate <= economic_limit:
        status = "Reached Economic Limit"
    elif latest_rate <= approaching_ceiling:
        status = "Approaching Limit"
    else:
        status = "Active"

    return dict(
        well_id=well_id, well_name=well["well_name"],
        initial_rate=initial_rate, latest_rate=latest_rate,
        latest_month=latest_month, economic_limit=economic_limit,
        status=status
    )


def get_all_well_statuses():
    conn = sqlite3.connect(DB_PATH)
    wells = pd.read_sql_query("SELECT well_id FROM wells", conn)
    statuses = [get_well_status(wid, conn) for wid in wells["well_id"]]
    conn.close()
    return pd.DataFrame(statuses)


if __name__ == "__main__":
    df = get_all_well_statuses()
    print(df[["well_name", "initial_rate", "latest_rate", "economic_limit", "status"]]
          .round(1).to_string(index=False))
    print()
    print("Status counts:")
    print(df["status"].value_counts().to_string())