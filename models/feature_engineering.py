"""
WPFI - Feature Engineering for XGBoost

Turns raw monthly production history into the input "clues" (features)
XGBoost needs. XGBoost has no built-in sense of time/sequence - unlike
Arps (which has time baked into its formula) XGBoost only sees whatever columns we hand it.
So WE have to manually create columns that capture recent history.
"""

import os
import sqlite3
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(SCRIPT_DIR, "..", "data", "wpfi.db")


def build_features_for_well(df):
    """
    df: a well's production history, ordered by month_index, with at
    least an 'oil_rate' column.
    Returns the same dataframe with new feature columns added.
    """
    df = df.copy().sort_values("month_index").reset_index(drop=True)

    # LAG FEATURES: what was production N months ago?
    # .shift(1) moves every value DOWN by one row - so row for month 5
    # now shows month 4's value in the lag_1 column. That's exactly
    # "1 month ago" from that row's perspective.
    df["lag_1"] = df["oil_rate"].shift(1)
    df["lag_2"] = df["oil_rate"].shift(2)
    df["lag_3"] = df["oil_rate"].shift(3)

    # ROLLING AVERAGE: the mean of the last 3 known months (excluding
    # the current month itself, so we're not leaking the answer)
    df["rolling_avg_3"] = df["oil_rate"].shift(1).rolling(window=3).mean()

    # TARGET: what we want to predict - NEXT month's oil rate.
    # .shift(-1) moves values UP by one row, so this row's target
    # becomes whatever oil_rate actually was the following month.
    df["target"] = df["oil_rate"].shift(-1)

    return df


def build_training_table(conn):
    """
    Builds features for ALL wells, keeping each well's lag calculations
    separate (never borrowing across wells), then combines everything
    into one table and drops incomplete rows.

    Returns one clean dataframe ready to hand to XGBoost.
    """
    wells = pd.read_sql_query("SELECT well_id FROM wells", conn)
    all_features = []

    for well_id in wells["well_id"]:
        well_df = pd.read_sql_query(
            f"SELECT * FROM production WHERE well_id = {well_id} ORDER BY month_index", conn
        )
        featured = build_features_for_well(well_df)
        all_features.append(featured)

    # Stack every well's feature table on top of each other into one
    combined = pd.concat(all_features, ignore_index=True)

    # Drop any row missing a lag value OR a target - these can't be
    # used for training (early months / each well's last month)
    feature_cols = ["lag_1", "lag_2", "lag_3", "rolling_avg_3", "target"]
    clean = combined.dropna(subset=feature_cols).reset_index(drop=True)

    return clean


if __name__ == "__main__":
    conn = sqlite3.connect(DB_PATH)

    # First, the single-well test we already did (kept for reference)
    df = pd.read_sql_query(
        "SELECT * FROM production WHERE well_id = 1 ORDER BY month_index", conn
    )
    features = build_features_for_well(df)
    print("WELL-01 with engineered features (first 8 months):")
    print(features[["month_index", "oil_rate", "lag_1", "lag_2", "lag_3",
                     "rolling_avg_3", "target"]].head(8).to_string(index=False))

    print()
    print("=" * 60)
    print()

    # Now the full combined table across ALL 18 wells
    training_table = build_training_table(conn)
    conn.close()

    print(f"Combined training table: {len(training_table)} rows "
          f"(from 18 wells x 24 months = 432 raw rows, minus incomplete edges)")
    print()
    print("Sample rows from different wells:")
    print(training_table[["well_id", "month_index", "oil_rate", "lag_1",
                           "lag_2", "lag_3", "rolling_avg_3", "target"]].sample(6, random_state=1).to_string(index=False))