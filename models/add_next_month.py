"""
WPFI - Monthly Automation: Add Next Month

Generates ONE new month of production for a well, continuing its
decline using its MOST RECENT fitted curve (via segmented_fit) rather
than the original generation-time parameters. This matters because,
months from now, a well's true current behavior may have drifted from
its original "recipe" - we want automation to follow what the data
actually shows NOW, the same way a real analyst would refit a model
on fresh data rather than trusting an old assumption forever.
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

NOISE_STD = 0.04  # same +/-4% noise used in the original data generation
REMEDIATION_CHANCE = 0.08       # 8% chance per month, IF water cut is elevated
REMEDIATION_THRESHOLD = 0.50    # water cut must be above this to be eligible for remediation
REMEDIATION_REDUCTION = (0.12, 0.22)  # water cut drops by a random amount in this range


def generate_next_month_oil(oil_history, water_history):
    """
    Given a well's full history so far, fits its CURRENT curve (most
    recent segment) and projects exactly one month beyond it.
    Returns the new oil rate (with the same realistic noise as before).
    """
    result = segmented_fit(oil_history, water_history)
    last_segment = result["segments"][-1]
    fit = last_segment["fit"]

    segment_length = last_segment["end_month"] - last_segment["start_month"] + 1
    next_t = segment_length + 1  # one step beyond this segment's known history

    predicted = arps_rate(next_t, fit["qi"], fit["Di"], max(fit["b"], 1e-6))
    noisy_predicted = predicted * np.random.normal(1.0, NOISE_STD)
    return max(noisy_predicted, 0)


def generate_next_month_water(oil_history, water_history, new_oil_rate):
    """
    Continues the well's water cut trend forward by one month, with a
    small chance of a REMEDIATION event if water cut is already
    elevated - reflecting real interventions (water shut-off
    treatments, recompletions) that can genuinely reduce water cut,
    not just let it climb forever.

    Returns (new_water_rate, event_note)
    """
    oil_history = np.array(oil_history)
    water_history = np.array(water_history)
    total_liquid = oil_history + water_history
    water_cut_history = np.divide(
        water_history, total_liquid,
        out=np.zeros_like(water_history, dtype=float),
        where=total_liquid > 0
    )

    last_water_cut = water_cut_history[-1]

    # Recent trend: average month-over-month change over the last 3 months.
    # We floor this at a small positive value (0.005) so that a ONE-TIME
    # remediation drop never gets mistaken for an ongoing declining
    # trend and extrapolated forever - after an intervention, water cut
    # should resume its normal slow climb, not keep crashing toward zero.
    recent_changes = np.diff(water_cut_history[-4:])  # up to 3 changes
    raw_trend = np.mean(recent_changes) if len(recent_changes) > 0 else 0.01
    trend = max(raw_trend, 0.005)

    event_note = ""

    # Check for a remediation event FIRST, before applying the normal trend
    if last_water_cut > REMEDIATION_THRESHOLD and np.random.random() < REMEDIATION_CHANCE:
        reduction = np.random.uniform(*REMEDIATION_REDUCTION)
        next_water_cut = max(last_water_cut - reduction, 0.05)
        event_note = "Water shut-off treatment applied"
    else:
        next_water_cut = min(last_water_cut + trend, 0.95)

    total_next_liquid = new_oil_rate / max(1 - next_water_cut, 0.05)
    new_water_rate = max(total_next_liquid - new_oil_rate, 0)

    return new_water_rate, event_note


def generate_next_month_gas(oil_history, gas_history, new_oil_rate):
    """
    Continues the gas-oil ratio (GOR) trend forward by one month -
    same idea as water, just following the GOR instead of water cut.
    """
    oil_history = np.array(oil_history)
    gas_history = np.array(gas_history)
    gor_history = np.divide(
        gas_history, oil_history,
        out=np.zeros_like(gas_history, dtype=float),
        where=oil_history > 0
    )

    recent_gor = np.mean(gor_history[-3:])  # average of the last 3 months' GOR
    new_gas_rate = new_oil_rate * recent_gor * np.random.normal(1.0, NOISE_STD)
    return max(new_gas_rate, 0)


if __name__ == "__main__":
    conn = sqlite3.connect(DB_PATH)

    # Test 1: WELL-01, oil only (as before)
    df1 = pd.read_sql_query(
        "SELECT * FROM production WHERE well_id = 1 ORDER BY month_index", conn
    )
    last_known = df1["oil_rate"].iloc[-1]
    next_value = generate_next_month_oil(df1["oil_rate"].values, df1["water_rate"].values)
    print(f"WELL-01 last known month ({len(df1)}): {last_known:.1f}")
    print(f"WELL-01 generated next month ({len(df1)+1}): {next_value:.1f}")
    print(f"Change: {(next_value/last_known - 1)*100:.2f}%")

    print()
    print("=" * 60)
    print()

    # Test 2: WELL-15, water breakthrough well - simulate 12 NEW months
    # forward to see whether the remediation logic ever kicks in
    df15 = pd.read_sql_query(
        "SELECT * FROM production WHERE well_id = 15 ORDER BY month_index", conn
    )
    conn.close()

    oil_hist = list(df15["oil_rate"].values)
    water_hist = list(df15["water_rate"].values)
    gas_hist = list(df15["gas_rate"].values)

    print("WELL-15 (water breakthrough well) - simulating 12 future months:")
    print(f"{'Month':>6} {'Oil':>8} {'Water':>8} {'WaterCut':>9} {'Event'}")
    for i in range(12):
        new_oil = generate_next_month_oil(oil_hist, water_hist)
        new_water, event_note = generate_next_month_water(oil_hist, water_hist, new_oil)
        new_gas = generate_next_month_gas(oil_hist, gas_hist, new_oil)

        water_cut = new_water / (new_oil + new_water) if (new_oil + new_water) > 0 else 0
        month_num = len(oil_hist) + 1
        print(f"{month_num:>6} {new_oil:>8.1f} {new_water:>8.1f} {water_cut:>9.2%} {event_note}")

        oil_hist.append(new_oil)
        water_hist.append(new_water)
        gas_hist.append(new_gas)