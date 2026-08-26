"""
WPFI - Arps Curve Fitting

Given only a well's raw monthly oil production history (no knowledge of
the true parameters that generated it), this fits the best Arps curve
(qi, Di, b) to explain that history - exactly what a real petroleum
engineer would do with real field data.
"""

import os
import sqlite3
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(SCRIPT_DIR, "..", "data", "wpfi.db")


def arps_rate(t, qi, Di, b):
    """
    The Arps equation, written so scipy can use it for fitting.
    b is clipped to a tiny positive number instead of exactly 0, because
    the general hyperbolic formula divides by b - as b approaches 0 it
    smoothly approaches the exponential case anyway, so this avoids a
    divide-by-zero error while still letting the fit find b close to 0.
    """
    b_safe = max(b, 1e-6)
    return qi / ((1 + b_safe * Di * t) ** (1 / b_safe))


def fit_well(oil_rates):
    """
    oil_rates: a plain list/array of monthly oil rates for ONE well,
    in order (month 1, month 2, ...).
    Returns the best-fit (qi, Di, b) and the fitted curve's predicted
    values, so we can compare fit-vs-actual.
    """
    t = np.arange(1, len(oil_rates) + 1)
    y = np.array(oil_rates)

    # Starting guesses for the optimizer - qi starts at the first
    # observed value (or a small positive floor if that's ~0), Di and
    # b start at modest middle-of-the-road guesses
    initial_guess = [max(y[0], 1.0), 0.05, 0.3]

    # Bounds keep the fit within physically sensible territory. We use
    # max(y[0]*2, 1.0) instead of just y[0]*2 - if a segment happens to
    # start at or near zero (e.g. a shut-in well cut short by a time
    # window), y[0]*2 would also be ~0, making the upper bound equal
    # to the lower bound, which curve_fit rejects as invalid.
    bounds = ([0, 0.001, 0.0001], [max(y[0] * 2, 1.0), 0.5, 1.0])

    try:
        params, _ = curve_fit(arps_rate, t, y, p0=initial_guess, bounds=bounds, maxfev=5000)
        qi_fit, Di_fit, b_fit = params
        predicted = arps_rate(t, *params)
        return dict(qi=qi_fit, Di=Di_fit, b=b_fit, predicted=predicted, success=True)
    except RuntimeError:
        # curve_fit failed to converge - flag it rather than crash
        return dict(qi=None, Di=None, b=None, predicted=None, success=False)


if __name__ == "__main__":
    # Quick standalone test: fit ONE well and see how close we get
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(
        "SELECT * FROM production WHERE well_id = 1 ORDER BY month_index", conn
    )
    true_params = pd.read_sql_query(
        "SELECT well_name, decline_type, qi_oil, Di, b FROM wells WHERE well_id = 1", conn
    ).iloc[0]
    conn.close()

    result = fit_well(df["oil_rate"].values)

    print(f"Well: {true_params['well_name']} ({true_params['decline_type']})")
    print(f"TRUE parameters:   qi={true_params['qi_oil']:.1f}  Di={true_params['Di']:.4f}  b={true_params['b']:.2f}")
    print(f"FITTED parameters: qi={result['qi']:.1f}  Di={result['Di']:.4f}  b={result['b']:.2f}")