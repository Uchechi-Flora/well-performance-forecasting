"""
WPFI - Data Quality Checks

Runs 5 automated checks against the wells/production tables:
completeness, uniqueness, validity, referential integrity, timeliness.
Combines them into one composite score, decides PASS/FAIL, and logs
the result into dq_results so quality can be trended over time.
"""

import os
import sqlite3
from datetime import datetime, timezone
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(SCRIPT_DIR, "..", "data", "wpfi.db")

EXPECTED_MONTHS = 24  # how many months each well SHOULD have right now

conn = sqlite3.connect(DB_PATH)
wells = pd.read_sql_query("SELECT * FROM wells", conn)
production = pd.read_sql_query("SELECT * FROM production", conn)

issues = []  # collect plain-English notes about anything found wrong

# ---------------------------------------------------------------------------
# CHECK 1: COMPLETENESS
# Does every well have exactly the expected number of monthly rows?
# A well missing months means gaps in its history - a real problem for
# forecasting, since a model can't learn from data that isn't there.
# ---------------------------------------------------------------------------
rows_per_well = production.groupby("well_id").size()
wells_with_gaps = rows_per_well[rows_per_well < EXPECTED_MONTHS]

completeness_score = 1 - (len(wells_with_gaps) / len(wells))
if len(wells_with_gaps) > 0:
    issues.append(f"{len(wells_with_gaps)} well(s) missing expected months: "
                   f"{list(wells_with_gaps.index)}")

# ---------------------------------------------------------------------------
# CHECK 2: UNIQUENESS
# Is there ever more than one row for the same well + same month?
# Duplicates would double-count production and silently break any
# forecast trained on this data.
# ---------------------------------------------------------------------------
duplicate_count = production.duplicated(subset=["well_id", "month_index"]).sum()
uniqueness_score = 1 - (duplicate_count / len(production)) if len(production) > 0 else 1.0
if duplicate_count > 0:
    issues.append(f"{duplicate_count} duplicate well/month row(s) found")

# ---------------------------------------------------------------------------
# CHECK 3: VALIDITY
# Are the values actually sensible? No negative production, nothing
# impossibly large (a sign of a calculation bug, not a real well).
# ---------------------------------------------------------------------------
invalid_mask = (
    (production["oil_rate"] < 0) | (production["gas_rate"] < 0) | (production["water_rate"] < 0)
    | (production["oil_rate"] > 5000)  # sanity ceiling - no well in this dataset should exceed this
)
invalid_count = invalid_mask.sum()
validity_score = 1 - (invalid_count / len(production)) if len(production) > 0 else 1.0
if invalid_count > 0:
    issues.append(f"{invalid_count} row(s) with invalid (negative or unrealistic) values")

# ---------------------------------------------------------------------------
# CHECK 4: REFERENTIAL INTEGRITY
# Does every production row point to a well_id that actually exists
# in the wells table? (The database's FOREIGN KEY constraint should
# already prevent this - this check independently verifies it held.)
# ---------------------------------------------------------------------------
valid_well_ids = set(wells["well_id"])
orphan_rows = production[~production["well_id"].isin(valid_well_ids)]
referential_score = 1 - (len(orphan_rows) / len(production)) if len(production) > 0 else 1.0
if len(orphan_rows) > 0:
    issues.append(f"{len(orphan_rows)} production row(s) reference a well_id that doesn't exist")

# ---------------------------------------------------------------------------
# CHECK 5: TIMELINESS
# Is the data current? For a static one-time dataset this is less
# meaningful - it matters most once monthly automation is running,
# where "timely" means the latest month isn't older than expected.
# For now we just check the data isn't literally empty.
# ---------------------------------------------------------------------------
timeliness_score = 1.0 if len(production) > 0 else 0.0
if len(production) == 0:
    issues.append("Production table is empty")

# ---------------------------------------------------------------------------
# COMPOSITE SCORE
# A weighted average - completeness and validity matter most for THIS
# project (a forecasting model is only as good as having full, sensible
# history), so they're weighted higher than timeliness for now.
# ---------------------------------------------------------------------------
weights = {
    "completeness": 0.30,
    "uniqueness": 0.20,
    "validity": 0.25,
    "referential": 0.15,
    "timeliness": 0.10,
}

composite_score = (
    completeness_score * weights["completeness"]
    + uniqueness_score * weights["uniqueness"]
    + validity_score * weights["validity"]
    + referential_score * weights["referential"]
    + timeliness_score * weights["timeliness"]
)

PASS_THRESHOLD = 0.95
pass_fail = "PASS" if composite_score >= PASS_THRESHOLD else "FAIL"

# ---------------------------------------------------------------------------
# LOG THE RESULT
# ---------------------------------------------------------------------------
issues_summary = "; ".join(issues) if issues else "No issues found"

conn.execute("""
    INSERT INTO dq_results (run_timestamp, completeness_score, uniqueness_score,
                             validity_score, referential_score, timeliness_score,
                             composite_score, pass_fail, issues_found)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
""", (
    datetime.now(timezone.utc).isoformat(),
    round(completeness_score, 4), round(uniqueness_score, 4),
    round(validity_score, 4), round(referential_score, 4),
    round(timeliness_score, 4), round(composite_score, 4),
    pass_fail, issues_summary
))
conn.commit()
conn.close()

# ---------------------------------------------------------------------------
# REPORT TO SCREEN
# ---------------------------------------------------------------------------
print("=" * 60)
print("WPFI DATA QUALITY REPORT")
print("=" * 60)
print(f"Completeness:        {completeness_score:.2%}")
print(f"Uniqueness:          {uniqueness_score:.2%}")
print(f"Validity:            {validity_score:.2%}")
print(f"Referential Integrity: {referential_score:.2%}")
print(f"Timeliness:          {timeliness_score:.2%}")
print("-" * 60)
print(f"COMPOSITE SCORE:     {composite_score:.2%}  (threshold: {PASS_THRESHOLD:.0%})")
print(f"RESULT:              {pass_fail}")
print("-" * 60)
print(f"Issues: {issues_summary}")
print("=" * 60)