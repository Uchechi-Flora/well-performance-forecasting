"""
WPFI - Well Performance Forecasting Intelligence
Phase 1: Synthetic Data Generation

Generates 18 fictional wells (12 normal + 6 with injected complications),
24 months of monthly oil/gas/water production history each, following
Arps decline curve behavior (exponential, hyperbolic, harmonic).

Output:
- data/wpfi.db          (SQLite database - wells + production tables)
- data/production_preview.csv  (flat CSV export, for easy viewing in Excel)
"""

import sqlite3
import numpy as np
import pandas as pd
from datetime import date
from dateutil.relativedelta import relativedelta

np.random.seed(42)  # reproducible results

import os

# Build paths relative to this script's own location, so it works on
# any computer (Windows, Mac, Linux) without hardcoding a specific folder
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(SCRIPT_DIR, "..", "data", "wpfi.db")
CSV_PATH = os.path.join(SCRIPT_DIR, "..", "data", "production_preview.csv")
N_MONTHS = 24
START_DATE = date(2024, 1, 1)

NIGERIAN_FIELD_LOCATIONS = [
    "Bonny", "Forcados", "Escravos", "Bomu", "Oloibiri",
    "Nembe", "Okan", "Ughelli", "Afam", "Soku",
    "Idama", "Ogbainbiri", "Ekulama", "Cawthorne Channel",
    "Robertkiri", "Diebu Creek", "Belema", "Ubie"
]

# ---------------------------------------------------------------------------
# STEP 1: Define each well's "recipe" — its Arps parameters + complication
# ---------------------------------------------------------------------------
# decline_type: exponential (b=0), hyperbolic (0<b<1), harmonic (b=1)
# qi_oil = initial oil rate (bbl/day equivalent, but we store monthly avg)
# Di = monthly nominal decline rate
# complication_type: none / workover / water_breakthrough / shut_in / recompletion
# complication_month: which month (1-24) the event starts

wells_config = []

# 12 NORMAL wells — genuine variety, not clones. Mix of decline types,
# mix of decline rates and starting rates, deliberately not clustered
# to artificially help any one model later.
normal_wells = [
    dict(name="WELL-01", decline_type="exponential", qi_oil=850,  Di=0.035, b=0.0),
    dict(name="WELL-02", decline_type="hyperbolic",   qi_oil=1200, Di=0.06,  b=0.4,
         shift_month=14, shift_to_b=0.0, shift_to_Di=0.03),   # hyperbolic -> exponential
    dict(name="WELL-03", decline_type="exponential",  qi_oil=430,  Di=0.02,  b=0.0),
    dict(name="WELL-04", decline_type="hyperbolic",   qi_oil=950,  Di=0.08,  b=0.65,
         shift_month=17, shift_to_b=0.0, shift_to_Di=0.035),  # hyperbolic -> exponential
    dict(name="WELL-05", decline_type="harmonic",     qi_oil=610,  Di=0.045, b=1.0),
    dict(name="WELL-06", decline_type="exponential",  qi_oil=1400, Di=0.05,  b=0.0,
         shift_month=12, shift_to_b=0.5, shift_to_Di=0.06),   # exponential -> hyperbolic (reverse case)
    dict(name="WELL-07", decline_type="hyperbolic",   qi_oil=780,  Di=0.07,  b=0.3),
    dict(name="WELL-08", decline_type="hyperbolic",   qi_oil=1100, Di=0.055, b=0.55,
         shift_month=15, shift_to_b=0.0, shift_to_Di=0.03),   # hyperbolic -> exponential
    dict(name="WELL-09", decline_type="exponential",  qi_oil=520,  Di=0.028, b=0.0),
    dict(name="WELL-10", decline_type="harmonic",     qi_oil=690,  Di=0.06,  b=1.0),
    dict(name="WELL-11", decline_type="hyperbolic",   qi_oil=1050, Di=0.045, b=0.2,
         shift_month=18, shift_to_b=0.0, shift_to_Di=0.025),  # hyperbolic -> exponential
    dict(name="WELL-12", decline_type="exponential",  qi_oil=340,  Di=0.018, b=0.0),
]
for w in normal_wells:
    w["complication_type"] = "none"
    w["complication_month"] = None
    w.setdefault("shift_month", None)
    w.setdefault("shift_to_b", None)
    w.setdefault("shift_to_Di", None)

# 6 COMPLICATED wells — one clear "gotcha" story each
complicated_wells = [
    dict(name="WELL-13", decline_type="hyperbolic", qi_oil=900,  Di=0.05, b=0.45,
         complication_type="workover", complication_month=10),
    dict(name="WELL-14", decline_type="exponential", qi_oil=760, Di=0.04, b=0.0,
         complication_type="workover", complication_month=15),
    dict(name="WELL-15", decline_type="hyperbolic", qi_oil=1150, Di=0.06, b=0.5,
         complication_type="water_breakthrough", complication_month=12),
    dict(name="WELL-16", decline_type="exponential", qi_oil=630, Di=0.03, b=0.0,
         complication_type="water_breakthrough", complication_month=14),
    dict(name="WELL-17", decline_type="hyperbolic", qi_oil=880, Di=0.05, b=0.35,
         complication_type="shut_in", complication_month=11),
    dict(name="WELL-18", decline_type="exponential", qi_oil=1000, Di=0.045, b=0.0,
         complication_type="recompletion", complication_month=16),
]

for w in complicated_wells:
    w.setdefault("shift_month", None)
    w.setdefault("shift_to_b", None)
    w.setdefault("shift_to_Di", None)

wells_config = normal_wells + complicated_wells

for i, w in enumerate(wells_config):
    w["well_id"] = i + 1
    w["location"] = NIGERIAN_FIELD_LOCATIONS[i]
    w["start_date"] = START_DATE

# ---------------------------------------------------------------------------
# STEP 2: Arps decline formulas
# ---------------------------------------------------------------------------
def arps_rate(qi, Di, b, t):
    """Return production rate at month t using the Arps equation."""
    if b == 0:
        # Exponential decline
        return qi * np.exp(-Di * t)
    else:
        # Hyperbolic / harmonic decline (b=1 is harmonic, special case)
        return qi / ((1 + b * Di * t) ** (1 / b))


# ---------------------------------------------------------------------------
# STEP 3: Generate monthly production, injecting complications where relevant
# ---------------------------------------------------------------------------
def generate_well_production(well):
    rows = []
    qi = well["qi_oil"]
    Di = well["Di"]
    b = well["b"]
    comp_type = well["complication_type"]
    comp_month = well["complication_month"]

    # Track a "shift" applied after certain events (workover/recompletion
    # restart the decline clock from a boosted rate)
    effective_start_t = 0
    boost_applied_qi = qi

    # Track the CURRENT curve shape being used — this can change mid-well
    # if the well has a shift_month defined (e.g. hyperbolic -> exponential)
    current_Di = Di
    current_b = b
    shift_month = well.get("shift_month")
    shift_to_b = well.get("shift_to_b")
    shift_to_Di = well.get("shift_to_Di")
    has_shifted = False

    # Water cut behaves like a slowly rising fraction of total liquid,
    # rising faster after a water_breakthrough event
    base_water_cut = np.random.uniform(0.05, 0.15)  # starting water fraction

    is_shut_in = False
    shut_in_remaining = 0

    for month in range(1, N_MONTHS + 1):
        t = month - effective_start_t
        event_note = ""

        # ---- Handle shut-in wells: zero/near-zero production for a stretch
        if comp_type == "shut_in" and month == comp_month:
            is_shut_in = True
            shut_in_remaining = np.random.choice([2, 3])  # 2-3 month outage
            event_note = "Unplanned shut-in begins"

        if is_shut_in and shut_in_remaining > 0:
            oil_rate = 0.0
            gas_rate = 0.0
            water_rate = 0.0
            shut_in_remaining -= 1
            if shut_in_remaining == 0:
                is_shut_in = False
            rows.append(dict(
                well_id=well["well_id"], month_index=month,
                date=well["start_date"] + relativedelta(months=month - 1),
                oil_rate=round(oil_rate, 1), gas_rate=round(gas_rate, 1),
                water_rate=round(water_rate, 1), event_note=event_note or "Shut-in"
            ))
            continue

        # ---- Handle workover: brief dip then rate jumps back up (restart curve)
        if comp_type == "workover" and month == comp_month:
            event_note = "Workover performed - production restored"
            boost_applied_qi = arps_rate(qi, Di, b, t) * np.random.uniform(1.25, 1.45)
            effective_start_t = month - 1
            t = month - effective_start_t

        # ---- Handle recompletion: bigger jump, resets decline like a "new" well
        if comp_type == "recompletion" and month == comp_month:
            event_note = "Well recompleted - new decline cycle begins"
            boost_applied_qi = qi * np.random.uniform(0.85, 1.05)  # near-original rate
            effective_start_t = month - 1
            t = month - effective_start_t

        # ---- Handle curve-type transition: the well keeps declining
        # smoothly (no rate jump), but the SHAPE of the curve changes
        # going forward — e.g. hyperbolic slows into exponential once
        # the strong pressure support driving it depletes.
        if shift_month is not None and month == shift_month and not has_shifted:
            # Freeze the rate right where it currently is, then treat
            # that as the new "starting point" for the new curve shape
            rate_at_shift = arps_rate(
                boost_applied_qi if comp_type in ("workover", "recompletion") and month >= (comp_month or 9999) else qi,
                current_Di, current_b, t
            )
            boost_applied_qi = rate_at_shift
            current_Di = shift_to_Di
            current_b = shift_to_b
            effective_start_t = month - 1
            t = month - effective_start_t
            has_shifted = True
            from_shape = "hyperbolic" if b > 0 else "exponential"
            to_shape = "hyperbolic" if current_b > 0 else "exponential"
            event_note = f"Decline shape shifted: {from_shape} to {to_shape}"

        base_qi = boost_applied_qi if (has_shifted or (comp_type in ("workover", "recompletion") and month >= comp_month)) else qi
        oil_rate = arps_rate(base_qi, current_Di, current_b, t)

        # Small realistic noise (+/- 4%)
        oil_rate *= np.random.normal(1.0, 0.04)
        oil_rate = max(oil_rate, 0)

        # ---- Water cut: rises gradually; accelerates sharply for
        # water_breakthrough wells from the point of the event onward
        normal_water_cut = min(0.6, base_water_cut + 0.01 * month)
        if comp_type == "water_breakthrough" and month >= comp_month:
            months_since = month - comp_month
            # continue from wherever the water cut already was, then
            # accelerate the climb rather than resetting downward
            water_cut_at_event = min(0.6, base_water_cut + 0.01 * comp_month)
            water_cut = min(0.9, water_cut_at_event + 0.07 * months_since)
            if month == comp_month:
                event_note = "Water breakthrough detected"
        else:
            water_cut = normal_water_cut

        total_liquid = oil_rate / max(1 - water_cut, 0.05)
        water_rate = max(total_liquid - oil_rate, 0)

        # ---- Gas rate: tracks oil via a gas-oil ratio (GOR) that climbs
        # slowly as reservoir pressure drops (realistic late-life behavior)
        base_gor = np.random.uniform(0.9, 1.3)  # Mcf per bbl, simplified units
        gor = base_gor * (1 + 0.01 * month)
        gas_rate = oil_rate * gor * np.random.normal(1.0, 0.05)
        gas_rate = max(gas_rate, 0)

        rows.append(dict(
            well_id=well["well_id"], month_index=month,
            date=well["start_date"] + relativedelta(months=month - 1),
            oil_rate=round(oil_rate, 1), gas_rate=round(gas_rate, 1),
            water_rate=round(water_rate, 1), event_note=event_note
        ))

    return rows


# ---------------------------------------------------------------------------
# STEP 4: Build SQLite database using the standalone schema.sql file
# ---------------------------------------------------------------------------
conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()

SCHEMA_PATH = os.path.join(SCRIPT_DIR, "schema.sql")
with open(SCHEMA_PATH, "r") as f:
    schema_sql = f.read()
cur.executescript(schema_sql)

# Insert wells (dimension table)
for w in wells_config:
    cur.execute("""
        INSERT INTO wells (well_id, well_name, location, decline_type, qi_oil, Di, b,
                            complication_type, complication_month, start_date)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (w["well_id"], w["name"], w["location"], w["decline_type"], w["qi_oil"],
          w["Di"], w["b"], w["complication_type"], w["complication_month"],
          w["start_date"].isoformat()))

# Generate + insert production (fact table)
all_rows = []
for w in wells_config:
    well_rows = generate_well_production(w)
    for r in well_rows:
        r["date"] = r["date"].isoformat()
        all_rows.append(r)
        cur.execute("""
            INSERT INTO production (well_id, month_index, date, oil_rate, gas_rate, water_rate, event_note)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (r["well_id"], r["month_index"], r["date"], r["oil_rate"], r["gas_rate"],
              r["water_rate"], r["event_note"]))

conn.commit()

# ---------------------------------------------------------------------------
# STEP 5: Export a flat CSV preview (wells + production joined) for Flora
# to open directly in Excel and sanity-check
# ---------------------------------------------------------------------------
preview_df = pd.read_sql_query("""
    SELECT p.well_id, w.well_name, w.location, w.decline_type,
           w.complication_type, p.month_index, p.date,
           p.oil_rate, p.gas_rate, p.water_rate, p.event_note
    FROM production p
    JOIN wells w ON p.well_id = w.well_id
    ORDER BY p.well_id, p.month_index
""", conn)
preview_df.to_csv(CSV_PATH, index=False)

conn.close()

print(f"Database created: {DB_PATH}")
print(f"CSV preview created: {CSV_PATH}")
print(f"Total wells: {len(wells_config)}")
print(f"Total production rows: {len(all_rows)}")
print("\nComplication summary:")
for w in wells_config:
    if w["complication_type"] != "none":
        print(f"  {w['name']}: {w['complication_type']} at month {w['complication_month']}")