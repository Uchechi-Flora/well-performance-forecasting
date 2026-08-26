-- ============================================================================
-- WPFI Star Schema
-- ============================================================================
-- One dimension table (wells) + one fact table (production).
-- This file ONLY defines structure - no data is inserted here.
-- Other scripts (data generation, quality checks, automation) all reuse
-- this same schema so the database structure lives in exactly one place.
-- ============================================================================

DROP TABLE IF EXISTS production;
DROP TABLE IF EXISTS wells;

-- ----------------------------------------------------------------------------
-- DIMENSION TABLE: wells
-- Grain: ONE ROW PER WELL. Fixed, descriptive facts that do not change
-- month to month (a well's location doesn't change; its decline TYPE can,
-- but that's tracked as an EVENT in the fact table, not rewritten here).
-- ----------------------------------------------------------------------------
CREATE TABLE wells (
    well_id             INTEGER PRIMARY KEY,   -- surrogate key: simple auto ID,
                                                 -- not tied to any real-world code
    well_name           TEXT NOT NULL,          -- e.g. "WELL-01"
    location             TEXT NOT NULL,          -- field/location name
    decline_type         TEXT NOT NULL,          -- starting curve shape at month 1
    qi_oil                REAL NOT NULL,          -- initial oil rate (bbl/month)
    Di                    REAL NOT NULL,          -- initial nominal decline rate
    b                     REAL NOT NULL,          -- initial hyperbolic exponent
    complication_type    TEXT NOT NULL,          -- none / workover / water_breakthrough /
                                                    -- shut_in / recompletion
    complication_month   INTEGER,                -- which month the complication starts
    start_date            TEXT NOT NULL           -- when this well's history begins
);

-- ----------------------------------------------------------------------------
-- FACT TABLE: production
-- Grain: ONE ROW PER WELL, PER MONTH. This is where the numbers that
-- actually change over time live.
-- ----------------------------------------------------------------------------
CREATE TABLE production (
    production_id   INTEGER PRIMARY KEY AUTOINCREMENT,  -- surrogate key
    well_id          INTEGER NOT NULL,                    -- FOREIGN KEY -> wells.well_id
    month_index      INTEGER NOT NULL,                    -- 1 through 24
    date              TEXT NOT NULL,
    oil_rate          REAL NOT NULL,
    gas_rate          REAL NOT NULL,
    water_rate        REAL NOT NULL,
    event_note        TEXT,                                -- non-empty only on event months

    FOREIGN KEY (well_id) REFERENCES wells(well_id),
    UNIQUE(well_id, month_index)   -- prevents duplicate rows for the same
                                     -- well/month if a script runs twice
);

-- ----------------------------------------------------------------------------
-- INDEXES
-- An index is like a book's index page - it lets the database jump straight
-- to relevant rows instead of scanning the entire table. We add one on
-- well_id in the fact table since almost every query will filter or join
-- on "give me this well's production history."
-- ----------------------------------------------------------------------------
CREATE INDEX idx_production_well_id ON production(well_id);

-- ----------------------------------------------------------------------------
-- RESULTS TABLE: dq_results
-- Grain: ONE ROW PER QUALITY CHECK RUN. Every time the quality-check
-- script runs, it logs one row here - this is what lets you TREND data
-- quality over time (e.g. "did quality drop after last month's update?"),
-- rather than only ever seeing the most recent result.
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS dq_results (
    run_id                INTEGER PRIMARY KEY AUTOINCREMENT,
    run_timestamp         TEXT NOT NULL,

    completeness_score    REAL NOT NULL,   -- 0.0 to 1.0
    uniqueness_score      REAL NOT NULL,
    validity_score        REAL NOT NULL,
    referential_score     REAL NOT NULL,
    timeliness_score      REAL NOT NULL,

    composite_score       REAL NOT NULL,   -- weighted combination of the above
    pass_fail              TEXT NOT NULL,   -- 'PASS' or 'FAIL'

    issues_found           TEXT             -- short human-readable summary of problems
);