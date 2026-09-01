"""
WPFI - Streamlit App

Ported directly from the finalized mockup.html design. Navigation
uses URL query params (?page=...) so the custom nav bar can use plain
links instead of Streamlit's default sidebar.
"""

import os
import sys
import sqlite3
import io
import warnings
from datetime import date
from dateutil.relativedelta import relativedelta

import numpy as np
import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(SCRIPT_DIR, "models")
sys.path.insert(0, MODELS_DIR)

from forecast import forecast_well
from economic_limit import get_all_well_statuses
from eur import estimate_eur
from well_report import build_full_report

DB_PATH = os.path.join(SCRIPT_DIR, "data", "wpfi.db")

st.set_page_config(page_title="WPFI", page_icon="🛢️", layout="wide")

# ============================================================================
# DESIGN SYSTEM - ported directly from mockup.html
# ============================================================================
st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700&family=DM+Sans:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
:root {
    --white: #FFFFFF; --olive: #556B2F; --olive-dark: #40521f;
    --sage: #DCE5D4; --sage-line: #C9D6BE; --terracotta: #C96F4A;
    --terracotta-dark: #b25e3b; --charcoal: #29302A; --muted: #68736A;
    --border: #E5EAE2;
    --row-olive-light: #E8EDE1; --row-terracotta-light: #F3E1D8;
}
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; color: var(--charcoal); }
#MainMenu, footer, header {visibility: hidden;}
.block-container { padding-top: 0 !important; max-width: 1100px; }
.wpfi-nav {
    display: flex; align-items: center; justify-content: space-between;
    padding: 16px 4px; border-bottom: 1px solid var(--border); margin-bottom: 10px;
}
.wpfi-nav .mark { font-weight: 700; letter-spacing: 0.06em; color: var(--olive); font-size: 18px; }
.wpfi-nav .sub { font-size: 11px; color: var(--muted); letter-spacing: 0.02em; }
.wpfi-nav-links a { margin-left: 28px; font-size: 14px; font-weight: 500; color: var(--charcoal); text-decoration: none; padding-bottom: 4px; }
.wpfi-nav-links a.active { color: var(--olive); border-bottom: 2px solid var(--terracotta); }
.wpfi-back { font-size: 14px; font-weight: 600; color: var(--olive); text-decoration: none; }
.eyebrow { font-size: 12px; font-weight: 700; letter-spacing: 0.14em; color: var(--olive); text-transform: uppercase; margin-bottom: 12px; }
.wpfi-h1 { font-family: 'Playfair Display', serif; font-weight: 700; font-size: 34px; margin: 0 0 10px; color: var(--charcoal); }
.wpfi-subtitle { color: var(--muted); font-size: 15.5px; max-width: 70ch; margin-bottom: 18px; }
.stat-box { text-align: center; padding: 22px 10px; border-top: 3px solid var(--olive); }
.stat-box .num { font-family: 'Playfair Display', serif; font-size: 38px; font-weight: 700; color: var(--olive); }
.stat-box .label { font-size: 12.5px; color: var(--muted); text-transform: uppercase; letter-spacing: 0.06em; margin-top: 6px; }
.how-section { background: var(--sage); padding: 48px 32px; border-radius: 14px; margin: 32px 0; }
.section-title { font-family: 'Playfair Display', serif; font-size: 28px; font-weight: 700; text-align: center; margin-bottom: 36px; color: var(--charcoal); }
.step-num { font-family: 'Playfair Display', serif; font-size: 15px; color: var(--terracotta); font-weight: 700; margin-bottom: 8px; }
.step-title { font-size: 17px; font-weight: 700; margin-bottom: 8px; color: var(--charcoal); }
.step-desc { font-size: 14px; color: var(--muted); }
.step-flow { text-align: center; margin-top: 32px; font-size: 13px; color: var(--olive); font-weight: 600; }
.wpfi-card { border: 1px solid var(--border); border-radius: 12px; padding: 26px; background: var(--white); height: 100%; }
.wpfi-card h3 { font-size: 17px; font-weight: 700; margin-bottom: 8px; color: var(--olive); }
.wpfi-card p { font-size: 14px; color: var(--muted); margin-bottom: 14px; }
.wpfi-card .explore-link { font-size: 13.5px; font-weight: 700; color: var(--terracotta); }
.article-callout { background: var(--olive); color: white; text-align: center; padding: 48px 32px; border-radius: 14px; margin: 32px 0; }
.article-callout h3 { font-family: 'Playfair Display', serif; font-size: 24px; margin-bottom: 12px; }
.article-callout p { color: var(--sage); font-size: 14px; margin-bottom: 18px; }
.article-callout a { font-weight: 700; color: #F5D9C8; }
.wpfi-footer { background: var(--charcoal); color: #9AA394; text-align: center; padding: 22px; font-size: 13px; border-radius: 10px; margin-top: 24px; }
.wpfi-table-wrap { overflow-x: auto; margin: 12px 0 20px; border-radius: 8px; border: 1px solid var(--border); }
table.wpfi-table { width: 100%; border-collapse: collapse; font-size: 13.5px; }
table.wpfi-table th { background: var(--olive); color: #000000; text-align: left; padding: 10px 14px; font-weight: 700; white-space: nowrap; }
table.wpfi-table td { padding: 9px 14px; color: var(--charcoal); white-space: nowrap; }
table.wpfi-table tr:nth-child(odd)  td { background: var(--row-olive-light); }
table.wpfi-table tr:nth-child(even) td { background: var(--row-terracotta-light); }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# NAVIGATION
# ============================================================================
current_page = st.query_params.get("page", "overview")


def render_nav():
    def link(label, key):
        cls = "active" if current_page == key else ""
        return f'<a href="?page={key}" class="{cls}">{label}</a>'

    st.markdown(f"""
    <div class="wpfi-nav">
        <div><div class="mark">WPFI</div><div class="sub">Well Performance Forecasting Intelligence</div></div>
        <div class="wpfi-nav-links">
            {link("Overview", "overview")}{link("Wells", "wells")}{link("Forecasts", "forecasts")}{link("Data Quality", "data")}
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_back_link():
    st.markdown('<a href="?page=overview" class="wpfi-back">&larr; Back to Overview</a>', unsafe_allow_html=True)
    st.write("")


# ============================================================================
# DATA HELPERS
# ============================================================================
@st.cache_data(ttl=3600)
def load_wells():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM wells", conn)
    conn.close()
    return df


@st.cache_data(ttl=3600)
def load_production(well_id):
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(f"SELECT * FROM production WHERE well_id = {well_id} ORDER BY month_index", conn)
    conn.close()
    return df


@st.cache_data(ttl=3600)
def load_model_selection():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM model_selection", conn)
    conn.close()
    return df


def render_html_table(df):
    html = df.to_html(index=False, classes="wpfi-table", border=0)
    st.markdown(f'<div class="wpfi-table-wrap">{html}</div>', unsafe_allow_html=True)


def download_buttons(df, filename_base):
    buffer = io.BytesIO()
    df.to_excel(buffer, index=False, engine="openpyxl")
    st.download_button("Download Excel", buffer.getvalue(), f"{filename_base}.xlsx",
                        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")


# ============================================================================
# PAGE: OVERVIEW
# ============================================================================
def page_overview():
    render_nav()

    hero_col1, hero_col2 = st.columns([1.1, 1])
    with hero_col1:
        st.markdown("""
        <div class="eyebrow">Well Performance Forecasting Intelligence</div>
        <div class="wpfi-h1" style="font-size:44px; line-height:1.08;">Predicting what a well does next.</div>
        <p class="wpfi-subtitle">WPFI uses production history, decline-curve analysis, and machine learning
        to forecast future well performance and flag changes that may need attention — updated
        automatically every month.</p>
        """, unsafe_allow_html=True)
        st.link_button("Explore the Wells →", "?page=wells", type="primary")

    with hero_col2:
        st.markdown("""
        <div style="background:var(--sage); border:1px solid var(--sage-line); border-radius:14px; padding:24px;">
            <div style="font-size:12px; font-weight:600; letter-spacing:0.08em; color:var(--olive); text-transform:uppercase; margin-bottom:10px;">
                Oil Production — Sample Well
            </div>
            <svg viewBox="0 0 400 200" width="100%" height="180">
                <polyline points="10,40 60,55 110,72 160,90 200,105" fill="none" stroke="#556B2F" stroke-width="3" stroke-linecap="round"/>
                <polyline points="200,105 240,116 280,126 320,134 360,140 390,145" fill="none" stroke="#C96F4A" stroke-width="3" stroke-dasharray="2 6" stroke-linecap="round"/>
                <circle cx="200" cy="105" r="4" fill="#29302A"/>
            </svg>
            <div style="display:flex; gap:18px; font-size:12px; color:var(--muted); margin-top:8px;">
                <span>⬤ Historical</span><span style="color:#C96F4A;">⬤ Forecast</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.write("")
    st.write("")
    stats = [("18", "Wells Analyzed"), ("24+", "Months History"),
             ("2", "Forecasting Approaches"), ("12", "Month Forecast Horizon")]
    cols = st.columns(4)
    for col, (num, label) in zip(cols, stats):
        col.markdown(f"""
        <div class="stat-box"><div class="num">{num}</div><div class="label">{label}</div></div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="how-section">
        <div class="section-title">How WPFI Works</div>
    """, unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    steps = [
        ("01", "Analyze", "Production history is evaluated to understand each well's decline behavior and any operational events."),
        ("02", "Select", "Arps decline-curve analysis and XGBoost are compared per well; the better-performing approach is selected."),
        ("03", "Forecast", "The selected model generates the next month's forecast automatically as new production data arrives."),
    ]
    for col, (num, title, desc) in zip([c1, c2, c3], steps):
        col.markdown(f"""
        <div class="step-num">{num}</div>
        <div class="step-title">{title}</div>
        <div class="step-desc">{desc}</div>
        """, unsafe_allow_html=True)
    st.markdown("""
        <div class="step-flow">Production Data → Model Selection → Forecast</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">Explore WPFI</div>', unsafe_allow_html=True)
    e1, e2, e3 = st.columns(3)
    cards = [
        ("Wells", "Explore individual well performance, production history, and which model was selected for each well.", "wells"),
        ("Forecasts", "View projected production and compare forecast performance across all 18 wells.", "forecasts"),
        ("Data Quality", "See how the underlying production data is checked before it feeds the forecasting pipeline.", "data"),
    ]
    for col, (title, desc, target) in zip([e1, e2, e3], cards):
        with col:
            st.markdown(f"""
            <div class="wpfi-card"><h3>{title}</h3><p>{desc}</p>
            <a class="explore-link" href="?page={target}">Explore →</a></div>
            """, unsafe_allow_html=True)

    st.markdown("""
    <div class="article-callout">
        <h3>Want to see how it was built?</h3>
        <p>From synthetic production data to automated model selection and monthly
        forecasting, read the full methodology behind WPFI.</p>
        <a href="https://medium.com/@Nwokocha_Uchechi_Flora/forecasting-the-future-of-an-oil-well-inside-my-automated-well-performance-forecasting-system-bb525ee95046
">Read the article →</a>
    </div>
    <div class="wpfi-footer">Built by Nwokocha Uchechi Flora &nbsp;·&nbsp; © 2026</div>
    """, unsafe_allow_html=True)


# ============================================================================
# PAGE: WELLS
# ============================================================================
def page_wells():
    render_nav()
    render_back_link()
    st.markdown('<div class="wpfi-h1">Wells</div>', unsafe_allow_html=True)
    st.markdown('<p class="wpfi-subtitle">Browse each well\'s production history, current status, '
                'and which forecasting model was selected for it.</p>', unsafe_allow_html=True)

    wells = load_wells()
    model_sel = load_model_selection()
    statuses = get_all_well_statuses()

    well_name = st.selectbox("Select a well", wells["well_name"].tolist())
    well_row = wells[wells["well_name"] == well_name].iloc[0]
    well_id = int(well_row["well_id"])

    status_row = statuses[statuses["well_id"] == well_id].iloc[0]
    model_row = model_sel[model_sel["well_id"] == well_id]
    selected_model = model_row["selected_model"].iloc[0] if len(model_row) else "N/A"

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Location", well_row["location"])
    c2.metric("Decline Type", well_row["decline_type"].capitalize())
    c3.metric("Status", status_row["status"])
    c4.metric("Selected Model", selected_model)

    production = load_production(well_id)
    display_df = production[["month_index", "date", "oil_rate", "gas_rate", "water_rate", "event_note"]].rename(
        columns={"month_index": "Month", "date": "Date", "oil_rate": "Oil Rate",
                 "gas_rate": "Gas Rate", "water_rate": "Water Rate", "event_note": "Event"})
    render_html_table(display_df)
    download_buttons(display_df, f"{well_name}_production")


# ============================================================================
# PAGE: FORECASTS
# ============================================================================
def page_forecasts():
    render_nav()
    render_back_link()
    st.markdown('<div class="wpfi-h1">Forecasts</div>', unsafe_allow_html=True)
    st.markdown(
        '<p class="wpfi-subtitle">A 12-month forward projection and estimated ultimate recovery '
        '(EUR) for every well, using whichever model was proven most accurate for it. '
        '"12-Month Forecast" shows how many barrels of oil each well is expected to produce '
        '<b>per month</b>, 12 months from now — not a total sum over the year. For a full '
        'month-by-month breakdown of each well\'s next 12 months, download the PDF report below.</p>',
        unsafe_allow_html=True
    )

    conn = sqlite3.connect(DB_PATH)
    wells = load_wells()
    model_sel = load_model_selection()

    rows = []
    for _, w in wells.iterrows():
        well_id = int(w["well_id"])
        production = load_production(well_id)
        latest_actual = production["oil_rate"].iloc[-1]

        result = forecast_well(production["oil_rate"].values, production["water_rate"].values)
        forecast_end = result["forecast_values"][-1]
        pct_change = (forecast_end / latest_actual - 1) * 100

        eur = estimate_eur(well_id, conn)
        eur_display = f"{eur['eur_total']:,.0f}" + (" (min.)" if eur["hit_cap"] else "")

        model_row = model_sel[model_sel["well_id"] == well_id]
        selected_model = model_row["selected_model"].iloc[0] if len(model_row) else "N/A"

        rows.append({
            "Well": w["well_name"], "Location": w["location"], "Model": selected_model,
            "Latest Actual": round(latest_actual, 1), "12-Month Forecast": round(forecast_end, 1),
            "% Change": f"{pct_change:.1f}%", "EUR (Barrels)": eur_display,
        })
    conn.close()

    forecast_df = pd.DataFrame(rows)
    render_html_table(forecast_df)
    download_buttons(forecast_df, "wpfi_forecasts")
    st.caption("EUR = cumulative production to date + forecasted future production until each well "
               "crosses its own economic limit. '(min.)' marks wells still above their limit after a "
               "20-year projection cap — a conservative lower-bound estimate, not a full projection.")

    st.write("")
    st.markdown("#### View a well's forecast chart")
    well_name = st.selectbox("Select a well", wells["well_name"].tolist(), key="forecast_well_select")
    well_row_chart = wells[wells["well_name"] == well_name].iloc[0]
    well_id = int(well_row_chart["well_id"])
    production = load_production(well_id)
    result = forecast_well(production["oil_rate"].values, production["water_rate"].values)

    start_date = date.fromisoformat(well_row_chart["start_date"])
    hist_dates = [start_date + relativedelta(months=int(m) - 1) for m in production["month_index"]]
    forecast_dates = [start_date + relativedelta(months=int(m) - 1) for m in result["forecast_months"]]

    chart_df = pd.DataFrame({
        "Date": hist_dates + forecast_dates,
        "Oil Rate (Actual)": production["oil_rate"].tolist() + [None] * len(forecast_dates),
        "Oil Rate (Forecast)": [None] * len(production) + result["forecast_values"].tolist(),
    }).set_index("Date")
    st.line_chart(chart_df, color=["#556B2F", "#C96F4A"])

    st.write("")
    st.markdown(
        "**Want a deeper explanation of every well?** Generate a single plain-language PDF "
        "report covering all 18 wells — decline history, forecasting approach, 12-month "
        "outlook, and estimated ultimate recovery for each — for anyone who wants more "
        "insight than a spreadsheet gives."
    )

    if st.button("Generate Full WPFI Report (All Wells)"):
        conn = sqlite3.connect(DB_PATH)
        statuses = get_all_well_statuses()
        production_by_well = {int(w["well_id"]): load_production(int(w["well_id"])) for _, w in wells.iterrows()}
        eur_by_well = {int(w["well_id"]): estimate_eur(int(w["well_id"]), conn) for _, w in wells.iterrows()}
        conn.close()

        pdf_bytes = build_full_report(wells, production_by_well, statuses, model_sel, eur_by_well)
        st.download_button(
            "Download Full WPFI Report (PDF)", pdf_bytes, "wpfi_full_report.pdf", "application/pdf"
        )


# ============================================================================
# PAGE: DATA QUALITY
# ============================================================================
def page_data():
    render_nav()
    render_back_link()
    st.markdown('<div class="wpfi-h1">Data Quality</div>', unsafe_allow_html=True)
    st.markdown("""
    <p class="wpfi-subtitle">WPFI is built on a synthetic dataset of 18 wells with realistic decline
    curves, injected operational events (workovers, water breakthrough, shut-ins, recompletions), and
    +/-4% random noise — designed to mirror the messiness of real production data without using any
    real, confidential field data.</p>
    """, unsafe_allow_html=True)

    conn = sqlite3.connect(DB_PATH)
    dq = pd.read_sql_query("SELECT * FROM dq_results ORDER BY run_timestamp DESC LIMIT 1", conn)
    full_dataset = pd.read_sql_query("""
        SELECT p.well_id, w.well_name, w.location, p.month_index, p.date,
               p.oil_rate, p.gas_rate, p.water_rate, p.event_note
        FROM production p JOIN wells w ON p.well_id = w.well_id
        ORDER BY p.well_id, p.month_index
    """, conn)
    conn.close()

    if len(dq) > 0:
        latest = dq.iloc[0]
        st.markdown(f"""
        <div class="article-callout">
            <h3>Latest check: {latest['pass_fail']}</h3>
            <p>Composite score: {latest['composite_score']:.1%} — checked completeness, uniqueness,
            validity, referential integrity, and timeliness.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("#### Full dataset")
    st.caption(f"{len(full_dataset)} rows across {full_dataset['well_id'].nunique()} wells. "
               f"Download only — not displayed in full below.")
    download_buttons(full_dataset, "wpfi_full_dataset")


# ============================================================================
# ROUTER
# ============================================================================
pages = {"overview": page_overview, "wells": page_wells, "forecasts": page_forecasts, "data": page_data}
pages.get(current_page, page_overview)()