"""
WPFI - Decline Report

Instead of only ever showing an open-ended forecast, this builds a
HISTORICAL SUMMARY: for every well, how it has actually performed
from its start date to its most recent recorded month - initial
rate, current rate, overall % decline, average monthly decline rate,
and its current economic status.

Produces both an in-terminal table and a downloadable PDF.
"""

import os
import sys
import sqlite3
from datetime import date
import pandas as pd

from reportlab.lib.pagesizes import letter, landscape
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from economic_limit import get_well_status, ECONOMIC_LIMIT_PCT

DB_PATH = os.path.join(SCRIPT_DIR, "..", "data", "wpfi.db")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "..", "data")


def build_decline_report():
    conn = sqlite3.connect(DB_PATH)
    wells = pd.read_sql_query("SELECT well_id, well_name, location, start_date FROM wells", conn)

    rows = []
    for _, w in wells.iterrows():
        well_id = w["well_id"]
        status_info = get_well_status(well_id, conn)

        production = pd.read_sql_query(
            f"SELECT month_index, date FROM production WHERE well_id = {well_id} "
            f"ORDER BY month_index", conn
        )
        latest_date = production["date"].iloc[-1]
        n_months = len(production)

        initial_rate = status_info["initial_rate"]
        latest_rate = status_info["latest_rate"]
        pct_decline = (1 - latest_rate / initial_rate) * 100
        avg_monthly_decline = pct_decline / n_months  # simple average over the period

        rows.append(dict(
            well_name=w["well_name"], location=w["location"],
            start_date=w["start_date"], latest_date=latest_date,
            months_tracked=n_months,
            initial_rate=round(initial_rate, 1), latest_rate=round(latest_rate, 1),
            pct_decline=round(pct_decline, 1),
            avg_monthly_decline=round(avg_monthly_decline, 2),
            status=status_info["status"]
        ))

    conn.close()
    return pd.DataFrame(rows)


def export_pdf(report_df, filepath):
    doc = SimpleDocTemplate(filepath, pagesize=landscape(letter),
                             topMargin=40, bottomMargin=40)
    styles = getSampleStyleSheet()
    story = []

    title = Paragraph("WPFI Well Decline Report", styles["Title"])
    story.append(title)

    subtitle_style = ParagraphStyle(
        "Subtitle", parent=styles["Normal"], textColor=colors.grey, spaceAfter=16
    )
    period_start = report_df["start_date"].min()
    period_end = report_df["latest_date"].max()
    subtitle = Paragraph(
        f"Covering all wells from {period_start} to {period_end} &nbsp;|&nbsp; "
        f"Economic limit defined as {int(ECONOMIC_LIMIT_PCT*100)}% of each well's own starting rate",
        subtitle_style
    )
    story.append(subtitle)
    story.append(Spacer(1, 12))

    # Build the table data: header row + one row per well
    headers = ["Well", "Location", "Start", "Latest", "Months",
               "Initial Rate", "Latest Rate", "% Decline", "Avg Monthly Decline", "Status"]
    table_data = [headers]
    for _, row in report_df.iterrows():
        table_data.append([
            row["well_name"], row["location"], row["start_date"], row["latest_date"],
            str(row["months_tracked"]), f"{row['initial_rate']:.1f}", f"{row['latest_rate']:.1f}",
            f"{row['pct_decline']:.1f}%", f"{row['avg_monthly_decline']:.2f}%/mo", row["status"]
        ])

    table = Table(table_data, repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f2937")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("ALIGN", (4, 0), (-1, -1), "CENTER"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f3f4f6")]),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(table)

    story.append(Spacer(1, 20))
    status_counts = report_df["status"].value_counts()
    summary_text = " &nbsp;|&nbsp; ".join(
        f"{status}: {count} wells" for status, count in status_counts.items()
    )
    story.append(Paragraph(f"Summary: {summary_text}", styles["Normal"]))

    doc.build(story)


if __name__ == "__main__":
    report_df = build_decline_report()
    print(report_df.to_string(index=False))

    pdf_path = os.path.join(OUTPUT_DIR, "wpfi_decline_report.pdf")
    export_pdf(report_df, pdf_path)
    print(f"\nPDF report saved to: {pdf_path}")