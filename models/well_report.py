"""
WPFI - Full Fleet PDF Report

Generates ONE PDF covering every well - each gets its own section
explaining its story in plain language: decline behavior, forecasting
approach, 12-month outlook, and estimated ultimate recovery. Built
for people who want more insight than a spreadsheet of numbers gives.
"""

import os
import sys
import io
import warnings

warnings.filterwarnings("ignore")

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from forecast import forecast_well


def _status_sentence(status):
    if status == "Reached Economic Limit":
        return ("this well has reached its economic limit - the point where continued "
                "production is no longer expected to justify operating costs.")
    if status == "Approaching Limit":
        return ("this well is approaching its economic limit and should be monitored "
                "closely over the coming months.")
    return "this well remains comfortably active, well above its economic limit."


def _well_section(well_row, production_df, status_row, selected_model, model_mape, eur_dict, styles):
    """Builds the list of PDF flowables (paragraphs, table, spacers) for
    ONE well's section. Returned as a plain list, NOT wrapped in its own
    document - the caller stitches these together into one shared PDF."""
    story = []

    heading1 = ParagraphStyle("H1well", parent=styles["Heading1"], textColor=colors.HexColor("#556B2F"), spaceAfter=4)
    subtitle_style = ParagraphStyle("Sub", parent=styles["Normal"], textColor=colors.grey, spaceAfter=14)
    body = ParagraphStyle("Body", parent=styles["Normal"], fontSize=10.5, leading=16, spaceAfter=12)
    heading2 = ParagraphStyle("H2", parent=styles["Heading2"], textColor=colors.HexColor("#556B2F"), spaceBefore=12, spaceAfter=6)

    story.append(Paragraph(well_row["well_name"], heading1))
    story.append(Paragraph(f"Location: {well_row['location']}", subtitle_style))

    # --- Overview ---
    story.append(Paragraph("Overview", heading2))
    oil_history = production_df["oil_rate"].values
    initial_rate = well_row["qi_oil"]
    latest_rate = oil_history[-1]
    pct_decline = (1 - latest_rate / initial_rate) * 100
    months_tracked = len(production_df)

    decline_desc = {
        "exponential": "a steady, constant-percentage decline",
        "hyperbolic": "a decline that started fast and has gradually slowed",
        "harmonic": "a very gradually slowing, long-tailing decline",
    }.get(well_row["decline_type"], well_row["decline_type"])

    overview_text = (
        f"{well_row['well_name']} has been tracked for {months_tracked} months. It began production "
        f"at approximately {initial_rate:,.0f} barrels per month and currently produces "
        f"{latest_rate:,.0f} barrels per month — an overall decline of {pct_decline:.1f}% since it "
        f"began. Its production follows {decline_desc}. Based on current production levels, "
        f"{_status_sentence(status_row['status'])}"
    )
    story.append(Paragraph(overview_text, body))

    if well_row["complication_type"] != "none":
        event_text = (
            f"This well's history includes a recorded operational event: "
            f"{well_row['complication_type'].replace('_', ' ')}, around month "
            f"{well_row['complication_month']}. WPFI accounts for this by fitting separate curves "
            f"to the well's behavior before and after the event."
        )
        story.append(Paragraph(event_text, body))

    # --- Model selection ---
    story.append(Paragraph("Forecasting Approach", heading2))
    story.append(Paragraph(
        f"WPFI tested multiple forecasting approaches and selected <b>{selected_model}</b> as the "
        f"most accurate for this well (mean absolute percentage error of {model_mape:.1f}% on "
        f"held-out validation months).", body
    ))

    # --- Forecast ---
    story.append(Paragraph("12-Month Forecast", heading2))
    result = forecast_well(oil_history, production_df["water_rate"].values)
    forecast_end = result["forecast_values"][-1]
    forecast_pct = (forecast_end / latest_rate - 1) * 100
    direction = "continue declining" if forecast_pct < 0 else "hold relatively steady"
    story.append(Paragraph(
        f"Over the next 12 months, this well is forecasted to {direction}, moving from "
        f"{latest_rate:,.0f} to approximately {forecast_end:,.0f} barrels per month "
        f"({forecast_pct:+.1f}%).", body
    ))

    table_data = [["Month", "Forecasted Oil Rate"]]
    for m, v in zip(result["forecast_months"], result["forecast_values"]):
        table_data.append([str(int(m)), f"{v:,.1f}"])
    t = Table(table_data, colWidths=[120, 200])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#556B2F")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F3E1D8")]),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(t)
    story.append(Spacer(1, 10))

    # --- EUR ---
    story.append(Paragraph("Estimated Ultimate Recovery (EUR)", heading2))
    eur_note = (
        " This is a conservative lower-bound estimate: the well had not yet reached its economic "
        "limit within a 20-year projection window, so its true EUR may be higher."
        if eur_dict["hit_cap"] else ""
    )
    story.append(Paragraph(
        f"Combining {eur_dict['cumulative_to_date']:,.0f} barrels already produced with an estimated "
        f"{eur_dict['remaining_to_limit']:,.0f} barrels of forecasted future production, "
        f"{well_row['well_name']}'s Estimated Ultimate Recovery (EUR) is approximately "
        f"<b>{eur_dict['eur_total']:,.0f} barrels</b>.{eur_note}", body
    ))

    return story


def build_full_report(wells_df, production_by_well, statuses_df, model_sel_df, eur_by_well):
    """
    Builds ONE PDF covering every well.

    wells_df: the full wells table
    production_by_well: dict of {well_id: production DataFrame}
    statuses_df: output of get_all_well_statuses()
    model_sel_df: the model_selection table
    eur_by_well: dict of {well_id: estimate_eur() result}
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, topMargin=48, bottomMargin=48,
                             leftMargin=54, rightMargin=54)
    styles = getSampleStyleSheet()
    story = []

    # --- Cover section ---
    title_style = ParagraphStyle("Title", parent=styles["Title"], textColor=colors.HexColor("#556B2F"))
    story.append(Paragraph("WPFI Well Performance Report", title_style))
    subtitle_style = ParagraphStyle("Sub", parent=styles["Normal"], textColor=colors.grey, spaceAfter=20)
    story.append(Paragraph(f"Covering all {len(wells_df)} wells — generated from live WPFI data", subtitle_style))
    story.append(Spacer(1, 10))

    for i, (_, well_row) in enumerate(wells_df.iterrows()):
        well_id = int(well_row["well_id"])
        production_df = production_by_well[well_id]
        status_row = statuses_df[statuses_df["well_id"] == well_id].iloc[0]
        model_row = model_sel_df[model_sel_df["well_id"] == well_id]
        selected_model = model_row["selected_model"].iloc[0] if len(model_row) else "Arps"
        model_mape = model_row["selected_mape"].iloc[0] if len(model_row) else 0
        eur_dict = eur_by_well[well_id]

        section = _well_section(well_row, production_df, status_row, selected_model,
                                 model_mape, eur_dict, styles)
        story.extend(section)

        if i < len(wells_df) - 1:
            story.append(PageBreak())

    footer_style = ParagraphStyle("Footer", parent=styles["Normal"], fontSize=8, textColor=colors.grey)
    story.append(Spacer(1, 16))
    story.append(Paragraph(
        "WPFI — Well Performance Forecasting Intelligence. This report is generated from a synthetic "
        "dataset built for portfolio demonstration purposes.", footer_style
    ))

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()