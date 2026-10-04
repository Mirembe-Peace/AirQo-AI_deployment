"""Builds the assignment slide deck on the UCU PowerPoint template from the pipeline outputs.

Run from the repo root after the notebook has been executed:
    python notebook_src/build_presentation.py <path/to/UCU-power-point template.pptx>
"""
import json
import sys
from pathlib import Path

import pandas as pd
from PIL import Image
from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.dml.color import RGBColor
from pptx.enum.chart import XL_CHART_TYPE, XL_LABEL_POSITION
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.util import Inches, Pt

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "outputs" / "figures"
TEMPLATE = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "templates" / "UCU-power-point template.pptx"
OUT = ROOT / "deliverables" / "AirQo_PM25_Presentation.pptx"
OUT.parent.mkdir(exist_ok=True)

NAVY, CRIMSON, GOLD = RGBColor(0x0B, 0x3D, 0x91), RGBColor(0xD7, 0x01, 0x4D), RGBColor(0xFF, 0xD9, 0x32)
TITLE_BLUE, INK, MUTED = RGBColor(0x1F, 0x49, 0x7D), RGBColor(0x22, 0x22, 0x22), RGBColor(0x5F, 0x6B, 0x7A)
TINT = RGBColor(0xEE, 0xF3, 0xFA)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)

comp = pd.read_csv(ROOT / "outputs" / "model_comparison.csv", index_col=0)
card = json.load(open(ROOT / "models" / "best_model_card.json"))
best = card["model_name"]
bm = card["test_metrics"]

prs = Presentation(TEMPLATE)
L_TITLE, L_TITLE_ONLY, L_LAST = prs.slide_layouts[0], prs.slide_layouts[5], prs.slide_layouts[11]

# drop the two example slides that ship with the template
sldIdLst = prs.slides._sldIdLst
for sldId in list(sldIdLst):
    prs.part.drop_rel(sldId.rId)
    sldIdLst.remove(sldId)

CONTENT_TOP, CONTENT_BOTTOM, LEFT, RIGHT = 2.05, 6.65, 0.92, 12.42


def set_title(slide, text, size=28):
    t = slide.shapes.title
    # fixed geometry: one line, sitting just above the template's colour band and clear of the logo
    t.left, t.top, t.width, t.height = Inches(0.92), Inches(0.8), Inches(9.7), Inches(0.8)
    t.text_frame.word_wrap = True
    t.text = text
    for p in t.text_frame.paragraphs:
        for r in p.runs:
            r.font.size = Pt(size)
            r.font.bold = True
            r.font.color.rgb = TITLE_BLUE
    t.text_frame.vertical_anchor = MSO_ANCHOR.BOTTOM


def textbox(slide, x, y, w, h, paras, size=15, color=INK, bold_first=False, bullets=False, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP):
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = Inches(0.05)
    for i, item in enumerate(paras):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        p.space_after = Pt(6)
        runs = item if isinstance(item, list) else [(item, {})]
        if bullets:
            runs = [("•  ", {"color": CRIMSON, "bold": True})] + runs
        for txt, opt in runs:
            r = p.add_run()
            r.text = txt
            r.font.size = Pt(opt.get("size", size))
            r.font.bold = opt.get("bold", bold_first and i == 0)
            r.font.italic = opt.get("italic", False)
            r.font.color.rgb = opt.get("color", color)
    return tb


def picture(slide, path, x, y, w, h):
    """Fit an image inside the box (x, y, w, h) keeping its aspect ratio, centred."""
    iw, ih = Image.open(path).size
    scale = min(w / iw, h / ih)
    pw, ph = iw * scale, ih * scale
    return slide.shapes.add_picture(str(path), Inches(x + (w - pw) / 2), Inches(y + (h - ph) / 2), Inches(pw), Inches(ph))


def card_box(slide, x, y, w, h, fill=TINT):
    s = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    s.adjustments[0] = 0.08
    s.fill.solid(); s.fill.fore_color.rgb = fill
    s.line.fill.background()
    s.shadow.inherit = False
    return s


def stat(slide, x, y, w, value, label, color=NAVY):
    card_box(slide, x, y, w, 1.55)
    textbox(slide, x + 0.1, y + 0.12, w - 0.2, 0.8, [value], size=34, color=color, bold_first=True, align=PP_ALIGN.CENTER)
    textbox(slide, x + 0.1, y + 0.9, w - 0.2, 0.6, [label], size=12, color=MUTED, align=PP_ALIGN.CENTER)


def table(slide, x, y, w, rows, col_w, size=12, header_fill=NAVY, row_h=0.36):
    shp = slide.shapes.add_table(len(rows), len(rows[0]), Inches(x), Inches(y), Inches(w), Inches(row_h * len(rows)))
    tbl = shp.table
    for j, cw in enumerate(col_w):
        tbl.columns[j].width = Inches(cw)
    for i, row in enumerate(rows):
        for j, val in enumerate(row):
            c = tbl.cell(i, j)
            c.text = str(val)
            c.margin_left = c.margin_right = Inches(0.06)
            c.margin_top = c.margin_bottom = Inches(0.03)
            for p in c.text_frame.paragraphs:
                for r in p.runs:
                    r.font.size = Pt(size)
                    r.font.bold = i == 0
                    r.font.color.rgb = WHITE if i == 0 else INK
            c.fill.solid()
            c.fill.fore_color.rgb = header_fill if i == 0 else (TINT if i % 2 else WHITE)
    return tbl


def new_slide(title, notes=""):
    s = prs.slides.add_slide(L_TITLE_ONLY)
    set_title(s, title)
    if notes:
        s.notes_slide.notes_text_frame.text = notes
    return s


# 1 — Title ---------------------------------------------------------------------------------
s = prs.slides.add_slide(L_TITLE)
s.shapes.title.text = "Predicting PM2.5 Air Quality from AirQo Sensor Data"
for r in s.shapes.title.text_frame.paragraphs[0].runs:
    r.font.size = Pt(40); r.font.bold = True; r.font.color.rgb = TITLE_BLUE
sub = s.placeholders[1].text_frame
sub.text = "CSC3119 AI Deployment & Scalability · Assignment 4"
sub.add_paragraph().text = "Mirembe Peace Mercy / M24B23/005 / B27499"
sub.add_paragraph().text = "Norah Angel Nakaye / M24B23/008 / B27502"
for i, p in enumerate(sub.paragraphs):
    for r in p.runs:
        r.font.size = Pt(20 if i == 0 else 18); r.font.color.rgb = CRIMSON if i == 0 else INK; r.font.bold = i == 0
s.notes_slide.notes_text_frame.text = ("We built a complete pipeline that turns raw AirQo low-cost sensor readings into a trained, evaluated, "
                                      "monitored and deployable PM2.5 forecasting model.")

# 2 — Why it matters --------------------------------------------------------------------------
s = new_slide("Why forecast PM2.5 in Uganda?",
              "PM2.5 penetrates deep into the lungs and bloodstream. Kampala's annual mean is about eight times the WHO guideline. "
              "Forecasts let the health sector, NEMA, the Ministry of Energy and Works & Transport act before bad-air days.")
stat(s, LEFT, 2.15, 3.6, "39 µg/m³", "Kampala annual mean PM2.5 (2018–21), ≈ 8× WHO guideline of 5 µg/m³", CRIMSON)
stat(s, LEFT + 3.95, 2.15, 3.6, "7,257", "deaths in Kampala attributed to long-term PM2.5 exposure (2018–21)", CRIMSON)
stat(s, LEFT + 7.9, 2.15, 3.6, "149", "AirQo sensors in this dataset · 14,600 hourly rows · 2026", NAVY)
textbox(s, LEFT, 4.05, 5.6, 2.6, [
    [("Objective  ", {"bold": True, "color": NAVY}), ("Build a deployable pipeline that predicts hourly PM2.5 (as a Z-score) from sensor, time and weather data, and forecasts August 2026.", {})],
    [("Who uses it  ", {"bold": True, "color": NAVY}), ("Health sector (early warning), NEMA (enforcement), Ministry of Energy (clean cooking), Works & Transport (traffic).", {})],
], size=14)
textbox(s, LEFT + 5.9, 4.05, 5.6, 2.6, [
    [("Deliverables  ", {"bold": True, "color": NAVY}), ("cleaned data → EDA → 5 tuned models + ensemble → saved model → August forecasts → Evidently monitoring → RunPod run.", {})],
    [("AQI bands used throughout  ", {"bold": True, "color": NAVY}), ("Good ≤ 9 · Moderate ≤ 35.49 · USG ≤ 55.49 · Unhealthy ≤ 125.49 · Very Unhealthy ≤ 225.49 · Hazardous above.", {})],
], size=14)

# 3 — Pipeline ------------------------------------------------------------------------------
s = new_slide("Methodology pipeline", "Each stage feeds the next; every heavy step is wrapped in try/except so the pipeline never crashes.")
picture(s, FIG / "00_pipeline_flowchart.png", LEFT, CONTENT_TOP, RIGHT - LEFT, CONTENT_BOTTOM - CONTENT_TOP)

# 4 — Data reality check --------------------------------------------------------------------
s = new_slide("The data: 3 short windows, many nulls",
              "Although labelled Jan–Jul, the file holds only 1–3 Jan, 1–2 Apr and 1–3 Jun. After resampling to a full hourly grid "
              "only 1.9% of sensor-hours are observed. Zeros in temp/humidity/coords are nulls per the brief.")
stat(s, LEFT, 2.15, 2.7, "3", "observation windows (1–3 Jan, 1–2 Apr, 1–3 Jun)")
stat(s, LEFT + 2.95, 2.15, 2.7, "758,112", "rows in the full hourly grid (149 sensors × 5,088 h)")
stat(s, LEFT + 5.9, 2.15, 2.7, "1.9%", "of grid hours have a valid PM2.5 reading", CRIMSON)
stat(s, LEFT + 8.85, 2.15, 2.65, "28–30%", "of temperature / humidity values are 0 (= null)", CRIMSON)
table(s, LEFT, 4.0, 6.2, [
    ["Field", "Empty", "Zero", "% null"],
    ["latitude / longitude", "1", "3,085", "21%"],
    ["temperature", "54", "4,066", "28%"],
    ["humidity", "206", "4,159", "30%"],
    ["PM2.5 calibrated", "325", "0", "2%"],
    ["site_id", "7,508", "–", "51%"],
], [2.6, 1.1, 1.2, 1.3], size=12)
textbox(s, LEFT + 6.6, 4.0, 4.9, 2.6, [
    "PM2.5 maximum of 998 µg/m³ → sensor saturation",
    "frequency is constant; site_id duplicates site_name",
    "GPS jitters up to 119 positions per fixed device",
    "6 devices never report a valid GPS fix",
], size=14, bullets=True)

# 5 — Cleaning decisions ----------------------------------------------------------------------
s = new_slide("Data preparation decisions",
              "Junninen et al. (2004): linear interpolation is best for short gaps; long gaps should not be filled. AirQo papers (Okure 2022, Adong 2022) and Barkjohn 2021 guide validity checks.")
table(s, LEFT, 2.1, RIGHT - LEFT, [
    ["Situation", "Action", "Why"],
    ["0 or empty in temp, RH, PM, lat, lon", "→ NaN", "Brief: 0 is a placeholder, not a measurement"],
    ["PM2.5 > 500 µg/m³ (3 rows)", "→ NaN, logged", "Beyond low-cost sensor range (Barkjohn et al., 2021)"],
    ["Per-sensor IQR outliers (650 rows)", "Flag & keep", "Real rush-hour / inversion episodes, not errors"],
    ["Gap ≤ 3 h", "Linear time interpolation", "Accurate for short gaps (Junninen et al., 2004)"],
    ["Gap > 3 h (device offline)", "Keep NaN + is_observed = 0", "Never fabricate labels for long outages (Okure et al., 2022)"],
    ["Missing temp / RH at observed hour", "IDW of 3 nearest sensors → hour median", "Weather is spatially smooth"],
    ["Missing coordinates", "Device median; gazetteer for 6", "Fixed installations; GPS noise averaged out"],
    ["Hourly resampling", "Full grid 1 Jan – 31 Jul", "Offline hours missing, not absent"],
], [3.6, 3.3, 4.6], size=12, row_h=0.47)

# 6 — Feature selection ----------------------------------------------------------------------
s = new_slide("Feature selection",
              "PM10 is excluded: it's co-measured (leakage), unknown at forecast time, and driven by dust rather than temperature/humidity.")
card_box(s, LEFT, 2.15, 5.55, 4.45)
textbox(s, LEFT + 0.25, 2.25, 5.1, 0.5, ["Used (9 features)"], size=18, color=NAVY, bold_first=True)
textbox(s, LEFT + 0.25, 2.8, 5.1, 3.7, [
    [("Hour of day (sin/cos)", {"bold": True}), (" – diurnal traffic, cooking, inversions", {})],
    [("Temperature, humidity", {"bold": True}), (" – boundary layer & particle growth", {})],
    [("Latitude, longitude", {"bold": True}), (" – city centre vs suburbs vs towns", {})],
    [("Peak dry season flag", {"bold": True}), (" – Jan ≈ 2× Apr/Jun levels", {})],
    [("Device level", {"bold": True}), (" – season-adjusted site encoding", {})],
    [("Network", {"bold": True}), (" – AirQo vs AirGradient hardware", {})],
], size=14, bullets=True)
card_box(s, LEFT + 5.95, 2.15, 5.55, 4.45, fill=RGBColor(0xFB, 0xEC, 0xF1))
textbox(s, LEFT + 6.2, 2.25, 5.1, 0.5, ["Excluded"], size=18, color=CRIMSON, bold_first=True)
textbox(s, LEFT + 6.2, 2.8, 5.1, 3.7, [
    [("PM10", {"bold": True}), (" – same-sensor leakage; unavailable in a forecast", {})],
    [("site_name, site_id", {"bold": True}), (" – identifiers, covered by device level", {})],
    [("frequency", {"bold": True}), (" – constant ('hourly')", {})],
    [("Day of week", {"bold": True}), (" – only Mon–Fri observed; unlearnable", {})],
    [("Target", {"bold": True}), (f" – PM2.5 Z-score, z = (x − μ)/σ with μ = {card['mu']:.1f}, σ = {card['sigma']:.1f} µg/m³ (train set)", {})],
], size=14, bullets=True)

# 7 — EDA distributions & box plots ------------------------------------------------------------
s = new_slide("EDA: distributions and outliers",
              "Five Kampala sensors: Nansana, Nakasero II, Lower Nsooba, Civic Centre, Salama Road. Skewness ≈ 2.1. Outliers flagged but kept.")
picture(s, FIG / "03_distributions.png", LEFT, 2.0, RIGHT - LEFT, 2.35)
picture(s, FIG / "03c_boxplots.png", LEFT, 4.35, RIGHT - LEFT, 2.35)

# 8 — Time series with AQI bands ---------------------------------------------------------------
s = new_slide("EDA: hourly PM2.5 vs AQI bands",
              "January is far dirtier (Unhealthy to Very Unhealthy) than April and June (mostly Moderate). Morning and evening peaks are visible every day.")
picture(s, FIG / "03d_timeseries_aqi.png", LEFT, 1.95, 8.2, 4.75)
textbox(s, LEFT + 8.4, 2.2, 3.1, 4.4, [
    [("January ", {"bold": True, "color": CRIMSON}), ("(peak dry season) daily means 44–55 µg/m³", {})],
    [("April & June ", {"bold": True, "color": NAVY}), ("≈ 21–28 µg/m³", {})],
    [("Salama Road ", {"bold": True}), ("and Civic Centre are the worst sites", {})],
    [("Peaks ", {"bold": True}), ("06:00–09:00 and 19:00–23:00", {})],
], size=14, bullets=True)

# 9 — Calendar -----------------------------------------------------------------------------
s = new_slide("EDA: AQI calendar, Jan–Jul 2026",
              "Grey days are offline/no data — the calendar makes the three observation windows obvious.")
picture(s, FIG / "03f2_calendar_observed_months.png", LEFT, 1.95, 6.4, 4.75)
picture(s, FIG / "03g_aqi_share.png", LEFT + 6.55, 2.1, 5.0, 2.4)
textbox(s, LEFT + 6.6, 4.65, 4.9, 2.0, [
    "Only 0–9% of hours are Good at any site",
    "Salama Road: ~49% of hours Unhealthy or worse",
    "Nansana (peri-urban): 79% Moderate",
], size=14, bullets=True)

# 10 — Z-score & correlation -----------------------------------------------------------------
s = new_slide("Z-scores and the weather link",
              "z = 0 is the network-average hour; z = +1 one standard deviation dirtier. The USG threshold (35.49) sits at z ≈ 0.1, so above-average already means unhealthy for sensitive groups.")
picture(s, FIG / "03h_zscore.png", LEFT, 1.95, 6.0, 2.4)
picture(s, FIG / "03i_correlation.png", LEFT, 4.4, 6.0, 2.3)
textbox(s, LEFT + 6.3, 2.1, 5.2, 4.5, [
    [("Z-score ", {"bold": True, "color": NAVY}), ("re-centres and re-scales but keeps the skew: lowest z ≈ −1.0, spikes reach about +14", {})],
    [("4.7% ", {"bold": True}), ("of hours have |z| > 2; 1.5% have |z| > 3", {})],
    [("Temperature ", {"bold": True, "color": NAVY}), ("is negatively correlated (r up to −0.50)", {})],
    [("Humidity ", {"bold": True, "color": NAVY}), ("is positively correlated (r up to +0.41)", {})],
    [("Physics: ", {"bold": True}), ("cool, humid nights and mornings mean a shallow boundary layer that traps smoke", {})],
], size=14, bullets=True)

# 11 — Models & hyperparameters ----------------------------------------------------------------
s = new_slide("Five tuned models + ensemble",
              "RandomizedSearchCV with GroupKFold by calendar day (blocked CV). TimeSeriesSplit was rejected because early folds trained only on January.")
bp = card["best_params"]
table(s, LEFT, 2.1, RIGHT - LEFT, [
    ["Model", "Hyperparameters tuned", "Why they matter"],
    ["Ridge (+ poly)", "alpha, polynomial degree", "Shrinkage vs overfitting; interactions"],
    ["K-Nearest Neighbours", "k, weights, distance p", "Small k memorises noise, large k over-smooths"],
    ["SVR (RBF)", "C, gamma, epsilon", "Error penalty, kernel width, noise tolerance"],
    ["Random Forest", "n_estimators, max_depth, min_samples_leaf, max_features", "Variance reduction; tree de-correlation"],
    ["XGBoost", "learning_rate, n_estimators, max_depth, min_child_weight, subsample, colsample, lambda", "Step size × rounds; complexity; regularisation"],
    ["Stacking ensemble", "Meta RidgeCV alpha over RF + XGB + KNN + SVR", "Combines models that err differently"],
], [2.5, 4.9, 4.1], size=12, row_h=0.55)
textbox(s, LEFT, 6.05, RIGHT - LEFT, 0.6, [
    [("Validation: ", {"bold": True, "color": NAVY}), ("train = Jan, Apr, 1 Jun; test = last full 24 h (2 Jun 06:00 → 3 Jun 06:00), never used in tuning", {})]], size=13)

# 12 — Results (native chart) -------------------------------------------------------------------
s = new_slide(f"Results: {best} has the lowest RMSE",
              "RMSE is the primary metric because big misses on pollution spikes are the costly errors. All models beat both baselines.")
order = [m for m in comp.index if not m.startswith("Baseline")] + ["Baseline: sensor mean", "Baseline: global mean"]
cd = CategoryChartData()
cd.categories = [m.replace("Baseline: ", "Base: ") for m in order]
cd.add_series("Test RMSE (µg/m³)", [round(comp.loc[m, "RMSE_ugm3"], 1) for m in order])
gf = s.shapes.add_chart(XL_CHART_TYPE.BAR_CLUSTERED, Inches(LEFT), Inches(2.0), Inches(5.6), Inches(4.65), cd)
ch = gf.chart
ch.has_legend = False
ch.has_title = True
ch.chart_title.text_frame.text = "Test RMSE (µg/m³) — lower is better"
ch.chart_title.text_frame.paragraphs[0].runs[0].font.size = Pt(13)
plot = ch.plots[0]
plot.has_data_labels = True
plot.data_labels.font.size = Pt(11)
plot.data_labels.position = XL_LABEL_POSITION.OUTSIDE_END
plot.gap_width = 60
ch.category_axis.reverse_order = True
ch.category_axis.tick_labels.font.size = Pt(11)
ch.value_axis.visible = False
ch.value_axis.has_major_gridlines = False
for i, pt in enumerate(plot.series[0].points):
    pt.format.fill.solid()
    pt.format.fill.fore_color.rgb = NAVY if i == 0 else (RGBColor(0xBF, 0xBF, 0xBF) if order[i].startswith("Baseline") else RGBColor(0x8F, 0xB3, 0xE0))
rows = [["Model", "RMSE µg/m³", "MAE µg/m³", "R²", "AQI acc."]]
for m in order:
    r = comp.loc[m]
    rows.append([m.replace("Baseline: ", "Base: "), f"{r.RMSE_ugm3:.1f}", f"{r.MAE_ugm3:.1f}", f"{r.R2:.2f}", f"{r['AQI_accuracy_%']:.0f}%"])
table(s, LEFT + 5.8, 2.05, 5.7, rows, [2.1, 1.05, 0.95, 0.7, 0.9], size=11, row_h=0.4)
textbox(s, LEFT + 5.8, 5.85, 5.7, 0.8, [
    "Note: the global-mean baseline scores 73% AQI accuracy simply because most test hours are Moderate — accuracy alone misleads"],
    size=12, color=MUTED)

# 13 — Diagnostics -------------------------------------------------------------------------------
s = new_slide(f"Best model: {best} diagnostics",
              f"Test RMSE {bm['RMSE_ugm3']:.1f} µg/m³, MAE {bm['MAE_ugm3']:.1f}, R² {bm['R2']:.2f}. Residuals are centred near zero but the model under-predicts the largest spikes.")
picture(s, FIG / "04b_best_model_diagnostics.png", LEFT, 1.95, RIGHT - LEFT, 2.75)
picture(s, FIG / "04d_feature_importance.png", LEFT, 4.7, 5.3, 2.0)
picture(s, FIG / "04c_aqi_confusion.png", LEFT + 5.5, 4.7, 2.5, 2.0)
textbox(s, LEFT + 8.2, 4.75, 3.3, 1.95, [
    [(f"R² = {bm['R2']:.2f}", {"bold": True, "color": NAVY}), (": explains ~22% of hourly variance with no recent readings", {})],
    [("Overfit gap ", {"bold": True}), (f"{bm['overfit_gap']:.2f} z (train {bm['train_RMSE_z']:.2f} vs test {bm['RMSE_z']:.2f})", {})],
], size=12, bullets=True)

# 14 — Forecasts ---------------------------------------------------------------------------------
s = new_slide("August 2026 forecasts",
              "pm2.5 = z × σ + μ. Daily forecasts are near-constant because August weather is climatological; the hourly forecast shows the morning and evening peaks.")
picture(s, FIG / "05a_forecast_calendar.png", LEFT, 1.95, RIGHT - LEFT, 2.3)
picture(s, FIG / "05c_forecast_hourly_aug01.png", LEFT, 4.3, 8.3, 2.4)
textbox(s, LEFT + 8.5, 4.35, 3.0, 2.3, [
    "Daily (1–16 Aug): 93% of sensor-days Moderate, 6% USG",
    "1 Aug peaks at 06:00–08:00 and 21:00",
    "Band = ±1 test RMSE (≈ 20 µg/m³)",
], size=12, bullets=True)

# 15 — Bonus map -----------------------------------------------------------------------------------
s = new_slide("Bonus: at-risk sensors, 1 Aug 13:00",
              "Natural Earth shapefiles. At midday the boundary layer is deep, so only 2 of 149 sensors are forecast in the USG range.")
picture(s, FIG / "05e_bonus_map_13h.png", LEFT, 1.95, 7.6, 4.75)
textbox(s, LEFT + 7.8, 2.3, 3.7, 4.2, [
    [("2 of 149 ", {"bold": True, "color": CRIMSON}), ("sensors in the USG → Very Unhealthy range", {})],
    [("Sir Apollo Kagwa Road ", {"bold": True}), ("≈ 50 µg/m³", {})],
    [("Kamwokya Primary School ", {"bold": True}), ("≈ 37 µg/m³", {})],
    "Both in central Kampala: dense traffic and roadside exposure",
    "At 07:00 or 21:00 many more sites cross 35.5 µg/m³",
], size=14, bullets=True)

# 16 — Evidently -----------------------------------------------------------------------------------
s = new_slide("Monitoring with Evidently AI",
              "Wasserstein distance > 0.1 or Jensen-Shannon > 0.1 marks drift. Temperature and humidity drift is seasonal; target drift means the pollution regime changed.")
picture(s, FIG / "06a_evidently_drift.png", LEFT, 1.95, 7.3, 2.6)
table(s, LEFT, 4.75, 5.4, [
    ["Metric (z units)", "Train", "Test"],
    ["RMSE", "0.52", "0.70"],
    ["MAE", "0.35", "0.43"],
    ["Mean error (bias)", "0.00", "−0.01"],
    ["R²", "0.73", "0.22"],
], [2.6, 1.4, 1.4], size=12, row_h=0.36)
textbox(s, LEFT + 7.5, 2.1, 4.0, 4.6, [
    [("Data drift ", {"bold": True, "color": NAVY}), ("5 of 11 columns drift train → test: temperature, humidity, season flag, target and prediction", {})],
    [("No drift ", {"bold": True, "color": NAVY}), ("in hour, location or device mix: coverage is stable", {})],
    [("Model drift ", {"bold": True, "color": CRIMSON}), ("RMSE +36% on new data with bias ≈ 0: generalisation gap, not a systematic shift", {})],
    [("Action ", {"bold": True}), ("retrain when RMSE rises > 20% or bias moves from 0; keep the previous version for rollback", {})],
], size=13, bullets=True)

# 17 — RunPod --------------------------------------------------------------------------------------
s = new_slide("Deployment on RunPod",
              "Debugged locally first (full run ≈ 4 min on 4 CPUs). On the pod, run deployment/run_on_runpod.sh; it logs GPU, pod ID, stage timings and metrics. Terminate the pod afterwards.")
steps = [("1", "Pod", "Community Cloud GPU under $2/h (RTX A4000 / 3090 ≈ $0.2–0.7/h)"),
         ("2", "Template", "RunPod PyTorch 2.x with Jupyter Lab, 20 GB disk"),
         ("3", "Import", "git clone the repo / upload data, pip install -r requirements"),
         ("4", "Run", "bash deployment/run_on_runpod.sh, XGBoost on CUDA"),
         ("5", "Evidence", "pipeline_run.log, run_summary_runpod.json, screenshots"),
         ("6", "Stop", "download outputs, terminate the pod (≈ $0.10 total)")]
for i, (n, head, body) in enumerate(steps):
    col, row = i % 3, i // 3
    x, y = LEFT + col * 3.9, 2.15 + row * 2.25
    card_box(s, x, y, 3.6, 2.0)
    c = s.shapes.add_shape(MSO_SHAPE.OVAL, Inches(x + 0.2), Inches(y + 0.2), Inches(0.55), Inches(0.55))
    c.fill.solid(); c.fill.fore_color.rgb = CRIMSON if i == 5 else NAVY; c.line.fill.background()
    c.text_frame.text = n
    c.text_frame.paragraphs[0].alignment = PP_ALIGN.CENTER
    c.text_frame.paragraphs[0].runs[0].font.size = Pt(16); c.text_frame.paragraphs[0].runs[0].font.bold = True
    c.text_frame.paragraphs[0].runs[0].font.color.rgb = WHITE
    textbox(s, x + 0.9, y + 0.22, 2.6, 0.5, [head], size=17, color=NAVY, bold_first=True)
    textbox(s, x + 0.2, y + 0.85, 3.25, 1.1, [body], size=13)

# 18 — Conclusions ---------------------------------------------------------------------------------
s = new_slide("Conclusions and next steps", "Be upfront about the data limitation: seven days of history caps what any model can learn.")
card_box(s, LEFT, 2.15, 5.55, 4.45)
textbox(s, LEFT + 0.25, 2.25, 5.1, 0.5, ["What we found"], size=18, color=NAVY, bold_first=True)
textbox(s, LEFT + 0.25, 2.8, 5.1, 3.7, [
    "Kampala air is mostly Moderate to USG, with Unhealthy spikes in the peak dry season",
    "Clear diurnal cycle; cool, humid hours are dirtier",
    f"{best} is the best model: RMSE {bm['RMSE_ugm3']:.1f} µg/m³, R² {bm['R2']:.2f}, {bm['AQI_accuracy_%']:.0f}% AQI accuracy",
    "Tree ensembles beat linear, KNN and SVR models",
], size=14, bullets=True)
card_box(s, LEFT + 5.95, 2.15, 5.55, 4.45, fill=RGBColor(0xFF, 0xF8, 0xD6))
textbox(s, LEFT + 6.2, 2.25, 5.1, 0.5, ["Limitations → next steps"], size=18, color=CRIMSON, bold_first=True)
textbox(s, LEFT + 6.2, 2.8, 5.1, 3.7, [
    "Only 7 days of history → pull months of data from the AirQo API",
    "Climatological weather → use UNMA / Open-Meteo forecasts",
    "No lag features → add a nowcasting model for the next hours",
    "Automate Evidently checks, retraining and model rollback",
], size=14, bullets=True)

# 19 — Last slide ----------------------------------------------------------------------------------
s = prs.slides.add_slide(L_LAST)
s.notes_slide.notes_text_frame.text = "Thank you. Questions?"

prs.save(OUT)
print("saved", OUT, len(prs.slides), "slides")
