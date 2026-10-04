// Builds the Word study guide for Assignment 4 from the pipeline outputs.
// Run from the repo root after the notebook has been executed:  node notebook_src/build_study_guide.js
const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, AlignmentType, Table, TableRow, TableCell,
  WidthType, ShadingType, BorderStyle, ImageRun, LevelFormat, PageBreak, TableOfContents, Footer,
  PageNumber, Header,
} = require("docx");

const ROOT = path.resolve(__dirname, "..");
const FIG = path.join(ROOT, "outputs", "figures");
const card = JSON.parse(fs.readFileSync(path.join(ROOT, "models", "best_model_card.json")));
const M = card.test_metrics;
const comp = fs.readFileSync(path.join(ROOT, "outputs", "model_comparison.csv"), "utf8").trim().split("\n").map((l) => l.split(","));

const NAVY = "0B3D91", CRIMSON = "D7014D", MUTED = "5F6B7A", TINT = "EEF3FA", GOLDTINT = "FFF8D6", ROSE = "FBECF1";
const PAGE_W = 11906, MARGIN = 1134, CONTENT_W = PAGE_W - 2 * MARGIN; // A4, 2 cm margins

// ---------- helpers ----------
// **bold** and _italic_ inline markup
function runs(text, base = {}) {
  const out = [];
  const re = /(\*\*[^*]+\*\*|~[^~]+~)/g;
  let last = 0, m;
  while ((m = re.exec(text)) !== null) {
    if (m.index > last) out.push(new TextRun({ text: text.slice(last, m.index), ...base }));
    const t = m[0];
    if (t.startsWith("**")) out.push(new TextRun({ text: t.slice(2, -2), bold: true, ...base }));
    else out.push(new TextRun({ text: t.slice(1, -1), italics: true, ...base }));
    last = m.index + t.length;
  }
  if (last < text.length) out.push(new TextRun({ text: text.slice(last), ...base }));
  return out;
}
const P = (text, opts = {}) => new Paragraph({ children: runs(text, opts.run || {}), spacing: { after: 120, line: 276 }, ...opts.para });
const H1 = (text) => new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun(text)], pageBreakBefore: true });
const H2 = (text) => new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun(text)] });
const H3 = (text) => new Paragraph({ heading: HeadingLevel.HEADING_3, children: [new TextRun(text)] });
const B = (text, level = 0) => new Paragraph({ numbering: { reference: "bullets", level }, children: runs(text), spacing: { after: 60, line: 264 } });
let LIST = 0;
const newList = () => { LIST += 1; };
const N = (text) => new Paragraph({ numbering: { reference: "numbers", level: 0, instance: LIST }, children: runs(text), spacing: { after: 60, line: 264 } });

function img(file, widthIn, caption) {
  const p = path.join(FIG, file);
  const buf = fs.readFileSync(p);
  // PNG width/height from IHDR
  const w = buf.readUInt32BE(16), h = buf.readUInt32BE(20);
  const wpx = widthIn * 96, hpx = (wpx * h) / w;
  const out = [new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 120, after: 60 },
    children: [new ImageRun({ type: "png", data: buf, transformation: { width: wpx, height: hpx },
      altText: { title: caption || file, description: caption || file, name: file } })] })];
  if (caption) out.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 200 },
    children: [new TextRun({ text: caption, italics: true, size: 18, color: MUTED })] }));
  return out;
}

const border = { style: BorderStyle.SINGLE, size: 4, color: "C9D3E3" };
const borders = { top: border, bottom: border, left: border, right: border };

function table(rows, widths, opts = {}) {
  const total = widths.reduce((a, b) => a + b, 0);
  return new Table({
    width: { size: total, type: WidthType.DXA }, columnWidths: widths,
    rows: rows.map((r, i) => new TableRow({
      tableHeader: i === 0,
      children: r.map((c, j) => new TableCell({
        borders, width: { size: widths[j], type: WidthType.DXA },
        shading: { fill: i === 0 ? NAVY : (opts.zebra !== false && i % 2 === 0 ? TINT : "FFFFFF"), type: ShadingType.CLEAR },
        margins: { top: 60, bottom: 60, left: 100, right: 100 },
        children: [new Paragraph({ spacing: { after: 20 },
          children: runs(String(c), i === 0 ? { bold: true, color: "FFFFFF", size: 19 } : { size: 19 }) })],
      })),
    })),
  });
}

// shaded call-out box (single-cell table)
function box(title, lines, fill = TINT, titleColor = NAVY) {
  const children = [new Paragraph({ spacing: { after: 80 }, children: [new TextRun({ text: title, bold: true, color: titleColor, size: 22 })] })];
  lines.forEach((l) => children.push(typeof l === "string"
    ? new Paragraph({ spacing: { after: 60, line: 264 }, children: runs(l, { size: 20 }) }) : l));
  return new Table({
    width: { size: CONTENT_W, type: WidthType.DXA }, columnWidths: [CONTENT_W],
    rows: [new TableRow({ children: [new TableCell({
      width: { size: CONTENT_W, type: WidthType.DXA }, shading: { fill, type: ShadingType.CLEAR },
      borders: { top: { style: BorderStyle.NONE, size: 0, color: "FFFFFF" }, bottom: { style: BorderStyle.NONE, size: 0, color: "FFFFFF" },
        left: { style: BorderStyle.NONE, size: 0, color: "FFFFFF" }, right: { style: BorderStyle.NONE, size: 0, color: "FFFFFF" } },
      margins: { top: 140, bottom: 140, left: 200, right: 200 }, children })] })],
  });
}
const gap = () => new Paragraph({ spacing: { after: 80 }, children: [] });

// metric explainer: definition / large / small / decision
function metric(name, def, large, small, decision) {
  return [
    H3(name),
    table([
      ["", "Meaning"],
      ["**What it is**", def],
      ["**Large value means**", large],
      ["**Small value means**", small],
      ["**Decision you can make**", decision],
    ], [2300, CONTENT_W - 2300], { zebra: true }),
    gap(),
  ];
}

const f1 = (x) => Number(x).toFixed(1), f2 = (x) => Number(x).toFixed(2);

// ---------- content ----------
const c = [];

// Title page
c.push(new Paragraph({ spacing: { before: 2400, after: 200 }, alignment: AlignmentType.LEFT,
  children: [new TextRun({ text: "STUDY GUIDE", bold: true, size: 28, color: CRIMSON, characterSpacing: 40 })] }));
c.push(new Paragraph({ spacing: { after: 200 }, children: [new TextRun({ text: "Predicting PM2.5 Air Quality from AirQo Sensor Data", bold: true, size: 56, color: NAVY })] }));
c.push(new Paragraph({ spacing: { after: 600 }, children: [new TextRun({ text: "CSC3119 AI Deployment & Scalability · Assignment 4 · Uganda Christian University", size: 26, color: MUTED })] }));
c.push(box("How to use this guide", [
  "This guide explains **what was done in the notebook, why it was done, and how to interpret every result** — so you can present the work, answer viva questions and write the report with confidence.",
  "Each section follows the pipeline order: data → EDA → models → metrics → forecasts → monitoring → deployment. The **metric sections** tell you what a large or small value means and what decision it should lead to.",
  "Look for the boxes: **Key idea** (core concept), **Exam / viva tip** (likely question), and **Our result** (the numbers from this project).",
]));
c.push(gap());
c.push(table([
  ["Key number", "Value", "Where it comes from"],
  ["Raw rows / sensors", "14,600 / 149", "Input CSV"],
  ["Observation windows", "1–3 Jan, 1–2 Apr, 1–3 Jun 2026", "Data audit (not continuous Jan–Jul)"],
  ["Hourly grid", "758,112 rows, 1.9% observed", "149 sensors × 5,088 hours"],
  ["Target scaler (train)", `μ = ${f1(card.mu)} µg/m³, σ = ${f1(card.sigma)} µg/m³`, "z = (x − μ)/σ"],
  ["Best model", card.model_name, "Lowest hold-out RMSE"],
  ["Test RMSE / MAE", `${f1(M.RMSE_ugm3)} / ${f1(M.MAE_ugm3)} µg/m³`, "Last 24 h (2–3 Jun)"],
  ["Test R² / AQI accuracy", `${f2(M.R2)} / ${f1(M["AQI_accuracy_%"])}%`, "Same test set"],
], [2600, 3300, CONTENT_W - 5900]));
c.push(new Paragraph({ children: [new PageBreak()] }));
c.push(new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun("Contents")] }));
c.push(new TableOfContents("Contents", { hyperlink: true, headingStyleRange: "1-2" }));
c.push(P("~If the contents list is empty, right-click it in Word and choose Update Field.~", { run: { color: MUTED } }));

// 1. Problem
c.push(H1("1. The problem in one page"));
c.push(P("**PM2.5** is particulate matter smaller than 2.5 micrometres — about 30 times thinner than a human hair. It passes the nose and throat, reaches deep into the lungs and can enter the bloodstream. It is linked to asthma, heart disease, stroke and premature death. **PM10** (up to 10 µm) mostly comes from road dust and construction and stops in the upper airways."));
c.push(P("Kampala's annual mean PM2.5 (2018–2021) was **39 µg/m³ — about 8× the WHO guideline of 5 µg/m³**, and about 7,257 deaths were attributed to long-term exposure. Forecasting PM2.5 lets the health sector, NEMA, the Ministry of Energy and the Ministry of Works & Transport act **before** bad-air days."));
c.push(H2("The AQI categories used everywhere"));
c.push(table([
  ["PM2.5 (µg/m³)", "Category", "What it means for people"],
  ["0 – 9", "Good", "Safe for everyone"],
  ["9.1 – 35.49", "Moderate", "Acceptable; very sensitive people may react"],
  ["35.49 – 55.49", "Unhealthy for Sensitive Groups (USG)", "Children, elderly, asthma/heart patients should limit outdoor exertion"],
  ["55.49 – 125.49", "Unhealthy", "Everyone may feel effects"],
  ["125.49 – 225.49", "Very Unhealthy", "Health alert"],
  ["> 225.49", "Hazardous", "Emergency conditions"],
], [2000, 3200, CONTENT_W - 5200]));
c.push(gap());
c.push(box("Key idea — why temperature and humidity matter for PM2.5 but not PM10", [
  "Cool, humid nights and early mornings create a **shallow boundary layer / temperature inversion** — a lid of warm air that traps smoke from cooking and traffic near the ground. In the warm afternoon the boundary layer deepens and pollution dilutes. Humidity also makes particles absorb water and grow (**hygroscopic growth**).",
  "PM10 is mostly mechanical dust (roads, construction), so it responds far less to temperature and humidity. That is why the two pollutants are not modelled the same way, and why **PM10 is not used as a predictor** in our model.",
]));

// 2. Data
c.push(H1("2. The data and how we prepared it"));
c.push(H2("2.1 What the audit revealed"));
c.push(B("The file says ~January–July 2026~ but contains only **three short windows**: 1–3 Jan, 1–2 Apr and 1–3 Jun (147 distinct hours). Every sensor has 20–147 rows."));
c.push(B("**Zeros mean null** (brief, §5): 28% of temperature, 30% of humidity and 21% of coordinates are 0. Uganda is never 0 °C, and no sensor sits at 0°N 0°E."));
c.push(B("**frequency** is constant and **site_id** is 51% empty and duplicates site_name — both dropped."));
c.push(B("PM2.5 reaches **998 µg/m³** — a sensor-saturation value, not real air."));
c.push(B("GPS positions jitter (up to 119 distinct fixes for one fixed sensor); 6 devices never report a valid fix."));
c.push(H2("2.2 Every cleaning decision and its justification"));
c.push(table([
  ["Problem", "What we did", "Why (what to say in a viva)"],
  ["0 / empty in temp, RH, PM, lat, lon", "Converted to NaN", "The brief defines 0 as null; keeping zeros would drag means down and teach the model false relationships."],
  ["Mixed text/number types, UTC strings", "Datetime → Kampala local time (EAT); numeric → float64; IDs → string", "Consistent types (brief requirement). Local time matters: rush hours happen at local 07:00, not UTC 07:00."],
  ["PM2.5 > 500 µg/m³ (3 rows)", "Set to NaN and logged", "Beyond the effective range of low-cost optical sensors (Barkjohn et al., 2021)."],
  ["Statistical outliers (650 rows beyond 1.5×IQR per sensor)", "**Flagged, kept**", "They happen at rush hours and night inversions: real pollution. Removing them makes the model blind to unhealthy hours."],
  ["Short gaps ≤ 3 h", "Linear interpolation in time, per sensor", "Junninen et al. (2004): linear interpolation is the most accurate method for short gaps."],
  ["Long gaps (device offline)", "Left as NaN, is_observed = 0, not used as targets", "Filling weeks of missing data invents labels the model would learn from (Okure et al., 2022)."],
  ["Missing temp/RH at an observed hour", "Inverse-distance weighting of the 3 nearest sensors (≤ 50 km) at the same hour → sensor hour-median → network median", "Weather is spatially smooth, so neighbours are the best proxy (the team's 'closest location' idea)."],
  ["Missing / jittery coordinates", "One median location per device; documented gazetteer for 6 devices", "Sensors are fixed; the median removes GPS noise."],
  ["Full hourly time index", "Every sensor gets every hour 1 Jan – 31 Jul (758,112 rows)", "Brief: offline hours must be **missing rather than absent**, which makes outages visible."],
], [2500, 3100, CONTENT_W - 5600]));
c.push(gap());
c.push(box("Exam / viva tip — 'Why didn't you just drop all rows with nulls?'", [
  "Because 28–30% of rows have a null temperature or humidity. Dropping them would throw away a third of the PM2.5 labels, and not at random: whole sensors without weather probes would disappear, biasing the model toward certain locations. Imputing weather spatially keeps those PM2.5 labels while only filling the predictor.",
]));
c.push(H2("2.3 Feature selection"));
c.push(table([
  ["Feature", "Used?", "Reason"],
  ["Hour of day as sin/cos", "Yes", "Captures the daily cycle. Sin/cos make 23:00 and 00:00 neighbours (a plain 0–23 number puts them 23 apart)."],
  ["Temperature, humidity", "Yes", "Physical drivers of PM2.5 (boundary layer, particle growth)."],
  ["Latitude, longitude", "Yes", "Spatial pattern: central Kampala vs suburbs vs upcountry towns."],
  ["Peak dry season flag (Dec–Feb)", "Yes", "January daily means ≈ 44–55 µg/m³ vs ≈ 21–28 in April and June. August is coded 0, so it is expected to look like June."],
  ["Device level (season-adjusted target encoding)", "Yes", "Each site's typical deviation from the network average in the same period (roadside / market sites high, suburbs low)."],
  ["Network (AirQo vs AirGradient)", "Yes", "Hardware may have systematic offsets."],
  ["PM10", "**No**", "Same sensor, same moment → leakage; not known in a real forecast; dust-driven."],
  ["Day of week", "No", "Only Monday–Friday were observed (7 days total), so a weekday effect can't be learned."],
], [3000, 900, CONTENT_W - 3900]));
c.push(gap());
c.push(box("Key idea — target encoding and leakage", [
  "**Target encoding** replaces a category (the device name) with the average target for that category. It is powerful but can **leak**: if computed on the test rows the model sees the answers. We computed it **on training rows only**, adjusted for season (some sensors were only online in dirty January), and **shrunk** toward 0 for devices with few rows: (n·mean)/(n + 10).",
]));

// 3. EDA
c.push(H1("3. Exploratory data analysis (EDA)"));
c.push(P("EDA uses 5 Kampala sensors with full weather data and the most observations: **Nansana east ward (Wakiso), Nakasero II, Lower Nsooba (Kawempe), Civic Centre (Kampala Central) and Salama Road**."));
c.push(H2("3.1 Distributions"));
c.push(...img("03_distributions.png", 6.5, "Figure 1. Distributions of PM2.5, temperature and humidity (dashed lines = AQI thresholds)."));
c.push(P("**How to read it:** the histogram shows how often each value occurs; the curve is a smoothed version (KDE). PM2.5 has a long tail to the right — **skewness ≈ 2.1** (above 1 = strongly skewed): many moderate hours, a few very high spikes. The mean (45.7 µg/m³) is pulled above the median (31.7 µg/m³) by those spikes."));
c.push(P("**What it means for modelling:** skewed targets make squared-error metrics (RMSE) dominated by a few spikes, and models tend to under-predict extremes. That is why we report MAE next to RMSE and check how the model does on high values."));
c.push(H2("3.2 Box plots and outliers"));
c.push(...img("03c_boxplots.png", 6.5, "Figure 2. Box plots per sensor. Red points = beyond 1.5 × IQR."));
c.push(P("**How to read a box plot:** the box spans the 25th to 75th percentile (the **interquartile range, IQR**); the line inside is the median; whiskers reach 1.5 × IQR beyond the box; points outside are flagged outliers."));
c.push(P("**Our result:** Salama Road and Civic Centre have the highest medians and longest tails. Outliers were **flagged and kept**: Civic Centre 17, Nansana 11, Salama Road 5. Tree models handle such spikes well; for distance-based models we standardise features."));
c.push(H2("3.3 Time series with AQI bands"));
c.push(...img("03d_timeseries_aqi.png", 6.5, "Figure 3. Hourly PM2.5 at 5 Kampala sensors, shaded by AQI category."));
c.push(P("**Our result:** January (peak dry season) sits mostly in **Unhealthy / Very Unhealthy**; April and June are mostly **Moderate** with spikes. Every day shows a **morning peak (≈ 06:00–09:00)** and an **evening peak (≈ 19:00–23:00)**, matching traffic, cooking and night inversions."));
c.push(...img("03e_diurnal_profile.png", 6.0, "Figure 4. Average PM2.5 by hour of day (local time)."));
c.push(H2("3.4 Calendar plot"));
c.push(...img("03f2_calendar_observed_months.png", 4.6, "Figure 5. Daily mean PM2.5 coloured by AQI (months with data; the notebook also has the full Jan–Jul calendar)."));
c.push(P("**How to read it:** each square is a day, coloured by its daily mean's AQI category; grey = no data. It shows at a glance that January was worst and that the network had big gaps. The full Jan–Jul version in the notebook makes the gaps obvious: most of the calendar is grey."));
c.push(H2("3.5 The Z-score transformation"));
c.push(box("Key idea — Z-score", [
  "**z = (x − μ) / σ**, where μ is the mean and σ the standard deviation. The result says **how many standard deviations a value is above or below average**. To go back: **x = z·σ + μ** (the inverse transform used on forecasts).",
  "z = 0 → an average hour. z = +1 → one σ dirtier than average. z = −1 → one σ cleaner. |z| > 2 is unusual (≈ 5% of hours if the data were normal); |z| > 3 is rare (≈ 0.3%).",
]));
c.push(gap());
c.push(...img("03h_zscore.png", 6.5, "Figure 6. Global Z-score (left; dashed lines are AQI thresholds in Z units) and per-sensor Z-score (right)."));
c.push(table([
  ["AQI threshold (µg/m³)", "Category above it", "Equivalent global z (EDA, all data)"],
  ["9.0", "Moderate", "−0.86"],
  ["35.49", "USG", "+0.09"],
  ["55.49", "Unhealthy", "+0.81"],
  ["125.49", "Very Unhealthy", "+3.34"],
  ["225.49", "Hazardous", "+6.94"],
], [2800, 2600, CONTENT_W - 5400]));
c.push(gap());
c.push(P("**Interpretation:** the USG threshold is at z ≈ 0.09, so **'above average' in this network already means unhealthy for sensitive groups**. Z-scores range from ≈ −1.0 (PM2.5 can't go below 0) up to ≈ +13.8: scaling does **not remove skew**. 4.7% of hours have |z| > 2 and 1.5% have |z| > 3 — far more than a normal distribution would give, confirming heavy tails."));
c.push(P("**Global vs per-sensor Z:** the model target uses the **global** z (one μ and σ, easy to invert, keeps differences between sites). A **per-sensor** z says how unusual an hour is ~for that location~, which is useful for anomaly alerts (e.g. a nearby fire at a normally clean site)."));
c.push(box("Exam / viva tip — 'Why fit μ and σ on the training set only?'", [
  "If μ and σ were computed on all data, information from the test period would leak into training. We describe the data with all-data Z in EDA, but the **model's scaler uses training rows only** (μ = " + f1(card.mu) + ", σ = " + f1(card.sigma) + " µg/m³) and is saved with the model for the inverse transform.",
]));
c.push(H2("3.6 Correlation with temperature and humidity"));
c.push(...img("03i_correlation.png", 6.5, "Figure 7. Pearson correlations (left) and pooled scatter plots."));
c.push(P("**Pearson r** measures straight-line association from −1 to +1. **Spearman ρ** works on ranks, so it captures any monotonic relationship and is robust to outliers — useful because PM2.5 is skewed."));
c.push(table([
  ["|r| value", "Strength", "What we can conclude"],
  ["< 0.1", "None / negligible", "Variable alone tells you almost nothing"],
  ["0.1 – 0.3", "Weak", "Useful hint, needs other features"],
  ["0.3 – 0.5", "Moderate", "Clear relationship"],
  ["> 0.5", "Strong", "Major driver (but correlation ≠ causation)"],
], [1800, 2400, CONTENT_W - 4200]));
c.push(gap());
c.push(P("**Our result:** temperature is **negative** (r from −0.08 at Civic Centre to −0.50 at Salama Road) and humidity **positive** (up to +0.41). Pooled: r = −0.19 (temperature) and +0.14 (humidity). The physics holds, but weather alone is not enough: location and time of day carry much of the signal. At Civic Centre, traffic emissions swamp the weather effect (r ≈ 0)."));

// 4. Modelling
c.push(H1("4. Machine-learning models"));
c.push(H2("4.1 Set-up: target, split and validation"));
c.push(B("**Target:** global Z-score of PM2.5 (brief §7.3). Predictions are converted back to µg/m³ with x = z·σ + μ."));
c.push(B("**Chronological split:** train = Jan, Apr and 1 Jun (13,780 rows); test = the **last full 24 hours** (2 Jun 06:00 → 3 Jun 06:00, 2,468 rows). This mimics forecasting the future, covers a whole day–night cycle, and is in the same season as August."));
c.push(B("**Blocked cross-validation for tuning:** GroupKFold with each **calendar day** as a group (Roberts et al., 2017). Whole days are held out, so neighbouring, highly similar hours never sit on both sides of a fold."));
c.push(box("Exam / viva tip — 'Why not a random train/test split?'", [
  "PM2.5 at 10:00 is very similar to 09:00 and 11:00 (**autocorrelation**). A random split puts these near-copies in both train and test, so the model looks far better than it really is. A time-based split tests what we actually need: predicting hours the model has never seen.",
  "We first tried TimeSeriesSplit and rejected it: its early folds trained only on January (dry, dirty) and validated on April (wet, clean), so the season flag never varied in training and tuning chose over-smoothed models.",
]));
c.push(gap());
c.push(box("Key idea — baselines", [
  "A **baseline** is a 'dumb' prediction every model must beat. **Global mean:** always predict the average. **Sensor mean:** predict each sensor's usual level. If a model doesn't beat these, it has learned nothing useful. All our models beat both.",
], GOLDTINT, CRIMSON));
c.push(H2("4.2 The models and their hyperparameters"));
c.push(P("**Hyperparameters** are settings chosen before training (not learned from data). We tuned them with **RandomizedSearchCV**, which tries random combinations and keeps the one with the lowest cross-validated RMSE. The table says what happens when each setting goes up or down."));
c.push(table([
  ["Model", "Hyperparameter", "Larger value →", "Smaller value →"],
  ["Ridge regression", "alpha (L2 penalty)", "Coefficients shrink → simpler, risk of **underfitting**", "≈ ordinary least squares → risk of **overfitting** with correlated inputs"],
  ["", "polynomial degree", "More curves/interactions, more overfitting risk", "Straight-line model"],
  ["KNN", "n_neighbors (k)", "Smoother, averages more hours → misses spikes (bias)", "Copies a few neighbours → noisy (variance)"],
  ["", "weights = distance", "Closer neighbours count more", "(uniform) all k count equally"],
  ["SVR (RBF)", "C", "Fits training errors hard → overfits spikes", "Tolerates errors → smoother, may underfit"],
  ["", "gamma", "Very local, wiggly function", "Very smooth, almost linear"],
  ["", "epsilon", "Wider no-penalty tube → ignores more small errors", "Narrow tube → tries to fit every small error"],
  ["Random Forest", "n_estimators", "More trees → more stable (slower)", "Fewer trees → noisier predictions"],
  ["", "max_depth / min_samples_leaf", "Deeper / smaller leaves → more detailed rules, overfit", "Shallower / bigger leaves → simpler, underfit"],
  ["", "max_features", "Trees look alike (less diversity)", "More diverse trees, better averaging"],
  ["XGBoost", "learning_rate", "Big steps → fast but can overshoot/overfit", "Small careful steps → needs more trees"],
  ["", "n_estimators", "More boosting rounds → fits more (overfit if too many)", "Too few rounds → underfit"],
  ["", "max_depth, min_child_weight", "Depth ↑ / weight ↓ → more complex trees", "Simpler trees"],
  ["", "subsample, colsample_bytree", "Close to 1 → uses all rows/features each round", "Lower → more randomness, less overfitting"],
  ["", "reg_lambda", "Stronger L2 regularisation of leaf values", "Weaker regularisation"],
  ["Stacking", "meta-learner alpha", "Meta-model averages base models more evenly", "Trusts individual base models more"],
], [1650, 2150, 2950, CONTENT_W - 6750]));
c.push(gap());
c.push(box("Our result — best hyperparameters found", [
  "**XGBoost (best):** learning_rate 0.014, n_estimators 204, max_depth 8, min_child_weight 3, subsample 0.91, colsample_bytree 0.79, reg_lambda 1.96.",
  "**Random Forest:** 423 trees, max_depth None, min_samples_leaf 7, max_features 0.4.  **KNN:** k = 24, distance weights, Manhattan distance.  **SVR:** C = 2.09, gamma = 0.031, epsilon = 0.071.  **Ridge:** alpha ≈ 99 with degree-2 terms.",
  "**Stacking:** meta-weights RF 0.36, XGB 0.53, SVR 0.22, KNN −0.10 (a negative weight means KNN mostly repeats what the trees say, and the meta-model corrects for it).",
]));

// 5. Metrics
c.push(H1("5. Evaluation metrics: what they mean and what to decide"));
c.push(P("All error metrics are computed on the Z scale and, after inverse transform, in **µg/m³** (RMSE in µg/m³ = RMSE in z × σ). Below: what each metric is, what a **large** or **small** value means, and the decision it should drive."));
c.push(...metric("RMSE — Root Mean Squared Error (our primary metric)",
  "√(mean of (observed − predicted)²). Same units as the target (µg/m³ or z). Squaring makes large errors count much more than small ones.",
  "Predictions are often far off, **or** a few hours are badly missed (spikes). RMSE in z close to 1 means the model is barely better than predicting the mean, because the target's standard deviation is 1.",
  "Predictions are close to reality, including on spikes. RMSE in z of 0.5 = typical error of half a standard deviation (≈ 14 µg/m³ here).",
  "Choose the model with the lowest **test** RMSE (we did). Use it when big misses are costly, as here: under-predicting a pollution spike means people are not warned. If RMSE rises in monitoring → investigate drift and retrain."));
c.push(...metric("MAE — Mean Absolute Error",
  "Mean of |observed − predicted|. Every error counts in proportion to its size; robust to outliers. Easiest to explain: 'on average we are off by X µg/m³'.",
  "Typical hourly error is large; forecasts can't separate neighbouring AQI bands reliably.",
  "Typical error is small; the model is reliable on ordinary hours.",
  "Use MAE to communicate accuracy to non-experts. **Compare RMSE with MAE:** if RMSE ≫ MAE, errors are dominated by a few big misses → focus on spikes (more data on extreme episodes, quantile models)."));
c.push(...metric("R² — Coefficient of determination",
  "1 − (model's squared error ÷ squared error of always predicting the mean). The share of variation in PM2.5 the model explains. 1 = perfect, 0 = no better than the mean, negative = worse than the mean.",
  "Close to 1: the model explains most of the ups and downs. (Suspicious near 1 on time series: check for leakage.)",
  "Near 0 or negative: the model captures little beyond the average level. Negative test R² means it is worse than a flat line on new data (e.g. a level shift between seasons).",
  "R² compares models on the same test set and judges whether features carry signal. Low R² with acceptable MAE means the model gets typical levels right but not the hour-to-hour swings → add better predictors (weather forecasts, recent readings)."));
c.push(...metric("AQI-category accuracy (and the confusion matrix)",
  "Share of hours where the predicted AQI band (Good, Moderate…) equals the observed band. The confusion matrix shows which bands get mixed up.",
  "Most hours land in the right band: useful for public messaging.",
  "Many hours land in the wrong band.",
  "**Read it with care**: when most hours are Moderate, a model that always says 'Moderate' scores high. Our global-mean baseline gets 73% this way! For health warnings, track **recall on unhealthy hours** (share of real USG+ hours we caught) instead."));
c.push(...metric("Cross-validation RMSE and the overfitting gap",
  "CV RMSE = average RMSE over held-out days during tuning. Overfitting gap = test RMSE − train RMSE.",
  "Large gap: the model memorised training data and generalises poorly (overfitting). Our KNN has train RMSE 0.06 vs test 0.73: distance weighting lets it copy training points exactly.",
  "Small gap: train and test performance agree. A small gap with high error on both = **underfitting** (model too simple, e.g. Ridge).",
  "Large gap → regularise more (bigger alpha, larger min_samples_leaf, smaller learning rate) or get more data. Small gap with high error → more flexible model or better features."));
c.push(box("Why we did not use MAPE", [
  "MAPE (mean absolute percentage error) divides by the observed value. On **Z-scores** values cross zero, so MAPE explodes (Evidently reports astronomical MAPE on z). Even in µg/m³ it over-punishes errors at very clean hours (an error of 5 on an observed 5 = 100%). Use it only when all true values are well above zero.",
  "**When a different metric is better:** for a yes/no health alert ('will PM2.5 exceed 55?') use **precision / recall / F1** on that event; when every error matters equally, use MAE as primary; for skewed targets, RMSE on log-PM2.5 is also common.",
], GOLDTINT, CRIMSON));
c.push(H2("5.1 Our results"));
const rows = [["Model", "RMSE (z)", "RMSE µg/m³", "MAE µg/m³", "R²", "AQI acc.", "Train RMSE (z)"]];
comp.slice(1).forEach((r) => rows.push([r[0], f2(r[1]), f1(r[4]), f1(r[5]), f2(r[3]), f1(r[6]) + "%", r[8] ? f2(r[8]) : "–"]));
c.push(table(rows, [2300, 1050, 1250, 1150, 850, 1100, CONTENT_W - 7700]));
c.push(...img("04a_model_comparison.png", 6.6, "Figure 8. Test RMSE, R² and AQI accuracy for every model."));
c.push(H3("How to interpret these results"));
newList();
c.push(N(`**${card.model_name} is best** (RMSE ${f1(M.RMSE_ugm3)} µg/m³), with the stacking ensemble and Random Forest within 0.1 µg/m³. The tree ensembles are effectively tied; XGBoost was saved because it has the lowest RMSE and is small and fast.`));
c.push(N("**All models beat both baselines** (≈ 24 µg/m³), so features add real information: about 4 µg/m³ less error than the sensor-mean baseline."));
c.push(N(`**R² = ${f2(M.R2)}**: the model explains about 22% of hour-to-hour variance on unseen hours. That is modest but honest for a model with **no recent readings** (no lag features) trained on only 7 days. It captures typical levels and the daily cycle, not sudden events.`));
c.push(N(`**RMSE (${f1(M.RMSE_ugm3)}) is much larger than MAE (${f1(M.MAE_ugm3)})**: errors come from a few big misses. On hours above 35.5 µg/m³ the model under-predicts by ≈ 28 µg/m³ on average; on cleaner hours it over-predicts by ≈ 6. It **shrinks toward the middle**.`));
c.push(N("**SVR has the best AQI accuracy (74%) but the worst R²**: it predicts 'Moderate' most of the time, which scores well on accuracy but misses the swings. This is the clearest example of why one metric is not enough."));
c.push(N("**KNN overfits** (train 0.06 vs test 0.73); **Ridge underfits** (train 0.69 ≈ test 0.73, both high)."));

// 6. Diagnostics
c.push(H1("6. Best-model diagnostics"));
c.push(...img("04b_best_model_diagnostics.png", 6.6, "Figure 9. Predicted vs observed, residuals vs prediction, residual distribution (test set)."));
c.push(B("**Predicted vs observed:** a perfect model puts every point on the dashed diagonal. Our points flatten below the line at high values: spikes are under-predicted (the max prediction is ≈ 124 µg/m³; the max observed is 246)."));
c.push(B("**Residuals (observed − predicted):** centred near 0 (mean bias ≈ 0) but with a long positive tail: when the model is wrong, it is usually too low. That matters for public health, since it means missed warnings."));
c.push(...img("04c_aqi_confusion.png", 3.8, "Figure 10. AQI confusion matrix (rows = observed, columns = predicted)."));
c.push(P("**Our result:** 1,676 of 1,790 Moderate hours are correct, but of the 456 hours that were really USG or worse, the model flagged only **32% (recall)**; when it did predict USG or worse it was right **55% (precision)** of the time. Decision: for an early-warning service, lower the alert threshold (e.g. warn from 30 µg/m³ predicted) to raise recall, accepting more false alarms."));
c.push(...img("04d_feature_importance.png", 5.2, "Figure 11. Permutation importance: how much test RMSE worsens when one feature is shuffled."));
c.push(P("**How to read permutation importance:** shuffle one feature's values, breaking its link with the target, and measure how much the error grows. A big increase means the model relies on it."));
c.push(P("**Our result:** ~is_airgradient~ ranks first because the few AirGradient sensors in the test day are far more polluted (mean ≈ 58 vs 24 µg/m³ for AirQo devices), so this flag acts as a 'very polluted site' marker. Then device level and latitude (where you are), then temperature and hour of day (when, and under what weather). **The season flag scores 0 here** because it is constant (0) in the June test day: shuffling a constant changes nothing. That doesn't mean it is useless in training."));

// 7. Forecasting
c.push(H1("7. Forecasting August 2026"));
c.push(H2("7.1 How the forecast is produced"));
newList();
c.push(N("Re-load the saved model from **models/best_pm25_model.joblib** (proves the artifact works on its own)."));
c.push(N("Build future rows for every sensor and hour: 1–16 Aug for the daily forecast (covers both 'Aug 1–14' and 'the full weeks Mon 3 – Sun 16'), and the 24 hours of 1 Aug for the hourly forecast."));
c.push(N("Fill the features: hour, coordinates, device level, network and season flag are known. **Temperature and humidity are unknown**, so we use each sensor's **June hour-of-day average** (the closest observed month to August)."));
c.push(N("Predict z, then **inverse-transform**: PM2.5 = z × σ + μ. Daily value = mean of the 24 hourly predictions."));
c.push(box("Worked example of the inverse transform", [
  `σ = ${f2(card.sigma)} and μ = ${f2(card.mu)} µg/m³. If the model predicts z = 0.55 for a sensor: PM2.5 = 0.55 × ${f2(card.sigma)} + ${f2(card.mu)} = **${f1(0.55 * card.sigma + card.mu)} µg/m³** → USG band. A prediction of z = −0.5 gives ${f1(-0.5 * card.sigma + card.mu)} µg/m³ → Moderate.`,
]));
c.push(gap());
c.push(...img("05a_forecast_calendar.png", 6.6, "Figure 12. Daily forecast calendar, 1–16 August 2026 (5 sensors across zones)."));
c.push(...img("05c_forecast_hourly_aug01.png", 6.6, "Figure 13. Hourly forecast for Saturday 1 August 2026."));
c.push(H2("7.2 How to interpret the forecasts"));
c.push(B("**Daily (1–16 Aug):** 93% of sensor-days are **Moderate**, 6% **USG** and 1% **Unhealthy**. Central-Kampala and roadside sites are highest; Gulu and Jinja are lowest of the five shown."));
c.push(B("**Why are the daily values the same every day?** None of the inputs change from one August day to the next (same hours, same climatological weather, same season flag). The model gives a **climatological forecast**: the typical level for that site and season. With real daily weather forecasts the values would vary. Say this openly; it shows you understand the model."));
c.push(B("**Hourly (1 Aug):** clear **morning peak (06:00–08:00)** and **evening rise (≈ 21:00)**, with the cleanest air at midday when the boundary layer is deepest. Civic Centre reaches ≈ 45 µg/m³ (USG) at 06:00."));
c.push(B(`**Uncertainty:** the shaded band in the daily plot is ±1 test RMSE (≈ ±${f1(M.RMSE_ugm3)} µg/m³). A forecast of 29 µg/m³ could realistically be anywhere from 9 to 49, i.e. from Moderate to USG.`));
c.push(H2("7.3 Bonus map"));
c.push(...img("05e_bonus_map_13h.png", 6.4, "Figure 14. Sensors forecast in the USG → Very Unhealthy range at 13:00 on 1 Aug 2026."));
c.push(P("At 13:00 only **2 of 149** sensors are forecast above 35.5 µg/m³ (Sir Apollo Kagwa Road ≈ 50, Kamwokya Primary School ≈ 37), both in central Kampala. 13:00 is near the daily minimum; the same map at 07:00 would show many more sites at risk."));

// 8. Evidently
c.push(H1("8. Model monitoring with Evidently AI"));
c.push(P("**Data drift** = the distribution of inputs changes (e.g. the weather shifts with the season). **Target / concept drift** = the thing we predict, or its relationship to the inputs, changes. **Model drift** = performance gets worse on new data. Evidently compares a **reference** dataset (training period) with a **current** one (the test day) and tests every column."));
c.push(H2("8.1 The statistics Evidently reports"));
c.push(table([
  ["Statistic", "Used for", "Large value means", "Small value means", "Drift threshold"],
  ["K-S test p-value", "Numeric columns, small samples (≤ 1,000 rows)", "Distributions look the same (no evidence of drift)", "Distributions differ (drift)", "p < 0.05 → drift"],
  ["Wasserstein distance (normed)", "Numeric columns, large samples (our case)", "Distribution moved a lot (in standard deviations)", "Distributions are close", "> 0.1 → drift"],
  ["Jensen-Shannon distance", "Categorical / binary columns (season, network)", "Category proportions changed", "Same proportions", "> 0.1 → drift"],
  ["Share of drifted columns", "Whole dataset", "Many features changed: the model is extrapolating", "Stable inputs", "≥ 50% → dataset drift"],
], [1900, 2200, 2200, 1900, CONTENT_W - 8200]));
c.push(gap());
c.push(box("Watch out: a p-value is backwards compared with distances", [
  "For the **K-S p-value**, a **small** number means drift. For **Wasserstein / Jensen-Shannon**, a **large** number means drift. Always check which method the report names.",
], GOLDTINT, CRIMSON));
c.push(gap());
c.push(...img("06a_evidently_drift.png", 6.6, "Figure 15. Drift scores train → test (left) and January → April (right). Red = drift detected."));
c.push(H2("8.2 Our drift results and what they mean"));
c.push(table([
  ["Column", "Train → test score", "Drift?", "Interpretation"],
  ["hour_sin / hour_cos", "0.04 / 0.07", "No", "Both periods cover all hours: the monitoring window is complete"],
  ["latitude / longitude / device_level", "0.04 / 0.05 / 0.08", "No", "The mix of online sensors is stable (no coverage problem)"],
  ["temperature / humidity", "0.27 / 0.31", "Yes", "June weather differs from the Jan + Apr training mix: seasonal drift; the model is extrapolating"],
  ["is_peak_dry_season", "0.41 (JS)", "Yes", "By construction: test is all non-peak, training is 41% peak"],
  ["target (true PM2.5)", "0.29", "Yes", "Pollution level changed vs. training (January was much dirtier): concept-drift risk"],
  ["prediction", "0.42", "Yes", "The model's outputs moved with the inputs, as they should"],
], [2600, 1700, 900, CONTENT_W - 5200]));
c.push(gap());
c.push(P("**Seasonal report (Jan → Apr):** target drift 0.78 and humidity 0.35: the wet season is much cleaner and more humid. This is what drift looks like when the season changes."));
c.push(H2("8.3 Model drift (performance)"));
c.push(table([
  ["Metric (z units)", "Reference (train)", "Current (test)", "What it tells us"],
  ["RMSE", "0.52", "0.70", "+36% error on new data: a generalisation gap"],
  ["MAE", "0.35", "0.43", "Typical error grows less than RMSE: the extra error is in spikes"],
  ["Mean error (bias)", "0.00", "−0.01", "No systematic over- or under-prediction on average"],
  ["R²", "0.73", "0.22", "Much less variance explained on unseen hours"],
  ["Max absolute error", "5.94", "5.89", "Worst single miss ≈ 6 σ ≈ 168 µg/m³ in both"],
], [2200, 1700, 1700, CONTENT_W - 5600]));
c.push(gap());
c.push(box("Decisions from monitoring (what you would do in production)", [
  "Run Evidently daily on the latest labelled sensor data. **Retrain** when test RMSE rises by more than ~20% over the reference, when the mean error moves clearly away from 0 (systematic bias), or when the **target** drifts (a new pollution regime such as the dry season).",
  "Drift in **hour, location or device mix** points to a **data-pipeline problem** (sensors offline), not a change in the air: fix the data, don't retrain.",
  "Keep the previous model version so you can **roll back** if a retrained model performs worse (as discussed in the team meeting).",
  "If Evidently fails on RunPod, the notebook catches the error and continues (brief §9: 'thou shall not crash').",
]));

// 9. RunPod
c.push(H1("9. Deployment on RunPod"));
c.push(P("RunPod is a pay-as-you-go GPU cloud. A **Pod** is a dedicated GPU machine for development and testing. The budget is ≤ $5; our full pipeline takes ≈ 4–5 minutes, so the run costs about **$0.10**."));
newList();
c.push(N("Add credit; go to **Pods → Deploy**; choose **Community Cloud** and an available GPU under $2/h (RTX A4000, 3090 or 4090 ≈ $0.2–0.7/h). Screenshot the **Pod Summary** and **Pricing Summary**."));
c.push(N("Template: **RunPod PyTorch 2.x** (Jupyter Lab included); 20 GB container disk; On-Demand."));
c.push(N("Open Jupyter Lab → terminal → git clone the repository (or upload the zip) → `bash deployment/run_on_runpod.sh`."));
c.push(N("The script records the GPU (nvidia-smi), pod ID, pip install log, executes the notebook headless and writes **outputs/logs/pipeline_run.log** and **run_summary_runpod.json** (stage timings, metrics, best model). XGBoost automatically uses CUDA when a GPU is found."));
c.push(N("Download outputs/ and models/, screenshot the logs and billing page, then **Stop and Terminate** the pod."));
c.push(box("Exam / viva tip — 'How did you show local debugging first?'", [
  "The notebook writes **run_summary_local.json** and a timed **pipeline_run.log** on every local run (environment: CPU only, no GPU, 'not on RunPod'). Comparing it with the RunPod summary (GPU name, pod ID, runtime) is your evidence that the pipeline was debugged locally before you spent money on the cloud.",
]));

// 10. Viva questions
c.push(H1("10. Likely viva / exam questions with model answers"));
const qa = [
  ["Why is the target a Z-score and not raw µg/m³?", "The brief requires it. It puts the target on a unit-free scale (mean 0, sd 1), which helps distance- and gradient-based models (KNN, SVR, Ridge) and makes errors comparable. We always inverse-transform (x = zσ + μ) before reporting forecasts."],
  ["Your R² is only 0.22. Is the model bad?", "It is modest but honest. The model has no recent readings, only 7 days of history and climatological weather. It beats both baselines by ≈ 4 µg/m³ and gets 71% of AQI bands right. A random split would have shown a much higher, misleading R². Better weather inputs and longer history are the way to improve it."],
  ["Why is XGBoost better than Ridge?", "PM2.5 depends on interactions (e.g. the morning peak is bigger at roadside sites and in cool weather). Trees learn these non-linear interactions automatically; Ridge can only add them through polynomial terms."],
  ["What's the difference between bagging and boosting?", "Bagging (Random Forest) trains many independent trees on bootstrap samples and averages them, which reduces variance. Boosting (XGBoost) trains trees one after another, each correcting the previous errors, which reduces bias. The learning rate controls each tree's contribution."],
  ["Why did stacking not win?", "Its base models (RF, XGB) already capture almost the same patterns, so there is little complementary information to combine. Stacking helps most when base models make different errors."],
  ["Why didn't you use PM10 as a feature?", "Leakage: it is measured by the same sensor at the same moment, so it would not be available in a real forecast. The brief also stresses that PM10 is dust-driven and responds differently to weather."],
  ["How did you handle sensors that go offline?", "Offline hours are explicit missing rows (is_observed = 0). Short gaps (≤ 3 h) are interpolated; long gaps are never filled, and those hours are not used as training targets. The model itself needs no history, so it can predict for any sensor at any hour."],
  ["What does data drift tell you here?", "Temperature, humidity and PM2.5 itself shift between seasons. When those drift and the error rises, retrain on recent data. Drift in location or device mix means a coverage problem."],
  ["Why are the August daily forecasts flat?", "All inputs are known in advance or climatological, so nothing changes from day to day; it is a climatological forecast. Feeding daily weather forecasts would add variation."],
  ["What would you improve first?", "(1) Pull months of continuous data from the AirQo API; (2) add real weather forecasts; (3) add a short-term nowcasting model with lag features; (4) automate Evidently checks, retraining and rollback."],
];
qa.forEach(([q, a]) => { c.push(new Paragraph({ spacing: { before: 160, after: 60 }, children: [new TextRun({ text: "Q: " + q, bold: true, color: NAVY })] })); c.push(P("**A:** " + a)); });

// 11. Glossary
c.push(H1("11. Glossary"));
c.push(table([
  ["Term", "Plain-language meaning"],
  ["Autocorrelation", "A value is similar to the values just before and after it in time"],
  ["Baseline", "A simple reference prediction (e.g. the mean) every model must beat"],
  ["Bias (model)", "Systematic error: always too high or too low"],
  ["Blocked CV", "Cross-validation that holds out whole blocks (days) to respect time structure"],
  ["Boundary layer", "The lowest layer of the atmosphere where pollution mixes; shallow at night, deep at midday"],
  ["Concept drift", "The relationship between inputs and target changes over time"],
  ["Data leakage", "Information from the test set or the future sneaks into training, inflating scores"],
  ["Hyperparameter", "A model setting chosen before training (e.g. number of trees)"],
  ["IDW", "Inverse distance weighting: closer neighbours get more weight when filling a gap"],
  ["IQR", "Interquartile range: 75th minus 25th percentile; basis of box-plot outlier fences"],
  ["Overfitting / underfitting", "Memorising noise (great train, poor test) / too simple (poor on both)"],
  ["Permutation importance", "Increase in error when one feature is randomly shuffled"],
  ["Recall / precision", "Share of real events caught / share of alarms that were real"],
  ["Regularisation", "A penalty that keeps a model simple to reduce overfitting (alpha, lambda)"],
  ["Target encoding", "Replacing a category with its average target value (computed on training data)"],
  ["Wasserstein distance", "How far one distribution must be 'moved' to match another"],
  ["Z-score", "Number of standard deviations from the mean: z = (x − μ)/σ"],
], [2600, CONTENT_W - 2600]));

// 12. Checklist
c.push(H1("12. Deliverables checklist"));
c.push(table([
  ["Deliverable (brief §8)", "File in the repository", "Status"],
  ["Code: full pipeline", "AirQo_PM25_Pipeline.ipynb (source: notebook_src/airqo_pm25_pipeline.py)", "Done, runs end-to-end locally (≈ 4 min)"],
  ["Saved best model", "models/best_pm25_model.joblib (+ model card JSON, stacking ensemble)", "Done"],
  ["Presentation", "deliverables/AirQo_PM25_Presentation.pptx (UCU template)", "Done"],
  ["Forecasts & figures", "outputs/forecasts/*.csv, outputs/figures/*.png", "Done"],
  ["Evidently reports", "outputs/evidently/*.html (open in a browser)", "Done"],
  ["RunPod run", "deployment/run_on_runpod.sh + RUNPOD_GUIDE.md", "**You must run it on your own RunPod account and add screenshots/logs**"],
  ["LaTeX report", "–", "**Not produced yet: required by the brief** (this guide and the notebook contain all the content)"],
], [2400, 4300, CONTENT_W - 6700]));
c.push(gap());
c.push(H2("References"));
[
  "Adong, P., Bainomugisha, E., Okure, D., & Sserunjogi, R. (2022). Applying machine learning for large scale field calibration of low-cost PM2.5 and PM10 air pollution sensors. Applied AI Letters, 3(3), e76.",
  "Barkjohn, K. K., Gantt, B., & Clements, A. L. (2021). Development and application of a United States-wide correction for PM2.5 data collected with the PurpleAir sensor. Atmospheric Measurement Techniques, 14(6), 4617–4637.",
  "Junninen, H., Niska, H., Tuppurainen, K., Ruuskanen, J., & Kolehmainen, M. (2004). Methods for imputation of missing values in air quality data sets. Atmospheric Environment, 38(18), 2895–2907.",
  "Okure, D., Ssematimba, J., Sserunjogi, R., Gracia, N. L., Soppelsa, M. E., & Bainomugisha, E. (2022). Characterization of ambient air quality in selected urban areas in Uganda using low-cost sensing and measurement technologies. Environmental Science & Technology, 56(6), 3324–3339.",
  "Roberts, D. R., et al. (2017). Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure. Ecography, 40(8), 913–929.",
  "World Health Organization (2021). WHO global air quality guidelines.",
].forEach((r) => c.push(new Paragraph({ spacing: { after: 80 }, indent: { left: 360, hanging: 360 }, children: [new TextRun({ text: r, size: 20 })] })));

// ---------- document ----------
const doc = new Document({
  creator: "AirQo Assignment 4",
  features: { updateFields: true },
  title: "Study Guide — AirQo PM2.5 Prediction",
  styles: {
    default: { document: { run: { font: "Calibri", size: 22 } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 34, bold: true, font: "Cambria", color: NAVY }, paragraph: { spacing: { before: 240, after: 200 }, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 27, bold: true, font: "Cambria", color: NAVY }, paragraph: { spacing: { before: 280, after: 120 }, outlineLevel: 1 } },
      { id: "Heading3", name: "Heading 3", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 23, bold: true, font: "Calibri", color: CRIMSON }, paragraph: { spacing: { before: 200, after: 80 }, outlineLevel: 2 } },
    ],
  },
  numbering: { config: [
    { reference: "bullets", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT,
      style: { paragraph: { indent: { left: 540, hanging: 270 } } } }, { level: 1, format: LevelFormat.BULLET, text: "–", alignment: AlignmentType.LEFT,
      style: { paragraph: { indent: { left: 1080, hanging: 270 } } } }] },
    { reference: "numbers", levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT,
      style: { paragraph: { indent: { left: 540, hanging: 300 } } } }] },
  ] },
  sections: [{
    properties: { page: { size: { width: PAGE_W, height: 16838 }, margin: { top: MARGIN, bottom: MARGIN, left: MARGIN, right: MARGIN } } },
    headers: { default: new Header({ children: [new Paragraph({ alignment: AlignmentType.RIGHT,
      children: [new TextRun({ text: "CSC3119 · Assignment 4 Study Guide", size: 16, color: MUTED })] })] }) },
    footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER,
      children: [new TextRun({ children: ["Page ", PageNumber.CURRENT], size: 16, color: MUTED })] })] }) },
    children: c,
  }],
});

const out = path.join(ROOT, "deliverables", "AirQo_PM25_Study_Guide.docx");
Packer.toBuffer(doc).then((buf) => { fs.writeFileSync(out, buf); console.log("saved", out); });
