# AirQo PM2.5 Prediction — Assignment 4 (CSC3119 AI Deployment & Scalability, UCU)

End-to-end pipeline that predicts hourly PM2.5 (as a Z-score) from AirQo sensor data, forecasts August 2026,
monitors drift with Evidently AI and runs on RunPod.

## Deliverables
| Deliverable | Path |
|---|---|
| Notebook (full pipeline, executed) | `AirQo_PM25_Pipeline.ipynb` |
| Saved best model (XGBoost) + model card | `models/best_pm25_model.joblib`, `models/best_model_card.json` |
| Saved ensemble model | `models/stacking_ensemble_pm25.joblib` |
| Presentation (UCU template) | `deliverables/AirQo_PM25_Presentation.pptx` |
| Study guide (Word) | `deliverables/AirQo_PM25_Study_Guide.docx` |
| LaTeX report (+ compiled PDF) | `report/AirQo_PM25_Report.tex`, `report/AirQo_PM25_Report.pdf` |
| Forecasts (daily 1–16 Aug, hourly 1 Aug, 13:00 map list) | `outputs/forecasts/` |
| Figures | `outputs/figures/` |
| Evidently reports (open in a browser) | `outputs/evidently/` |
| Run logs (local; RunPod run adds `*_runpod*`) | `outputs/logs/` |
| RunPod scripts & guide | `deployment/` |

## Results (hold-out: last 24 h, 2–3 Jun 2026)
| Model | RMSE µg/m³ | MAE µg/m³ | R² | AQI accuracy |
|---|---|---|---|---|
| **XGBoost (best)** | 20.0 | 12.3 | 0.22 | 71% |
| Stacking ensemble | 20.0 | 12.1 | 0.22 | 72% |
| Random Forest | 20.1 | 12.0 | 0.21 | 71% |
| KNN | 20.8 | 12.4 | 0.16 | 69% |
| Ridge | 20.8 | 12.4 | 0.15 | 72% |
| SVR | 21.6 | 11.8 | 0.09 | 74% |
| Baseline: sensor mean | 23.9 | 18.4 | −0.11 | 49% |

## Reproduce
```bash
pip install -r deployment/requirements.txt
jupyter nbconvert --to notebook --execute --inplace AirQo_PM25_Pipeline.ipynb   # ~4 min on CPU
python notebook_src/build_presentation.py "templates/UCU-power-point template.pptx"
node notebook_src/build_study_guide.js        # needs the `docx` npm package
cd report && pdflatex AirQo_PM25_Report.tex && pdflatex AirQo_PM25_Report.tex   # figures are read from ../outputs/figures
```
The notebook is generated from `notebook_src/airqo_pm25_pipeline.py` (jupytext percent format):
`jupytext --to notebook notebook_src/airqo_pm25_pipeline.py -o AirQo_PM25_Pipeline.ipynb`.

## Data
`data/sensor_data_jan_jul_2026.csv` (raw), `data/processed/hourly_grid_clean.parquet` (cleaned hourly grid),
`data/shapefiles/` (Natural Earth Uganda boundary, districts, lakes, for the bonus map).
