# Deploying the pipeline on RunPod (budget ≤ $5)

1. **Create account / add credit** – runpod.io → Billing → add $5 (the minimum is enough).
2. **Deploy a Pod** – *Pods → + Deploy*.
   - Cloud: **Community Cloud** (cheapest).
   - GPU: pick an available card well under $2/h – e.g. **RTX A4000 / RTX 3090 / RTX 4090 / RTX A5000** (~$0.20–0.70/h).
     Our pipeline runs in ~5 min on CPU, so any small GPU is plenty.
   - Template: **RunPod PyTorch 2.x** (includes Jupyter Lab).
   - Container disk 20 GB, volume 20 GB, **On-Demand** pricing.
   - Check the **Pod Summary** and **Pricing Summary** (hourly GPU + storage cost) → screenshot → *Deploy On-Demand*.
3. **Connect** – *Connect → Jupyter Lab (port 8888)*.
4. **Import code & data** – in a Jupyter terminal:
   ```bash
   cd /workspace
   git clone https://github.com/Mirembe-Peace/AirQo-AI_deployment.git
   cd AirQo-AI_deployment
   git checkout claude/assignment-notebook-model-presentation-nu5e92
   ```
   (or drag-and-drop the repo zip into Jupyter Lab and `unzip` it).
5. **Run the pipeline** –
   ```bash
   bash deployment/run_on_runpod.sh
   ```
   or open `AirQo_PM25_Pipeline.ipynb` and *Run → Run All Cells*.
6. **Collect evidence** for the report/slides:
   - `outputs/logs/runpod_environment.txt` (GPU, pod ID, runtime)
   - `outputs/logs/pipeline_run.log` (stage timings, metrics, best model)
   - `outputs/logs/run_summary_runpod.json`
   - `AirQo_PM25_Pipeline_executed_runpod.ipynb`
   - screenshots: pod summary, pricing, Jupyter run, pod *Logs* tab, billing page.
7. **Download** `outputs/` and `models/` (right-click → Download, or zip them), then **Stop → Terminate** the pod so billing stops.

**Cost estimate:** ~15 min of pod time at ≈ $0.40/h ≈ **$0.10**, plus a few cents of storage.

If Evidently errors on the pod, the notebook already catches it, logs a warning and continues (brief §9).
