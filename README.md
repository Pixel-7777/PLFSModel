# PLFS Regular-Wage Model

A reproducible end-to-end project for estimating the monthly regular wage of
Indian regular salaried/wage workers from the Periodic Labour Force Survey
(PLFS), 2023–2025.

The model is a tabular quantile MLP with categorical embeddings and a weighted
pinball loss. It predicts the **5th, 50th and 95th percentiles** of monthly
regular earnings in 2025 prices.

This repository contains the code and notebooks only. **No PLFS microdata is
redistributed here.** Download the source files from the official MoSPI
microdata portal using the instructions below.

## Contents

```text
.
├── notebooks/
│   ├── 01_wage_merge.ipynb              # harmonize 2023 + 2024 + 2025 PLFS files
│   └── 02_wage_model_end_to_end.ipynb   # train, evaluate, SHAP, robustness, figures
├── app/
│   ├── server.py                        # local HTTP server + JSON API
│   ├── inference.py                     # loads the trained weights/preprocessing
│   ├── labels.json                      # categorical labels and form defaults
│   ├── static/                          # HTML, CSS and JavaScript interface
│   └── data/                            # SQLite prediction log (created at runtime)
├── data/
│   ├── raw/                             # place downloaded PLFS person files here
│   └── processed/                       # created by notebook 01
├── models/                              # created by notebook 02
├── figures/
│   ├── data/                            # created by notebook 01
│   ├── training/                        # created by notebook 02
│   └── presentation/                    # created by notebook 02
├── run_ui.ps1                           # start the interface on Windows
├── run_ui.bat                           # double-click launcher
├── README.md
└── requirements.txt
```

The `data/`, `models/` and generated `figures/` contents are ignored by Git.
The notebooks, `app/` source, `README.md`, `requirements.txt`, launchers and
`.gitignore` are the files meant to be tracked. The trained weights and
preprocessing bundle are not committed; run notebook 02 to create them before
starting the interface.

## 1. Set up the environment

Python 3.10+ is recommended. A CUDA GPU is strongly recommended for training
and SHAP; the notebooks also run on CPU but much more slowly.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Then start Jupyter:

```bash
jupyter lab
```

## 2. Download the PLFS data

The project uses three **person-level** PLFS files. They are available from the
MoSPI Microdata Portal:

- Portal: <https://microdata.gov.in/NADA/index.php/catalog/PLFS>
- 2023–24 survey: <https://microdata.gov.in/NADA/index.php/catalog/213>
- Calendar Year 2024: <https://microdata.gov.in/NADA/index.php/catalog/254>
- Calendar Year 2025: <https://microdata.gov.in/NADA/index.php/catalog/284>

The 2024 release may appear on the portal as **"Key Employment Unemployment
Indicators for (January 2024 - December 2024)"**; open its unit-level data
files and choose the person-level file.

For each release:

1. Open the catalog page and sign in / register if required.
2. Open **Get Microdata** and download the unit-level data.
3. Extract the archive.
4. Select the **person-level** CSV — the file with one row per person, not the
   household-level file.
5. Rename and place it as follows:

| Release | Put the person file at | Expected raw rows |
|---|---|---:|
| PLFS July 2023–June 2024 | `data/raw/2023.csv` | 418,159 |
| PLFS Calendar Year 2024 | `data/raw/2024.csv` | 415,549 |
| PLFS Calendar Year 2025 | `data/raw/2025.csv` | 1,148,634 |

The final layout must be:

```text
data/raw/2023.csv
data/raw/2024.csv
data/raw/2025.csv
```

### Expected columns

The harmonization notebook contains explicit column mappings:

- 2023: `MAP_2023` in cell 5 (raw `perv1` columns such as `b4q6_perv1`,
  `b5pt1q5_perv1`, `b6q9_perv1`, `b6q7_3pt*_perv1`).
- 2024: `MAP_2024` in cell 8 (columns such as `Age`, `General_Education_Level`,
  `Principal_Industry_Code`, `CWS_Earnings_Salaried`, `Day1_Total_Hours`).
- 2025: `required25` in cell 11 plus the hour columns `hr1`–`hr7`.

If MoSPI changes a release, the notebook will stop with a clear
`missing columns` error. In that case, update the corresponding mapping in
`01_wage_merge.ipynb` before continuing.

The processed dataset should contain **213,241 rows** (2023: 45,569;
2024: 46,034; 2025: 121,638) and **41 columns**. The notebook checks this
contract explicitly.

> **Why raw data is not included:** PLFS unit-level data is distributed by
> MoSPI under its own access terms. This repository does not redistribute the
> source files.

## 3. Run the notebooks

Run in this order.

### Step 1 — harmonize the data

Open and run `notebooks/01_wage_merge.ipynb` top to bottom.

Outputs:

```text
data/processed/plfs_2023_2024_2025_harmonized.csv
data/processed/wage_merge_summary.csv
data/processed/wage_merge_report.json
figures/data/01_data_summary_table.png
figures/data/02_rows_by_year.png
figures/data/03_median_wages.png
figures/data/04_wage_quantiles.png
figures/data/05_education_missingness.png
```

The notebook preserves the year-specific quirks used by the original model,
including the 2025 education mapping and the 69 retained duplicate rows.

### Step 2 — train, evaluate and generate the slide figures

Open and run `notebooks/02_wage_model_end_to_end.ipynb` top to bottom.

This notebook:

- loads and validates the 213,241-row processed dataset;
- creates the stratified 70/15/15 train / validation / test split (seed 42);
- fits the preprocessing objects on the training split only;
- trains the 5/50/95 quantile MLP with the weighted pinball loss;
- evaluates coverage, MAE, bias, interval width and quantile crossings;
- runs GradientSHAP, permutation importance and ablation importance;
- runs realistic and out-of-range perturbations, bootstrap intervals and
  MC-dropout stability;
- runs an in-memory categorical-format/schema-drift audit;
- writes all training and presentation figures.

Outputs are written to `models/wage/`, `figures/training/` and
`figures/presentation/`.

The main artifacts are:

```text
models/wage/wage_quantile_mlp.pt
models/wage/wage_preprocessing.pkl
models/wage/model_info.json
models/wage/metrics.json
models/wage/test_predictions.csv.gz
models/wage/shap_importance.csv
models/wage/shap_permutation.csv
models/wage/shap_ablation.csv
models/wage/wage_perturbations.csv
models/wage/wage_subgroups.csv
models/wage/wage_reliability.csv
models/wage/wage_drift_summary.json
models/wage/model_card.md
```

## 4. Use the web interface

Notebook 02 must have finished first, because the interface loads:

- `models/wage/wage_quantile_mlp.pt`
- `models/wage/wage_preprocessing.pkl`
- `models/wage/model_info.json`
- `models/wage/metrics.json`

Start the local server:

```bash
python app/server.py
```

Then open <http://127.0.0.1:8000>.

On Windows you can also run `.\run_ui.ps1` or double-click `run_ui.bat`.

The interface:

- renders labelled dropdowns for all 27 categorical features from
  `app/labels.json`;
- loads the current trained model and preprocessing bundle at startup;
- derives potential experience and average daily hours with the same formulas
  as notebook 01;
- returns the 5th, 50th and 95th percentile of monthly regular earnings in
  2025 prices;
- shows warnings for missing education, out-of-range inputs and high-wage
  estimates;
- stores every submission and result in `app/data/predictions.db`;
- lists recent predictions and exports them as CSV.

The server uses only the Python standard library for HTTP and SQLite. Use
`python app/server.py --port 8080` if port 8000 is busy.

### API

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/health` | Model version, device, weight hash and stored prediction count. |
| `GET` | `/api/schema` | Form labels, defaults and current model facts. |
| `POST` | `/api/predict` | Score one profile. Body: `{"inputs": {...}}`. |
| `GET` | `/api/predictions?limit=20` | Recent predictions from SQLite. |
| `GET` | `/api/predictions.csv` | Download all stored predictions. |

### Resetting the local prediction log

Stop the server, delete `app/data/predictions.db`, and restart the server. The
database is recreated automatically. The database is ignored by Git.

## Runtime

On a CUDA GPU, the full run is roughly:

- `01_wage_merge.ipynb`: about 1.5 minutes.
- `02_wage_model_end_to_end.ipynb`: about 10–15 minutes, dominated by
  GradientSHAP. Training itself is about 2–3 minutes for 50 epochs.

The SHAP sample sizes are set at the top of notebook 02:

```python
SHAP_N_TEST = 2000
SHAP_N_BG = 250
SHAP_NSAMPLES = 150
```

Reduce them for a faster exploratory run, or increase them for a final
analysis. The notebook also reads these environment variables:

```bash
export WAGE_EPOCHS=3
export SHAP_N_TEST=100
export SHAP_N_BG=50
export SHAP_NSAMPLES=20
export N_BOOT=100
export MC_PASSES=3
```

On Windows PowerShell use `$env:WAGE_EPOCHS="3"` etc.

## Generated figures

`figures/data/` — data audit:

- `01_data_summary_table.png`
- `02_rows_by_year.png`
- `03_median_wages.png`
- `04_wage_quantiles.png`
- `05_education_missingness.png`

`figures/training/` — training and test evaluation:

- `01_training_history.png`
- `02_test_metrics_table.png`
- `03_calibration_by_decile.png`
- `04_coverage_by_wage_band.png`

`figures/presentation/` — model, SHAP, robustness and drift:

- `01_test_metrics_table.png`
- `02_coverage_by_group.png`
- `03_calibration_deciles.png`
- `04_shap_top10_q50.png`
- `05_shap_top5_table.png`
- `06_shap_crosscheck_table.png`
- `07_robustness_realistic.png`
- `08_robustness_failures.png`
- `09_schema_drift.png`

The figures are deliberately simple: one table or one bar/line chart per slide.

## Model summary

- **Target:** `ern_reg_current` — monthly regular earnings of a regular
  salaried/wage worker, CPI-adjusted to 2025 prices.
- **Architecture:** tabular MLP with categorical embeddings, four residual
  blocks, three-output quantile head.
- **Quantiles:** 0.05, 0.50, 0.95.
- **Features:** 7 numeric + 27 categorical.
- **Loss:** weighted pinball loss, 5× weight above ₹100,000.
- **Split:** stratified by survey year and wage band, seed 42.
- **Documented frozen-model test results:** 93.13% 5–95 coverage,
  ₹6,544 MAE, −₹1,323 bias, ₹22,388 median width, 0 crossings.

The notebook retrains the model from scratch. Results may differ slightly from
the documented frozen artifact because of hardware, library versions and
non-deterministic GPU operations.

## Limitations

- The model is trained only on regular salaried/wage workers (PLFS principal
  activity status 31). Other worker types are outside the target population.
- The high-wage tail is less reliable. On the documented frozen model,
  >₹100k coverage is 84.08% and MAE is ₹50,303.
- The 5–95 interval is a weighted quantile interval, not a conformal interval.
- Categorical codes must keep their original string formatting. Normalizing
  float-string codes to integers silently maps many rows to an unknown
  embedding and can collapse coverage.
- No input-range validation is built into the model; `days_worked=31` and
  `age=100` are high-risk inputs.
- Unknown categories map to an embedding row that was never trained.
- GradientSHAP is approximate, not exact Shapley value.
- SHAP and robustness results are model-behaviour diagnostics, not causal
  estimates.

## Data source and attribution

The survey data is produced by the National Statistical Office (NSO), Ministry
of Statistics and Programme Implementation (MoSPI), Government of India, and is
distributed through the MoSPI Microdata Portal. Follow the portal's terms of use
when downloading and using the data. This repository does not redistribute the
microdata.
