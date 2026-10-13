# AAVAIL Revenue Forecasting — AI Enterprise Workflow Capstone

![CI status](https://img.shields.io/badge/tests-pytest-informational) ![Python](https://img.shields.io/badge/python-3.11%2B-blue) ![Docker](https://img.shields.io/badge/docker-included-blue)

**Business objective.** Help AAVAIL managers estimate expected invoice revenue over the next 30 days for **one country** or **all countries combined**, enabling market-specific planning. This independent educational implementation follows the [IBM AI Enterprise Workflow Capstone brief](https://github.com/aavail/ai-workflow-capstone). It uses chronological validation, has a runnable HTTP API, automated tests, JSONL prediction logs, performance monitoring and Docker.

> **DATA PROVENANCE — VERIFIED:** `data/sample/aavail_demo_transactions.csv` is deterministic **synthetic demo data** used by CI/tests. The committed `reports/` charts, holdout predictions, and metrics were regenerated from 21 upstream IBM/AAVAIL `cs-train` JSON files (815,011 transactions across 43 countries, downloaded on 2026-10-13). The course data files themselves are excluded from Git for size and provenance; rerun `python scripts/fetch_official_data.py` and the official-data commands below to reproduce. **Observed limitation:** the lag-7 seasonal-naive baseline (MAE 75.0376) outperforms Ridge (MAE 98.2823) and Random Forest (MAE 125.8091) on the held-out period. Do not claim the ML candidates outperform the baseline. Source: https://github.com/aavail/ai-workflow-capstone

## Quick start (no Docker)

```bash
python -m pip install -r requirements.txt
python run_tests.py
python -m src.ingest --data-dir data/sample
python -m src.train --data-dir data/sample
python -m src.plots --data-dir data/sample
python -m src.app
```

Then visit **http://127.0.0.1:8000/** for a responsive browser dashboard. Run these commands from the repository root. Model training and plotting work without a network connection when the included sample data is used.

## Use the official course dataset (recommended before submitting)

```bash
python scripts/fetch_official_data.py
python -m src.ingest --data-dir data/official
python -m src.train --data-dir data/official
python -m src.plots --data-dir data/official
python -m src.app
```

The download script fetches the **public original course dataset**, not someone else's solution. Requires an internet connection. Place original `.json` files manually inside `data/official/` if downloading the zip is blocked. Original source: https://github.com/aavail/ai-workflow-capstone/tree/master/cs-train. This project intentionally excludes course data and other students' code from its committed files.

For the grading repository, include **your generated `reports/metrics.json`, `reports/holdout_predictions.csv` and the PNG charts**, produced using official data, so peer reviewers can verify the numerical and visual evidence. You may additionally commit a small suitably licensed sample of original records if permitted; retain attribution.

## Docker

```bash
docker build -t aavail-revenue-capstone .
docker run --rm -p 127.0.0.1:8000:8000 aavail-revenue-capstone
# or
# docker compose up --build
```

Docker builds a self-contained **synthetic demo** and automatically trains the bundled sample before serving. For real course data, copy original invoices into `data/official/`, adjust `INPUT_DIR=data/official` in the Docker runtime, and mount a data volume or build a local image that includes authorized inputs. Avoid publishing proprietary data or credentials.

## API

No need to learn a special UI: predictions work via curl, Postman or the built-in web page.

```bash
# Health & available countries
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/countries

# Prediction for a market, by JSON body
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d '{"country":"France","date":"2025-03-01","duration":30}'

# All countries combined, using query parameters
curl -X POST 'http://127.0.0.1:8000/predict?date=2025-03-01&duration=30'

# Chronological test metrics and model selection
curl http://127.0.0.1:8000/metrics
```

Accepted optional request fields: `country` defaults to `all` (case insensitive), `date` defaults to the day after the final available invoice; `duration` defaults to 30 (maximum 90 days). Both JSON fields and query parameters are accepted. `all` is the **sum of separately forecast market revenues**, making country and global estimates consistent. Revenue is expressed in unspecified dataset units (no invented currency). Historic dates are retrospective, in-sample predictions: **do not use those responses as evidence of a fair backtest**; use `reports/holdout_predictions.csv` instead.

API also has: `GET /logs?limit=100` requiring `Authorization: Bearer $ADMIN_TOKEN`; `POST /train` requiring `Authorization: Bearer $TRAIN_TOKEN`. When tokens are empty these protected endpoints are disabled. Do not expose this demonstration server directly to the public internet; add TLS, strict authorization, request limits, and a production server before actual deployment.

## Model selection and evaluation

- **Target:** per-day summed invoice `price`, including zero-sales days; the forecasted 30-day business figure is the sum of 30 recursively predicted daily totals.
- **Features:** previous day/week/2 weeks/4 weeks of sales, rolling 7- and 28-day averages, weekly and annual seasonality and elapsed time. Only previous-day information is used when engineering training features.
- **Candidates:** regularized linear regression (`Ridge`) and nonlinear random forest (`RandomForestRegressor`); compared with a **seasonal-naive lag-7 baseline**.
- **Evaluation:** last 14–30 calendar days are withheld chronologically for every market. Validation predictions are rolled forward using predictions only, **not actual held-out future values**, avoiding obvious future-label leakage. The lowest average absolute-error trained candidate is refit on all available history.
- **Limits:** no forecast intervals, external drivers, or holiday features; recursive multi-step errors grow with horizon. Markets with very little history need further validation. Negative invoices may indicate returns and are counted in net revenue.

`reports/model_vs_baseline.png` shows the actual withheld revenue versus both ML candidates and the baseline. `reports/eda_daily_revenue.png` and `reports/eda_country_revenue.png` visualize EDA. `reports/metrics.json` and `reports/holdout_predictions.csv` contain reproducible numeric evidence.

## Monitoring and feedback

```bash
# Held-out MAE and RMSE from a genuinely unseen final time interval
python -m src.monitor

# Compare logged production forecasts to newly available actual 30-day totals:
# actuals.csv columns: country,date,duration_days,actual_revenue
python -m src.monitor --actuals actuals.csv --alert-mae 25
```

Forecast activity is appended to `logs/predictions.jsonl`. Monitoring joins actual 30-day totals to predictions using exact `(country, date, duration_days)` keys and reports real MAE/RMSE. It does **not** manufacture ground-truth labels. The optional threshold warns if observed MAE exceeds a user-defined limit. Log access in the API is access-controlled when configured.

## Test isolation

- Run **all tests with one command:** `python run_tests.py`.
- API: `tests/test_api.py` starts an actual HTTP server on a free loopback port.
- Models: `tests/test_model.py` verifies features, leakage protection, country/global inference, validation.
- Logging and monitoring: `tests/test_logging.py`; ingestion: `tests/test_ingest.py`.
- Tests use a **temporary directory** created by pytest for model binaries, evaluation files and logs; they never delete, overwrite or append to production artifacts.
- CI in `.github/workflows/ci.yml` runs tests, trains the sample, generates figures and builds/smoke-tests a Docker image when pushed to GitHub.

## Peer review checklist (12 grading criteria)

| # | Assessed criterion | Evidence in this repository |
|---|---|---|
| 1 | Unit tests for API | `tests/test_api.py` |
| 2 | Unit tests for model | `tests/test_model.py` |
| 3 | Unit tests for logging | `tests/test_logging.py` |
| 4 | One script runs all tests | `python run_tests.py`; GitHub Actions workflow |
| 5 | Monitor model performance | `src/monitor.py` (holdout and later realized-label MAE/RMSE) |
| 6 | Isolate read/write tests from production | pytest `tmp_path`, test fixture in `tests/conftest.py` |
| 7 | Country-specific and all-country API | `POST /predict` API + live integration tests |
| 8 | Automated data ingestion | `src/ingest.py`, `scripts/fetch_official_data.py` |
| 9 | Compare multiple models | Ridge, random forest, seasonal-naive; `reports/metrics.json` |
| 10 | EDA visualizations | `reports/eda_daily_revenue.png`, `reports/eda_country_revenue.png` |
| 11 | Dockerized working solution | `Dockerfile`, `compose.yaml`, CI smoke test |
| 12 | Visualization versus baseline | `reports/model_vs_baseline.png` |

## Business hypotheses

1. Revenue varies systematically by day of week and season, rather than as pure noise.
2. Short-term history (7 to 28 days) adds predictive value over a repeat-last-week baseline.
3. Country-specific forecasts provide actionable differences in revenue planning and can be summed for a global estimate.

Success is evaluated using **holdout MAE / RMSE**, with improved errors over the baseline indicating value. The implementation always shows the baseline even when it performs better; it does not conceal negative results.

## Architecture

`Invoice JSON/CSV → ingest/validate → daily by market → lag/seasonal features → chronological holdout → Ridge vs RandomForest vs lag-7 benchmark → retrain → model artifact → HTTP API → JSONL logs → realized-revenue monitoring + EDA/report plots`

## Ethics, provenance, and reproducibility

- Original implementation for educational purposes; no code imported from peers' capstone solutions.
- Original IBM/AAVAIL case-study data may be obtained from the [public source repository](https://github.com/aavail/ai-workflow-capstone); check source distribution rights before publishing data.
- Demo data generator: `python scripts/create_demo_data.py`, NumPy seeded with 42.
- Deterministic train/test split by calendar date, with random forest seed 42.
- This is a course demonstration, not a deployed financial service. Running Docker locally is containerization, not proof of public-cloud deployment.
