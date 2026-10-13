# Reproducibility and submission gates

Evidence collected on 2026-10-13, Windows Python 3.12.10.

| Gate | Status | Evidence |
|---|---|---|
| G1. API, model, logging and ingestion tests | PASS | `python run_tests.py`: **12 passed** (warnings from NumPy/Pandas deprecations) |
| G2. HTTP API integration | PASS | `tests/test_api.py` starts a real loopback HTTP server; country and all-country prediction tested |
| G3. Visual evidence | PASS | `reports/eda_daily_revenue.png`, `reports/eda_country_revenue.png`, `reports/model_vs_baseline.png` |
| G4. Model comparison and chronological holdout | PASS | `reports/metrics.json`, `reports/holdout_predictions.csv` |
| G5. Official IBM/AAVAIL data | PASS | Downloaded **21** JSON files; ingested **815,011** transactions across **43** countries; trained/evaluated all candidates |
| G6. Docker build and smoke | NOT VERIFIED BY REQUEST | Dockerfile and compose included, CI workflow defines a Docker smoke test. No local Docker execution performed |
| G7. Public GitHub repo | CREATED | https://github.com/Klein1411/aavail-revenue-forecasting |
| G8. Public cloud deployment | NOT REQUESTED | This is a reproducible demonstration repository, not a cloud-deployed service |

## Official-data holdout scores

| Model | MAE | RMSE |
|---|---:|---:|
| Ridge | 98.2823 | 746.5428 |
| Random forest | 125.8091 | 867.6618 |
| Seasonal naive (lag-7) | **75.0376** | **737.6694** |

**Interpretation:** The baseline outperformed both trained ML candidates on this validation set. Ridge is currently selected among trained estimators for serving, but its measured improvement relative to the naive baseline is **negative**. These figures are not synthetic-demo figures. They are based on the official public course dataset, and are measured by per-day recursive predictions on a chronological holdout.

## Reproduce

```bash
python -m pip install -r requirements.txt
python run_tests.py
python scripts/fetch_official_data.py
python -m src.ingest --data-dir data/official
python -m src.train --data-dir data/official
python -m src.plots --data-dir data/official
python -m src.app
```

The original official invoice JSON files are intentionally excluded from Git. The synthetic fixture `data/sample/aavail_demo_transactions.csv` is committed for offline tests/CI.

## Caveats

- Docker was not locally validated, per explicit user instruction.
- Python dependencies are range-constrained, not pinned; consider a lock file for strict bitwise reproducibility.
- Deprecation warnings do not fail the tests but should be fixed in a maintenance pass.
- Avoid treating this demonstration as a production financial forecasting service.
