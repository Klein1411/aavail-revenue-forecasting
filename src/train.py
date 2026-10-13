"""Chronological model evaluation and retraining on all historical data."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from .ingest import load_transactions, daily_by_country
from .model import MIN_HISTORY, supervised, recursive_forecast, candidates


def train_pipeline(data_dir: str | Path, artifacts_dir: str | Path,
                   reports_dir: str | Path) -> dict:
    histories = daily_by_country(load_transactions(data_dir))
    if min(map(len, histories.values())) < 70:
        raise ValueError('Need at least 70 calendar days of data for chronological validation')
    reports = Path(reports_dir)
    models_path = Path(artifacts_dir)
    reports.mkdir(parents=True, exist_ok=True)
    models_path.mkdir(parents=True, exist_ok=True)
    # Every country has the same calendar, avoiding inconsistently aligned targets.
    holdout = min(30, max(14, len(next(iter(histories.values()))) // 6))
    names = [*candidates(), 'seasonal_naive']
    all_errors = {name: [] for name in names}
    detailed = []
    for country, series in histories.items():
        history, actual = series.iloc[:-holdout], series.iloc[-holdout:]
        X, y = supervised(history)
        for name in names:
            if name == 'seasonal_naive':
                pred = recursive_forecast(None, history.tolist(), actual.index[0], holdout, baseline=True)
            else:
                estimator = candidates()[name]
                estimator.fit(X, y)
                pred = recursive_forecast(estimator, history.tolist(), actual.index[0], holdout)
            errors = np.abs(actual.to_numpy() - np.asarray(pred))
            all_errors[name].extend(errors.tolist())
            detailed.extend({'country': country, 'date': dt.strftime('%Y-%m-%d'),
                            'actual': float(a), 'prediction': float(p), 'model': name}
                            for dt, a, p in zip(actual.index, actual.to_numpy(), pred))
    metrics = {
        name: {'mae': round(float(np.mean(values)), 4),
               'rmse': round(float(np.sqrt(np.mean(np.square(values)))), 4)}
        for name, values in all_errors.items()
    }
    # Baseline is kept as a benchmark, not chosen as the trained ML estimator.
    selected = min(candidates(), key=lambda name: metrics[name]['mae'])
    fitted = {}
    for country, series in histories.items():
        X, y = supervised(series)
        estimator = candidates()[selected]
        estimator.fit(X, y)
        fitted[country] = estimator
    last_date = max(series.index.max() for series in histories.values()).strftime('%Y-%m-%d')
    bundle = {'models': fitted, 'histories': histories, 'model_name': selected,
              'last_date': last_date, 'metrics': metrics, 'holdout_days': holdout,
              'data_mode': 'sample' if Path(data_dir).name == 'sample' else 'provided-data'}
    joblib.dump(bundle, models_path / 'model_bundle.joblib')
    rows = pd.DataFrame(detailed)
    rows.to_csv(reports / 'holdout_predictions.csv', index=False)
    (reports / 'metrics.json').write_text(json.dumps({
        'model_selected': selected, 'holdout_days': holdout,
        'scores': metrics, 'countries': list(histories), 'source': str(data_dir),
        'explanation': 'Chronological withheld final days; recursive predictions only use earlier history.'},
        indent=2), encoding='utf-8')
    print('Selected trained model:', selected, '\nScores:', json.dumps(metrics, indent=2))
    return bundle


def main() -> None:
    parser = argparse.ArgumentParser(description='Train evaluated revenue forecasters')
    parser.add_argument('--data-dir', default='data/sample')
    parser.add_argument('--artifacts-dir', default='artifacts')
    parser.add_argument('--reports-dir', default='reports')
    args = parser.parse_args()
    train_pipeline(args.data_dir, args.artifacts_dir, args.reports_dir)


if __name__ == '__main__':
    main()
