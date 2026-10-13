"""Out-of-sample performance dashboard and realized-revenue monitoring."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import pandas as pd
import numpy as np
from .logging_service import read_events


def summarize_holdout(path: str | Path) -> dict:
    rows = pd.read_csv(path)
    required = {'model', 'actual', 'prediction'}
    if not required.issubset(rows):
        raise ValueError('Holdout file missing required columns')
    return {
        str(name): {
            'mae': round(float(np.mean(np.abs(group.actual - group.prediction))), 4),
            'rmse': round(float(np.sqrt(np.mean((group.actual - group.prediction) ** 2))), 4),
        }
        for name, group in rows.groupby('model')
    }


def realized_performance(log_path: str | Path, actuals_path: str | Path) -> dict:
    """Match predictions with externally supplied realized 30-day totals.

    Actuals CSV must contain country,date,duration_days,actual_revenue.
    Only exact matches are scored; no synthetic labels are fabricated.
    """
    actuals = pd.read_csv(actuals_path)
    keys = ['country', 'date', 'duration_days']
    if not set([*keys, 'actual_revenue']).issubset(actuals):
        raise ValueError('Actuals require country,date,duration_days,actual_revenue')
    predictions = pd.DataFrame([
        {key: event[key] for key in [*keys, 'predicted_revenue']}
        for event in read_events(log_path, limit=1000)
        if event.get('event') == 'prediction' and all(key in event for key in [*keys, 'predicted_revenue'])
    ])
    if predictions.empty:
        return {'matched': 0, 'status': 'insufficient_data'}
    result = predictions.merge(actuals, how='inner', on=keys)
    if result.empty:
        return {'matched': 0, 'status': 'insufficient_data'}
    errors = result.predicted_revenue - result.actual_revenue
    return {'matched': int(len(result)), 'status': 'evaluated',
            'mae': round(float(errors.abs().mean()), 4),
            'rmse': round(float(np.sqrt((errors ** 2).mean())), 4)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--holdout', default='reports/holdout_predictions.csv')
    parser.add_argument('--logs', default='logs/predictions.jsonl')
    parser.add_argument('--actuals', default=None)
    parser.add_argument('--alert-mae', type=float, default=None)
    args = parser.parse_args()
    score = {'holdout': summarize_holdout(args.holdout)}
    if args.actuals:
        runtime = realized_performance(args.logs, args.actuals)
        score['realized'] = runtime
        if args.alert_mae is not None and runtime.get('mae', 0) > args.alert_mae:
            score['alert'] = 'MAE threshold exceeded. Consider retraining.'
    print(json.dumps(score, indent=2))


if __name__ == '__main__':
    main()
