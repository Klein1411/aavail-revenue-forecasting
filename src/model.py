"""Leakage-aware temporal features and recursive daily-revenue predictions."""
from __future__ import annotations
from datetime import timedelta
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor

MIN_HISTORY = 28
FEATURES = ('lag1', 'lag7', 'lag14', 'lag28', 'avg7', 'avg28',
            'dow_sin', 'dow_cos', 'doy_sin', 'doy_cos', 'trend')


def feature_vector(history: list[float], day: pd.Timestamp, elapsed: int) -> np.ndarray:
    if len(history) < MIN_HISTORY:
        raise ValueError('Need at least 28 historical days')
    dow, doy = day.dayofweek, day.dayofyear
    return np.asarray([
        history[-1], history[-7], history[-14], history[-28],
        np.mean(history[-7:]), np.mean(history[-28:]),
        np.sin(2 * np.pi * dow / 7), np.cos(2 * np.pi * dow / 7),
        np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25),
        elapsed / 365.0,
    ], dtype=float)


def supervised(series: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    history = series.to_numpy(dtype=float).tolist()
    if len(history) <= MIN_HISTORY:
        raise ValueError('Need more than 28 days to fit model')
    X = [feature_vector(history[:i], pd.Timestamp(series.index[i]), i) for i in range(MIN_HISTORY, len(history))]
    return np.vstack(X), np.asarray(history[MIN_HISTORY:], dtype=float)


def candidates() -> dict:
    return {
        'ridge': Ridge(alpha=10.0),
        'random_forest': RandomForestRegressor(n_estimators=48, max_depth=9,
                                               min_samples_leaf=3, random_state=42, n_jobs=1),
    }


def recursive_forecast(model, history: list[float], start_day: pd.Timestamp,
                       days: int, *, baseline: bool = False, origin_days: int | None = None) -> list[float]:
    if days < 1 or days > 500:
        raise ValueError('days must be in 1..500')
    if len(history) < MIN_HISTORY:
        raise ValueError('Not enough historical observations')
    past = list(history)
    index_base = len(past) if origin_days is None else origin_days
    predictions = []
    for i in range(days):
        current = start_day + pd.Timedelta(days=i)
        if baseline:
            value = past[-7]  # seasonal-naive baseline
        else:
            vector = feature_vector(past, current, index_base + i)
            value = float(model.predict(vector.reshape(1, -1))[0])
        value = max(0.0, float(value))
        predictions.append(value)
        past.append(value)
    return predictions


def predict_window(bundle: dict, country: str | None, date: str | None, duration: int = 30) -> dict:
    if not isinstance(duration, int) or not 1 <= duration <= 90:
        raise ValueError('duration must be an integer in 1..90')
    available = list(bundle['histories'].keys())
    if country is None or str(country).strip().casefold() in {'all', 'global', 'worldwide'}:
        countries = available
        selected = 'all'
    else:
        matches = [c for c in available if c.casefold() == str(country).strip().casefold()]
        if not matches:
            raise KeyError(f"Unknown country: {country}. Available: {', '.join(available)}")
        countries = matches
        selected = matches[0]
    last = pd.Timestamp(bundle['last_date'])
    start = pd.Timestamp(date) if date is not None else last + pd.Timedelta(days=1)
    if pd.isna(start) or start != start.normalize():
        raise ValueError('date must be YYYY-MM-DD')
    if start > last + pd.Timedelta(days=365):
        raise ValueError('Prediction date exceeds one-year forecast limit')
    outputs = []
    for c in countries:
        s = bundle['histories'][c]
        oldest = pd.Timestamp(s.index[0])
        if start < oldest + pd.Timedelta(days=MIN_HISTORY):
            raise ValueError(f'Date must be >= {(oldest + pd.Timedelta(days=MIN_HISTORY)).date()}')
        if start <= last:
            seen = s.loc[s.index < start]
            hist = seen.tolist()
        else:
            hist = s.tolist()
            bridge = (start - last).days - 1
            if bridge:
                generated = recursive_forecast(bundle['models'][c], hist, last + pd.Timedelta(days=1), bridge)
                hist.extend(generated)
        result = recursive_forecast(bundle['models'][c], hist, start, duration)
        outputs.append(result)
    by_day = np.asarray(outputs, dtype=float).sum(axis=0)
    return {
        'country': selected,
        'date': start.strftime('%Y-%m-%d'),
        'duration_days': duration,
        'predicted_revenue': round(float(by_day.sum()), 2),
        'model': bundle['model_name'],
        'currency': 'AAVAIL dataset units (unspecified currency)',
        'series': [
            {'date': (start + pd.Timedelta(days=i)).strftime('%Y-%m-%d'),
             'revenue': round(float(v), 2)} for i, v in enumerate(by_day)
        ],
        'note': ('Retrospective estimate on in-sample history; use held-out plots for unbiased evaluation.'
                 if start <= last else 'Out-of-sample recursive forecast.'),
    }
