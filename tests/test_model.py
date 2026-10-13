import numpy as np
import pandas as pd
import pytest
from src.model import feature_vector, predict_window, supervised, recursive_forecast


def test_feature_and_training_no_leakage():
    series = pd.Series(np.arange(42, dtype=float), index=pd.date_range('2024-01-01', periods=42))
    X, y = supervised(series)
    assert X.shape == (14, 11)
    assert y[0] == 28
    assert X[0, 0] == 27  # previous day, never target day's value
    assert X[0, 3] == 0


def test_forecast_country_and_all(setup_model):
    _, model = setup_model
    day = '2025-03-01'
    one = predict_window(model, 'France', day, 30)
    world = predict_window(model, None, day, 30)
    assert one['predicted_revenue'] > 0
    assert world['predicted_revenue'] > one['predicted_revenue']
    assert len(one['series']) == 30
    assert sum(x['revenue'] for x in world['series']) == pytest.approx(world['predicted_revenue'], abs=.2)
    assert predict_window(model, 'fRaNcE', day, 1)['country'] == 'France'


def test_forecast_rejects_bad_input(setup_model):
    _, model = setup_model
    with pytest.raises(KeyError):
        predict_window(model, 'Atlantis', None)
    with pytest.raises(ValueError):
        predict_window(model, 'France', None, 0)
    with pytest.raises(ValueError):
        predict_window(model, 'France', '2023-01-01')


def test_chronological_model_scores(setup_model):
    _, item = setup_model
    assert 'seasonal_naive' in item['metrics']
    assert 'ridge' in item['metrics']
    assert 'random_forest' in item['metrics']
    assert item['holdout_days'] >= 14
