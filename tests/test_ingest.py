import json
import pandas as pd
import pytest
from src.ingest import load_transactions, daily_by_country


def test_json_aliases_and_bad_data(tmp_path):
    payload = [
        {'Country': ' Canada ', 'year': 2022, 'month': 1, 'day': 2, 'unit_price': 50},
        {'Country': 'Canada', 'year': 2022, 'month': 1, 'day': 3, 'unit_price': 'broken'},
    ]
    (tmp_path / 'invoices.json').write_text(json.dumps(payload), encoding='utf-8')
    records = load_transactions(tmp_path)
    assert len(records) == 1
    assert records.country.iloc[0] == 'Canada'
    assert records.price.iloc[0] == 50


def test_missing_input_and_missing_required_column(tmp_path):
    with pytest.raises(ValueError, match='No JSON or CSV'):
        load_transactions(tmp_path)
    (tmp_path / 'broken.csv').write_text('country,year,month,day\nCanada,2022,1,1\n')
    with pytest.raises(ValueError, match='Missing columns'):
        load_transactions(tmp_path)


def test_daily_fills_missing_days(tmp_path):
    (tmp_path / 'sales.csv').write_text('country,date,price\nCanada,2024-01-01,10\nCanada,2024-01-03,5\n')
    series = daily_by_country(load_transactions(tmp_path))
    assert series['Canada'].tolist() == [10.0, 0.0, 5.0]
