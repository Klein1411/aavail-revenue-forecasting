from src.logging_service import read_events, write_event
from src.monitor import realized_performance, summarize_holdout


def test_log_paths_are_test_only(tmp_path):
    sandbox_log = tmp_path / 'logs' / 'test_predictions.jsonl'
    assert read_events(sandbox_log) == []
    write_event(sandbox_log, {'event': 'prediction', 'country': 'France', 'predicted_revenue': 45})
    events = read_events(sandbox_log)
    assert len(events) == 1 and events[0]['country'] == 'France'
    assert 'timestamp_utc' in events[0]


def test_monitor_and_realized_labels(tmp_path, setup_model):
    folder, _ = setup_model
    scores = summarize_holdout(folder / 'reports' / 'holdout_predictions.csv')
    assert scores['seasonal_naive']['mae'] >= 0
    log = tmp_path / 'activity.jsonl'
    write_event(log, {'event': 'prediction', 'country': 'France', 'date': '2025-03-01',
                      'duration_days': 30, 'predicted_revenue': 100.0})
    feedback = tmp_path / 'feedback.csv'
    feedback.write_text('country,date,duration_days,actual_revenue\nFrance,2025-03-01,30,90\n')
    result = realized_performance(log, feedback)
    assert result['matched'] == 1 and result['mae'] == 10.0
