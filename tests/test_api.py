import json
import threading
from urllib.request import urlopen, Request
from urllib.error import HTTPError
import pytest
from src.app import Config, create_server


@pytest.fixture()
def api(tmp_path, setup_model):
    work, _ = setup_model
    conf = Config(artifacts=work / 'artifacts', reports=work / 'reports',
                  log_file=tmp_path / 'private-test-logs.jsonl',
                  admin_token='unit-test-admin', train_token='unit-test-train')
    server = create_server(conf, host='127.0.0.1', port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f'http://127.0.0.1:{server.server_address[1]}'
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)


def call(url, method='GET', body=None, token=None):
    raw = json.dumps(body).encode() if body is not None else None
    headers = {'Content-Type': 'application/json'}
    if token:
        headers['Authorization'] = f'Bearer {token}'
    req = Request(url, data=raw, headers=headers, method=method)
    try:
        with urlopen(req, timeout=15) as response:
            return response.status, json.load(response)
    except HTTPError as exc:
        return exc.code, json.load(exc)


def test_health_countries_and_metrics(api):
    assert call(api + '/health')[0] == 200
    assert 'France' in call(api + '/countries')[1]['countries']
    assert 'seasonal_naive' in call(api + '/metrics')[1]['metrics']


def test_country_and_all_country_post(api):
    country, specific = call(api + '/predict', 'POST', {'country': 'France', 'date': '2025-03-01', 'duration': 30})
    total, combined = call(api + '/predict?date=2025-03-01', 'POST', {})
    assert country == total == 200
    assert specific['country'] == 'France'
    assert combined['country'] == 'all'
    assert combined['predicted_revenue'] > specific['predicted_revenue']
    assert len(combined['series']) == 30


def test_bad_input_and_protected_logs(api):
    assert call(api + '/predict', 'POST', {'country': 'Neverland'})[0] == 404
    assert call(api + '/predict', 'POST', {'date': 'not-a-date'})[0] == 400
    assert call(api + '/logs')[0] == 403
    assert call(api + '/train', 'POST', {})[0] == 403
    assert call(api + '/logs?limit=10', token='unit-test-admin')[0] == 200
    assert call(api + '/missing')[0] == 404
