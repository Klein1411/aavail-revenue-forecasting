from pathlib import Path
import pytest
from src.train import train_pipeline

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope='session')
def setup_model(tmp_path_factory):
    work = tmp_path_factory.mktemp('aavail_test')
    model_dir = work / 'artifacts'
    report_dir = work / 'reports'
    bundle = train_pipeline(ROOT / 'data' / 'sample', model_dir, report_dir)
    return work, bundle
