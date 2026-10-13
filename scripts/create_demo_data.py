"""Generate reproducible SYNTHETIC transactions for an offline runnable demo."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd


def make_demo(path: str | Path):
    rng = np.random.default_rng(42)
    rows = []
    countries = [('United Kingdom', 125.0), ('France', 72.0), ('Germany', 93.0)]
    for d in pd.date_range('2024-01-01', '2025-02-28', freq='D'):
        for country, base in countries:
            weekly = 1.35 if d.dayofweek in (4, 5) else 0.87 if d.dayofweek == 0 else 1.0
            seasonal = 1.0 + 0.2 * np.sin(2 * np.pi * d.dayofyear / 365.25)
            growth = 1.0 + (d - pd.Timestamp('2024-01-01')).days * 0.00055
            daily = max(0.0, base * weekly * seasonal * growth + rng.normal(0, 8))
            for k, fraction in enumerate((.38, .35, .27)):
                rows.append({'date': d.strftime('%Y-%m-%d'), 'country': country,
                             'price': round(float(daily * fraction), 2),
                             'invoice': f'SYN-{country[:2].upper()}-{d.strftime("%Y%m%d")}-{k}'})
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False)
    print(f'Created {len(rows):,} synthetic transactions -> {output}')


if __name__ == '__main__':
    make_demo('data/sample/aavail_demo_transactions.csv')
