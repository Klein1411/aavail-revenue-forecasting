"""Automated ingestion of AAVAIL invoice JSON and CSV transaction files."""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd


def load_transactions(folder: str | Path) -> pd.DataFrame:
    folder = Path(folder)
    if not folder.is_dir():
        raise FileNotFoundError(f"Input folder does not exist: {folder}")
    paths = sorted([*folder.rglob('*.json'), *folder.rglob('*.csv')])
    if not paths:
        raise ValueError(f"No JSON or CSV input files in {folder}")
    parts = []
    for path in paths:
        if path.suffix.lower() == '.json':
            raw = pd.read_json(path)
        else:
            raw = pd.read_csv(path)
        if raw.empty:
            continue
        # The source dataset uses partly inconsistent column naming.
        raw.columns = [str(c).strip().lower().replace(' ', '_') for c in raw.columns]
        aliases = {'unit_price': 'price', 'unitprice': 'price', 'total_price': 'price',
                   'country_name': 'country'}
        raw = raw.rename(columns={k: v for k, v in aliases.items() if k in raw.columns and v not in raw.columns})
        required = {'country', 'price'}
        if not required.issubset(raw.columns):
            raise ValueError(f"Missing columns {required - set(raw.columns)} in {path.name}")
        if 'date' in raw.columns:
            raw['date'] = pd.to_datetime(raw['date'], errors='coerce')
        elif {'year', 'month', 'day'}.issubset(raw.columns):
            raw['date'] = pd.to_datetime(raw[['year', 'month', 'day']], errors='coerce')
        else:
            raise ValueError(f"Missing date or year/month/day in {path.name}")
        raw['country'] = raw['country'].astype('string').str.strip()
        raw['price'] = pd.to_numeric(raw['price'], errors='coerce')
        clean = raw[['date', 'country', 'price']].dropna()
        clean = clean.loc[clean['country'].str.len() > 0].copy()
        parts.append(clean)
    if not parts:
        raise ValueError('No valid transaction rows found')
    data = pd.concat(parts, ignore_index=True)
    data['date'] = pd.to_datetime(data['date']).dt.normalize()
    data['country'] = data['country'].astype(str)
    return data.sort_values(['date', 'country']).reset_index(drop=True)


def daily_by_country(data: pd.DataFrame) -> dict[str, pd.Series]:
    if data.empty:
        raise ValueError('Cannot aggregate empty transactions')
    first, last = data['date'].min(), data['date'].max()
    dates = pd.date_range(first, last, freq='D')
    series = {}
    for name, subset in data.groupby('country'):
        revenue = subset.groupby('date')['price'].sum().reindex(dates, fill_value=0.0)
        revenue.index.name = 'date'
        series[str(name)] = revenue.astype(float)
    return series


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='data/sample')
    parser.add_argument('--output', default='reports/daily_revenue.csv')
    args = parser.parse_args()
    source = load_transactions(args.data_dir)
    agg = source.groupby(['country', 'date'], as_index=False)['price'].sum()
    dest = Path(args.output)
    dest.parent.mkdir(parents=True, exist_ok=True)
    agg.to_csv(dest, index=False)
    print(f"Ingested {len(source):,} transactions from {source.country.nunique()} countries -> {dest}")


if __name__ == '__main__':
    main()
