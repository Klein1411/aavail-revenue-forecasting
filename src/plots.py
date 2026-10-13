"""Create real EDA and baseline/model validation plots as saved PNGs."""
from __future__ import annotations
import argparse
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from .ingest import load_transactions


def build_plots(data_dir: str | Path, reports_dir: str | Path) -> list[Path]:
    reports = Path(reports_dir)
    reports.mkdir(parents=True, exist_ok=True)
    transactions = load_transactions(data_dir)
    daily = transactions.groupby('date').price.sum().sort_index()
    top = transactions.groupby('country').price.sum().sort_values(ascending=False).head(12)
    output = []
    fig, ax = plt.subplots(figsize=(10, 4))
    daily.plot(ax=ax, color='#2563eb')
    ax.set(title='Daily revenue — all countries', xlabel='Date', ylabel='Revenue (dataset units)')
    ax.grid(alpha=.2)
    fig.tight_layout()
    path = reports / 'eda_daily_revenue.png'
    fig.savefig(path, dpi=140)
    plt.close(fig)
    output.append(path)
    fig, ax = plt.subplots(figsize=(9, 4))
    top.sort_values().plot(kind='barh', ax=ax, color='#0891b2')
    ax.set(title='Revenue by country (top 12)', xlabel='Revenue (dataset units)', ylabel='Country')
    fig.tight_layout()
    path = reports / 'eda_country_revenue.png'
    fig.savefig(path, dpi=140)
    plt.close(fig)
    output.append(path)
    holdout = reports / 'holdout_predictions.csv'
    if holdout.exists():
        rows = pd.read_csv(holdout)
        most = rows.loc[rows.country == rows.country.value_counts().index[0]]
        fig, ax = plt.subplots(figsize=(11, 4.5))
        for model, subset in most.groupby('model'):
            style = dict(linewidth=1.7, linestyle='--' if model == 'seasonal_naive' else '-')
            ax.plot(pd.to_datetime(subset.date), subset.prediction, label=model, **style)
        truth = most.loc[most.model == 'seasonal_naive']
        ax.plot(pd.to_datetime(truth.date), truth.actual, label='actual', linewidth=2.8, color='black')
        ax.set(title=f"Unseen chronological holdout — {truth.country.iloc[0]}",
               ylabel='Daily revenue', xlabel='Date')
        ax.legend()
        ax.grid(alpha=.18)
        fig.autofmt_xdate()
        fig.tight_layout()
        path = reports / 'model_vs_baseline.png'
        fig.savefig(path, dpi=140)
        plt.close(fig)
        output.append(path)
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='data/sample')
    parser.add_argument('--reports-dir', default='reports')
    args = parser.parse_args()
    print(*build_plots(args.data_dir, args.reports_dir), sep='\n')


if __name__ == '__main__':
    main()
