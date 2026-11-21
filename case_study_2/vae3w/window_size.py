"""Window size from the decorrelation time of each sensor.

For each sampled instance and sensor, the ADF test decides whether the series is differenced, then
the ACF is computed and the decorrelation time is the first lag after which |ACF| stays below 1/e
for min_consecutive lags in a row. Requiring a sustained crossing avoids stopping at the first dip
of an oscillating signal. The window is an aggregate (median by default) of these lags.

This is the initial analysis, on raw data and all sensors. scripts/window_size_final_sensors.py
repeats it on the cleaned training instances and the sensors kept by select_sensors.py.
"""
from __future__ import annotations

import argparse
import math
import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import acf, adfuller

from .data import DATA, list_instances, read, sensor_columns

THRESHOLD = 1 / math.e
warnings.filterwarnings('ignore', category=FutureWarning, module='statsmodels')


def is_stationary(x, alpha=0.05, max_obs=5000):
    if len(x) > max_obs:                      # central stretch, ADF is slow on long series
        start = (len(x) - max_obs) // 2
        x = x.iloc[start:start + max_obs]
    try:
        return adfuller(x, autolag='AIC')[1] < alpha
    except Exception:
        return False


def decorrelation_lag(r, threshold=THRESHOLD, min_consecutive=15):
    for lag in range(1, len(r)):
        seg = r[lag:lag + min_consecutive]
        if len(seg) < min_consecutive:
            break
        if np.all(np.abs(seg) < threshold):
            return lag
    return None


def series_lag(x, max_lag=4000, threshold=THRESHOLD, min_consecutive=15, min_obs=30, difference='auto'):
    """Returns (lag, differenced) or None when the series is too short or constant."""
    x = x.dropna()
    if len(x) < min_obs or x.nunique() <= 1:
        return None
    diff = difference == 'always' or (difference == 'auto' and not is_stationary(x))
    if diff:
        x = x.diff().dropna()
        if len(x) < min_obs or x.nunique() <= 1:
            return None
    try:
        r = acf(x, nlags=min(max_lag, len(x) - 1), fft=True, missing='drop')
    except Exception:
        return None
    return decorrelation_lag(r, threshold, min_consecutive), diff


def run(root, n_per_class=15, max_lag=4000, seed=42, difference='auto', min_consecutive=15):
    rng = random.Random(seed)
    by_class = {}
    for inst in list_instances(root):
        by_class.setdefault(inst.class_dir, []).append(inst)
    sample = [i for insts in by_class.values() for i in rng.sample(insts, min(n_per_class, len(insts)))]

    rows = []
    for inst in sample:
        df = read(inst)
        nlags = min(max_lag, max(1, len(df) - 1))
        for s in sensor_columns(df):
            if s.startswith('ESTADO-'):           # valve states, not continuous signals
                continue
            res = series_lag(df[s], nlags, min_consecutive=min_consecutive, difference=difference)
            if res is not None:
                rows.append({'instance': inst.path.name, 'class_dir': inst.class_dir, 'sensor': s,
                             'decorrelation_lag': res[0], 'differenced': res[1]})
    return pd.DataFrame(rows)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data', default=DATA)
    p.add_argument('--n-per-class', type=int, default=15)
    p.add_argument('--max-lag', type=int, default=4000)
    p.add_argument('--difference', default='auto', choices=['auto', 'always', 'never'])
    p.add_argument('--min-consecutive', type=int, default=15)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--output', default='outputs/acf_summary.csv')
    a = p.parse_args()

    summary = run(a.data, a.n_per_class, a.max_lag, a.seed, a.difference, a.min_consecutive)
    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(a.output, index=False)
    valid = summary.dropna(subset=['decorrelation_lag'])
    print(valid.groupby('sensor')['decorrelation_lag'].median().sort_values(ascending=False).round(1).to_string())
    print(f'\n{summary["differenced"].mean():.0%} of the series differenced, '
          f'{summary["decorrelation_lag"].isna().sum()} without a sustained crossing')
    print(f'median over {valid["sensor"].nunique()} sensors and {len(valid)} series: '
          f'{valid["decorrelation_lag"].median():.0f} samples')
