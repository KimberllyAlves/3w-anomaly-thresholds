"""Instance split, cleaning, scaling and windowing.

1. Split train/validation/test by whole instance, stratified by class.
2. Sentinel values (|x| > 1e10) and frozen stretches (same value for >= 300 samples, 5 min at 1 Hz)
   become NaN. Pressures are in Pa, so 1e10 is far above any real reading.
3. Sensors with a valid value in at least min_presence of the training instances are kept,
   minus the ones excluded in the config.
4. z-score with training statistics only.
5. Non-overlapping windows labelled with the most frequent label of their labelled samples
   (0 normal, 1 transient, 2 steady state). Windows with any NaN are dropped, not imputed:
   imputation would create constant stretches that are easy to reconstruct.

Usage: python -m vae3w.preprocessing --config configs/w32.yaml [--data PATH]
"""
from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from vae3w.data import DATA, Instance, LABEL_COLUMNS, list_instances, read, sensor_columns, state_of

SENTINEL = 1e10
FROZEN_RUN = 300


@dataclass
class Split:
    train: list[Instance]
    val: list[Instance]
    test: list[Instance]


def split_instances(root=DATA, train_frac=0.7, val_frac=0.15, test_frac=0.15, seed=42) -> Split:
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-6
    rng = random.Random(seed)
    by_class: dict[int, list[Instance]] = {}
    for inst in list_instances(root):
        by_class.setdefault(inst.class_dir, []).append(inst)
    train, val, test = [], [], []
    for insts in by_class.values():
        shuffled = insts[:]
        rng.shuffle(shuffled)
        n_train, n_val = round(len(shuffled) * train_frac), round(len(shuffled) * val_frac)
        train += shuffled[:n_train]
        val += shuffled[n_train:n_train + n_val]
        test += shuffled[n_train + n_val:]
    return Split(train, val, test)


def split_from_config(cfg, root=DATA) -> Split:
    p = cfg['preprocessing']
    return split_instances(root, p['train_frac'], p['val_frac'], p['test_frac'], cfg['random_seed'])


def frozen_mask(series: pd.Series, min_run=FROZEN_RUN) -> pd.Series:
    """True for samples inside a run of >= min_run identical consecutive values."""
    v = series.to_numpy()
    if len(v) == 0:
        return pd.Series([], index=series.index, dtype=bool)
    same = np.empty(len(v), dtype=bool)
    same[0] = False
    same[1:] = v[1:] == v[:-1]
    run_id = np.cumsum(~same)
    run_len = pd.Series(run_id).map(pd.Series(run_id).value_counts()).to_numpy()
    return pd.Series(run_len >= min_run, index=series.index)


def read_clean(inst: Instance) -> pd.DataFrame:
    df = read(inst)
    cols = sensor_columns(df)
    df[cols] = df[cols].mask(df[cols].abs() > SENTINEL)
    for c in cols:
        df.loc[frozen_mask(df[c]), c] = np.nan
    return df


def coverage(instances, exclude_prefixes=('ESTADO-',)) -> pd.Series:
    """Fraction of instances with at least one valid value, per sensor."""
    count: dict[str, int] = {}
    for inst in instances:
        df = read_clean(inst)
        for c in sensor_columns(df):
            if c.startswith(exclude_prefixes):
                continue
            count.setdefault(c, 0)
            if df[c].notna().any():
                count[c] += 1
    return pd.Series({c: n / len(instances) for c, n in count.items()}).sort_values(ascending=False)


def normalization_stats(instances, sensors) -> pd.DataFrame:
    sums = pd.Series(0.0, index=sensors)
    sums_sq = pd.Series(0.0, index=sensors)
    counts = pd.Series(0, index=sensors)
    for inst in instances:
        df = read_clean(inst).reindex(columns=sensors)
        sums += df.fillna(0).sum()
        sums_sq += (df.fillna(0) ** 2).sum()
        counts += df.notna().sum()
    counts = counts.replace(0, np.nan)
    mean = sums / counts
    std = ((sums_sq / counts) - mean ** 2).clip(lower=1e-12) ** 0.5
    return pd.DataFrame({'mean': mean, 'std': std})


def make_windows(df, sensors, stats, size, stride):
    """(n, size, n_sensors) windows and their labels; windows with NaN or no label are dropped."""
    x = (df.reindex(columns=sensors) - stats['mean']) / stats['std']
    labels = df['class']
    windows, y = [], []
    for start in range(0, len(x) - size + 1, stride):
        chunk = x.iloc[start:start + size]
        if chunk.isna().any().any():
            continue
        lab = labels.iloc[start:start + size].dropna()
        if lab.empty:
            continue
        windows.append(chunk.to_numpy(dtype=np.float32))
        y.append(state_of(lab.mode().iloc[0]))
    if not windows:
        return np.empty((0, size, len(sensors)), dtype=np.float32), np.empty((0,), dtype=np.int64)
    return np.stack(windows), np.array(y, dtype=np.int64)


def process_split(instances, sensors, stats, size, stride):
    """Windows, labels, index of the source instance in `instances` and its class folder."""
    W, Y, ids, cls = [], [], [], []
    for i, inst in enumerate(instances):
        w, y = make_windows(read_clean(inst), sensors, stats, size, stride)
        if len(w) == 0:
            continue
        W.append(w)
        Y.append(y)
        ids.append(np.full(len(w), i, dtype=np.int64))
        cls.append(np.full(len(w), inst.class_dir, dtype=np.int64))
    if not W:
        e = np.empty((0,), dtype=np.int64)
        return np.empty((0, size, len(sensors)), dtype=np.float32), e, e, e
    return np.concatenate(W), np.concatenate(Y), np.concatenate(ids), np.concatenate(cls)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--data', default=DATA)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config, encoding='utf-8'))
    p = cfg['preprocessing']
    size, stride = p['window_size'], p.get('stride', p['window_size'])

    split = split_from_config(cfg, args.data)
    print(f'instances: train {len(split.train)}, val {len(split.val)}, test {len(split.test)}')
    cov = coverage(split.train)
    print('sensor coverage in training instances:')
    print(cov.round(3).to_string())
    sensors = sorted(c for c, f in cov.items() if f >= p['min_presence_frac'])
    sensors = [s for s in sensors if s not in p.get('exclude_sensors', [])]
    print('sensors:', sensors)

    stats = normalization_stats(split.train, sensors)
    out = Path(cfg['output_dir']) / 'processed'
    out.mkdir(parents=True, exist_ok=True)
    for name, insts in [('train', split.train), ('val', split.val), ('test', split.test)]:
        w, y, ids, cls = process_split(insts, sensors, stats, size, stride)
        np.savez_compressed(out / f'{name}.npz', windows=w, labels=y, instance_ids=ids,
                            class_dirs=cls, sensors=np.array(sensors))
        print(f'{name}: {len(w)} windows, states {np.bincount(y, minlength=3).tolist()}')
    stats.to_csv(out / 'normalization_stats.csv')


if __name__ == '__main__':
    main()
