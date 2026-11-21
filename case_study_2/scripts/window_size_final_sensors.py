"""Decorrelation time restricted to the final sensors, on cleaned training instances.

Same criterion as vae3w.window_size (ADF, 15 consecutive lags with |ACF| < 1/e), with 15 training
instances per class and series of at least 200 valid samples.
"""
import argparse

import numpy as np
import yaml
from statsmodels.tsa.stattools import acf

from vae3w.data import DATA
from vae3w.preprocessing import read_clean, split_from_config
from vae3w.window_size import THRESHOLD, decorrelation_lag, is_stationary

SENSORS = ['P-MON-CKP', 'P-TPT', 'T-JUS-CKP']
MAX_LAG = 400
MIN_CONSECUTIVE = 15
N_PER_CLASS = 15
MIN_OBS = 200


def lag_of(x):
    x = x.dropna()
    if len(x) < MIN_OBS or x.nunique() <= 1:
        return None
    if not is_stationary(x):
        d = x.diff().dropna()
        if len(d) >= MIN_OBS and d.nunique() > 1:
            x = d
    nlags = min(MAX_LAG, len(x) - 1)
    if nlags < MIN_CONSECUTIVE + 1:
        return None
    return decorrelation_lag(acf(x, nlags=nlags, fft=True), THRESHOLD, MIN_CONSECUTIVE)


p = argparse.ArgumentParser()
p.add_argument('--config', default='configs/w32.yaml')
p.add_argument('--data', default=DATA)
a = p.parse_args()
cfg = yaml.safe_load(open(a.config))
split = split_from_config(cfg, a.data)

rng = np.random.default_rng(cfg['random_seed'])
by_class = {}
for inst in split.train:
    by_class.setdefault(inst.class_dir, []).append(inst)
sample = [insts[i] for insts in by_class.values()
          for i in rng.choice(len(insts), size=min(N_PER_CLASS, len(insts)), replace=False)]

lags = {s: [] for s in SENSORS}
for inst in sample:
    df = read_clean(inst)
    for s in SENSORS:
        if s in df.columns and (lag := lag_of(df[s])) is not None:
            lags[s].append(lag)

for s, v in lags.items():
    print(f'{s:<12} n={len(v):4d}  median={np.median(v):5.0f}  IQR={np.percentile(v, 25):.0f}-{np.percentile(v, 75):.0f}')
allv = sum(lags.values(), [])
print(f'\nmedian over the {len(SENSORS)} sensors: {np.median(allv):.0f} samples, '
      f'nearest power of two {2 ** round(np.log2(np.median(allv)))}')
