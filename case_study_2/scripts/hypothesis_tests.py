"""Normal vs transient comparison of the epistemic uncertainty, per class.

For each split (validation, test) and class:
  - ratio of means and of medians (transient / normal), Mann-Whitney p and Cliff's delta;
  - Cliff's delta of the residual of log(epistemic) after a log-log linear fit on log(recon_error),
    fitted once on all normal windows of the split (does uncertainty add anything to the error?);
  - the same comparison split by instance origin (real, simulated, drawn).
Also prints the Spearman correlation between epistemic and recon_error and how the origins are
mixed between normal and transient windows.

The origin of each window comes from instance_ids in the npz, which index the instance lists of
the split, so the split is rebuilt from the dataset folder (only the file names are needed).

Usage:
  python scripts/hypothesis_tests.py --config configs/w32.yaml \
      --val outputs_w32/uncertainty/val_uncertainty.csv --test outputs_w32/uncertainty/test_uncertainty.csv
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import mannwhitneyu, spearmanr

from vae3w.data import DATA
from vae3w.preprocessing import split_from_config

CLASSES = [1, 5, 6, 7, 8, 9]
SCORE = 'epistemic'
SMALL = 30


def cliff(t, n):
    u = mannwhitneyu(t, n).statistic
    return 2 * u / (len(t) * len(n)) - 1


def compare(n, t):
    p = mannwhitneyu(t, n).pvalue
    return {'n_normal': len(n), 'n_transient': len(t), 'mean_ratio': t.mean() / n.mean(),
            'median_ratio': np.median(t) / np.median(n), 'cliff': cliff(t, n), 'p': p}


def load(csv, npz, instances):
    df = pd.read_csv(csv)
    ids = np.load(npz, allow_pickle=True)['instance_ids']
    assert len(ids) == len(df), f'{csv} and {npz} do not match'
    df['instance'] = ids
    df['origin'] = [instances[i].origin for i in ids]
    normal = df[df.label == 0]
    fit = np.polyfit(np.log(normal.recon_error), np.log(normal[SCORE]), 1)
    df['residual'] = np.log(df[SCORE]) - np.polyval(fit, np.log(df.recon_error))
    return df


p = argparse.ArgumentParser()
p.add_argument('--config', default='configs/w32.yaml')
p.add_argument('--data', default=DATA)
p.add_argument('--val', required=True)
p.add_argument('--test', required=True)
p.add_argument('--processed', help='folder with val.npz and test.npz (default: output_dir/processed)')
a = p.parse_args()
cfg = yaml.safe_load(open(a.config))
processed = Path(a.processed or Path(cfg['output_dir']) / 'processed')
split = split_from_config(cfg, a.data)
data = {'val': load(a.val, processed / 'val.npz', split.val),
        'test': load(a.test, processed / 'test.npz', split.test)}

rows = []
for name, df in data.items():
    print(f'{name}: Spearman(epistemic, recon_error) = {spearmanr(df[SCORE], df.recon_error)[0]:.3f}')
    for c in CLASSES:
        d = df[df.class_dir == c]
        n, t = d[d.label == 0], d[d.label == 1]
        if len(n) and len(t):
            r = compare(n[SCORE].to_numpy(), t[SCORE].to_numpy())
            r['cliff_residual'] = cliff(t.residual.to_numpy(), n.residual.to_numpy())
            rows.append({'split': name, 'class': c, **r})
res = pd.DataFrame(rows)
res['small_sample'] = res[['n_normal', 'n_transient']].min(axis=1) < SMALL
print('\n' + res.round(3).to_string(index=False))

rows = []
for name, df in data.items():
    for c in CLASSES:
        for origin, d in df[df.class_dir == c].groupby('origin'):
            n, t = d[d.label == 0], d[d.label == 1]
            if len(n) and len(t):
                r = compare(n[SCORE].to_numpy(), t[SCORE].to_numpy())
                rows.append({'split': name, 'class': c, 'origin': origin, 'instances': d.instance.nunique(),
                             'n_normal': r['n_normal'], 'n_transient': r['n_transient'],
                             'cliff': f"{r['cliff']:+.2f}" + ('*' if r['p'] >= 0.05 else '')})
print('\nby origin (* p >= 0.05)')
print(pd.DataFrame(rows).to_string(index=False))

print('\ncomposition: share of simulated windows and median epistemic of normal windows by origin')
for name, df in data.items():
    for c in CLASSES:
        d = df[df.class_dir == c]
        sim = d.groupby('label').origin.apply(lambda o: (o == 'simulated').mean())
        med = d[d.label == 0].groupby('origin')[SCORE].median()
        print(f'{name:5} class {c}: simulated normal {sim.get(0, np.nan):.0%} transient {sim.get(1, np.nan):.0%}  '
              + '  '.join(f'{o} {v:.3f}' for o, v in med.items()))
