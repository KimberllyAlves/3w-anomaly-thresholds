"""Comparison of detectors with the well as unit (Demšar, 2006).

Friedman test on per-well means, followed by pairwise Wilcoxon tests with Holm correction.
Reads static_failures.csv (static_thresholds.py). Usage: python stats_tests.py
"""
import itertools

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon

from protocol import RESULTS


def holm(p):
    order = np.argsort(p)
    adj, acc = np.empty(len(p)), 0
    for k, i in enumerate(order):
        acc = max(acc, (len(p) - k) * p[i])
        adj[i] = min(1, acc)
    return adj


d = pd.read_csv(RESULTS / 'static_failures.csv')
d = d[d.set == 'benchmark']
for config in ['original', 'extended']:
    for metric in ['f1', 'auc']:
        x = d[(d.config == config) & (d.rule == 'max')]
        w = x.groupby(['well', 'det'])[metric].mean().unstack().dropna()
        p = friedmanchisquare(*[w[c] for c in w.columns]).pvalue
        print(f'\n{config}, {metric}, max rule, {len(w)} wells: Friedman p = {p:.4f}')
        pairs = list(itertools.combinations(w.columns, 2))
        ps = np.array([wilcoxon(w[a], w[b], zero_method='zsplit').pvalue for a, b in pairs])
        for (a, b), q in zip(pairs, holm(ps)):
            print(f'  {a} x {b}: Holm-adjusted p = {q:.4f}')
    for det in ['IF', 'LOF', 'VAE-Rec', 'VAE-Maha']:
        x = d[(d.config == config) & (d.det == det)]
        w = x.groupby(['well', 'rule']).f1.mean().unstack().dropna()
        print(f'{config} {det}: Friedman across rules p = {friedmanchisquare(*[w[c] for c in w.columns]).pvalue:.2e}')
