"""Transient/normal ratio of the mean epistemic uncertainty, per class and sensor.

Hatched bars: Mann-Whitney p >= 0.05.
Usage: python scripts/fig_uncertainty_by_sensor.py VAL_CSV [OUTPUT_BASENAME]
"""
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from scipy.stats import mannwhitneyu

SENSORS = ['P-MON-CKP', 'P-TPT', 'T-JUS-CKP']
CLASSES = [1, 5, 6, 7, 8, 9]
SERIES = {'epistemic': ('Aggregated', '#4d4d4d'), 'epistemic_P-MON-CKP': ('P-MON-CKP', '#1b6ca8'),
          'epistemic_P-TPT': ('P-TPT', '#e08214'), 'epistemic_T-JUS-CKP': ('T-JUS-CKP', '#2a9d5c')}

df = pd.read_csv(sys.argv[1])
out = sys.argv[2] if len(sys.argv) > 2 else 'uncertainty_by_sensor'

ratios = {}
for c in CLASSES:
    n, t = df[(df.class_dir == c) & (df.label == 0)], df[(df.class_dir == c) & (df.label == 1)]
    if len(n) and len(t):
        ratios[c] = {k: (t[k].mean() / n[k].mean(), mannwhitneyu(t[k], n[k]).pvalue) for k in SERIES}
for c, r in ratios.items():
    print(c, '  '.join(f'{SERIES[k][0]} {v:.2f} (p={p:.1e})' for k, (v, p) in r.items()))

plt.rcParams.update({'font.family': 'serif', 'font.size': 10.5})
fig, ax = plt.subplots(figsize=(8.5, max(4.2, len(ratios))))
h = 0.19
for i, (k, (label, color)) in enumerate(SERIES.items()):
    for j, c in enumerate(ratios):
        v, p = ratios[c][k]
        y = j + (i - 1.5) * h
        ax.barh(y, v, height=h * 0.92, color=color, hatch='' if p < 0.05 else '///',
                edgecolor='white', linewidth=0.4, zorder=3)
        ax.text(v * (1.12 if v >= 1 else 0.88), y, f'{v:.2f}', va='center',
                ha='left' if v >= 1 else 'right', fontsize=7.3)
values = [v for r in ratios.values() for v, _ in r.values()]
ax.axvline(1, color='#b02418', ls='--', lw=1.1, zorder=2)
ax.set_xscale('log')
ax.set_xlim(min(values) * 0.5, max(values) * 2)
ax.set_yticks(range(len(ratios)), [f'Class {c}' for c in ratios])
ax.set_xlabel('Epistemic uncertainty ratio, transient / normal (log scale)')
ax.grid(axis='x', which='both', alpha=0.2)
ax.invert_yaxis()
ax.legend(handles=[Patch(facecolor=col, label=lab) for lab, col in SERIES.values()], fontsize=9)
fig.tight_layout()
fig.savefig(f'{out}.png', dpi=300, bbox_inches='tight')
fig.savefig(f'{out}.pdf', bbox_inches='tight')
