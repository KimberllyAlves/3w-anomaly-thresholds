"""Mean ROC curves of the four detectors with the operating point of each static rule.

Each instance has its own model and the reported AUC is the mean over instances, so the curve of a
detector is the vertical average of the per-instance curves, and each operating point is the mean
false alarm and detection rate of a rule across instances. Reads static_scores.pkl.
Usage: python fig_roc.py
"""
import pickle

import numpy as np
from matplotlib.lines import Line2D
from sklearn.metrics import roc_auc_score, roc_curve

from detectors import RULES
from protocol import RESULTS
from style import plt, DETS, COLOR, grid

NAMES = {'max': ('Training maximum', 'o'), 'p95': ('95th percentile', 's'),
         'm2sd': ('Mean + 2 sd', '^'), 'p50': ('Median', 'D')}
FA_GRID = np.linspace(0, 1, 201)

data = [e for e in pickle.load(open(RESULTS / 'static_scores.pkl', 'rb')) if e['set'] == 'benchmark']
fig, axs = plt.subplots(1, 2, figsize=(8.6, 4.9), sharey=True)
for ax, (config, title) in zip(axs, [('original', 'Original training'), ('extended', 'Extended training')]):
    x = [e for e in data if e['config'] == config and 0 < e['y'].mean() < 1]
    ax.plot([0, 1], [0, 1], ls='--', lw=0.8, color='#b5b4ae')
    for det in DETS:
        curves, aucs, pts = [], [], {r: [] for r in RULES}
        for e in x:
            s_tr, s_te = e['scores'][det]
            y = e['y']
            fa, dr, _ = roc_curve(y, s_te)
            curves.append(np.interp(FA_GRID, fa, dr))
            aucs.append(roc_auc_score(y, s_te))
            for r, f in RULES.items():
                al = s_te > f(s_tr)
                pts[r].append((al[y == 0].mean(), al[y == 1].mean()))
        mean = np.mean(curves, axis=0)
        mean[0] = 0.0
        ax.plot(FA_GRID, mean, color=COLOR[det], lw=1.8, label=f'{det} (AUC = {np.mean(aucs):.3f})')
        for r, (_, marker) in NAMES.items():
            fa_m, dr_m = np.mean(pts[r], axis=0)
            ax.plot(fa_m, dr_m, marker=marker, ms=7, color=COLOR[det], mec='white', mew=0.8, ls='none', zorder=5)
    ax.set_title(f'{title} ({len(x)} instances)', loc='left', fontsize=10)
    ax.set_xlabel('False alarm rate')
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.02)
    ax.set_aspect('equal')
    grid(ax, 'both')
    ax.legend(loc='lower right', fontsize=8, frameon=False)
axs[0].set_ylabel('Detection rate')
markers = [Line2D([], [], marker=m, ls='none', color='#52514e', mec='white', ms=7, label=n) for n, m in NAMES.values()]
fig.legend(handles=markers, loc='lower center', ncol=4, fontsize=8.5, frameon=False,
           title='Operating points (threshold rules)', title_fontsize=8.5, bbox_to_anchor=(0.5, -0.02))
fig.tight_layout(rect=(0, 0.12, 1, 1))
fig.savefig(RESULTS / 'fig_roc.png', dpi=300, facecolor='white')
