"""Confusion matrix per detector with the failure split into transient and steady state
(threshold at the maximum training score). Reads static_scores.pkl.
Usage: python fig_confusion_matrix.py [original|extended]"""
import pickle
import sys

import numpy as np

from protocol import RESULTS
from style import plt, DETS

CONFIG = sys.argv[1] if len(sys.argv) > 1 else 'extended'
data = [e for e in pickle.load(open(RESULTS / 'static_scores.pkl', 'rb'))
        if e['config'] == CONFIG and e['set'] == 'benchmark']
ROWS = ['Normal', 'Transient', 'Steady state']
GOOD, BAD = np.array([0x1b, 0xaf, 0x7a]) / 255, np.array([0xeb, 0x68, 0x34]) / 255
fig, axs = plt.subplots(1, 4, figsize=(12, 3.6))
summary = {}
for a, det in zip(axs, DETS):
    M, first = np.zeros((3, 2), int), [0, 0]
    for e in data:
        s_tr, s_te = e['scores'][det]
        al, st = s_te > s_tr.max(), e['state']
        for k in range(3):
            M[k, 1] += al[st == k].sum()
            M[k, 0] += (~al[st == k]).sum()
        it = np.where(st == 1)[0]
        if len(it):
            first[0] += 1
            first[1] += int(al[it[0]])
    F = M / M.sum(1, keepdims=True)
    summary[det] = (M, first)
    color = np.zeros((3, 2, 3))
    for k in range(3):
        for j in range(2):
            ok = (k == 0 and j == 0) or (k > 0 and j == 1)
            color[k, j] = 1 - F[k, j] * (1 - (GOOD if ok else BAD))
    a.imshow(color, aspect='auto')
    for k in range(3):
        for j in range(2):
            a.text(j, k, f'{F[k, j]:.0%}\n({M[k, j]})', ha='center', va='center', fontsize=9,
                   color='white' if F[k, j] > 0.6 else '#1a1a1a')
    a.set_xticks([0, 1])
    a.set_xticklabels(['No alarm', 'Alarm'])
    a.set_yticks(range(3))
    a.set_yticklabels(ROWS if det == 'IF' else [])
    a.set_title(det, loc='left', fontsize=10)
    for s in a.spines.values():
        s.set_visible(False)
    a.set_xticks([0.5], minor=True)
    a.set_yticks([0.5, 1.5], minor=True)
    a.grid(which='minor', color='white', lw=2)
    a.tick_params(which='minor', length=0)
fig.tight_layout()
fig.savefig(RESULTS / f'fig_confusion_matrix_{CONFIG}.png', dpi=300, facecolor='white')
for det, (M, first) in summary.items():
    t, n = M[1, 1] - first[1], M[1].sum() - first[0]
    print(f'{det}: normal {M[0, 1] / M[0].sum():.1%} | transient {M[1, 1] / M[1].sum():.1%} | '
          f'steady {M[2, 1] / M[2].sum():.1%} | first transient window {first[1]}/{first[0]} | '
          f'transient without the first window {t / n:.1%}')
