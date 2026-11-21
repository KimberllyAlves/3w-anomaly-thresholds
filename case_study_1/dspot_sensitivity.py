"""Sensitivity of DSPOT with persistence to the risk q, the depth d of the moving mean and the
k-out-of-n persistence rule, on the VAE-Rec scores. Reads the cached scores of streaming.py.
Output: dspot_sensitivity.csv
Usage: python dspot_sensitivity.py
"""
import numpy as np
import pandas as pd

from protocol import RESULTS, WINDOW as S
from streaming import cached_scores, dspot, events, persistence

DET = 'VAE-Rec'
normal, failure = cached_scores('normal', DET), cached_scores('failure', DET)


def evaluate(fn):
    ev = pd.DataFrame([(d['well'], events(fn(*d['scores'])) / (len(d['scores'][2]) * S / 3600)) for d in normal],
                      columns=['well', 'e']).groupby('well').e.mean().mean()
    dw, delay, fa = [], [], []
    for d in failure:
        y = d['y']
        al = fn(*d['scores'])
        if (y == 1).any():
            dw.append(al[y == 1].mean())
        if (y == 0).any():
            fa.append(al[y == 0].mean())
        idx = [d['starts'][i] for i in np.where(al & (y == 1))[0]]
        delay.append((idx[0] - d['onset']) / 60 if idx else np.nan)
    return ev, np.mean(dw), np.nanmedian(delay), np.mean(fa), np.mean(~np.isnan(delay))


rows = []
for risk in [1e-2, 1e-3, 1e-4]:
    for depth in [5, 10, 20]:
        for k, n in [(1, 1), (2, 3), (3, 5), (5, 8)]:
            fn = lambda s_tr, s_cal, st: persistence(dspot(np.concatenate([s_tr, s_cal]), st, d=depth, risk=risk), k=k, n=n)
            ev, dw, delay, fa, det = evaluate(fn)
            rows.append(dict(q=risk, depth=depth, persistence=f'{k}-of-{n}', events_h=ev, det_windows=dw,
                             delay_min=delay, fa_before=fa, detected=det))
r = pd.DataFrame(rows)
r.to_csv(RESULTS / 'dspot_sensitivity.csv', index=False)
print(r.round(3).to_string(index=False))
