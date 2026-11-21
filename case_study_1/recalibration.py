"""Acknowledged recalibration and alarm grouping, on top of persistence over the conformal threshold.

  S0  persistence (3 of 5) over the conformal threshold, updated with the last W scores without alarm
  S1  S0 + grouping: an alarm that returns within M windows after clearing counts as the same event
  S2  S0 + recalibration: after N consecutive windows in alarm the condition is taken as a new
      operating point and the threshold is recomputed from the scores of those N windows
  S3  S2 + S1 (proposed), with N and M in {5, 10, 20} windows (15, 30 and 60 min), plus N = 3

Reads the cached scores of streaming.py. Output: recalibration.csv
Usage: python recalibration.py
"""
import pickle

import numpy as np
import pandas as pd

from protocol import RESULTS, WINDOW as S
from streaming import ALPHA, W, K_PERS, N_PERS, q

CONFIGS = [('S0', None, 0), ('S1', None, 10)] + [('S3', n, m) for n in (5, 10, 20) for m in (5, 10, 20)] + [('S2', 10, 0), ('S3', 3, 10)]


def persist(al):
    al = np.asarray(al, int)
    return np.array([al[max(0, i - N_PERS + 1):i + 1].sum() >= K_PERS for i in range(len(al))])


def alarms(s_cal, stream, n=None):
    """Persistence over the conformal threshold; with n, recalibrate after n windows in alarm."""
    hist, lim, run, out = list(s_cal), q(s_cal, ALPHA), 0, []
    for i, s in enumerate(stream):
        a = s > lim
        out.append(a)
        run = run + 1 if a else 0
        if not a:
            hist.append(s)
            if n is not None:
                lim = q(hist[-W:], ALPHA)
        if n is not None and run >= n:
            hist = list(stream[i - n + 1:i + 1])
            lim = q(hist, ALPHA)
            run = 0
    return persist(out)


def events(al, m=0):
    """Alarm events; an alarm returning within m windows is not a new event."""
    ev, gap = 0, 10 ** 9
    for a in np.asarray(al, bool):
        if a:
            if gap > m:
                ev += 1
            gap = 0
        else:
            gap += 1
    return ev


def main():
    rows = []
    for det in ['IF', 'LOF', 'VAE-Rec', 'VAE-Maha']:
        f_norm, f_fail = RESULTS / f'stream_scores_normal_{det}.pkl', RESULTS / f'stream_scores_failure_{det}.pkl'
        if not (f_norm.exists() and f_fail.exists()):
            continue
        normal, failure = pickle.load(open(f_norm, 'rb')), pickle.load(open(f_fail, 'rb'))
        for name, n, m in CONFIGS:
            r0 = []
            for d in normal:
                _, s_cal, st = d['scores']
                al = alarms(s_cal, st, n)
                r0.append(dict(well=d['well'], events_h=events(al, m) / (len(st) * S / 3600),
                               time_in_alarm=al.mean(), permanent=al.mean() > .9))
            r0 = pd.DataFrame(r0)
            r1 = []
            for d in failure:
                _, s_cal, st = d['scores']
                y = d['y']
                al = alarms(s_cal, st, n)
                idx = [d['starts'][i] for i in np.where(al & (y == 1))[0]]
                before = al[y == 0]
                r1.append(dict(detected=float(bool(idx)), delay=(idx[0] - d['onset']) / 60 if idx else np.nan,
                               det_windows=al[y == 1].mean() if (y == 1).any() else np.nan,
                               on_before=bool(before[-1]) if len(before) else False))
            r1 = pd.DataFrame(r1)
            clean = r1[~r1.on_before]
            rows.append(dict(det=det, config=name, N=n, M=m,
                             events_h=r0.groupby('well').events_h.mean().mean(),
                             time_in_alarm=r0.groupby('well').time_in_alarm.mean().mean(),
                             permanent=r0.permanent.mean(), detected=r1.detected.mean(), delay=r1.delay.median(),
                             det_windows=r1.det_windows.mean(), n_no_prior_alarm=len(clean),
                             detected_no_prior_alarm=clean.detected.mean()))
    r = pd.DataFrame(rows)
    r.to_csv(RESULTS / 'recalibration.csv', index=False)
    pd.set_option('display.width', 200)
    print(r.round(3).to_string())


if __name__ == '__main__':
    main()
