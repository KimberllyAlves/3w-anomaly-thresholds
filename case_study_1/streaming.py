"""Threshold strategies in streaming mode.

Each instance becomes a stream of non-overlapping 180 s windows scored in time order. The
threshold of each window uses only scores already seen and no labels. The initial normal period
(extended training of protocol.py) is split into 80% to train the detector and the last 20% as a
calibration set (out-of-sample scores).

Strategies
  static_max      maximum of the training scores (Vargas et al., 2019)
  static_m2sd     mean + 2 sd of the training scores (Schmöckel et al., 2026)
  conformal       (1 - alpha) quantile of the calibration scores (Vovk et al., 2005)
  moving          (1 - alpha) quantile of the last W scores that raised no alarm
  aci             adaptive conformal inference (Gibbs and Candès, 2021)
  persist_*       k-out-of-n rule over a base strategy
  cusum           one-sided CUSUM of the score standardized by the calibration set (Page, 1954)
  dspot           DSPOT (Siffer et al., 2017) on the deviation from the mean of the last d normal scores

Metrics
  class 0         fraction of windows in alarm, alarms per hour, alarm events per hour
  failures        false alarm before the failure, instance detection, fraction of failure windows
                  in alarm and delay of the first alarm in a failure window

Scores are cached in RESULTS/stream_scores_<mode>_<detector>.pkl; delete a file to recompute.
Usage: python streaming.py [--detectors IF LOF VAE-Rec VAE-Maha]
"""
import argparse
import glob
import os
import pickle
import warnings

import numpy as np
import pandas as pd

import protocol as P
from detectors import DETECTORS, score

warnings.filterwarnings('ignore')
S = P.WINDOW
ALPHA, W, GAMMA, CAL_FRAC = 0.01, 20, 0.005, 0.2
K_PERS, N_PERS = 3, 5                  # 3 out of the last 5 windows (15 min)
CUSUM_K, CUSUM_H = 0.5, 5.0            # slack and decision limit, in standard deviations
DSPOT_Q, DSPOT_D, DSPOT_LEVEL = 1e-3, 10, 0.90


def q(x, alpha):
    """Conformal quantile with finite-sample correction. With fewer than 1/alpha - 1 scores the
    quantile would be infinite (never alarms); the maximum is used instead."""
    x = np.sort(np.asarray(x))
    n = len(x)
    k = int(np.ceil((n + 1) * (1 - alpha))) - 1
    return x[min(max(k, 0), n - 1)]


def persistence(al, k=None, n=None):
    k = K_PERS if k is None else k
    n = N_PERS if n is None else n
    al = np.asarray(al, int)
    out = np.zeros_like(al, bool)
    for i in range(len(al)):
        out[i] = al[max(0, i - n + 1):i + 1].sum() >= k
    return out


def _gpd_mom(y):
    """Generalized Pareto fitted by the method of moments (stable with few peaks)."""
    m, v = y.mean(), y.var()
    if v <= 0:
        return 0.0, max(m, 1e-12)
    g = 0.5 * (1 - m * m / v)
    s = 0.5 * m * (m * m / v + 1)
    return float(np.clip(g, -0.5, 0.5)), float(max(s, 1e-12))


def _zq(t, peaks, n, risk=None):
    risk = DSPOT_Q if risk is None else risk
    if len(peaks) < 3:
        return t + (max(peaks) if len(peaks) else 0.0) * 1.5
    g, s = _gpd_mom(np.asarray(peaks))
    r = risk * n / len(peaks)
    return t + (s / g) * (r ** (-g) - 1) if abs(g) > 1e-6 else t - s * np.log(r)


def dspot(init, stream, d=None, risk=None, level=None):
    d = DSPOT_D if d is None else d
    level = DSPOT_LEVEL if level is None else level
    init = np.asarray(init, float)
    if len(init) <= d + 3:
        return stream > init.max()
    res = np.array([init[i] - init[i - d:i].mean() for i in range(d, len(init))])
    t = np.quantile(res, level)
    peaks = list(res[res > t] - t)
    n = len(res)
    zq = _zq(t, peaks, n, risk)
    recent, al = list(init[-d:]), []
    for s in stream:
        r = s - np.mean(recent)
        if r > zq:
            al.append(True)
            continue
        al.append(False)
        if r > t:
            peaks.append(r - t)
            zq = _zq(t, peaks, n + 1, risk)
        n += 1
        recent = recent[1:] + [s]
    return np.array(al)


def cusum(cal, stream):
    mu, sd = np.mean(cal), np.std(cal)
    sd = sd if sd > 0 else 1e-12
    c, al = 0.0, []
    for s in stream:
        c = max(0.0, c + (s - mu) / sd - CUSUM_K)
        if c > CUSUM_H:
            al.append(True)
            c = 0.0
        else:
            al.append(False)
    return np.array(al)


def strategies(s_tr, s_cal, stream):
    """{strategy: boolean alarm vector over the stream}."""
    out = {'static_max': stream > s_tr.max(),
           'static_m2sd': stream > s_tr.mean() + 2 * s_tr.std(),
           'conformal': stream > q(s_cal, ALPHA)}
    hist, al = list(s_cal[-W:]), []
    for s in stream:
        a = s > q(hist[-W:], ALPHA) if len(hist) >= 5 else False
        al.append(a)
        if not a:
            hist.append(s)
    out['moving'] = np.array(al)
    alpha_t, ref, al = ALPHA, list(s_cal), []
    for s in stream:
        a = s > q(ref, min(max(alpha_t, 1e-4), 0.5))
        al.append(a)
        alpha_t += GAMMA * (ALPHA - float(a))
        if not a:
            ref.append(s)
    out['aci'] = np.array(al)
    out['persist_max'] = persistence(out['static_max'])
    out['persist_conformal'] = persistence(out['conformal'])
    out['cusum'] = cusum(s_cal, stream)
    out['dspot'] = dspot(np.concatenate([s_tr, s_cal]), stream)
    out['dspot_persist'] = persistence(out['dspot'])
    return out


def events(al):
    al = np.asarray(al, int)
    return int(((al[1:] == 1) & (al[:-1] == 0)).sum() + (al[:1] == 1).sum())


def _score_stream(train_windows, stream_windows, det):
    n_cal = max(3, int(round(CAL_FRAC * len(train_windows))))
    if len(train_windows) - n_cal < 5:
        return None
    r = P.Run('', '', 0)
    r.train = train_windows[:-n_cal]
    r.test = train_windows[-n_cal:] + stream_windows
    Xtr, Xrest, _ = P.prepare(r)
    if Xtr is None:
        return None
    s_tr, s_rest = score(Xtr, Xrest, [det])[det]
    return s_tr, s_rest[:n_cal], s_rest[n_cal:]


def scores_normal(det):
    out = []
    for r in P.normal_runs(P.DATA):
        if not r.test:
            continue
        res = _score_stream(r.train, r.test, det)
        if res is not None:
            out.append(dict(well=r.well, file=r.file, scores=res))
    return out


def scores_failure(det):
    out = []
    for c in P.CLASSES:
        for p in sorted(glob.glob(os.path.join(P.DATA, str(c), 'WELL*.parquet'))):
            labels, X = P.load(p)
            if sum(o == 0 for o in labels) < P.MIN_NORMAL or not any(o in (c, c + 100) for o in labels):
                continue
            _, _, _, block, f_idx = P.sample_windows(labels, c)
            tr = P.extended_train(f_idx, block)
            starts = list(range(block, len(labels) - S + 1, S))   # from the test block to the end
            lab, y = np.array(labels, dtype=float), []
            for a in starts:
                seg = lab[a:a + S]
                seg = seg[~np.isnan(seg)]
                y.append(np.nan if len(seg) == 0 else float((seg != 0).mean() > 0.5))
            onset = next(i for i, o in enumerate(labels) if o in (c, c + 100))
            res = _score_stream([X[a:a + S] for a in tr], [X[a:a + S] for a in starts], det)
            if res is not None:
                out.append(dict(cls=c, file=os.path.basename(p), well=P.well_of(p), scores=res,
                                y=np.array(y), starts=starts, onset=onset))
    return out


def cached_scores(mode, det):
    path = P.RESULTS / f'stream_scores_{mode}_{det}.pkl'
    if path.exists():
        return pickle.load(open(path, 'rb'))
    data = scores_normal(det) if mode == 'normal' else scores_failure(det)
    P.RESULTS.mkdir(parents=True, exist_ok=True)
    pickle.dump(data, open(path, 'wb'))
    return data


def evaluate(det):
    c0 = []
    for d in cached_scores('normal', det):
        s_tr, s_cal, st = d['scores']
        hours = len(st) * S / 3600
        for name, al in strategies(s_tr, s_cal, st).items():
            c0.append(dict(det=det, well=d['well'], file=d['file'], strategy=name, fa=al.mean(),
                           alarms_h=al.sum() / hours, events_h=events(al) / hours))
    fl = []
    for d in cached_scores('failure', det):
        s_tr, s_cal, st = d['scores']
        y = d['y']
        normal, failure = y == 0, y == 1
        for name, al in strategies(s_tr, s_cal, st).items():
            idx = [d['starts'][i] for i in np.where(al & failure)[0]]
            fl.append(dict(det=det, cls=d['cls'], well=d['well'], file=d['file'], strategy=name,
                           fa=al[normal].mean() if normal.any() else np.nan, detected=float(bool(idx)),
                           det_windows=al[failure].mean() if failure.any() else np.nan,
                           delay_min=(idx[0] - d['onset']) / 60 if idx else np.nan))
    return pd.DataFrame(c0), pd.DataFrame(fl)


def summary(c0, fl):
    c0 = c0.assign(permanent=c0.fa > 0.9)
    a = c0.groupby(['det', 'strategy', 'well'])[['events_h', 'fa']].mean().groupby(['det', 'strategy']).mean()
    a['permanent'] = c0.groupby(['det', 'strategy']).permanent.mean()
    b = fl.groupby(['det', 'strategy']).agg(detected=('detected', 'mean'), det_windows=('det_windows', 'mean'),
                                            delay_min=('delay_min', 'median'), fa_before=('fa', 'mean'))
    return a.join(b).reset_index()


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--detectors', nargs='+', default=DETECTORS)
    dets = ap.parse_args().detectors
    res = [evaluate(det) for det in dets]
    c0 = pd.concat([r[0] for r in res])
    fl = pd.concat([r[1] for r in res])
    c0.to_csv(P.RESULTS / 'stream_class0.csv', index=False)
    fl.to_csv(P.RESULTS / 'stream_failures.csv', index=False)
    t = summary(c0, fl)
    t.to_csv(P.RESULTS / 'stream_summary.csv', index=False)
    pd.set_option('display.width', 200)
    print(f'class 0: {c0.file.nunique()} instances | failures: {fl.file.nunique()} instances')
    print(t.set_index(['strategy', 'det']).sort_index().round(3).to_string())
