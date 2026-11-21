"""Static threshold rules on Benchmark 2 (classes 1, 2, 5, 6, 7, 8) and on class 9.

Outputs (in RESULTS):
  static_failures.csv   F1 micro, false alarm, detection and AUC-ROC per instance, detector and rule
  static_scores.pkl     training and test scores per instance (used by the figures)
  learning_curve.csv    same metrics training on the k most recent windows (--learning-curve)

F1 micro over the 12 test windows equals the accuracy, as in the benchmark.
Usage: python static_thresholds.py [--learning-curve]
"""
import argparse
import pickle
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import protocol as P
from detectors import RULES, score

warnings.filterwarnings('ignore')
KS = [9, 18, 36, 72, 144]


def metrics(s_tr, s_te, y):
    auc = roc_auc_score(y, s_te)
    out = []
    for rule, f in RULES.items():
        yp = (s_te > f(s_tr)).astype(int)
        out.append(dict(rule=rule, f1=(yp == y).mean(), fa=yp[y == 0].mean(), dr=yp[y == 1].mean(), auc=auc))
    return out


def main(learning_curve=False):
    rows, saved, curve = [], [], []
    for config in ['original', 'extended']:
        for classes, name in [(P.CLASSES, 'benchmark'), ([9], 'class 9')]:
            for r in P.failure_runs(P.DATA, classes, config):
                Xtr, Xte, good = P.prepare(r)
                if Xtr is None or len(Xtr) < 2:
                    continue
                sc = score(Xtr, Xte)
                for det, (s_tr, s_te) in sc.items():
                    for m in metrics(s_tr, s_te, r.y):
                        rows.append(dict(config=config, set=name, det=det, file=r.file, well=r.well, cls=r.cls,
                                         n_train=len(Xtr), **m))
                saved.append(dict(config=config, set=name, file=r.file, well=r.well, cls=r.cls,
                                  y=r.y, state=r.state, scores=sc))
                if learning_curve and name == 'benchmark' and config == 'extended' and len(r.train) >= max(KS):
                    full = r.train
                    for k in KS:
                        r.train = full[-k:]
                        Xk, Xk_te, _ = P.prepare(r)
                        if Xk is None:
                            continue
                        for det, (a, b) in score(Xk, Xk_te).items():
                            for m in metrics(a, b, r.y):
                                curve.append(dict(file=r.file, well=r.well, k=k, det=det, **m))
                    r.train = full
            print(config, name, 'dropped:', P.failure_runs.dropped, flush=True)

    P.RESULTS.mkdir(parents=True, exist_ok=True)
    d = pd.DataFrame(rows)
    d.to_csv(P.RESULTS / 'static_failures.csv', index=False)
    pickle.dump(saved, open(P.RESULTS / 'static_scores.pkl', 'wb'))
    for name in ['benchmark', 'class 9']:
        x = d[d.set == name]
        print(f'\n=== {name}: instances per config {x.groupby("config").file.nunique().to_dict()} ===')
        print('F1 micro (false alarm):')
        print(x.groupby(['config', 'rule', 'det'])[['f1', 'fa', 'dr']].mean().round(3).unstack('det').to_string())
        print('AUC-ROC:', x[x.rule == 'max'].groupby(['config', 'det']).auc.mean().round(3).to_dict())
    if learning_curve:
        c = pd.DataFrame(curve)
        c.to_csv(P.RESULTS / 'learning_curve.csv', index=False)
        print(f'\nLearning curve ({c.file.nunique()} instances, {c.well.nunique()} wells), max rule:')
        print(c[c.rule == 'max'].groupby(['det', 'k'])[['f1', 'fa', 'dr', 'auc']].mean().round(3).to_string())


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--learning-curve', action='store_true')
    main(ap.parse_args().learning_curve)
