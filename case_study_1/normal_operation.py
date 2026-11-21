"""Static threshold rules over hours of normal operation (class 0).

Each instance trains on its first 60% windows and the rest is scored in sequence, so every
alarm is a false alarm. Means are taken per well first, then across wells.
Output: static_class0.csv (fraction of windows in alarm and alarms per hour)
Usage: python normal_operation.py
"""
import warnings

import numpy as np
import pandas as pd

import protocol as P
from detectors import RULES, score

warnings.filterwarnings('ignore')


def main():
    rows, skipped = [], 0
    for r in P.normal_runs(P.DATA):
        if len(r.train) < 5 or not r.test:
            skipped += 1
            continue
        Xtr, Xte, _ = P.prepare(r)
        if Xtr is None:
            skipped += 1
            continue
        hours = len(Xte) * P.WINDOW / 3600
        for det, (a, b) in score(Xtr, Xte).items():
            for rule, f in RULES.items():
                al = b > f(a)
                rows.append(dict(well=r.well, file=r.file, det=det, rule=rule, fa=al.mean(),
                                 alarms_h=al.sum() / hours, hours=hours))
    d = pd.DataFrame(rows)
    P.RESULTS.mkdir(parents=True, exist_ok=True)
    d.to_csv(P.RESULTS / 'static_class0.csv', index=False)
    print(f'{d.file.nunique()} instances, {d.well.nunique()} wells, {skipped} skipped, '
          f'{d.drop_duplicates("file").hours.sum():.0f} h of test')
    t = d.groupby(['rule', 'det', 'well'])[['fa', 'alarms_h']].mean().groupby(['rule', 'det']).mean()
    print(t.round(3).unstack('det').to_string())


if __name__ == '__main__':
    main()
