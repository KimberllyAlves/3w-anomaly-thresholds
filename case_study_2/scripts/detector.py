"""Threshold detector on the window scores.

Calibration (validation split): for each candidate score, the threshold maximizes Youden's J
(TPR + TNR - 1) averaged over classes 6 and 7 with equal weight, separating normal and transient
windows. The score with the highest mean J is kept. As an alternative, a threshold on the
reconstruction error of the sensor associated with each class is calibrated within that class.

Evaluation: binary, normal windows negative, transient and steady-state windows positive. Reports
per class the F1 of the positive class on validation and test and the false alarm rate on test.

Usage: python scripts/detector.py outputs_w32/uncertainty/val_uncertainty.csv outputs_w32/uncertainty/test_uncertainty.csv
"""
import sys

import numpy as np
import pandas as pd

SCORES = ['epistemic', 'latent_variability', 'total_uncertainty', 'recon_error']
CALIBRATION_CLASSES = [6, 7]
SENSOR_OF_CLASS = {6: 'P-MON-CKP', 7: 'P-TPT'}


def youden(normal, transient, t):
    return (transient >= t).mean() + (normal < t).mean() - 1


def calibrate(df, score, classes):
    groups = [(df[(df.class_dir == c) & (df.label == 0)][score].to_numpy(),
               df[(df.class_dir == c) & (df.label == 1)][score].to_numpy()) for c in classes]
    groups = [g for g in groups if len(g[0]) and len(g[1])]
    candidates = np.unique(np.concatenate([np.concatenate(g) for g in groups]))[1:]
    j = np.array([np.mean([youden(n, tr, t) for n, tr in groups]) for t in candidates])
    best = int(np.argmax(j))      # first maximum
    return candidates[best], j[best]


def f1_positive(df, t, score):
    pred, true = df[score] >= t, df.label > 0
    tp, fp, fn = (pred & true).sum(), (pred & ~true).sum(), (~pred & true).sum()
    return 2 * tp / (2 * tp + fp + fn) if tp else (0.0 if true.any() else np.nan)


val, test = pd.read_csv(sys.argv[1]), pd.read_csv(sys.argv[2])

results = {s: calibrate(val, s, CALIBRATION_CLASSES) for s in SCORES}
for s, (t, j) in results.items():
    print(f'{s:<20} threshold {t:.6f}  mean J {j:.3f}')
score = max(results, key=lambda s: results[s][1])
t = results[score][0]

print('\nper-sensor alternative (calibrated within each class):')
for c, sensor in SENSOR_OF_CLASS.items():
    ts, js = calibrate(val, f'recon_error_{sensor}', [c])
    _, ja = calibrate(val, 'recon_error', [c])
    print(f'  class {c}, {sensor}: threshold {ts:.6f}  J {js:.3f}  (aggregated error J {ja:.3f})')

print(f'\ndetector: {score} >= {t:.6f}')
print(f'{"class":>5} {"normal":>8} {"anomalous":>10} {"F1 val":>7} {"F1 test":>8} {"FA test":>8}')
for c in sorted(test.class_dir.unique()):
    v, te = val[val.class_dir == c], test[test.class_dir == c]
    normal = te[te.label == 0]
    fa = f'{(normal[score] >= t).mean():.0%}' if len(normal) else '-'
    print(f'{c:>5} {len(normal):>8} {(te.label > 0).sum():>10} {f1_positive(v, t, score):>7.2f} '
          f'{f1_positive(te, t, score):>8.2f} {fa:>8}')
