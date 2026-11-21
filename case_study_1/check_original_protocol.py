"""Replica of demo_2_benchmark_anomaly_detection.ipynb (Vargas et al., 2019) on 3W v2.0.0, used
to validate protocol.py. Unavoidable differences: parquet instead of CSV, 7 variables (T-JUS-CKGL
is absent in v2) and the 6 features computed with numpy (the same as tsfresh MinimalFCParameters
without sum_values and length). Prints the mean F1 micro and AUC-ROC of the Isolation Forest with
the threshold at the maximum training score.
Usage: python check_original_protocol.py
"""
import glob
import os
from math import ceil

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from protocol import DATA, VARS, CLASSES, features

sample_size, min_normal, split_range, max_spp = 180, 1200, 0.6, 15
max_nan_percent, std_vars_min = 0.1, 0.01


def extract_samples(ols, X, class_code):
    """Line-by-line translation of extract_samples(). X: raw values after fillna(0)."""
    set_ols = set(int(o) for o in ols if not np.isnan(o))
    tr, te, yte = [], [], []
    f_idx = ols.index(0)
    l_idx = len(ols) - 1 - ols[::-1].index(0)
    max_samples_normal = l_idx - f_idx + 1 - sample_size
    if max_samples_normal > 0:
        num_normal = min(max_spp, max_samples_normal)
        num_train = int(split_range * num_normal)
        num_test = num_normal - num_train
    else:
        num_normal = num_train = num_test = 0
    tcode = class_code + 100
    if tcode in set_ols:
        f = ols.index(tcode)
        f = f - (sample_size - 1) if f - (sample_size - 1) > 0 else 0
        l = len(ols) - 1 - ols[::-1].index(tcode)
        max_tr = l - f + 1 - sample_size
    else:
        max_tr = 0
    if class_code in set_ols:
        f = ols.index(class_code)
        f = f - (sample_size - 1) if f - (sample_size - 1) > 0 else 0
        l = len(ols) - 1 - ols[::-1].index(class_code)
        l = l + (sample_size - 1) if l + (sample_size - 1) < len(ols) - 1 else len(ols) - 1
        max_ir = l - f + 1 - sample_size
    else:
        max_ir = 0
    n_t = ceil(num_test / 2)
    n_ir = num_test - n_t
    if max_tr >= n_t and max_ir < n_ir:
        n_ir = max_ir
        n_t = min(num_test - n_ir, max_tr)
    elif max_tr < n_t and max_ir >= n_ir:
        n_t = max_tr
        n_ir = min(num_test - n_t, max_ir)
    elif max_tr < n_t and max_ir < n_ir:
        n_t, n_ir = max_tr, max_ir
    if num_normal > 0:
        step_max = 1 if num_normal == max_samples_normal else (max_samples_normal - 1) // (max_spp - 1)
        step = min(sample_size, step_max)
        for idx in range(num_normal):
            a = l_idx - sample_size + 1 - (num_normal - 1 - idx) * step
            (tr if idx < num_train else te).append(X[a:a + sample_size])
            if idx >= num_train:
                yte.append(0)
    if n_t > 0:
        step = 1 if n_t == max_tr else (max_tr - 1) // (max_spp - 1)
        f = ols.index(tcode)
        f = f - (sample_size - 1) if f - (sample_size - 1) > 0 else 0
        for idx in range(n_t):
            te.append(X[f + idx * step: f + idx * step + sample_size])
            yte.append(1)
    if n_ir > 0:
        step = min(sample_size, 1 if n_ir == max_ir else (max_ir - 1) // (max_spp - 1))
        f = ols.index(class_code)
        f = f - (sample_size - 1) if f - (sample_size - 1) > 0 else 0
        for idx in range(n_ir):
            te.append(X[f + idx * step: f + idx * step + sample_size])
            yte.append(1)
    return tr, te, np.array(yte)


f1, auc, ignored = [], [], 0
for c in CLASSES:
    for p in sorted(glob.glob(os.path.join(DATA, str(c), 'WELL*.parquet'))):
        df = pd.read_parquet(p)
        for v in VARS:
            if v not in df.columns:
                df[v] = np.nan
        ols = df['class'].astype('float64').tolist()
        if (df['class'] == 0).sum() < min_normal:
            ignored += 1
            continue
        X = df[VARS].fillna(0).to_numpy(dtype=np.float64)        # fillna(0) on raw data, as in the notebook
        tr, te, y = extract_samples(ols, X, c)
        Tr = np.concatenate(tr)
        good = (np.isnan(Tr).mean(0) <= max_nan_percent) & (np.nanstd(Tr, 0) > std_vars_min)
        sc = StandardScaler().fit(Tr[:, good])
        Xtr = np.array([features(sc.transform(w[:, good])) for w in tr])
        Xte = np.array([features(sc.transform(w[:, good])) for w in te])
        clf = IsolationForest(random_state=1).fit(Xtr)
        s_tr, s_te = -clf.score_samples(Xtr), -clf.score_samples(Xte)
        f1.append(((s_te > s_tr.max()).astype(int) == y).mean())
        auc.append(roc_auc_score(y, s_te))
print(f'{len(f1)} instances used, {ignored} ignored')
print(f'IF, max rule: F1 micro = {np.mean(f1):.3f}, AUC-ROC = {np.mean(auc):.3f}')
