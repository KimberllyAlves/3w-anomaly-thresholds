"""Data protocol for case study 1.

Follows the code of Benchmark 2 in Vargas et al. (2019)
(demo_2_benchmark_anomaly_detection.ipynb, github.com/ricardovvargas/3w_dataset).
Every change to that protocol is marked with "Extension" and a short reason.

Two training configurations share the same test set:
  'original' -> 9 training windows, as in the benchmark
  'extended' -> all normal operation before the test block, non-overlapping windows
"""
from __future__ import annotations

import glob
import os
import re
from dataclasses import dataclass, field
from math import ceil
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

DATA = os.environ.get('DATA_3W', 'data/3W/dataset')   # folder with subfolders 0..9
RESULTS = Path(os.environ.get('RESULTS', 'results'))

# benchmark parameters
WINDOW = 3 * 60          # 180 observations at 1 Hz
MIN_NORMAL = 20 * 60     # minimum normal period before the event
SPLIT = 0.6              # train/test split of the normal windows
MAX_SPP = 15             # maximum windows per period
MAX_NAN = 0.10           # variable filter: fraction of missing values
STD_MIN = 0.01           # variable filter: minimum raw standard deviation
RANDOM_STATE = 1
CLASSES = [1, 2, 5, 6, 7, 8]
# Extension: T-JUS-CKGL is not in dataset.ini of 3W v2.0.0.
VARS = ['P-PDG', 'P-TPT', 'T-TPT', 'P-MON-CKP', 'T-JUS-CKP', 'P-JUS-CKGL', 'QGL']
# Extension: sentinel values (e.g. ~1e42 in P-PDG of WELL-00006) become NaN. They are neither
# missing nor constant, so they pass both filters of the benchmark and break the scaling.
SENTINEL = 1e10


@dataclass
class Run:
    file: str
    well: str
    cls: int
    train: list = field(default_factory=list)   # raw windows (180 x n_vars)
    test: list = field(default_factory=list)
    y: np.ndarray | None = None                 # 0 normal, 1 anomalous
    state: np.ndarray | None = None             # 0 normal, 1 transient, 2 steady state
    test_start: list = field(default_factory=list)


def well_of(path):
    m = re.match(r'(WELL-\d+)', os.path.basename(path))
    return m.group(1) if m else None


def load(path):
    """Labels as a list of floats (NaN = unlabeled) and the raw n x 7 matrix."""
    df = pd.read_parquet(path)
    for v in VARS:
        if v not in df.columns:
            df[v] = np.nan
    X = df[VARS].to_numpy(dtype=np.float64, copy=True)
    X[np.abs(X) > SENTINEL] = np.nan
    return df['class'].astype('float64').tolist(), X


def _last(labels, v):
    return len(labels) - 1 - labels[::-1].index(v)


def sample_windows(labels, cls):
    """extract_samples() of the benchmark, returning window start indices.
    Also returns the start of the normal test block, used by the extended configuration."""
    s = WINDOW
    present = set(int(o) for o in labels if not np.isnan(o))
    f_idx, l_idx = labels.index(0), _last(labels, 0)
    max_n = l_idx - f_idx + 1 - s
    if max_n > 0:
        n_normal = min(MAX_SPP, max_n)
        n_train = int(SPLIT * n_normal)
        n_test = n_normal - n_train
    else:
        n_normal = n_train = n_test = 0
    tcode = cls + 100
    if tcode in present:
        f = max(labels.index(tcode) - (s - 1), 0)
        max_t = _last(labels, tcode) - f + 1 - s
    else:
        max_t = 0
    if cls in present:
        f = max(labels.index(cls) - (s - 1), 0)
        l = min(_last(labels, cls) + (s - 1), len(labels) - 1)
        max_r = l - f + 1 - s
    else:
        max_r = 0
    n_t = ceil(n_test / 2)
    n_r = n_test - n_t
    if max_t >= n_t and max_r < n_r:
        n_r = max_r
        n_t = min(n_test - n_r, max_t)
    elif max_t < n_t and max_r >= n_r:
        n_t = max_t
        n_r = min(n_test - n_t, max_r)
    elif max_t < n_t and max_r < n_r:
        n_t, n_r = max_t, max_r

    tr, te, st = [], [], []
    if n_normal > 0:
        step = 1 if n_normal == max_n else min(s, (max_n - 1) // (MAX_SPP - 1))
        for i in range(n_normal):
            a = l_idx - s + 1 - (n_normal - 1 - i) * step
            if i < n_train:
                tr.append(a)
            else:
                te.append(a)
                st.append(0)
    if n_t > 0:
        step = 1 if n_t == max_t else (max_t - 1) // (MAX_SPP - 1)
        f = max(labels.index(tcode) - (s - 1), 0)
        te += [f + i * step for i in range(n_t)]
        st += [1] * n_t
    if n_r > 0:
        step = min(s, 1 if n_r == max_r else (max_r - 1) // (MAX_SPP - 1))
        f = max(labels.index(cls) - (s - 1), 0)
        te += [f + i * step for i in range(n_r)]
        st += [2] * n_r
    test_block = te[0] if n_test > 0 else None
    return tr, te, np.array(st), test_block, f_idx


def extended_train(f_idx, test_block):
    """Extension: all normal operation before the normal test block, in non-overlapping 180 s
    windows aligned backwards from the test block. With 9 windows a per-instance autoencoder
    does not beat a dummy classifier (Fernandes Junior, 2022), and overlapping windows make
    training scores artificially low. When the benchmark step is 180, the 9 'original'
    windows are the last 9 of this list."""
    out, a = [], test_block - WINDOW
    while a >= f_idx:
        out.append(a)
        a -= WINDOW
    return out[::-1]


def failure_runs(base=DATA, classes=CLASSES, config='original'):
    """One Run per real instance with a failure that meets the benchmark rules."""
    dropped = {'short_normal': 0, 'no_failure_label': 0, 'no_test': 0}
    for c in classes:
        for p in sorted(glob.glob(os.path.join(base, str(c), 'WELL*.parquet'))):
            labels, X = load(p)
            if sum(o == 0 for o in labels) < MIN_NORMAL:
                dropped['short_normal'] += 1
                continue
            # Extension: at least one observation labelled as failure (c or 100+c). In v2.0.0,
            # 43 real instances of class 9 have no failure label and the test would be all normal.
            if not any(o in (c, c + 100) for o in labels):
                dropped['no_failure_label'] += 1
                continue
            tr, te, st, block, f_idx = sample_windows(labels, c)
            if not te or (st > 0).sum() == 0:
                dropped['no_test'] += 1
                continue
            if config == 'extended':
                tr = extended_train(f_idx, block)
            r = Run(os.path.basename(p), well_of(p), c)
            r.train = [X[a:a + WINDOW] for a in tr]
            r.test = [X[a:a + WINDOW] for a in te]
            r.state = st
            r.y = (st > 0).astype(int)
            r.test_start = te
            yield r
    failure_runs.dropped = dropped


def normal_runs(base=DATA, train_frac=SPLIT):
    """Extension: class 0 instances, to measure alarms over hours of normal operation.
    Non-overlapping 180 s windows; the first 60% train and the rest is the test stream."""
    for p in sorted(glob.glob(os.path.join(base, '0', 'WELL*.parquet'))):
        labels, X = load(p)
        idx0 = [i for i, o in enumerate(labels) if o == 0]
        if len(idx0) < MIN_NORMAL:
            continue
        f, l = idx0[0], idx0[-1]
        starts = list(range(f, l - WINDOW + 2, WINDOW))
        k = int(train_frac * len(starts))
        r = Run(os.path.basename(p), well_of(p), 0)
        r.train = [X[a:a + WINDOW] for a in starts[:k]]
        r.test = [X[a:a + WINDOW] for a in starts[k:]]
        r.y = np.zeros(len(r.test), int)
        r.state = r.y.copy()
        r.test_start = starts[k:]
        yield r


def prepare(run):
    """Variable filter and scaling fitted on the training windows, then 6 features per variable.
    Deviation: the benchmark code fills NaN with 0 in the raw data before everything, which
    disables the 10% missing filter and inserts 0 Pa pressures. We follow the thesis text
    (Vargas, 2019, sec. 5.3): filter, scale, and only then replace remaining NaN by 0."""
    Tr = np.concatenate(run.train)
    good = (np.isnan(Tr).mean(0) <= MAX_NAN) & (np.nan_to_num(np.nanstd(Tr, 0), nan=0.0) > STD_MIN)
    if not good.any():
        return None, None, good
    sc = StandardScaler().fit(Tr[:, good])
    norm = lambda w: np.nan_to_num(sc.transform(w[:, good]), nan=0.0)
    Xtr = np.array([features(norm(w)) for w in run.train])
    Xte = np.array([features(norm(w)) for w in run.test])
    return Xtr, Xte, good


def features(w):
    """tsfresh MinimalFCParameters without sum_values and length, as in the benchmark:
    median, mean, standard deviation, variance, maximum and minimum of each variable."""
    return np.concatenate([np.median(w, 0), w.mean(0), w.std(0), w.var(0), w.max(0), w.min(0)])
