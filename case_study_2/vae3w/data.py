"""Reading 3W v2.0.0: one parquet file per instance, in subfolders 0..9 named after the class.

Column 'class': 0 normal, N steady state of event N, 100 + N transient towards event N.
Any column other than 'class' and 'state' is treated as a sensor.
"""
from __future__ import annotations

import argparse
import os
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

DATA = os.environ.get('DATA_3W', 'data/3W/dataset')
LABEL_COLUMNS = {'class', 'state'}
NORMAL, TRANSIENT, STEADY = 0, 1, 2


@dataclass
class Instance:
    path: Path
    class_dir: int
    origin: str       # real, simulated or drawn


def origin_of(name: str) -> str:
    name = name.upper()
    if 'SIMULATED' in name:
        return 'simulated'
    if 'DRAWN' in name:
        return 'drawn'
    return 'real'


def well_of(name: str) -> str | None:
    m = re.match(r'^(WELL-\d+)_', name)
    return m.group(1) if m else None


def list_instances(root=DATA, classes=None) -> list[Instance]:
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f'3W dataset folder not found: {root} (set DATA_3W or --data)')
    dirs = sorted(p for p in root.iterdir() if p.is_dir() and p.name.isdigit())
    if classes is not None:
        dirs = [p for p in dirs if p.name in {str(c) for c in classes}]
    return [Instance(f, int(d.name), origin_of(f.name)) for d in dirs for f in sorted(d.glob('*.parquet'))]


def read(inst: Instance) -> pd.DataFrame:
    return pd.read_parquet(inst.path, engine='pyarrow')


def sensor_columns(df):
    return [c for c in df.columns if c not in LABEL_COLUMNS]


def state_of(label) -> int:
    label = int(label)
    if label == 0:
        return NORMAL
    return TRANSIENT if label >= 100 else STEADY


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='Number of instances per class and origin.')
    ap.add_argument('--data', default=DATA)
    inst = list_instances(ap.parse_args().data)
    t = pd.DataFrame([(i.class_dir, i.origin) for i in inst], columns=['class', 'origin'])
    print(pd.crosstab(t['class'], t['origin'], margins=True))
