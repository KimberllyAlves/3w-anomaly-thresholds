"""Greedy forward selection of sensors.

Coverage is counted per instance after cleaning (does the sensor have at least one valid value?).
Sensors with coverage below 5% are discarded. At each step the sensor added is the one that most
increases the number of classes keeping at least 10 instances with every selected sensor present.
This counts instances, not windows, so the chosen set is confirmed afterwards with preprocessing.
P-PDG is left out from the start (65% of its samples are frozen).
"""
import argparse

import yaml

from vae3w.data import DATA, LABEL_COLUMNS
from vae3w.preprocessing import read_clean, split_from_config

MIN_COVERAGE = 0.05
MIN_INSTANCES = 10
MAX_SENSORS = 10
EXCLUDED = {'P-PDG'}

p = argparse.ArgumentParser()
p.add_argument('--config', default='configs/w32.yaml')
p.add_argument('--data', default=DATA)
a = p.parse_args()
cfg = yaml.safe_load(open(a.config))
split = split_from_config(cfg, a.data)
instances = split.train + split.val + split.test

present = {}
for i, inst in enumerate(instances):
    df = read_clean(inst)
    for c in df.columns:
        if c in LABEL_COLUMNS or c in EXCLUDED:
            continue
        present.setdefault(c, set())
        if df[c].notna().any():
            present[c].add(i)

cov = {s: len(v) / len(instances) for s, v in present.items()}
for s, v in sorted(cov.items(), key=lambda kv: -kv[1]):
    print(f'{s:<16}{v:7.1%}')
candidates = sorted((s for s in cov if cov[s] >= MIN_COVERAGE), key=lambda s: -cov[s])

by_class = {}
for i, inst in enumerate(instances):
    by_class.setdefault(inst.class_dir, set()).add(i)
classes = sorted(by_class)


def surviving(sensors, c):
    ok = set(by_class[c])
    for s in sensors:
        ok &= present[s]
    return len(ok)


def usable(sensors):
    return sum(surviving(sensors, c) >= MIN_INSTANCES for c in classes)


chosen = []
while candidates and len(chosen) < MAX_SENSORS:
    best = max(candidates, key=lambda s: usable(chosen + [s]))   # first maximum, as in a strict > loop
    chosen.append(best)
    candidates.remove(best)
    if len(chosen) >= 2:
        counts = ' '.join(f'{c}:{surviving(chosen, c)}' for c in classes)
        print(f'\n{len(chosen)} sensors {chosen}\n  usable classes {usable(chosen)}/{len(classes)}  instances {counts}')
