"""Real wells with instances in more than one split.

The split is by instance, so the same well can appear in training and in evaluation. Simulated and
hand-drawn instances have no well and are ignored.
"""
import argparse
from collections import defaultdict

import yaml

from vae3w.data import DATA, well_of
from vae3w.preprocessing import split_from_config

p = argparse.ArgumentParser()
p.add_argument('--config', default='configs/w32.yaml')
p.add_argument('--data', default=DATA)
a = p.parse_args()
split = split_from_config(yaml.safe_load(open(a.config)), a.data)

splits_of, classes_of = defaultdict(set), defaultdict(set)
for name, insts in [('train', split.train), ('val', split.val), ('test', split.test)]:
    for inst in insts:
        if (w := well_of(inst.path.name)):
            splits_of[w].add(name)
            classes_of[w].add(inst.class_dir)

shared = {w: s for w, s in splits_of.items() if len(s) > 1}
for w in sorted(shared):
    print(f'{w}  {sorted(shared[w])}  classes {sorted(classes_of[w])}')
print(f'\n{len(shared)} of {len(splits_of)} real wells ({len(shared) / len(splits_of):.1%}) appear in more than one split')
