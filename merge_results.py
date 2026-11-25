"""
Combine per-participant result files (e.g. results/sub-01/cmp_*linearResults.npy, each with a leading participant
axis of length 1) into group files (participants x ...) in participant order, for stats_contrasts.py.

    python merge_results.py --in results --out results/all
"""
import argparse
import glob
import os
import re

import numpy as np

ALL_PTS = [f'sub-{i:02d}' for i in range(1, 11)]
SUFFIXES = ('linearResults.npy', 'randomResults.npy', 'nComp.npy')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--in', dest='inp', default='results')
    ap.add_argument('--out', default='results/all')
    ap.add_argument('--pts', nargs='+', default=ALL_PTS)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    files = {}
    for pt in args.pts:
        for f in glob.glob(os.path.join(args.inp, pt, 'cmp_*')):
            name = os.path.basename(f)
            if name.endswith(SUFFIXES):
                files.setdefault(name, {})[pt] = f
    for name, per_pt in sorted(files.items()):
        missing = [p for p in args.pts if p not in per_pt]
        if missing:
            print(f'{name}: missing {missing} -> not merged')
            continue
        arr = np.concatenate([np.load(per_pt[p]) for p in args.pts], axis=0)
        np.save(os.path.join(args.out, name), arr)
        if name.endswith('linearResults.npy'):
            print(f'{name}: {arr.shape}, mean r = {arr.mean():.4f}')


if __name__ == '__main__':
    main()
