"""
SOMATA initialisation of the SSR oscillators, per participant and channel (resumable).

    python run_somata.py --pts sub-01 sub-02 --n-jobs 4
    python run_somata.py --pts sub-03 --channels 0-63 --n-jobs 16   # split a participant across machines

Every channel is fitted on the first 10 s of data (notebookfunctions.get_eeg: detrend + line-noise notches, see
extract_features.SOMATA_LINE_NOISE_HZ) with notebookfunctions.run_somata_single_channel, and written immediately to
{SSR_PATH_SOMATA}/{pt}/ch###.json, so an interrupted run only loses the channels in progress.
When all channels of a participant exist, the script writes

  {SSR_PATH_SOMATA}/{pt}_somata_results.npy   (format read by notebookfunctions.get_initParams)
  {SSR_PATH_SOMATA}/{pt}_somata_results.json  (same content, small; commit it so nobody has to rerun SOMATA)

and prints a summary, including any oscillator within +-2 Hz of a line-noise harmonic (50, 100, ..., 450 Hz).
SOMATA takes ~3 min per channel per core (10 s at 1024 Hz, up to 7 oscillators); the dataset has 1103 channels.
"""
import argparse
import json
import os
import time

import numpy as np
from joblib import Parallel, delayed

import notebookfunctions as nf
import extract_features as ef

SR = 1024
HARMONICS = np.arange(50, 500, 50)


def _to_jsonable(res):
    """Oscillator parameters as lists, everything else as Python scalars (JSON-serialisable)."""
    out = {}
    for k, v in res.items():
        a = np.asarray(v)
        if k in ('freqs', 'damping', 'sigma2'):
            out[k] = a.astype(float).ravel().tolist()
        elif a.size == 1:
            out[k] = a.ravel()[0].item()
        else:
            out[k] = a.ravel().tolist()
    return out


def _fit_and_save(ch, x, path):
    res = nf.run_somata_single_channel(ch, x, SR)
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(_to_jsonable(res), f)
    os.replace(tmp, path)
    return ch


def parse_channels(spec, n):
    if not spec:
        return list(range(n))
    chans = []
    for part in spec.split(','):
        a, _, b = part.partition('-')
        chans += list(range(int(a), (int(b) if b else int(a)) + 1))
    return [c for c in chans if c < n]


def assemble(pt, n_channels):
    ck = os.path.join(nf.path_somata, pt)
    files = [os.path.join(ck, f'ch{c:03d}.json') for c in range(n_channels)]
    if not all(os.path.exists(f) for f in files):
        return None
    chans = {}
    for c, f in enumerate(files):
        with open(f) as fh:
            chans[c] = json.load(fh)
    meta = dict(participant=pt, n_channels=n_channels, line_noise_notches_hz=list(ef.SOMATA_LINE_NOISE_HZ),
                segment_s=10, sr=SR, somata_osc_range=7)
    with open(os.path.join(nf.path_somata, f'{pt}_somata_results.json'), 'w') as fh:
        json.dump(dict(meta=meta, channels=chans), fh)
    as_np = {c: {k: (np.asarray(v) if isinstance(v, list) else v) for k, v in r.items()} for c, r in chans.items()}
    np.save(os.path.join(nf.path_somata, f'{pt}_somata_results.npy'), as_np)
    return as_np


def summarize(pt, res):
    freqs = np.concatenate([np.atleast_1d(r['freqs']) for r in res.values()])
    n_osc = [len(np.atleast_1d(r['freqs'])) for r in res.values()]
    no_hg = sum(not any(70 <= f <= 170 for f in np.atleast_1d(r['freqs'])) for r in res.values())
    near = [(c, float(f)) for c, r in res.items() for f in np.atleast_1d(r['freqs'])
            if np.min(np.abs(HARMONICS - f)) <= 2.0]
    print(f'{pt}: {len(res)} channels, {len(freqs)} oscillators ({min(n_osc)}-{max(n_osc)} per channel), '
          f'{no_hg} channels without own HG oscillator (template fallback)')
    print(f'{pt}: oscillators within +-2 Hz of a line-noise harmonic: {len(near)}'
          + (f' -> {near[:20]}' if near else ''))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--pts', nargs='+', default=nf.pts)
    ap.add_argument('--n-jobs', type=int, default=4)
    ap.add_argument('--channels', default='', help="subset, e.g. '0-63' or '0-9,20' (for splitting a participant)")
    args = ap.parse_args()
    print(f'SOMATA results -> {nf.path_somata}; line-noise notches {ef.SOMATA_LINE_NOISE_HZ} Hz')
    for pt in args.pts:
        eeg = nf.get_eeg(pt)
        ck = os.path.join(nf.path_somata, pt)
        os.makedirs(ck, exist_ok=True)
        todo = [c for c in parse_channels(args.channels, eeg.shape[1])
                if not os.path.exists(os.path.join(ck, f'ch{c:03d}.json'))]
        t0 = time.time()
        print(f'{pt}: {eeg.shape[1]} channels, {len(todo)} to fit', flush=True)
        if todo:
            Parallel(n_jobs=args.n_jobs, verbose=5)(
                delayed(_fit_and_save)(c, eeg[:, c], os.path.join(ck, f'ch{c:03d}.json')) for c in todo)
        print(f'{pt}: fitted {len(todo)} channels in {(time.time() - t0) / 60:.1f} min', flush=True)
        res = assemble(pt, eeg.shape[1])
        if res is not None:
            summarize(pt, res)


if __name__ == '__main__':
    main()
