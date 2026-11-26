"""
Inputs for the paper figures that need more than the committed result files (see paper_figures.py):

* SSR amplitude features of every participant (oscillator-level figures), and
* the reconstructed spectrograms of the main analysis, recomputed with the number of PLS components that the nested
  cross-validation chose in every outer fold (results/sub-XX/cmp_<set>_causal_nestednComp.npy). This reproduces the
  predictions of the main analysis exactly without repeating the inner selection; the script checks that the
  reconstruction correlations match the committed linearResults.

    export SSR_PATH_BIDS=... SSR_PATH_SOMATA=$PWD/somata_results SSR_PATH_FEATURES=...
    python figure_inputs.py --pts sub-01 ... --out <dir for predicted spectrograms>

Predicted spectrograms (~5 MB each) are written to --out as {pt}_fig_<set>_causal_predicted_spec.npy.
"""
import argparse
import os

import numpy as np

import run_comparisons as rc
from reconstruction_minimal import reconstruct

DEFAULT_SETS = ['SSR_pac', 'HG_LP_PAC_buf']


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--pts', nargs='+', default=rc.ALL_PTS)
    ap.add_argument('--sets', nargs='+', default=DEFAULT_SETS)
    ap.add_argument('--results', default='results', help='folder with sub-XX/cmp_* files of the main analysis')
    ap.add_argument('--out', required=True)
    ap.add_argument('--n-jobs', type=int, default=4)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    os.environ['SSR_PATH_RESULT'] = args.out
    for pt in args.pts:
        rc.ensure_features(pt, ['SSR_amp'] + args.sets, 'steady_state', args.n_jobs)
        for name in args.sets:
            prefix = f'fig_{name}_causal'
            if os.path.exists(os.path.join(args.out, f'{pt}_{prefix}_predicted_spec.npy')):
                print(f'[{pt}] {prefix} exists, skipping')
                continue
            ref = os.path.join(args.results, pt, f'cmp_{name}_causal_nested')
            k = np.load(ref + 'nComp.npy')
            res, _ = reconstruct([pt], model='PLS', n_comp=int(np.median(k)), fold_n_comp=k,
                                 feat_suffix=rc.FEATURE_SETS[name][0] + '.npy', unstacked=True, saveAs=prefix,
                                 causal_stack=True)
            diff = np.max(np.abs(res - np.load(ref + 'linearResults.npy')))
            print(f'[{pt}] {name}: mean r {res.mean():.6f}, max |r - committed| = {diff:.2e}', flush=True)


if __name__ == '__main__':
    main()
