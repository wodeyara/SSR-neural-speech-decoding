"""
Matched SSR-versus-bandpass comparison ladder.

Every SSR feature set gets a bandpass counterpart that differs in ONE respect only, so that
differences in decoding can be attributed to a single factor:

  factor                       SSR                       bandpass counterpart(s)
  ---------------------------  ------------------------  --------------------------------------
  amplitude only               SSR_amp                   HG, HG_LPamp, BPm_amp(_dc)
  + low-frequency phase        SSR_phase                 HG_LP, BPm_phase(_dc)
  + PAC                        SSR_pac                   HG_LP_PAC, BPm_pac(_dc)
  + phase-resolved PAC         SSR_pacc                  HG_LP_PACc
  filter delay (199 ms FIR)    -                         *_dc: same FIR, delay-compensated
  LF voltage (LMP) control     -                         HG_LP_LMP
  decoder look-ahead           any set, --causal-stack   any set, --causal-stack

HG / HG_LP*  : fixed bands (70-170 Hz envelope, <20 Hz low-pass), as in the paper.
BPm_*        : FIR bandpass + Hilbert at the SAME per-channel oscillator frequencies AND -3 dB
               bandwidths as the SSR filters (extract_features.extractBandpassMatched).
               This isolates the estimator (Kalman filter vs windowed bandpass) from the
               subject-specific choice of frequencies.

Usage
-----
    export SSR_PATH_BIDS=/path/to/SingleWordProductionDutch-iBIDS
    export SSR_PATH_SOMATA=/path/to/somata_results   # {pt}_somata_results.npy
    export SSR_PATH_FEATURES=./features SSR_PATH_RESULT=./results
    python run_comparisons.py --pts sub-01 sub-02 --sets HG HG_dc SSR_pac BPm_pac --n-comp 8
    python run_comparisons.py --sets all --n-comp nested            # recommended for the paper
    python run_comparisons.py --sets HG SSR_pac --n-comp 8 --causal-stack

Features are cached in SSR_PATH_FEATURES as {pt}{suffix}.npy (the published feature files are
reused if present). Results are written to SSR_PATH_RESULT with prefix
cmp_{set}[_causal]_{ncomp}; run stats_contrasts.py on them, e.g.

    python stats_contrasts.py --model HG=cmp_HG_n8 --model SSR=cmp_SSR_pac_n8 --contrast SSR:HG
"""
import argparse
import os

import numpy as np

import extract_features as ef
import notebookfunctions as nf
from reconstruction_minimal import reconstruct

ALL_PTS = [f'sub-{i:02d}' for i in range(1, 11)]

# name: (feature-file suffix, kind, kwargs, n_comp used in the paper or None)
FEATURE_SETS = {
    # --- published models (file names as used for the paper) ---
    'HG':          ('_final_bandpass_hg',    'hg',  dict(low_pass=False), 6),
    'HG_LP':       ('_final_bandpass_hg_lp', 'hg',  dict(low_pass=True), 9),
    'SSR_amp':     ('_onlyAMP_SSPEfeat',     'ssr', dict(include_phase=False, include_pac=False), 8),
    'SSR_phase':   ('_PHASE_SSPEfeat',       'ssr', dict(include_phase=True, include_pac=False), 8),
    'SSR_pac':     ('_PCA_SSPEfeat',         'ssr', dict(include_phase=True, include_pac=True), 8),
    # --- symmetric fixed-band bandpass ladder ---
    'HG_LPamp':    ('_bp_hg_lpamp',          'hg',  dict(low_pass=True, lp_phase=False), None),
    'HG_LP_PAC':   ('_bp_hg_lp_pac',         'hg',  dict(low_pass=True, include_pac=True), None),
    'HG_LP_PACc':  ('_bp_hg_lp_pacc',        'hg',  dict(low_pass=True, include_pac=True, pac_mode='complex'), None),
    'HG_LP_LMP':   ('_bp_hg_lp_lmp',         'hg',  dict(low_pass=True, include_lmp=True), None),
    'SSR_pacc':    ('_ssr_pacc',             'ssr', dict(include_phase=True, include_pac=True, pac_mode='complex'), None),
    # --- bandpass definitions from the literature (see extract_features.extractHG, `timing`) ---
    # Verwoert et al. 2022: zero-phase IIR + Hilbert on the whole recording (offline reference)
    'HG_vw':       ('_bp_hg_vw',             'hg',  dict(low_pass=False, timing='verwoert'), None),
    'HG_LP_vw':    ('_bp_hg_lp_vw',          'hg',  dict(low_pass=True, timing='verwoert'), None),
    'HG_LP_PAC_vw': ('_bp_hg_lp_pac_vw',     'hg',  dict(low_pass=True, include_pac=True, timing='verwoert'), None),
    # causal, buffer-wise: zero-phase IIR + Hilbert on the 400 ms ending at each window, mean of last 50 ms
    'HG_buf':      ('_bp_hg_buf',            'hg',  dict(low_pass=False, timing='buffered'), None),
    'HG_LP_buf':   ('_bp_hg_lp_buf',         'hg',  dict(low_pass=True, timing='buffered'), None),
    'HG_LP_PAC_buf': ('_bp_hg_lp_pac_buf',   'hg',  dict(low_pass=True, include_pac=True, timing='buffered'), None),
    # Angrick et al. 2021 real-time system: causal IIR with continuous state, log window power
    'HG_iir':      ('_bp_hg_iir',            'hg',  dict(low_pass=False, timing='streaming_iir'), None),
    # --- same FIR, delay-compensated (removes the 199 ms group delay) ---
    'HG_dc':       ('_bp_hg_dc',             'hg',  dict(low_pass=False, timing='delay_compensated'), None),
    'HG_LP_dc':    ('_bp_hg_lp_dc',          'hg',  dict(low_pass=True, timing='delay_compensated'), None),
    'HG_LP_PAC_dc': ('_bp_hg_lp_pac_dc',     'hg',  dict(low_pass=True, include_pac=True, timing='delay_compensated'), None),
    # --- bandpass matched to the SSR oscillators (frequency and bandwidth) ---
    'BPm_amp':     ('_bpm_amp',              'bpm', dict(include_phase=False, include_pac=False), None),
    'BPm_phase':   ('_bpm_phase',            'bpm', dict(include_phase=True, include_pac=False), None),
    'BPm_pac':     ('_bpm_pac',              'bpm', dict(include_phase=True, include_pac=True), None),
    'BPm_amp_dc':  ('_bpm_amp_dc',           'bpm', dict(include_phase=False, include_pac=False, timing='delay_compensated'), None),
    'BPm_phase_dc': ('_bpm_phase_dc',        'bpm', dict(include_phase=True, include_pac=False, timing='delay_compensated'), None),
    'BPm_pac_dc':  ('_bpm_pac_dc',           'bpm', dict(include_phase=True, include_pac=True, timing='delay_compensated'), None),
    # --- main analysis (agreed set-up): IIR estimators, 70-170 Hz HG, <20 Hz LP ---
    'HG_LPamp_buf': ('_bp_hg_lpamp_buf',     'hg',  dict(low_pass=True, lp_phase=False, timing='buffered'), None),
    'HG_LP_PACc_buf': ('_bp_hg_lp_pacc_buf', 'hg',  dict(low_pass=True, include_pac=True, pac_mode='complex', timing='buffered'), None),
    'BPm_amp_buf': ('_bpm_amp_buf',          'bpm', dict(include_phase=False, include_pac=False, timing='buffered'), None),
    'BPm_phase_buf': ('_bpm_phase_buf',      'bpm', dict(include_phase=True, include_pac=False, timing='buffered'), None),
    'BPm_pac_buf': ('_bpm_pac_buf',          'bpm', dict(include_phase=True, include_pac=True, timing='buffered'), None),
}

# Main analysis plan: (feature set, past-only context?). Causal baseline = buffered bandpass with past-only context;
# offline reference = Verwoert bandpass; SSR in both contexts; PLS components selected by nested CV.
MAIN_PLAN = [
    # causal (past-only context)
    ('HG_buf', True), ('HG_LPamp_buf', True), ('HG_LP_buf', True), ('HG_LP_PAC_buf', True),
    ('SSR_amp', True), ('SSR_phase', True), ('SSR_pac', True),
    ('BPm_amp_buf', True), ('BPm_phase_buf', True), ('BPm_pac_buf', True),
    # offline reference (Verwoert), past-only and +-200 ms
    ('HG_vw', True), ('HG_LP_vw', True), ('HG_LP_PAC_vw', True),
    ('HG_vw', False), ('HG_LP_vw', False), ('HG_LP_PAC_vw', False),
    # SSR with the original +-200 ms context
    ('SSR_amp', False), ('SSR_phase', False), ('SSR_pac', False),
    # phase-resolved PAC last: the largest feature sets (peak memory ~14 GB for SSR_pacc)
    ('HG_LP_PACc_buf', True), ('SSR_pacc', True),
]

# Contrasts that make the two frameworks symmetric (a - b)
MATCHED_CONTRASTS = [
    # published
    ('HG_LP', 'HG'), ('SSR_amp', 'HG'), ('SSR_phase', 'SSR_amp'), ('SSR_phase', 'HG_LP'),
    ('SSR_pac', 'SSR_phase'), ('SSR_pac', 'HG_LP'),
    # symmetric bandpass ladder
    ('HG_LPamp', 'HG'), ('HG_LP', 'HG_LPamp'), ('HG_LP_PAC', 'HG_LP'), ('SSR_pac', 'HG_LP_PAC'),
    ('SSR_amp', 'HG_LPamp'),
    # estimator, frequencies fixed (SSR vs matched bandpass)
    ('SSR_amp', 'BPm_amp'), ('SSR_phase', 'BPm_phase'), ('SSR_pac', 'BPm_pac'),
    ('SSR_amp', 'BPm_amp_dc'), ('SSR_phase', 'BPm_phase_dc'), ('SSR_pac', 'BPm_pac_dc'),
    # filter delay
    ('HG_dc', 'HG'), ('HG_LP_dc', 'HG_LP'),
    # bandpass definitions (run with and without --causal-stack)
    ('HG_vw', 'HG'), ('HG_buf', 'HG_vw'), ('HG_iir', 'HG_vw'), ('HG_LP_buf', 'HG_LP_vw'),
    ('SSR_amp', 'HG_buf'), ('SSR_phase', 'HG_LP_buf'), ('SSR_pac', 'HG_LP_PAC_buf'),
]


def feature_path(pt, name):
    return os.path.join(ef.path_output, f'{pt}{FEATURE_SETS[name][0]}.npy')


def ensure_features(pt, names, ssr_engine='steady_state', n_jobs=4, overwrite=False):
    os.makedirs(ef.path_output, exist_ok=True)
    if overwrite or not os.path.exists(os.path.join(ef.path_output, f'{pt}_spec.npy')):
        ef.extract_spectrogram(pt)
    todo = [n for n in names if overwrite or not os.path.exists(feature_path(pt, n))]
    if not todo:
        return
    eeg = ef.load_participant(pt)[0]
    params = None
    for n in todo:
        suffix, kind, kwargs, _ = FEATURE_SETS[n]
        print(f'[{pt}] extracting {n}')
        if kind == 'hg':
            feat = ef.extractHG(eeg, **kwargs)
        else:
            if params is None:
                params = nf.get_initParams(pt)
            if kind == 'ssr':
                feat = ef.extractSSPE(eeg, params, engine=ssr_engine, n_jobs=n_jobs, **kwargs)
            else:
                feat = ef.extractBandpassMatched(eeg, params, n_jobs=n_jobs, **kwargs)
        np.save(feature_path(pt, n), feat)


def result_prefix(name, n_comp, causal_stack):
    tag = 'nested' if isinstance(n_comp, (list, tuple)) else f'n{n_comp}'
    return f"cmp_{name}{'_causal' if causal_stack else ''}_{tag}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--pts', nargs='+', default=ALL_PTS)
    ap.add_argument('--sets', nargs='+', default=['all'], help=f"any of {list(FEATURE_SETS)} or 'all'")
    ap.add_argument('--plan', choices=['main'], default=None,
                    help="run the agreed main analysis (MAIN_PLAN: feature sets x context); overrides --sets/--causal-stack")
    ap.add_argument('--part', default='1/1', help="i/n: run only every n-th plan item starting at i (to split across machines)")
    ap.add_argument('--n-comp', default='nested',
                    help="'nested' (inner-CV selection, recommended), 'paper' (published values; 8 for new sets), "
                         "or an integer used for every set")
    ap.add_argument('--nested-grid', nargs='+', type=int, default=[4, 6, 8, 10, 12])
    ap.add_argument('--inner-folds', type=int, default=3)
    ap.add_argument('--causal-stack', action='store_true', help='stack only past context (no look-ahead)')
    ap.add_argument('--ssr-engine', default='steady_state', choices=['steady_state', 'kalman'])
    ap.add_argument('--n-jobs', type=int, default=4)
    ap.add_argument('--extract-only', action='store_true')
    ap.add_argument('--overwrite-features', action='store_true')
    args = ap.parse_args()

    if args.plan == 'main':
        items = MAIN_PLAN
    else:
        names = list(FEATURE_SETS) if args.sets == ['all'] else args.sets
        items = [(n, args.causal_stack) for n in names]
    i, n_parts = (int(x) for x in args.part.split('/'))
    items = items[i - 1::n_parts]
    names = list(dict.fromkeys(n for n, _ in items))
    for pt in args.pts:
        ensure_features(pt, names, args.ssr_engine, args.n_jobs, args.overwrite_features)
    if args.extract_only:
        return
    for n, causal in items:
        if args.n_comp == 'nested':
            n_comp = list(args.nested_grid)
        elif args.n_comp == 'paper':
            n_comp = FEATURE_SETS[n][3] or 8
        else:
            n_comp = int(args.n_comp)
        prefix = result_prefix(n, n_comp, causal)
        if os.path.exists(os.path.join(os.environ.get('SSR_PATH_RESULT', './results'), prefix + 'linearResults.npy')):
            print(f'=== {prefix} exists, skipping')
            continue
        print(f'=== reconstruct {n} ({"past-only" if causal else "+-200 ms"}) -> {prefix}', flush=True)
        reconstruct(args.pts, model='PLS', n_comp=n_comp, feat_suffix=FEATURE_SETS[n][0] + '.npy',
                    unstacked=True, saveAs=prefix, causal_stack=causal, inner_folds=args.inner_folds)
    print(f'PLAN_DONE ({len(items)} items, part {args.part})', flush=True)


if __name__ == '__main__':
    main()
