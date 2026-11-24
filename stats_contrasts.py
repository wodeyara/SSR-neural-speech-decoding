"""
Planned-contrast statistics for the speech-reconstruction results, as described in the paper:

* For every participant, fold and Mel bin the reconstruction correlation r is Fisher
  z-transformed and averaged over the 23 Mel bins -> one value per fold.
* Configurations are evaluated on identical folds, so each contrast uses paired fold-level
  differences, averaged within participant.
* Group level (participants are the unit of analysis, n = 10): one-sample t-test of the
  participant-mean differences (= paired t-test), Benjamini-Hochberg FDR over the contrasts.
* Within participant: corrected resampled t-test (Nadeau & Bengio, 2003) over the k = 10 folds,
  variance inflated by (1/k + n_test/n_train); BH-FDR over participants for every contrast.

Usage
-----
    python stats_contrasts.py                       # the six contrasts of the paper
    python stats_contrasts.py --results ./results \
        --model HG=res_final_bandpass_hg --model SSR=res_PCA_SSPEfeat --contrast SSR:HG

Each --model maps a short name to the prefix of a {prefix}linearResults.npy file written by
reconstruction_minimal.reconstruct (shape participants x folds x mel bins).
"""
import argparse
import os

import numpy as np
from scipy import stats
from statsmodels.stats.multitest import multipletests

PAPER_MODELS = {
    'HG': 'res_final_bandpass_hg',
    'HG_LP': 'res_final_bandpass_hg_lp',
    'SSR_amp': 'res_onlyAMP_SSPEfeat',
    'SSR_phase': 'res_PHASE_SSPEfeat',
    'SSR_pac': 'res_PCA_SSPEfeat',
}
PAPER_CONTRASTS = [('HG_LP', 'HG'), ('SSR_amp', 'HG'), ('SSR_phase', 'SSR_amp'),
                   ('SSR_phase', 'HG_LP'), ('SSR_pac', 'SSR_phase'), ('SSR_pac', 'HG_LP')]


def _m(name, past):
    """Result prefix written by run_comparisons.py --plan main (nested n_comp)."""
    return f"cmp_{name}{'_causal' if past else ''}_nested"


# Main analysis (run_comparisons.py --plan main). Keys: 'SET@past' (past-only context) or 'SET@sym' (+-200 ms).
# Contrasts are grouped into families; BH-FDR is applied within each family.
MAIN_FAMILIES = {
    'A. causal pipeline (past-only context): SSR vs buffered bandpass': [
        ('HG_LP_buf@past', 'HG_buf@past'), ('SSR_amp@past', 'HG_buf@past'), ('SSR_amp@past', 'HG_LPamp_buf@past'),
        ('SSR_phase@past', 'SSR_amp@past'), ('SSR_phase@past', 'HG_LP_buf@past'), ('SSR_pac@past', 'SSR_phase@past'),
        ('SSR_pac@past', 'HG_LP_PAC_buf@past'), ('HG_LP_PAC_buf@past', 'HG_LP_buf@past')],
    'B. estimator at matched frequencies (past-only)': [
        ('SSR_amp@past', 'BPm_amp_buf@past'), ('SSR_phase@past', 'BPm_phase_buf@past'), ('SSR_pac@past', 'BPm_pac_buf@past')],
    'C. causal SSR vs offline Verwoert bandpass (past-only) and cost of causal bandpass': [
        ('SSR_amp@past', 'HG_vw@past'), ('SSR_phase@past', 'HG_LP_vw@past'), ('SSR_pac@past', 'HG_LP_PAC_vw@past'),
        ('HG_buf@past', 'HG_vw@past'), ('HG_LP_buf@past', 'HG_LP_vw@past')],
    'D. temporal context (past-only minus +-200 ms)': [
        ('HG_vw@past', 'HG_vw@sym'), ('HG_LP_vw@past', 'HG_LP_vw@sym'), ('SSR_amp@past', 'SSR_amp@sym'),
        ('SSR_phase@past', 'SSR_phase@sym'), ('SSR_pac@past', 'SSR_pac@sym')],
    'E. phase-resolved PAC': [
        ('SSR_pacc@past', 'SSR_pac@past'), ('HG_LP_PACc_buf@past', 'HG_LP_PAC_buf@past')],
}


def main_preset_models():
    keys = {k for fam in MAIN_FAMILIES.values() for c in fam for k in c}
    return {k: _m(k.split('@')[0], k.endswith('@past')) for k in keys}


def fold_z(all_res):
    """(participants, folds, bins) r -> (participants, folds) bin-averaged Fisher z."""
    r = np.clip(np.asarray(all_res, float), -0.999999, 0.999999)
    return np.arctanh(r).mean(axis=2)


def corrected_resampled_ttest(d, test_train_ratio=1 / 9):
    """Nadeau & Bengio (2003) corrected resampled t-test on k fold-wise differences d."""
    d = np.asarray(d, float)
    k = len(d)
    var = np.var(d, ddof=1)
    if var == 0:
        return np.nan, 1.0
    t = d.mean() / np.sqrt((1.0 / k + test_train_ratio) * var)
    p = 2 * stats.t.sf(np.abs(t), df=k - 1)
    return t, p


def run_contrasts(results, contrasts, alpha=0.05, test_train_ratio=1 / 9):
    """
    results : dict name -> (participants, folds, bins) array of r
    contrasts : list of (a, b) tuples, testing a - b
    Returns a list of dicts (one per contrast).
    """
    rows = []
    for a, b in contrasts:
        za, zb = fold_z(results[a]), fold_z(results[b])
        d = za - zb                                     # (participants, folds)
        d_part = d.mean(axis=1)
        t, p = stats.ttest_1samp(d_part, 0.0)
        dr = np.mean(results[a], axis=(1, 2)) - np.mean(results[b], axis=(1, 2))
        within = [corrected_resampled_ttest(d[i], test_train_ratio) for i in range(d.shape[0])]
        p_within = np.array([w[1] for w in within])
        p_within_fdr = multipletests(np.nan_to_num(p_within, nan=1.0), method='fdr_bh')[1]
        rows.append(dict(contrast=f'{a} - {b}', a=a, b=b,
                         mean_r_a=float(np.mean(results[a])), mean_r_b=float(np.mean(results[b])),
                         delta_r=float(dr.mean()), delta_z=float(d_part.mean()),
                         t=float(t), df=len(d_part) - 1, p=float(p),
                         n_better=int(np.sum(d_part > 0)), n_participants=len(d_part),
                         n_sig_better=int(np.sum((p_within_fdr < alpha) & (d_part > 0))),
                         n_sig_worse=int(np.sum((p_within_fdr < alpha) & (d_part < 0))),
                         delta_r_participants=dr.tolist()))
    p_fdr = multipletests([r['p'] for r in rows], method='fdr_bh')[1]
    for r, pf in zip(rows, p_fdr):
        r['p_fdr'] = float(pf)
    return rows


def format_table(rows):
    head = (f"{'contrast':34s} {'mean r (a)':>10s} {'mean r (b)':>10s} {'dr':>8s} "
            f"{'t(df)':>10s} {'p':>8s} {'p_FDR':>8s}  better (sig+/sig-)")
    lines = [head, '-' * len(head)]
    for r in rows:
        lines.append(f"{r['contrast']:34s} {r['mean_r_a']:10.4f} {r['mean_r_b']:10.4f} {r['delta_r']:+8.4f} "
                     f"{r['t']:+6.2f}({r['df']})  {r['p']:8.4f} {r['p_fdr']:8.4f}  "
                     f"{r['n_better']}/{r['n_participants']} ({r['n_sig_better']}/{r['n_sig_worse']})")
    return '\n'.join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--results', default=os.environ.get('SSR_PATH_RESULT', './results'))
    ap.add_argument('--model', action='append', default=[], help='NAME=prefix')
    ap.add_argument('--contrast', action='append', default=[], help='A:B (tests A - B)')
    ap.add_argument('--csv', default=None, help='optional path to write the table as CSV')
    ap.add_argument('--preset', choices=['paper', 'main'], default='paper',
                    help="'paper': the six contrasts of the draft; 'main': contrast families of the main analysis")
    args = ap.parse_args()

    if args.preset == 'main' and not args.model:
        models = main_preset_models()
        all_rows = []
        for fam, contrasts in MAIN_FAMILIES.items():
            needed = sorted({m for c in contrasts for m in c})
            avail = {m: os.path.join(args.results, f'{models[m]}linearResults.npy') for m in needed}
            missing = [m for m, f in avail.items() if not os.path.exists(f)]
            if missing:
                print(f'\n{fam}: missing results for {missing}, skipped')
                continue
            rows = run_contrasts({m: np.load(f) for m, f in avail.items()}, contrasts)
            print(f'\n{fam}  (BH-FDR within family)')
            print(format_table(rows))
            for r in rows:
                r['family'] = fam
            all_rows += rows
        if args.csv and all_rows:
            import csv
            with open(args.csv, 'w', newline='') as f:
                w = csv.DictWriter(f, fieldnames=[k for k in all_rows[0] if k != 'delta_r_participants'])
                w.writeheader()
                for r in all_rows:
                    w.writerow({k: v for k, v in r.items() if k != 'delta_r_participants'})
        return

    models = dict(m.split('=', 1) for m in args.model) if args.model else PAPER_MODELS
    contrasts = [tuple(c.split(':', 1)) for c in args.contrast] if args.contrast else PAPER_CONTRASTS
    needed = sorted({m for c in contrasts for m in c})
    results = {m: np.load(os.path.join(args.results, f'{models[m]}linearResults.npy')) for m in needed}
    rows = run_contrasts(results, contrasts)
    print(format_table(rows))
    if args.csv:
        import csv
        with open(args.csv, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=[k for k in rows[0] if k != 'delta_r_participants'])
            w.writeheader()
            for r in rows:
                w.writerow({k: v for k, v in r.items() if k != 'delta_r_participants'})


if __name__ == '__main__':
    main()
