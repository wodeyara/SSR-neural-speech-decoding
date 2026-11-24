"""
Main-analysis figure (after merge_results.py):

    python plot_main.py --results results/all --out paper/images/main_results

A  mean reconstruction correlation (past-only context) per feature level and framework, mean +- 95% within-participant
   CI (Cousineau-Morey: participant means removed across all plotted configurations, Morey's correction);
B  paired differences (r) for the key contrasts, one dot per participant, mean +- 95% CI;
C  effect of temporal context: past-only minus +-200 ms, same layout as B.
Colours: categorical slots 1-4 of the default palette (validated; markers carry identity as well).
"""
import argparse
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

INK, MUTED, GRID = '#1f1f1e', '#6b6a64', '#e4e3df'
FRAMEWORKS = [  # (label, colour, marker, {level: (SET, context)})
    ('SSR (causal)', '#2a78d6', 'o',
     {'Amplitude': ('SSR_amp', 'past'), '+ LF phase': ('SSR_phase', 'past'), '+ PAC': ('SSR_pac', 'past')}),
    ('Bandpass, buffered (causal)', '#eb6834', 's',
     {'HG': ('HG_buf', 'past'), 'Amplitude': ('HG_LPamp_buf', 'past'), '+ LF phase': ('HG_LP_buf', 'past'),
      '+ PAC': ('HG_LP_PAC_buf', 'past')}),
    ('Matched bandpass (causal)', '#1baf7a', '^',
     {'Amplitude': ('BPm_amp_buf', 'past'), '+ LF phase': ('BPm_phase_buf', 'past'), '+ PAC': ('BPm_pac_buf', 'past')}),
    ('Bandpass, offline (Verwoert et al.)', '#eda100', 'D',
     {'HG': ('HG_vw', 'past'), '+ LF phase': ('HG_LP_vw', 'past'), '+ PAC': ('HG_LP_PAC_vw', 'past')}),
]
LEVELS = ['HG', 'Amplitude', '+ LF phase', '+ PAC']
CONTRASTS = [  # (label, a, b): a - b
    ('SSR − buffered BP, amplitude', ('SSR_amp', 'past'), ('HG_LPamp_buf', 'past')),
    ('SSR − buffered BP, + LF phase', ('SSR_phase', 'past'), ('HG_LP_buf', 'past')),
    ('SSR − buffered BP, + PAC', ('SSR_pac', 'past'), ('HG_LP_PAC_buf', 'past')),
    ('SSR − matched BP, + PAC', ('SSR_pac', 'past'), ('BPm_pac_buf', 'past')),
    ('SSR − offline BP, + PAC', ('SSR_pac', 'past'), ('HG_LP_PAC_vw', 'past')),
    ('buffered − offline BP, + LF phase', ('HG_LP_buf', 'past'), ('HG_LP_vw', 'past')),
    ('SSR: + LF phase − amplitude', ('SSR_phase', 'past'), ('SSR_amp', 'past')),
    ('buffered BP: + LF phase − HG', ('HG_LP_buf', 'past'), ('HG_buf', 'past')),
]
CONTEXT = [
    ('Offline BP, HG', 'HG_vw'), ('Offline BP, + LF phase', 'HG_LP_vw'), ('Offline BP, + PAC', 'HG_LP_PAC_vw'),
    ('SSR, amplitude', 'SSR_amp'), ('SSR, + LF phase', 'SSR_phase'), ('SSR, + PAC', 'SSR_pac'),
]


def load(results, name, ctx):
    f = os.path.join(results, f"cmp_{name}{'_causal' if ctx == 'past' else ''}_nestedlinearResults.npy")
    if not os.path.exists(f):
        return None
    a = np.load(f)
    return a.reshape(a.shape[0], -1).mean(axis=1)   # participant means over folds and Mel bins


def ci95(x):
    return stats.t.ppf(0.975, len(x) - 1) * np.std(x, ddof=1) / np.sqrt(len(x))


def style(ax):
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK, labelsize=8)
    ax.yaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def diff_panel(ax, items, results, title, xlabel):
    """Horizontal dot plot: one row per contrast, dots = participants, black = mean +- 95% CI."""
    rng = np.random.default_rng(0)
    for i, (label, a, b) in enumerate(items):
        xa, xb = load(results, *a), load(results, *b)
        if xa is None or xb is None:
            continue
        d = xa - xb
        ax.scatter(d, i + rng.uniform(-0.15, 0.15, len(d)), s=14, color='#9a9992', zorder=2, lw=0)
        ax.errorbar(d.mean(), i, xerr=ci95(d), fmt='o', ms=6, color=INK, capsize=0, lw=2, zorder=3)
    ax.axvline(0, color=MUTED, lw=1)
    ax.set_yticks(range(len(items)))
    ax.set_yticklabels([it[0] for it in items], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel(xlabel, fontsize=9, color=INK)
    ax.set_title(title, loc='left', fontsize=10, color=INK)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK, labelsize=8)
    ax.xaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def within_ci(table):
    """Cousineau-Morey within-participant 95% CI per column of a participants x conditions table."""
    k = table.shape[1]
    norm = table - table.mean(axis=1, keepdims=True) + table.mean()
    sd = norm.std(axis=0, ddof=1) * np.sqrt(k / (k - 1))
    return stats.t.ppf(0.975, table.shape[0] - 1) * sd / np.sqrt(table.shape[0])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--results', default='results/all')
    ap.add_argument('--out', default='paper/images/main_results')
    args = ap.parse_args()

    fig = plt.figure(figsize=(12, 7.2))
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 1.15], height_ratios=[1.25, 1], hspace=0.42, wspace=0.55)
    ax = fig.add_subplot(gs[:, 0])
    # all plotted configurations as one participants x conditions table (for the within-participant CI)
    cells = [(fi, lev, load(args.results, *sets[lev])) for fi, (_, _, _, sets) in enumerate(FRAMEWORKS)
             for lev in LEVELS if lev in sets]
    cells = [c for c in cells if c[2] is not None]
    table = np.column_stack([c[2] for c in cells])
    wci = dict(((c[0], c[1]), w) for c, w in zip(cells, within_ci(table)))
    offs = np.linspace(-0.21, 0.21, len(FRAMEWORKS))
    for fi, ((label, col, mk, sets), o) in enumerate(zip(FRAMEWORKS, offs)):
        xs, ms, cs = [], [], []
        for j, lev in enumerate(LEVELS):
            if (fi, lev) in wci:
                x = load(args.results, *sets[lev])
                xs.append(j + o); ms.append(x.mean()); cs.append(wci[(fi, lev)])
        ax.errorbar(xs, ms, yerr=cs, fmt=mk, color=col, ms=7, lw=1.5, capsize=0, label=label,
                    mec='white', mew=1.0, zorder=3)
        for k in range(len(xs) - 1):        # connect only adjacent feature levels (no line across a missing level)
            if round(xs[k + 1] - xs[k]) == 1:
                ax.plot(xs[k:k + 2], ms[k:k + 2], color=col, lw=1.5, zorder=2)
    ax.set_xticks(range(len(LEVELS)))
    ax.set_xticklabels(['High gamma\nonly', 'Amplitude', '+ LF phase', '+ PAC'], fontsize=8)
    ax.set_ylabel('Mean reconstruction r (past-only context)', fontsize=9, color=INK)
    ax.set_title('A  Reconstruction accuracy', loc='left', fontsize=10, color=INK)
    style(ax)
    ax.legend(fontsize=7.5, frameon=False, loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2)

    axb = fig.add_subplot(gs[0, 1])
    diff_panel(axb, CONTRASTS, args.results, 'B  Paired differences (past-only context)', 'Δr')
    axc = fig.add_subplot(gs[1, 1])
    diff_panel(axc, [(lab, (s, 'past'), (s, 'sym')) for lab, s in CONTEXT], args.results,
               'C  Past-only (−400…0 ms) minus ±200 ms context', 'Δr')
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    for ext in ('png', 'pdf'):
        fig.savefig(f'{args.out}.{ext}', dpi=200, bbox_inches='tight')
    print(f'wrote {args.out}.png/.pdf')


if __name__ == '__main__':
    main()
