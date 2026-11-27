"""
Regenerates the result-dependent figures of the paper from the main analysis (replaces the corresponding cells of
report_figures.ipynb, which used the earlier FIR / un-notched version of the analysis).

    export SSR_PATH_BIDS=... SSR_PATH_SOMATA=$PWD/somata_results SSR_PATH_FEATURES=...
    python paper_figures.py --figs all --preds <figure_inputs.py --out> [--out paper/images]

Figures (--figs): model_fits, amp_example, bp_corr, feat_audio, pls_weights, templates, ncomp, stability,
spectrograms (zoomed sub-01 excerpt + one full-length figure per participant, and the error-track correlations).
Numbers quoted in the paper are printed and written to <out>/figure_numbers.txt.
Inputs: results/all and results/sub-XX (main analysis), results/filters (compare_filters.py), somata_results,
the raw data (model_fits, amp_example), SSR_amp/HG_vw features (amp_example, feat_audio, pls_weights) and the
predicted spectrograms written by figure_inputs.py (spectrograms).
"""
import argparse
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scipy.signal
from scipy import stats
from sklearn.cross_decomposition import PLSRegression
from sklearn.neighbors import KernelDensity

import extract_features as ef
import notebookfunctions as nf
import run_comparisons as rc

PTS = rc.ALL_PTS
INK, MUTED, GRID = '#1f1f1e', '#6b6a64', '#e4e3df'
C_SSR, C_BUF, C_MATCH, C_OFF = '#2a78d6', '#eb6834', '#1baf7a', '#eda100'
DROP, HALF, FS_FRAME = 1000, 20, 100      # 10 s initialisation (frames), stacking trim, frames per second
NUMBERS = []


def note(s):
    print(s, flush=True)
    NUMBERS.append(s)


def style(ax, grid='y'):
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK, labelsize=9)
    if grid:
        getattr(ax, f'{grid}axis').grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def grid_fig(sharey=True, figsize=(16, 6.4)):
    fig, axes = plt.subplots(2, 5, figsize=figsize, sharex=True, sharey=sharey)
    return fig, axes.ravel()


def save(fig, out, name, dpi=200):
    fig.savefig(os.path.join(out, name), dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f'wrote {name}')


def osc_freqs(pt):
    return np.concatenate([np.asarray(p['freqs'], float) for p in nf.get_initParams(pt)])


def template_mask(pt):
    """Per oscillator (initParams order): True if it is a template high-gamma oscillator."""
    som, _ = nf.drop_line_noise_oscillators(nf.load_somata_results(pt))
    mask = []
    for ch, p in enumerate(nf.get_initParams(pt)):
        own = np.asarray(som[ch]['freqs'], float)
        for f in np.asarray(p['freqs'], float):
            mask.append(70 <= f <= 170 and not np.any(np.isclose(own, f)))
    return np.array(mask)


def load_feat(pt, name):
    return np.load(rc.feature_path(pt, name))


def aligned_spec_and_feat(pt, feat, lead_frames=0):
    """Spectrogram rows i (window i+20) with the neural feature window lead_frames earlier, first 10 s dropped."""
    spec = np.load(os.path.join(ef.path_output, f'{pt}_spec.npy'))[DROP:]
    f = feat[DROP + HALF - lead_frames:]
    n = min(len(spec), len(f))
    return spec[:n], f[:n]


# ------------------------------------------------------------------------------------------------------------------
def fig_model_fits(out):
    fig, axes = grid_fig(sharey=False, figsize=(17, 6.8))
    bins = np.arange(0, 201, 5)
    for ax, pt in zip(axes, PTS):
        raw = ef.load_participant(pt)[0].astype(float)
        raw = ef.notch_line_noise(scipy.signal.detrend(raw, axis=0), ef.SOMATA_LINE_NOISE_HZ)
        f, P = scipy.signal.welch(raw, fs=ef.sr, nperseg=2048, axis=0)
        del raw
        m = f <= 200
        mean, sem = P[m].mean(1), P[m].std(1) / np.sqrt(P.shape[1])
        ax.semilogy(f[m], mean, color=INK, lw=1.4)
        ax.fill_between(f[m], mean - sem, mean + sem, color=INK, alpha=0.15, lw=0)
        ax.set_xlim(0, 200)
        ax.set_title(pt, fontsize=11, color=INK)
        style(ax, grid=None)
        ax2 = ax.twinx()
        fr, tm = osc_freqs(pt), template_mask(pt)
        ax2.hist([fr[~tm], fr[tm]], bins=bins, stacked=True, color=[C_SSR, '#86b6ef'],
                 label=['SOMATA', 'HG template'], rwidth=0.9)
        ax2.spines['top'].set_visible(False)
        ax2.tick_params(labelsize=9, colors=MUTED, labelcolor=INK)
        ax.set_zorder(ax2.get_zorder() + 1)
        ax.patch.set_visible(False)
    axes[0].figure.text(0.5, 0.01, 'Frequency (Hz)', ha='center', fontsize=12)
    axes[0].figure.text(0.005, 0.5, 'Power spectral density (log)', va='center', rotation='vertical', fontsize=12)
    axes[0].figure.text(0.995, 0.5, 'Number of oscillators', va='center', rotation='vertical', fontsize=12)
    h, l = ax2.get_legend_handles_labels()
    fig.legend(h, l, loc='upper center', ncol=2, frameon=False, fontsize=10, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=[0.015, 0.03, 0.985, 0.97])
    save(fig, out, 'SSR_model_fits_all_subjects.png')


def fig_amp_example(out, pt='sub-01', ch=None, seconds=5):
    params = nf.get_initParams(pt)
    amp, hg = load_feat(pt, 'SSR_amp'), load_feat(pt, 'HG_vw')
    spec = np.load(os.path.join(ef.path_output, f'{pt}_spec.npy'))
    env = spec.mean(1)
    if ch is None:      # channel whose (offline) HG envelope tracks the speech energy best
        sp, h = aligned_spec_and_feat(pt, hg)
        ch = int(np.argmax([np.corrcoef(sp.mean(1), h[:, c])[0, 1] for c in range(h.shape[1])]))
    col = sum(len(p['freqs']) for p in params[:ch])
    fr = np.asarray(params[ch]['freqs'], float)
    w = seconds * FS_FRAME
    # 5 s window with most speech energy variation (spectrogram row i = window i + 20), after the 10 s initialisation
    starts = np.arange(DROP, len(env) - w, FS_FRAME)
    s0 = starts[np.argmax([env[s:s + w].std() for s in starts])] + HALF
    sl = slice(s0, s0 + w)
    t = (np.arange(s0, s0 + w) * ef.frameshift) + ef.winL / 2
    z = lambda x: (x - x.mean()) / x.std()
    raw = ef.load_participant(pt)[0][:, ch].astype(float)
    a = np.abs(scipy.signal.hilbert(scipy.signal.detrend(raw)))
    st, sp = ef._window_bounds(len(raw))
    rawenv = ef._window_mean(a[:, None], st, sp)[:, 0]
    order = np.argsort(fr)
    fig, axes = plt.subplots(2 + len(fr), 1, figsize=(10, 1.45 * (2 + len(fr))), sharex=True)
    axes[0].plot(t, z(env[s0 - HALF:s0 - HALF + w]), color=INK, lw=1)
    axes[0].set_title('Speech (mean log-mel energy)', loc='left', fontsize=10)
    axes[1].plot(t, z(rawenv[sl]), color=INK, lw=1)
    axes[1].set_title(f'Broadband iEEG envelope, {pt} channel {ch}', loc='left', fontsize=10)
    hgz = z(hg[sl, ch])
    for ax, j in zip(axes[2:], order):
        y = z(amp[sl, col + j])
        r = np.corrcoef(amp[DROP:, col + j], hg[DROP:, ch])[0, 1]
        ax.plot(t, hgz, color='#9a9992', lw=1, label='HG bandpass envelope (offline)')
        ax.plot(t, y, color=C_SSR, lw=1.2, label='SSR amplitude')
        ax.set_title(f'SSR oscillator {fr[j]:.1f} Hz  (r with HG envelope over the recording = {r:.2f})', loc='left',
                     fontsize=10)
    for ax in axes:
        style(ax)
        ax.set_yticks([])
    axes[2].legend(fontsize=8, frameon=False, loc='upper right', ncol=2)
    axes[-1].set_xlabel('Time (s)')
    fig.tight_layout()
    save(fig, out, 'amp_feature_viz.png')


def fig_bp_corr(out, filters='results/filters'):
    fig, axes = grid_fig()
    allr = []
    for ax, pt in zip(axes, PTS):
        zf = np.load(os.path.join(filters, f'lag_xcorr_{pt}.npz'))
        r = zf['verwoert'][:, zf['verwoert'].shape[1] // 2]
        allr.append(r)
        ax.hist(r, bins=np.arange(-0.2, 1.0001, 0.05), color=C_SSR, rwidth=0.9)
        ax.axvline(np.median(r), color=INK, lw=1, ls='--')
        ax.set_title(f'{pt} (median {np.median(r):.2f})', fontsize=11)
        style(ax)
        note(f'bp_corr {pt}: n={len(r)} mean={r.mean():.3f} median={np.median(r):.3f} '
             f'>0.7: {100 * np.mean(r > 0.7):.1f}%')
    a = np.concatenate(allr)
    note(f'bp_corr all: n={len(a)} mean={a.mean():.3f} median={np.median(a):.3f} >0.7: {100 * np.mean(a > 0.7):.1f}%; '
         f'participant medians {min(map(np.median, allr)):.3f}-{max(map(np.median, allr)):.3f}')
    fig.text(0.5, 0.01, 'Correlation of SSR high-gamma amplitude with the offline bandpass HG envelope (zero lag)',
             ha='center', fontsize=12)
    fig.text(0.005, 0.5, 'Number of HG oscillators', va='center', rotation='vertical', fontsize=12)
    fig.tight_layout(rect=[0.015, 0.03, 1, 1])
    save(fig, out, 'SSR_bandpass_correlation_grid.png')


def cohen_d(x, y):
    s = np.sqrt(((len(x) - 1) * x.var(ddof=1) + (len(y) - 1) * y.var(ddof=1)) / (len(x) + len(y) - 2))
    return (x.mean() - y.mean()) / s


def fig_feat_audio(out):
    fig, axes = grid_fig()
    for ax, pt in zip(axes, PTS):
        # neural window 200 ms before the audio window = centre of the decoder's past-only context (-400..0 ms)
        spec, f = aligned_spec_and_feat(pt, load_feat(pt, 'SSR_amp'), lead_frames=20)
        env = spec.mean(1)
        fz = (f - f.mean(0)) / np.where(f.std(0) > 0, f.std(0), 1)
        r = np.abs(fz.T @ ((env - env.mean()) / env.std()) / len(env))
        fr = osc_freqs(pt)
        hgm = (fr >= 70) & (fr <= 170)
        p = stats.mannwhitneyu(r[hgm], r[~hgm], alternative='greater').pvalue
        d = cohen_d(r[hgm], r[~hgm])
        ax.axvspan(70, 170, color=C_SSR, alpha=0.06, lw=0)
        ax.scatter(fr, r, s=6, color=C_SSR, alpha=0.5, lw=0)
        ax.hlines(r[hgm].mean(), 70, 170, color=INK, ls='--', lw=1.5)
        ax.hlines(r[~hgm].mean(), 0, 200, color=MUTED, ls=':', lw=1.5)
        ax.set_xlim(0, 200)
        ax.set_title(pt, fontsize=11)
        ax.text(0.03, 0.97, f'HG {r[hgm].mean():.2f}, other {r[~hgm].mean():.2f}\nd = {d:.2f}, p = {p:.1e}',
                transform=ax.transAxes, va='top', fontsize=8.5, color=INK)
        style(ax)
        note(f'feat_audio {pt}: HG mean |r|={r[hgm].mean():.3f} non-HG={r[~hgm].mean():.3f} d={d:.2f} p={p:.2e}')
    fig.text(0.5, 0.01, 'SSR oscillator frequency (Hz)', ha='center', fontsize=12)
    fig.text(0.005, 0.5, '|r| with speech energy 200 ms later', va='center', rotation='vertical', fontsize=12)
    fig.tight_layout(rect=[0.015, 0.03, 1, 1])
    save(fig, out, 'Feature_Audio_Correlations_All_Subjects_new.png')


def fig_pls_weights(out, results='results'):
    fig, axes = grid_fig(sharey=False)
    for ax, pt in zip(axes, PTS):
        feat = load_feat(pt, 'SSR_amp')[DROP:]
        X = ef.stackFeatures(feat, causal=True)
        spec = np.load(os.path.join(ef.path_output, f'{pt}_spec.npy'))[DROP:]
        n = min(len(X), len(spec))
        X, spec = X[:n], spec[:n]
        X = (X - X.mean(0)) / np.where(X.std(0) > 0, X.std(0), 1)
        k = int(np.median(np.load(os.path.join(results, pt, 'cmp_SSR_amp_causal_nestednComp.npy'))))
        pls = PLSRegression(n_components=k, max_iter=1000, copy=False).fit(X, spec)
        coef = np.abs(pls.coef_.reshape(spec.shape[1], -1))        # (mel bins, lags*features)
        imp = coef.reshape(spec.shape[1], 9, feat.shape[1]).mean(axis=(0, 1))
        fr = osc_freqs(pt)
        hgm = (fr >= 70) & (fr <= 170)
        ax.axvspan(70, 170, color=C_SSR, alpha=0.06, lw=0)
        ax.scatter(fr, imp / imp.max(), s=6, color=C_SSR, alpha=0.5, lw=0)
        ax.set_xlim(0, 200)
        ax.set_title(f'{pt} ({k} comp.)', fontsize=11)
        style(ax)
        note(f'pls_weights {pt}: n_comp={k} mean weight HG/other = {imp[hgm].mean() / imp[~hgm].mean():.2f}')
    fig.text(0.5, 0.01, 'SSR oscillator frequency (Hz)', ha='center', fontsize=12)
    fig.text(0.005, 0.5, 'Mean |PLS coefficient| (normalized)', va='center', rotation='vertical', fontsize=12)
    fig.tight_layout(rect=[0.015, 0.03, 1, 1])
    save(fig, out, 'PLS_weights_all_subjects_shaded_new.png')


def fig_templates(out):
    fig, axes = grid_fig(sharey=False)
    s = np.linspace(70, 170, 1000)
    for ax, pt in zip(axes, PTS):
        som, _ = nf.drop_line_noise_oscillators(nf.load_somata_results(pt))
        hgf = np.array([f for r in som.values() for f in r['freqs'] if 70 <= f <= 170])
        dens = np.exp(KernelDensity(kernel='gaussian', bandwidth=2).fit(hgf[:, None]).score_samples(s[:, None]))
        tmpl = nf.build_density_template(som)
        n_tmpl = sum(not any(70 <= f <= 170 for f in r['freqs']) for r in som.values())
        ax.vlines(hgf, 0, dens.max() * 0.08, color=C_SSR, alpha=0.5, lw=0.8)
        ax.plot(s, dens, color=INK, lw=1.3)
        for f in tmpl:
            ax.axvline(f, color=C_BUF, lw=1.2, ls='--')
            ax.text(f, dens.max() * 1.02, f'{f:.0f}', ha='center', fontsize=8, color=C_BUF)
        ax.set_title(f'{pt} (template: {n_tmpl}/{len(som)} ch.)', fontsize=11)
        ax.set_ylim(0, dens.max() * 1.15)
        style(ax)
    fig.text(0.5, 0.01, 'Frequency (Hz)', ha='center', fontsize=12)
    fig.text(0.005, 0.5, 'Kernel density of SOMATA HG frequencies', va='center', rotation='vertical', fontsize=12)
    fig.tight_layout(rect=[0.015, 0.03, 1, 1])
    save(fig, out, 'SOMATA_density_templates_all_subjects.png')


NCOMP_SETS = [('HG_buf', 'BP: HG'), ('HG_LPamp_buf', 'BP: HG + LP amp.'), ('HG_LP_buf', 'BP: HG + LP'),
              ('HG_LP_PAC_buf', 'BP: HG + LP + PAC'), ('BPm_amp_buf', 'matched BP: amp.'),
              ('BPm_phase_buf', 'matched BP: amp. + phase'), ('BPm_pac_buf', 'matched BP: + PAC'),
              ('SSR_amp', 'SSR: amp.'), ('SSR_phase', 'SSR: amp. + phase'), ('SSR_pac', 'SSR: + PAC'),
              ('HG_vw', 'offline BP: HG'), ('HG_LP_vw', 'offline BP: HG + LP'), ('HG_LP_PAC_vw', 'offline BP: + PAC')]


def fig_ncomp(out, results='results/all'):
    grid = [4, 6, 8, 10, 12]
    M = np.array([[np.mean(np.load(os.path.join(results, f'cmp_{s}_causal_nestednComp.npy')) == g) for g in grid]
                  for s, _ in NCOMP_SETS])
    fig, ax = plt.subplots(figsize=(6.5, 5.2))
    im = ax.imshow(M * 100, cmap='Blues', vmin=0, vmax=100, aspect='auto')
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f'{100 * M[i, j]:.0f}', ha='center', va='center', fontsize=8,
                    color='white' if M[i, j] > 0.55 else INK)
    ax.set_xticks(range(len(grid)), grid)
    ax.set_yticks(range(len(NCOMP_SETS)), [l for _, l in NCOMP_SETS], fontsize=9)
    ax.set_xlabel('PLS components chosen by the inner cross-validation')
    fig.colorbar(im, ax=ax, label='% of outer folds (10 participants x 10 folds)')
    fig.tight_layout()
    save(fig, out, 'hyperparameter.png')


STAB_SETS = [('HG_buf', 'Bandpass (buffered): HG', C_BUF, '--'), ('HG_LP_PAC_buf', 'Bandpass (buffered): HG + LP + PAC', C_BUF, '-'),
             ('SSR_amp', 'SSR: amplitude', C_SSR, '--'), ('SSR_pac', 'SSR: amplitude + phase + PAC', C_SSR, '-')]


def fig_stability(out, results='results/all'):
    fig, ax = plt.subplots(figsize=(9, 4.2))
    folds = np.arange(1, 11)
    for s, lab, c, ls in STAB_SETS:
        a = np.load(os.path.join(results, f'cmp_{s}_causal_nestedlinearResults.npy')).mean(2)   # participants x folds
        m = a.mean(0)
        slope = np.polyfit(folds, m, 1)[0]
        ps = [np.polyfit(folds, a[i], 1)[0] for i in range(len(a))]
        t, p = stats.ttest_1samp(ps, 0)
        ax.plot(folds, m, ls, marker='o', ms=4, color=c, lw=1.6, label=f'{lab} (slope {slope:+.4f}/fold)')
        note(f'stability {s}: group slope {slope:+.5f} per fold; participant slopes mean {np.mean(ps):+.5f}, '
             f't(9)={t:.2f}, p={p:.3f}; fold means {m.min():.3f}-{m.max():.3f}')
    ax.set_xticks(folds)
    ax.set_xlabel('Fold (consecutive ~29 s segments)')
    ax.set_ylabel('Mean reconstruction r (past-only context)')
    style(ax)
    ax.legend(fontsize=8, frameon=False, loc='upper center', bbox_to_anchor=(0.5, -0.18), ncol=2)
    fig.tight_layout()
    save(fig, out, 'fold_stability_trends_new.png')


def spec_panel(pt, preds, bp='HG_LP_PAC_buf', ssr='SSR_pac'):
    spec = np.load(os.path.join(ef.path_output, f'{pt}_spec.npy'))[DROP:]
    pb = np.load(os.path.join(preds, f'{pt}_fig_{bp}_causal_predicted_spec.npy'))
    ps = np.load(os.path.join(preds, f'{pt}_fig_{ssr}_causal_predicted_spec.npy'))
    n = min(len(spec), len(pb), len(ps))
    return spec[:n], pb[:n], ps[:n]


def plot_specs(spec, pb, ps, t0, title, path, dpi):
    eb, es = np.abs(spec - pb).mean(1), np.abs(spec - ps).mean(1)
    t = t0 + np.arange(len(spec)) * ef.frameshift
    ext = [t[0], t[-1], 0, spec.shape[1]]
    fig, axes = plt.subplots(5, 1, figsize=(12, 8.5), sharex=True, gridspec_kw={'height_ratios': [3, 3, 1, 3, 1]})
    vmin, vmax = np.percentile(spec, 5), np.percentile(spec, 99)
    for ax, x, lab in ((axes[0], spec, 'Recorded speech'), (axes[1], pb, 'Causal bandpass (HG + LP + PAC)'),
                       (axes[3], ps, 'Causal SSR (amplitude + phase + PAC)')):
        im = ax.imshow(x.T, aspect='auto', origin='lower', cmap='inferno', vmin=vmin, vmax=vmax, extent=ext,
                       interpolation='nearest')
        ax.set_title(lab, loc='left', fontsize=10)
        ax.set_ylabel('Mel bin')
    for ax, e, c in ((axes[2], eb, C_BUF), (axes[4], es, C_SSR)):
        ax.plot(t, e, color=c, lw=0.8)
        ax.set_ylabel('|error|')
        ax.set_ylim(0, np.percentile(np.r_[eb, es], 99.5))
        style(ax)
    axes[-1].set_xlabel('Time in recording (s)')
    fig.suptitle(title, x=0.01, ha='left', fontsize=11)
    fig.colorbar(im, ax=axes.tolist(), shrink=0.5, pad=0.01, label='log-mel energy')
    fig.savefig(path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f'wrote {os.path.basename(path)}')


def fig_spectrograms(out, preds):
    rs = []
    for pt in PTS:
        spec, pb, ps = spec_panel(pt, preds)
        t0 = (DROP + HALF) * ef.frameshift + ef.winL / 2
        eb, es = np.abs(spec - pb).mean(1), np.abs(spec - ps).mean(1)
        r = np.corrcoef(eb, es)[0, 1]
        rs.append(r)
        note(f'error tracks {pt}: r={r:.3f}; mean |error| bandpass {eb.mean():.3f}, SSR {es.mean():.3f}; '
             f'std ratio recon/recorded: bandpass {pb.std() / spec.std():.2f}, SSR {ps.std() / spec.std():.2f}')
        plot_specs(spec, pb, ps, t0, f'{pt}: full recording after the 10 s initialization segment',
                   os.path.join(out, f'{pt}_spectograms_vertical.png'), dpi=110)
        if pt == 'sub-01':
            a = min(27500, len(spec) - 1000)
            plot_specs(spec[a:a + 1000], pb[a:a + 1000], ps[a:a + 1000], t0 + a * ef.frameshift,
                       'sub-01: 10 s excerpt', os.path.join(out, 'zoomed_spectograms_sub1.png'), dpi=200)
    note(f'error tracks: r = {min(rs):.2f}-{max(rs):.2f} (median {np.median(rs):.2f})')


FIGS = ['ncomp', 'stability', 'bp_corr', 'templates', 'feat_audio', 'pls_weights', 'amp_example', 'model_fits',
        'spectrograms']


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--figs', nargs='+', default=['all'])
    ap.add_argument('--out', default='paper/images')
    ap.add_argument('--preds', default=None, help='folder with predicted spectrograms (figure_inputs.py --out)')
    args = ap.parse_args()
    figs = FIGS if args.figs == ['all'] else args.figs
    for name in figs:
        if name == 'spectrograms':
            fig_spectrograms(args.out, args.preds)
        else:
            globals()[f'fig_{name}'](args.out)
    with open(os.path.join(args.out, '..', 'figure_numbers.txt'), 'a') as f:
        f.write('\n'.join(NUMBERS) + '\n')


if __name__ == '__main__':
    main()
