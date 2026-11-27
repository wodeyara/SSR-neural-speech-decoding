"""
Timing of the feature pipelines and of the speech information in the neural signal.

Three analyses (results in results/timing, figure in paper/images/timing_overview.png):

--kernels   Temporal footprint of every high-gamma pipeline, measured with the actual feature code: a synthetic
            signal with short (30 ms) high-gamma bursts on white noise is passed through extractHG (earlier causal
            FIR, buffered causal, offline) and through the steady-state SSR filters of all channels of one
            participant. The burst-triggered average of the feature, as a function of (window centre - burst
            centre), shows when a feature responds to neural activity. Summed over the stacked context windows this
            gives the stretch of neural activity that the decoder sees for one audio frame.

--lag       Where the information is (per participant). Features are computed offline with zero-phase filters,
            so that they have no delay: window-mean low-passed voltage (LMP) and band amplitudes from 1 to 170 Hz.
            (a) A linear decoder using ONE 50 ms window per channel at lag L relative to the audio frame
                reconstructs the spectrogram (10 contiguous folds); r as a function of band and lag.
            (b) The usual 9-window, 400 ms context (PLS, 8 components) is slid along the lag axis; r as a function
                of the centre of the context window. Centre -200 ms is the past-only context, 0 the +-200 ms one.
            (c) Autocorrelation of the log-mel spectrogram (how slowly the target changes).

--speed     Cost per 10 ms frame of the causal estimators for one participant (all channels): buffered zero-phase
            bandpass (filter + Hilbert on the last 400 ms) and steady-state SSR update.

--plot      Figure from the saved results.

    export SSR_PATH_BIDS=... SSR_PATH_SOMATA=$PWD/somata_results SSR_PATH_FEATURES=...
    python timing_analysis.py --kernels --speed
    python timing_analysis.py --lag --pts sub-01 sub-02 ...
    python timing_analysis.py --plot
"""
import argparse
import os
import time

import numpy as np
import scipy.signal

import extract_features as ef
import notebookfunctions as nf
import run_comparisons as rc

FS = ef.sr
OUT = 'results/timing'
DROP = 1000                                   # 10 s initialisation segment (frames)
HALF = ef.modelOrder * ef.stepSize            # 20 frames: trim of the stacked features / spectrogram
MAX_LAG = 100                                 # frames (1 s)
LAGS = np.arange(-MAX_LAG, MAX_LAG + 1, 5)    # single-window lags, frames (50 ms steps)
CENTRES = np.arange(-60, 41, 10)              # centre of the 9-window context, frames
STACK = np.arange(-HALF, HALF + 1, ef.stepSize)
BANDS = [('1-4', (1, 4)), ('4-8', (4, 8)), ('8-13', (8, 13)), ('13-30', (13, 30)), ('30-70', (30, 70)),
         ('70-170', (70, 170))]


# ----------------------------------------------------------------------------------------------------------------
# Footprint of the pipelines
# ----------------------------------------------------------------------------------------------------------------

def burst_signal(seconds=300, seed=0):
    """White noise with 30 ms Hann-windowed bursts of 70-170 Hz noise every 1.5-1.8 s."""
    rng = np.random.default_rng(seed)
    n = int(seconds * FS)
    x = rng.standard_normal(n)
    carrier = scipy.signal.sosfiltfilt(ef._iir_sos('hg'), rng.standard_normal(n))
    carrier /= carrier.std()
    L = int(0.03 * FS)
    win = scipy.signal.windows.hann(L)
    centres, t = [], 12.0
    while t < seconds - 2:
        c = int(round(t * FS))
        x[c - L // 2:c - L // 2 + L] += 30 * win * carrier[c - L // 2:c - L // 2 + L]
        centres.append(c)
        t += 1.5 + rng.uniform(0, 0.3)
    return x, np.array(centres)


def triggered(feat, centres, n_samples, max_ms=700, bin_ms=5):
    """Burst-triggered average of the features versus (window centre - burst centre) in ms."""
    starts, stops = ef._window_bounds(n_samples)
    wc = (starts + stops) / 2 / FS * 1000
    edges = np.arange(-max_ms, max_ms + bin_ms, bin_ms)
    feat = feat - np.median(feat, axis=0)
    acc = np.zeros((len(edges) - 1, feat.shape[1]))
    cnt = np.zeros(len(edges) - 1)
    for c in centres:
        lag = wc[:len(feat)] - c / FS * 1000
        m = (lag > -max_ms) & (lag < max_ms)
        b = np.digitize(lag[m], edges) - 1
        np.add.at(acc, b, feat[m])
        np.add.at(cnt, b, 1)
    return (edges[:-1] + edges[1:]) / 2, acc / cnt[:, None]


def kernels(pt='sub-01', n_jobs=4):
    x, centres = burst_signal()
    out = {}
    for timing in ('causal', 'buffered', 'verwoert'):
        f = ef.extractHG(x[:, None], timing=timing)
        out['lag_ms'], out[timing] = triggered(f, centres, len(x))
    params = nf.get_initParams(pt)
    ssr = ef.extractSSPE(np.repeat(x[:, None], len(params), axis=1), params, include_phase=False, include_pac=False,
                         engine='steady_state', n_jobs=n_jobs)
    freqs = np.concatenate([np.asarray(p['freqs'], float) for p in params])
    hg = (freqs >= 70) & (freqs <= 170)
    _, out['ssr'] = triggered(ssr[:, hg], centres, len(x))
    out['ssr_freqs'] = freqs[hg]
    for k in ('causal', 'buffered', 'verwoert', 'ssr'):
        r = out[k] / out[k].max(axis=0)
        peak = out['lag_ms'][np.argmax(r, axis=0)]
        com = (out['lag_ms'][:, None] * np.clip(r, 0, None)).sum(0) / np.clip(r, 0, None).sum(0)
        print(f'{k:9s} peak at {np.median(peak):6.1f} ms, centre of mass {np.median(com):6.1f} ms '
              f'(range {com.min():.1f} to {com.max():.1f})')
    os.makedirs(OUT, exist_ok=True)
    np.savez(os.path.join(OUT, 'kernels.npz'), **out)


# ----------------------------------------------------------------------------------------------------------------
# Where the information is
# ----------------------------------------------------------------------------------------------------------------

def band_features(pt):
    """Delay-free (offline, zero-phase) window features per channel: LMP and band amplitudes."""
    eeg = ef.load_participant(pt)[0].astype(float)
    x = ef._ssr_preprocess(eeg)
    del eeg
    starts, stops = ef._window_bounds(x.shape[0])
    feats = {'LMP': ef._window_mean(scipy.signal.sosfiltfilt(ef._iir_sos('lp'), x, axis=0), starts, stops)}
    for name, (lo, hi) in BANDS:
        sos = scipy.signal.butter(4, [lo, hi], btype='bandpass', fs=FS, output='sos')
        amp = np.abs(ef.hilbert3(scipy.signal.sosfiltfilt(sos, x, axis=0)))
        assert np.all(np.isfinite(amp))
        feats[name] = ef._window_mean(amp, starts, stops)
    return feats


def _colcorr(a, b):
    a = a - a.mean(0)
    b = b - b.mean(0)
    return (a * b).sum(0) / np.sqrt((a ** 2).sum(0) * (b ** 2).sum(0))


def cv_ridge(X, Y, folds=10, lam=1.0):
    """Mean test correlation over Mel bins and contiguous folds of a (lightly regularised) linear decoder."""
    from sklearn.model_selection import KFold
    rs = []
    for tr, te in KFold(folds, shuffle=False).split(X):
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-12
        A = (X[tr] - mu) / sd
        ym = Y[tr].mean(0)
        W = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ (Y[tr] - ym))
        rs.append(_colcorr(((X[te] - mu) / sd) @ W + ym, Y[te]).mean())
    return float(np.mean(rs))


def cv_pls(X, Y, folds=10, n_comp=8):
    from sklearn.cross_decomposition import PLSRegression
    from sklearn.model_selection import KFold
    rs = []
    for tr, te in KFold(folds, shuffle=False).split(X):
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-12
        est = PLSRegression(n_components=n_comp, max_iter=1000, scale=False).fit((X[tr] - mu) / sd, Y[tr])
        rs.append(_colcorr(est.predict((X[te] - mu) / sd), Y[te]).mean())
    return float(np.mean(rs))


def lag_analysis(pt, n_jobs=4):
    from joblib import Parallel, delayed
    feats = band_features(pt)
    spec = np.load(os.path.join(os.environ['SSR_PATH_FEATURES'], f'{pt}_spec.npy'))
    n_win = len(feats['LMP'])
    w = np.arange(DROP + HALF + MAX_LAG, min(n_win, len(spec) + HALF) - HALF - MAX_LAG)   # audio frames used
    Y = spec[w - HALF]
    names = list(feats)
    single = np.array(Parallel(n_jobs=n_jobs)(delayed(cv_ridge)(feats[b][w + l], Y) for b in names for l in LAGS)
                      ).reshape(len(names), len(LAGS))
    sets = {'HG': ['70-170'], 'HG+LMP': ['70-170', 'LMP']}

    def window(cols, c):
        return cv_pls(np.hstack([feats[b][w + c + o] for b in cols for o in STACK]), Y)
    slide = np.array(Parallel(n_jobs=n_jobs)(delayed(window)(cols, c) for cols in sets.values() for c in CENTRES)
                     ).reshape(len(sets), len(CENTRES))
    Yz = (Y - Y.mean(0)) / Y.std(0)
    full = (spec - Y.mean(0)) / Y.std(0)
    acorr = np.array([np.mean(Yz * full[w - HALF + l]) for l in range(-MAX_LAG, MAX_LAG + 1)])
    os.makedirs(OUT, exist_ok=True)
    np.savez(os.path.join(OUT, f'lag_{pt}.npz'), bands=names, lags_ms=LAGS * 10, single=single,
             sets=list(sets), centres_ms=CENTRES * 10, slide=slide, acorr=acorr,
             acorr_lags_ms=np.arange(-MAX_LAG, MAX_LAG + 1) * 10)
    print(pt, 'single-window peak r per band:',
          ' '.join(f'{b}={single[i].max():.2f}@{LAGS[single[i].argmax()] * 10:+d}' for i, b in enumerate(names)))
    print(pt, 'sliding context:', {s: ' '.join(f'{v:.3f}' for v in slide[i]) for i, s in enumerate(sets)}, flush=True)


# ----------------------------------------------------------------------------------------------------------------
# Cost per frame
# ----------------------------------------------------------------------------------------------------------------

def speed(pt='sub-01', frames=300):
    import compare_filters as cf
    eeg = ef.load_participant(pt)[0].astype(float)
    params = nf.get_initParams(pt)
    x = scipy.signal.detrend(eeg[:int(30 * FS)], axis=0)
    n_buf = int(round(0.4 * FS))
    hg_sos = np.vstack([ef._iir_sos('hg'), ef._iir_sos('notch')])
    lp_sos = ef._iir_sos('lp')
    t_hg, t_all = [], []
    for i in range(frames):
        seg = x[1000 + 10 * i:1000 + 10 * i + n_buf]
        t0 = time.perf_counter()
        a = np.abs(scipy.signal.hilbert(scipy.signal.sosfiltfilt(hg_sos, seg, axis=0), axis=0))[-51:].mean(0)
        t1 = time.perf_counter()
        z = scipy.signal.hilbert(scipy.signal.sosfiltfilt(lp_sos, seg, axis=0), axis=0)[-51:]
        _ = np.abs(z).mean(0), np.exp(1j * np.angle(z)).mean(0)
        t2 = time.perf_counter()
        t_hg.append((t1 - t0) * 1000)
        t_all.append((t2 - t0) * 1000)
    ssr = cf.streaming_benchmark(params, eeg)
    res = dict(pt=pt, n_channels=x.shape[1], buffered_hg_ms=np.mean(t_hg[20:]), buffered_hg_lp_ms=np.mean(t_all[20:]),
               buffered_hg_lp_p95_ms=np.percentile(t_all[20:], 95), ssr_ms=ssr['mean_ms'], ssr_p95_ms=ssr['p95_ms'],
               ssr_modes=ssr['n_modes'])
    print(res)
    os.makedirs(OUT, exist_ok=True)
    np.save(os.path.join(OUT, f'speed_{pt}.npy'), res, allow_pickle=True)


# ----------------------------------------------------------------------------------------------------------------
# Figure
# ----------------------------------------------------------------------------------------------------------------

C_SSR, C_BUF, C_OFF, C_OLD, INK, MUTED = '#2a78d6', '#eb6834', '#eda100', '#7a7f8a', '#1f2430', '#6b7280'


def _title(ax, letter, title, sub, pad=16):
    ax.annotate(f'{letter}  {title}', (0.02, 1.0), xycoords=('figure fraction', 'axes fraction'), xytext=(0, pad + 11),
                textcoords='offset points', ha='left', va='bottom', fontsize=10, fontweight='bold')
    ax.annotate(sub, (0.02, 1.0), xycoords=('figure fraction', 'axes fraction'), xytext=(0, pad), textcoords='offset points',
                ha='left', va='bottom', fontsize=8, color=MUTED)


def decoder_footprint(lag_ms, resp, offsets_ms, tau):
    """Sensitivity of the stacked feature vector of one audio frame to neural activity at time tau (ms)."""
    resp = np.clip(resp, 0, None)
    k = sum(np.interp(o - tau, lag_ms, resp, left=0, right=0) for o in offsets_ms)
    return k / k.max()


def plot(out_png='paper/images/timing_overview.png', dpi=200):
    import glob
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.edgecolor': MUTED, 'xtick.color': MUTED, 'ytick.color': MUTED,
                         'axes.labelcolor': INK, 'text.color': INK, 'font.family': 'DejaVu Sans'})
    k = np.load(os.path.join(OUT, 'kernels.npz'))
    files = sorted(glob.glob(os.path.join(OUT, 'lag_sub-*.npz')))
    L = [np.load(f, allow_pickle=True) for f in files]
    n = len(L)
    single = np.mean([d['single'] for d in L], axis=0)
    slide = np.array([d['slide'] for d in L])                       # (pt, set, centre)
    acorr = np.mean([d['acorr'] for d in L], axis=0)
    bands, lags, centres = list(L[0]['bands']), L[0]['lags_ms'], L[0]['centres_ms']
    xlim = (-800, 600)
    tau = np.arange(xlim[0], xlim[1] + 1, 2.0)
    past, sym = np.arange(-400, 1, 50), np.arange(-200, 201, 50)
    ssr_resp = np.median(k['ssr'] / k['ssr'].max(axis=0), axis=1)
    rows = [('Earlier draft: bandpass\n(409-tap FIR), ±200 ms stack', k['causal'][:, 0], sym, C_OLD, 225),
            ('Earlier draft: SSR,\n±200 ms stack', ssr_resp, sym, C_SSR, 225),
            ('Main: causal bandpass\n(buffered), past-only stack', k['buffered'][:, 0], past, C_BUF, 25),
            ('Main: SSR,\npast-only stack', ssr_resp, past, C_SSR, 25),
            ('Reference: offline bandpass,\npast-only stack', k['verwoert'][:, 0], past, C_OFF, None)]

    fig = plt.figure(figsize=(8.8, 11.4))
    gs = fig.add_gridspec(4, 2, height_ratios=[3.3, 1.0, 2.5, 1.9], width_ratios=[1, 0.018], hspace=0.62, wspace=0.03,
                          left=0.27, right=0.92, top=0.95, bottom=0.05)
    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[1, 0], sharex=axA)
    axC = fig.add_subplot(gs[2, 0], sharex=axA)
    cax = fig.add_subplot(gs[2, 1])
    axD = fig.add_subplot(gs[3, 0], sharex=axA)

    # A: footprint of each pipeline for one audio frame
    for i, (label, resp, offs, col, need) in enumerate(rows):
        y0 = len(rows) - 1 - i
        f = decoder_footprint(k['lag_ms'], resp, offs, tau)
        axA.fill_between(tau, y0, y0 + 0.8 * f, color=col, alpha=0.85, lw=0)
        axA.plot(xlim, [y0, y0], color=MUTED, lw=0.5)
        axA.text(-0.02, y0 + 0.36, label, transform=axA.get_yaxis_transform(), ha='right', va='center', fontsize=8.5)
        com = (tau * f).sum() / f.sum()
        axA.plot([com], [y0 + 0.9], marker='v', ms=5, color=INK, clip_on=False)
        if need is None:
            axA.text(xlim[1], y0 + 0.4, 'needs the whole\nrecording (not real-time)', ha='right', va='center',
                     fontsize=7.5, color=MUTED)
        else:
            axA.plot([need, need], [y0, y0 + 0.8], color=INK, lw=1.5)
            axA.text(need + 12, y0 + 0.4, f'last sample\nneeded: +{need} ms', ha='left', va='center', fontsize=7.5)
    axA.axvspan(-25, 25, color=INK, alpha=0.10, lw=0)
    axA.text(0, len(rows) - 0.02, 'audio frame', ha='center', va='bottom', fontsize=8)
    axA.set_ylim(-0.1, len(rows) + 0.05)
    axA.set_yticks([])
    axA.spines['left'].set_visible(False)
    _title(axA, 'A', 'What the decoder sees for one audio frame',
           'Measured footprint of the stacked high-gamma features (▼ centre of mass); black bar: last neural sample '
           'needed before the frame can be decoded', pad=14)

    # B: autocorrelation of the target
    al = L[0]['acorr_lags_ms']
    axB.plot(al, acorr, color=INK, lw=2)
    axB.axhline(0, color=MUTED, lw=0.5)
    axB.set_ylabel('r')
    axB.set_ylim(-0.1, 1.02)
    for v in (-200, 200):
        axB.plot([v], [np.interp(v, al, acorr)], 'o', ms=5, color=INK)
    axB.annotate(f'r = {np.interp(200, al, acorr):.2f} at ±200 ms', (200, np.interp(200, al, acorr)),
                 xytext=(8, 6), textcoords='offset points', fontsize=8)
    _title(axB, 'B', 'How slowly the speech spectrogram changes',
           f'Autocorrelation of the log-mel spectrogram (mean over Mel bins and {n} participants)', pad=6)

    # C: information by frequency and lag
    order = ['70-170', '30-70', '13-30', '8-13', '4-8', '1-4', 'LMP']
    lab = {'LMP': 'slow voltage < 20 Hz\n(LMP, window mean)', '1-4': '1–4 Hz amplitude', '4-8': '4–8 Hz amplitude',
           '8-13': '8–13 Hz amplitude', '13-30': '13–30 Hz amplitude', '30-70': '30–70 Hz amplitude',
           '70-170': '70–170 Hz amplitude\n(high gamma)'}
    M = np.array([single[bands.index(b)] for b in order])
    step = lags[1] - lags[0]
    im = axC.imshow(M, aspect='auto', cmap='Blues', vmin=0, vmax=np.ceil(M.max() * 20) / 20,
                    extent=(lags[0] - step / 2, lags[-1] + step / 2, len(order) - 0.5, -0.5))
    for i in range(1, len(order)):
        axC.axhline(i - 0.5, color='white', lw=2)
    for i, b in enumerate(order):
        j = M[i].argmax()
        axC.plot([lags[j]], [i], 'o', ms=5, mfc='white', mec=INK, mew=1)
        axC.text(-0.02, i, lab[b], transform=axC.get_yaxis_transform(), ha='right', va='center', fontsize=8.5)
        axC.text(xlim[1] - 8, i, f'peak r = {M[i].max():.2f}', ha='right', va='center', fontsize=7.5,
                 color=INK if M[i, -1] < 0.5 * im.get_clim()[1] else 'white')
    axC.set_yticks([])
    axC.spines['left'].set_visible(False)
    axC.axvline(0, color=INK, lw=0.6, ls=':')
    cb = fig.colorbar(im, cax=cax)
    cb.set_label('reconstruction r', fontsize=8)
    cb.outline.set_visible(False)
    _title(axC, 'C', 'Where the information is, by frequency',
           f'Reconstruction r from ONE 50 ms window per channel placed at each lag (delay-free offline features, '
           f'mean of {n} participants); ○ best lag', pad=6)

    # D: sliding the 400 ms context
    sets = list(L[0]['sets'])
    for s, col in (('HG+LMP', '#1baf7a'), ('HG', C_OFF)):
        v = slide[:, sets.index(s)]
        m = v.mean(0)
        w = v - v.mean(1, keepdims=True) + v.mean()
        sem = w.std(0, ddof=1) / np.sqrt(n)
        axD.fill_between(centres, m - sem, m + sem, color=col, alpha=0.25, lw=0)
        axD.plot(centres, m, '-o', color=col, lw=2, ms=4, label={'HG': 'high gamma', 'HG+LMP': 'high gamma + LMP'}[s])
        axD.text(centres[-1] + 15, m[-1], {'HG': 'high gamma', 'HG+LMP': 'high gamma\n+ LMP'}[s], va='center',
                 fontsize=8)
    for c, txt, ha in ((-200, 'past-only context: main analysis,\nand in effect the earlier bandpass', 'right'),
                       (0, '±200 ms context:\nearlier SSR', 'left')):
        axD.axvline(c, color=INK, lw=0.8, ls=':')
        axD.annotate(txt, (c, 1.0), xycoords=('data', 'axes fraction'), xytext=(-4 if ha == 'right' else 4, 2),
                     textcoords='offset points', fontsize=7.5, va='bottom', ha=ha)
        y0 = 0.0 if c else 0.05
        axD.axvspan(c - 200, c + 200, ymin=y0, ymax=y0 + 0.035, color=INK, alpha=0.55, lw=0)
    axD.set_ylabel('reconstruction r')
    axD.set_xlabel('time of neural activity relative to the audio frame (ms); negative = brain before sound')
    _title(axD, 'D', 'Sliding the whole 400 ms context window',
           f'Reconstruction r versus the centre of the 9-window context (delay-free offline features, PLS; '
           f'mean ± within-participant SEM, n = {n})', pad=22)
    axA.set_xlim(xlim)
    for ax in (axB, axC):
        plt.setp(ax.get_xticklabels(), visible=False)
    fig.savefig(out_png, dpi=dpi)
    print('wrote', out_png)
    # numbers
    print('autocorr at 0, +-100, +-200, +-400 ms:', [round(float(np.interp(v, al, acorr)), 3) for v in (0, 100, 200, 400)])
    for i, b in enumerate(order):
        half = lags[M[i] >= M[i].max() - 0.5 * (M[i].max() - M[i].min())]
        print(f'{b:7s} peak r {M[i].max():.3f} at {lags[M[i].argmax()]:+d} ms; half-height {half.min():+d}..{half.max():+d} ms; '
              f'r at -400/-200/0/+200: ' + ' '.join(f'{np.interp(v, lags, M[i]):.3f}' for v in (-400, -200, 0, 200)))
    for s in sets:
        print(s, 'context centre', dict(zip(centres.tolist(), slide[:, sets.index(s)].mean(0).round(3).tolist())))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--kernels', action='store_true')
    ap.add_argument('--lag', action='store_true')
    ap.add_argument('--speed', action='store_true')
    ap.add_argument('--plot', action='store_true')
    ap.add_argument('--pts', nargs='+', default=rc.ALL_PTS)
    ap.add_argument('--n-jobs', type=int, default=4)
    args = ap.parse_args()
    if args.kernels:
        kernels(n_jobs=args.n_jobs)
    if args.speed:
        speed()
    if args.lag:
        for pt in args.pts:
            if os.path.exists(os.path.join(OUT, f'lag_{pt}.npz')):
                print(pt, 'exists, skipping')
                continue
            lag_analysis(pt, args.n_jobs)
    if args.plot:
        plot()


if __name__ == '__main__':
    main()
