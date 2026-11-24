"""
What happens between the bandpass filter and the SSR model?

For one or more participants this script characterises the SSR oscillators as the causal,
complex (quadrature) LTI filters they become once the Kalman gain has converged
(see ssr_filter.py), and compares them with the FIR filters of the bandpass baseline:

  A. magnitude responses of the SSR oscillators of an example channel, with the 70-170 Hz
     4th-order Butterworth band-pass (Verwoert et al. 2022; zero-phase |H|^2 and causal |H|)
     and the <20 Hz low-pass overlaid;
  B. -3 dB bandwidth and group delay of every SSR oscillator versus its centre frequency,
     compared with the group delay of the causal Butterworth band-pass (streaming IIR) and of
     the 409-tap FIR of the original draft (199 ms);
  C. lagged correlation between the SSR high-gamma amplitude features and the bandpass
     high-gamma envelope for the offline Verwoert pipeline (zero-phase, non-causal), the buffered
     causal pipeline and the causal FIR of the draft: the lag at the peak shows the relative
     delay of the feature sets;
  D. cost of a streaming (sample-by-sample) steady-state SSR update for the full montage.

Outputs (in --out): filter_summary_{pt}.csv, lag_xcorr_{pt}.npz, ssr_vs_bandpass_filters_{pt}.png/.pdf,
and a printed summary.

Usage:
    export SSR_PATH_BIDS=... SSR_PATH_SOMATA=... SSR_PATH_FEATURES=...
    python compare_filters.py --pts sub-01 --out ./results/filters
    python compare_filters.py --aggregate --pts sub-01 sub-02 ... --out ./results/filters   # group summary
"""
import argparse
import csv
import os
import time

import numpy as np
import scipy.signal
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import extract_features as ef
import notebookfunctions as nf
import ssr_filter as sf

FS = 1024


def filter_table(params_list):
    rows = []
    for ch, p in enumerate(params_list):
        f_max = min(FS / 2 - 1, 1.5 * max(p['freqs']) + 40)
        for j, s in enumerate(sf.summarize_oscillator_filters(p, f_max=f_max, df=0.05)):
            s.update(channel=ch, osc=j, n_osc=len(p['freqs']), damping=float(p['ampVec'][j]),
                     sigma2=float(p['sigmaFreqs'][j]))
            rows.append(s)
    return rows


def iir_response(f_axis):
    """4th-order Butterworth filters of the Verwoert pipeline: complex causal response and group delay (ms)."""
    out = {}
    for kind in ('hg', 'lp'):
        _, H = scipy.signal.sosfreqz(ef._iir_sos(kind), worN=f_axis, fs=FS)
        out[kind] = H
    gd = sf.group_delay(out['hg'], f_axis, FS)
    band = (f_axis >= 70) & (f_axis <= 170)
    return out, dict(hg_gd_median_ms=float(np.median(gd[band])), hg_gd_min_ms=float(gd[band].min()),
                     hg_gd_max_ms=float(gd[band].max()), hg_gd_120_ms=float(np.interp(120, f_axis, gd)))


def lagged_corr(a, b, max_lag):
    """corr(a[t], b[t+lag]) for lag in -max_lag..max_lag (frames). Positive lag: b lags a."""
    a = (a - a.mean()) / a.std()
    b = (b - b.mean()) / b.std()
    out = np.zeros(2 * max_lag + 1)
    for i, lag in enumerate(range(-max_lag, max_lag + 1)):
        if lag >= 0:
            out[i] = np.mean(a[:len(a) - lag] * b[lag:])
        else:
            out[i] = np.mean(a[-lag:] * b[:len(b) + lag])
    return out


LAG_TIMINGS = ('verwoert', 'buffered', 'causal')


def hg_lag_analysis(pt, params_list, eeg, max_lag_ms=400):
    """Lagged correlation between SSR HG-oscillator amplitudes and HG bandpass envelopes."""
    drop = int(10 / ef.frameshift)          # the 10 s initialisation segment, as in decoding
    ssr = ef.extractSSPE(eeg, params_list, include_phase=False, include_pac=False, engine='steady_state')
    hg = {t: ef.extractHG(eeg, timing=t) for t in LAG_TIMINGS}
    max_lag = int(max_lag_ms / (ef.frameshift * 1000))
    curves = {t: [] for t in hg}
    col = 0
    for ch, p in enumerate(params_list):
        for j, f in enumerate(p['freqs']):
            if 70 <= f <= 170:
                for t in hg:
                    curves[t].append(lagged_corr(ssr[drop:, col + j], hg[t][drop:, ch], max_lag))
        col += len(p['freqs'])
    lags_ms = np.arange(-max_lag, max_lag + 1) * ef.frameshift * 1000
    return lags_ms, {t: np.array(c) for t, c in curves.items()}


def pac_redundancy(params_list, eeg):
    """
    How much does the windowed PAC feature, |mean_W(A_HG exp(i phi_low))|, differ from the
    window-mean HG amplitude it is built from? For a slow oscillator the phase hardly changes
    within a 50 ms window, so |mean(A e^{i phi})| ~ mean(A) * |mean(e^{i phi})| ~ mean(A).
    Returns, per low-frequency oscillator, its frequency and corr(PAC feature, HG amplitude feature).
    """
    drop = int(10 / ef.frameshift)
    feat = ef.extractSSPE(eeg, params_list, include_phase=True, include_pac=True, engine='steady_state')
    out, col = [], 0
    for p in params_list:
        freqs = np.asarray(p['freqs'])
        starts = []
        for f in freqs:
            starts.append(col)
            col += 1 + (3 if f < ef.LOW_FREQ_THRESHOLD else 0)
        hg = next(j for j, f in enumerate(freqs) if ef.HG_RANGE[0] < f < ef.HG_RANGE[1])
        a_hg = feat[drop:, starts[hg]]
        for j, f in enumerate(freqs):
            if f < ef.LOW_FREQ_THRESHOLD:
                out.append((f, np.corrcoef(feat[drop:, starts[j] + 3], a_hg)[0, 1]))
    return np.array(out)


def streaming_benchmark(params_list, eeg, seconds=20, frame=10):
    """
    Per-frame cost of updating the steady-state SSR states of ALL channels sample by sample
    (frame = 10 samples ~ 10 ms) with the diagonalised form s_t = lam*s_{t-1} + b*y_t.
    """
    lam, b, idx = [], [], []
    for ch, p in enumerate(params_list):
        A, K = sf.ssr_lti(p)
        l, V = np.linalg.eig(A)
        lam.append(l); b.append(np.linalg.solve(V, K).ravel()); idx.append(np.full(len(l), ch))
    lam, b, idx = np.concatenate(lam), np.concatenate(b), np.concatenate(idx)
    x = ef._ssr_preprocess(eeg[:int(seconds * FS)].astype(float))
    s = np.zeros(len(lam), complex)
    times = []
    for start in range(0, x.shape[0] - frame, frame):
        t0 = time.perf_counter()
        for t in range(start, start + frame):
            s = lam * s + b * x[t, idx]
        times.append((time.perf_counter() - t0) * 1000)
    times = np.array(times[50:])
    return dict(n_modes=len(lam), mean_ms=times.mean(), p95_ms=np.percentile(times, 95), max_ms=times.max())


def plot(pt, rows, params_list, lags_ms, curves, out):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    # A: example channel = channel with most oscillators that has an own (non-template) HG oscillator
    ex = max(range(len(params_list)), key=lambda c: (len(params_list[c]['freqs']) <= 6,
                                                      any(70 <= f <= 170 for f in params_list[c]['freqs'][:4]),
                                                      len(params_list[c]['freqs'])))
    f_axis = np.arange(0.1, 200, 0.1)
    H = sf.frequency_response(params_list[ex], f_axis)
    Hiir, _ = iir_response(f_axis)
    ax = axes[0]
    for j, f in enumerate(params_list[ex]['freqs']):
        m = np.abs(H[j]); ax.plot(f_axis, m / m.max(), lw=1.5, label=f'SSR osc. {f:.0f} Hz')
    ax.plot(f_axis, np.abs(Hiir['hg']) ** 2, 'k--', lw=1.2, label='Butterworth 70-170 Hz (zero-phase)')
    ax.plot(f_axis, np.abs(Hiir['lp']) ** 2, 'k:', lw=1.2, label='Butterworth <20 Hz (zero-phase)')
    ax.set_xlabel('Frequency (Hz)'); ax.set_ylabel('Normalised |H(f)|'); ax.set_title(f'A  {pt}, channel {ex}')
    ax.legend(fontsize=7, loc='upper right'); ax.set_xlim(0, 200)
    # B: group delay vs centre frequency
    ax = axes[1]
    fr = np.array([r['freq'] for r in rows]); gd = np.array([r['group_delay_ms'] for r in rows])
    bw = np.array([r['bandwidth'] for r in rows])
    sc = ax.scatter(fr, gd, c=bw, s=8, cmap='viridis')
    _, gd_iir = iir_response(np.arange(0.1, 200, 0.1))
    ax.axhline(sf.fir_bandpass_group_delay_ms(ef._fir_taps(), FS), color='k', ls=':', lw=1.2, label='409-tap FIR of the draft')
    ax.axhline(gd_iir['hg_gd_120_ms'], color='k', ls='--', lw=1.2, label='causal Butterworth 70-170 Hz (at 120 Hz)')
    ax.set_xlabel('SSR oscillator frequency (Hz)'); ax.set_ylabel('Group delay at peak (ms)')
    ax.set_yscale('log'); ax.set_title('B  all channels'); ax.legend(fontsize=8)
    plt.colorbar(sc, ax=ax, label='-3 dB bandwidth (Hz)')
    # C: lagged correlation
    ax = axes[2]
    for t, lab, ls in (('verwoert', 'Verwoert (offline, zero-phase)', '-'), ('buffered', 'buffered causal', '--'),
                       ('causal', 'causal FIR (draft)', ':')):
        c = curves[t]; m = c.mean(0)
        ax.plot(lags_ms, m, ls, label=f'{lab}: peak {lags_ms[np.argmax(m)]:+.0f} ms')
        ax.fill_between(lags_ms, m - c.std(0) / np.sqrt(len(c)), m + c.std(0) / np.sqrt(len(c)), alpha=0.2)
    ax.axvline(0, color='grey', lw=0.8)
    ax.set_xlabel('Lag of bandpass envelope relative to SSR amplitude (ms)')
    ax.set_ylabel('Correlation (mean over HG oscillators)'); ax.set_title('C  SSR-HG vs bandpass HG'); ax.legend(fontsize=8)
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(out, f'ssr_vs_bandpass_filters_{pt}.{ext}'), dpi=300)
    plt.close(fig)


def aggregate(out, pts):
    """Group-level summary of the per-participant outputs written by main() (no data needed)."""
    rows = []
    for pt in pts:
        f = os.path.join(out, f'filter_summary_{pt}.csv')
        if not os.path.exists(f):
            print(f'{pt}: no filter summary, skipped'); continue
        with open(f) as fh:
            for r in csv.DictReader(fh):
                rows.append(dict(pt=pt, freq=float(r['freq']), bw=float(r['bandwidth']), gd=float(r['group_delay_ms'])))
    fr = np.array([r['freq'] for r in rows]); bw = np.array([r['bw'] for r in rows]); gd = np.array([r['gd'] for r in rows])
    q = lambda x: f'{np.median(x):.1f} [IQR {np.percentile(x, 25):.1f}-{np.percentile(x, 75):.1f}]'
    for name, m in (('HG (70-170 Hz)', (fr >= 70) & (fr <= 170)), ('LF (<20 Hz)', fr < 20)):
        print(f'all {name:15s} n={m.sum():5d}  bandwidth {q(bw[m])} Hz  group delay {q(gd[m])} ms, '
              f'95th pct {np.percentile(gd[m], 95):.1f}, max {gd[m].max():.1f} ms')
    pr = [np.load(os.path.join(out, f'pac_redundancy_{pt}.npy')) for pt in pts
          if os.path.exists(os.path.join(out, f'pac_redundancy_{pt}.npy'))]
    if pr:
        med = [np.median(x[:, 1]) for x in pr]; allp = np.concatenate(pr)
        print(f'PAC vs HG amplitude: participant medians {np.min(med):.2f}-{np.max(med):.2f}; pooled median '
              f'{np.median(allp[:, 1]):.3f} (n={len(allp)}), <8 Hz {np.median(allp[allp[:, 0] < 8, 1]):.3f}')
    peaks = {}
    for pt in pts:
        f = os.path.join(out, f'lag_xcorr_{pt}.npz')
        if not os.path.exists(f):
            continue
        z = np.load(f); lags = z['lags_ms']
        for t in LAG_TIMINGS:
            if t in z:
                m = z[t].mean(0)
                peaks.setdefault(t, []).append((pt, m[len(m) // 2], m.max(), lags[np.argmax(m)]))
    for t, v in peaks.items():
        r0 = np.array([x[1] for x in v]); pk = np.array([x[2] for x in v]); lg = np.array([x[3] for x in v])
        print(f'SSR-HG vs {t:9s}: r(0) {r0.mean():.3f} (range {r0.min():.3f}-{r0.max():.3f}), peak r {pk.mean():.3f} '
              f'(range {pk.min():.3f}-{pk.max():.3f}) at lags {sorted(set(lg.astype(int).tolist()))} ms (n={len(v)} participants)')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--pts', nargs='+', default=['sub-01'])
    ap.add_argument('--out', default=os.path.join(os.environ.get('SSR_PATH_RESULT', './results'), 'filters'))
    ap.add_argument('--no-lag', action='store_true', help='skip the lagged-correlation analysis')
    ap.add_argument('--aggregate', action='store_true', help='only summarise existing outputs in --out over --pts')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    if args.aggregate:
        aggregate(args.out, args.pts)
        return
    for pt in args.pts:
        params_list = nf.get_initParams(pt)
        rows = filter_table(params_list)
        with open(os.path.join(args.out, f'filter_summary_{pt}.csv'), 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
        fr = np.array([r['freq'] for r in rows]); gd = np.array([r['group_delay_ms'] for r in rows])
        bw = np.array([r['bandwidth'] for r in rows])
        for name, m in (('HG (70-170 Hz)', (fr >= 70) & (fr <= 170)), ('LF (<20 Hz)', fr < 20), ('other', (fr >= 20) & (fr < 70) | (fr > 170))):
            if m.any():
                print(f'{pt} {name:15s} n={m.sum():4d}  bandwidth median {np.median(bw[m]):6.1f} Hz '
                      f'[IQR {np.percentile(bw[m], 25):.1f}-{np.percentile(bw[m], 75):.1f}]  group delay median '
                      f'{np.median(gd[m]):6.1f} ms [IQR {np.percentile(gd[m], 25):.1f}-{np.percentile(gd[m], 75):.1f}], max {gd[m].max():.1f}')
        _, gd_iir = iir_response(np.arange(0.1, 200, 0.1))
        print(f"{pt} causal 4th-order Butterworth 70-170 Hz: group delay {gd_iir['hg_gd_min_ms']:.1f}-{gd_iir['hg_gd_max_ms']:.1f} ms "
              f"in the pass band (median {gd_iir['hg_gd_median_ms']:.1f}, at 120 Hz {gd_iir['hg_gd_120_ms']:.1f}); "
              f"409-tap FIR {sf.fir_bandpass_group_delay_ms(ef._fir_taps(), FS):.1f} ms")
        eeg = ef.load_participant(pt)[0]
        bench = streaming_benchmark(params_list, eeg)
        print(f"{pt} streaming steady-state SSR update, all {len(params_list)} channels ({bench['n_modes']} complex modes): "
              f"{bench['mean_ms']:.3f} ms per 10 ms frame (p95 {bench['p95_ms']:.3f}, max {bench['max_ms']:.3f})")
        if args.no_lag:
            continue
        pr = pac_redundancy(params_list, eeg)
        np.save(os.path.join(args.out, f'pac_redundancy_{pt}.npy'), pr)
        print(f'{pt} corr(SSR PAC feature, HG amplitude feature): median {np.median(pr[:, 1]):.3f} '
              f'[IQR {np.percentile(pr[:, 1], 25):.3f}-{np.percentile(pr[:, 1], 75):.3f}], min {pr[:, 1].min():.3f} '
              f'over {len(pr)} low-frequency oscillators; median for <8 Hz: {np.median(pr[pr[:, 0] < 8, 1]):.3f}')
        lags_ms, curves = hg_lag_analysis(pt, params_list, eeg)
        np.savez(os.path.join(args.out, f'lag_xcorr_{pt}.npz'), lags_ms=lags_ms, **curves)
        for t, c in curves.items():
            m = c.mean(0)
            print(f'{pt} SSR-HG vs {t:17s} HG envelope: r(0) = {m[len(m)//2]:.3f}, peak r = {m.max():.3f} at {lags_ms[np.argmax(m)]:+.0f} ms '
                  f'(n = {len(c)} HG oscillators)')
        plot(pt, rows, params_list, lags_ms, curves, args.out)


if __name__ == '__main__':
    main()
