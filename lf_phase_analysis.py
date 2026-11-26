"""
What do the low-frequency 'phase' features carry, and how good are the causal estimates?

For every channel of a participant (first 10 s dropped, as in decoding):
  * agreement between the causal (buffered) and offline (whole-recording, zero-phase) bandpass low-pass phase
    features: circular difference of the window-mean phase and correlation of the cos features;
  * phase concentration within the 50 ms window (|mean exp(i phi)|): how much the window average smooths;
  * share of the 0-20 Hz power below 4 Hz (how slow the 'low-pass' signal is);
  * R^2 with which the window-mean low-passed voltage (offline LMP) is linearly explained by the features of the same
    window: bandpass [amp, sin, cos] (causal and offline) and SSR low-frequency oscillators [amp, sin, cos each];
  * R^2 of the bandpass phase features (sin, cos) explained by the SSR low-frequency features of the channel;
  * with --narrowband: phase error of every low-frequency SSR oscillator and of the causal (buffered) bandpass filter
    matched to it, relative to the offline zero-phase matched bandpass at the same frequency (window-mean phase):
    median absolute error, and the error after removing each oscillator's constant phase offset (filter delay).

    python lf_phase_analysis.py --pts sub-01 [--out results/lf_phase]

Needs the feature files of HG_LP_buf, HG_LP_vw and SSR_phase in SSR_PATH_FEATURES (run_comparisons.ensure_features).
"""
import argparse
import os

import numpy as np
import scipy.signal

import extract_features as ef
import notebookfunctions as nf
import run_comparisons as rc

DROP = 1000          # 10 s at 10 ms frames


def r2(y, X):
    X = np.column_stack([np.ones(len(y)), X])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = y - X @ beta
    return 1 - res.var() / y.var()


def analyse(pt, n_jobs=3):
    rc.ensure_features(pt, ['HG_LP_buf', 'HG_LP_vw', 'SSR_phase'], 'steady_state', n_jobs)
    load = lambda n: np.load(rc.feature_path(pt, n))
    buf, off, ssr = load('HG_LP_buf'), load('HG_LP_vw'), load('SSR_phase')
    params = nf.get_initParams(pt)
    eeg = ef.load_participant(pt)[0].astype(float)
    eeg = scipy.signal.detrend(eeg, axis=0)
    lp = scipy.signal.sosfiltfilt(ef._iir_sos('lp'), eeg, axis=0)
    del eeg
    starts, stops = ef._window_bounds(lp.shape[0])
    lmp = ef._window_mean(lp, starts, stops)
    f, P = scipy.signal.welch(lp[:200 * 1024], fs=1024, nperseg=4096, axis=0)
    low_share = P[(f > 0) & (f < 4)].sum(0) / P[(f > 0) & (f <= 20)].sum(0)
    del lp
    n = min(len(buf), len(off), len(ssr), len(lmp))
    sl = slice(DROP, n)
    rows, col = [], 0
    for ch, p in enumerate(params):
        fr = np.asarray(p['freqs'], float)
        widths = [1 + (2 if x < ef.LOW_FREQ_THRESHOLD else 0) for x in fr]
        blocks = np.split(ssr[sl, col:col + sum(widths)], np.cumsum(widths)[:-1], axis=1)
        col += sum(widths)
        lf_ssr = np.column_stack([b for b, x in zip(blocks, fr) if x < ef.LOW_FREQ_THRESHOLD]) \
            if any(fr < ef.LOW_FREQ_THRESHOLD) else None
        a_b, s_b, c_b = (buf[sl, 4 * ch + j] for j in range(3))
        a_o, s_o, c_o = (off[sl, 4 * ch + j] for j in range(3))
        y = lmp[sl, ch]
        dphi = np.angle(np.exp(1j * (np.arctan2(s_b, c_b) - np.arctan2(s_o, c_o))))
        row = dict(ch=ch, low_share=low_share[ch],
                   dphi_med_deg=np.degrees(np.median(np.abs(dphi))),
                   cos_corr=np.corrcoef(c_b, c_o)[0, 1],
                   R_window=np.median(np.hypot(s_o, c_o)),
                   r2_lmp_off=r2(y, np.column_stack([a_o, s_o, c_o])),
                   r2_lmp_buf=r2(y, np.column_stack([a_b, s_b, c_b])),
                   r2_lmp_amp_only=r2(y, a_b[:, None]))
        if lf_ssr is not None:
            row.update(n_lf=int((fr < ef.LOW_FREQ_THRESHOLD).sum()),
                       r2_lmp_ssr=r2(y, lf_ssr),
                       r2_bpcos_from_ssr=r2(c_b, lf_ssr), r2_bpsin_from_ssr=r2(s_b, lf_ssr))
        rows.append(row)
    return rows


def narrowband(pt, n_jobs=2):
    """Phase error of causal SSR and causal matched bandpass versus offline matched bandpass, per LF oscillator."""
    # SSR phase features: from SSR_phase, or from SSR_pac (same columns plus one PAC column per LF oscillator)
    ssr_set = 'SSR_phase' if os.path.exists(rc.feature_path(pt, 'SSR_phase')) else 'SSR_pac'
    lf_width_ssr = 3 if ssr_set == 'SSR_phase' else 4
    rc.ensure_features(pt, [ssr_set, 'BPm_phase_buf', 'BPm_phase_vw'], 'steady_state', n_jobs)
    ssr, buf, ref = (np.load(rc.feature_path(pt, n)) for n in (ssr_set, 'BPm_phase_buf', 'BPm_phase_vw'))
    n = min(len(ssr), len(buf), len(ref))
    sl = slice(DROP, n)
    out, col, cs = [], 0, 0
    for p in nf.get_initParams(pt):
        for f in np.asarray(p['freqs'], float):
            if f < ef.LOW_FREQ_THRESHOLD:
                ph = {k: np.arctan2(x[sl, c + 1], x[sl, c + 2]) for k, x, c in (('ssr', ssr, cs), ('buf', buf, col),
                                                                              ('ref', ref, col))}
                row = dict(freq=f)
                for k in ('ssr', 'buf'):
                    e = np.angle(np.exp(1j * (ph[k] - ph['ref'])))
                    off = np.angle(np.mean(np.exp(1j * e)))
                    row[f'{k}_err'] = np.degrees(np.median(np.abs(e)))
                    row[f'{k}_offset'] = np.degrees(off)
                    row[f'{k}_jitter'] = np.degrees(np.median(np.abs(np.angle(np.exp(1j * (e - off))))))
                out.append(row)
                col += 3
                cs += lf_width_ssr
            else:
                col += 1
                cs += 1
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--pts', nargs='+', default=['sub-01'])
    ap.add_argument('--out', default=None)
    ap.add_argument('--n-jobs', type=int, default=3)
    ap.add_argument('--narrowband', action='store_true', help='only the narrowband phase-accuracy analysis')
    args = ap.parse_args()
    if args.narrowband:
        bands = [(0, 4), (4, 8), (8, 13), (13, 20)]
        for pt in args.pts:
            rows = narrowband(pt, args.n_jobs)
            if args.out:
                os.makedirs(args.out, exist_ok=True)
                np.save(os.path.join(args.out, f'lf_narrowband_{pt}.npy'), rows, allow_pickle=True)
            for lo, hi in bands:
                sel = [r for r in rows if lo <= r['freq'] < hi]
                if sel:
                    m = lambda k: np.median([r[k] for r in sel])
                    print(f"{pt} {lo:2d}-{hi:2d} Hz (n={len(sel):3d}): error SSR {m('ssr_err'):5.1f} deg, buffered BP "
                          f"{m('buf_err'):5.1f} deg | offset SSR {m('ssr_offset'):6.1f}, BP {m('buf_offset'):6.1f} | "
                          f"jitter SSR {m('ssr_jitter'):5.1f}, BP {m('buf_jitter'):5.1f}", flush=True)
        return
    keys = ['low_share', 'dphi_med_deg', 'cos_corr', 'R_window', 'r2_lmp_off', 'r2_lmp_buf', 'r2_lmp_amp_only',
            'r2_lmp_ssr', 'r2_bpcos_from_ssr', 'r2_bpsin_from_ssr']
    for pt in args.pts:
        rows = analyse(pt, args.n_jobs)
        med = {k: np.nanmedian([r.get(k, np.nan) for r in rows]) for k in keys}
        print(pt, ' '.join(f'{k}={v:.3f}' for k, v in med.items()), flush=True)
        if args.out:
            os.makedirs(args.out, exist_ok=True)
            np.save(os.path.join(args.out, f'lf_phase_{pt}.npy'), rows, allow_pickle=True)


if __name__ == '__main__':
    main()
