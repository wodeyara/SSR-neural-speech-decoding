"""
Steady-state (LTI) view of the state-space rhythms (SSR) Kalman filter.

With fixed oscillator parameters (flagNoFit=True in phaseEM), the Kalman gain of the
SSR model converges to a constant within a few hundred samples. From then on the
filter is a linear, time-invariant, *causal* system

    x_t = A x_{t-1} + K y_t,      A = (I - K M^T) Phi

so every oscillator j is simply a complex-valued (quadrature) IIR filter applied to
the channel signal:

    z_j(t) = x_{2j}(t) + i x_{2j+1}(t),     Z_j(e^{iw}) = H_j(e^{iw}) Y(e^{iw}).

This module makes that equivalence explicit. It is used to

* compute the effective frequency response, -3 dB bandwidth and group delay of each
  SSR oscillator, so SSR features can be compared like-for-like with bandpass
  filters (see `compare_filters.py`);
* run the SSR filter orders of magnitude faster than the per-sample Python loop in
  `phaseEM.causalPhaseEM_MKmdl_noSeg` (identical output after the gain transient);
* design bandpass comparators whose centre frequency and bandwidth are matched to
  the SSR oscillators (`extract_features.extractBandpassMatched`).
"""

import numpy as np
import scipy.linalg
import scipy.signal

from phaseEM import genParametersSoulatMdl_sspp


def ssr_matrices(freqs, damping, sigma2, Fs):
    """Return (Phi, Q, M) exactly as phaseEM builds them (M is a column vector)."""
    Phi, Q, M = genParametersSoulatMdl_sspp(np.asarray(freqs, float), Fs,
                                            np.asarray(damping, float),
                                            np.asarray(sigma2, float))
    return Phi, Q, M


def steady_state_gain(Phi, Q, M, R):
    """
    Steady-state prior covariance and Kalman gain of the SSR model.

    Solves the discrete algebraic Riccati equation for the one-step-ahead (prior)
    covariance P and returns K = P M / (M^T P M + R), the gain used in
    phaseEM.oneStepKFupdate_sspp once it has converged.
    """
    R = np.atleast_2d(float(np.squeeze(R)))
    P = scipy.linalg.solve_discrete_are(Phi.T, M, Q, R)
    P = 0.5 * (P + P.T)
    K = P @ M / (M.T @ P @ M + R).item()
    return K, P


def ssr_lti(params, R=None):
    """
    LTI form (A, K) of the SSR filter for one channel.

    Parameters
    ----------
    params : dict
        An initParams dict as produced by notebookfunctions.get_initParams
        (keys freqs, ampVec, sigmaFreqs, sigmaObs, Fs).
    R : float, optional
        Observation-noise variance. Defaults to params['sigmaObs'] (what the
        decoding pipeline uses).
    """
    Fs = params['Fs']
    Phi, Q, M = ssr_matrices(params['freqs'], params['ampVec'], params['sigmaFreqs'], Fs)
    R = params['sigmaObs'] if R is None else R
    K, _ = steady_state_gain(Phi, Q, M, R)
    A = (np.eye(Phi.shape[0]) - K @ M.T) @ Phi
    return A, K


def frequency_response(params, f_axis, R=None):
    """
    Complex frequency response H_j(f) from the channel signal to the complex state
    of every oscillator j, evaluated at the frequencies in f_axis (Hz, may be
    negative).

    Returns
    -------
    H : ndarray (n_osc, len(f_axis)) complex
    """
    A, K = ssr_lti(params, R)
    Fs = params['Fs']
    n = A.shape[0]
    n_osc = n // 2
    C = np.zeros((n_osc, n), complex)
    C[np.arange(n_osc), 2 * np.arange(n_osc)] = 1.0
    C[np.arange(n_osc), 2 * np.arange(n_osc) + 1] = 1j
    I = np.eye(n)
    H = np.zeros((n_osc, len(f_axis)), complex)
    for k, f in enumerate(f_axis):
        zinv = np.exp(-2j * np.pi * f / Fs)
        H[:, k] = (C @ np.linalg.solve(I - A * zinv, K)).ravel()
    return H


def group_delay(H, f_axis, Fs):
    """Group delay in milliseconds, -d(arg H)/d(omega), via finite differences."""
    w = 2 * np.pi * np.asarray(f_axis) / Fs
    ph = np.unwrap(np.angle(H), axis=-1)
    gd_samples = -np.gradient(ph, w, axis=-1)
    return gd_samples / Fs * 1000.0


def summarize_oscillator_filters(params, R=None, f_max=None, df=0.05):
    """
    Centre frequency, peak gain, -3 dB band edges/bandwidth, and group delay at the
    peak for each oscillator of a channel.
    """
    Fs = params['Fs']
    f_max = Fs / 2 if f_max is None else f_max
    f_axis = np.arange(df, f_max, df)
    H = frequency_response(params, f_axis, R)
    gd = group_delay(H, f_axis, Fs)
    out = []
    for j, f0 in enumerate(params['freqs']):
        mag = np.abs(H[j])
        k = int(np.argmax(mag))
        half = mag >= mag[k] / np.sqrt(2)
        lo = k
        while lo > 0 and half[lo - 1]:
            lo -= 1
        hi = k
        while hi < len(mag) - 1 and half[hi + 1]:
            hi += 1
        out.append({'freq': float(f0), 'peak_freq': float(f_axis[k]), 'peak_gain': float(mag[k]),
                    'f_lo': float(f_axis[lo]), 'f_hi': float(f_axis[hi]),
                    'bandwidth': float(f_axis[hi] - f_axis[lo]),
                    'group_delay_ms': float(gd[j, k])})
    return out


def ssr_states_fast(y, params, start=None, R=None):
    """
    Steady-state SSR filtering of one channel, vectorised with scipy.signal.lfilter.

    Equivalent to phaseEM.causalPhaseEM_MKmdl_noSeg(y, params, flagNoFit=True)[1]
    after the Kalman-gain transient (the loop starts with P = 0.001 I and converges
    to the steady-state gain within a few hundred samples). Like the loop, states are
    zero before `start` (default: params['windowSize']).

    Returns
    -------
    allX_full : ndarray (T, n_osc, 2)
        Real and imaginary oscillator states, same layout as phaseEM.
    """
    y = np.asarray(y, float)
    start = params['windowSize'] if start is None else start
    A, K = ssr_lti(params, R)
    # Diagonalise A: x = V s, s_t = L s_{t-1} + (V^-1 K) y_t  -> first-order complex IIRs.
    lam, V = np.linalg.eig(A)
    b = np.linalg.solve(V, K).ravel()
    seg = y[start:]
    S = np.empty((len(lam), len(seg)), complex)
    for m in range(len(lam)):
        S[m] = scipy.signal.lfilter([b[m]], [1.0, -lam[m]], seg)
    X = np.real(V @ S)
    n_osc = A.shape[0] // 2
    out = np.zeros((len(y), n_osc, 2))
    out[start:] = X.reshape(n_osc, 2, -1).transpose(2, 0, 1)
    return out


def fir_bandpass_group_delay_ms(n_taps, Fs):
    """Group delay of a linear-phase FIR (e.g. extract_features.extractHG): (N-1)/2 samples."""
    return (n_taps - 1) / 2 / Fs * 1000.0
