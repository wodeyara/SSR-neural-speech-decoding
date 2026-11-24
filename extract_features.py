import os

import pandas as pd
import numpy as np
import numpy.matlib as matlib
import scipy
import scipy.signal
import scipy.io.wavfile
import scipy.fftpack

from pynwb import NWBHDF5IO
import MelFilterBank as mel
import phaseEM
from joblib import Parallel, delayed

winL = 0.05 #windowlength
frameshift = 0.01
modelOrder = 4
stepSize = 5
# Paths can be overridden with environment variables so the pipeline runs on other machines.
path_bids = os.environ.get('SSR_PATH_BIDS', '/Users/lisa/Documents/DSAI_year2/Marble/SingleWordProductionDutch/SingleWordProductionDutch-iBIDS')
path_output = os.environ.get('SSR_PATH_FEATURES', r'./features')
sr=1024

# Oscillators / bands below this frequency get phase (and PAC) features
LOW_FREQ_THRESHOLD = 20.0
# Range searched for the high-gamma partner of the PAC features
HG_RANGE = (70.0, 170.0)
# Length of the FIR filters used for all bandpass features (409 taps -> 199 ms group delay)
FIR_MEMORY_MS = 400

#Small helper function to speed up the hilbert transform by extending the length of data to the next power of 2
hilbert3 = lambda x: scipy.signal.hilbert(x, scipy.fftpack.next_fast_len(len(x)),axis=0)[:len(x)]


# ---------------------------------------------------------------------------------------
# Shared helpers: windowing and per-oscillator feature definitions
# ---------------------------------------------------------------------------------------

def _window_bounds(n_samples):
    """Start/stop sample of every 50 ms window with a 10 ms shift (same rounding as before)."""
    numWindows = int(np.floor((n_samples - winL * sr) / (frameshift * sr)))
    starts = np.floor(np.arange(numWindows) * frameshift * sr).astype(int)
    stops = np.floor(starts + winL * sr).astype(int)
    return starts, stops


def _window_mean(x, starts, stops):
    """Mean of x[start:stop] along axis 0 for every window (vectorised with a cumulative sum)."""
    x = np.asarray(x)
    cs = np.cumsum(x, axis=0)
    cs = np.concatenate([np.zeros((1,) + x.shape[1:], dtype=cs.dtype), cs], axis=0)
    n = (stops - starts).reshape((-1,) + (1,) * (x.ndim - 1))
    return (cs[stops] - cs[starts]) / n


def _channel_oscillator_features(z, freqs, include_phase=True, include_pac=True, pac_mode='mvl'):
    """
    Windowed features for the oscillators of ONE channel, given their complex (analytic)
    signals. This single definition is used for both the SSR states and the matched
    bandpass analytic signals so the two feature sets are computed identically.

    Parameters
    ----------
    z : complex array (samples, n_osc)
        Analytic signal of every oscillator (real part = signal, imaginary = quadrature).
    freqs : array (n_osc,)
        Centre frequency of every oscillator, in the same order as the columns of z.
    include_phase : bool
        Add sin and cos of the window-mean phase for oscillators < LOW_FREQ_THRESHOLD.
    include_pac : bool
        Add phase-amplitude coupling with the first high-gamma oscillator of the channel.
    pac_mode : {'mvl', 'complex'}
        'mvl'     : |mean(A_HG * exp(i*phi_low))| (as in the paper; 1 column).
        'complex' : real and imaginary part of mean(A_HG * exp(i*phi_low)) (2 columns).
                    These keep the phase at which high gamma is strong, i.e. the
                    amplitude-phase interaction that a linear decoder cannot build itself
                    from separate amplitude and phase features.

    Returns
    -------
    feat : array (windows, n_features)
        Column order per oscillator: amplitude, [sin phase, cos phase], [PAC ...].
    """
    freqs = np.asarray(freqs, dtype=float)
    assert z.shape[1] == len(freqs), "smth wrong with nr of oscillators and features"
    starts, stops = _window_bounds(z.shape[0])
    amp = np.abs(z)
    unit = np.exp(1j * np.angle(z))   # angle(0) = 0, as in the original implementation
    amp_w = _window_mean(amp, starts, stops)

    high_osc_idx = None
    if include_pac:
        # first high-gamma oscillator = the most prominent one found by SOMATA (or the template)
        for idx, freq in enumerate(freqs):
            if HG_RANGE[0] < freq < HG_RANGE[1]:
                high_osc_idx = idx
                break
        assert high_osc_idx is not None, "Channel appears to have no HG oscillator"

    cols = []
    for k, f in enumerate(freqs):
        cols.append(amp_w[:, k])
        if f < LOW_FREQ_THRESHOLD:
            if include_phase:
                cm = _window_mean(unit[:, k], starts, stops)
                cols.append(np.imag(cm))   # sin
                cols.append(np.real(cm))   # cos
            if include_pac:
                pz = _window_mean(amp[:, high_osc_idx] * unit[:, k], starts, stops)
                if pac_mode == 'mvl':
                    cols.append(np.abs(pz))
                elif pac_mode == 'complex':
                    cols.append(np.real(pz))
                    cols.append(np.imag(pz))
                else:
                    raise ValueError(f"unknown pac_mode {pac_mode}")
    return np.column_stack(cols)


def _ssr_preprocess(data):
    """Linear detrend and notch filtering used before SSR (and matched-bandpass) features."""
    data = scipy.signal.detrend(data, axis=0)
    #remove (harmonics of) line noise
    #NOTE: (i*100+49, i*100+51) notches 50, 150, 250 and 350 Hz but NOT 100 Hz; kept as is so that
    #features reproduce the published results (see paper/REVIEW_NOTES.md)
    for i in range (0,4):
        sos = scipy.signal.iirfilter(4, [(i*100+49)/(sr/2),(i*100+51)/(sr/2)],btype='bandstop',output='sos')
        data = scipy.signal.sosfiltfilt(sos,data,axis=0)
    return data


def _apply_fir(b, x, timing='causal'):
    """
    Apply an FIR filter along axis 0.

    timing='causal'            : y[t] = sum_k b[k] x[t-k]  (identical to scipy.signal.lfilter);
                                 a linear-phase FIR then delays the signal by (len(b)-1)/2 samples.
    timing='delay_compensated' : the same filter, shifted back by its group delay (a centred,
                                 non-causal, zero-phase FIR). Same magnitude response, no delay.
    """
    x = np.asarray(x, dtype=float)
    bb = np.asarray(b, dtype=float).reshape((-1,) + (1,) * (x.ndim - 1))
    full = scipy.signal.oaconvolve(x, bb, mode='full', axes=0)
    T = x.shape[0]
    if timing == 'causal':
        return full[:T]
    if timing == 'delay_compensated':
        D = (len(b) - 1) // 2
        return full[D:D + T]
    raise ValueError(f"unknown timing {timing}")


def _fir_taps():
    n_taps = int((FIR_MEMORY_MS / 1000.0) * sr)
    if n_taps % 2 == 0:
        n_taps += 1
    return n_taps


# ---------------------------------------------------------------------------------------
# SSR features
# ---------------------------------------------------------------------------------------

def _ssr_channel_features(y, params, include_phase, include_pac, pac_mode, engine):
    if engine == 'kalman':
        _, allX_full, _ = phaseEM.causalPhaseEM_MKmdl_noSeg(y, params, flagNoFit=True)
        if allX_full is None:
            raise RuntimeError("phaseEM failed for this channel (see traceback above)")
    elif engine == 'steady_state':
        import ssr_filter
        allX_full = ssr_filter.ssr_states_fast(y, params)
    else:
        raise ValueError(f"unknown engine {engine}")
    # allX_full shape is (time, freq, state_dim) where state_dim is 2 (real/imaginary)
    z = allX_full[:, :, 0] + 1j * allX_full[:, :, 1]
    return _channel_oscillator_features(z, params['freqs'], include_phase, include_pac, pac_mode)


def extractSSPE(data, initParams_list, include_phase = True, include_pac = True, pac_mode='mvl',
                engine='kalman', n_jobs=4):
    """
    Extract State-Space Phase Estimation (SSPE) features from EEG data.

    This is an alternative to extractHG with the same output shapes. It tracks multiple oscillators per channel and extract
    amplitude, phase, and phase-amplitude coupling (PAC) features.

    Parameters
    ----------
    data : array (samples, channels)
        EEG time series data
    initParams_list : list of dict
        List of dictionaries containing initialization parameters for the phaseEM
        algorithm for each channel. Each dict should contain: freqs, Fs, ampVec,
        sigmaFreqs, sigmaObs, windowSize, lowFreqBand
    include_phase : bool, optional
        Whether to include phase features for low-frequency oscillators (<20 Hz).
        Phase features are represented as sine and cosine components. Default is True.
    include_pac : bool, optional
        Whether to include phase-amplitude coupling features between low-frequency
        oscillators and high-gamma oscillators. Default is True.
    pac_mode : {'mvl', 'complex'}
        See _channel_oscillator_features. 'mvl' reproduces the published features.
    engine : {'kalman', 'steady_state'}
        'kalman' runs the per-sample Kalman filter in phaseEM (published results).
        'steady_state' runs the equivalent steady-state LTI filter (ssr_filter.py);
        identical after the ~1 s gain transient (dropped anyway with the first 10 s) and
        orders of magnitude faster.

    Returns
    -------
    feat : array (windows, oscillators)
        Feature matrix where each column represents an oscillator feature.
        Features include: amplitude for all oscillators, sine/cosine phase for
        low-frequency oscillators (if include_phase=True), and PAC features for
        low-frequency oscillators (if include_pac=True).
    """
    data = _ssr_preprocess(data)
    n_channels = data.shape[1]
    print(f"Starting parallel SSPE extraction for {n_channels} channels ({engine})...")
    blocks = Parallel(n_jobs=n_jobs)(
        delayed(_ssr_channel_features)(data[:, ch], initParams_list[ch], include_phase,
                                       include_pac, pac_mode, engine)
        for ch in range(n_channels)
    )
    return np.hstack(blocks)


# ---------------------------------------------------------------------------------------
# Bandpass features
# ---------------------------------------------------------------------------------------

def extractHG(data, low_pass=False, lp_phase=True, include_pac=False, pac_mode='mvl',
              include_lmp=False, timing='causal'):
    """
    Extract high-gamma band envelope features and optional low-pass amplitude/phase/PAC.

    With the default arguments this reproduces the published 'HG bandpass' (low_pass=False)
    and 'HG + LP bandpass' (low_pass=True) features exactly.

    Parameters
    ----------
    data : array (samples, channels)
        EEG time series data.
    low_pass : bool, optional
        If True, include low-pass (<20 Hz) amplitude (and by default phase) components.
    lp_phase : bool, optional
        Include sin/cos of the window-mean low-pass phase (only with low_pass=True).
        Set to False for an 'HG + LP amplitude' model that mirrors amplitude-only SSR.
    include_pac : bool, optional
        Add bandpass phase-amplitude coupling between the low-pass phase and the
        high-gamma envelope, computed exactly as the SSR PAC feature
        (|mean(A_HG * exp(i*phi_LP))| per 50 ms window, or its real/imaginary parts
        with pac_mode='complex'). Only with low_pass=True.
    include_lmp : bool, optional
        Add the window mean of the low-pass filtered voltage itself (local motor
        potential). Control for what the low-frequency 'phase' features encode.
    timing : {'causal', 'delay_compensated'}
        'causal' (default, published): causal 409-tap FIR, i.e. features lag the signal
        by 199 ms. 'delay_compensated': the same FIR applied centred (zero phase), which
        removes the 199 ms delay and makes the features time-aligned with SSR features.
        NB: in both cases the notch filters (sosfiltfilt) and the Hilbert transform are
        applied to the whole recording, i.e. they are not causal.

    Returns
    -------
    feat : array (windows, channels) when low_pass=False, otherwise (windows, channels*k)
        with per-channel column order [lp_amp, (lp_sin, lp_cos), (lmp), (pac...), hg_amp].
        The default low_pass=True layout is [lp_amp, lp_sin, lp_cos, hg_amp] (k=4).
    """
    # Linear detrend
    data = scipy.signal.detrend(data, axis=0)
    n_channels = data.shape[1]
    starts, stops = _window_bounds(data.shape[0])

    # FIR Filter setup (400 ms memory window)
    n_taps = _fir_taps()

    # HIGH-GAMMA BANDPASS (70 - 170 Hz)
    hg_filter = scipy.signal.firwin(
        n_taps,
        [70 / (sr / 2), 170 / (sr / 2)],
        pass_zero=False,
        window='hamming'
    )
    data_hg_filtered = _apply_fir(hg_filter, data, timing)

    # Notch filters for line noise harmonics (98-102 Hz, 148-152 Hz)

    # first harmonic of line noise
    sos = scipy.signal.iirfilter(4, [98/(sr/2),102/(sr/2)],btype='bandstop',output='sos')
    data_hg_filtered = scipy.signal.sosfiltfilt(sos,data_hg_filtered,axis=0)

    # second harmonic of line noise
    sos = scipy.signal.iirfilter(4, [148/(sr/2),152/(sr/2)],btype='bandstop',output='sos')
    data_hg_filtered = scipy.signal.sosfiltfilt(sos,data_hg_filtered,axis=0)

    # Hilbert transform for envelope
    hg_amplitude = np.abs(scipy.signal.hilbert(data_hg_filtered, axis=0))
    del data_hg_filtered
    hg_w = _window_mean(hg_amplitude, starts, stops)

    if not low_pass:
        return hg_w

    # LOW-PASS BAND (< 20 Hz)
    lp_filter = scipy.signal.firwin(
        n_taps,
        20 / (sr / 2),
        pass_zero='lowpass',
        window='hamming'
    )
    data_lp_filtered = _apply_fir(lp_filter, data, timing)

    analytic_lp = scipy.signal.hilbert(data_lp_filtered, axis=0)
    lp_amplitude = np.abs(analytic_lp)
    lp_unit = np.exp(1j * np.angle(analytic_lp))
    del analytic_lp

    # FEATURE EXTRACTION WINDOWING
    blocks = [_window_mean(lp_amplitude, starts, stops)]          # lp_amp
    if lp_phase:
        cm = _window_mean(lp_unit, starts, stops)
        blocks += [np.imag(cm), np.real(cm)]                         # lp_sin, lp_cos
    if include_lmp:
        blocks.append(_window_mean(data_lp_filtered, starts, stops))  # lmp
    if include_pac:
        pz = _window_mean(hg_amplitude * lp_unit, starts, stops)
        if pac_mode == 'mvl':
            blocks.append(np.abs(pz))
        elif pac_mode == 'complex':
            blocks += [np.real(pz), np.imag(pz)]
        else:
            raise ValueError(f"unknown pac_mode {pac_mode}")
    blocks.append(hg_w)                                              # hg_amp
    # interleave per channel: [ch0 features..., ch1 features..., ...]
    feat = np.stack(blocks, axis=2).reshape(len(starts), n_channels * len(blocks))
    return feat


def _matched_band(params, osc_idx, bandwidth, ssr_summary):
    f0 = float(params['freqs'][osc_idx])
    if bandwidth == 'ssr':
        lo, hi = ssr_summary[osc_idx]['f_lo'], ssr_summary[osc_idx]['f_hi']
    else:
        lo, hi = f0 - bandwidth / 2.0, f0 + bandwidth / 2.0
    hi = min(hi, sr / 2 - 2.0)
    return lo, hi


def _bandpass_matched_channel(y, params, include_phase, include_pac, pac_mode, bandwidth, timing):
    """Analytic signals of FIR bandpass filters centred on the channel's SSR oscillators."""
    ssr_summary = None
    if bandwidth == 'ssr':
        import ssr_filter
        f_max = min(sr / 2 - 1, 1.5 * max(params['freqs']) + 40)
        ssr_summary = ssr_filter.summarize_oscillator_filters(params, f_max=f_max, df=0.1)
    n_taps = _fir_taps()
    z = np.zeros((len(y), len(params['freqs'])), dtype=complex)
    bands = []
    for k in range(len(params['freqs'])):
        lo, hi = _matched_band(params, k, bandwidth, ssr_summary)
        if lo < 1.0:   # band reaching (close to) DC: low-pass
            b = scipy.signal.firwin(n_taps, hi / (sr / 2), pass_zero='lowpass', window='hamming')
        else:
            b = scipy.signal.firwin(n_taps, [lo / (sr / 2), hi / (sr / 2)], pass_zero=False, window='hamming')
        z[:, k] = scipy.signal.hilbert(_apply_fir(b, y, timing))
        bands.append((lo, hi))
    return _channel_oscillator_features(z, params['freqs'], include_phase, include_pac, pac_mode), bands


def extractBandpassMatched(data, initParams_list, include_phase=True, include_pac=True, pac_mode='mvl',
                           bandwidth='ssr', timing='causal', n_jobs=4, return_bands=False):
    """
    Bandpass counterpart of extractSSPE: for every SSR oscillator of every channel, an FIR
    bandpass (same 409-tap Hamming design as extractHG) + Hilbert transform is used instead of
    the Kalman filter. Same preprocessing, same oscillator frequencies, same feature layout and
    feature definitions as extractSSPE, so the ONLY difference between the two feature sets is
    how the analytic signal of each oscillator is estimated.

    Parameters
    ----------
    bandwidth : 'ssr' or float
        'ssr': use the -3 dB passband of the steady-state SSR filter of each oscillator
        (ssr_filter.summarize_oscillator_filters), i.e. match centre frequency AND bandwidth.
        float: fixed bandwidth in Hz centred on the oscillator frequency.
    timing : {'causal', 'delay_compensated'}
        See extractHG.
    """
    data = _ssr_preprocess(data)
    n_channels = data.shape[1]
    print(f"Starting matched-bandpass extraction for {n_channels} channels ({timing}, bw={bandwidth})...")
    out = Parallel(n_jobs=n_jobs)(
        delayed(_bandpass_matched_channel)(data[:, ch], initParams_list[ch], include_phase, include_pac,
                                           pac_mode, bandwidth, timing)
        for ch in range(n_channels)
    )
    feat = np.hstack([f for f, _ in out])
    if return_bands:
        return feat, [b for _, b in out]
    return feat


def extractHG_old(data, low_pass = False):
    """
    Extract high-gamma band envelope features from EEG data using Hilbert transform. If low-pass=True, also frequencies <20 Hz are included.

    This function windows the data, applies bandpass filtering in the high-gamma
    range (70-170 Hz) and low-pass filtering (<20 Hz) if low_pass=True, removes line noise harmonics,
    and extracts the envelope using the Hilbert transform.


    Parameters
    ----------
    data : array (samples, channels)
        EEG time series
    low_pass : bool, optional
        If True, apply low-pass filtering before high-gamma extraction. Default is False.

    Returns
    -------
    feat : array (windows, channels) or (windows, channels * 2) if low_pass=True
        Envelope feature matrix. Each row corresponds to a time window
        and each column to a channel.
        If low_pass=ture, the matrix has interleaved low-pass and high-gamma feature columns:
        [ch0_low, ch0_hg, ch1_low, ch1_hg, ...]
    """
    #Linear detrend
    data = scipy.signal.detrend(data,axis=0)
    #Number of windows
    numWindows = int(np.floor((data.shape[0]-winL*sr)/(frameshift*sr)))
    #Filter High-Gamma Band
    #sos = scipy.signal.iirfilter(4, [70/(sr/2),170/(sr/2)],btype='bandpass',output='sos')
    #data = scipy.signal.sosfiltfilt(sos,data,axis=0) (original uncausal method)
    #data = scipy.signal.sosfilt(sos,data,axis=0) (original causal method, produces roughly the same result as uncausal method)

    # Design FIR filter with 400ms memory for causal filtering
    filter_memory_samples = int(0.4 * sr)

    # Process high-gamma band
    data_hg = data.copy()
    #Filter High-Gamma Band
    sos = scipy.signal.iirfilter(4, [70/(sr/2),170/(sr/2)],btype='bandpass',output='sos')
    data_hg = scipy.signal.sosfilt(sos,data_hg,axis=0)

    #Attenuate first harmonic of line noise
    sos = scipy.signal.iirfilter(4, [98/(sr/2),102/(sr/2)],btype='bandstop',output='sos')
    data_hg = scipy.signal.sosfiltfilt(sos,data_hg,axis=0)
    #Attenuate second harmonic of line noise
    sos = scipy.signal.iirfilter(4, [148/(sr/2),152/(sr/2)],btype='bandstop',output='sos')
    data_hg = scipy.signal.sosfiltfilt(sos,data_hg,axis=0)
    data_hg = np.abs(hilbert3(data_hg))

    if low_pass:
        # Process low-pass band
        data_lp = data.copy()
        # Low-pass FIR filter (<20 Hz) using a Hamming window
        lp_filter = scipy.signal.firwin(filter_memory_samples, 20/(sr/2),
                                        pass_zero='lowpass', window='hamming')
        data_lp = scipy.signal.lfilter(lp_filter, [1.0], data_lp, axis=0)
        data_lp = np.abs(hilbert3(data_lp))

    #Create feature space
    n_channels = data.shape[1]
    feat = np.zeros((numWindows, n_channels * 2)) if low_pass else np.zeros((numWindows, n_channels))
    for win in range(numWindows):
        start= int(np.floor((win*frameshift)*sr))
        stop = int(np.floor(start+winL*sr))
        if low_pass:
            # Interleave low-pass and high-gamma features
            feat[win, 0::2] = np.mean(data_lp[start:stop,:],axis=0)  # even indices: low-pass
            feat[win, 1::2] = np.mean(data_hg[start:stop,:],axis=0)  # odd indices: high-gamma
        else:
            feat[win, :] = np.mean(data_hg[start:stop,:],axis=0)
    return feat

def stackFeatures(features, modelOrder=4, stepSize=5, causal=False):
    """
    Add temporal context to each window by stacking neighboring feature vectors

    Parameters
    ----------
    features: array (windows, channels)
        Feature time series
    modelOrder: int
        Number of temporal context to include prior to and after current window
    stepSize: float
        Number of temporal context to skip for each next context (to compensate for frameshift)
    causal: bool
        False (default, published): frames t-modelOrder*stepSize ... t+modelOrder*stepSize,
        i.e. +-200 ms around the current frame (200 ms look-ahead).
        True: the same number of frames, but only from the past,
        t-2*modelOrder*stepSize ... t (0 to -400 ms), so the decoder needs no future data.
        Output rows are aligned with the same frames t as in the non-causal case; the first
        modelOrder*stepSize rows reuse the first frame for unavailable past context.
    Returns
    ----------
    featStacked: array (windows, feat*(2*modelOrder+1))
        Stacked feature matrix
    """
    half = modelOrder * stepSize
    centres = np.arange(half, features.shape[0] - half)
    if causal:
        offsets = np.arange(-2 * half, 1, stepSize)
    else:
        offsets = np.arange(-half, half + 1, stepSize)
    idx = np.clip(centres[:, None] + offsets[None, :], 0, features.shape[0] - 1)
    featStacked = features[idx, :].reshape(len(centres), -1)
    return featStacked

def downsampleLabels(labels):
    """
    Downsamples non-numerical data by using the mode

    Parameters
    ----------
    labels : array of str
        Label time series
    sr: int
        Sampling rate of the data
    winL: float
        Length of window (in seconds) in which mode will be used
    frameshift: float
        Shift (in seconds) after which next window will be extracted
    Returns
    ----------
    newLabels: array of str
        Downsampled labels
    """
    numWindows=int(np.floor((labels.shape[0]-winL*sr)/(frameshift*sr)))
    newLabels = np.empty(numWindows, dtype="S15")
    for w in range(numWindows):
        start = int(np.floor((w*frameshift)*sr))
        stop = int(np.floor(start+winL*sr))
        vals, counts = np.unique(labels[start:stop], return_counts=True)
        newLabels[w] = vals[np.argmax(counts)]
        #newLabels[w]=scipy.stats.mode(labels[start:stop])[0][0].encode("ascii", errors="ignore").decode()
    return newLabels

def extractMelSpecs(audio, sr):
    """
    Extract logarithmic mel-scaled spectrogram, traditionally used to compress audio spectrograms

    Parameters
    ----------
    audio: array
        Audio time series
    sr: int
        Sampling rate of the audio
    winL: float
        Length of window (in seconds) in which spectrogram will be calculated
    frameshift: float
        Shift (in seconds) after which next window will be extracted
    numFilter: int
        Number of triangular filters in the mel filterbank
    Returns
    ----------
    spectrogram: array (numWindows, numFilter)
        Logarithmic mel scaled spectrogram
    """
    numWindows=int(np.floor((audio.shape[0]-winL*sr)/(frameshift*sr)))
    win = scipy.signal.windows.hann(int(np.floor(winL*sr + 1)))[:-1]
    spectrogram = np.zeros((numWindows, int(np.floor(winL*sr / 2 + 1))),dtype='complex')
    for w in range(numWindows):
        start_audio = int(np.floor((w*frameshift)*sr))
        stop_audio = int(np.floor(start_audio+winL*sr))
        a = audio[start_audio:stop_audio]
        spec = np.fft.rfft(win*a)
        spectrogram[w,:] = spec
    mfb = mel.MelFilterBank(spectrogram.shape[1], 23, sr)
    spectrogram = np.abs(spectrogram)
    spectrogram = (mfb.toLogMels(spectrogram)).astype('float')
    return spectrogram

def nameVector(elecs, modelOrder=4):
    """
    Creates list of electrode names

    Parameters
    ----------
    elecs: array of str
        Original electrode names
    modelOrder: int
        Temporal context stacked prior and after current window
        Will be added as T-modelOrder, T-(modelOrder+1), ...,  T0, ..., T+modelOrder
        to the elctrode names
    Returns
    ----------
    names: array of str
        List of electrodes including contexts, will have size elecs.shape[0]*(2*modelOrder+1)
    """
    names = matlib.repmat(elecs.astype(np.dtype(('U', 10))),1,2 * modelOrder +1).T
    for i, off in enumerate(range(-modelOrder,modelOrder+1)):
        names[i,:] = [e[0] + 'T' + str(off) for e in elecs]
    return names.flatten()  #Add 'F' if stacked the same as matlab


def load_participant(pt):
    """Load sEEG, audio, word markers and channel names of one participant from the iBIDS dataset."""
    io = NWBHDF5IO(os.path.join(path_bids,pt,'ieeg',f'{pt}_task-wordProduction_ieeg.nwb'), 'r')
    nwbfile = io.read()
    eeg = nwbfile.acquisition['iEEG'].data[:]
    audio = nwbfile.acquisition['Audio'].data[:]
    words = np.array(nwbfile.acquisition['Stimulus'].data[:], dtype=str)
    io.close()
    channels = pd.read_csv(os.path.join(path_bids,pt,'ieeg',f'{pt}_task-wordProduction_channels.tsv'), delimiter='\t')
    channels = np.array(channels['name'])
    return eeg, audio, words, channels


def extract_spectrogram(pt, audio=None, words=None, channels=None, save=True):
    """
    Ground-truth log-mel spectrogram, word labels and feature names, aligned to STACKED features
    (both ends trimmed by modelOrder*stepSize windows, as in the original Verwoert et al. code).
    This is the {pt}_spec.npy that reconstruct() expects.
    """
    if audio is None:
        _, audio, words, channels = load_participant(pt)
    audio_sr = 48000
    target_SR = 16000
    audio = scipy.signal.decimate(audio,int(audio_sr / target_SR))
    audio_sr = target_SR
    scaled = np.int16(audio/np.max(np.abs(audio)) * 32767)
    melSpec = extractMelSpecs(scaled,audio_sr)
    words = downsampleLabels(words)
    words = words[modelOrder*stepSize:words.shape[0]-modelOrder*stepSize]
    melSpec = melSpec[modelOrder*stepSize:melSpec.shape[0]-modelOrder*stepSize,:]
    feature_names = nameVector(channels[:,None], modelOrder=modelOrder)
    if save:
        os.makedirs(path_output, exist_ok=True)
        scipy.io.wavfile.write(os.path.join(path_output,f'{pt}_orig_audio.wav'),audio_sr,scaled)
        np.save(os.path.join(path_output,f'{pt}_procWords.npy'), words)
        np.save(os.path.join(path_output,f'{pt}_spec.npy'), melSpec)
        np.save(os.path.join(path_output,f'{pt}_feat_names.npy'), feature_names)
    return melSpec


def run_extract_features(pt, initParams_list=None, SSPE=True, include_phase = True, include_pac = True, low_pass=False,
                         saveAs = None, stack=False, **kwargs):
    """
    Extract EEG features for a single participant and save them.

    Parameters
    ----------
    pt : str
        Participant identifier (e.g., 'sub-01')
    initParams_list : list of dict
        List of initialization parameters for SSPE algorithm per channel.
        Required if SSPE=True.
    SSPE : bool, optional
        If True, use SSPE feature extraction. If False, use high-gamma extraction.
        Default is True.
    include_phase : bool, optional
        If True, include phase information in SSPE features. Default is True.
    include_pac : bool, optional
        If True, include phase-amplitude coupling in SSPE features. Default is True.
    low_pass : bool, optional
        If True, include low-pass features in HG extraction. Default is False.
    saveAs : str, optional
        Custom filename suffix for the saved feature file.
    stack : bool, optional
        Stack temporal context before saving. Default False: features are saved UNSTACKED,
        which is what reconstruct(..., unstacked=True) expects and how all published
        results were produced (the 10 s initialisation segment is dropped before stacking).
        (Previously this function always stacked, which double-stacked features when combined
        with reconstruct(..., unstacked=True) in run_pipeline.)
    **kwargs
        Passed on to extractSSPE / extractHG (e.g. pac_mode, engine, timing, lp_phase).

    The ground-truth spectrogram is produced by extract_spectrogram(pt).
    Saves {pt}{saveAs}.npy (or {pt}_SSPEfeat.npy / {pt}_causalfeat.npy) in path_output.
    """
    eeg, _, _, _ = load_participant(pt)

    #Extract features
    if not SSPE:
        #Extract HG features
        feat = extractHG(eeg, low_pass=low_pass, **kwargs)
    else:
        #Extract SSPE features
        feat = extractSSPE(eeg, initParams_list, include_phase = include_phase, include_pac = include_pac, **kwargs)
    print("extracted features")

    if stack:
        feat = stackFeatures(feat,modelOrder=modelOrder,stepSize=stepSize)

    #Save everything
    os.makedirs(path_output, exist_ok=True)
    if saveAs:
        np.save(os.path.join(path_output,f'{pt}{saveAs}.npy'), feat)
    else:
        if SSPE:
            np.save(os.path.join(path_output,f'{pt}_SSPEfeat.npy'), feat)
        else:
            np.save(os.path.join(path_output,f'{pt}_causalfeat.npy'), feat)
    return feat



if __name__=="__main__":
    import notebookfunctions  # imported here to avoid a circular import (notebookfunctions imports this module)
    pt = 'sub-06'
    initParams_list = notebookfunctions.get_initParams(pt)
    run_extract_features(pt,initParams_list)
