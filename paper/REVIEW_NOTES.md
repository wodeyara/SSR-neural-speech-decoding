# Review notes: code, numbers, and SSR-vs-bandpass comparability

These notes accompany the revised `main.tex`. Items still open are marked **TODO** here and in red in the PDF
(`\showtodosfalse` in `main.tex` hides the red markers).

How the checks were done: the code was read in full; the saved outputs of the notebooks (`report_figures.ipynb`, `PAC.ipynb`)
were extracted and compared with the manuscript; the per-participant results printed during the PLS grid search
(`report_figures.ipynb`, cell 15) were used to recompute the group statistics; and the public dataset was downloaded to rerun parts of the
pipeline for sub-01 (SOMATA initialisation, feature extraction, decoding).

---

## 1. Are the numbers in the manuscript right?

| Claim in the draft | Status | Evidence / correction |
|---|---|---|
| Mean r: HG 0.728, HG+LP 0.761, SSR amp 0.719, amp+phase 0.763, full 0.765/0.766 | ✅ except rounding | Grid search: 0.72813, 0.76139, 0.71872, 0.76284, **0.76548**. The full model is **0.765**; abstract, Discussion and Limitations said 0.766. Fixed. |
| Six contrasts: t = 4.03, −0.47, 4.37, 0.37, 2.04, 0.77; p_FDR = 0.009, 0.723, 0.009, 0.723, 0.143, 0.689 | ✅ consistent | The paper's own t-values reproduce its BH-corrected p-values exactly. Participant-level Fisher-z tests on the per-participant means give t = 4.16, −0.39, 4.44, 0.41, 1.92, 0.78: same signs and same significance. The code that produced the paper's fold-level numbers was not in the repo; `stats_contrasts.py` now implements the described procedure. |
| The repo's stats cell (`report_figures.ipynb`, cell 24) | ⚠️ different analysis | Holm correction on raw r over different families: t = 3.60, −0.65, 3.67, 0.19, 1.97, 0.52. This is where the abstract's p_corrected = 0.01 / 0.08 came from. Abstract now uses the BH values (0.009 / 0.14). |
| "Full SSR … while it **did** differ significantly from HG+LP (t = 0.77, p = 0.689)" | ❌ | Should be "did **not**". Fixed. |
| HG+LP better in 9/10; +phase better in 9/10; +PAC better in 6/10; full vs HG+LP better in 5/10 | ✅ | Recomputed from the participant means. |
| Limitations: PAC "improved performance in 7 of 10 participants" | ❌ | 6/10 (the 7/10 is from an older run). Fixed. |
| Limitations: "weaker or absent SSR benefit in sub-03 and sub-06" | ❌ outdated | With the final models sub-03 (+0.037) and sub-06 (+0.010) gain. The participant who loses with every alternative feature set is **sub-09**; amplitude-only SSR loses most in sub-01, sub-02, sub-05 and sub-09. Fixed. |
| Table 2 random control / z-scores | ✅ but mis-described | The values match cell 22. However, the "random control" correlates the ground-truth spectrogram with a circularly shifted copy **of itself** (inherited from Verwoert et al.), so it does not depend on the features. That is why it is ≈0.000 ± 0.160 for every model. The caption and Methods now say so. |
| SSR-vs-bandpass HG correlation: mean 0.57, 30.3% > 0.70, medians up to 0.765 | ⚠️ not reproducible from repo | The saved notebook output (cell 1) gives mean **0.10**, 0.1% > 0.7, because it uses the causal FIR features, which lag by 199 ms (see §3). The paper numbers came from an earlier zero-phase run that is not in the repo. **TODO**: regenerate with `extractHG(timing='delay_compensated')`. |
| Error-track correlations "0.54 (sub-05) to 0.76 (sub-10)" | ❌ | Notebook cell 7 gives **0.64 (sub-01) to 0.78 (sub-10)**, and it reads `{pt}_bandpass_predicted_spec.npy` / `{pt}_PAC_predicted_spec.npy`, which are not the final `res_*` models. Text updated to the notebook values. **TODO** recompute from final predictions. |
| "Smoothing more prominent for the bandpass reconstruction" | ⚠️ unquantified | The variance figure was commented out. **TODO** quantify (std ratio) or drop. |
| HG cluster "75–90 Hz for participants 1–8" | ⚠️ | sub-05 clusters at 100–110 Hz (next to an un-notched 100 Hz line-noise peak, see §2). sub-06 / sub-09 spikes at 75/125 Hz and >130 Hz are template oscillators. Reworded. |
| PLS components 6 / 9 / 8 | ✅ | Grid search output. Note: n_comp was chosen on the reporting folds (not nested). |
| Runtime: "per-frame execution completes within 7 ms … substantial headroom" | ❌ | The 5.37 ms covers feature derivation only. Kalman filtering costs 1.44 ms per channel per 50 ms block, i.e. ~0.29 ms per channel per 10 ms frame. For 127 channels on 4 threads that is ~9 ms per frame, so ~14.5 ms in total, which is over the 10 ms budget in Python. The steady-state filter (§3) fixes this. Text corrected. |
| Fold length "30-second" | minor | 290 s after dropping 10 s → ~29 s per fold. |
| Introduction: HG "80–150 Hz" | ❌ | Methods and code use 70–170 Hz. Fixed. |
| Zoomed spectrogram figure | ❌ | x-axis labelled 10–300 s but shows a 10 s excerpt (frames 27500–28500); titles "Ground-Thruth", "PCA". **TODO** fix in `report_figures.ipynb` cell 11. |

### Template fallback (should be reported)
The high-gamma template was appended to **428 of 1103 channels (39%)**:
sub-01 50/127, sub-02 62/127, sub-03 17/127, sub-04 45/115, sub-05 7/60, sub-06 114/127 (90%), sub-07 7/127, sub-08 15/54, sub-09 60/117, sub-10 51/122.
This is now in Methods and Limitations. Note that sub-06, where 90% of the HG oscillators are template copies, is also the participant with the smallest SSR gains.

---

## 2. Code review

### Fixed (committed, behaviour of the published pipeline unchanged)
* `phaseEM.causalPhaseEM_MKmdl_noSeg(time=True)` crashed: the `time` argument shadowed the `time` module, and the `except` block swallowed the error and returned `None`. So the timing in `PAC.ipynb` cannot have come from this version of the file (that cell also uses an undefined `initParams_`).
* `phaseEM`: `float()` of 1×1 arrays raises on NumPy ≥ 2.5, which again made every channel silently return `None`. Replaced by `.item()` (identical result).
* `run_extract_features` always stacked before saving, while `run_pipeline` then called `reconstruct(..., unstacked=True)`, which **double-stacks** SSR features. It also used n_comp 9/5 instead of 8/6/9 and lacked HG+LP. The function now saves unstacked features (as for all published results) and `run_pipeline` mirrors the published runs.
* `highest_hg_mel_correlations` called `extractHG(eeg, eeg_sr)`, which passed 1024 as `low_pass` and returned 4 features per channel. It also compared features with a spectrogram trimmed by 20 frames.
* Circular import `extract_features` ↔ `notebookfunctions` ↔ `reconstruction_minimal` (it worked only with the notebooks' import order).
* `phaseEM` demo unpacked 5 return values from a 3-value function.

### Not changed (they would change published numbers): decide and rerun
1. **Line noise at 100 Hz is not removed before SSR.** `range(0,4)` with `(i*100+49, i*100+51)` notches 50/150/250/350 Hz. The PSD figure shows the 100 Hz peak (sub-05, sub-08), and sub-05's HG oscillators/template sit at 100–110 Hz. Recommend notching 100 (and 200) Hz.
2. **Observation noise fixed at σ²_R = 1 µV²** instead of SOMATA's estimate (`obs_noise_R`, ≈0.3 µV² for sub-01 ch 1; signal variance ≈3×10⁴ µV²). The steady-state filters for R = 1 and R = 0.31 are practically identical (same bandwidths/delays to <0.1 Hz / 0.1 ms), so this is harmless. It is now stated in Methods.
3. **Preprocessing is not causal in either pipeline**: whole-recording detrend, zero-phase notch filters, and (bandpass) a Hilbert transform of the whole recording. For the online claims, use causal notches for SSR. A truly causal bandpass needs a causal analytic signal (e.g. complex FIR / cFIR, Smetanin 2020).
4. **Oscillator–spectrogram correlations (Fig. 5A, `report_figures` cell 4)**: SSR features (row j = window j) are correlated with `_spec.npy` (row j = window j+20). So neural activity at t is compared with audio at t+200 ms, and the initialisation segment (with zero states before 2 s) is included. The Mann–Whitney test treats oscillators of the same channel as independent.
5. **PLS weights (Fig. 5B)** come from the last CV fold only (`est.coef_`), and weight magnitude ≠ relevance (Haufe et al. 2014). The cache branch in cell 9 leaves `weights` undefined.
6. **`report_figures` cell 21** uses stale indices after HG+LP was added: `amp_sf = results[1]` is HG+LP. Its printed "gains" are amp+phase SSR − HG. Cell 25/26 labels are also off (e.g. "Adding phase improved … 4 of 10" compares amp+phase SSR with HG+LP).

---

## 3. What happens between the bandpass filter and the SSR model?

**SSR with fixed parameters is a causal, linear filter bank.** The Kalman gain converges in <1 s, after which
`X_t = A X_{t-1} + K y_t` with `A = (I − K M) Φ`. Each oscillator's complex state is the output of a complex (quadrature)
IIR filter `H_j(e^{iω}) = c_jᵀ (I − A e^{−iω})⁻¹ K` applied to the channel signal (`ssr_filter.py`; matches the Kalman loop to
<10⁻⁸ and runs ~50–1000× faster). This gives a common language for both feature types:

| | bandpass HG (published) | SSR HG oscillator (sub-01, ch 1, example) |
|---|---|---|
| passband | 70–170 Hz (100 Hz) | 62.7–81.4 Hz (18.7 Hz), centred on the channel's own peak |
| delay | **199 ms** (409-tap linear-phase FIR, applied causally) | **18.5 ms** group delay |
| analytic signal | Hilbert transform of the whole recording (non-causal) | the state itself (causal) |
| LF features | 0–20 Hz low-pass, 199 ms delay, one broadband "phase" | separate oscillators (e.g. 2.6, 16.2 Hz), 30–40 ms delay |

Sub-01, all channels (`compare_filters.py`): PENDING

Consequences:
* The causal bandpass features **lag the SSR features by ~180–200 ms**. With the ±200 ms context, the decoder sees bandpass
  activity from about −400 to 0 ms relative to each audio frame, but SSR activity from about −200 to +200 ms. The two models therefore use
  different effective neural windows, which confounds the comparison in either direction. This is also why the causal-FIR features
  correlate at r ≈ 0.1 with SSR HG amplitude, versus ≈0.57 for the zero-phase run.
* The **decoder is not causal for any model**: ±200 ms stacking means 200 ms of look-ahead. With past-only context the SSR's short delay should be
  an advantage.
* The SSR and bandpass "phase" features measure different things. The bandpass phase is the Hilbert phase of a 0–20 Hz broadband signal; together with
  the LP amplitude it largely encodes the low-frequency voltage itself (A·cos φ is the low-passed signal; cf. LMP, Schalk 2007).
* **PAC as implemented is almost a copy of HG amplitude.** Within 50 ms the phase of a <20 Hz oscillator barely moves, so
  |mean(A_HG e^{iφ})| ≈ mean(A_HG)·|mean(e^{iφ})| ≈ mean(A_HG). Sub-01: PENDING. This explains the tiny PAC gain.
  The phase-resolved variant (Re/Im of the same complex mean = A_HG cos φ, A_HG sin φ) carries the amplitude–phase interaction
  that a linear decoder cannot construct; it is implemented as `pac_mode='complex'`.

### How to make the two frameworks comparable (implemented in `run_comparisons.py`)
1. **Symmetric ladders.** Each SSR feature set gets a bandpass counterpart that differs in one factor (paper Table 1):
   `HG_LPamp` (mirrors amplitude-only SSR, which also has LF amplitudes), `HG_LP` (mirrors +phase), `HG_LP_PAC`
   (**bandpass PAC**, computed exactly like SSR PAC), `HG_LP_PACc` / `SSR_pacc` (phase-resolved PAC), and `HG_LP_LMP` (LF-voltage control).
2. **Equal delays.** `*_dc` sets use the same FIR filters, centred (zero phase): identical magnitude response, no 199 ms lag.
3. **Same frequencies, different estimator.** `BPm_*`: FIR bandpass + Hilbert at each SSR oscillator's frequency **and** −3 dB
   bandwidth, with identical preprocessing, windows and feature definitions as SSR (`extractBandpassMatched`). SSR vs BPm isolates the
   Kalman estimator; HG/HG_LP vs BPm isolates subject-specific frequency selection.
4. **Online setting.** `--causal-stack` uses only past context (0 to −400 ms) for any feature set.
5. **Hyperparameters.** `--n-comp nested` selects the PLS components inside each training fold (exact and cheap: one fit per inner fold). At a
   common n_comp = 8 the published conclusions hold; at n_comp = 5, HG+LP vs HG is not significant (t = 2.02, p_BH = 0.22). So nested CV matters.

### First results, sub-01 only (n_comp = 8; preliminary, not in the paper)
PENDING

---

## 4. Text and references

* The Introduction is reframed around parity as the benchmark. The pipeline was built for bandpass envelopes, so matching it means that SSR's extra
  properties (sample-by-sample, low-delay, subject-specific amplitude/phase, native phase/PAC) come at no accuracy cost. New Methods subsections:
  "SSR oscillators as causal quadrature filters", "Bandpass features", and the matched comparison design (Table 1). New Results subsections on filters/delays
  and PAC redundancy, and a Discussion subsection on latency.
* The abstract is restructured into JNE's Objective / Approach / Main results / Significance format.
* Bibliography moved to `references.bib` (BibTeX, `iopart-num`). Corrections:
  * `Silva2024` combined the title/pages of **Metzger et al. 2023** with another author list. Now two entries: Silva et al. 2024 *Nat Rev Neurosci* (review) and Metzger et al. 2023.
  * `dataset` (Verwoert et al. 2022) had the **wrong author list**. The authors are Verwoert, Ottenhoff, Goulis, Colon, Wagner, Tousseyn, van Dijk, Kubben, Herff.
  * `Kearney2019` ("High gamma power features for speech decoding from EEG", JNE 16 066015) could not be found and appears not to exist. Replaced by Crone et al. 2006 and Ray & Maunsell 2011.
  * `Wairagkar2024` (bioRxiv) is now published: *Nature* 644, 145–152 (2025).
  * `Beck2023SOMATA`: preprint is 2022 and the authors are Beck, He, Gutierrez, Purdon (no "G.C. Hotan").
  * `Canolty2006`: pages 1626–1628 (not "5793").
  * `benjamini1995` and `nadeau2003` were cited but missing from the bibliography.
  * `Chang2010` (speech *perception* in STG) was cited for the speech-production network. Now Flinker 2015 and Bouchard 2013.
  * Added: Moses 2021, Metzger 2023, Willett 2023, Card 2024, Anumanchipalli 2019, Herff 2015/2019/2020, Angrick 2019, Crone 2006,
    Ray & Maunsell 2011, Giraud & Poeppel 2012, Proix 2022 (low- and cross-frequency iEEG features decode imagined speech: very relevant),
    Schalk 2007 (LMP), Tort 2010, Aru 2015, Beck 2018, Soulat 2022 (state-space PAC), Zrenner 2020, McIntosh & Sajda 2020,
    Kalman 1960, Anderson & Moore 1979, Wold 2001, Haufe 2014, Donoghue 2020.
* Figures for sub-02…sub-10 spectrograms are referenced but not in the repo; they render as labelled placeholder boxes. Drop the PNGs into
  `paper/images/` with the same names and they will be used.

## 5. Open TODOs (in order of importance)
1. Run the matched comparisons for all participants: `python run_comparisons.py --sets all --n-comp nested` (and with `--causal-stack`), then
   `stats_contrasts.py` with the `cmp_*` prefixes. Fill Table 1 results and the red TODOs.
2. Run `compare_filters.py --pts sub-01 … sub-10` with the saved SOMATA results to report filter bandwidths/delays, lags and PAC redundancy for all participants.
3. Regenerate Fig. 4B (SSR vs bandpass correlation) with the delay-compensated bandpass and commit the code. Recompute the error-track correlations from the
   final predictions. Fix the time axis of the zoomed spectrogram.
4. Decide on the 100 Hz notch; if changed, rerun SOMATA + all models.
5. Forward-chaining stability analysis (train early, test later) if parameter stability is to be claimed strongly.
6. Author list (R. Cavill?), affiliations, funding, CRediT, code DOI.
