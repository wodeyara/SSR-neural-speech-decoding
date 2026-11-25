# SingleWordProductionDutch

Neural speech decoding pipeline using intracranial EEG data. This project implements State-Space Phase Estimation (SSPE) for extracting neural features and reconstructing speech spectrograms from brain activity.

## Dataset
The project uses intracranial EEG data from [here](https://osf.io/nrgx6/) described 
in this [article](https://www.nature.com/articles/s41597-022-01542-9).

## Dependencies
The scripts require Python >= 3.6 and the following packages
* [numpy](http://www.numpy.org/)
* [scipy](https://www.scipy.org/scipylib/index.html)
* [scikit-learn](https://scikit-learn.org/stable/)
* [pandas](https://pandas.pydata.org/) 
* [pynwb](https://github.com/NeurodataWithoutBorders/pynwb)
* [somata](https://github.com/ckyrkou/somata) - Oscillator detection

## Repository content
To recreate the experiments, run the following scripts.
* __notebookfunctions.py__: Utility functions including `run_pipeline()` to execute the complete analysis pipeline
* __extract_features.py__: Reads in the iBIDS dataset and extracts features which are then saved to './features'
* __phaseEM.py__: Causal phase estimation using state-space oscillator models
* __reconstruction_minimal.py__: Reconstructs the spectrogram from the neural features in a 10-fold cross-validation and synthesizes the audio using the Method described by Griffin and Lim.
* __reconstuctWave.py__: Synthesizes an audio waveform using the method described by Griffin-Lim
* __MelFilterBank.py__: Applies mel filter banks to spectrograms.
* __ssr_filter.py__: Steady-state (LTI) form of the SSR Kalman filter: frequency response, -3 dB bandwidth and group delay of every oscillator, and a fast vectorised SSR filter (`extractSSPE(..., engine='steady_state')`) that is identical to the per-sample Kalman filter after the ~1 s gain transient.
* __compare_filters.py__: Compares SSR oscillators with the bandpass filters (passbands, group delays, lag between SSR and bandpass high-gamma features, PAC redundancy, streaming cost); `--aggregate` summarises all participants.
* __run_somata.py__: Resumable SOMATA initialisation per participant and channel; the results used in the paper are in `somata_results/*.json`.
* __run_comparisons.py__: Matched SSR-versus-bandpass comparison ladder (bandpass PAC, HG + LP amplitude, delay-compensated filters, bandpass filters matched to the SSR oscillators, past-only temporal context, nested selection of PLS components).
* __stats_contrasts.py__: Planned-contrast statistics used in the paper (Fisher z, paired t-tests over participants with Benjamini-Hochberg FDR within contrast families, 95% CIs of the differences, corrected resampled t-tests within participants); `--preset main` for the main analysis.
* __merge_results.py__, __summarize_main.py__, __plot_main.py__: combine per-participant results of the main analysis, write the results table and draw the main figure.

## Paper
The manuscript lives in `paper/` (`latexmk -pdf main.tex`). `paper/REVIEW_NOTES.md` lists the code review, the verification of the numbers in the manuscript and the open TODOs (marked in red in the PDF).

## Jupyter Notebooks
The jupyter notebook `report_figures.ipynb` can be used to reproduce the results from the paper. (The rest of the notebooks were for exploratory analysis, and are a bit chaotic)

## Usage
Make sure to set the correct paths for 
`path_bids` (location to dataset), `path_somata` (folder in which you want to store somata results) and `path_result` (folder in which you want to store results).
They can also be set with the environment variables `SSR_PATH_BIDS`, `SSR_PATH_SOMATA`, `SSR_PATH_RESULT` and `SSR_PATH_FEATURES` (feature folder, default `./features`), e.g.

```
export SSR_PATH_BIDS=/data/SingleWordProductionDutch-iBIDS SSR_PATH_SOMATA=$PWD/somata_results
# main analysis of the paper (21 configurations per participant, nested PLS; ~4-6 h and up to 14 GB per participant)
SSR_PATH_RESULT=results/sub-01 python run_comparisons.py --pts sub-01 --plan main
python merge_results.py --in results --out results/all
python stats_contrasts.py --results results/all --preset main --latex paper/tables/main_contrasts.tex
python summarize_main.py --results results/all --latex paper/tables/main_results.tex
python plot_main.py --results results/all --out paper/images/main_results
python compare_filters.py --pts sub-01   # filter view of the SSR models
```
The reconstruction correlations and chosen PLS components of the main analysis are in `results/sub-XX/` and, merged, in `results/all/` (chance-level `randomResults` files are not committed because of their size).
