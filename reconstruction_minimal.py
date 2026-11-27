import os

import numpy as np
import scipy.io.wavfile as wavfile
from scipy.stats import pearsonr
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.cross_decomposition import PLSRegression
from extract_features import *

import reconstructWave as rW
import MelFilterBank as mel


def createAudio(spectrogram, audiosr=16000, winLength=0.05, frameshift=0.01):
    """
    Create a reconstructed audio wavefrom
    
    Parameters
    ----------
    spectrogram: array
        Spectrogram of the audio
    sr: int
        Sampling rate of the audio
    windowLength: float
        Length of window (in seconds) in which spectrogram was calculated
    frameshift: float
        Shift (in seconds) after which next window was extracted
    Returns
    ----------
    scaled: array
        Scaled audio waveform
    """
    mfb = mel.MelFilterBank(int((audiosr*winLength)/2+1), spectrogram.shape[1], audiosr)
    nfolds = 10
    hop = int(spectrogram.shape[0]/nfolds)
    rec_audio = np.array([])
    for_reconstruction = mfb.fromLogMels(spectrogram)
    
    for w in range(0,spectrogram.shape[0],hop):
        spec = for_reconstruction[w:min(w+hop,for_reconstruction.shape[0]),:]
        
        rec = rW.reconstructWavFromSpectrogram(spec,spec.shape[0]*spec.shape[1],fftsize=int(audiosr*winLength),overlap=int(winLength/frameshift))
        rec_audio = np.append(rec_audio,rec)
    scaled = np.int16(rec_audio/np.max(np.abs(rec_audio)) * 32767)
    return scaled


def _pls_predict_k(pls, X, k):
    """
    Predict with only the first k components of a fitted PLSRegression.
    NIPALS extracts components sequentially (with deflation), so the first k components of a
    model fitted with n_components >= k are identical to those of a k-component model.
    """
    Xs = (X - pls._x_mean) / pls._x_std
    coef = (pls.x_rotations_[:, :k] @ pls.y_loadings_[:, :k].T) * pls._y_std
    return Xs @ coef + pls._y_mean


def _mean_corr(Y, Yhat):
    Y = Y - Y.mean(axis=0); Yhat = Yhat - Yhat.mean(axis=0)
    r = np.sum(Y * Yhat, axis=0) / np.sqrt(np.sum(Y**2, axis=0) * np.sum(Yhat**2, axis=0))
    return np.nanmean(r)


def select_n_comp_pls(X, Y, grid, inner_folds=5):
    """
    Nested selection of the number of PLS components using contiguous (non-shuffled) inner
    folds of the TRAINING data only. Returns the best n_components and the inner-CV score curve.
    """
    grid = sorted(int(g) for g in grid)
    scores = np.zeros((inner_folds, len(grid)))
    for i, (tr, te) in enumerate(KFold(inner_folds, shuffle=False).split(X)):
        pls = PLSRegression(n_components=max(grid), max_iter=1000, copy=False).fit(X[tr], Y[tr])
        for j, k in enumerate(grid):
            scores[i, j] = _mean_corr(Y[te], _pls_predict_k(pls, X[te], k))
    curve = scores.mean(axis=0)
    return grid[int(np.argmax(curve))], curve


def reconstruct(pts, model = 'PLS', n_comp = 9, feat_suffix='_feat.npy', unstacked =False, saveAs=None, synthesize=False,
                causal_stack=False, inner_folds=5, fold_n_comp=None):
    """
    Reconstruct mel spectrograms from neural features using regression models.
    
    This function trains a regression model (PLS, Linear Regression, or Lasso) to
    predict mel spectrograms from neural features using cross-validation. It evaluates
    reconstruction quality using Pearson correlation coefficients and compares against random baselines.
    Note that the function drops the first 10 seconds of data to avoid using data the SSPE model was initialized on.

    Parameters
    ----------
    pts : list of str or None
        List of participant identifiers. If None, uses all 10 participants.
    model : str, optional
        Regression model to use: 'PLS', 'LR', or 'Lasso'. Default is 'PLS'.
    n_comp : int or list of int, optional
        Number of components for PLS regression. Default is 9.
        The paper uses 6 (HG bandpass), 9 (HG + LP bandpass) and 8 (all SSR feature sets),
        selected on the outer test folds (not nested).
        If a list/tuple of ints is given, the number of components is selected separately
        in every outer fold by an inner contiguous cross-validation on the training data
        only (nested CV); the chosen values are saved as {saveAs}nComp.npy.
    feat_suffix : str, optional
        Suffix for feature files. Default is '_feat.npy'.
    unstacked : bool, optional
        If True, the saved features are unstacked and temporal context is stacked here,
        after dropping the initialisation segment. Default is False.
    causal_stack : bool, optional
        Only with unstacked=True. If True, stack only past context frames (0 to -400 ms)
        instead of +-200 ms, so the decoder does not use future neural data.
    inner_folds : int, optional
        Number of inner folds for nested selection of n_comp (if n_comp is a list).
    fold_n_comp : array (n_participants, n_folds) of int, optional
        Number of PLS components to use in every outer fold, e.g. the {saveAs}nComp.npy of an earlier nested run.
        Reproduces that run's predictions without repeating the inner selection (n_comp is then ignored).
    saveAs : str or None, optional
        Prefix for saving results. If None or 'dont', returns results without saving.
    synthesize : bool, optional
        If True, synthesize audio from reconstructed spectrograms. Default is False.
        
    Returns
    -------
    allRes : array (n_participants, n_folds, n_mel_bins)
        Correlation coefficients for each participant, fold, and mel bin
    coef : array
        Model coefficients
        
    Notes
    -----
    This function uses global variables for paths:
    - feat_path: Path to feature files
    - result_path: Path to save results
    """
    feat_path = os.environ.get('SSR_PATH_FEATURES', r'./features')
    result_path = os.environ.get('SSR_PATH_RESULT', r'./results')
    if pts == None:
        pts = ['sub-01', 'sub-02', 'sub-03', 'sub-04', 'sub-05','sub-06','sub-07', 'sub-08', 'sub-09', 'sub-10']

    winLength = 0.05
    frameshift = 0.01
    audiosr = 16000

    nfolds = 10
    kf = KFold(nfolds,shuffle=False)

    #seconds_to_drop = 2.0 #to allow sspe model to 'warm-up'
    seconds_to_drop = 10.0 #to not include the data that we initialized the parameters with
    frames_to_drop = int(seconds_to_drop / frameshift)

    #pca = PCA()
    nested = isinstance(n_comp, (list, tuple, np.ndarray))
    chosen_n_comp = np.zeros((len(pts), nfolds), dtype=int)
    if model == 'PLS': 
        # copy=False: PLS centres/scales the (temporary) training matrix in place instead of copying it
        est = PLSRegression(n_components=(max(n_comp) if nested else n_comp), max_iter=1000, copy=False)
    elif model == 'LR':
        est = LinearRegression(n_jobs=1)
    elif model == 'Lasso':
        est = Lasso()
    else:
        print('model name not recognized, using LR')
        est = LinearRegression(n_jobs=1)


    #Initialize empty matrices for correlation results, randomized contols and amount of explained variance
    allRes = np.zeros((len(pts),nfolds,23))
    explainedVariance = np.zeros((len(pts),nfolds))
    numRands = 1000
    randomControl = np.zeros((len(pts),numRands, 23))

    for pNr, pt in enumerate(pts):
        #Load the data
        data = np.load(os.path.join(feat_path,f'{pt}{feat_suffix}'))
        
        # Slice data along the time dimension (axis 0)
        data = data[frames_to_drop:, :] #this works now because i know we have unstacked data, else, change

        if unstacked:
            data = stackFeatures(data, causal=causal_stack)
    
        #Check for NaNs or Infinities
        nan_count = np.isnan(data).sum()
        inf_count = np.isinf(data).sum()
        if nan_count > 0 or inf_count > 0:
            print(f"!Warning: {pt} has {nan_count} NaNs and {inf_count} Infs in features.")
            
            #Replace NaNs with 0 and Infs with a very large/small number or 0
            data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

        spectrogram = np.load(os.path.join(feat_path,f'{pt}_spec.npy'))
        spectrogram = spectrogram[frames_to_drop:, :]
        #adjust length (differences might occur due to rounding in the number of windows)
        if spectrogram.shape[0] != data.shape[0]:
            tLen = min(spectrogram.shape[0], data.shape[0])
            print(f'{pt}: cropping features {data.shape[0]} and spectrogram {spectrogram.shape[0]} to {tLen} frames')
            spectrogram = spectrogram[:tLen, :]
            data = data[:tLen, :]
        #labels = np.load(os.path.join(feat_path,f'{pt}_procWords.npy'))
        #featName = np.load(os.path.join(feat_path,f'{pt}_feat_names.npy'))
        
        #Initialize an empty spectrogram to save the reconstruction to
        rec_spec = np.zeros(spectrogram.shape)
        #Save the correlation coefficients for each fold
        rs = np.zeros((nfolds,spectrogram.shape[1]))
        for k,(train, test) in enumerate(kf.split(data)):
            #Z-Normalize with mean and std from the training data
            mu=np.mean(data[train,:],axis=0)
            std=np.std(data[train,:],axis=0)
            std[std == 0] = 1.0  # Prevent division by zero

            # (in place, to avoid extra copies of the large stacked feature matrix; same values as (x-mu)/std)
            trainData = data[train,:]
            trainData -= mu
            trainData /= std
            testData = data[test,:]
            testData -= mu
            testData /= std

            # Convert all NaNs to 0.0 before passing to PCA
            trainData = np.nan_to_num(trainData, nan=0.0, copy=False)

            ''''
            #Fit PCA to training data
            pca.fit(trainData)
            #Get percentage of explained variance by selected components
            explainedVariance[pNr,k] =  np.sum(pca.explained_variance_ratio_[:numComps])
            #Tranform data into component space
            trainData=np.dot(trainData, pca.components_[:numComps,:].T)
            testData = np.dot(testData, pca.components_[:numComps,:].T)
            '''

            #Fit the regression model
            if fold_n_comp is not None and model == 'PLS':
                chosen_n_comp[pNr, k] = int(fold_n_comp[pNr][k])
                est = PLSRegression(n_components=chosen_n_comp[pNr, k], max_iter=1000, copy=False)
            elif nested and model == 'PLS':
                best_k, _ = select_n_comp_pls(trainData, spectrogram[train, :], n_comp, inner_folds)
                chosen_n_comp[pNr, k] = best_k
                est = PLSRegression(n_components=best_k, max_iter=1000, copy=False)
    
            est.fit(trainData, spectrogram[train, :])
            del trainData
            #Predict the reconstructed spectrogram for the test data
            rec_spec[test, :] = est.predict(testData)

            #Evaluate reconstruction of this fold
            for specBin in range(spectrogram.shape[1]):
                if np.any(np.isnan(rec_spec)):
                    print('%s has %d broken samples in reconstruction' % (pt, np.sum(np.isnan(rec_spec))))
                r, p = pearsonr(spectrogram[test, specBin], rec_spec[test, specBin])
                rs[k,specBin] = r

        #Show evaluation result
        print('%s has mean correlation of %f' % (pt, np.mean(rs)))
        allRes[pNr,:,:]=rs

        #Estimate random baseline
        for randRound in range(numRands):
            #Choose a random splitting point at least 10% of the dataset size away
            splitPoint = np.random.choice(np.arange(int(spectrogram.shape[0]*0.1),int(spectrogram.shape[0]*0.9)))
            #Swap the dataset on the splitting point 
            shuffled = np.concatenate((spectrogram[splitPoint:,:],spectrogram[:splitPoint,:]))
            
            #Calculate the correlations
            '''
            for specBin in range(spectrogram.shape[1]):
                if np.any(np.isnan(rec_spec)):
                    print('%s has %d broken samples in reconstruction' % (pt, np.sum(np.isnan(rec_spec))))
                r, p = pearsonr(spectrogram[:,specBin], shuffled[:,specBin])
                randomControl[pNr, randRound,specBin]=r
'''
            # Vectorized correlation:
            # This calculates all 23 bins in one go
            # (Subtract mean, divide by std, then dot product)
            s_ms = shuffled - shuffled.mean(axis=0)
            r_ms = spectrogram - spectrogram.mean(axis=0)
            r = np.sum(s_ms * r_ms, axis=0) / (np.sqrt(np.sum(s_ms**2, axis=0) * np.sum(r_ms**2, axis=0)))
            randomControl[pNr, randRound, :] = r


        #Save reconstructed spectrogram
        os.makedirs(os.path.join(result_path), exist_ok=True)
        if saveAs:
            np.save(os.path.join(result_path,f'{pt}_{saveAs}_predicted_spec.npy'), rec_spec)
        
        if synthesize:
            #Synthesize waveform from spectrogram using Griffin-Lim
            print(f"Max value in rec_spec: {np.max(rec_spec)}")
            if np.max(rec_spec) > 100: # Threshold depends on your scaling
                print("Warning: Predicted spectrogram values are too high!")
                rec_spec = np.clip(rec_spec, -100, 50) # Forces values into a mathematically safe range
            reconstructedWav = createAudio(rec_spec,audiosr=audiosr,winLength=winLength,frameshift=frameshift)
            wavfile.write(os.path.join(result_path,f'{pt}_predicted.wav'),int(audiosr),reconstructedWav)

            #For comparison synthesize the original spectrogram with Griffin-Lim
            origWav = createAudio(spectrogram,audiosr=audiosr,winLength=winLength,frameshift=frameshift)
            wavfile.write(os.path.join(result_path,f'{pt}_orig_synthesized.wav'),int(audiosr),origWav)

    #Save results in numpy arrays  
    if saveAs:
        np.save(os.path.join(result_path, saveAs + 'linearResults.npy'),allRes)
        np.save(os.path.join(result_path, saveAs + 'randomResults.npy'),randomControl)
        np.save(os.path.join(result_path, saveAs + 'explainedVariance.npy'),explainedVariance)
        np.save(os.path.join(result_path, saveAs + 'est_coef.npy'), est.coef_)
        if nested:
            np.save(os.path.join(result_path, saveAs + 'nComp.npy'), chosen_n_comp)
    return allRes, est.coef_



if __name__=="__main__":
    reconstruct(None)