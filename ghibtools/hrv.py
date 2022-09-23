import numpy as np
import pandas as pd
import neurokit2 as nk
import matplotlib.pyplot as plt
from .signals import time_vector, spectre
from scipy import signal

def ecg_to_hrv(ecg, srate, show = False, inverse_sig = False):

    if inverse_sig:
        ecg = -ecg
    
    clean = nk.ecg_clean(ecg, sampling_rate=srate, method='neurokit')
    peaks, info_ecg = nk.ecg_peaks(clean, sampling_rate=srate,method='neurokit', correct_artifacts=True)
    
    R_peaks = info_ecg['ECG_R_Peaks'] # get R time points

    if show: 
        fig, ax = plt.subplots()
        ax.plot(ecg, label = 'ecg')
        ax.plot(R_peaks, ecg[R_peaks], 'x', label = 'peaks')
        plt.show()

    diff_R_peaks = np.diff(R_peaks) 
    x = time_vector(ecg, srate)
    xp = R_peaks[1::]/srate
    fp = diff_R_peaks
    interpolated_hrv = np.interp(x, xp, fp, left=None, right=None, period=None) / srate
    fci = 60 / interpolated_hrv
    
    return fci

def get_hrv_metrics(ecg, srate, kind = 'all', show = False):
    peaks, info = nk.ecg_peaks(ecg, sampling_rate=srate, correct_artifacts=True)
    if show: 
        pics = info['ECG_R_Peaks']
        fig, ax = plt.subplots()
        ax.plot(ecg, label = 'ecg')
        ax.plot(pics, ecg[pics], 'x', label = 'peaks')
        plt.show()

    if kind == 'all':
        return nk.hrv(peaks, srate).dropna(axis = 'columns')
    elif kind == 'time':
        return nk.hrv_time(peaks, srate).dropna(axis = 'columns')
    elif kind == 'freq':
        return nk.hrv_frequency(peaks, srate).dropna(axis = 'columns')
    elif kind == 'time_and_freq':
        return pd.concat([nk.hrv_time(peaks, srate).dropna(axis = 'columns'), nk.hrv_frequency(peaks, srate).dropna(axis = 'columns')], axis = 1)


def get_rsa(ecg, rsp, srate, show = False):
    ecg_signals, info = nk.ecg_process(ecg, sampling_rate = srate)
    rsp_signals, _ = nk.rsp_process(rsp, sampling_rate=srate)
    rsa = nk.hrv_rsa(ecg_signals, rsp_signals, info, sampling_rate=srate)
    if show:
        nk.signal_plot([ecg_signals["ECG_Rate"], rsp_signals["RSP_Rate"], rsa], standardize=True)
    return pd.DataFrame.from_dict(rsa, orient='index').T

def ecg_peaks(ecg, srate, method = 'neurokit', show = False):
    if method == 'neurokit':
        _,info = nk.ecg_peaks(ecg, sampling_rate=srate)
        peaks = info['ECG_R_Peaks']

    elif method == 'homemade':
        peaks,_ = signal.find_peaks(ecg, distance=1000/1.5)

    if show: 
        fig, ax = plt.subplots()
        ax.plot(ecg, label = 'ecg')
        ax.plot(peaks, ecg[peaks], 'x', label = 'peaks')
        plt.show()
        
    return peaks

def manual_peak_correction(peaks, to_remove=None, to_add=None, sig=None, error_size = 50):

    if not to_remove is None:
        remove_peaks = []
        for remove in to_remove:
            remove_peak = peaks[(peaks > remove - error_size) & (peaks < remove + error_size)]
            remove_peaks.append(int(remove_peak))

        corrected_peaks = peaks[~np.isin(peaks, remove_peaks)]
    else:
        corrected_peaks = peaks

    if not to_add is None:
        indices_where_inserting = [np.where(corrected_peaks > add)[0][0] for add in to_add if add < corrected_peaks[-1]]
        corrected_peaks_added = np.insert(corrected_peaks, indices_where_inserting, to_add)
    else:
        corrected_peaks_added = corrected_peaks

    if not sig is None:
        fig, ax = plt.subplots()
        ax.plot(sig, label = 'sig')
        ax.plot(peaks, sig[peaks], 'x', color = 'orange')
        
        if not to_remove is None:
            removed = peaks[~np.isin(peaks, corrected_peaks_added)]
            ax.plot(removed, sig[removed], 'o' , color = 'r', label = 'removed')
        if not to_add is None:
            ax.plot(to_add, sig[to_add], 'o' , color = 'g', label = 'added')

        ax.legend()
        plt.show()

    return corrected_peaks_added

def peaks_to_RRI(peaks, srate):
    peaks_time = peaks / srate

    RRIs = []
    for i, time in enumerate(peaks_time):
        if i != 0:
            RRIs.append(time - peaks_time[i-1])
    return np.array(RRIs)*1000

def RRI_to_successive_differences(RRIs):
    successive_differences = []
    for i, RRI in enumerate(RRIs):
        if i != 0:
            successive_differences.append(RRIs[i-1] - RRI)
    return successive_differences

def MeanNN(RRIs):
    return np.mean(RRIs)

def SDNN(RRIs):
    return np.std(RRIs)

def RMSSD(RRIs):
    square_of_successive_differences = []
    for i, RRI in enumerate(RRIs):
        if i != 0:
            square_of_successive_differences.append((RRIs[i-1] - RRI)**2)
        
    return np.sqrt(np.mean(square_of_successive_differences))

def pNN50(RRIs):
    return (sum(RRIs) > 50) / RRIs.size

def freq_domain(ecg, srate):
    fci = ecg_to_hrv(ecg, srate)
    f, Pxx = spectre(fci, srate, lowest_freq=0.04)
    lf = np.trapz(Pxx[(f > 0.04) & (f < 0.15)])
    hf = np.trapz(Pxx[(f > 0.15) & (f < 0.4)])
    lfhf = lf / hf
    return {'LF':lf, 'HF':hf, 'LFHF':lfhf}

def get_hrv_metrics_homemade(ecg, srate, show = False):
    peaks = ecg_peaks(ecg, srate)
    rri = peaks_to_RRI(peaks, srate)
    mean_nn = MeanNN(rri)
    sdnn = SDNN(rri)
    rmssd = RMSSD(rri)
    pnn50 = pNN50(rri)
    freqs = freq_domain(ecg, srate)
    data = [mean_nn , sdnn , rmssd , pnn50, freqs['LF'], freqs['HF'], freqs['LFHF']]
    return pd.Series(data=data, index = ['MeanNN','SDNN','RMSSD','pNN50','LF','HF','LFHF']).to_frame().T





