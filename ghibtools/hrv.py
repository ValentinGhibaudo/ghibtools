import numpy as np
import pandas as pd
import neurokit2 as nk
import matplotlib.pyplot as plt
from .signals import time_vector

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
    
    return clean, fci

def get_hrv_metrics(ecg, srate, show = False):
    peaks, info = nk.ecg_peaks(ecg, sampling_rate=srate, correct_artifacts=True)
    if show: 
        pics = info['ECG_R_Peaks']
        fig, ax = plt.subplots()
        ax.plot(ecg, label = 'ecg')
        ax.plot(pics, ecg[pics], 'x', label = 'peaks')
        plt.show()

    return nk.hrv(peaks).dropna(axis = 'columns')

def get_rsa(ecg, rsp, srate, show = False):
    ecg_signals, info = nk.ecg_process(ecg, sampling_rate = srate)
    rsp_signals, _ = nk.rsp_process(rsp, sampling_rate=srate)
    rsa = nk.hrv_rsa(ecg_signals, rsp_signals, info, sampling_rate=srate)
    if show:
        nk.signal_plot([ecg_signals["ECG_Rate"], rsp_signals["RSP_Rate"], rsa], standardize=True)
    return pd.DataFrame.from_dict(rsa, orient='index').T

def ecg_peaks(ecg, srate, show = False):
    _,info = nk.ecg_peaks(ecg, sampling_rate=srate)

    if show: 
        pics = info['ECG_R_Peaks']
        fig, ax = plt.subplots()
        ax.plot(ecg, label = 'ecg')
        ax.plot(pics, ecg[pics], 'x', label = 'peaks')
        plt.show()
        
    return info['ECG_R_Peaks']

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
    return 

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
