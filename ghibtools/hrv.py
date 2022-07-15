import numpy as np
import pandas as np
import neurokit2 as nk

def ecg_to_hrv(ecg, srate, show = False, inverse_sig = True):

    if inverse_sig:
        ecg = -ecg

    if show:
        plt.figure(figsize=(15,10))
        ecg_signals, info_ecg = nk.ecg_process(ecg, sampling_rate=srate, method='neurokit')
        nk.ecg_plot(ecg_signals, rpeaks=info_ecg, sampling_rate=srate, show_type='default')
    
    clean = nk.ecg_clean(ecg, sampling_rate=srate, method='neurokit')
    peaks, info_ecg = nk.ecg_peaks(clean, sampling_rate=srate,method='neurokit', correct_artifacts=True)
    
    R_peaks = info_ecg['ECG_R_Peaks'] # get R time points
    diff_R_peaks = np.diff(R_peaks) 
    x = vector_time
    xp = R_peaks[1::]/srate
    fp = diff_R_peaks
    interpolated_hrv = np.interp(x, xp, fp, left=None, right=None, period=None) / srate
    fci = 60 / interpolated_hrv
    
    return clean, fci

def ecg_peaks(ecg, srate):
    _,info = nk.ecg_peaks(ecg, sampling_rate=srate)
    return info['ECG_R_Peaks']

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
