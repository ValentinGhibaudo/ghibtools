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