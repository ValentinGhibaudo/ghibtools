import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from .signals import filter_sig
from .stats import med_mad

def detect_zerox(sig, show = False):
    """
    Detect zero-crossings ("zerox")

    ------
    inputs =
    - sig : numpy 1D array
    - show : plot figure showing rising zerox in red and decaying zerox in green (default = False)

    output =
    - pandas dataframe with index of rises and decays
    """
    rises, = np.where((sig[:-1] <=0) & (sig[1:] >0)) # detect where sign inversion from - to +
    decays, = np.where((sig[:-1] >=0) & (sig[1:] <0)) # detect where sign inversion from + to -
    if rises[0] > decays[0]: # first point detected has to be a rise
        decays = decays[1:] # so remove the first decay if is before first rise
    if rises[-1] > decays[-1]: # last point detected has to be a decay
        rises = rises[:-1] # so remove the last rise if is after last decay

    if show:
        fig, ax = plt.subplots(figsize = (15,5))
        ax.plot(sig)
        ax.plot(rises, sig[rises], 'o', color = 'r', label = 'rise')
        ax.plot(decays, sig[decays], 'o', color = 'g', label = 'decay')
        ax.set_title('Zero-crossing')
        ax.legend()
        plt.show()

    return pd.DataFrame.from_dict({'rises':rises, 'decays':decays}, orient = 'index').T


def get_cycle_features(zerox, sig, srate):
    """
    Compute respirations features from zerox df output of detect_zerox() function

    -----
    Inputs =
    - zerox : zerox df output of detect_zerox()
    - sig : respiration signal to compute inspi and expi amplitudes and volumes by cycle
    - srate : sampling rate of the respiratory signal

    Outputs = 
    - pandas dataframe containg respiration features for each cycle
    """
    features = []
    for i , row in zerox.iterrows(): # last cycle is probably not complete so it is removed in any case
        if i != zerox.index[-1]:
            start = int(row['rises'])
            transition = int(row['decays'])
            stop = int(zerox.loc[i+1, 'rises'])
            start_t = start / srate
            transition_t = transition / srate
            stop_t = stop / srate
            cycle_duration = stop_t - start_t
            inspi_duration = transition_t - start_t
            expi_duration = stop_t - transition_t
            cycle_freq = 1 / cycle_duration
            cycle_ratio = inspi_duration / cycle_duration
            inspi_amplitude = np.max(np.abs(sig[start:transition]))
            expi_amplitude = np.max(np.abs(sig[transition:stop]))
            cycle_amplitude = inspi_amplitude + expi_amplitude
            inspi_volume = np.trapz(np.abs(sig[start:transition]))
            expi_volume = np.trapz(np.abs(sig[transition:stop]))
            cycle_volume = inspi_volume + expi_volume
            second_volume = cycle_freq * cycle_volume
            features.append([start, transition , stop, start_t, transition_t, stop_t, cycle_duration,
                             inspi_duration, expi_duration, cycle_freq, cycle_ratio, inspi_amplitude,
                             expi_amplitude,cycle_amplitude, inspi_volume, expi_volume, cycle_volume, second_volume])
    df_features = pd.DataFrame(features, columns = ['start','transition','stop','start_time','transition_time',
                                                    'stop_time','cycle_duration','inspi_duration','expi_duration','cycle_freq','cycle_ratio',
                                                    'inspi_amplitude','expi_amplitude','cycle_amplitude','inspi_volume','expi_volume','cycle_volume','second_volume'])
    return df_features

def get_resp_features(rsp, srate):
    """
    High level function that directly return respiration features from respiratory signal based on zerox cycle detection

    ---
    Input = 
    - rsp : respiratory signal (1d np vector)
    - srate : sampling rate 

    Output = 
    - pandas dataframe containg respiration features for each cycle
    """
    zerox = detect_zerox(rsp)
    features = get_cycle_features(zerox, rsp, srate)
    return features

def robust_zscore(sig): # center by median and reduce by std
    return (sig - np.median(sig)) / np.std(sig)

def robust_mad_scaling(sig): # center by median and reduce by mad
    med, mad = med_mad(sig)
    return (sig - med) / mad

