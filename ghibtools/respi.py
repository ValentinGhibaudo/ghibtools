import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from .signals import filter_sig

def detect_zerox(sig, show = False):
    rises = []
    decays = []
    for i in range(sig.size):
        if i != 0:
            if np.sign(sig[i]) != np.sign(sig[i-1]):
                if sig[i] > 0:
                    rises.append(i)
                elif sig[i] < 0:
                    decays.append(i)

    if show:
        fig, ax = plt.subplots(figsize = (15,5))
        ax.plot(sig)
        ax.plot(rises, sig[rises], 'o', color = 'r', label = 'rise')
        ax.plot(decays, sig[decays], 'o', color = 'g', label = 'decay')
        ax.set_title('Zero-crossing')
        ax.legend()
        plt.show()

    return pd.DataFrame.from_dict({'rises':rises, 'decays':decays}, orient = 'index').T

def get_cycle_features(zerox, srate, show = False):
    features = []
    for i , row in zerox.iterrows():
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
            features.append([start, transition , stop, start_t, transition_t, stop_t, cycle_duration, inspi_duration, expi_duration, cycle_freq, cycle_ratio])
    df_features = pd.DataFrame(features, columns = ['start','transition','stop','start_time','transition_time','stop_time', 'cycle_duration','inspi_duration','expi_duration','cycle_freq','cycle_ratio'])

    if show:
        fig, ax = plt.subplots()
        ax.hist(df_features['cycle_freq'], bins = 100)
        ax.set_ylabel('n_cycles')
        ax.set_xlabel('Freq [Hz]')
        median_cycle = df_features['cycle_freq'].median()
        ax.axvline(median_cycle, linestyle = '--', color='m')
        ax.set_title(f'Median freq : {round(median_cycle, 2)}')
        plt.show()
    return df_features

def get_resp_features(rsp, srate, manual_baseline_correction = 0, low = 0.05, high=0.8, show = False):
    sig = rsp - np.mean(rsp)
    sig_filtered = filter_sig(sig, srate, low, high) + manual_baseline_correction

    if show:
        fig, ax = plt.subplots()
        ax.plot(sig, label = 'raw')
        ax.plot(sig_filtered, label = 'filtered')
        ax.set_title('Filtering')
        ax.legend()
        plt.show()
    
    zerox = detect_zerox(sig_filtered, show)
    features = get_cycle_features(zerox, srate, show)

    return features