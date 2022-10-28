import numpy as np
import pandas as pd
import neurokit2 as nk
import matplotlib.pyplot as plt
from .signals import time_vector, spectre
from scipy import signal
from scipy.interpolate import interp1d

def ecg_to_hrv(ecg, srate, show = False, inverse_sig = False, method = 'scipy'):

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

    rris = np.diff(R_peaks) 

    t = time_vector(ecg, srate)

    if method == 'numpy':
        xp = R_peaks[1:]/srate
        fp = rris
        interpolated_hrv = np.interp(t, xp, fp, left=None, right=None, period=None) / srate
        fci = 60 / interpolated_hrv

    elif method == 'scipy':
        x = t[R_peaks][1:]
        y = rris
        f = interp1d(x, y, fill_value="extrapolate", kind = 'cubic')
        xnew = t
        ynew = 60 * srate / f(xnew)
        fci = ynew
    
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



def pqrst_cycle(ecg, srate):
    signals, info = nk.ecg_process(-ecg, srate)
    columns = [
        'ECG_P_Onsets',
        'ECG_R_Onsets',
        'ECG_Q_Peaks',
        'ECG_R_Peaks',
        'ECG_S_Peaks',
        'ECG_T_Offsets',
    ]

    r, = np.nonzero(signals['ECG_R_Peaks'].values)
    # print(r.size)

    cycles = pd.DataFrame(index=np.arange(r.size), columns=[col.replace('ECG_', '') for col in columns])
    cycles['R_Peaks'] = r / srate

    for col in columns:
        if col == 'ECG_R_Peaks':
            continue
        col2 = col.replace('ECG_', '')
        ind, = np.nonzero(signals[col].values)

        pre = columns.index(col) < columns.index('ECG_R_Peaks')

        if pre:
            ind = ind[ind<r[-1]]

            # first cycle
            valid, = np.nonzero(ind < r[0])
            if valid.size == 1:
                cycles.at[0, col2] = ind[valid[0]] / srate

            for c in range(1, r.size):
                valid, = np.nonzero((ind > r[c -1 ]) & (ind < r[c]))
                if valid.size == 1:
                    cycles.at[c, col2] = ind[valid[0]] / srate
        else:
            ind = ind[ind>r[0]]


            for c in range(r.size -1):
                valid, = np.nonzero((ind > r[c]) & (ind < r[c + 1]))
                if valid.size == 1:
                    cycles.at[c, col2] = ind[valid[0]] / srate

            # last cycle
            valid, = np.nonzero(ind > r[-1])
            if valid.size == 1:
                cycles.at[r.size - 1, col2] = ind[valid[0]] / srate

    return cycles




def segment_variability(participant,bloc, start_letter, stop_letter, start_pattern, stop_pattern):
    ecg = -da.loc[participant,bloc,:].values
    srate = 500
    signals, info = nk.ecg_process(ecg, sampling_rate=srate)
    sig = signals['ECG_Clean']
    init = signals[f'ECG_{start_letter}_{start_pattern}']
    end = signals[f'ECG_{stop_letter}_{stop_pattern}']

    init_timings = np.where(init==1)[0] / srate
    end_timings = np.where(end==1)[0] / srate

    if not start_letter == stop_letter:
        concat = []
        for i in range(min([init_timings.size,end_timings.size])):
            init_timing = init_timings[i]
            end_timing = end_timings[i]
            segment_i = end_timing - init_timing
            concat.append(segment_i)
        mean = np.mean(concat)
        print(f'{start_letter}{stop_letter} moyen :' , round(mean,2))
    else:
        concat = []
        for i in range(min([init_timings.size,end_timings.size])-1):
            init_timing = init_timings[i]
            end_timing = end_timings[i+1]
            segment_i = end_timing - init_timing
            concat.append(segment_i)
        mean = np.mean(concat)
        print(f'{start_letter}{stop_letter} moyen :' , round(mean,2))
    return concat

def segment_variability_mean(participant,bloc, start_letter, stop_letter, start_pattern, stop_pattern):
    ecg = -da.loc[participant,bloc,:].values
    srate = 500
    signals, info = nk.ecg_process(ecg, sampling_rate=srate)
    sig = signals['ECG_Clean']
    init = signals[f'ECG_{start_letter}_{start_pattern}']
    end = signals[f'ECG_{stop_letter}_{stop_pattern}']

    init_timings = np.where(init==1)[0] / srate
    end_timings = np.where(end==1)[0] / srate
    if not start_letter == stop_letter:
        concat = []
        for i in range(min([init_timings.size,end_timings.size])):
            init_timing = init_timings[i]
            end_timing = end_timings[i]
            segment_i = end_timing - init_timing
            concat.append(segment_i)
        mean = np.mean(concat)
        # print(f'{start_letter}{stop_letter} moyen :' , round(mean,2))
    else:
        concat = []
        for i in range(min([init_timings.size,end_timings.size])-1):
            init_timing = init_timings[i]
            end_timing = end_timings[i+1]
            segment_i = end_timing - init_timing
            concat.append(segment_i)
        mean = np.mean(concat)
        # print(f'{start_letter}{stop_letter} moyen :' , round(mean,2))
    return mean


def corr_rr_st(da, participant, bloc):
    qt = segment_variability(da, participant,bloc, start_letter='Q', stop_letter='T', start_pattern='Peaks', stop_pattern='Offsets')
    rr = segment_variability(da, participant,bloc, start_letter='R', stop_letter='R', start_pattern='Peaks', stop_pattern='Peaks')
    df = pd.DataFrame()
    df.insert(0,'rr',rr)
    df.insert(0,'qt',qt[0:len(rr)])
    rcorr = df.rcorr()
    df.plot()
    return rcorr

def participants_analysis():
    participants = ['sub01','sub02','sub03','sub04','sub05','sub06','sub07','sub08','sub09']
    blocs = ['coherence','free','confort']
    rows = []
    for participant in participants:
        print(participant)
        for bloc in blocs:
            print(bloc)
            pr = segment_variability_mean(participant,bloc, start_letter='P', stop_letter='R', start_pattern='Onsets', stop_pattern='Onsets')
            qt = segment_variability_mean(participant,bloc, start_letter='Q', stop_letter='T', start_pattern='Peaks', stop_pattern='Offsets')
            st = segment_variability_mean(participant,bloc, start_letter='S', stop_letter='T', start_pattern='Peaks', stop_pattern='Onsets')
            rr = segment_variability_mean(participant,bloc, start_letter='R', stop_letter='R', start_pattern='Peaks', stop_pattern='Peaks')

            if pr <= 0.2:
                pr_interpretation = 'Valide'
            else:
                pr_interpretation = 'BAV'
            if st <= 0.15:
                st_interpretation = 'Valide'
            else:
                st_interpretation = 'ST Long'
            if qt <= 0.5:
                qt_interpretation = 'Valide'
            else:
                qt_interpretation = 'QT Long'


            if bloc == 'coherence':
                FR = 0.1
            elif bloc == 'free':
                FR = 0.15
            elif bloc == 'confort':
                FR = 0.2
            row = [participant, bloc , pr , st , qt, rr, FR ,  pr_interpretation, st_interpretation, qt_interpretation]
            rows.append(row)
    df = pd.DataFrame(rows , columns = ['participant','bloc','pr','st','qt','rr', 'fresp', 'PR interp','ST interp','QT interp'])
    return df


def plot_ecg_features(ecg):
    signals, info = nk.ecg_process(-ecg, sampling_rate=512)
    sig = signals['ECG_Clean']
    p = signals['ECG_P_Onsets']
    q = signals['ECG_R_Onsets']
    r = signals['ECG_R_Peaks']
    s = signals['ECG_S_Peaks']
    t = signals['ECG_T_Offsets']
    plt.figure()
    plt.plot(sig)
    plt.vlines(x = np.where(p==1) , ymin = min(signals['ECG_Clean']), ymax = max(signals['ECG_Clean']), colors = 'y')
    plt.vlines(x = np.where(q==1) , ymin = min(signals['ECG_Clean']), ymax = max(signals['ECG_Clean']), colors = 'g')
    plt.vlines(x = np.where(r==1) , ymin = min(signals['ECG_Clean']), ymax = max(signals['ECG_Clean']), colors = 'r')
    plt.vlines(x = np.where(s==1) , ymin = min(signals['ECG_Clean']), ymax = max(signals['ECG_Clean']), colors = 'm')
    plt.vlines(x = np.where(t==1) , ymin = min(signals['ECG_Clean']), ymax = max(signals['ECG_Clean']), colors = 'c')
    # plt.plot(p_peaks*signals['ECG_P_Peaks'][p_peaks == 1] , "x")
    plt.show()


def plot_ecg_features_sam(ecg):
    fig, ax = plt.subplots()
    signals, info = nk.ecg_process(-ecg, sampling_rate=512)
    sig = signals['ECG_Clean']
    ax.plot(sig)
    peaks = {
        'ECG_P_Onsets' : 'y',
        'ECG_R_Onsets' : 'g',
        'ECG_Q_Peaks' : 'k',
        'ECG_R_Peaks' : 'r',
        'ECG_S_Peaks' : 'm',
        'ECG_T_Offsets' : 'c',

    }
    for k, color in peaks.items():
        x,  = np.nonzero(signals[k].values==1)
        y = sig.values[x]
        ax.scatter(x, y, color=color)


def segments_cycles(cycles):
    # cycles = cycles.dropna()
    c = cycles.iloc[1:, :]
    pr = c['Q_Peaks'].iloc[:-1].values - c['P_Onsets'].iloc[:-1].values
    st = c['T_Offsets'].iloc[:-1].values - c['S_Peaks'].iloc[:-1].values
    qt = c['T_Offsets'].iloc[:-1].values - c['Q_Peaks'].iloc[:-1].values
    rr = c['R_Peaks'].iloc[1:].values - c['R_Peaks'].iloc[:-1].values
    qs = c['S_Peaks'].iloc[:-1].values - c['Q_Peaks'].iloc[:-1].values
    segments = [pr, st, qt, rr , qs]
    segments_reshape = [ segment.reshape(segment.shape[0], 1) for segment in segments]
    concat = np.concatenate(segments_reshape, axis = 1)
    # print(concat.shape)
    df = pd.DataFrame(concat, columns = ['pr','st','qt','rr','qs'])
    df = df.astype(float)
    df = df.dropna()
    return df





