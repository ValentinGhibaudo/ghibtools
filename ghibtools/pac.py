import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy import stats
import pandas as pd
import xarray as xr
from .signals import init_da, iirfilt

def Kullback_Leibler_Distance(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return np.sum(np.where(a != 0, a * np.log(a / b), 0))

def Shannon_Entropy(a):
    a = np.asarray(a, dtype=float)
    return - np.sum(a*np.log(a))

def Modulation_Index(distrib, show=False, verbose=False):
    distrib = np.asarray(distrib, dtype = float)
    
    if verbose:
        if np.sum(distrib) != 1:
            print(f'(!)  The sum of all bins is not 1 (sum = {round(np.sum(distrib), 2)})  (!)')
        
    N = distrib.size
    uniform_distrib = np.ones(N) * (1/N)
    mi = Kullback_Leibler_Distance(distrib, uniform_distrib) / np.log(N)
    
    if show:
        bin_width_deg = 360 / N
        
        doubled_distrib = np.concatenate([distrib,distrib] )
        x = np.arange(0, doubled_distrib.size*bin_width_deg, bin_width_deg)
        fig, ax = plt.subplots(figsize = (8,4))
        
        doubled_uniform_distrib = np.concatenate([uniform_distrib,uniform_distrib] )
        ax.scatter(x, doubled_uniform_distrib, s=2, color='r')
        
        ax.bar(x=x, height=doubled_distrib, width = bin_width_deg/1.1, align = 'edge')
        ax.set_title(f'Modulation Index = {round(mi, 4)}')
        ax.set_xlabel(f'Phase (Deg)')
        ax.set_ylabel(f'Amplitude (Normalized)')
        ax.set_xticks([0,360,720])

    return mi

def Shannon_MI(a):
    a = np.asarray(a, dtype = float)
    N = a.size
    kl_divergence_shannon = np.log(N) - Shannon_Entropy(a)
    return kl_divergence_shannon / np.log(N)

def get_phase(sig):
    analytic_signal = signal.hilbert(sig)
    instantaneous_phase = np.angle(analytic_signal)
    return instantaneous_phase

def get_amp(sig):
    analytic_signal = signal.hilbert(sig)
    amplitude_envelope = np.abs(analytic_signal)
    return amplitude_envelope

def shuffle_sig(sig, reverse=True):
    random_clip_index = np.random.randint(0, sig.size) # create a random index along sig
    clipped_1, clipped_2 = sig[0:random_clip_index] , sig[random_clip_index:] # clip sig at this random index
    
    if reverse:
        if np.random.randint(0,2) == 1: # one chance by two to reverse second clipped sig (that will be first clipped sig)
            clipped_2 = -clipped_2
            
    sig_shuffled = np.concatenate([clipped_2,clipped_1])
    return sig_shuffled

def get_phase_amplitude_vectors(sig, modulant_freqs, target_freqs, srate, show = False, window_size=2):
    
    sig = sig - np.mean(sig) # center sig
    
    modulant_filtered_sig = iirfilt(sig, srate, lowcut = modulant_freqs[0], highcut=modulant_freqs[1])
    target_filtered_sig = iirfilt(sig, srate, lowcut = target_freqs[0], highcut=target_freqs[1])
    phase_modulant = get_phase(modulant_filtered_sig)
    amp_target = get_amp(target_filtered_sig)
    
    if show:
        index_end = int(window_size * srate)
        sig_plot = sig[:index_end]
        modulant_filtered_sig_plot = modulant_filtered_sig[:index_end]
        phase_modulant_plot = phase_modulant[:index_end]
        target_filtered_sig_plot = target_filtered_sig[:index_end]
        amp_target_plot = amp_target[:index_end]
        
        time = np.arange(0,sig_plot.size/srate,1/srate)
        
        fig, axs = plt.subplots(nrows = 4, figsize = (8,6), constrained_layout =True)
        
        ax = axs[0]
        ax.plot(time, sig_plot)
        ax.set_title('Raw signal')
        ax.set_ylabel('Amp [mV]')
        ax.set_xlabel('Time [s]')
        
        ax = axs[1]
        ax.plot(time, modulant_filtered_sig_plot)
        ax.set_title(f'Modulant signal filtered [{modulant_freqs[0]}-{modulant_freqs[1]}] Hz')
        ax.set_ylabel('Amp [mV]')
        ax.set_xlabel('Time [s]')
        
        ax = axs[2]
        ax.plot(time, np.degrees(phase_modulant_plot) + 180)
        ax.set_title(f'Modulant signal Phases [{modulant_freqs[0]}-{modulant_freqs[1]}] Hz')
        ax.set_ylabel('Phase [Deg]')
        ax.set_xlabel('Time [s]')
        
        ax = axs[3]
        ax.plot(time, target_filtered_sig_plot, label = 'filtered target sig')
        ax.plot(time, amp_target_plot, label='envelope')
        ax.set_title(f'Target signal filtered and amplitude envelope [{target_freqs[0]}-{target_freqs[1]}] Hz')
        ax.set_ylabel('Amp [mV]')
        ax.set_xlabel('Time [s]')
        
        plt.show()
        
    return phase_modulant, amp_target

def get_amplitude_distribution(phase_modulant, amp_target, nbins=18):
    bin_edges = np.linspace(-np.pi,np.pi,nbins+1)
    mean_phase_amplitude_vector = []
    for j in range(bin_edges.size - 1):
        phase_inf = bin_edges[j]
        phase_sup = bin_edges[j+1]
        mean_amplitude_between_phase_bin_edges = np.mean( amp_target[(phase_modulant >= phase_inf) & (phase_modulant < phase_sup)] )
        mean_phase_amplitude_vector.append(mean_amplitude_between_phase_bin_edges)
        
    amplitude_distribution = np.array(mean_phase_amplitude_vector) / sum(mean_phase_amplitude_vector)
    return amplitude_distribution

def phase_amplitude_plot(pac_distribution, modulant_freqs, target_freqs):
    N = pac_distribution.size
    bin_width_deg = 360 / N
    pac_distrib_doubled = np.concatenate([pac_distribution,pac_distribution] )
    x = np.arange(0, pac_distrib_doubled.size*bin_width_deg, bin_width_deg)
    fig, ax = plt.subplots(figsize = (8,4))
    ax.bar(x=x, height=pac_distrib_doubled, width = bin_width_deg/1.1, align = 'edge')
    ax.set_title(f'Modulation Index = {round(Modulation_Index(pac_distribution), 4)}')
    ax.set_xlabel(f'Phase [{modulant_freqs[0]}-{modulant_freqs[1]}] Hz (Deg)')
    ax.set_ylabel(f'Amplitude [{target_freqs[0]}-{target_freqs[1]}] Hz (Normalized)')
    ax.set_xticks([0,360,720])
    plt.show()
    
def raw_to_mi(sig, modulant_freqs, target_freqs, srate, N=18, shuffle = False):
    sig = sig - np.mean(sig)
    phase_modulant, amp_target = get_phase_amplitude_vectors(sig=sig, modulant_freqs=modulant_freqs, target_freqs=target_freqs, srate=srate)
    if shuffle:
        phase_modulant = shuffle_sig(phase_modulant)
        amp_target = shuffle_sig(amp_target)
    pac_distribution = get_amplitude_distribution(phase_modulant, amp_target, N)
    mi = Modulation_Index(pac_distribution)
    return mi

def get_heights_ratio(distrib):
    return (np.max(distrib) - np.min(distrib)) / np.max(distrib)

def psd_of_amplitude_envelope(sig, target_freqs, srate, nperseg, show = False):
    sig = sig - np.mean(sig)
    filtered_sig_target = iirfilt(sig, srate, lowcut = target_freqs[0], highcut=target_freqs[1])
    amp_target = get_amp(filtered_sig_target)
    f, Pxx = signal.welch(amp_target, fs=srate, nperseg = nperseg)
    dominant_freq = f[np.argmax(Pxx)]
    if show:
        fig, ax = plt.subplots()
        ax.plot(f[f <= target_freqs[0]], Pxx[f <= target_freqs[0]])
        ax.set_title(f'[{target_freqs[0]}-{target_freqs[1]}] Hz Dominant Frequency of Envelope = {round(dominant_freq, 2)} Hz')
        ax.set_ylabel(f'PSD of Envelope [{target_freqs[0]}-{target_freqs[1]}]')
        ax.set_xlabel(f'Freq [Hz]')
        ax.vlines(x=dominant_freq, ymin=0, ymax = np.max(Pxx), linestyle = '--', color = 'r')
        ax.set_xticks([0, dominant_freq, f[f <= target_freqs[0]][-1]])
        plt.show()
    return dominant_freq

def get_mean_vector_length(sig, modulant_freqs, target_freqs, srate):
    sig = sig - np.mean(sig)
    phase_modulant, amp_target = get_phase_amplitude_vectors(sig, modulant_freqs, target_freqs, srate)
    complex_vector = amp_target * np.exp(1j * phase_modulant)
    mean_vector = np.mean(complex_vector)
    mean_vector_length = np.abs(mean_vector)
    mean_vector_angle = np.angle(mean_vector)
    return mean_vector_length


def phase_locking_value(sig, modulant_freqs, target_freqs, srate):
    sig = sig - np.mean(sig)
    phase_modulant, amp_target = get_phase_amplitude_vectors(sig, modulant_freqs, target_freqs, srate)
    phase_envelope = get_phase(amp_target)
    PLV = np.abs(np.mean(np.exp(1j*phase_modulant - 1j*phase_envelope)))
    return PLV

def correlation_coefficient(sig, modulant_freqs, target_freqs, srate):
    sig = sig - np.mean(sig)
    phase_modulant, amp_target = get_phase_amplitude_vectors(sig, modulant_freqs, target_freqs, srate)
    r, p = stats.spearmanr(amp_target, phase_modulant)
    return r

def coherence_value(sig, target_freqs, srate, show=False):
    sig = sig - np.mean(sig)
    filtered_sig_target = iirfilt(sig, srate, lowcut = target_freqs[0], highcut = target_freqs[1])
    amp_target = get_amp(filtered_sig_target)
    f, Cxy = signal.coherence(amp_target, sig, fs=srate, nperseg=srate*2)
    dominant_freq = f[np.argmax(Cxy)]
    if show:
        fig, ax = plt.subplots()
        ax.plot(f, Cxy)
        ax.set_title(f'[{target_freqs[0]}-{target_freqs[1]}] Hz filtered sig Envelope Coherence with Raw Sig')
        ax.set_ylabel(f'Coherence of Envelope [{target_freqs[0]}-{target_freqs[1]}] with Raw sig')
        ax.set_xlabel(f'Freq [Hz]')
        ax.vlines(x=dominant_freq, ymin=0, ymax = np.max(Cxy), linestyle = '--', color = 'r')
        ax.set_xticks([0, dominant_freq, f[-1]])
        plt.show()
    return dominant_freq

def get_all_features(sig, modulant_freqs, target_freqs, srate):
    sig = sig - np.mean(sig)
    phase_modulant, amp_target = get_phase_amplitude_vectors(sig, modulant_freqs, target_freqs, srate)
    pac_distibution = get_amplitude_distribution(phase_modulant, amp_target)
    
    modulation_index = Modulation_Index(pac_distribution)
    heights_ratio = get_heights_ratio(pac_distribution)
    psd_dom_freq = psd_of_amplitude_envelope(sig, target_freqs, srate)
    cxx_dom_freq = coherence_value(sig, target_freqs, srate)
    mean_vector_length = get_mean_vector_length(sig, modulant_freqs, target_freqs, srate)
    plv = phase_locking_value(sig, modulant_freqs, target_freqs, srate)
    r = correlation_coefficient(sig, modulant_freqs, target_freqs, srate)
    
    data = np.array([str(modulant_freqs), str(target_freqs), modulation_index, heights_ratio, psd_dom_freq, cxx_dom_freq, mean_vector_length, plv, r])
    df_features = pd.DataFrame(data=data, index = ['Modulant Freqs', 'Target Freqs','Modulation Index','Heights Ratio','Main PSD Freq','Main Cxx Freq','Mean Vector Length','Phase Locking Value','r'])
    return df_features.T.round(2)


def get_comodulogram(sig, srate, range_fp , range_fa, bandwidth_fp=4, bandwidth_fa=10, fp_resolution=0.5, fa_resolution=0.5):
    modulants = np.arange(range_fp[0], range_fp[1], fp_resolution)
    modulated = np.arange(range_fa[0], range_fa[1], fa_resolution)
    da_comodulogram = None
    for fp in modulants:
        modulants_freqs = (fp - bandwidth_fp/2, fp + bandwidth_fp/2)
        for fa in modulated:
            target_freqs = (fa - bandwidth_fa/2, fa + bandwidth_fa/2)
            mi = raw_to_mi(sig, modulants_freqs, target_freqs, srate)
            if da_comodulogram is None:
                da_comodulogram = init_da({'Modulated_freqs':modulated, 'Modulant_Freqs':modulants})
            da_comodulogram.loc[fa, fp] = mi
    return da_comodulogram

def simu_pac_sig(pac_value, freq_modulant, amp_modulant, freq_modulated, amp_modulated, noise_amp, duration, srate):
    time = np.arange(0 , duration , 1 / srate)
    X = pac_value
    Amp_Envelope = amp_modulant * ((1 - X)*np.sin(2*np.pi*freq_modulant*time) + 1 + X) / 2
    simulated_signal = Amp_Envelope * np.sin(2*np.pi*freq_modulated*time) + amp_modulant * np.sin(2*np.pi*freq_modulant*time) + np.random.randn(time.size) * noise_amp
    return time, simulated_signal
