import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy import fftpack
import scipy.interpolate
import xarray as xr
import joblib
import pandas as pd
import mne

def notch(sig, fs):
    sig_notched = mne.filter.notch_filter(sig, Fs=fs, freqs=np.arange(50,101,50),  verbose=False)
    return sig_notched
    
def get_wsize(srate, lowest_freq , n_cycles=5):
    nperseg = ( n_cycles / lowest_freq) * srate
    return int(nperseg)

def get_memory(numpy_array):
    memory = numpy_array.nbytes * 1e-9
    print(f'{memory} Go')
    
def filter_sig(sig,fs, low, high , order=1, btype = 'mne', show = False):
    if btype == 'bandpass':
        # Paramètres de notre filtre :
        fe = fs
        f_lowcut = low
        f_hicut = high
        nyq = 0.5 * fe
        N = order                # Ordre du filtre
        Wn = [f_lowcut/nyq,f_hicut/nyq]  # Nyquist frequency fraction

        # Création du filtre :
        b, a = signal.butter(N, Wn, btype)

        # Calcul de la reponse en fréquence du filtre
        w, h = signal.freqz(b, a)

        # Applique le filtre au signal :
        filtered_sig = signal.filtfilt(b, a, sig)
        
    elif btype == 'lowpass':
        
        # Paramètres de notre filtre :
        fe = fs
        f_hicut = high
        nyq = 0.5 * fe
        N = order                  # Ordre du filtre
        Wn = f_hicut/nyq  # Nyquist frequency fraction

        # Création du filtre :
        b, a = signal.butter(N, Wn, btype)

        # Calcul de la reponse en fréquence du filtre
        w, h = signal.freqz(b, a)

        # Applique le filtre au signal :
        filtered_sig = signal.filtfilt(b, a, sig)
        
    elif btype == 'highpass':
        
        # Paramètres de notre filtre :
        fe = fs
        f_lowcut = low
        nyq = 0.5 * fe
        N = order                  # Ordre du filtre
        Wn = f_lowcut/nyq  # Nyquist frequency fraction

        # Création du filtre :
        b, a = signal.butter(N, Wn, btype)

        # Calcul de la reponse en fréquence du filtre
        w, h = signal.freqz(b, a)

        # Applique le filtre au signal :
        filtered_sig = signal.filtfilt(b, a, sig)
        
    elif btype == 'mne':
        filtered_sig = mne.filter.filter_data(sig, sfreq=fs, l_freq = low, h_freq = high, verbose = False)
        
    if show:
        # Tracé de la réponse en fréquence du filtre
        fig, ax = plt.subplots(figsize=(8,5)) 
        ax.plot(0.5*fe*w/np.pi, np.abs(h), 'b')
        ax.set_xlabel('frequency [Hz]')
        ax.set_ylabel('Amplitude [dB]')
        ax.grid(which='both', axis='both')
        plt.show()

    return filtered_sig

def norm(data):
    data = (data - np.mean(data)) / np.std(data)
    return data

def detrend(sig):
    dentrended = signal.detrend(sig)
    return dentrended

def center(sig):
    sig_centered = sig - np.mean(sig)
    return sig_centered

def time_vector(sig, srate):
    time = np.arange(0, sig.size / srate , 1 / srate)
    return time

def down_sample(sig, factor): 
    sig_down = signal.decimate(sig, q=factor, n=None, ftype='iir', axis=- 1, zero_phase=True)
    return sig_down

def spectre(sig, srate, lowest_freq, n_cycles = 5, nfft_factor = 2, verbose = False):
    nperseg = get_wsize(srate, lowest_freq, n_cycles)
    nfft = nperseg * nfft_factor
    f, Pxx = signal.welch(sig, fs=srate, nperseg = nperseg , nfft = nfft, scaling='spectrum')
    if verbose:
        n_windows = 2 * sig.size // nperseg
        print(f'nperseg : {nperseg}')
        print(f'sig size : {sig.size}')
        print(f'total cycles lowest freq : {int(sig.size / ((1 / lowest_freq)*srate))}')
        print(f'nwindows : {n_windows}')
    return f, Pxx

def coherence(sig1,sig2, srate, lowest_freq, n_cycles = 5, nfft_factor = 2, verbose= False):
    nperseg = get_wsize(srate, lowest_freq, n_cycles)
    nfft = nperseg * nfft_factor
    f, Cxy = signal.coherence(sig1,sig2, fs=srate, nperseg = nperseg , nfft = nfft )
    if verbose:
        n_windows = 2 * sig.size // nperseg
        print(f'nperseg : {nperseg}')
        print(f'sig size : {sig.size}')
        print(f'total cycles lowest freq : {int(sig.size / ((1 / lowest_freq)*srate))}')
        print(f'nwindows : {n_windows}')
    return f, Cxy

def init_da(coords, name = None):
    dims = list(coords.keys())
    coords = coords

    def size_of(element):
        element = np.array(element)
        size = element.size
        return size

    shape = tuple([size_of(element) for element in list(coords.values())])
    data = np.zeros(shape)
    da = xr.DataArray(data=data, dims=dims, coords=coords, name = name)
    return da

def parallelize(iterator, function, n_jobs):
    result = joblib.Parallel(n_jobs = n_jobs, prefer = 'threads')(joblib.delayed(function)(i) for i in iterator)
    return result

def shuffle_sig_one_inversion(sig):
    half_size = sig.shape[0]//2
    ind = np.random.randint(low=0, high=half_size)
    sig2 = sig.copy()
    
    sig2[ind:ind+half_size] *= -1
    if np.random.rand() >=0.5:
        sig2 *= -1

    return sig2

def shuffle_sig_one_break(sig):
    ind = np.random.randint(low=0, high=sig.shape[0])
    sig2 = np.hstack([sig[ind:], sig[:ind]])
    return sig2



def discrete_FT_homemade(sig, srate):
    t = np.arange(0, sig.size/srate, 1/srate)
    f = np.linspace(0, int(srate/2), int(t.size / 2)) 
    # "number of unique freqs that can be extracted from a time series is 1/2 nb of points of the time series plus the zero freq, because of nyquist Th"
    # nyquist th = you need at least two points per cycle to measure a sine wave, and thus half the number of points in the data to the fastest frequency that can be extracted from t"
    fourier = np.zeros(t.size, dtype = 'complex_') # initialize fourier coeff with sig.size == time.size, dtype = complexe to be able to receive complex values
    for i in range(t.size): # loop on bins of time
        freq = i # freq of the kernel = the sine wave = 0, 1, 2 ... N
        kernel = np.exp(-1j * 2 * np.pi * freq * t ) # kernel = sine wave of freq i , and size = time = sig.size, but imaginary defined sine wave !
        dot_product = np.sum(sig * kernel) # dot product = sum of sig * kernel simply, because sig.size == kernel.size, so sliding/zero-padding is not needed
        fourier[i] = dot_product # fourier coefficient in bin i == dot_product of bin i
    fourier_full = fourier 
    fourier = fourier[0:int(t.size/2)] # see below (keep only positive freqs by selecting first half of fourier) 
    f = f * duration
    power = np.abs(fourier) ** 2
    phase = np.angle(fourier, deg = False) # Return the angle of the complex argument. In radians or degrees according to deg param
    return f, fourier, fourier_full, power, phase # fourier coef = concatenation of dot products of sig vs kernel with kernel = imaginary sine waves of freq = idx of iteration across time bins

def discrete_FT_homemade_short(sig, srate):
    t = np.arange(0, sig.size/srate, 1/srate)
    f = np.linspace(0, int(srate/2), int(t.size / 2)) 
    # "number of unique freqs that can be extracted from a time series is 1/2 nb of points of the time series plus the zero freq, because of nyquist Th"
    # nyquist th = you need at least two points per cycle to measure a sine wave, and thus half the number of points in the data to the fastest frequency that can be extracted from t"
    fourier = np.zeros(t.size, dtype = 'complex_') # initialize fourier coeff with sig.size == time.size, dtype = complexe to be able to receive complex values
    for i in range(t.size): # loop on bins of time
        freq = i # freq of the kernel = the sine wave = 0, 1, 2 ... N
        kernel = np.exp(-1j * 2 * np.pi * freq * t ) # kernel = sine wave of freq i , and size = time = sig.size, but imaginary defined sine wave !
        dot_product = np.sum(sig * kernel) # dot product = sum of sig * kernel simply, because sig.size == kernel.size, so sliding/zero-padding is not needed
        fourier[i] = dot_product # fourier coefficient in bin i == dot_product of bin i
    fourier_full = fourier 
    fourier = fourier[0:int(t.size/2)] # see below (keep only positive freqs by selecting first half of fourier) 
    f = f * duration
    power = np.abs(fourier) ** 2
    phase = np.angle(fourier, deg = False) # Return the angle of the complex argument. In radians or degrees according to deg param
    return f, power, fourier_full # fourier coef = concatenation of dot products of sig vs kernel with kernel = imaginary sine waves of freq = idx of iteration across time bins

def inverse_FT_homemade(N, fourier_full):
    sine_waves = np.zeros((N,N))
    for fi in range(N):
        sine_wave = fourier_full[fi] * np.exp(-1j * 2 * np.pi * fi * time)
        sine_wave = np.real(sine_wave)
        sine_waves[fi,:] = sine_wave
    sig_reconstructed = - np.sum(sine_waves, axis = 0) / (N * duration)
    return sig_reconstructed

def inverse_FT_homemade_modif(N, fourier_full):
    sine_waves = np.zeros((N,N))
    for fi in range(N):
        sine_wave = fourier_full[fi] * np.exp(-1j * 2 * np.pi * fi * time)
        sine_wave = np.real(sine_wave)
        sine_waves[fi,:] = sine_wave
    sig_reconstructed = - np.sum(sine_waves, axis = 0) / (N * duration)
    return sig_reconstructed


def convolution_time_domain(sig, kernel):
    kernel_flip = kernel[::-1] # flip the kernel
    zero_padding = np.zeros(kernel.size - 1) # prepare the amount of zeros to add to the beginning and end of the sig to begin and end the convolution and extremes of sig
    sig_zero_padded = np.concatenate((zero_padding, sig, zero_padding)) # add zero_padding to beginning and end of sig
    conv = np.zeros(sig_zero_padded.size - kernel.size + 1) # prepare shape of convolution to be filled with time bin results

    for i in range(sig_zero_padded.size - kernel.size + 1): # loop on each bin of sig (nb iterations = conv.size)
        bin_product = []
        for j in range(kernel.size): # loop on kernel bins
            bin_product.append(sig_zero_padded[i+j]*kernel_flip[j]) # append values (= bin_sig * bin_kernel) to be summed to get dot product
        dot_product = np.sum(bin_product) # sum values to get the dot product
        conv[i] = dot_product # add dot_product in a time series
        
    convolution = conv[int(kernel.size/2):conv.size - int(kernel.size/2) + 1 ] # cut result with one-half size of kernel at the beginning and one-half + 1 size of the kernel at the end to get same size of sig
    
    return convolution

def convolution_freq_domain(sig, kernel):
    f_sig, power_sig, fourier_full_sig  = discrete_FT_homemade_short(sig, srate)
    f_kernel, power_kernel, fourier_full_kernel = discrete_FT_homemade_short(kernel, srate)
    
    multiple_fourier_full = fourier_full_sig * fourier_full_kernel
    
    convo_by_freq = inverse_FT_homemade(N = sig.size, fourier_full = multiple_fourier_full)
    
    return convo_by_freq, power_sig, power_kernel, f_sig, multiple_fourier_full

def gaussian_win(a , time, m , n , freq):
    
    # a = 2 # amplitude
    # time = time # time
    # m  = time[-1] / 2 # offset (not relevant for eeg analyses and can be always set to zero)
    # n = 10 # refers to the number of wavelet cycles , defines the trade-off between temporal precision and frequency precision
    # freq = 10 # change the width of the gaussian and therefore of the wavelet
    # s = n / (2 * np.pi * freq) # std of the width of the gaussian, freq = in Hz
    
    GaussWin = a * np.exp( -(time - m)** 2 / (2 * s**2))
    
    return GaussWin

def morlet_wavelet(a , time, m , n , freq):
    s = n / (2 * np.pi * freq)
    GaussWin = a * np.exp( -(time - m)** 2 / (2 * s**2))
    SinWin = np.sin ( 2 * np.pi * freq * time )
    MorletWavelet = GaussWin * SinWin
    return MorletWavelet

def complex_mw(a , time, n , freq, m = 0):
    s = n / (2 * np.pi * freq)
    GaussWin = a * np.exp( -(time - m)** 2 / (2 * s**2))
    complex_sinewave = np.exp(1j * 2 *np.pi * freq * time)
    cmw = GaussWin * complex_sinewave
    return cmw

def extract_features_from_cmw_family(sig, time_sig, cmw_family_params, return_cmw_family=False, module_method='abs'): # cmw_family_params = {'amp':amp, 'time':time_cmw, 'n_cycles':n_cycles, 'm':m, 'range':range_freqs}
    
    shape = (2, cmw_family_params['range'].size, cmw_family_params['time'].size)
    cmw_family = np.zeros(shape)

    dims = ['axis','freq','time']
    coords = {'axis':['real','imag'],'freq':cmw_family_params['range'], 'time':cmw_family_params['time']}
    da_cmw_family = xr.DataArray(data = cmw_family, dims = dims, coords = coords, name = 'cmw_family')

    shape = (cmw_family_params['range'].size , time_sig.size)
    
    reals = np.zeros(shape)
    imags = np.zeros(shape)
    modules = np.zeros(shape)
    angles = np.zeros(shape)

    features = ['filtered','i_filtered','phase','power']
    data = np.zeros(shape = (len(features),reals.shape[0] , reals.shape[1]))
    dims = ['feature','freqs','time']
    coords = {'feature':features, 'freqs':cmw_family_params['range'], 'time':time_sig}
    da_features = xr.DataArray(data = data, dims = dims, coords = coords, name = 'features')

    cmw_family_freq_range = cmw_family_params['range']
    cmw_family_n_range = cmw_family_params['n_cycles']
    idx = np.arange(0,cmw_family_freq_range.size,1)
    
    for i, fi, ni in zip(idx, cmw_family_freq_range, cmw_family_n_range):
     
        a = cmw_family_params['amp']
        time = cmw_family_params['time']
        m = cmw_family_params['m']
        
        cmw_f = complex_mw(a=a, time=time,n=ni, freq=fi, m = m)

        da_cmw_family.loc['real',fi,:] = np.real(cmw_f)
        da_cmw_family.loc['imag',fi,:] = np.imag(cmw_f)

        complex_conv = signal.convolve(sig, cmw_f, mode = 'same')

        real = np.real(complex_conv)
        imag = np.imag(complex_conv)
        angle = np.angle(complex_conv)
        
        if module_method == 'old':
            module = np.zeros((time_sig.size))
            for i_bin in range(time_sig.size):
                module_i_bin = np.sqrt((real[i_bin])**2 + (imag[i_bin])**2)
                module[i_bin] = module_i_bin
        elif module_method == 'abs':       
            module = np.abs(complex_conv)**2
        elif module_method == 'conjugate':
            module = complex_conv * np.conjugate(complex_conv)
        
        reals[i,:] = real
        imags[i,:] = imag
        modules[i,:] = module
        angles[i,:] = angle

        da_features.loc['filtered',:,:] = reals
        da_features.loc['i_filtered',:,:] = imags
        da_features.loc['phase',:,:] = angle
        da_features.loc['power',:,:] = modules
    if return_cmw_family:
        return da_features, da_cmw_family
    else:
        return da_features.loc['power',:,:]
    
    
def stretch_data(resp_features, nb_point_by_cycle, data, srate, inspi_ratio = 0.35):

    # params
    cycle_times = resp_features[['inspi_time', 'expi_time']].values
    mean_cycle_duration = np.mean(resp_features[['insp_duration', 'exp_duration']].values, axis=0)
    mean_inspi_ratio = mean_cycle_duration[0]/mean_cycle_duration.sum()
    times = np.arange(0,np.size(data))/srate

    clipped_times, times_to_cycles, cycles, cycle_points, data_stretch_linear = deform_to_cycle_template(
            data, times, cycle_times, nb_point_by_cycle=nb_point_by_cycle, inspi_ratio=inspi_ratio)

    nb_cycle = data_stretch_linear.shape[0]//nb_point_by_cycle
    phase = np.arange(nb_point_by_cycle)/nb_point_by_cycle
    data_stretch = data_stretch_linear.reshape(int(nb_cycle), int(nb_point_by_cycle))

    return data_stretch


def deform_to_cycle_template(data, times, cycle_times, nb_point_by_cycle=40, inspi_ratio = 0.4):
    """
    Input:
    data: ND array time axis must always be 0
    times: real timestamps associated to data
    cycle_times: N*2 array columns are inspi and expi times. If expi is "nan", corresponding cycle is skipped
    nb_point_by_cycle: number of respi phase per cycle
    inspi_ratio: relative length of the inspi in a full cycle (between 0 and 1)
    
    Output:
    clipped_times: real times used (when both respi and signal exist)
    times_to_cycles: conversion of clipped_times in respi cycle phase
    cycles: array of cycle indices (rows of cycle_times) used
    cycle_points: respi cycle phases where deformed_data is computed
    deformed_data: data rescaled to have cycle_points as "time" reference
    """
    
    #~ nb_point_inspi = int(nb_point_by_cycle*inspi_ratio)
    #~ nb_point_expi = nb_point_by_cycle - nb_point_inspi
    #~ one_cycle = np.linspace(0,1,nb_point_by_cycle)
    #~ two_cycle = np.linspace(0,2,nb_point_by_cycle*2)
    
    #~ print('cycle_times.shape', cycle_times.shape)
    
    #clip cycles if data/times smaller than cycles
    keep_cycle = (cycle_times[:, 0]>=times[0]) & (cycle_times[:, 1]<times[-1]) # keep cycles whose times are conatained in provided time vector
    first_cycle = np.where(keep_cycle)[0].min()
    last_cycle = np.where(keep_cycle)[0].max()+1
    n_cycles_removed = 0
    if last_cycle==cycle_times.shape[0]:
        #~ print('yep')
        last_cycle -= 1 # remove last cycle if last cycle is the last of provided cycles times
        n_cycles_removed = n_cycles_removed+1
    #~ print('first_cycle', first_cycle, 'last_cycle', last_cycle)

    #clip times/data if cycle_times smaller than times
    keep = (times>=cycle_times[first_cycle,0]) & (times<cycle_times[last_cycle,0]) # keep times from first cycle point to last cycle point
    #~ print(keep)
    clipped_times = times[keep] # = times - ( (time between start and first cycle point) + (time between end of pre last cycle to end time) )
    clipped_data = data[keep] # = data kept according to clipped times
    #~ print('clipped_times', clipped_times.shape, clipped_times[0], clipped_times[-1])
    
    # construct cycle_step
    times_to_cycles = np.zeros(clipped_times.shape)*np.nan # make a nan array of shape of clipped times
    cycles = np.arange(first_cycle, last_cycle) # rename cycles from 0 to last cycle
    t_start = clipped_times[0]
    sr = np.median(np.diff(clipped_times)) # get sampling rate
    #~ print('t_start', t_start, 'sr', sr)
    for c in cycles:
        #2 segments : inspi + expi
        
        if not np.isnan(cycle_times[c, 1]):
            #no missing cycles
            mask_inspi_times=(clipped_times>=cycle_times[c, 0])&(clipped_times<cycle_times[c, 1]) # where are inspi times of the cycles
            mask_expi_times=(clipped_times>=cycle_times[c, 1])&(clipped_times<cycle_times[c+1, 0]) # where are expi times of the cycles
            times_to_cycles[mask_inspi_times]=(clipped_times[mask_inspi_times]-cycle_times[c, 0])/(cycle_times[c, 1]-cycle_times[c, 0])*inspi_ratio+c
            times_to_cycles[mask_expi_times]=(clipped_times[mask_expi_times]-cycle_times[c, 1])/(cycle_times[c+1, 0]-cycle_times[c, 1])*(1-inspi_ratio)+c+inspi_ratio
                    
        else:
            #there is a missing cycle
            mask_cycle_times=(clipped_times>=cycle_times[c, 0])&(clipped_times<cycle_times[c+1, 0])
            times_to_cycles[mask_cycle_times]=(clipped_times[mask_cycle_times]-cycle_times[c, 0])/(cycle_times[c+1, 0]-cycle_times[c, 0])+c
    
    # new clip with cycle
    keep = ~np.isnan(times_to_cycles)
    times_to_cycles = times_to_cycles[keep]
    clipped_times = clipped_times[keep]
    clipped_data = clipped_data[keep]
    
    
    interp = scipy.interpolate.interp1d(times_to_cycles, clipped_data, kind='linear', axis=0, bounds_error=False, fill_value='extrapolate')
    cycle_points = np.arange(first_cycle, last_cycle, 1./nb_point_by_cycle)

    if cycle_points[-1]>times_to_cycles[-1]:
        # it could append that the last bins of the last cycle is out last cycle
        # due to rounding effect so:
        last_cycle = last_cycle-1
        cycles = np.arange(first_cycle, last_cycle)
        cycle_points = np.arange(first_cycle, last_cycle, 1./nb_point_by_cycle)
        n_cycles_removed = n_cycles_removed+1
    
    deformed_data = interp(cycle_points)
    
    #put NaN for missing cycles
    missing_ind,  = np.nonzero(np.isnan(cycle_times[:, 1]))
    #~ print('missing_ind', missing_ind)
    for c in missing_ind:
        #mask = (cycle_points>=c) & (cycle_points<(c+1))
        #due to rounding problem add esp
        esp = 1./nb_point_by_cycle/10.
        mask = (cycle_points>=(c-esp)) & (cycle_points<(c+1-esp))
        deformed_data[mask] = np.nan
    
    if n_cycles_removed != 0:
        cycles_times_final = cycle_times[:-n_cycles_removed , :]
    else:
        cycles_times_final = cycle_times
    print('n cycle removed =' , n_cycles_removed)
    return cycles_times_final, cycles, deformed_data
    
def detect_respiration_cycles(resp_sig, sampling_rate, t_start = 0., output = 'index',

                                    # preprocessing
                                    inspiration_sign = '-',
                                    high_pass_filter = None,
                                    constrain_frequency = None,
                                    median_windows_filter = None,
                                    
                                    # baseline
                                    baseline_with_average = False,
                                    manual_baseline = 0.,
                                    
                                    # clean
                                    eliminate_time_shortest_ratio = 10,
                                    eliminate_amplitude_shortest_ratio = 10,
                                    eliminate_mode = 'OR', # 'AND'
                                    
                                    ):


    sig = resp_sig.copy()
    sr = sampling_rate

    # STEP 1 : preprocessing
    sig = sig  - manual_baseline

    if inspiration_sign =='-' :
        sig = -sig
    
    if median_windows_filter is not None:
        k = int(np.round(median_windows_filter*sr/2.)*2+1)
        sig = scipy.signal.medfilt(sig, kernel_size = k)
    
    original_sig = resp_sig.copy()
    
    #baseline center
    if baseline_with_average:
        centered_sig = sig - sig.mean()
    else:
        centered_sig = sig

    if high_pass_filter is not None:
        sig =  fft_filter(sig, f_low =high_pass_filter, f_high=None, sampling_rate=sr)
    
    # hard filter to constrain frequency
    if constrain_frequency is not None:
        filtered_sig =  fft_filter(centered_sig, f_low =None, f_high=constrain_frequency, sampling_rate=sr)
    else :
        filtered_sig = centered_sig
    
    # STEP 2 : crossing zeros on filtered_sig
    ind1, = np.where( (filtered_sig[:-1] <=0) & (filtered_sig[1:] >0))
    ind2, = np.where( (filtered_sig[:-1] >=0) & (filtered_sig[1:] <0))
    if ind1.size==0 or ind2.size==0:
        return np.zeros((0,2), dtype='int64')
    ind2 = ind2[ (ind2>ind1[0]) & (ind2<ind1[-1]) ]
    
    # STEP 3 : crossing zeros on centered_sig
    ind_inspi_possible, = np.where( (centered_sig[:-1]<=0 ) &  (centered_sig[1:]>0 ) )
    list_inspi = [ ]
    for i in range(len(ind1)) :
        ind = np.argmin( np.abs(ind1[i] - ind_inspi_possible) )
        list_inspi.append( ind_inspi_possible[ind] )
    list_inspi = np.unique(list_inspi)

    ind_expi_possible, = np.where( (centered_sig[:-1]>0 ) &  (centered_sig[1:]<=0 ) )
    list_expi = [ ]
    for i in range(len(list_inspi)-1) :
        ind_possible = ind_expi_possible[ (ind_expi_possible>list_inspi[i]) & (ind_expi_possible<list_inspi[i+1]) ]
        
        ind_possible2 = ind2[ (ind2>list_inspi[i]) & (ind2<list_inspi[i+1]) ]
        ind_possible2.sort()
        if ind_possible2.size ==1 :
            ind = np.argmin( abs(ind_possible2 - ind_possible ) )
            list_expi.append( ind_possible[ind] )
        elif ind_possible2.size >=1 :
            ind = np.argmin( np.abs(ind_possible2[-1] - ind_possible ) )
            list_expi.append( ind_possible[ind]  )
        else :
            list_expi.append( max(ind_possible)  )
    
    list_inspi,list_expi =  np.array(list_inspi,dtype = 'int64')+1, np.array(list_expi,dtype = 'int64')+1
    
    
    # STEP 4 :  cleaning for small amplitude and duration
    nb_clean_loop = 20
    if eliminate_mode == 'OR':
        # eliminate cycle with too small duration or too small amplitude
    
        if eliminate_amplitude_shortest_ratio is not None :
            for b in range(nb_clean_loop) :
                max_inspi = np.zeros((list_expi.size))
                for i in range(list_expi.size) :
                    max_inspi[i] = np.max( np.abs(centered_sig[list_inspi[i]:list_expi[i]]) )
                ind, = np.where( max_inspi < np.median(max_inspi)/eliminate_amplitude_shortest_ratio)
                list_inspi[ind] = -1
                list_expi[ind] = -1
                list_inspi = list_inspi[list_inspi != -1]
                list_expi = list_expi[list_expi != -1]
                
                max_expi = np.zeros((list_expi.size))
                for i in range(list_expi.size) :
                    max_expi[i] = np.max( abs(centered_sig[list_expi[i]:list_inspi[i+1]]) )
                ind, = np.where( max_expi < np.median(max_expi)/eliminate_amplitude_shortest_ratio)
                list_inspi[ind+1] = -1
                list_expi[ind] = -1
                list_inspi = list_inspi[list_inspi != -1]
                list_expi = list_expi[list_expi != -1]
            
        if eliminate_time_shortest_ratio is not None :
            for i in range(nb_clean_loop) :
                l = list_expi - list_inspi[:-1]
                ind, = np.where(l< np.median(l)/eliminate_time_shortest_ratio )
                list_inspi[ind] = -1
                list_expi[ind] = -1
                list_inspi = list_inspi[list_inspi != -1]
                list_expi = list_expi[list_expi != -1]
                
                l = list_inspi[1:] - list_expi
                ind, = np.where(l< np.median(l)/eliminate_time_shortest_ratio )
                list_inspi[ind+1] = -1
                list_expi[ind] = -1
                list_inspi = list_inspi[list_inspi != -1]
                list_expi = list_expi[list_expi != -1]
    
    
    elif eliminate_mode == 'AND':
        # eliminate cycle with both too small duration and too small amplitude
        max_inspi = np.zeros((list_expi.size))
        for b in range(nb_clean_loop) :
            
            max_inspi = np.zeros((list_expi.size))
            for i in range(list_expi.size) :
                max_inspi[i] = np.max( np.abs(centered_sig[list_inspi[i]:list_expi[i]]) )
            l = list_expi - list_inspi[:-1]
            cond = ( max_inspi < np.median(max_inspi)/eliminate_amplitude_shortest_ratio ) & (l< np.median(l)/eliminate_time_shortest_ratio)
            ind,  = np.where(cond)
            list_inspi[ind] = -1
            list_expi[ind] = -1
            list_inspi = list_inspi[list_inspi != -1]
            list_expi = list_expi[list_expi != -1]
            
            max_expi = np.zeros((list_expi.size))
            for i in range(list_expi.size) :
                max_expi[i] = np.max( abs(centered_sig[list_expi[i]:list_inspi[i+1]]) )
            l = list_inspi[1:] - list_expi
            cond = ( max_expi < np.median(max_expi)/eliminate_amplitude_shortest_ratio) & (l< np.median(l)/eliminate_time_shortest_ratio )
            ind,  = np.where(cond)
            list_inspi[ind+1] = -1
            list_expi[ind] = -1
            list_inspi = list_inspi[list_inspi != -1]
            list_expi = list_expi[list_expi != -1]
    
    
    # STEP 5 : take crossing zeros on original_sig, last one before min for inspiration
    ind_inspi_possible, = np.where( (original_sig[:-1]<=0 ) &  (original_sig[1:]>0 ) )
    for i in range(len(list_inspi)-1) :
        ind_max = np.argmax(centered_sig[list_inspi[i]:list_expi[i]])
        ind = ind_inspi_possible[ (ind_inspi_possible>=list_inspi[i]) & (ind_inspi_possible<=list_inspi[i]+ind_max) ]
        if ind.size!=0:
            list_inspi[i] = ind.max()
    
    if output == 'index':
        cycles = -np.ones( (list_inspi.size, 2),  dtype = 'int64')
        cycles[:,0] = list_inspi
        cycles[:-1,1] = list_expi
    elif output == 'times':
        cycles = zeros( (list_inspi.size, 2),  dtype = 'float64')*np.nan
        times = np.arange(sig.size, dtype = float)/sr + t_start
        cycles[:,0] = times[list_inspi]
        cycles[:-1,1] = times[list_expi]
    
    return cycles


def generate_wavelet_fourier(len_wavelet, f_start, f_stop, delta_freq, sampling_rate, f0, normalisation):
    """
    Compute the wavelet coefficients at all scales and makes its Fourier transform.
    When different signal scalograms are computed with the exact same coefficients, 
        this function can be executed only once and its result passed directly to compute_morlet_scalogram
        
    Output:
        wf : Fourier transform of the wavelet coefficients (after weighting), Fourier frequencies are the first 
    """
    # compute final map scales
    scales = f0/np.arange(f_start,f_stop,delta_freq)*sampling_rate
    # compute wavelet coeffs at all scales
    xi=np.arange(-len_wavelet/2.,len_wavelet/2.)
    xsd = xi[:,np.newaxis] / scales
    wavelet_coefs=np.exp(complex(1j)*2.*np.pi*f0*xsd)*np.exp(-np.power(xsd,2)/2.)

    weighting_function = lambda x: x**(-(1.0+normalisation))
    wavelet_coefs = wavelet_coefs*weighting_function(scales[np.newaxis,:])

    # Transform the wavelet into the Fourier domain
    #~ wf=fft(wavelet_coefs.conj(),axis=0) <- FALSE
    wf=fftpack.fft(wavelet_coefs,axis=0)
    wf=wf.conj() # at this point there was a mistake in the original script
    
    return wf


def convolve_scalogram(sig, wf):
    """
    Convolve with fft the signal (in time domain) with the wavelet
    already computed in freq domain.
    
    Parameters
    ----------
    sig: numpy.ndarray (1D, float)
        The signal
    wf: numpy.array (2D, complex)
        The wavelet coefficient in fourrier domain.
    """
    n = wf.shape[0]
    assert sig.shape[0]<=n, 'the sig.size is longer than wf.shape[0] {} {}'.format(sig.shape[0], wf.shape[0])
    sigf=fftpack.fft(sig,n)
    wt_tmp=fftpack.ifft(sigf[:,np.newaxis]*wf,axis=0)
    wt = fftpack.fftshift(wt_tmp,axes=[0])
    return wt

def cmo_tf(sig, len_wavelet, f_start, f_stop, delta_freq, srate, f0=5, normalisation=0, return_as_da=True):
    wf = generate_wavelet_fourier(len_wavelet=len_wavelet, f_start=f_start, f_stop=f_stop, delta_freq=delta_freq, sampling_rate=srate, f0=f0, normalisation=normalisation)
    wt = convolve_scalogram(sig, wf)
    tf_matrix = np.abs(wt).T # axis 0 = freqs, axis 1 = time
    
    if return_as_da:
        da_matrix = xr.DataArray(data = tf_matrix, dims = ['freqs','time'] , coords = {'freqs':np.arange(f_start,f_stop,delta_freq), 'time':time_vector(sig, srate)})
        return da_matrix
    else:
        return tf_matrix 
    
def tf_cycle_stretch(da, chan, rsp_features, nb_point_by_cycle=1000, inspi_ratio = 0.4, save_path=None):
    # da = 3d da (chan * freqs * time)
    cycles_times_final, cycles, deformed_data = deform_to_cycle_template(data = da.loc[chan,:,:].values.T,
                                                                                                   times = da.coords['time'].values , 
                                                                                                   cycle_times=rsp_features[['inspi_time','expi_time']].values, 
                                                                                                   nb_point_by_cycle=nb_point_by_cycle, 
                                                                                                   inspi_ratio = inspi_ratio)
    if not save_path is None:
        new_rsp_features = rsp_features[ (rsp_features['inspi_time'] >= cycles_times_final[0,0]) & ( rsp_features['inspi_time'] <= cycles_times_final[-1,0]) ] # mask rsp_features to cycles kept
        new_rsp_features.to_excel(save_path)
    
    deformed = deformed_data.T
    
    shape = (cycles.size , deformed.shape[0] , nb_point_by_cycle)
    data = np.zeros(shape)
    da_stretch_cycle = xr.DataArray(data=data , dims = ['cycle','freqs','point'], coords = {'cycle' : cycles, 'freqs': da.coords['freqs'].values , 'point':np.arange(0,nb_point_by_cycle,1)})
    for cycle in cycles:
        data_of_the_cycle = deformed[:,cycle*nb_point_by_cycle:(cycle+1)*nb_point_by_cycle]
        da_stretch_cycle.loc[cycle, : , :] = data_of_the_cycle
    return da_stretch_cycle

def tf(sig, srate, f_start, f_stop, n_step, cycle_start, cycle_stop, wavelet_duration = 2, squaring=True, increase = 'linear', extracted_feature = 'power'):

    a = 1 # amplitude of the cmw
    m = 0 # max time point of the cmw
    time_cmw = np.arange(-wavelet_duration,wavelet_duration,1/srate) # time vector of the cmw

    if increase == 'linear':
        range_freqs = np.linspace(f_start,f_stop,n_step) 
    elif increase == 'log':
        log_start = np.log10(f_start)
        log_stop = np.log10(f_stop)
        range_freqs = np.logspace(log_start,log_stop,n_step) 

    n_cycles = np.linspace(cycle_start,cycle_stop,n_step) # n cycles depends on fi

    time_sig = np.arange(0, sig.size / srate , 1 / srate)

    shape = (range_freqs.size , time_sig.size)
    data = np.zeros(shape)
    dims = ['freqs','time']
    coords = {'freqs':range_freqs, 'time':time_sig}
    tf = xr.DataArray(data = data, dims = dims, coords = coords)
    
    for i, fi in enumerate(range_freqs):
        
        ni = n_cycles[i]
        cmw_f = complex_mw(a=a, time=time_cmw, n=ni, freq=fi, m = m) # make the complex mw
        complex_conv = signal.convolve(sig, cmw_f, mode = 'same')

        if extracted_feature == 'power':
            if squaring:
                module = np.abs(complex_conv) ** 2
            else:
                module = np.abs(complex_conv) # abs method without squaring (more "real")

            tf.loc[fi,:] = module
        elif extracted_feature == 'phase':
            tf.loc[fi,:] = np.angle(complex_conv)
        elif extracted_feature == 'filter':
            tf.loc[fi,:] = np.real(complex_conv)
        else:
            assert ValueError("Possible arguments in extracted_features : ['power','phase','filter']")

    return tf

def tf_power_law(tf, start_baseline, stop_baseline, method = 'decibel', show = False, center_estimator='median',decimate_factor=1): # input = 2D Xarray : freqs * time
    if center_estimator == 'mean':
        baseline_fi = tf.loc[:,start_baseline:stop_baseline].mean('time')
    elif center_estimator == 'median':
        baseline_fi = tf.loc[:,start_baseline:stop_baseline].median('time')

    if show:
        fig, ax = plt.subplots()
        ax.plot(baseline_fi.coords['freqs'].values, baseline_fi.values)
        ax.set_title('Power law of selected baseline')
        ax.set_ylabel('Power')
        ax.set_xlabel('Freqs')
        plt.show()

    tf_scaled = tf.copy()
    for fi in tf.coords['freqs'].values:
        if method == 'decibel':
            tf_scaled.loc[fi, :] = 10*np.log10(tf_scaled.loc[fi,:] / baseline_fi.loc[fi])
        elif method == 'prctchange':
            tf_scaled.loc[fi, :] = 100*((tf_scaled.loc[fi,:] - baseline_fi.loc[fi]) / baseline_fi.loc[fi])
        elif method == 'ztransform':
            std_baseline_over_time = tf.loc[fi,start_baseline:stop_baseline].std().values
            tf_scaled.loc[fi, :] = (tf_scaled.loc[fi,:] - baseline_fi.loc[fi]) / std_baseline_over_time
        elif method == 'divide':
            tf_scaled.loc[fi, :] = tf_scaled.loc[fi,:] / baseline_fi.loc[fi]

    if decimate_factor != 1:
        dims = tf_scaled.dims
        freqs = tf_scaled.coords['freqs']
        time = signal.decimate(tf_scaled.coords['time'].values, decimate_factor)
        coords = {'freqs':freqs,'time':time}
        tf_scaled = xr.DataArray(data=signal.decimate(tf_scaled, decimate_factor), dims = dims, coords = coords)
        
    return tf_scaled  

def get_itpc(da): # input = 3D Xarray : trials * freqs * time
    da_itpc = None
    for f in da.coords['freqs'].values:
        for t in da.coords['time'].values:
            phase_angles = da.loc[:,f,t]
            mean_phase_angle_over_trials = np.abs(np.mean(np.exp(1j*phase_angles)))
            if da_itpc is None:
                da_itpc = init_da({'freqs':da.coords['freqs'].values, 'time':da.coords['time'].values})
            da_itpc.loc[f,t] = mean_phase_angle_over_trials
    return da_itpc # output = 2D Xarray : freqs * time
