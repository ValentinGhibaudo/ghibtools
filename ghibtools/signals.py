import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy import fftpack
import xarray as xr
import pandas as pd
 
def get_wsize(srate, lowest_freq , n_cycles=5):
    nperseg = ( n_cycles / lowest_freq) * srate
    return int(nperseg)

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

def down_sample(sig, factor, axis=-1): 
    sig_down = signal.decimate(sig, q=factor, n=None, ftype='iir', axis=axis, zero_phase=True)
    return sig_down

def spectre(sig, srate, lowest_freq, n_cycles = 5, nfft_factor = 1, axis = -1, scaling = 'spectrum', verbose = False):

    """
    Compute Power Spectral Density of the signal with Welch method

    -----------------
    Inputs =
    - sig : Nd array with time in last dim
    - srate : samping rate
    - lowest_freq : Lowest frequency of interest, window sizes will be automatically computed based on this freq and set min number of cycle in window
    - n_cycles : Minimum cycles of the lowest frequency in the window size (default = 5)
    - nfft_factor : Factor of zero-padding (default = 1)
    - verbose : if True, print informations about windows length (default = False)
    - scaling : 'spectrum' or 'density' (cf scipy.signal.welch) (default = 'scaling')

    Outputs = 
    - f : frequency vector
    - Pxx : Power Spectral Density vector (scaling = spectrum so unit = V**2)

    """

    nperseg = get_wsize(srate, lowest_freq, n_cycles)
    nfft = int(nperseg * nfft_factor)
    f, Pxx = signal.welch(sig, fs=srate, nperseg = nperseg , nfft = nfft, scaling=scaling, axis=axis)

    if verbose:
        n_windows = 2 * sig.size // nperseg
        print(f'nperseg : {nperseg}')
        print(f'sig size : {sig.size}')
        print(f'total cycles lowest freq : {int(sig.size / ((1 / lowest_freq)*srate))}')
        print(f'nwindows : {n_windows}')

    return f, Pxx

def coherence(sig1,sig2, srate, lowest_freq, n_cycles = 5, nfft_factor = 2):
    nperseg = get_wsize(srate, lowest_freq, n_cycles)
    nfft = nperseg * nfft_factor
    f, Cxy = signal.coherence(sig1,sig2, fs=srate, nperseg = nperseg , nfft = nfft )
    return f, Cxy

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


def gaussian_win(a , time, m , n , freq):
    
    # a = 2 # amplitude
    # time = time # time
    # m  = time[-1] / 2 # offset (not relevant for eeg analyses and can be always set to zero)
    # n = 10 # refers to the number of wavelet cycles , defines the trade-off between temporal precision and frequency precision
    # freq = 10 # change the width of the gaussian and therefore of the wavelet
    # s = n / (2 * np.pi * freq) # std of the width of the gaussian, freq = in Hz
    s = n / (2 * np.pi * freq)
    GaussWin = a * np.exp( -(time - m)** 2 / (2 * s**2))
    
    return GaussWin

def morlet_wavelet(a , time, m , n , freq):
    s = n / (2 * np.pi * freq)
    GaussWin = a * np.exp( -(time - m)** 2 / (2 * s**2))
    SinWin = np.sin ( 2 * np.pi * freq * time )
    MorletWavelet = GaussWin * SinWin
    return MorletWavelet

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
                da_itpc = xr.DataArray(data = np.nan,
                                       dims = ['freqs','time'],
                                       coords = {'freqs':da.coords['freqs'].values, 'time':da.coords['time'].values}
                                      )
            da_itpc.loc[f,t] = mean_phase_angle_over_trials
    return da_itpc # output = 2D Xarray : freqs * time

def iirfilt(sig, srate, lowcut=None, highcut=None, order = 4, ftype = 'butter', verbose = False, show = False, axis = 0):

    """
    IIR-Filter of signal

    -------------------
    Inputs : 
    - sig : nd array
    - srate : sampling rate of the signal
    - lowcut : lowcut of the filter. Lowpass filter if lowcut is None and highcut is not None
    - highcut : highcut of the filter. Highpass filter if highcut is None and low is not None
    - order : N-th order of the filter (the more the order the more the slope of the filter)
    - ftype : Type of the IIR filter, could be butter or bessel
    - verbose : if True, will print information of type of filter and order (default is False)
    - show : if True, will show plot of frequency response of the filter (default is False)
    """

    if lowcut is None and not highcut is None:
        btype = 'lowpass'
        cut = highcut

    if not lowcut is None and highcut is None:
        btype = 'highpass'
        cut = lowcut

    if not lowcut is None and not highcut is None:
        btype = 'bandpass'

    if btype in ('bandpass', 'bandstop'):
        band = [lowcut, highcut]
        assert len(band) == 2
        Wn = [e / srate * 2 for e in band]
    else:
        Wn = float(cut) / srate * 2

    filter_mode = 'sos'
    sos = signal.iirfilter(order, Wn, analog=False, btype=btype, ftype=ftype, output=filter_mode)

    filtered_sig = signal.sosfiltfilt(sos, sig, axis=axis)

    if verbose:
        print(f'{ftype} iirfilter of {order}th-order')
        print(f'btype : {btype}')


    if show:
        w, h = signal.sosfreqz(sos,fs=srate, worN = 2**18)
        fig, ax = plt.subplots()
        ax.plot(w, np.abs(h))
        ax.scatter(w, np.abs(h), color = 'k', alpha = 0.5)
        full_energy = w[np.abs(h) >= 0.99]
        ax.axvspan(xmin = full_energy[0], xmax = full_energy[-1], alpha = 0.1)
        ax.set_title('Frequency response')
        ax.set_xlabel('Frequency [Hz]')
        ax.set_ylabel('Amplitude')
        plt.show()

    return filtered_sig


def complex_mw(time, n_cycles , freq, a= 1, m = 0): 
    """
    Create a complex morlet wavelet by multiplying a gaussian window to a complex sinewave of a given frequency
    
    ------------------------------
    a = amplitude of the wavelet
    time = time vector of the wavelet
    n_cycles = number of cycles in the wavelet
    freq = frequency of the wavelet
    m = 
    """
    s = n_cycles / (2 * np.pi * freq)
    GaussWin = a * np.exp( -(time - m)** 2 / (2 * s**2)) # real gaussian window
    complex_sinewave = np.exp(1j * 2 *np.pi * freq * time) # complex sinusoidal signal
    cmw = GaussWin * complex_sinewave
    return cmw

def morlet_family(srate, f_start, f_stop, n_steps, n_cycles):
    """
    Create a family of morlet wavelets
    
    ------------------------------
    srate : sampling rate
    f_start : lowest frequency of the wavelet family
    f_stop : highest frequency of the wavelet family
    n_steps : number of frequencies from f_start to f_stop
    n_cycles : number of waves in the wavelet
    """
    tmw = np.arange(-5,5,1/srate)
    freqs = np.linspace(f_start,f_stop,n_steps) 
    mw_family = np.zeros((freqs.size, tmw.size), dtype = 'complex')
    for i, fi in enumerate(freqs):
        mw_family[i,:] = complex_mw(tmw, n_cycles = n_cycles, freq = fi)
    return freqs, mw_family

def morlet_power(sig, srate, f_start, f_stop, n_steps, n_cycles, amplitude_exponent=2):
    """
    Compute time-frequency matrix by convoluting wavelets on a signal
    
    ------------------------------
    Inputs =
    - sig : the signal (1D np vector)
    - srate : sampling rate
    - f_start : lowest frequency of the wavelet family
    - f_stop : highest frequency of the wavelet family
    - n_steps : number of frequencies from f_start to f_stop
    - n_cycles : number of waves in the wavelet
    - amplitude_exponent : amplitude values extracted from the length of the complex vector will be raised to this exponent factor (default = 2 = V**2 as unit)

    Outputs = 
    - freqs : frequency 1D np vector
    - power : 2D np array , axis 0 = freq, axis 1 = time

    """
    freqs, family = morlet_family(srate, f_start = f_start, f_stop = f_stop, n_steps = n_steps, n_cycles = n_cycles)
    sigs = np.tile(sig, (n_steps,1))
    tf = signal.fftconvolve(sigs, family, mode = 'same', axes = 1)
    power = np.abs(tf) ** amplitude_exponent
    return freqs , power

