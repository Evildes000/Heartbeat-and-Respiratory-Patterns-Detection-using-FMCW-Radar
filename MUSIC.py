import pprint
import numpy as np
import os
import shutil
import matplotlib.pyplot as plt
# import matplotlib.animation as animation
import time
from sklearn.cluster import DBSCAN
from scipy import signal, constants
from scipy.ndimage import maximum_filter
from ifxradarsdk import get_version
from ifxradarsdk.fmcw import DeviceFmcw
from ifxradarsdk.fmcw.types import create_dict_from_sequence
from ifxradarsdk.fmcw.types import FmcwSimpleSequenceConfig, FmcwSequenceChirp, FmcwMetrics
from helpers.DistanceAlgo import *
from draw_doppler_range import *
# from findpeaks import findpeaks
import pickle
from scipy.signal import firwin, freqz, filtfilt, czt,czt_points
from scipy.linalg import norm
from scipy.signal import medfilt
from scipy.signal import hilbert
from scipy.optimize import minimize_scalar
from numpy import linalg as la
import plot_all
import KalmanFilter as kf
import nls_funcs 
import anls_kf
import music_funcs


config = FmcwSimpleSequenceConfig(
    # frame_repetition_time_s=307.325e-3,  # Frame repetition time
    frame_repetition_time_s=0.05,  # Frame repetition time
    chirp_repetition_time_s=0.000150,  # Chirp repetition time

    num_chirps=64,  # chirps per frame
    
    tdm_mimo=False,  # set True to enable MIMO mode, which is only valid for sensors with 2 Tx antennas
    chirp=FmcwSequenceChirp(
        start_frequency_Hz=59e9,  # start RF frequency, where Tx is ON
        end_frequency_Hz=62e9,  # stop RF frequency, where Tx is OFF
        sample_rate_Hz=2000000,  # ADC sample rate
        num_samples=256,  # samples per chirp
        rx_mask=7,  # RX mask is a 4-bit, each bit set enables that RX e.g. [1,3,7,15]
        tx_mask=1,  # TX antenna mask is a 2-bit (use value 3 for MIMO)
        tx_power_level=31,  # TX power level of 31
        lp_cutoff_Hz=500000,  # Anti-aliasing filter cutoff frequency, select value from data-sheet
        hp_cutoff_Hz=80000,  # High-pass filter cutoff frequency, select value from data-sheet
        if_gain_dB=30,  # IF-gain
    ),
)

class Doppler_fft():
    """compute Doppler fft""" 
    def __init__(self, num_chirps_per_frame: int, num_samples: int, mti_alpha: float = 1.0):
        self.num_chirps = num_chirps_per_frame
        self.mti_alpha = mti_alpha
        self.num_samples = num_samples
        self.mti_history = np.zeros((self.num_chirps, self.num_samples))

                # compute Blackman-Harris Window matrix over chirp samples(range)
        try:
            self.range_window = signal.blackmanharris(num_samples).reshape(1, num_samples)
        except AttributeError:
            self.range_window = signal.windows.blackmanharris(num_samples).reshape(1, num_samples)

        # compute Blackman-Harris Window matrix over number of chirps(velocity)
        try:
            self.doppler_window = signal.blackmanharris(self.num_chirps).reshape(1, self.num_chirps)
        except AttributeError:
            self.doppler_window = signal.windows.blackmanharris(self.num_chirps).reshape(1, self.num_chirps)
    


    def compute_range_fft(self, mat: np.ndarray, range_bin_length: float):
        """
        Parameters
        ----------
        mat : a frame data(num_chirps * num_samples ) of a rx antenna
        
        Return
        ------
        range_fft_result : a list. The first element is range frequency(only positive part)
                           The second is range_fft value(also only positive part)
        """
        range_fft_result = []
        [num_chirps, num_samples] = np.shape(mat)
        avrgs = np.average(mat, axis=1).reshape(num_chirps, 1)
        mat = mat - avrgs

        # perform range window on each row of mat
        mat = np.multiply(mat, self.range_window)
        # since IF is not perfect cosinus function, it might still a peak value at 0 Hz 
        
        range_fft_size = 2 * num_samples    # fft size for calculating range fft
        # velo_fft_size = 2 * config.num_chirps    # fft size for calculating velocity fft

        # apply fft on rows of mat
        range_fft = np.fft.fft(mat, axis=1 ,n=range_fft_size)[:,0:num_samples]
        # velocity_fft = np.fft.fft(range_fft, axis = 0, n=config.num_chirps)[0:config.num_chirps,:]

        # range_freq = (range_bin_length * np.fft.fftfreq(range_fft_size, d=1/config.chirp.sample_rate_Hz))[0:num_samples]
        # velo_freq = (velocity_bin_length * np.fft.fftfreq(velo_fft_size, d=1/config.chirp.sample_rate_Hz))[0:config.num_chirps]

        # plot_all.plot_range(range_fft[:,30:num_samples], range_freq)
        # plot_all.plot_range_doppler(velocity_fft, range_freq, velo_freq)

        # range_fft[:,0] = 0
        range_freq = (range_bin_length * np.fft.fftfreq(range_fft_size, d=1/config.chirp.sample_rate_Hz))[0:num_samples]
        # range_res = range_freq[1] - range_freq[0]
        """
        skip = round(0.5/(range_freq[1] - range_freq[0]))
        plt.plot(range_freq[skip:], np.abs(range_fft[1][skip:]))
        max_indix = np.argmax(np.abs(range_fft[1][skip:]))
        plt.plot(range_freq[skip:][max_indix], np.abs(range_fft[1][skip:][max_indix]), "*")
        
        plt.title("range fft")
        plt.xlabel("range/m", fontsize=12)
        plt.ylabel("amp", fontsize=12)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.show()
        """

        range_fft = range_fft[:,0:num_samples]
        range_fft_result.append(range_freq)
        range_fft_result.append(range_fft)
        return range_fft_result
    

    def compute_doppler_range_fft(self, mat:np.ndarray, range_bin: float, velo_bin: float)-> list:
        """
        Parameters
        ----------
        mat : a frame of a antenna
        range_bin : length of fast-time FFT bin
        velo_bin : length of slow-time FFT bin
        
        Return
        ------
        doppler_range_fft_result : a list
        """
        doppler_range_fft_result = []

        #mti_data = mat - self.mti_history
        #self.mti_history = mat * self.mti_alpha - (1 - self.mti_alpha) * self.mti_history
        # fft_range = target_detction(mti_data, range_bin)
        range_fft_result = self.compute_range_fft(mat, range_bin, velo_bin)

        fft1d = range_fft_result[1]
        fft1d = np.transpose(fft1d)

        fft1d = np.multiply(fft1d, self.doppler_window)
        # ffd1d_abs  = abs(fft1d)
        fft2d = np.fft.fft(fft1d, axis=1, n=2 * self.num_chirps)
        fft2d = np.fft.fftshift(fft2d,(1,))
        # rows are velocity and columns are range
        velo = velo_bin * np.fft.fftfreq(len(fft2d[0]), d=1)
        # fft2d = np.fft.fftshift(fft2d)
        # append range into doppler range fft result list
        doppler_range_fft_result.append(range_fft_result[0])
        doppler_range_fft_result.append(velo)
        doppler_range_fft_result.append(fft2d)
        doppler_range_fft_result.append(range_fft_result[1])

        # return np.fft.fftshift(fft2d)
        return doppler_range_fft_result
    



def mean_phase_elimination(mat:np.ndarray):
    """
    Calculate mean value of each column and substrate them from the input marix
    Input:
        mat: frames matrix
    """
    [r,c] = np.shape(mat)
    
    buffer = np.zeros((r,c), dtype= complex)
    for i in np.arange(c):
        m_v = np.mean(mat[:,i])
        buffer[:,i] = mat[:,i] - m_v
    
    return buffer



def max_peak_detection(mat:np.ndarray, range_freq:np.ndarray):
    """
    Detect the peak of mat

    Parameters
    ----------
    mat : fast FFT matrices
    range_freq : ranges 
    
    Return
    ------
    indices of the peak 
    """

    # print(f"shape of mat : {np.shape(mat)}")
    #3 print(f"shape of mat is: {np.shape(mat)}")
    # range_freq = range_freq * range_factor
    range_bin_length = range_freq[3] - range_freq[2]
    # print(f"range_bin_length is: {range_bin_length}")
    skip = round(0.5/range_bin_length)
    # print(f"skipped bins is: {skip}")
    # plot_all.plot_range(mat, range_fft=range_freq, skip=skip)


    #max_index = np.argmax(np.abs(mat[:,skip:]))


    eliminated_mat = mean_phase_elimination(mat)
    # plot_all.plot_range(mat, range_fft=range_freq, skip=skip)
    max_index = np.argmax(np.abs(eliminated_mat[:,skip:]))
    # plot_all.plot_range(eliminated_mat, range_fft=range_freq, skip=skip)
    """
    plt.plot(range_freq[skip:], np.abs(mat[1][skip:]))
    max_indix = np.argmax(np.abs(mat[1][skip:]))
    plt.plot(range_freq[skip:][max_indix], np.abs(mat[1][skip:][max_indix]), "*")
    # plt.title("range fft")
    plt.xlabel("range/m", fontsize=12)
    plt.ylabel("amp", fontsize=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.show()
    """
    
    # Convert flattened index to (row, column)
    max_coords = np.unravel_index(max_index, np.shape(mat[:, skip:]))
    """
    print(f"coords is: {max_coords}")

    plt.imshow(np.abs(mat),
               extent=[range_freq[0], range_freq[255], 999, 0],
               aspect='auto')
    
    plt.colorbar(label='Value')
    plt.scatter(range_freq[max_coords[1]+skip], max_coords[0], s=100, marker='*', color='red')
    plt.xlabel("range/m")
    plt.ylabel("frames")
    plt.show()
    # print(indices)
    # (indices[0])[0] += skip
    """ 
    # max_coords is a tuple and element in tuple can"t be changed, we need another list to store the coordinates
    coords = [max_coords[0], max_coords[1]+skip]
    return coords


def coherent_averaging(mat:np.ndarray):

    """
    compute average along rows of mat

    Parameters
    ----------
    mat : fast FFT matrices

    Return
    ------
    None
    """

    # sum over all chirps
    # conj_mat = np.conjugate(mat)
    # temp = np.multiply(mat, conj_mat)
    # co_averaged = np.sum(temp,axis=0)/config.num_chirps
    # skip = round(0.2/(range_freq[1] - range_freq[0]))
    co_averaged = np.mean(mat, axis=0)
    # index = np.argmax(np.abs(co_averaged[skip:]))
    # plt.plot(range_freq, np.abs(co_averaged))
    # print(f"target range is: {range_freq[index+skip]}, the index is: {index+skip}")
    # plt.plot(range_freq[index+skip], np.abs(co_averaged[skip:])[index], 'rx', markersize=10, label='Marked Point')
    # plt.grid()
    # plt.show()
    return co_averaged



def peak_region_detection(vector:np.ndarray, range_freq:np.ndarray):
    
    skip = round(0.2/(range_freq[1] - range_freq[0]))
    index = np.argmax(np.abs(vector[skip:]))
    index = index + skip
    buffer = np.zeros(5)
    # pick 2 adjacent bins on right and left side of target bin 
    buffer = vector[index - 2:index + 3]
    return buffer


def phase_correlation(mat:np.ndarray, index:int, factor: float):
    """
    To agrregate energy adjacent to the target bin and returns the displacement signal
    
    Parameters
    ----------
    mat : fast FFT matrices without static interference
    index : index of the target bin
    factor : factor that transfers phase signal to the displacement signal
    
    Return
    ------
    buffer : displacement signal that combined with adjacent bins' energy
    """
    
    target_col = mat[:,index]
    disp_target_col = factor * np.unwrap(np.angle(target_col))
    t = np.arange(len(disp_target_col)) * config.frame_repetition_time_s
    # plt.plot(t, disp_target_col, label = "target_bin")
    # plt.plot(disp_target_col)
    # plt.show()
    #print(f"phase developemnt is: {np.unwrap(np.angle(target_col))}")
    #plt.plot(np.unwrap(np.angle(target_col)))
    #plt.show()
    # std_target_col = np.std(disp_target_col)
    buffer = disp_target_col

    # compute covariance
    scan_range = 5 
    i = 1
    while i < scan_range: 
        # p means one more range bin to right
        # n means one more less range bin to left
        # bias_pcol = phase_difference(mat[:,index + i])
        # bias_ncol = phase_difference(mat[:,index - i])
        
        bias_pcol = mat[:,index + i]
        bias_ncol = mat[:,index - i]
        
        disp_bias_pcol = factor * np.unwrap(np.angle(bias_pcol))
        disp_bias_ncol = factor * np.unwrap(np.angle(bias_ncol))
        
        # compute pearson correlation coefficient
        # pccp = covp / (std_target_col * std_p)
        # pccn = covn / (std_target_col * std_n)
        pccp = np.corrcoef(disp_bias_pcol, disp_target_col)[0, 1]
        pccn = np.corrcoef(disp_bias_ncol, disp_target_col)[0, 1]
        # print(f"pccp/pccn is: {pccp}/{pccn}")
        threshold = 0.8
        # print(f"The {i}-th pccp/pccn is {pccp}/{pccn}")
        if pccp > threshold:
            buffer = buffer + disp_bias_pcol
            # plt.plot(t, disp_bias_pcol, label = f"r_{i}:{pccp}")
        if pccn > threshold:
            buffer = buffer + disp_bias_ncol
            # plt.plot(t, disp_bias_ncol, label = f"l_{i}:{pccn}")
        
        i = i + 1
    # buffer = buffer / i


    return  buffer



def normalization(disp: np.ndarray):
    """
    Normalize the given displacement signal into [-1,1]
    
    Parameters
    ----------
    disp : displacement signal
    
    Return
    ------
    disp : normalized displacement signal
    """
    max_value = np.max(disp)
    min_value = np.min(disp)
    # print(f"mat is: {mat}")
    # print(f"min_value is: {min_value}")
    disp = (disp - min_value) / (max_value - min_value)
    return disp


def remove_pulse_noise(disp:np.ndarray):
    """
    Remove pulse noise of differential displacement signal

    Parameters
    ----------
    disp : differential displacement signal
    
    Return
    ------
    disp : differential displacement signal without pulse noise
    """
    len_disp = len(disp)
    
    if len_disp % 3 == 0:
        for i in np.arange(1, len_disp-1):
            disp[i] = (disp[i-1] + disp[i+1]) / 2
    else:
        res = len_disp % 3
        res_vetor = np.zeros(res) 
        disp = np.concatenate((disp, res_vetor))
        for i in np.arange(1, len_disp-1):
            disp[i] = (disp[i-1] + disp[i+1]) / 2
    
    return disp



def remove_baseline(disp: np.ndarray):
    """
    Remove baseline drift of the displacement signal

    Parameters
    ----------
    disp : displacement signal

    Return
    ------
    disp : displacement signal without baseline drift

    """
    # remove baseline by using medium filter
    # print(f"type of element is: {type(disp[10])}")
    filtered = medfilt(disp, kernel_size=101)
    #plt.plot(filtered)
    #plt.title("median filter")
    #plt.show()
    disp = disp - filtered
    return disp


def kaiser_filtering_b(disp:np.ndarray):
    """
    Band-pass filter for breathing signal

    Parameters
    ----------
    disp : displacement signal

    Return
    ------
    filtered_signal : breathing signal
    """
    fs = 1 / config.frame_repetition_time_s
    nyq = fs / 2
    low_cut = 0.1 / nyq
    high_cut = 0.9 / nyq
    len_disp = len(disp)
    numtaps = 195
    beta = 6.5
    # try:
    #    range_window = signal.blackmanharris(len_disp)
    # except AttributeError:
    #     range_window = signal.windows.blackmanharris(len_disp) 
    # print(f"length of range_window is: {len(range_window)}")
    bandpass_fir = firwin(numtaps, [low_cut, high_cut], pass_zero=False, window=('kaiser', beta))
    
    # disp = disp * range_window
    #  print(f"length of disp is: {len(disp)}")
    """
    padlen = 3 * len(bandpass_fir)
    if len(disp) <= padlen:
        # Pad the signal manually (symmetric padding works well)
        pad_width = padlen - len(disp) + 1  # add a bit extra to be safe
        padded_signal = np.pad(disp, (pad_width,), mode='reflect')

        filtered_signal = filtfilt(bandpass_fir, 1.0, padded_signal)
    else:
    """
    filtered_signal = filtfilt(bandpass_fir, 1.0, disp)

    """
    # print(f"length of filtered signal: {len(filtered_signal)}")
    w, h = freqz(bandpass_fir, worN=8000)
    
    
    plt.figure(figsize=(10, 4))
    plt.plot(w / np.pi, 20 * np.log10(np.abs(h)), 'b')
    plt.title('Bandpass FIR Filter Breath')
    plt.xlabel('Normalized Frequency (×π rad/sample)', fontsize=12)
    plt.ylabel('Gain (dB)', fontsize=12)
    plt.xticks(fontsize=12)
    plt.xticks(fontsize=12)
    plt.grid(True)
    plt.ylim(-80, 5)
    plt.axvline(low_cut, color='red', linestyle='--', label='Cutoff Frequencies')
    plt.axvline(high_cut, color='red', linestyle='--')
    plt.legend()
    plt.show()
    
    """
    return filtered_signal


def kaiser_filtering_h(disp:np.ndarray):
    """
    Band-pass filter for heartbeat signal

    Parameters
    ----------
    disp : displacement signal

    Return
    ------
    filtered_signal : heartbeat signal
    """


    fs = 1 / config.frame_repetition_time_s
    nyq = fs / 2
    low_cut = 1 / nyq
    high_cut = 5 / nyq
    numtaps = 195
    beta = 6.5
    # len_disp = len(disp)
    # try:
    #    range_window = signal.blackmanharris(len_disp)
    # except AttributeError:
    #    range_window = signal.windows.blackmanharris(len_disp) 
    
    # disp = disp * range_window
    bandpass_fir = firwin(numtaps, [low_cut, high_cut], pass_zero=False, window=('kaiser', beta))
    """
    padlen = 3 * len(bandpass_fir)
    if len(disp) <= padlen:
        # Pad the signal manually (symmetric padding works well)
        pad_width = padlen - len(disp) + 1  # add a bit extra to be safe
        padded_signal = np.pad(disp, (pad_width,), mode='reflect')

        filtered_signal = filtfilt(bandpass_fir, 1.0, padded_signal)
    else:
    """
    filtered_signal = filtfilt(bandpass_fir, 1.0, disp)

    
    """
    # print(f"length of filtered signal: {len(filtered_signal)}")
    
    w, h = freqz(bandpass_fir, worN=8000)
    

    plt.figure(figsize=(10, 4))
    plt.plot(w / np.pi, 20 * np.log10(np.abs(h)), 'b')
    plt.title('Bandpass FIR Filter Heartbeat')
    plt.xlabel('Normalized Frequency (×π rad/sample)', fontsize=12)
    plt.ylabel('Gain (dB)', fontsize=12)
    plt.xticks(fontsize=12)
    plt.xticks(fontsize=12)
    plt.grid(True)
    plt.ylim(-80, 5)
    plt.axvline(low_cut, color='red', linestyle='--', label='Cutoff Frequencies')
    plt.axvline(high_cut, color='red', linestyle='--')
    plt.legend()
    plt.show()
    """
    # order = 20 

    # sos = signal.butter(order, [low_cut, high_cut], btype='band', fs=fs, output='sos')

    # Filter the signal
    # filtered = signal.sosfilt(sos, disp)

    return filtered_signal


def map_real_to_complex(vector:np.ndarray):
    """
    Use hilbert function to transfer real signal to complex signal (anaytic signal)
    
    Parameters
    ----------
    disp : displacement signal

    Return
    ------
    buffer : complex displacement signal 
    """
    # use hilbert function to transfer real signal to complex signal(anaytic signal)
    # hilbert function 
    buffer = hilbert(vector)
    return buffer



def dc_remove(disp:np.ndarray):
    """
    Remove dc of displacement signal

    Parameters
    ----------
    disp : displacement signal

    Return
    ------
    disp : displacement signal without dc
    """
    disp = disp - np.sum(disp)/len(disp)
    return disp


def fisrt_order_differentiators(sig:np.ndarray, sample_period:float):
    """
    Calculate the fisrt order difference of the input signal

    Parameters
    ----------
    sig : displacement signal
    sample_period : sampling period (1/fs)
    
    Return
    ------
    buffer :  fisrt order differential signal

    """
    extra = np.zeros(3)
    extra[0] = sig[0]
    extra[1] = sig[0]
    extra[2] = sig[0]
    padded_sig = np.concatenate((extra,sig))

    extra[0] = sig[-1]
    extra[1] = sig[-1]
    extra[2] = sig[-1]
    padded_sig = np.concatenate((padded_sig,extra))
    buffer = np.zeros(len(sig))
    
    for i in np.arange(3, len(sig)):
        # each time taking 7 samples from sig 
        seg_sig = padded_sig[i-3:i+4]
        # print(f"length of seg_sig is: {len(seg_sig)}")
        # calculate the first order difference 
        buffer[i-3] = (5 * (seg_sig[4] - seg_sig[2]) + 4 *  (seg_sig[5] - seg_sig[1]) + (seg_sig[6] -  seg_sig[0]))/ (32 * sample_period)

    return buffer


def ANF(disp:np.ndarray, breath_rate:float, harnic:int):
    """
    Use notch filter to filter out sepcific harmonic

    Parameters
    ----------
    disp : heartbeat signal
    breath_arte : breathing rate
    harmoic : which harmonic frequency will be suppressed

    Return
    ------
    filtered : heartbeat signal without the harmonic frequency
    """


    omega1 = 2*np.pi*harnic*breath_rate / (1/config.frame_repetition_time_s)  # notch frequency (adjust as needed)
    mu = 0.1            # notch width (0 < mu < 1)

    # Numerator and denominator coefficients
    b = [1, -2 * np.cos(omega1), 1]
    a = [1, -2 * (1 - mu / 2) * np.cos(omega1), 1 - mu]
    
    # Frequency response
    w, h = signal.freqz(b, a, worN=1000)
    filtered = signal.lfilter(b, a, disp)

    """
        # Plotting
    plt.figure(figsize=(12, 5))

    # Magnitude response
    plt.subplot(1, 2, 1)
    plt.plot(w / np.pi, 20 * np.log10(abs(h)))
    plt.title('Magnitude Response')
    plt.xlabel('Normalized Frequency [×π rad/sample]')
    plt.ylabel('Magnitude [dB]')
    plt.grid(True)

    # Phase response
    plt.subplot(1, 2, 2)
    plt.plot(w / np.pi, np.angle(h))
    plt.title('Phase Response')
    plt.xlabel('Normalized Frequency [×π rad/sample]')
    plt.ylabel('Phase [radians]')
    plt.grid(True)

    plt.tight_layout()
    plt.show()
    """

    return filtered

def heart_sig_average(disp:np.ndarray):
    """
    This function averages heartbeat signal
    
    Parameters
    ----------
    disp : heartbeat signal

    Return
    ------
    buffer : averaged heartbeat signal
    """
    len_win = 10
    win_start = 0
    win_end = len_win
    step = 1
    len_buffer = len(disp)-len_win
    buffer = np.zeros(len_buffer)
    # for index, _ in np.ndenumerate(disp):
    i = 0
    while win_end <= len_buffer:
        buffer[i] = np.average(disp[win_start:win_end])
        i = i + 1
        win_start = win_start + step
        win_end = win_end + step
    
    return buffer
    


def plot_music(heart_wave, m_list, fs):
    """
    This fucntion plots music spectrum with different lags

    Parameters
    ----------

    heart_wave : heartbeat signal
    m_lsit : list of different lags
    fs : slow time sampling frequency

    """
    corr_mat_50 = music_funcs.autocorr_mat(heart_wave, m_list[0])
    corr_mat_150 = music_funcs.autocorr_mat(heart_wave, m_list[1])
    corr_mat_250 = music_funcs.autocorr_mat(heart_wave, m_list[2])
    corr_mat_350 = music_funcs.autocorr_mat(heart_wave, m_list[3])
    corr_mat_450 = music_funcs.autocorr_mat(heart_wave, m_list[4])
    corr_mat_550 = music_funcs.autocorr_mat(heart_wave, m_list[5])

    [freq_region_50, music_heart_50]   = music_funcs.decompose(corr_mat_50, m_list[0], fs)  # k = 50
    [freq_region_150, music_heart_150] = music_funcs.decompose(corr_mat_150, m_list[1], fs)  # k = 150
    [freq_region_250, music_heart_250] = music_funcs.decompose(corr_mat_250, m_list[2], fs)  # k = 250
    [freq_region_350, music_heart_350] = music_funcs.decompose(corr_mat_350, m_list[3], fs)  # k = 350
    [freq_region_450, music_heart_450] = music_funcs.decompose(corr_mat_450, m_list[4], fs)  # k = 450
    [freq_region_550, music_heart_550] = music_funcs.decompose(corr_mat_550, m_list[5], fs)  # k = 550


    plt.subplot(3,2,1)
    plt.plot(freq_region_50, music_heart_50,label="K=50")
    #plt.xlabel("Frequency/Hz",fontsize=12)
    plt.ylabel("Amplitude/-",fontsize=12)
    #plt.title("music spectrum",fontsize=12)
    #plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend()
    plt.grid()

    plt.subplot(3,2,2)
    plt.plot(freq_region_150, music_heart_150,label="K=150")
    #plt.xlabel("Frequency/Hz",fontsize=12)
    #plt.ylabel("Amplitude/-",fontsize=12)
    #plt.title("music spectrum",fontsize=12)
    #plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend()
    plt.grid()

    plt.subplot(3,2,3)
    plt.plot(freq_region_250, music_heart_250,label="K=250")
    #plt.xlabel("Frequency/Hz",fontsize=12)
    plt.ylabel("Amplitude/-",fontsize=12)
    #plt.title("music spectrum",fontsize=12)
    #plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend()
    plt.grid()

    plt.subplot(3,2,4)
    plt.plot(freq_region_350, music_heart_350,label="K=350")
    #plt.xlabel("Frequency/Hz",fontsize=12)
    #plt.ylabel("Amplitude/-",fontsize=12)
    #plt.title("music spectrum",fontsize=12)
    #plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend()
    plt.grid()

    plt.subplot(3,2,5)
    plt.plot(freq_region_450, music_heart_450,label="K=450")
    plt.xlabel("Frequency/Hz",fontsize=12)
    plt.ylabel("Amplitude/-",fontsize=12)
    # plt.title("music spectrum",fontsize=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend()
    plt.grid()

    plt.subplot(3,2,6)
    plt.plot(freq_region_550, music_heart_550,label="K=550")
    plt.xlabel("Frequency/Hz",fontsize=12)
    #plt.ylabel("Amplitude/-",fontsize=12)
    #plt.title("music spectrum",fontsize=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend()
    plt.grid()


    plt.show()






def music_analysis(mat:np.ndarray, case:str): 
    """
    Analysis the raw data from the three antennas sperately and compute the average of them as the final result

    Parameters
    ----------
    mat : 4D Matrices with the shape (number of frames * number of antennas * number of chirps per frame * number of samples per chirp)
    case : name of volunteers and in which state are the volunteers (before or after sport)

    Returns
    -------
    None 
    """

    bandwidth = config.chirp.end_frequency_Hz - config.chirp.start_frequency_Hz
    fsta = config.chirp.start_frequency_Hz
    lamda = constants.c / fsta
    velo_bin_res = lamda / (2* config.chirp.num_samples / config.chirp.sample_rate_Hz)
    # range_bin_res = constants.c * (config.chirp.num_samples / config.chirp.sample_rate_Hz) / (2 * bandwidth)
    range_bin_res = constants.c * (config.chirp.num_samples / config.chirp.sample_rate_Hz) / (2 * bandwidth)

    print(f"range bin is: {range_bin_res}, velo bin is: {velo_bin_res} ")
    # used to transfer frequency to range
    disp_factor = lamda / (4*np.pi)
    num_antennas = 3
    # fft_freq = np.zeros(config.chirp.num_samples)

    frames = np.shape(mat)[0]
    print(f"number of frames from wei is: {frames}")

    buffer_1 = np.zeros((frames, config.chirp.num_samples), dtype=complex)
    buffer_2 = np.zeros((frames, config.chirp.num_samples), dtype=complex)
    buffer_3 = np.zeros((frames, config.chirp.num_samples), dtype=complex)

    doppler_range_ins = Doppler_fft(config.num_chirps, config.chirp.num_samples)

    print(f"loading data...")
    for i_Ant in np.arange(num_antennas):
        for i_frame in np.arange(frames): 
            chirps_and_samples = mat[i_frame, i_Ant]
            # chirps_and_samples = load_frames(i_Ant, i_frame)
            # doppler_range_fft_result[3] is the range_fft(num_chirsp * num_samples)
            range_fft_result = doppler_range_ins.compute_range_fft(chirps_and_samples, 
                                                            range_bin_length=range_bin_res)
            # range_freq indicates range
            range_freq = range_fft_result[0]
            co_averaged = coherent_averaging(range_fft_result[1])
                
            if i_Ant == 0:
                buffer_1[i_frame,:] = co_averaged
                        
            if i_Ant == 1:
                buffer_2[i_frame,:] = co_averaged

            if i_Ant == 2:
                buffer_3[i_frame,:] = co_averaged
    
    # len_pw = 1200
    window_start = 0
    window_end = 600
    sliding_step = 20

    M=50
    M_list = [50, 150, 250, 350, 450, 550] # lags
    
    breath_rates_1 = np.array([])
    heart_rates_1 = np.array([])

    breath_rates_2 = np.array([])
    heart_rates_2 = np.array([])

    breath_rates_3 = np.array([])
    heart_rates_3 = np.array([])

    print(f"processing data")
    while window_end <= frames:
        # processing_window_1 = buffer_1[window_start:window_end,:]
        # processing_window_2 = buffer_2[window_start:window_end,:]
        # processing_window_3 = buffer_3[window_start:window_end,:]
        
        # print(f"shape of buffer 1 is: {np.shape(buffer_1)}")
        # print(f"shape of buffer 2 is: {np.shape(buffer_2)}")
        # print(f"shape of buffer 3 is: {np.shape(buffer_3)}")

        processing_window_1 = buffer_1[window_start:window_end,:]
        processing_window_2 = buffer_2[window_start:window_end,:]
        processing_window_3 = buffer_3[window_start:window_end,:]
        
        # print(f"shape of processing window 1 is: {np.shape(processing_window_1)}")
        # print(f"shape of processing window 2 is: {np.shape(processing_window_2)}")
        # print(f"shape of processing window 3 is: {np.shape(processing_window_3)}")

        """
        extend = [min(fft_freq[10:]), max(fft_freq[10:])/2, 0,1500]
        plt.imshow(np.abs(processing_window_3[:,10:]), extent=extend, origin='upper', aspect="auto")
        plt.gca().invert_yaxis()
        plt.tick_params(labelsize=16)
        plt.xlabel("Range/m",fontsize=16)
        plt.ylabel("Frames",fontsize=16)
        plt.show()"
        """

        max_col_1 = max_peak_detection(processing_window_1, range_freq)
        max_col_2 = max_peak_detection(processing_window_2, range_freq)
        max_col_3 = max_peak_detection(processing_window_3, range_freq)


        # plot_all.plot_mat(processing_window_1, range_freq, max_col_1)
        # plot_all.plot_mat(processing_window_2, range_freq, max_col_2)
        # plot_all.plot_mat(processing_window_3, range_freq, max_col_3)

        # combine energy in adjacent bins of the target bin together  
        disp_1 = phase_correlation(processing_window_1, max_col_1[1], disp_factor)
        disp_2 = phase_correlation(processing_window_2, max_col_2[1], disp_factor)
        disp_3 = phase_correlation(processing_window_3, max_col_3[1], disp_factor)
        # figure out why length of disps changes

        # print(f"length of the signal 1 is: {len(disp_1)}")
        # print(f"length of the signal 2 is: {len(disp_2)}")
        # print(f"length of the signal 3 is: {len(disp_3)}")

        # normalize displacement signals
        
        disp_1 = normalization(disp_1)
        disp_2 = normalization(disp_2)
        disp_3 = normalization(disp_3)

      #  plot_all.plot_displacement(disp_1=disp_1, disp_2=disp_2, disp_3=disp_3)


    
        # disp_1 = remove_baseline(disp_1)
        # disp_2 = remove_baseline(disp_2)
        # disp_3 = remove_baseline(disp_3)
        

        # use differential signal for heart rate detection 
        # diff_disp_1 = fisrt_order_differentiators(disp_1, config.frame_repetition_time_s)
        # diff_disp_2 = fisrt_order_differentiators(disp_2, config.frame_repetition_time_s)
        # diff_disp_3 = fisrt_order_differentiators(disp_3, config.frame_repetition_time_s)
        # plot_all.plot_displacement(disp_1=diff_disp_1, disp_2=disp_2, disp_3=disp_3)



        # use nomral signal for breath detection
        disp_1 = remove_baseline(disp_1)
        disp_2 = remove_baseline(disp_2)
        disp_3 = remove_baseline(disp_3)
        
        disp_1 = dc_remove(disp_1)
        disp_1 = dc_remove(disp_1)
        disp_1 = dc_remove(disp_1)
        # plot_all.plot_displacement(disp_1=disp_1, disp_2=disp_2, disp_3=disp_3)


        # print(f"length of the signal 1 is: {len(disp_1)}")
        # print(f"length of the signal 2 is: {len(disp_2)}")
        # print(f"length of the signal 3 is: {len(disp_3)}")

        # plot_all.plot_unwrapped(disp_1, disp_2, disp_3)

        # compute differential signals
        disp_1_diff = fisrt_order_differentiators(disp_1, config.frame_repetition_time_s)
        disp_2_diff = fisrt_order_differentiators(disp_2, config.frame_repetition_time_s)
        disp_3_diff = fisrt_order_differentiators(disp_3, config.frame_repetition_time_s)


        disp_1_diff = dc_remove(disp_1_diff)
        disp_2_diff = dc_remove(disp_2_diff)
        disp_3_diff = dc_remove(disp_3_diff)

        # remove pulse noise
        disp_1_diff = remove_pulse_noise(disp_1_diff)
        disp_2_diff = remove_pulse_noise(disp_2_diff)
        disp_3_diff = remove_pulse_noise(disp_3_diff)


        breath_wave_1 = kaiser_filtering_b(disp_1)
        heart_wave_1 = kaiser_filtering_h(disp_1_diff)
        # print(f"length of signal is: {len(breath_wave_1)}")

        breath_wave_2 = kaiser_filtering_b(disp_2)
        heart_wave_2 = kaiser_filtering_h(disp_2_diff)

        breath_wave_3 = kaiser_filtering_b(disp_3)
        heart_wave_3 = kaiser_filtering_h(disp_3_diff)


        # plot_all.plot_breath_heart_disp(breath_wave_3, heart_wave_3)

        #heart_wave_1 = heart_sig_average(heart_wave_1)
        #heart_wave_2 = heart_sig_average(heart_wave_2)
        #heart_wave_3 = heart_sig_average(heart_wave_3)

        breath_autocorrmat_1 = music_funcs.autocorr_mat(breath_wave_1, M)
        breath_autocorrmat_2 = music_funcs.autocorr_mat(breath_wave_2, M)
        breath_autocorrmat_3 = music_funcs.autocorr_mat(breath_wave_3, M)


        [freq_region, music_breath_1] = music_funcs.decompose(breath_autocorrmat_1, M, 1/config.frame_repetition_time_s)
        [freq_region, music_breath_2] = music_funcs.decompose(breath_autocorrmat_2, M, 1/config.frame_repetition_time_s)

                    
        [freq_region, music_breath_3] = music_funcs.decompose(breath_autocorrmat_3, M, 1/config.frame_repetition_time_s)
        
        breath_rate_1 = music_funcs.peak_selection(freq_region, music_breath_1)
        breath_rate_2 = music_funcs.peak_selection(freq_region, music_breath_2)
        breath_rate_3 = music_funcs.peak_selection(freq_region, music_breath_3)

        print(f"breath rates are: {breath_rate_1}, {breath_rate_2}, {breath_rate_3}")

        heart_wave_1 = ANF(heart_wave_1, breath_rate_1, 4)
        heart_wave_2 = ANF(heart_wave_2, breath_rate_2, 4)
        heart_wave_3 = ANF(heart_wave_3, breath_rate_3, 4)

        # comp_heart_wave_1 = map_real_to_complex(heart_wave_1)
        # comp_heart_wave_2 = map_real_to_complex(heart_wave_2)
        # comp_heart_wave_3 = map_real_to_complex(heart_wave_3)

        heart_autocorrmat_1 = music_funcs.autocorr_mat(heart_wave_1, M)
        heart_autocorrmat_2 = music_funcs.autocorr_mat(heart_wave_2, M)
        heart_autocorrmat_3 = music_funcs.autocorr_mat(heart_wave_3, M)

        [freq_region, music_heart_1] = music_funcs.decompose(heart_autocorrmat_1, M, 1/config.frame_repetition_time_s)
        [freq_region, music_heart_2] = music_funcs.decompose(heart_autocorrmat_2, M, 1/config.frame_repetition_time_s)

        # plot_music(heart_wave_3, M_list, 1/config.frame_repetition_time_s)
        [freq_region, music_heart_3] = music_funcs.decompose(heart_autocorrmat_3, M, 1/config.frame_repetition_time_s)
        
        heart_rate_1 = music_funcs.peak_selection(freq_region, music_heart_1)
        heart_rate_2 = music_funcs.peak_selection(freq_region, music_heart_2)
        heart_rate_3 = music_funcs.peak_selection(freq_region, music_heart_3)

        print(f"heart rates are: {heart_rate_1}, {heart_rate_2}, {heart_rate_3}")

        breath_rates_1 = np.append(breath_rates_1, breath_rate_1)
        breath_rates_2 = np.append(breath_rates_2, breath_rate_2)
        breath_rates_3 = np.append(breath_rates_3, breath_rate_3)
        

        heart_rates_1 = np.append(heart_rates_1, heart_rate_1)
        heart_rates_2 = np.append(heart_rates_2, heart_rate_2)
        heart_rates_3 = np.append(heart_rates_3, heart_rate_3)

        window_start = window_start + sliding_step
        window_end = window_end + sliding_step

    
    breath_rates_1 = np.array(breath_rates_1) * 60
    heart_rates_1 = np.array(heart_rates_1) * 60

    breath_rates_2 = np.array(breath_rates_2) * 60
    heart_rates_2 = np.array(heart_rates_2) * 60

    breath_rates_3 = np.array(breath_rates_3) * 60
    heart_rates_3 = np.array(heart_rates_3) * 60
    
    breath_rates = (breath_rates_1 + breath_rates_2 + breath_rates_3) / 3
    heart_rates = (heart_rates_1 + heart_rates_2 + heart_rates_3) / 3
    plot_all.plot_hrs_brs(breath_rates, heart_rates, case)

    np.save("masterarbeit-radar/candidates_hrs_brs_combine_music/"+case+"_breath_rates", breath_rates)
    np.save("masterarbeit-radar/candidates_hrs_brs_combine_music/"+case+"_heart_rates", heart_rates)
    














if __name__ == '__main__':
    
    # delte_previous_data()
    # make_dir()
    # record_data() 
    
    # daphne = np.load('daphne/RadarIfxAvian_00/radar.npy')
    # wei = np.load('wei/RadarIfxAvian_00/radar.npy')
    # wei = np.load('wei/wei_data_2/RadarIfxAvian_00/radar.npy')
    
    name_folder_dict = {#"benjamin_before":"masterarbeit-radar/record_on_0909/Benjamin_before20250909-100656/RadarIfxAvian_00/radar.npy",
                        #"benjamin_after":"masterarbeit-radar/record_on_0909/Benjamin_after20250909-101734/RadarIfxAvian_00/radar.npy",
                        #"daphne_before":"masterarbeit-radar/record_on_0909/daphne_before20250909-102441/RadarIfxAvian_00/radar.npy",
                        
                        #"daphne_after":"masterarbeit-radar/record_on_0909/daphne_after20250909-103516/RadarIfxAvian_00/radar.npy",
                        #"roman_before":"masterarbeit-radar/record_on_0909/roman_before20250909-152453/RadarIfxAvian_00/radar.npy",
                        #"roman_after":"masterarbeit-radar/record_on_0909/roman_after20250909-153726/RadarIfxAvian_00/radar.npy",
                        #"stephan_before":"masterarbeit-radar/record_on_0909/stephan_before20250909-150057/RadarIfxAvian_00/radar.npy",
                        "stephan_after":"masterarbeit-radar/record_on_0909/stephan_after20250909-151129/RadarIfxAvian_00/radar.npy",
                        #"hassan_before":"masterarbeit-radar/record_on_0909/hassan_before20250916-135424/RadarIfxAvian_00/radar.npy",
                        #"hassan_after":"masterarbeit-radar/record_on_0909/hassan_after20250916-140450/RadarIfxAvian_00/radar.npy",
                        #"alex_before":"masterarbeit-radar/record_on_0909/alex_before20250916-143105/RadarIfxAvian_00/radar.npy",
                        #"alex_after":"masterarbeit-radar/record_on_0909/alex_after20250916-144220/RadarIfxAvian_00/radar.npy",
                        #"nils_before":"masterarbeit-radar/record_on_0909/nils_before20250916-145102/RadarIfxAvian_00/radar.npy",
                        #"nils_after":"masterarbeit-radar/record_on_0909/nils_after20250916-150139/RadarIfxAvian_00/radar.npy",
                        #"awis_before":"masterarbeit-radar/record_on_0909/awis_before20250916-152604/RadarIfxAvian_00/radar.npy",
                        "awis_after":"masterarbeit-radar/record_on_0909/awis_after20250916-153822/RadarIfxAvian_00/radar.npy"}
                      

    for key in name_folder_dict:
        mat = np.load(name_folder_dict[key])
        music_analysis(mat, key)