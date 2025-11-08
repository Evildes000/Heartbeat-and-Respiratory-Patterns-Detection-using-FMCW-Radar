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
from scipy.signal import find_peaks
# import pickle
from scipy.signal import firwin, freqz, filtfilt, czt,czt_points
from scipy.linalg import norm
from scipy.signal import medfilt
from scipy.signal import hilbert
import plot_all
import emd

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
        # mat: a frame data(num_chirps * num_samples ) of a rx antenna
        # return: a list. The first element is range frequency(only positive part)
        # The second is range_fft value(also only positive part)
        range_fft_result = []
        [num_chirps, num_samples] = np.shape(mat)
        avrgs = np.average(mat, axis=1).reshape(num_chirps, 1)
        mat = mat - avrgs

        mat = np.multiply(mat, self.range_window)
        # since IF is not perfect cosinus function, it might still a peak value at 0 Hz 
        fft_size = 2 * num_samples
        range_fft = np.fft.fft(mat, axis=1 ,n=fft_size)

        range_fft[:,0] = 0
        range_freq = (range_bin_length * np.fft.fftfreq(fft_size, d=1/config.chirp.sample_rate_Hz))[0:num_samples]
        # range_res = range_freq[1] - range_freq[0]
        
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
        # mat: a frame of a antenna
        # range_bin: length of range bin
        doppler_range_fft_result = []
        #mti_data = mat - self.mti_history
        #self.mti_history = mat * self.mti_alpha - (1 - self.mti_alpha) * self.mti_history
        # fft_range = target_detction(mti_data, range_bin)
        range_fft_result = self.compute_range_fft(mat, range_bin)

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
    # this function averages heart signal for every 10 samples
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
    # mat: doppler_range_map
    # return indices of max peak
    # print(f"shape of mat : {np.shape(mat)}")
    #3 print(f"shape of mat is: {np.shape(mat)}")
    range_bin_length = range_freq[3] - range_freq[2]
    skip = round(0.5/range_bin_length)
    eliminated_mat = mean_phase_elimination(mat)

    max_index = np.argmax(np.abs(eliminated_mat[:,skip:]))
    # plot_all.plot_range(np.abs(mat[1]), range_fft=range_freq, skip=skip)
    # plot_all.plot_frame_matrix(mat, range_fft=range_freq, skip=skip)
    # plot_all.plot_range(np.abs(eliminated_mat[1]), range_fft=range_freq, skip=skip)
    # plot_all.plot_frame_matrix(eliminated_mat, range_fft=range_freq, skip=skip)
    # Convert flattened index to (row, column)
    max_coords = np.unravel_index(max_index, np.shape(mat[:, skip:]))
    """
    print(f"coords is: {max_coords}")

    plt.imshow(np.abs(mat[:,skip:]),
               extent=[range_freq[0], range_freq[255], 999, 0],
               aspect='auto')
    
    plt.colorbar(label='Value')
    plt.scatter(range_freq[max_coords[1]+skip], max_coords[0], s=100, marker='*', color='red')
    plt.xlabel("range/m")
    plt.ylabel("frames")
    plt.show()
    """
    # print(indices)
    # (indices[0])[0] += skip
    
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
    plt.plot(t, disp_target_col, label = "target_bin")
    # plt.plot(disp_target_col)
    # plt.show()
    #print(f"phase developemnt is: {np.unwrap(np.angle(target_col))}")
    #plt.plot(np.unwrap(np.angle(target_col)))
    #plt.show()
    # std_target_col = np.std(disp_target_col)
    # phase_buffer = target_col
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
            # phase_buffer = phase_buffer + bias_pcol
            plt.plot(t, disp_bias_pcol, label = f"dr_{i}:{pccp}")
            print(pccp)
            print("r")
        if pccn > threshold:
            buffer = buffer + disp_bias_ncol
            # phase_buffer = phase_buffer + bias_ncol
            plt.plot(t, disp_bias_ncol, label = f"dl_{i}:{pccn}")
            print(pccp)
            print("l")
        
        i = i + 1
    # buffer = buffer / i
    # plt.plot(t, buffer, )
    plt.xlabel("t/s", fontsize=12)
    plt.ylabel("Amplitude/m",fontsize=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid()
    plt.legend()
    plt.show()
    # [imag, real] = iq_balance(phase_buffer)
    # another_buffer = factor * np.unwrap(np.angle(imag+1j*real)) 
    # return  [buffer, another_buffer]
    return buffer


def normalization(mat: np.ndarray):
    """
    Normalize the given displacement signal into [-1,1]
    
    Parameters
    ----------
    disp : displacement signal
    
    Return
    ------
    disp : normalized displacement signal
    """
    max_value = np.max(mat)
    min_value = np.min(mat)
    # print(f"mat is: {mat}")
    # print(f"min_value is: {min_value}")
    mat = (mat - min_value) / (max_value - min_value)
    return mat


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
    # print(f"type of element is: {type(disp[10])}")
    filtered = medfilt(disp, kernel_size=51)
    len_disp = len(disp)
    t = np.arange(len_disp) * config.frame_repetition_time_s
    plt.plot(t, 100*filtered)
    plt.xlabel("t/s", fontsize=12)
    plt.ylabel("Amplitude/mm", fontsize=12)
    plt.ticklabel_format(style="sci", axis="y", scilimits=(0,0))
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid()
    plt.show()
    # plt.title("median filter")
    plt.show()
    disp = disp - filtered
    return disp



def plot_filters():

    fs = 1 / config.frame_repetition_time_s
    nyq = fs / 2
    numtaps = 121
    beta = 6.5
    b_low_cut = 0.1 / nyq
    b_high_cut = 0.9 / nyq
    h_low_cut = 0.8 / nyq
    h_high_cut = 3 / nyq

    b_bandpass_fir = firwin(numtaps, [b_low_cut, b_high_cut], pass_zero=False, window=('kaiser', beta))
    h_bandpass_fir = firwin(numtaps, [h_low_cut, h_high_cut], pass_zero=False, window=('kaiser', beta))
    b_w, b_h = freqz(b_bandpass_fir, worN=8000)
    h_w, h_h = freqz(h_bandpass_fir, worN=8000)

    plt.figure()
    plt.subplot(2,1,1)
    plt.plot(b_w / np.pi, 20 * np.log10(np.abs(b_h)), 'b')
    # plt.title('Bandpass FIR Filter Breathing')
    # plt.xlabel('Normalized Frequency (×π rad/sample)', fontsize=12)
    plt.ylabel('Gain/dB', fontsize=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True)
    plt.ylim(-80, 5)
    plt.axvline(b_low_cut, color='red', linestyle='--', label='Cutoff Frequencies')
    plt.axvline(b_high_cut, color='red', linestyle='--')
    plt.legend()

    plt.subplot(2,1,2)
    plt.plot(h_w / np.pi, 20 * np.log10(np.abs(h_h)), 'b')
    # plt.title('Bandpass FIR Filter heart')
    plt.xlabel('Normalized Frequency/(×π rad/sample)',fontsize=12)
    plt.ylabel('Gain/dB',fontsize=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True)
    plt.ylim(-80, 5)
    plt.axvline(h_low_cut, color='red', linestyle='--', label='Cutoff Frequencies')
    plt.axvline(h_high_cut, color='red', linestyle='--')
    plt.legend()


    plt.show()


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
    
    numtaps = 121
    beta = 6.5

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

    # print(f"length of filtered signal: {len(filtered_signal)}")
    # c
    
    """
    plt.figure(figsize=(10, 4))
    plt.plot(w / np.pi, 20 * np.log10(np.abs(h)), 'b')
    plt.title('Bandpass FIR Filter Breath')
    plt.xlabel('Normalized Frequency (×π rad/sample)')
    plt.ylabel('Gain (dB)')
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
    low_cut = 0.8 / nyq
    high_cut = 3 / nyq
    numtaps = 121
    beta = 6.5

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

    # print(f"length of filtered signal: {len(filtered_signal)}")
    
    w, h = freqz(bandpass_fir, worN=8000)
    
    """
    plt.figure(figsize=(10, 4))
    plt.plot(w / np.pi, 20 * np.log10(np.abs(h)), 'b')
    plt.title('Bandpass FIR Filter Heart')
    plt.xlabel('Normalized Frequency (×π rad/sample)')
    plt.ylabel('Gain (dB)')
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


    plt.show()

    return filtered_signal


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
    # hanic: harmonic frequency of breath rate
    # to remove certain frequency component from the given data

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


def rubost_breath_rate_detection(disp:np.ndarray, cut_off_freq:float):
    """
    detect breathing rate both in time and frequency domin
    Parameters
    ----------

    disp : breathing signal
    cut_off_freq : opper cut off frequency of the bandpass filter for breathing signal

    Return
    ------
    breathing rate
    """

    skip = 5
    # disp_len = len(disp)
    disp_len = len(disp)
    t = np.arange(disp_len) * config.frame_repetition_time_s
    disp_fft = (np.abs(np.fft.fft(disp, n=2*disp_len))/disp_len)[0:int(disp_len/4)]
    fft_freq = np.fft.fftfreq(2*disp_len, d=config.frame_repetition_time_s)[0:int(disp_len/4)]
    index = np.argmax(disp_fft[skip:])
    ff = fft_freq[index+skip] # frequency estimated in frequency domin
    print(f"ff is: {ff}")
    peaks_indices, _ = find_peaks(disp, distance=int((1/cut_off_freq)/config.frame_repetition_time_s))
    # print(peaks_indices)
    ft = len(peaks_indices)/30 # frequency estimated in time dommin
    print(f"ft is: {ft}")
    if ff > 0.5:
        ff = ff /2
    if ft > 0.5:
        ft = ft / 2

    
    plt.plot(fft_freq,  disp_fft)
    plt.plot(fft_freq[index+skip], disp_fft[index+skip], "*")
    plt.xlabel("Frequency/Hz", fontsize=12)
    plt.ylabel("Amplitude/-", fontsize=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.ticklabel_format(style="sci",axis="y",scilimits=(0,0))
    plt.grid()
    plt.show()
    
    """
    plt.plot(t, disp)
    plt.plot(t[peaks_indices], disp[peaks_indices], "*")
    plt.show()
    """

    return min(ff, ft)



"""
def second_order_differentiators(sig:np.ndarray, sample_period:float):
    # calculate the second order difference of the input signal
    
    # padding zeros at the begining and end of the input signal
    extra = np.zeros(3)
    padded_sig = np.concatenate((extra,sig))
    padded_sig = np.concatenate((padded_sig,extra))
    buffer = np.zeros(len(sig))
    
    for i in np.arange(3, len(sig)):
        # each time taking 7 samples from sig 
        seg_sig = padded_sig[i-3:i+4]
        # print(f"length of seg_sig is: {len(seg_sig)}")
        # calculate the first order difference 
        buffer[i-3] =(4 * seg_sig[3] + (seg_sig[4] - seg_sig[2]) - 2 * (seg_sig[1] + seg_sig[5]) - (seg_sig[6] + seg_sig[0])) / (16 * (sample_period)**2)

    buffer[0] = 0
    return buffer
"""


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


def peak_detection(disp, step=10):
    """
    detect how many peaks are there within a processing window
    Parameters
    ----------
    disp : heartbeat signal
    step : minimal distance between two peaks

    Return
    ------
    indices of all peaks
    """
    len_disp = len(disp)
    len_window = 2*step + 1 # length of processing window
    start_indix = 0
    end_indix = len_window
    shift_step = 1
    peaks_indices = []
    i = 0
    while end_indix <= len_disp:
        if disp[i+step] == max(disp[start_indix:end_indix]):
            peaks_indices.append(i+step)
        start_indix = start_indix + shift_step
        end_indix = end_indix + shift_step
        i = i + 1
    
    """
    t = np.arange(len(disp)) * 0.05
    plt.plot(t, disp)
    plt.plot(t[peaks_indices], disp[peaks_indices], "*")
    plt.grid()
    plt.show()

    
    len_disp = len(disp)
    res_vector = np.zeros(step)
    # buffer = np.concatenate((res_vector, disp, res_vector))
    buffer = np.append(res_vector, disp)
    buffer = np.append(buffer, res_vector)
    peaks_indices = []
    for i in np.arange(len_disp):
        middle = i + step
        # check if the midlle value bigger than other values around it
        # others = np.concatenate((buffer[middle-step:middle], buffer[middle+1:middle+step]))
        # print(f"others is: {others}")
        # print(f"middle value is: {buffer[middle]}")
        # if buffer[middle] >= max(buffer[middle-step:middle+step]):
        if buffer[middle] == max(buffer[middle-step:middle+step]):
            peaks_indices.append(i)
    
    # indices_to_delete = np.concatenate(np.arange(step), len_disp+np.arange(step))
    """
    return np.array(peaks_indices)


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



def remove_pulse_noise(disp:np.ndarray):
    # remove pulse noise after phase difference
    len_disp = len(disp)
    buffer = np.zeros(len_disp)

    for i in np.arange(1, len_disp-1):
        buffer[i] = (disp[i-1] + disp[i+1]) / 2
    buffer[0] = disp[0]
    buffer[-1] = disp[-1]

    return buffer


def czt_analysis(sig:np.ndarray ,coarse_hf:float, fs:float):
    # sig: 
    # f: detected frequency
    # using czt to have more accurate frequency
    a = np.exp(-coarse_hf/fs)
    # num_points points after start points in the circle
    disp_len = len(sig)
    num_points = disp_len
    w = np.exp(-1j*np.pi/disp_len)

    # disp_fft_heart = (np.abs(np.fft.fft(sig, n=2*disp_len))/disp_len)[0:int(disp_len/8)]
    
    points = czt_points(m=num_points, w=w, a=a)
    czt_result = np.abs(czt(x=sig, a=a, w=w, m=num_points))[0:int(disp_len/8)]
    freqs = (np.angle(points)*fs/(2*np.pi))[0:int(disp_len/8)]
    # max = np.argmax(czt_result)

    # detect heart rate in the more precious spectrum
    peaks_indices = peak_detection(czt_result,step=3)
    candidate_freqs = freqs[peaks_indices]
    freqs_diff = np.abs(candidate_freqs - coarse_hf)

    
    index = np.argmin(freqs_diff)
    hr = candidate_freqs[index]

    """
    plt.figure(0)
    #plt.plot(fft_freq, disp_fft_heart, label = "heart_fft")
    plt.plot(freqs, czt_result, label = "heart_czt")
    plt.plot(freqs[peaks_indices], czt_result[peaks_indices], "x", label = "peaks")
    # plt.text(fft_freq[peaks_indices], disp_fft_heart[peaks_indices], disp_fft_heart[peaks_indices])
    plt.plot(freqs[peaks_indices][index] ,czt_result[peaks_indices][index], "o", label = "estimated hr")
    plt.xlabel("Hz",fontsize=16)
    plt.ylabel("amplitude",fontsize=16)
    plt.legend()
    plt.tick_params(labelsize=16)
    plt.grid()
    plt.show()    
    """

    if hr < 1:
        return coarse_hf * 60
    else:
        return hr * 60



def heartrate_estimation_fft(disp:np.ndarray, cut_off_freq:float):
    """
    Estimate heart rate in time region and frequency region
    
    Parameters
    ----------
    disp : heartbeat signal
    cut_off_freq : upper cut off frequency of the bandpass filter for heartbeat signal
    
    Retrun
    ------
    Heart rate
    """
    disp = (disp - np.mean(disp)) / (np.std(disp) + 1e-12)

    peak_indices,_ = find_peaks(disp, prominence=1, distance=int((1/cut_off_freq)/config.frame_repetition_time_s))
    ft = len(peak_indices)/30
    n_size = 1024
    t = np.arange(len(disp)) * config.frame_repetition_time_s
    disp_fft = np.abs(np.fft.fft(disp, n_size))[0:int(n_size/8)] 
    fft_freq = np.fft.fftfreq(n_size, d=config.frame_repetition_time_s)[0:int(n_size/8)]
    
    freq_res = fft_freq[1] - fft_freq[0] # 0.0195
    # print(f"freq_res is: {freq_res}")
    start_f = int((ft - 0.33)/freq_res)
    end_f = int((ft + 0.33)/freq_res)
    # print(f"start: {start_f}, end: {end_f}")
    ff_arr, _ = find_peaks(disp_fft[start_f:end_f], distance=int(0.05/freq_res))
    ff_indix = np.argmin(np.abs(fft_freq[start_f:end_f][ff_arr] - ft))

    """    
    fig, axs = plt.subplots(2,1)
    axs[0].plot(t, disp)
    axs[0].plot(t[peak_indices], disp[peak_indices], "*")
    axs[0].legend()

    axs[1].plot(fft_freq, disp_fft)
    axs[1].plot(fft_freq[start_f:end_f][ff_arr], disp_fft[start_f:end_f][ff_arr], "*")
    axs[1].plot(fft_freq[start_f:end_f][ff_arr][ff_indix], disp_fft[start_f:end_f][ff_arr][ff_indix], "o")
    axs[1].legend()
    """

    
    plt.plot(t, disp)
    plt.plot(t[peak_indices], disp[peak_indices], "*")
    # plt.title("peaks of heartbeat signal", fontsize=12)
    plt.xlabel("t/s", fontsize=12)
    plt.ylabel("Amplitude/mm", fontsize=12)
    plt.grid()
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.show()
    
    
    plt.plot(fft_freq, disp_fft)
    plt.plot(fft_freq[start_f:end_f][ff_arr], disp_fft[start_f:end_f][ff_arr], "*")
    plt.plot(fft_freq[start_f:end_f][ff_arr][ff_indix], disp_fft[start_f:end_f][ff_arr][ff_indix], "o")
    plt.xlabel("Frequency/Hz", fontsize=12)
    plt.ylabel("Amplitude/--", fontsize=12)
    plt.grid()
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.show()
    
    return fft_freq[start_f:end_f][ff_arr][ff_indix]

    """
    start_indice = 0
    end_indice = 200
    step = 20 # 1s
    n_size = 1024
    buffer = np.array([])
    while end_indice <= 1200:
        sliced = disp[start_indice:end_indice]
        disp_fft = np.abs(np.fft.fft(sliced, n_size))[0:int(n_size/2)]
        fft_freq = np.fft.fftfreq(n_size, d=config.frame_repetition_time_s)[0:int(n_size/2)]
        plt.plot(fft_freq, disp_fft)
        plt.show()
        max_indix = np.argmax(disp_fft)
        freq = fft_freq[max_indix]
        print(f"heart rate is: {freq}")
        buffer = np.append(buffer, freq)    
        start_indice = start_indice + step
        end_indice = end_indice + step
    """   


def averaging(data:np.ndarray):
    
    window_start = 0
    window_end = 60
    step = 1
    len_data = len(data)
    buffer = np.array([])

    while window_end <= len_data:
        ave = np.average(data[window_start:window_end])
        window_start = window_start + step
        window_end = window_end + step
        # extra = np.full(10, ave)
        buffer = np.append(buffer, ave)
    return buffer


def emd_analysis(disp:np.ndarray):
    # use emd to decompose the input signal
    imf = emd.sift.sift(disp)
    t_disp = np.arange(len(disp)) * config.frame_repetition_time_s
    # take the first imf 
    heart_wave = imf[:,0]
    breath_wave = imf[:,2]
    # emd.plotting.plot_imfs(imf[:, :])
    
    #plt.plot(t_disp, imf_1)
    #plt.title("emd decomposition")
    #plt.grid()
    #plt.show()
    return [heart_wave, breath_wave]


def sig_average(disp:np.ndarray):


    start_indice = 0
    end_indice = 10
    step = 1
    buffer = np.array([])
    while end_indice<= len(disp):
        ave = np.average(disp[start_indice:end_indice])
        buffer = np.append(buffer, ave)
        start_indice = start_indice + step
        end_indice = end_indice + step
    
    return buffer


def iq_balance(disp:np.ndarray):

    Q = np.real(disp)
    I = np.imag(disp)

    plt.plot(Q, I, "*")
    plt.title("I-Q")
    plt.grid()
    plt.show()

        # 1) remove DC (ellipse center)
    I0 = I - I.mean()
    Q0 = Q - Q.mean()

    # 2) whiten (ellipse -> circle)
    X = np.vstack([I0, Q0])  # shape (2, N)
    C = np.cov(X)            # 2x2
    eigvals, eigvecs = np.linalg.eigh(C)
    W = eigvecs @ np.diag(1.0/np.sqrt(eigvals)) @ eigvecs.T
    Iw, Qw = (W @ X)

    # 3) fine rotation to enforce quadrature
    varI = np.var(Iw)
    varQ = np.var(Qw)
    covIQ = np.mean((Iw - Iw.mean()) * (Qw - Qw.mean()))
    theta = 0.5 * np.arctan2(2*covIQ, (varI - varQ))

    c, s = np.cos(-theta), np.sin(-theta)
    Ic = c*Iw - s*Qw
    Qc = s*Iw + c*Qw

    plt.plot(Qc, Ic, "*")
    plt.title("balanced I-Q")
    plt.grid()
    plt.show()

    return Ic, Qc


def hampel_filter(x, window_size=15, n_sigma=3.0, mode="reflect", cval=0.0, ignore_nan=False, replace="median"):
    """
    Hampel outlier filter (1D).

    Parameters
    ----------
    x : array_like
        Input 1D signal.
    window_size : int
        Sliding window length (must be odd >= 3). Typical: 9–21.
    n_sigma : float
        Outlier threshold in robust-sigma units (2.5–3.5 common).
    mode : {"reflect","edge","constant"}
        How to pad at the edges.
    cval : float
        Pad value if mode="constant".
    ignore_nan : bool
        If True, compute median/MAD ignoring NaNs in each window.
    replace : {"median","clip","none"}
        - "median": replace outliers with local median (classic Hampel).
        - "clip": clip to m ± n_sigma * sigma.
        - "none": just return the mask; y=x unchanged.

    Returns
    -------
    y : np.ndarray
        Output signal (same shape as x).
    mask : np.ndarray (bool)
        True where samples were flagged as outliers.
    med : np.ndarray
        Local median per sample.
    sigma : np.ndarray
        Local robust sigma (1.4826 * MAD) per sample.
    """
    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        raise ValueError("hampel_filter expects a 1D array")
    if window_size < 3 or window_size % 2 == 0:
        raise ValueError("window_size must be odd and >= 3")

    k = window_size // 2
    if mode not in {"reflect", "edge", "constant"}:
        raise ValueError("mode must be 'reflect', 'edge', or 'constant'.")

    # Pad
    if mode == "constant":
        xp = np.pad(x, k, mode="constant", constant_values=cval)
    else:
        xp = np.pad(x, k, mode=mode)

    # Sliding windows
    try:
        from numpy.lib.stride_tricks import sliding_window_view
        W = sliding_window_view(xp, window_shape=window_size)
    except Exception:
        # Fallback (slower)
        N = len(x)
        W = np.empty((N, window_size), dtype=float)
        for i in range(N):
            W[i] = xp[i:i+window_size]

    # Local median and MAD
    if ignore_nan:
        med = np.nanmedian(W, axis=1)
        mad = np.nanmedian(np.abs(W - med[:, None]), axis=1)
    else:
        med = np.median(W, axis=1)
        mad = np.median(np.abs(W - med[:, None]), axis=1)

    sigma = 1.4826 * mad + 1e-12  # Gaussian-consistent scale
    # Outlier mask
    mask = np.abs(x - med) > (n_sigma * sigma)

    # Replace or clip
    y = x.copy()
    if replace == "median":
        y[mask] = med[mask]
    elif replace == "clip":
        upper = med + n_sigma * sigma
        lower = med - n_sigma * sigma
        y = np.minimum(np.maximum(y, lower), upper)
    elif replace == "none":
        pass
    else:
        raise ValueError("replace must be 'median', 'clip', or 'none'.")

    # return y, mask, med, sigma
    return y


def seperate_analysis(mat:np.ndarray, case:str): 
    # mat: recorded data
    # pts: name of the file to save
    # processing raw data

    bandwidth = config.chirp.end_frequency_Hz - config.chirp.start_frequency_Hz
    fsta = config.chirp.start_frequency_Hz
    lamda = constants.c / fsta
    velo_bin_res = lamda / (2* config.chirp.num_samples / config.chirp.sample_rate_Hz)
    range_bin_res = constants.c * (config.chirp.num_samples / config.chirp.sample_rate_Hz) / (2 * bandwidth)
    # print(f"range_bin_res is: {range_bin_res}")
    disp_factor = lamda / (4*np.pi)
    num_antennas = 3
    # fft_freq = np.zeros(config.chirp.num_samples)

    hr_cut_off_freq = 3 # Hz
    br_cut_off_freq = 0.9 # Hz
    frames = np.shape(mat)[0]
    print(f"number of frames is: {frames}")

    buffer_1 = np.zeros((frames, config.chirp.num_samples), dtype=complex)
    buffer_2 = np.zeros((frames, config.chirp.num_samples), dtype=complex)
    buffer_3 = np.zeros((frames, config.chirp.num_samples), dtype=complex)

    doppler_range_ins = Doppler_fft(config.num_chirps, config.chirp.num_samples)

    print(f"loading data...")
    for i_Ant in np.arange(num_antennas):
        for i_frame in np.arange(frames):
            chirps_and_samples = mat[i_frame, i_Ant]
            # chirps_and_samples = load_frames(i_Ant, i_frame)
            # print(chirps_and_samples)
            # doppler_range_fft_result[3] is the range_fft(num_chirsp * num_samples)
            doppler_range_fft_result = doppler_range_ins.compute_doppler_range_fft(chirps_and_samples, 
                                                            range_bin=range_bin_res,
                                                            velo_bin=velo_bin_res)
            range_freq = doppler_range_fft_result[0]
            co_averaged = coherent_averaging(doppler_range_fft_result[3])
            
            if i_Ant == 0:
                buffer_1[i_frame,:] = co_averaged
                    
            if i_Ant == 1:
                buffer_2[i_frame,:] = co_averaged

            if i_Ant == 2:
                buffer_3[i_frame,:] = co_averaged

    # each time take 500 frames(so called processing window) from the buffer and after each loop sliding the
    # processing window for 1 sec(this corresponding to 50 frame)

    window_start = 0
    window_end = 600
    sliding_step = 20 # shift the processing window by 50s for every iteration

    breath_rates_1 = np.array([])
    heart_rates_1 = np.array([])

    breath_rates_2 = np.array([])
    heart_rates_2 = np.array([])

    breath_rates_3 = np.array([])
    heart_rates_3 = np.array([])

    print(f"processing data")
    while window_end <= frames:

        processing_window_1 = buffer_1[window_start:window_end,:]
        processing_window_2 = buffer_2[window_start:window_end,:]
        processing_window_3 = buffer_3[window_start:window_end,:]

        max_col_1 = max_peak_detection(processing_window_1, range_freq)
        max_col_2 = max_peak_detection(processing_window_2, range_freq)
        max_col_3 = max_peak_detection(processing_window_3, range_freq)


        disp_1 = phase_correlation(processing_window_1, max_col_1[1], disp_factor)
        disp_2 = phase_correlation(processing_window_2, max_col_2[1], disp_factor)
        disp_3 = phase_correlation(processing_window_3, max_col_3[1], disp_factor)

        plot_all.plot_displacement(disp_1, disp_2, disp_3, "unwrapped")

        
        
        disp_1 = remove_baseline(disp_1)
        disp_2 = remove_baseline(disp_2)
        disp_3 = remove_baseline(disp_3)

        plot_all.plot_displacement(disp_1, disp_2, disp_3, "unwrapped")
        
        # disp_1 = normalization(disp_1)
        # disp_2 = normalization(disp_2)
        # disp_3 = normalization(disp_3)

        diff_disp_1 = fisrt_order_differentiators(disp_1, config.frame_repetition_time_s)
        diff_disp_2 = fisrt_order_differentiators(disp_2, config.frame_repetition_time_s)
        diff_disp_3 = fisrt_order_differentiators(disp_3, config.frame_repetition_time_s) 
        
        
        # plot_all.plot_displacement(diff_disp_1, diff_disp_2, diff_disp_3, "differential signal")

        # remove pulse noise after differentiation
        diff_disp_1 = hampel_filter(diff_disp_1)
        diff_disp_2 = hampel_filter(diff_disp_2)
        diff_disp_3 = hampel_filter(diff_disp_3)


        """
        [emd_heart_wave_1, emd_breath_wave_1] = emd_analysis(disp_1)
        [emd_heart_wave_2, emd_breath_wave_2] = emd_analysis(disp_2)
        [emd_heart_wave_3, emd_breath_wave_3] = emd_analysis(disp_3)
        """

        # plot_differential(disp_1_diff, disp_2_diff, disp_3_diff)
        # plot_displacement(disp_1, disp_2, disp_3)

        # disp_1 = remove_baseline(disp_1)
        # disp_2 = remove_baseline(disp_2)
        # disp_3 = remove_baseline(disp_3)

        # plot_displacement(disp_1, disp_2, disp_3)

        # original_disp = disp_1 + disp_2 + disp_3
        # disp_diff = disp_1_diff + disp_2_diff + disp_3_diff
        # plt.plot(disp_diff)
        # plt.show()
        # disp = remove_pulse_noise(disp_diff)

        # after_remove_pulse_noise(disp, disp_diff)
        # diff_disp = phase_difference(disp)

        # plot_diff_original(diff_disp, disp)
        plot_filters()
        breath_wave_1 = kaiser_filtering_b(disp_1)
        heart_wave_1 = kaiser_filtering_h(diff_disp_1)
        
        breath_wave_2 = kaiser_filtering_b(disp_2)
        heart_wave_2 = kaiser_filtering_h(diff_disp_2)
        
        breath_wave_3 = kaiser_filtering_b(disp_3)
        heart_wave_3 = kaiser_filtering_h(diff_disp_3)



        plot_all.plot_breath_heart_disp(breath_disp=breath_wave_1, heart_disp=heart_wave_1)
        #heart_wave_1 = heart_sig_average(heart_wave_1)
        #heart_wave_2 = heart_sig_average(heart_wave_2)
        #heart_wave_3 = heart_sig_average(heart_wave_3)

        # emd_heart_wave_1 = normalization(emd_heart_wave_1)
        # emd_heart_wave_2 = normalization(emd_heart_wave_2)
        # emd_heart_wave_3 = normalization(emd_heart_wave_3)
        
        """
        heart_wave_1 = dc_remove(heart_wave_1)
        heart_wave_2 = dc_remove(heart_wave_2)
        heart_wave_3 = dc_remove(heart_wave_3)

        emd_heart_wave_1 = dc_remove(emd_heart_wave_1)
        emd_heart_wave_2 = dc_remove(emd_heart_wave_2)
        emd_heart_wave_3 = dc_remove(emd_heart_wave_3)

        similarity(breath_wave_1, emd_breath_wave_1)    
        similarity(breath_wave_2, emd_breath_wave_2)
        similarity(breath_wave_3, emd_breath_wave_3)

        similarity(heart_wave_1, emd_heart_wave_1)    
        similarity(heart_wave_2, emd_heart_wave_2)
        similarity(heart_wave_3, emd_heart_wave_3)
        """
        breath_wave_1 = dc_remove(breath_wave_1)
        breath_wave_2 = dc_remove(breath_wave_2)
        breath_wave_3 = dc_remove(breath_wave_3)

        breath_rate_1 = rubost_breath_rate_detection(breath_wave_1, br_cut_off_freq)
        breath_rate_2 = rubost_breath_rate_detection(breath_wave_2, br_cut_off_freq)
        breath_rate_3 = rubost_breath_rate_detection(breath_wave_3, br_cut_off_freq)

        """
        heart_wave_1 = ANF(heart_wave_1, breath_rate_1, 3)
        heart_wave_2 = ANF(heart_wave_2, breath_rate_2, 3)
        heart_wave_3 = ANF(heart_wave_3, breath_rate_3, 3)

        heart_wave_1 = ANF(heart_wave_1, breath_rate_1, 4)
        heart_wave_2 = ANF(heart_wave_2, breath_rate_2, 4)
        heart_wave_3 = ANF(heart_wave_3, breath_rate_3, 4)
        """

        # suppress the second and third harmonics of breathing signal in heartbeat signals
        heart_wave_1 = ANF(heart_wave_1, breath_rate_1, 2)
        heart_wave_1 = ANF(heart_wave_1, breath_rate_1, 3)
        
        heart_wave_2 = ANF(heart_wave_2, breath_rate_2, 2)
        heart_wave_2 = ANF(heart_wave_2, breath_rate_2, 3)
        
        heart_wave_3 = ANF(heart_wave_3, breath_rate_3, 2)
        heart_wave_3 = ANF(heart_wave_3, breath_rate_3, 3)
        
        heart_rates_1 = np.append(heart_rates_1 ,heartrate_estimation_fft(heart_wave_1, hr_cut_off_freq))
        heart_rates_2 = np.append(heart_rates_2 ,heartrate_estimation_fft(heart_wave_2, hr_cut_off_freq))
        heart_rates_3 = np.append(heart_rates_3 ,heartrate_estimation_fft(heart_wave_3, hr_cut_off_freq))
        # There are 1200 frames for each large processing window, for each processing window, we have (1200 - 200) / 20 = 50 heart rates
        # so it's needed to pad 50 breath rate for each processing window
        breath_rates_1 = np.append(breath_rates_1, breath_rate_1)
        breath_rates_2 = np.append(breath_rates_2, breath_rate_2)
        breath_rates_3 = np.append(breath_rates_3, breath_rate_3)

        window_start = window_start + sliding_step
        window_end = window_end + sliding_step

        # heart_rate = czt_analysis(heart_wave, heart_freq, 1/config.frame_repetition_time_s)
        # print(f"heartrate is: {heart_rate}")

        # plot_displacement(disp_1=disp_1, disp_2=disp_2, disp_3=disp_3)
    
    # breath_rates_1 = breath_rates_1 * 60
    # breath_rates_2 = breath_rates_2 * 60
    # breath_rates_3 = breath_rates_3 * 60

    # print(f"length of breath rates: {len(breath_rates_1)}, heart rates is: {len(heart_rates_1)}")


    # heart_rates_1 = averaging(heart_rates_1)
    # heart_rates_2 = averaging(heart_rates_2)
    # heart_rates_3 = averaging(heart_rates_3)

    heart_rates_1 = heart_rates_1 * 60
    heart_rates_2 = heart_rates_2 * 60
    heart_rates_3 = heart_rates_3 * 60

    breath_rates_1 = breath_rates_1 * 60
    breath_rates_2 = breath_rates_2 * 60
    breath_rates_3 = breath_rates_3 * 60

    # plot_all.plot_hrs_brs(heart_rates_1, breath_rates_1, case)
    # plot_all.plot_hrs_brs(heart_rates_2, breath_rates_2, case)
    # plot_all.plot_hrs_brs(heart_rates_3, breath_rates_3, case)

    heart_rates = (heart_rates_1 + heart_rates_2 + heart_rates_3) / 3
    breath_rates = (breath_rates_1 + breath_rates_2 + breath_rates_3) / 3

    # heart_rates = sig_average(heart_rates)
    # breath_rates = sig_average(breath_rates)


    plot_all.plot_hrs_brs(heart_rates, breath_rates, case)
    np.save("masterarbeit-radar/candidates_hrs_brs_combine_ctf/"+case+"_breath_rates", breath_rates)
    np.save("masterarbeit-radar/candidates_hrs_brs_combine_ctf/"+case+"_heart_rates", heart_rates)





if __name__ == '__main__':
    
    # delte_previous_data()
    # make_dir()
    # record_data() 
    
    # daphne = np.load('data/test_sensor_d20250704-143627/RadarIfxAvian_00/radar.npy')
    # wei = np.load('data/test_sensor_220250704-141359/RadarIfxAvian_00/radar.npy')
    # wei = np.load('wei/wei_data_2/RadarIfxAvian_00/radar.npy')
    # wei = np.load("record_on_0909/BGT60TR13C_record_20250908-231501/RadarIfxAvian_00/radar.npy")
    # wei = np.load('wei_220250704-141359/RadarIfxAvian_00/radar.npy')

    # [frames_wei, antenna, chirps, samples] = np.shape(daphne)
    # [frames_wei, antenna, chirps, samples] = np.shape(wei)

    name_folder_dict = {#"benjamin_before":"masterarbeit-radar/record_on_0909/Benjamin_before20250909-100656/RadarIfxAvian_00/radar.npy",
                        # "benjamin_after":"masterarbeit-radar/record_on_0909/Benjamin_after20250909-101734/RadarIfxAvian_00/radar.npy",
                        #"daphne_before":"masterarbeit-radar/record_on_0909/daphne_before20250909-102441/RadarIfxAvian_00/radar.npy",
                        #"daphne_after":"masterarbeit-radar/record_on_0909/daphne_after20250909-103516/RadarIfxAvian_00/radar.npy",
                        #"roman_before":"masterarbeit-radar/record_on_0909/roman_before20250909-152453/RadarIfxAvian_00/radar.npy",
                        #"roman_after":"masterarbeit-radar/record_on_0909/roman_after20250909-153726/RadarIfxAvian_00/radar.npy",
                        # "stephan_before":"masterarbeit-radar/record_on_0909/stephan_before20250909-150057/RadarIfxAvian_00/radar.npy",
                        "stephan_after":"masterarbeit-radar/record_on_0909/stephan_after20250909-151129/RadarIfxAvian_00/radar.npy",
                        #"hassan_before":"masterarbeit-radar/record_on_0909/hassan_before20250916-135424/RadarIfxAvian_00/radar.npy",
                        #"hassan_after":"masterarbeit-radar/record_on_0909/hassan_after20250916-140450/RadarIfxAvian_00/radar.npy",
                        "alex_before":"masterarbeit-radar/record_on_0909/alex_before20250916-143105/RadarIfxAvian_00/radar.npy"}
                        #"alex_after":"masterarbeit-radar/record_on_0909/alex_after20250916-144220/RadarIfxAvian_00/radar.npy",
                        #"nils_before":"masterarbeit-radar/record_on_0909/nils_before20250916-145102/RadarIfxAvian_00/radar.npy",
                        #"nils_after":"masterarbeit-radar/record_on_0909/nils_after20250916-150139/RadarIfxAvian_00/radar.npy",
                        #"awis_before":"masterarbeit-radar/record_on_0909/awis_before20250916-152604/RadarIfxAvian_00/radar.npy"}
                        #"awis_after":"masterarbeit-radar/record_on_0909/awis_after20250916-153822/RadarIfxAvian_00/radar.npy"}
                      

    for key in name_folder_dict:
        mat = np.load(name_folder_dict[key])
        seperate_analysis(mat, key)

        