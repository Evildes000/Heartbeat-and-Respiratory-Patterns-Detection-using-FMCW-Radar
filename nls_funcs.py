# this script contains NLS functions
import numpy as np
from scipy.optimize import minimize_scalar
from ifxradarsdk.fmcw.types import FmcwSimpleSequenceConfig, FmcwSequenceChirp


config = FmcwSimpleSequenceConfig(
    # frame_repetition_time_s=307.325e-3,  # Frame repetition time
    frame_repetition_time_s=0.05,  # Frame repetition time
    chirp_repetition_time_s=0.000150,  # Chirp repetition time

    num_chirps=64,  # chirps per frame
    
    tdm_mimo=False,  # set True to enable MIMO mode, which is only valid for sensors with 2 Tx antennas
    chirp=FmcwSequenceChirp(
        start_frequency_Hz=59e9,  # start RF frequency, where Tx is ON
        end_frequency_Hz=63e9,  # stop RF frequency, where Tx is OFF
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


def NLS_breath_first(disp:np.ndarray):
    """
    This function use NLS to eatimate breathing rate for the first time

    Parameters
    ----------
    disp : breathing signal

    Return
    ------
    estimated breathing rate
    """
    len_disp = len(disp)
    # fft_vals = np.fft.fft(disp)[0:len_disp]
    # freqs = np.fft.fftfreq(2*len_disp, config.frame_repetition_time_s)[0:len_disp]
    start_freq = 0.1
    end_freq = 0.5
    fs = 1 / config.frame_repetition_time_s
    # number of harmonic
    Lk = 5
    n = np.arange(len_disp)

    def cost_function(f):
        total = 0
        for l in range(1, Lk + 1):
            omega = 2 * np.pi * f * l * n/ fs
            basis = np.exp(-1j * omega)
            # inner_product = np.sum( np.abs(basis.conj()*disp)**2)
            inner_product = np.abs(np.dot(basis, disp)) ** 2 
            total += inner_product
        return -total

    result = minimize_scalar(
            cost_function,
            bounds=(start_freq, end_freq),
            method='bounded')

    return result.x


def NLS_breath(disp, pre_breath_rate):
    """
    This function use NLS to eatimate breathing rate for the first time

    Parameters
    ----------
    disp : breathing signal
    pre_breath_rate : the estimated breathing rate from the last time
    Return
    ------
    estimated breathing rate
    """
    
    
    len_disp = len(disp)
    # fft_vals = np.fft.fft(disp)[0:len_disp]
    # freqs = np.fft.fftfreq(2*len_disp, config.frame_repetition_time_s)[0:len_disp]
    # start_freq = pre_breath_rate - 0.033
    # end_freq = pre_breath_rate + 0.033

    start_freq = max(0.1,  pre_breath_rate - 0.033)
    end_freq = min(0.5,  pre_breath_rate + 0.033)

    n = np.arange(len_disp)
    fs = 1 / config.frame_repetition_time_s
    Lk = 5

    def cost_function(f):
        total = 0
        for l in range(1, Lk + 1):
            # 严格计算内积 z^H(lω) d
            omega = 2 * np.pi * f * l * n/ fs
            basis = np.exp(-1j * omega)
            # inner_product = np.sum( np.abs(basis.conj()*disp)**2)
            inner_product = np.abs(np.dot(basis, disp)) ** 2 
            total += inner_product
        return -total

    result = minimize_scalar(
            cost_function,
            bounds=(start_freq, end_freq),
            method='bounded')

    return result.x



def NLS_heart(disp:np.ndarray):
    """
    This function use NLS to eatimate heart rate

    Parameters
    ----------
    disp : breathing signal
    pre_breath_rate : the estimated breathing rate from the last time
    Return
    ------
    estimated breathing rate
    """

    # complex_disp = hilbert(disp)
    len_disp = len(disp)
    # fft_vals = np.fft.fft(disp)[0:len_disp]
    # freqs = np.fft.fftfreq(2*len_disp, config.frame_repetition_time_s)[0:len_disp]
    
    # 3*1 heart rates array
    heart_rates = np.zeros((3,1), dtype=float)
    print(f" shape is: {np.shape(heart_rates)}")
    fs = 1 / config.frame_repetition_time_s
    n = np.arange(len_disp)

    start_freq_1 = 1.0
    end_freq_1 = 1.5
    
    start_freq_2 = 1.5
    end_freq_2 = 3

    start_freq_3 = 2.25
    end_freq_3 = 4.5
    
    Lk = 3
    def cost_function(f):
        total = 0
        for l in range(1, Lk + 1):
            # 严格计算内积 z^H(lω) d
            omega = 2 * np.pi * f * n*l / fs
            basis = np.exp(-1j * omega)
            # inner_product = np.sum( np.abs(basis.conj()*disp)**2)
            inner_product = np.abs(np.dot(basis, disp)) ** 2 
            total += inner_product
        return -total

    result_1 = minimize_scalar(
            cost_function,
            bounds=(start_freq_1, end_freq_1),
            method='bounded')
    
    result_2 = minimize_scalar(
            cost_function,
            bounds=(start_freq_2, end_freq_2),
            method='bounded')
    
    result_3 = minimize_scalar(
            cost_function,
            bounds=(start_freq_3, end_freq_3),
            method='bounded')
    
    heart_rates[0] = result_1.x
    heart_rates[1] = result_2.x / 2
    heart_rates[2] = result_3.x / 3
    return heart_rates 