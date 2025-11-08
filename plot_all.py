# this script all functions that plot some figures
import numpy as np
import matplotlib.pyplot as plt
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

def plot_diff_original(diff_disp:np.ndarray, disp:np.ndarray):
    # diff_disp: differential signal
    # disp: original signal
    # plot the differential signal and original signal
    t = np.arange(len(disp)) * config.frame_repetition_time_s
    plt.plot(t, diff_disp, label = "diff signal")
    plt.plot(t, disp, label = "original signal")
    plt.xlabel("t/s")
    plt.ylabel("amplitude")
    plt.grid()
    plt.legend()
    plt.show()




def plot_mat(mat:np.ndarray, range_freq:np.ndarray, col:int):
    # mat: matrix to plot
    # range_freq: 
    # print(f"col is: {col}")
    plt.imshow(np.abs(mat[:,30:255]),
               extent=[range_freq[30], range_freq[255], 1499, 0],
               aspect='auto')
    # plt.imshow(np.abs(mat))
    # plt.xlim(20,255)
    # plt.ylim(499,0)
    plt.colorbar(label='Value')
    # plt.plot(col[0],col[1]+30, 'rx', markersize=12, label='Peak')
    plt.xlabel("range/m")
    plt.ylabel("frames")
    plt.show()



def plot_displacement(disp_1:np.ndarray, disp_2:np.ndarray, disp_3:np.ndarray, title_name:str):
    # disp_1, disp_2, disp_3: three displacement signals from three receive antennas repectively
    # plot displacement signals and spectrums of three receive antennas 
    len_disp = len(disp_1)
    t = np.arange(len_disp) * config.frame_repetition_time_s
    fft_size = 2*len_disp
    disp_1_fft = np.abs(np.fft.fft(disp_1, n=fft_size)[0:len_disp])/fft_size
    disp_2_fft = np.abs(np.fft.fft(disp_2, n=fft_size)[0:len_disp])/fft_size
    disp_3_fft = np.abs(np.fft.fft(disp_3, n=fft_size)[0:len_disp])/fft_size
    fft_freq = np.fft.fftfreq(fft_size, d=config.frame_repetition_time_s)[0:len_disp]

    plt.plot(t, 100*disp_3)
    plt.xlabel("t/s", fontsize=12)
    plt.ylabel("Amplitude/mm", fontsize=12)
    plt.ticklabel_format(style="sci", axis="y", scilimits=(0,0))
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid()
    plt.show()
    """
    fig, axs = plt.subplots(3,2)
    axs[0,0].plot(t, disp_1)
    axs[0,0].set_xlabel("t/s")
    axs[0,0].set_ylabel("amplitude")
    axs[0,0].set_title("disp_1")
    axs[0,0].grid()

    axs[1,0].plot(t, disp_2)
    axs[1,0].set_xlabel("t/s")
    axs[1,0].set_ylabel("amplitude")
    axs[1,0].set_title("disp_2")
    axs[1,0].grid()

    axs[2,0].plot(t, disp_3)
    axs[2,0].set_xlabel("t/s")
    axs[2,0].set_ylabel("amplitude")
    axs[2,0].set_title("disp_3")
    axs[2,0].grid()

    axs[0,1].plot(fft_freq, disp_1_fft)
    axs[0,1].set_xlabel("f/Hz")
    axs[0,1].set_ylabel("amplitude")
    axs[0,1].set_title("spectrum_1")
    axs[0,1].grid()

    axs[1,1].plot(fft_freq, disp_1_fft)
    axs[1,1].set_xlabel("f/Hz")
    axs[1,1].set_ylabel("amplitude")
    axs[1,1].set_title("spectrum_2")
    axs[1,1].grid()

    axs[2,1].plot(fft_freq, disp_1_fft)
    axs[2,1].set_xlabel("f/Hz")
    axs[2,1].set_ylabel("amplitude")
    axs[2,1].set_title("spectrum_3")
    axs[2,1].grid()
    """
    # fig.suptitle(title)




def plot_breath_heart_disp(breath_disp:np.ndarray, heart_disp: np.ndarray):
    # breath_disp, heart_disp: breath and heart signal respectively
    # this function plots breath and heart displacement and their spectrum
    
    """
    try:
        len(breath_disp) == len(heart_disp)
    except:
        print(f"length of two inputs are not match")
    """
    
    t_breath = np.arange(len(breath_disp)) * config.frame_repetition_time_s
    t_heart = np.arange(len(heart_disp)) * config.frame_repetition_time_s
    n_size = 2*len(breath_disp)
    breath_fft = np.fft.fft(breath_disp ,n=n_size)/n_size
    heart_fft = np.fft.fft(heart_disp ,n=n_size)/n_size
    
    # length = min(len(breath_disp), len(heart_disp))
    fftfreq = np.fft.fftfreq(n_size, d=config.frame_repetition_time_s)
    
    fig, axs = plt.subplots(2,1)
    
    axs[0].plot(t_breath, 1000*breath_disp, label = "breath_disp")
    axs[0].plot(t_heart, 1000*heart_disp, label = "heart_disp")
    axs[0].set_xlabel("t/s", fontsize=24)
    axs[0].set_ylabel("Amplitude/mm", fontsize=24)
    axs[0].tick_params(labelsize=24)
    axs[0].grid()
    axs[0].legend(fontsize=24, loc ="upper right")

    axs[1].plot(fftfreq[0:int(n_size/10)], np.abs(breath_fft[0:int(n_size/10)]), label = "breath_fft")
    axs[1].plot(fftfreq[0:int(n_size/10)], np.abs(heart_fft[0:int(n_size/10)]), label = "heart_fft")
    axs[1].set_xlabel("Frequency/Hz", fontsize=24)
    axs[1].set_ylabel("Amplitude/-", fontsize=24)
    axs[1].ticklabel_format(style="sci", axis="y", scilimits=(0,0))
    axs[1].tick_params(labelsize=24)
    axs[1].grid()
    axs[1].legend(fontsize=24)

    fig.subplots_adjust(hspace=0.5)

    plt.show()
    


def plot_hrs_brs(hrs:np.ndarray, brs:np.ndarray, case:str):
    # hrs, brs: array of heart rate and breath rate, respectively from each processing window 
    # plot the estimated heart rates over a period of time

    t = np.arange(len(hrs)) * 1
    # plt.plot(hrs, marker='o', linestyle='-', label="hrs")
    # plt.plot(brs, marker='o', linestyle='-', label="brs")
    plt.plot(hrs, label="hrs")
    plt.plot(brs, label="brs")
    plt.xlabel("t/s", fontsize=16)
    plt.ylabel("bpm", fontsize=16)
    plt.tick_params(labelsize=16)
    plt.legend()
    plt.grid()
    plt.title(case)
    plt.show()


def after_remove_pulse_noise(disp:np.ndarray, org_disp):
    # disp: after noise removed
    # org_disp: before noise remove 
    t = np.arange(len(disp)) * config.frame_repetition_time_s
    plt.plot(t, disp, label = "after noise remove")
    plt.plot(t, org_disp, label = "before noise remove")
    plt.xlabel("t/s", fontsize=16)
    plt.ylabel("amp", fontsize=16)
    # plt.title("after removing pulse noise")
    plt.legend()
    plt.tick_params(labelsize=16)
    plt.grid()
    plt.show()



def plot_unwrapped(disp_1:np.ndarray, disp_2:np.ndarray, disp_3:np.ndarray):
    # plot displacement data acquicred from three antennas and their spectrum

    length = len(disp_1)
    print(f"length of the signal is: {length}")
    t = np.arange(length) * config.frame_repetition_time_s
    # disp = (disp_1 + disp_2 + disp_3) / 3
    fft_size = 2 * length
    fft_disp_1 = np.fft.fft(disp_1, n=fft_size)/fft_size
    fft_disp_2 = np.fft.fft(disp_2, n=fft_size)/fft_size
    fft_disp_3 = np.fft.fft(disp_3, n=fft_size)/fft_size
    # fft_disp = np.fft.fft(disp, n=fft_size)/fft_size

    fft_freq = np.fft.fftfreq(fft_size, d=config.frame_repetition_time_s)
    # zero_freq_index = np.where(fft_freq == 0)[0][0]

    fig, axs = plt.subplots(3, 2)
    axs[0,0].plot(t,disp_1, label="unwrapped signal 1")
    # axs[0,0].set_title("disp_1")
    # axs[0,0].set_xlabel("time/s")
    axs[0,0].set_ylabel("amplitude")
    axs[0,0].tick_params(labelsize = 12)
    axs[0,0].legend()
    axs[0,0].grid()

    axs[1,0].plot(t, disp_2, label="unwrapped signal 2")
    # axs[1,0].set_title("disp_2")
    # axs[1,0].set_xlabel("time/s")
    axs[1,0].set_ylabel("amplitude")
    axs[1,0].tick_params(labelsize = 12)
    axs[1,0].legend()
    axs[1,0].grid()

    axs[2,0].plot(t, disp_3, label="unwrapped signal 3")
    # axs[2,0].set_title("disp_3")
    axs[2,0].set_xlabel("time/s")
    axs[2,0].set_ylabel("amplitude")
    axs[2,0].tick_params(labelsize = 12)
    axs[2,0].legend()
    axs[2,0].grid()


    divide_factor = 32
    axs[0,1].plot(fft_freq[0:int(fft_size/divide_factor)], np.abs(fft_disp_1[0:int(fft_size/divide_factor)]), label="spectrum for unwrapped signal 1")
    # axs[0,1].set_title("spectrum for disp_1")
    # axs[0,1].set_xlabel("freq/Hz")
    axs[0,1].set_ylabel("amplitude")
    axs[0,1].tick_params(labelsize = 12)
    axs[0,1].legend()
    axs[0,1].grid()

    axs[1,1].plot(fft_freq[0:int(fft_size/divide_factor)], np.abs(fft_disp_2[0:int(fft_size/divide_factor)]), label="spectrum for unwrapped signal 2")
    #axs[1,1].set_title("spectrum for disp_2")
    # axs[1,1].set_xlabel("freq/Hz")
    axs[1,1].set_ylabel("amplitude")
    axs[1,1].tick_params(labelsize = 12)
    axs[1,1].legend()
    axs[1,1].grid()

    axs[2,1].plot(fft_freq[0:int(fft_size/divide_factor)], np.abs(fft_disp_3[0:int(fft_size/divide_factor)]), label="spectrum for unwrapped signal 3")
    #axs[2,1].set_title("spectrum for disp_3")
    axs[2,1].set_xlabel("freq/Hz")
    axs[2,1].tick_params(labelsize = 12)
    axs[2,1].set_ylabel("amplitude")
    axs[2,1].legend()
    axs[2,1].grid()



    fig.text(0.51, 0.90, 'antenna 1', ha='center', va='center', fontsize=12)
    fig.text(0.51, 0.63, 'antenna 2', ha='center', va='center', fontsize=12)
    fig.text(0.51, 0.35, 'antenna 3', ha='center', va='center', fontsize=12)

    """
    axs[3,1].plot(fft_freq[0:int(fft_size/2)], np.abs(fft_disp[0:int(fft_size/2)]), label = "spectrum for disp")
    #axs[3,1].set_title("spectrum for disp")
    axs[3,1].set_xlabel("freq/Hz")
    axs[3,1].set_ylabel("amplitude")
    axs[3,1].legend()
    axs[3,1].grid()
    """
    # plt.tick_params(labelsize = 12)
    plt.show()



def plot_hearts(disp_1:np.ndarray, disp_2:np.ndarray, disp_3:np.ndarray):
    # plot breath signals and thier spectrums

    length = len(disp_1)
    print(f"length of the signal is: {length}")
    t = np.arange(length) * config.frame_repetition_time_s
    # disp = (disp_1 + disp_2 + disp_3) / 3
    fft_size = 2 * length
    fft_disp_1 = np.fft.fft(disp_1, n=fft_size)/fft_size
    fft_disp_2 = np.fft.fft(disp_2, n=fft_size)/fft_size
    fft_disp_3 = np.fft.fft(disp_3, n=fft_size)/fft_size
    # fft_disp = np.fft.fft(disp, n=fft_size)/fft_size

    fft_freq = np.fft.fftfreq(fft_size, d=config.frame_repetition_time_s)
    # zero_freq_index = np.where(fft_freq == 0)[0][0]

    fig, axs = plt.subplots(3, 2)
    axs[0,0].plot(t,disp_1, label="heart signal 1")
    # axs[0,0].set_title("disp_1")
    # axs[0,0].set_xlabel("time/s")
    axs[0,0].set_ylabel("amplitude")
    axs[0,0].tick_params(labelsize = 12)
    axs[0,0].legend()
    axs[0,0].grid()

    axs[1,0].plot(t, disp_2, label="heart signal 2")
    # axs[1,0].set_title("disp_2")
    # axs[1,0].set_xlabel("time/s")
    axs[1,0].set_ylabel("amplitude")
    axs[1,0].tick_params(labelsize = 12)
    axs[1,0].legend()
    axs[1,0].grid()

    axs[2,0].plot(t, disp_3, label="heart signal 3")
    # axs[2,0].set_title("disp_3")
    axs[2,0].set_xlabel("time/s")
    axs[2,0].set_ylabel("amplitude")
    axs[2,0].tick_params(labelsize = 12)
    axs[2,0].legend()
    axs[2,0].grid()


    divide_factor = 8
    axs[0,1].plot(fft_freq[0:int(fft_size/divide_factor)], np.abs(fft_disp_1[0:int(fft_size/divide_factor)]), label="spectrum for heart signal 1")
    # axs[0,1].set_title("spectrum for disp_1")
    # axs[0,1].set_xlabel("freq/Hz")
    axs[0,1].set_ylabel("amplitude")
    axs[0,1].tick_params(labelsize = 12)
    axs[0,1].legend()
    axs[0,1].grid()

    axs[1,1].plot(fft_freq[0:int(fft_size/divide_factor)], np.abs(fft_disp_2[0:int(fft_size/divide_factor)]), label="spectrum for heart signal 2")
    #axs[1,1].set_title("spectrum for disp_2")
    # axs[1,1].set_xlabel("freq/Hz")
    axs[1,1].set_ylabel("amplitude")
    axs[1,1].tick_params(labelsize = 12)
    axs[1,1].legend()
    axs[1,1].grid()

    axs[2,1].plot(fft_freq[0:int(fft_size/divide_factor)], np.abs(fft_disp_3[0:int(fft_size/divide_factor)]), label="spectrum for heart signal 3")
    #axs[2,1].set_title("spectrum for disp_3")
    axs[2,1].set_xlabel("freq/Hz")
    axs[2,1].tick_params(labelsize = 12)
    axs[2,1].set_ylabel("amplitude")
    axs[2,1].legend()
    axs[2,1].grid()



    fig.text(0.51, 0.90, 'antenna 1', ha='center', va='center', fontsize=12)
    fig.text(0.51, 0.63, 'antenna 2', ha='center', va='center', fontsize=12)
    fig.text(0.51, 0.35, 'antenna 3', ha='center', va='center', fontsize=12)

    """
    axs[3,1].plot(fft_freq[0:int(fft_size/2)], np.abs(fft_disp[0:int(fft_size/2)]), label = "spectrum for disp")
    #axs[3,1].set_title("spectrum for disp")
    axs[3,1].set_xlabel("freq/Hz")
    axs[3,1].set_ylabel("amplitude")
    axs[3,1].legend()
    axs[3,1].grid()
    """
    # plt.tick_params(labelsize = 12)
    plt.show()







def plot_breaths(disp_1:np.ndarray, disp_2:np.ndarray, disp_3:np.ndarray):
    # plot breath signals and thier spectrums

    length = len(disp_1)
    print(f"length of the signal is: {length}")
    t = np.arange(length) * config.frame_repetition_time_s
    # disp = (disp_1 + disp_2 + disp_3) / 3
    fft_size = 2 * length
    fft_disp_1 = np.fft.fft(disp_1, n=fft_size)/fft_size
    fft_disp_2 = np.fft.fft(disp_2, n=fft_size)/fft_size
    fft_disp_3 = np.fft.fft(disp_3, n=fft_size)/fft_size
    # fft_disp = np.fft.fft(disp, n=fft_size)/fft_size

    fft_freq = np.fft.fftfreq(fft_size, d=config.frame_repetition_time_s)
    # zero_freq_index = np.where(fft_freq == 0)[0][0]

    fig, axs = plt.subplots(3, 2)
    axs[0,0].plot(t,disp_1, label="breath signal 1")
    # axs[0,0].set_title("disp_1")
    # axs[0,0].set_xlabel("time/s")
    axs[0,0].set_ylabel("amplitude")
    axs[0,0].tick_params(labelsize = 12)
    axs[0,0].legend()
    axs[0,0].grid()

    axs[1,0].plot(t, disp_2, label="breath signal 2")
    # axs[1,0].set_title("disp_2")
    # axs[1,0].set_xlabel("time/s")
    axs[1,0].set_ylabel("amplitude")
    axs[1,0].tick_params(labelsize = 12)
    axs[1,0].legend()
    axs[1,0].grid()

    axs[2,0].plot(t, disp_3, label="breath signal 3")
    # axs[2,0].set_title("disp_3")
    axs[2,0].set_xlabel("time/s")
    axs[2,0].set_ylabel("amplitude")
    axs[2,0].tick_params(labelsize = 12)
    axs[2,0].legend()
    axs[2,0].grid()


    divide_factor = 16
    axs[0,1].plot(fft_freq[0:int(fft_size/divide_factor)], np.abs(fft_disp_1[0:int(fft_size/divide_factor)]), label="spectrum for breath signal 1")
    # axs[0,1].set_title("spectrum for disp_1")
    # axs[0,1].set_xlabel("freq/Hz")
    axs[0,1].set_ylabel("amplitude")
    axs[0,1].tick_params(labelsize = 12)
    axs[0,1].legend()
    axs[0,1].grid()

    axs[1,1].plot(fft_freq[0:int(fft_size/divide_factor)], np.abs(fft_disp_2[0:int(fft_size/divide_factor)]), label="spectrum for breath signal 2")
    #axs[1,1].set_title("spectrum for disp_2")
    # axs[1,1].set_xlabel("freq/Hz")
    axs[1,1].set_ylabel("amplitude")
    axs[1,1].tick_params(labelsize = 12)
    axs[1,1].legend()
    axs[1,1].grid()

    axs[2,1].plot(fft_freq[0:int(fft_size/divide_factor)], np.abs(fft_disp_3[0:int(fft_size/divide_factor)]), label="spectrum for breath signal 3")
    #axs[2,1].set_title("spectrum for disp_3")
    axs[2,1].set_xlabel("freq/Hz")
    axs[2,1].tick_params(labelsize = 12)
    axs[2,1].set_ylabel("amplitude")
    axs[2,1].legend()
    axs[2,1].grid()



    fig.text(0.51, 0.90, 'antenna 1', ha='center', va='center', fontsize=12)
    fig.text(0.51, 0.63, 'antenna 2', ha='center', va='center', fontsize=12)
    fig.text(0.51, 0.35, 'antenna 3', ha='center', va='center', fontsize=12)

    """
    axs[3,1].plot(fft_freq[0:int(fft_size/2)], np.abs(fft_disp[0:int(fft_size/2)]), label = "spectrum for disp")
    #axs[3,1].set_title("spectrum for disp")
    axs[3,1].set_xlabel("freq/Hz")
    axs[3,1].set_ylabel("amplitude")
    axs[3,1].legend()
    axs[3,1].grid()
    """
    # plt.tick_params(labelsize = 12)
    plt.show()
    



def plot_differential(disp_1:np.ndarray, disp_2:np.ndarray, disp_3:np.ndarray):
    # plot displacement data acquicred from three antennas and their spectrum

    length = len(disp_1)
    # print(f"length of the element is: {length}")
    t = np.arange(length) * config.frame_repetition_time_s
    # disp = (disp_1 + disp_2 + disp_3) / 3
    fft_size = 2 * len(t)
    fft_disp_1 = np.fft.fft(disp_1, n=fft_size)/fft_size
    fft_disp_2 = np.fft.fft(disp_2, n=fft_size)/fft_size
    fft_disp_3 = np.fft.fft(disp_3, n=fft_size)/fft_size
    # fft_disp = np.fft.fft(disp, n=fft_size)/fft_size

    fft_freq = np.fft.fftfreq(fft_size, d=config.frame_repetition_time_s)
    zero_freq_index = np.where(fft_freq == 0)[0][0]

    fig, axs = plt.subplots(3, 2)
    axs[0,0].plot(t,disp_1, label="differential signal")
    # axs[0,0].set_title("disp_1")
    # axs[0,0].set_xlabel("time/s")
    axs[0,0].set_ylabel("amplitude")
    axs[0,0].tick_params(labelsize = 12)
    axs[0,0].legend()
    axs[0,0].grid()

    axs[1,0].plot(t, disp_2, label="differential signal")
    # axs[1,0].set_title("disp_2")
    # axs[1,0].set_xlabel("time/s")
    axs[1,0].set_ylabel("amplitude")
    axs[1,0].tick_params(labelsize = 12)
    axs[1,0].legend()
    axs[1,0].grid()

    axs[2,0].plot(t, disp_3, label="differential signal")
    # axs[2,0].set_title("disp_3")
    # axs[2,0].set_xlabel("time/s")
    axs[2,0].set_ylabel("amplitude")
    axs[2,0].tick_params(labelsize = 12)
    axs[2,0].legend()
    axs[2,0].grid()

    divide_factor = 16
    axs[0,1].plot(fft_freq[0:int(fft_size/divide_factor)], np.abs(fft_disp_1[0:int(fft_size/divide_factor)]), label="spectrum for differential signal")
    # axs[0,1].set_title("spectrum for disp_1")
    # axs[0,1].set_xlabel("freq/Hz")
    axs[0,1].set_ylabel("amplitude")
    axs[0,1].tick_params(labelsize = 12)
    axs[0,1].legend()
    axs[0,1].grid()

    axs[1,1].plot(fft_freq[0:int(fft_size/divide_factor)], np.abs(fft_disp_2[0:int(fft_size/divide_factor)]), label="spectrum for differential signal")
    #axs[1,1].set_title("spectrum for disp_2")
    # axs[1,1].set_xlabel("freq/Hz")
    axs[1,1].set_ylabel("amplitude")
    axs[1,1].tick_params(labelsize = 12)
    axs[1,1].legend()
    axs[1,1].grid()

    axs[2,1].plot(fft_freq[0:int(fft_size/divide_factor)], np.abs(fft_disp_3[0:int(fft_size/divide_factor)]), label="spectrum for differential signal")
    axs[2,1].set_title("")
    #axs[2,1].set_xlabel("freq/Hz")
    axs[2,1].set_ylabel("amplitude")
    axs[2,1].tick_params(labelsize = 12)
    axs[2,1].legend()
    axs[2,1].grid()


    fig.text(0.51, 0.90, 'antenna 1', ha='center', va='center', fontsize=12)
    fig.text(0.51, 0.63, 'antenna 2', ha='center', va='center', fontsize=12)
    fig.text(0.51, 0.35, 'antenna 3', ha='center', va='center', fontsize=12)

    """
    axs[3,1].plot(fft_freq[0:int(fft_size/2)], np.abs(fft_disp[0:int(fft_size/2)]), label = "spectrum for disp")
    #axs[3,1].set_title("spectrum for disp")
    axs[3,1].set_xlabel("freq/Hz")
    axs[3,1].set_ylabel("amplitude")
    axs[3,1].legend()
    axs[3,1].grid()
    """
    plt.show()


def compare_plots(disp_diff:np.ndarray, filtered_disp:np.ndarray):
    # compare signal spectrum before and after ANF

    n_size = 2*len(disp_diff)
    t = np.arange(len(disp_diff)) * config.frame_repetition_time_s
    # print(f"length of the element is: {length}")
    # t = np.arange(len(disp)) * config.frame_repetition_time_s
    disp_fft = (np.abs(np.fft.fft(disp_diff, n=n_size))[0:int(n_size/2)])/n_size
    filtered_disp_fft = (np.abs(np.fft.fft(filtered_disp, n=n_size))[0:int(n_size/2)])/n_size
    fft_freq = np.fft.fftfreq(n_size,d=config.frame_repetition_time_s)[0:int(n_size/2)]

    fig,axs = plt.subplots(2,1)
    axs[0].plot(t, disp_diff, label = "heart_disp")
    axs[0].plot(t, filtered_disp, label = "ANF_filtered_disp")
    axs[0].set_xlabel("t/s")
    axs[0].set_ylabel("amp")
    axs[0].grid()
    axs[0].legend()
    axs[0].set_title("time domin")

    axs[1].plot(fft_freq, disp_fft, label = "heart_disp")
    axs[1].plot(fft_freq, filtered_disp_fft, label = "ANF_filtered_disp")
    axs[1].set_xlabel("freq/Hz")
    axs[1].set_ylabel("amp")
    axs[1].grid()
    axs[1].legend()
    axs[1].set_title("filtered disp")

    plt.show()


def plot_range(range_amp:np.ndarray, range_fft:np.ndarray, skip:int):
    """
    range_amp: amplitude of ranges
    range_fft: ranges tranformed from fft
    skip:      number of fft bins need to be skipped
    index:     maxmimal value's index of skipped range_amp 
    """
    index = np.argmax(range_amp[skip:])
    plt.plot(range_fft[skip:], range_amp[skip:])
    plt.plot(range_fft[skip:][index],  range_amp[skip:][index], "*")
    plt.xlabel("Range/m")
    plt.ylabel("Amplitude/-")
    plt.grid()
    plt.show()

def plot_frame_matrix(mat:np.ndarray, range_fft:np.ndarray, skip:int):
    abs_mat = np.abs(mat[:,skip:])
    num_rows, _ = np.shape(abs_mat)
    range_fft = range_fft[skip:]
    plt.imshow(abs_mat, origin="upper", aspect="auto", extent=(range_fft[0], range_fft[-1], num_rows-1,0))
    plt.xlabel("Range/m")
    plt.ylabel("Frame/-")
    cbar = plt.colorbar()
    cbar.set_label("Amplitude/-")
    # plt.title("2D fft")
    plt.show()



def plot_range_doppler(mat:np.ndarray, range_fft:np.ndarray, velocity_fft:np.ndarray):
    """
    Inputs:
        mat: 2D fft matrix(with only positive)
        range_fft: range(m) along fast time
        velocity_fft: velocity along slow time 
    """

    abs_mat = np.abs(mat)
    plt.imshow(abs_mat, origin="lower", aspect="auto", extent=[range_fft[0], range_fft[-1], velocity_fft[0], velocity_fft[-1]])
    cbar = plt.colorbar()
    cbar.set_label("mag")
    # plt.title("2D fft")
    plt.show()




def plot_ref_ant(algo_type, t,
                 breath_rates_after, ref_breath_rates_after, breath_rates_before, ref_breath_rates_before,
                 heart_rates_after, ref_heart_rates_after, heart_rates_before, ref_heart_rates_before):
    """
    Plot reference and estimatd breathing and heart rate
    input:
        algo_type: type of algorithm that has been chosed
    """

    plt.plot(t, breath_rates_after, label = algo_type + "_br_after")
    plt.plot(t, ref_breath_rates_after, label = "ref_" + "_br_after")
    plt.plot(t, heart_rates_after, label = algo_type + "_hr_after")
    plt.plot(t, ref_heart_rates_after, label = "ref_" + "_hr_after")
    plt.xlabel("t/s", fontsize=28)
    plt.ylabel("bpm", fontsize=28)
    plt.xticks(fontsize=28)
    plt.yticks(fontsize=28)
    plt.grid()
    plt.legend(fontsize=20, loc='center right' ,bbox_to_anchor=(1.0, 0.3), borderaxespad=0.)
    #plt.legend(fontsize=20, loc='upper right')
    plt.show()


    ######################### plot rates before ####################################
    plt.plot(t, breath_rates_before, label = algo_type + "_br_before")
    plt.plot(t, ref_breath_rates_before, label = "ref_" + "_br_before")
    plt.plot(t, heart_rates_before, label = algo_type  + "_hr_before")
    plt.plot(t, ref_heart_rates_before, label = "ref_"  + "_hr_before")
    plt.xlabel("t/s", fontsize=28)
    plt.ylabel("bpm", fontsize=28)
    plt.xticks(fontsize=28)
    plt.yticks(fontsize=28)
    plt.grid()
    plt.legend(fontsize=20, loc='center right' ,bbox_to_anchor=(1.0, 0.3), borderaxespad=0.)
    #plt.legend(fontsize=20, loc='upper right')
    plt.show()


