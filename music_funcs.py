import numpy as np
from numpy.linalg import eigh
from scipy.linalg import toeplitz
import matplotlib.pyplot as plt


def decompose(r, m, fs):
    """
    Parameters
    ----------
    r: covariance matrix
    m: number of lags
    fs: sampling rate
    
    Returns
    -------
    search_region: frequecny sweep region
    buffer:         music peaks at frequencies
    """

    evals, evcts  = eigh(r)
    # eigenvalues are in ascending order
    # print(f"eigenvalues are: {evals}")

    # np.argsort order the indices of eigenvalues in terms of eigenvalues in ascending order
    # [::-1] reverse the indices   
    idx = np.argsort(evals)[::-1]
    # order eigenvalues also in descending order
    evals = evals[idx]
    # np.maximum compare two input arrays element-wise and pick the maximal one
    # in this way the tiny negative values can be set to 0
    evals = np.maximum(evals, 0)

    # take eigenvectors corresponding to eigenvalues
    evcts = evcts[:, idx]
    # print(f"eigenvectors are: {evcts}")

    # p is number of signal tones
    p = 1
    # take the first p eigenvectors as signal subspace
    E_s = evcts[:,:p]
    # take other eigenvectors as noise subspace
    E_n = evcts[:,p:]

    # generate frequency grid
    f_start = 0.05
    f_end = 3
    f_step = 0.05
    search_region = np.arange(f_start, f_end, f_step)
    buffer = np.zeros(len(search_region))

    for i in np.arange(len(search_region)):
        sum = 0
        samples = np.arange(m)
        a = np.exp(-1j * 2*np.pi*search_region[i]*samples/fs)
        for j in np.arange(m-p):
            sum = sum + np.abs(a @ E_n[:,j])**2/evals[j]

        buffer[i] = 1/sum
    
    return [search_region, buffer]


def peak_selection(freq:np.ndarray,  music:np.ndarray):
    """
    Select the peak in the music
    Parameters
    ----------
    music : values of each frequency after music algorithm

    Return
    ------
    the frequency corresponding to the peak
    """


    """
    plt.plot(freq, music)
    plt.xlabel("Frequency/Hz",fontsize=12)
    plt.ylabel("Amplitude/-",fontsize=12)
    plt.title("music spectrum",fontsize=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid()
    plt.show()
    """
    max_index = np.argmax(np.abs(music))
    return freq[max_index]




def autocorr_mat(arr:np.ndarray, m:int):
    """
    This function builds autocorrelation matrix

    Parameters
    ----------
    arr : 1D vector
    m : number of lags

    Return
    ------
    Rxx : autocorrelation matrix
    """

    arr_size = np.size(arr)
    rows = arr_size + m - 1
    cols = m
    buffer = np.zeros((rows, cols))
    for i in np.arange(m):
        buffer[i:arr_size+i,i] = arr
    # print(f"buffer is: {buffer}")

    Rxx = np.transpose(buffer) @ buffer
    
    return Rxx



if __name__ == "__main__":
    rng = np.random.default_rng(42)
    noise_std=0.3
    f1=1.2 
    A1=1.5
    f2=2.7
    A2=1.5,
    N = 600
    t = np.arange(N) / 20

    s1 = A1 * np.sin(2*np.pi*f1*t)
    s2 = A2 * np.sin(2*np.pi*f2*t)
    n  = rng.normal(0.0, noise_std, size=N)

    arr = s1 + s2 + n
    # arr = np.array([1,2,3,4,5,6,7,8,9,10])


    """    
    cov = corrmtx(arr, 1000)
    # new_arr, k = hankel_embed(arr, 300)
    print(f"new arr is: {cov}")

    r =(cov @ np.transpose(cov))
    print(f"covariance matrix is: {r}")

    [search_region, f_music] = decompose(r=r, n=1000, fs=20)
    plt.plot(search_region, np.abs(f_music))
    plt.grid()
    plt.show()
    """
    # arr = np.array([1,2,3,4,5])
    # N = arr.size
    # corr = np.correlate(arr, arr, mode="full")[N-1:]
    
    Rxx = autocorr_mat(arr=arr, m=300) # 3 lags
    print(f"Rxx is: {Rxx}")
    [freq, music] = decompose(Rxx, m=300, fs=20)

    plt.plot(freq, np.abs(music))
    plt.show()