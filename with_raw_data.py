import numpy as np
import matplotlib.pyplot as plt
import adaptfilt as adf
# from scipy.signal import firwin, freqz, filtfilt
import pandas as pd
from pathlib import Path
import os
import plot_all


# because in the first 7, 8 s the reference data are 0, so we only consider the data after 10 s.   

def pick_samples(data:np.ndarray):
    len_data = len(data)
    # print(f"length of data is: {len_data}")
    num_loops = len_data // 2048 
    # print(f"number of loops is: {num_loops}")
    i = 0
    picked_samples = np.array([])
    while i <= num_loops:
        # extra = np.full(10, data[i*2047])
        # picked_samples = np.concatenate((picked_samples, extra))
        picked_samples = np.append(picked_samples, data[i*2047])
        i = i + 1
    
    # picked_samples = np.array(picked_samples)
    # print(f"picked_samples are: {picked_samples}")
    num_zero = 0
    for i in picked_samples:
        if i == 0:
            num_zero = num_zero + 1
        else:
            pass
    
    print(f"number of zeros are: {num_zero}")
    return picked_samples


def rmse(reference:np.ndarray, estimated:np.ndarray):
    """
    calculate rmse of reference and estimated data

    Parameters
    ----------
    reference : reference data
    estimated : estimated data

    Return 
    ------
    rmse : rmse
    """
    num = len(reference)
    rmse = np.sqrt(np.sum(np.abs(reference - estimated) ** 2) / num)
    return  rmse


def averaging(data:np.ndarray):
    """
    Take all samples in 30 sec and calculate the average value
    
    Parameters
    ----------
    data : the reference data

    Return
    ------
    buffer : the average value
    """
    window_start = 0
    window_end = 200
    step = 10
    len_data = len(data)
    buffer = np.array([])

    while window_end <= len_data:
        ave = np.average(data[window_start:window_end])
        window_start = window_start + step
        window_end = window_end + step
        extra = np.full(10, ave)
        buffer = np.concatenate((buffer, extra))
    return buffer

def remove_zeros(data:np.ndarray):
    """
    remove zeros that located at the beginning of the input array
    
    Parameters
    ----------
    data : the reference data

    Return
    ------
    data : the reference data without 0 values at the beginning
    """
    for index, value in np.ndenumerate(data):
        if value <= 1:
            data = np.delete(data, index)
    return data


def compute_average(data:np.ndarray):
    """
    Take all samples in 30 sec and calculate the average value
    
    Parameters
    ----------
    data : the reference data

    Return
    ------
    buffer : the average value
    """

    len_data = len(data)
    start = 0
    end = 30
    step = 1 
    buffer = np.array([])
    while end<= len_data:
        ave = np.average(data[start:end])
        buffer = np.append(buffer, ave)
        start = start + step
        end = end + step

    return buffer


# Skip the metadata lines and load only the data section
def load_data(path_to_data:str):
    """
    Load the files in the path_to_data

    Parameters
    ----------
    path_to_data : the path

    Return
    ------
    breath_rate : all breathing rates in the file
    heart_rate : all heart rates in the file
    """
    file_path = path_to_data
    usecols = ["Zeit", "heart rate", "Respiration rate"]

    df = pd.read_csv(file_path, 
                 sep=';', 
                 skiprows=5, 
                 usecols=usecols,
                 engine='python')

    # print(df)
    # Step 2: Replace commas with dots and convert to float
    # df["Zeit"] = df["Zeit"].str.replace(",", ".", regex=False).astype(float)
    df["heart rate"] = df["heart rate"].str.replace(",", ".", regex=False).astype(float)
    df["Respiration rate"] = df["Respiration rate"].str.replace(",", ".", regex=False).astype(float)

    # t = np.array(df["Zeit"])
    # t = pick_samples(t)
    # print(f"t is: {t}")
    breath_rate = np.array(df["Respiration rate"])
    breath_rate = pick_samples(breath_rate)
    
    heart_rate = np.array(df["heart rate"])
    heart_rate = pick_samples(heart_rate)

    # print(f"frequency is: {frequenz}")
    # frequenz = [float(x) for x in frequenz]
    
    return [breath_rate, heart_rate]


def load_ref_data():
    """
    load the reference breath and heart recorded by medical devices
    """
    # candidate_names = ["stephan", "alex", "awis", "benjamin", "daphne", "hassan", "nile", "roman"]
    data_dict_ref = {"stephan_after":[], "stephan_before":[], 
                "alex_after":[], "alex_before":[],
                "awis_after":[], "awis_before":[],
                "benjamin_after":[], "benjamin_before":[], 
                "daphne_after":[], "daphne_before":[],
                "hassan_after":[], "hassan_before":[],
                "nils_after":[],  "nils_before":[],
                "roman_after":[], "roman_before":[]}

    upper_folder_path = "masterarbeit-radar/record_on_0909/"

    for name in candidate_names:
        lower_folder_path = upper_folder_path + name
        # print(lower_folder_path)
        files = os.listdir(lower_folder_path)
        for file in files:
            path_to_file = lower_folder_path + "/" + file
            [breath_rates, heart_rates] = load_data(path_to_file)
            #breath_rates = remove_zeros(breath_rates)
            breath_rates = compute_average(breath_rates)
            #heart_rates = remove_zeros(heart_rates)
            # heart_rates = compute_average(heart_rates)
            
            """
            plt.plot(breath_rates, label = "br")
            plt.plot(heart_rates, label = "hr")
            plt.title(file)
            plt.legend()
            plt.show()
            """
            # print(f"length of breath_rates: {len(breath_rates)}, heart_rates: {len(heart_rates)}")
            data_dict_ref[file[:-4]].append(breath_rates)
            data_dict_ref[file[:-4]].append(heart_rates)
            
    return data_dict_ref


def load_ant_data():
    """
    load the estimated breathing and heart rates
    """
    # load 
    # candidate_names = ["stephan", "alex", "awis", "benjamin", "daphne", "hassan", "nils", "roman"]
    # subnames = ["_after_breath_rates_3", "_before_breath_rates_3", "_after_heart_rates_3", "_before_heart_rates_3"]
    subnames = ["_after_breath_rates", "_before_breath_rates", "_after_heart_rates", "_before_heart_rates"]
    data_dict = {"stephan":[], "alex":[], "awis":[], "benjamin":[], "daphne":[], "hassan":[], "nils":[], "roman":[]}

    for name in candidate_names:
        # "masterarbeit-radar/candidates_hrs_brs_ctf/alex"
        # the_path = "masterarbeit-radar/candidates_hrs_brs_combine_ctf/" + name 
        # the_path = "masterarbeit-radar/candidates_hrs_brs_combine_ctf/"
        # the_path = "masterarbeit-radar/candidates_hrs_brs_combine_nls/"
        the_path = "masterarbeit-radar/candidates_hrs_brs_combine_music/"
        for key in data_dict:
            # the first two elements are breath and heart rates after training
            # data_dict[key].append(np.load(the_path + "/" + name + subnames[0] + ".npy"))
            # data_dict[key].append(np.load(the_path + "/" + name + subnames[2] + ".npy"))
            data_dict[key].append(np.load(the_path + name + subnames[0] + ".npy"))
            data_dict[key].append(np.load(the_path + name + subnames[2] + ".npy"))

            # the last two elements are breath and heart rates before training
            data_dict[key].append(np.load(the_path + name + subnames[1] + ".npy"))
            data_dict[key].append(np.load(the_path + name + subnames[3] + ".npy"))
    
    return data_dict


def plot_rmse_br(br_music_before, br_ctf_before, br_anlskf_before,
                 br_music_after, br_ctf_after, br_anlskf_after,
                 hr_music_before, hr_ctf_before, hr_anlskf_before,
                 hr_music_after, hr_ctf_after, hr_anlskf_after):

    plt.subplot(2,2,1)
    plt.plot(br_music_before,label="RMSE of MUSIC")
    plt.plot(br_ctf_before, label="RMSE of CTF")
    plt.plot(br_anlskf_before, label="RMSE OF ANLS&KF")
    plt.ylabel("RMSE/-",fontsize=28)
    #plt.xlabel("Volunteer/-",fontsize=12)
    plt.xticks(fontsize=28)
    plt.yticks(fontsize=28)
    plt.title("Breathing before sport", fontsize=18)
    plt.legend(fontsize=18,loc='upper right')
    plt.grid()


    plt.subplot(2,2,2)
    plt.plot(br_music_after,label="RMSE of MUSIC")
    plt.plot(br_ctf_after, label="RMSE of CTF")
    plt.plot(br_anlskf_after, label="RMSE oF ANLS&KF")
    #plt.ylabel("RMSE/-",fontsize=12)
    #plt.xlabel("Volunteer/-",fontsize=12)
    plt.xticks(fontsize=28)
    plt.yticks(fontsize=28)
    plt.title("Breathing after sport", fontsize=18)
    plt.legend(fontsize=18,loc='upper right')
    plt.grid()

    plt.subplot(2,2,3)
    plt.plot(hr_music_before,label="RMSE of MUSIC")
    plt.plot(hr_ctf_before, label="RMSE of CTF")
    plt.plot(hr_anlskf_before, label="RMSE OF ANLS&KF")
    plt.ylabel("RMSE/-",fontsize=28)
    plt.xlabel("Volunteer/-",fontsize=28)
    plt.xticks(fontsize=28)
    plt.yticks(fontsize=28)
    plt.title("heart before sport", fontsize=18)
    plt.legend(fontsize=18,loc='upper right')
    plt.grid()

    plt.subplot(2,2,4)
    plt.plot(hr_music_after,label="RMSE of MUSIC")
    plt.plot(hr_ctf_after, label="RMSE of CTF")
    plt.plot(hr_anlskf_after, label="RMSE OF ANLS&KF")
    #plt.ylabel("RMSE/-",fontsize=12)
    plt.xlabel("Volunteer/-",fontsize=28)
    plt.xticks(fontsize=28)
    plt.yticks(fontsize=28)
    plt.title("heart after sport", fontsize=18)
    plt.legend(fontsize=18,loc='upper right')
    plt.grid()

    plt.show()
    


# candidate_names = ["stephan" , "alex", "awis", "benjamin", "daphne", "hassan", "nils", "roman"]
candidate_names = ["alex"]
if __name__ == "__main__":
    
    
    """
    br_ctf_before = np.array([0.949, 0.578, 2.322, 1.769, 2.889, 1.551, 0.411, 2.624])
    br_music_before = np.array([0.712, 9.849, 1.955, 8.360, 4.789, 1.444, 2.865, 5.104])
    br_anlskf_before = np.array([0.243, 9.978, 1.997, 8.401, 4.493, 1.773, 5.791, 2.496])
    """

    """
    # Breathing — Before sport
    ANLSKF_breathing_before = np.array([0.331, 10.884, 3.076, 8.279, 5.374, 3.034, 0.333, 1.151])
    CTF_breathing_before    = np.array([0.409, 0.596, 3.054, 1.379, 2.868, 1.516, 0.497, 1.173])
    MUSIC_breathing_before  = np.array([0.663, 11.015, 3.127, 8.051, 6.690, 0.973, 3.005, 1.316])

    # Breathing — After sport
    ANLSKF_breathing_after = np.array([0.419, 7.578, 2.432, 0.586, 1.125, 5.032, 0.844, 3.635])
    CTF_breathing_after    = np.array([0.558, 3.986, 2.747, 0.585, 2.503, 2.176, 0.906, 1.419])
    MUSIC_breathing_after  = np.array([0.483, 10.501, 1.415, 1.511, 1.469, 2.354, 1.560, 1.190])

    # Heartbeat — Before sport
    ANLSKF_heartbeat_before = np.array([7.861, 15.893, 11.626, 13.558, 11.150, 13.552, 12.334, 16.364])
    CTF_heartbeat_before    = np.array([3.980, 20.879, 12.119, 10.774, 8.476, 13.568, 5.922, 20.518])
    MUSIC_heartbeat_before  = np.array([9.531, 12.780, 20.658, 15.761, 13.676, 27.067, 26.969, 9.502])

    # Heartbeat — After sport
    ANLSKF_heartbeat_after = np.array([24.642, 26.257, 20.915, 15.286, 26.914, 28.084, 6.312, 19.184])
    CTF_heartbeat_after    = np.array([6.490, 16.549, 20.189, 16.929, 7.090, 14.098, 12.829, 4.687])
    MUSIC_heartbeat_after  = np.array([4.972, 15.251, 19.802, 13.632, 4.812, 19.602, 15.502, 3.895])

    plot_rmse_br(MUSIC_breathing_before, CTF_breathing_before, ANLSKF_breathing_before,
                 ANLSKF_breathing_after, CTF_breathing_after, MUSIC_breathing_after,
                 ANLSKF_heartbeat_before, CTF_heartbeat_before, MUSIC_heartbeat_before,
                 ANLSKF_heartbeat_after,CTF_heartbeat_after, MUSIC_heartbeat_after)
    


    ant_data = load_ant_data()
    ref_data = load_ref_data()

    for name in candidate_names:
        print(name)
        breath_rates_after = ant_data[name][0]   
        heart_rates_after = ant_data[name][1]

        breath_rates_before = ant_data[name][2]
        heart_rates_before = ant_data[name][3]

        ref_breath_rates_after = ref_data[name+"_after"][0]
        ref_heart_rates_after = ref_data[name+"_after"][1]

        ref_breath_rates_before = ref_data[name+"_before"][0]
        ref_heart_rates_before = ref_data[name+"_before"][1] 

        t_len = 200
        # t_len = max(len(ctf_breath_rates_after), len(ref_breath_rates_before))
        t = np.arange(t_len)
        skip = 10



        ######################### plot rates after ####################################

        #plot_all.plot_ref_ant("music", t[skip:], 
        #                      breath_rates_after[skip:t_len], ref_breath_rates_after[skip:t_len],  breath_rates_before[skip:t_len], ref_breath_rates_before[skip:t_len],
        #                      heart_rates_after[skip:t_len], ref_heart_rates_after[skip:t_len], heart_rates_before[skip:t_len], ref_heart_rates_before[skip:t_len])
        

        rmse_br_before = rmse(ref_breath_rates_before[skip:t_len] ,breath_rates_before[skip:t_len])
        rmse_br_after = rmse(ref_breath_rates_after[skip:t_len] ,breath_rates_after[skip:t_len])

        rmse_hr_before = rmse(ref_heart_rates_before[skip:t_len] ,heart_rates_before[skip:t_len])
        rmse_hr_after = rmse(ref_heart_rates_after[skip:t_len] ,heart_rates_after[skip:t_len])


        #corr = np.correlate(ref_heart_rates_after[:t_len], heart_rates_after[skip:t_len])
        #print(corr)
        print(f"rmse of br " + name + " before:")
        print(f"{rmse_br_before}")
        print(f"rmse of br " + name + " after:")
        print(f"{rmse_br_after}")
        print(f"rmse of hr "  + name + "before:")
        print(f"{rmse_hr_before}")
        print(f"rmse of hr "  + name + "after:")
        print(f"{rmse_hr_after}")
        """
    volunteers = ["volunteer 1", "volunteer 2","volunteer 3","volunteer 4","volunteer 5","volunteer 6","volunteer 7","volunteer 8"]
    cases = ["anls&kf_aft_spt_br", "ctf_aft_spt_br", "music_aft_spt_br",
                "anls&kf_bef_spt_br", "ctf_bef_spt_br", "music_bef_spt_br",
                "anls&kf_aft_spt_hr", "ctf_aft_spt_hr", "music_aft_spt_hr",
                "anls&kf_bef_spt_hr", "ctf_bef_spt_hr", "music_bef_spt_hr"]
    rmse_table = np.array([[0.331, 0.409,0.663,0.419,0.558,0.483,7.861,3.980,9.531,24.642,6.490,4.972],
                    [10.884,0.596,11.015,7.578,3.986,10.501,15.893,20.879,12.780,26.257,16.549,15.251],
                    [3.076, 3.054,3.127,2.432,2.747,1.415,11.616,12.209,11.205,20.915,10.887,19.182],
                    [8.279,1.379,8.051,0.586,0.585,1.451,13.558,10.774,15.761,15.286,16.992,13.632],
                    [5.374,2.868,6.690,1.125,2.503,1.469,11.150,8.476,13.676,26.914,7.090,4.812],
                    [3.034,1.516,0.973,5.032,2.176,2.354,13.552,13.568,27.067,23.084,14.098,19.602],
                    [0.333,0.497,3.005,0.844,0.906,1.560,12.334,5.922,26.969,3.612,12.829,15.502],
                    [1.151,1.173,1.316,3.635,1.419,1.900,16.364,20.518,9.502,19.184,4.687,3.895]])

    heatmap = plt.imshow(rmse_table, aspect='auto')
        
    # Add color bar to represent temperature
    colorbar = plt.colorbar(heatmap, orientation='vertical')
    # colorbar.set_label('Temperature (°C)', labelpad=10)

    # Label the axes
    # plt.xlabel('Day of the Month')
    # plt.ylabel('Month')

    # Set y-ticks to display month names instead of numbers
    plt.yticks(ticks=np.arange(len(volunteers)), labels=volunteers)
    plt.xticks(ticks=np.arange(len(cases)), labels=cases, rotation=30, ha="right", rotation_mode="anchor")
    for i in range(len(volunteers)):
        for j in range(len(cases)):
            plt.text(j,i, rmse_table[i,j],ha="center", va="center", color="r")
    
    plt.show()









    """
    # load data that recorded 10 Sept.
    path_to_folder = "masterarbeit-radar/candidates_hrs_brs"
    folder_path = Path(path_to_folder)

    files = [f.name for f in folder_path.iterdir() if f.is_file()]
    number_of_files = int(len(files)/6)

    for i in np.arange(number_of_files):
        sub_files = files[0+(i*6):5+(i*6)]
        for sub_file in sub_files:
            candidate_data = np.load(path_to_folder+"/"+sub_file)
            len_data = len(candidate_data)
            t = np.arange(len_data) * 0.05
            plt.plot(t, candidate_data)
        plt.show()
    """





"""
# [br_wei_reference, hr_wei_reference] = load_data("data/rohdaten_wei_neu.txt")
# [br_daphne_reference, hr_daphne_reference] = load_data("data/rohdaten_daphne_neu.txt")
[br_wei_reference, hr_wei_reference] = load_data("wei/row_data_wei_parallel_2.txt")
# br_wei_reference = averaging(br_wei_reference)
# hr_wei_reference = averaging(hr_wei_reference)
# br_daphne_reference = averaging(br_daphne_reference)
# hr_daphne_reference = averaging(hr_daphne_reference)


br_wei_antenna = np.load("wei/br_wei_antenna.npy")
hr_wei_antenna = np.load("wei/hr_wei_antenna.npy")
ctf_br_wei_antenna = np.load("wei/ctf_br_wei_antenna.npy")
ctf_hr_wei_antenna = np.load("wei/ctf_hr_wei_antenna.npy")
# anf_br_wei_antenna = np.load("wei/anf_br_wei_antenna.npy")
# anf_hr_wei_antenna = np.load("wei/anf_hr_wei_antenna.npy")


br_daphne_antenna = np.load("daphne/br_daphne_antenna.npy")
hr_daphne_antenna = np.load("daphne/hr_daphne_antenna.npy")
ctf_br_daphne_antenna = np.load("daphne/ctf_br_daphne_antenna.npy")
ctf_hr_daphne_antenna = np.load("daphne/ctf_hr_daphne_antenna.npy")


# t_len_wei = min(len(br_wei_reference), len(br_wei_antenna), len(ctf_br_wei_antenna), len(anf_br_wei_antenna))
t_len_wei = min(len(br_wei_reference), len(br_wei_antenna), len(ctf_br_wei_antenna))
print(f"length of br_wei_reference: {len(br_wei_reference)}")
print(f"length of br_wei_reference: {len(br_wei_antenna)}")
print(f"length of br_wei_reference: {len(ctf_br_wei_antenna)}")
# t_len_daphne = min(len(br_daphne_reference), len(br_daphne_antenna), len(ctf_br_daphne_antenna))
print(f"t_len_wei is: {t_len_wei}")
# print(f"t_len_daphne is: {t_len_daphne}")

start_index = 70
end_index = len(br_wei_reference) - 70

rmse_nls_wei_br = rmse(br_wei_reference[start_index:end_index], br_wei_antenna)
rmse_ctf_wei_br = rmse(br_wei_reference[start_index:end_index], ctf_br_wei_antenna)
# rmse_anf_wei_br = rmse(br_wei_reference[start_index:t_len_wei], anf_br_wei_antenna[start_index:t_len_wei])

rmse_nls_wei_hr = rmse(hr_wei_reference[start_index:end_index], hr_wei_antenna)
rmse_ctf_wei_hr = rmse(hr_wei_reference[start_index:end_index], ctf_hr_wei_antenna)
# rmse_anf_wei_hr = rmse(hr_wei_reference[start_index:t_len_wei], anf_hr_wei_antenna[start_index:t_len_wei])

# rmse_nls_daphne_br = rmse(br_daphne_reference[start_index:t_len_daphne], br_daphne_antenna[start_index:t_len_daphne])
# rmse_ctf_daphne_br = rmse(br_daphne_reference[start_index:t_len_daphne], ctf_br_daphne_antenna[start_index:t_len_daphne])
# rmse_nls_daphne_hr = rmse(hr_daphne_reference[start_index:t_len_daphne], hr_daphne_antenna[start_index:t_len_daphne])
# rmse_ctf_daphne_hr = rmse(hr_daphne_reference[start_index:t_len_daphne], ctf_hr_daphne_antenna[start_index:t_len_daphne])


# len_t = min(len(t_wei), len(t_daphne))
t_interval = 0.1
t_wei = np.arange(t_len_wei) * t_interval
# t_daphne = np.arange(t_len_daphne) * t_interval

plt.figure(0)
plt.plot(t_wei, br_wei_reference[start_index:end_index], label = "br_reference")
plt.plot(t_wei, hr_wei_reference[start_index:end_index], label = "hr_reference")

plt.plot(t_wei, ctf_br_wei_antenna, label = "br_CTF")
plt.plot(t_wei, ctf_hr_wei_antenna, label = "hr_CTF")

plt.plot(t_wei, br_wei_antenna, label = "br_NLS")
plt.plot(t_wei, hr_wei_antenna, label = "hr_NLS")

# plt.plot(t_wei[start_index:t_len_wei], anf_br_wei_antenna[start_index:t_len_wei], label = "br_ANF")
# plt.plot(t_wei[start_index:t_len_wei], anf_hr_wei_antenna[start_index:t_len_wei], label = "hr_ANF")


# plt.title("wei")
plt.xlabel("t/s",fontsize=16)
plt.tick_params(labelsize=16)
plt.legend()
plt.grid()
plt.show()



plt.figure(1)
plt.plot(t_daphne[start_index:t_len_daphne], br_daphne_reference[start_index:t_len_daphne], label = "br_reference")
plt.plot(t_daphne[start_index:t_len_daphne], hr_daphne_reference[start_index:t_len_daphne], label = "hr_reference")

plt.plot(t_daphne[start_index:t_len_daphne], ctf_br_daphne_antenna[start_index:t_len_daphne], label = "br_CTF")
plt.plot(t_daphne[start_index:t_len_daphne], ctf_hr_daphne_antenna[start_index:t_len_daphne], label = "hr_CTF")

plt.plot(t_daphne[start_index:t_len_daphne], br_daphne_antenna[start_index:t_len_daphne], label = "br_NLS")
plt.plot(t_daphne[start_index:t_len_daphne], hr_daphne_antenna[start_index:t_len_daphne], label = "hr_NLS")

# plt.plot(t_hr_wei[0:len_wei_hr], ctf_hr_wei_antenna[0:len_wei_hr], label = "CTF")
# plt.title("daphne")
plt.xlabel("t/s", fontsize=16)
plt.tick_params(labelsize=16)
plt.legend()
plt.grid()
plt.show()



print(f"rmse of weis breath rate using NLS: {rmse_nls_wei_br}")
print(f"rmse of weis breath rate using CTF: {rmse_ctf_wei_br}")
# print(f"rmse of weis breath rate using ANf: {rmse_anf_wei_br}")

print(f"rmse of weis heart rate using NLS: {rmse_nls_wei_hr}")
print(f"rmse of weis heart rate using CTF: {rmse_ctf_wei_hr}")
# print(f"rmse of weis heart rate using ANF: {rmse_anf_wei_hr}")

# print(f"rmse of daphnes breath rate using NLS: {rmse_nls_daphne_br}")
# print(f"rmse of daphnes breath rate using CTF: {rmse_ctf_daphne_br}")

# print(f"rmse of daphnes heart rate using NLS: {rmse_nls_daphne_hr}")
# print(f"rmse of daphnes heart rate using CTF: {rmse_ctf_daphne_hr}")
"""

