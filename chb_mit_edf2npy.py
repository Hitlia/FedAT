import numpy as np
import pyedflib
import os

EEG_CHANNELS = [
    "FP1-F7",
    "F7-T7",
    "T7-P7",
    "P7-O1",
    "FP1-F3",
    "F3-C3",
    "C3-P3",
    "P3-O1",
    "FP2-F4",
    "F4-C4",
    "C4-P4",
    "P4-O2",
    "FP2-F8",
    "F8-T8",
    "T8-P8",
    "P8-O2",
    "FZ-CZ",
    "CZ-PZ",
]

def get_selected_channels(channels):
    channels = list(channels)
    ordered_channels = []
    for ch in EEG_CHANNELS:
        try:
            ordered_channels.append(channels.index(ch))
        except:
            return None
    return ordered_channels

def getEDFsignals(edf):
    """
    Get EEG signal in edf file
    Args:
        edf: edf object
    Returns:
        signals: shape (num_channels, num_data_points)
    """
    n = edf.signals_in_file
    samples = edf.getNSamples()[0]
    signals = np.zeros((n, samples))
    for i in range(n):
        try:
            signals[i, :] = edf.readSignal(i)
        except:
            pass
    return signals

subject = [str(i).rjust(2,'0') for i in range(1,2)]
data_path = '/data/physionet.org/files/chbmit/1.0.0/'
num_channels = 18
save_path = '/data/lixinying/chb-mit_18ch/'
if not os.path.exists(save_path):
    os.makedirs(save_path)
fs = 256
# print(subject)
for sub in subject:
    data_path_label = data_path+'chb'+sub+'/chb'+sub+'-summary.txt'
    with open(data_path_label, "r") as f:
        data = f.readlines()

    target = "Number of Seizures in File"
    target2 = "Channels"
    channels_pos = [i for i, item in enumerate(data) if item.find(target2) != -1]
    # print(sub)
    # print(channels_pos)
    num_of_seizure = np.array([int(data[i].split()[-1]) for i, item in enumerate(data) if item.find(target) != -1])
    file_index = np.array([i-3 for i, item in enumerate(data) if item.find(target) != -1])
    seizure_index = [i+1 for i, item in enumerate(num_of_seizure) if item > 0]
    # print(seizure_index)
    start_time_all = list()
    end_time_all = list()
    for seizure_file in seizure_index:
        seizure_num = num_of_seizure[seizure_file-1]
        start_index = file_index[seizure_file-1]+2
        if len(channels_pos) == 1:
            channels = [data[i].split()[-1] for i, item in enumerate(data) if item.find("Channel ") != -1]
            # print(len(channels))
            # print(channels)
        else:
            if sub == '24':
                si = start_index
            else:
                si = start_index-2
            pos = [i for i, item in enumerate(channels_pos) if item < si][-1]
            if pos == len(channels_pos)-1:
                channels = [data[i].split()[-1] for i, item in enumerate(data) if item.find("Channel ") != -1 and i > channels_pos[pos]]
            else:
                channels = [data[i].split()[-1] for i, item in enumerate(data) if item.find("Channel ") != -1 and i > channels_pos[pos] and i < channels_pos[pos+1]]
            # print(channels)
            # print(len(channels))
        selected_channels = get_selected_channels(channels)
        # print(selected_channels)
        if selected_channels != None:
            if sub == '24':
                data_file = data_path+'chb'+sub+'/'+data[start_index].split()[-1]
            else:
                data_file = data_path+'chb'+sub+'/'+data[start_index-2].split()[-1]
            f = pyedflib.EdfReader(data_file)
            print(data_file.split('/')[-1].split('.edf')[0])
            signals = getEDFsignals(f)
            selected_signals = signals[selected_channels,:]
            labels = np.zeros(signals.shape[1])
            # print(selected_signals.shape)
            start_time_list = list()
            end_time_list = list()
            for i in range(1,seizure_num+1):
                start_index += 2
                start_time_list.append(int(data[start_index].split()[-2]))
                end_time_list.append(int(data[start_index+1].split()[-2]))
            for idx in range(len(start_time_list)):
                labels[start_time_list[idx]*fs:end_time_list[idx]*fs] = 1
            # print(labels.shape)
            save_file_name = save_path + data_file.split('/')[-1].split('.edf')[0] + '.npy'
            save_label_name = save_path + data_file.split('/')[-1].split('.edf')[0] + '_label.npy'
            # print(np.sum(labels))
            # np.save(save_file_name,np.array(selected_signals).transpose(1,0))
            # np.save(save_label_name,np.array(labels))
            start_time_all.append(start_time_list)
            end_time_all.append(end_time_list)
    # print(start_time_all)
    # print(end_time_all)
