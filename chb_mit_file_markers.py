import numpy as np
import pyedflib
import os
import csv

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
    # "P7-T7",
    # "T7-FT9",
    # "FT9-FT10",
    # "FT10-T8",
    # "T8-P8",
]
for index in range(24):
    subject = [str(i).rjust(2,'0') for i in range(index+1,index+2)]
    data_path = '/data/lixinying/chb-mit_18ch/'
    num_channels = 18
    save_path = '/home/lixinying/anomaly_transformer/Anomaly-Transformer/chb-mit_file_markers/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    outputDir = os.path.join(save_path, str(index+1).rjust(2,'0')+"_test_file_markers.csv")
    with open(outputDir, "a+") as f:
        csv_write = csv.writer(f)
        inputData = ["record_id","clip_index" , "label"] # label = 1
        csv_write.writerow(inputData)

    fs = 256
    window_len = 256

    for xpath, dirnanme, fname in os.walk(data_path):
        target = "_label.npy"
        target1 = "chb"+str(index+1).rjust(2,'0')
        label_files = [f for f in fname if f.find(target) != -1 and f.find(target1) != -1]
        # print(label_files[0].split('_label.npy')[0])
    for file in label_files:
        labels = np.load(os.path.join(xpath, file))
        # print(np.sum(labels))
        # print(labels.shape)
        with open(outputDir, "a+") as f: 
            for i in range(0,len(labels)-window_len,window_len):#
                label = labels[i:i+window_len]
                # print(label)
                if np.sum(label)>window_len/2:
                    clip_label = 1
                else:
                    clip_label = 0
                data = [file.split('_label.npy')[0]+'.npy',int(i//window_len),clip_label]
                csv_write = csv.writer(f)
                csv_write.writerow(data)
    
