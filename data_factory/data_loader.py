import torch
import os
import random
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
# from PIL import Image
import numpy as np
# import collections
# import numbers
# import math
import pandas as pd
from sklearn.preprocessing import StandardScaler
# import pickle
# import tqdm

class myStandardScaler:
    """
    Standardize the input
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    # def transform(self, data):
    #     return (data - self.mean) / self.std
    def transform(self, data):
        # print(data.shape)
        _, num_nodes = self.mean.shape
        for i in range(num_nodes):
            data[:,i] = (data[:,i]-self.mean[:,i])/self.std[:,i]
        return data



class PSMSegLoader(object):
    def __init__(self, data_path, win_size, step, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = pd.read_csv(data_path + '/train.csv')
        data = data.values[:, 1:]

        data = np.nan_to_num(data)

        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = pd.read_csv(data_path + '/test.csv')

        test_data = test_data.values[:, 1:]
        test_data = np.nan_to_num(test_data)

        self.test = self.scaler.transform(test_data)

        self.train = data
        self.val = self.test

        self.test_labels = pd.read_csv(data_path + '/test_label.csv').values[:, 1:]

        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):
        """
        Number of images in the object dataset.
        """
        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


class MSLSegLoader(object):
    def __init__(self, data_path, win_size, step, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(data_path + "/MSL_train.npy")
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(data_path + "/MSL_test.npy")
        self.test = self.scaler.transform(test_data)

        self.train = data
        self.val = self.test
        self.test_labels = np.load(data_path + "/MSL_test_label.npy")
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):

        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


class SMAPSegLoader(object):
    def __init__(self, data_path, win_size, step, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(data_path + "/SMAP_train.npy")
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(data_path + "/SMAP_test.npy")
        self.test = self.scaler.transform(test_data)

        self.train = data
        self.val = self.test
        self.test_labels = np.load(data_path + "/SMAP_test_label.npy")
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):

        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


class SMDSegLoader(object):
    def __init__(self, data_path, win_size, step, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(data_path + "/SMD_train.npy")
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(data_path + "/SMD_test.npy")
        self.test = self.scaler.transform(test_data)
        self.train = data
        data_len = len(self.train)
        self.val = self.train[(int)(data_len * 0.8):]
        self.test_labels = np.load(data_path + "/SMD_test_label.npy")

    def __len__(self):

        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])

class CHBMITLoader(object):
    def __init__(self, data_path, win_size, marker_path):
        # self.mode = mode
        # self.step = step
        self.win_size = win_size
        self.data_path = data_path
        self.df_file = marker_path
        self.records = self.df_file["record_id"].tolist()
        self.labels = self.df_file["label"].tolist()
        self.clip_idxs = self.df_file["clip_index"].tolist()
        self.mean, self.std = self._compute_mean_std(list(set(self.records)))
        # print(self.mean.shape)
        self.scaler = myStandardScaler(mean=self.mean, std=self.std)

    def __len__(self):
        return len(self.df_file)

    def __getitem__(self, index):
        file_name = self.records[index]
        y = self.labels[index]
        clip_idx = int(self.df_file.iloc[index]["clip_index"])

        physical_len = self.win_size
        start_idx = clip_idx * physical_len
        end_idx = start_idx + physical_len

        signals = np.load(os.path.join(self.data_path, file_name))
        x = signals[start_idx:end_idx,:]
        labels = np.load(os.path.join(self.data_path, file_name.split('.npy')[0]+'_label.npy'))
        label = labels[start_idx:end_idx]
        # import h5py
        # raw_data_path = '/home/lixinying/data/resampled_tuh/'
        # with h5py.File(os.path.join(raw_data_path, file_name), "r") as hf:
        #         signals = hf["resampled_signal"][()]
        # x = signals[:-1,start_idx:end_idx].transpose((1,0)) 
        if y == 0:
            label = np.zeros(self.win_size)
        else:
            label = np.ones(self.win_size)


        if self.scaler is not None:
            # standardize
            x = self.scaler.transform(x)
        
        return np.float32(x), np.float32(label)
        
    def _compute_mean_std(self, train_files, num_nodes=18):
        count = 0
        signal_sum = np.zeros((num_nodes))
        signal_sum_sqrt = np.zeros((num_nodes))
        print("Computing mean and std of training data...")
        for idx in range(len(train_files)):
            # import h5py
            # raw_data_path = '/home/lixinying/data/resampled_tuh/'
            # with h5py.File(os.path.join(raw_data_path, train_files[idx]), "r") as hf:
            #     signals = hf["resampled_signal"][()]
            #     signal = signals[:-1,:].transpose((1,0))
            signal = np.load(os.path.join(self.data_path, train_files[idx]))
            signal_sum += signal.sum(axis=0)
            signal_sum_sqrt += (signal**2).sum(axis=0)
            count += signal.shape[0]
        total_mean = signal_sum / count
        total_var = (signal_sum_sqrt / count) - (total_mean**2)
        total_std = np.sqrt(total_var)

        return np.expand_dims(total_mean, 0), np.expand_dims(total_std, 0)



def get_loader_segment(data_path, batch_size, win_size=100, step=100, mode='train', subject_id=0,dataset='KDD'):
    if (dataset == 'SMD'):
        dataset = SMDSegLoader(data_path, win_size, step, mode)
    elif (dataset == 'MSL'):
        dataset = MSLSegLoader(data_path, win_size, 1, mode)
    elif (dataset == 'SMAP'):
        dataset = SMAPSegLoader(data_path, win_size, 1, mode)
    elif (dataset == 'PSM'):
        dataset = PSMSegLoader(data_path, win_size, 1, mode)
    elif (dataset == 'CHBMIT'):
        if mode == 'train' or mode == 'val':
            marker_file = '/home/lixinying/anomaly_transformer/code/chb-mit_file_markers/train_file_markers.csv'
        else:
            marker_file = '/home/lixinying/anomaly_transformer/code/chb-mit_file_markers/'+str(subject_id+1).rjust(2,'0')+'_test_file_markers.csv'
        # print(marker_file)
        marker_path = pd.read_csv(marker_file)
        dataset = CHBMITLoader(data_path, win_size, marker_path)

    shuffle = False
    if mode == 'train':
        shuffle = True

    data_loader = DataLoader(dataset=dataset,
                             batch_size=batch_size,
                             shuffle=shuffle,
                             num_workers=0)
    return data_loader

