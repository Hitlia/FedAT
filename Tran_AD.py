# 编辑于4.10 epoch=20 save_model='/home/YaoGuo/code/TranAD/checkpoints/TranAD/2024-04-10 13:34:0860'
import pickle
import os
import scipy.signal
import matplotlib.pyplot as plt
from src.errors import Errors
import numpy as np
import pandas as pd
from tqdm import tqdm
from src.models import *
from src.constants import *
from src.plotting import *
from src.pot import *
from src.utils import *
from src.diagnosis import *
from src.merlin import *
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch.nn as nn
from time import time
from pprint import pprint
import scipy.io as sio
# from beepy import beep
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
# device = "cpu"
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
import src.helpers as helpers
from src.helpers import Config
logger = helpers.setup_logging()
np.set_printoptions(suppress=True)
from itertools import groupby
from Pan_tompkins_algorithms import Pan_tompkins
from pyentrp import entropy as ent
import random
def convert_to_windows(data, model):
    windows = [];
    w_size = model.n_window
    new_data = data
    new_data_array = new_data.numpy()
    new_data = new_data[:, 0:18]
    for i, g in enumerate(new_data):
        if i >= w_size:
            if new_data_array[i - 1, -1] - new_data_array[i - w_size, -1] == 0:
                w = new_data[i - w_size:i]
            else:
                temp = np.diff(new_data_array[i - w_size:i, -1])
                temp_value = np.where(temp != 0)[0] + 1
                w = torch.cat([new_data[i - w_size + temp_value].repeat(int(temp_value), 1),
                               new_data[int(i - w_size + temp_value):i]])
        else:
            w = torch.cat([new_data[0].repeat(w_size - i, 1), new_data[0:i]])
        # print(w.contiguous().view(-1).shape)
        windows.append(w if 'TranAD' in args.model or 'Attention' in args.model else w.contiguous().view(-1))
    # print(w.view(-1).shape)
    return torch.stack(windows)



def load_dataset(dataset, subject):
    loader_EEG = []
    loader_label = []
    loader_group = []
    n = 0
    for i in range(subject, subject + 1):
        if i < 10:
            para_folder = output_folder_chb_mit_aEEG + 'chb0' + str(i) + '/'
        else:
            para_folder = output_folder_chb_mit_aEEG + 'chb' + str(i) + '/'
        all_file_name = os.listdir(para_folder)
        #  if subject==2:
        #     all_file_name.sort(key=lambda x: int(x.split('_chb')[1].split('_')[1].split('.mat')[0]), reverse=True)
        # elif subject==14:
        #     all_file_name.sort(key=lambda x: int(x.split('_chb')[1].split('_')[1].split('.mat')[0]))
        # elif subject == 24:
        #     all_file_name.sort(key=lambda x: int(x.split('_chb')[1].split('_')[1].split('.mat')[0]))
    
        all_ch_ind = range(0, 18)
        for j in range(len(all_file_name)):
            n = n + 1
            sub_file = os.path.join(para_folder, all_file_name[j])
            temp = sio.loadmat(sub_file)
            label_temp = temp['label']
            temp_EEG = np.transpose(temp['aEEG_Up'])
            # EEG = np.concatenate((temp_EEG[:_el_temp, (EEG.shape[0], 1)), temp_EEG.shape[1], axis=1)
            EEG = np.concatenate((temp_EEG, n * np.ones((np.shape(temp_EEG)[0], 1))), axis=1)
            label = np.repeat(np.reshape(label_temp, (EEG.shape[0], 1)), EEG.shape[1], axis=1)

            if n == 1:
                loader_EEG = EEG
                loader_label = label
            else:
                loader_EEG = np.concatenate((loader_EEG, EEG), axis=0)
                loader_label = np.concatenate((loader_label, label), axis=0)
        
        scaler = MinMaxScaler()
        scaler = scaler.fit(loader_EEG)
        loader_EEG = scaler.transform(loader_EEG)



    train_loader = DataLoader(loader_EEG, batch_size=loader_EEG.shape[0])
    labels = loader_label
    return train_loader, labels


def save_model(model, optimizer, scheduler, epoch, accuracy_list, folder):
    # folder = f'{save_road}/aEEG_patient_{args.subject}'
    file_path = f'{folder}/model.ckpt'
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'accuracy_list': accuracy_list}, file_path)


def load_model(modelname, dims):
    import src.models
    model_class = getattr(src.models, modelname)
    model = model_class(dims).double()
    optimizer = torch.optim.AdamW(model.parameters(), lr=model.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 5, 0.9)
    fname = f'checkpoints/{args.model}_{args.dataset}/aEEG_patient_{args.subject}/model.ckpt'
    if os.path.exists(fname) and (not args.retrain or args.test):
        print(f"{color.GREEN}Loading pre-trained model: {model.name}{color.ENDC}")
        checkpoint = torch.load(fname)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        epoch = checkpoint['epoch']
        accuracy_list = checkpoint['accuracy_list']
    else:
        print(f"{color.GREEN}Creating new model: {model.name}{color.ENDC}")
        epoch = -1;
        accuracy_list = []
    return model, optimizer, scheduler, epoch, accuracy_list


def backprop(epoch, model, data, dataO, optimizer, scheduler, save_road, training=True):
    l = nn.MSELoss(reduction='mean' if training else 'none')
    # feats = dataO.shape[1]
    feats = 18
    if 'TranAD' in model.name:
        l = nn.MSELoss(reduction='none')
        data_x = torch.DoubleTensor(data)
        dataset = TensorDataset(data_x, data_x)
        # bs = model.batch if training else len(data)
        bs = model.batch
        dataloader = DataLoader(dataset, batch_size=bs)
        n = epoch + 1
        w_size = model.n_window
        l1s, l2s = [], []
        if training:
            for d, _ in dataloader:

                local_bs = d.shape[0]
                window = d.permute(1, 0, 2)
                elem = window[-1, :, :].view(1, local_bs, feats)
                window, elem = window.to(device), elem.to(device)
                window[-1, :, :] = 0
                z = model(window, elem) ## 
                l1 = l(z, elem) if not isinstance(z, tuple) else (1 / n) * l(z[0], elem) + (1 - 1 / n) * l(z[1], elem)
                if isinstance(z, tuple): z = z[1]
                l1s.append(torch.mean(l1).item())
                loss = torch.mean(l1)
                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                optimizer.step()
            scheduler.step()
            tqdm.write(f'Epoch {epoch},\tL1 = {np.mean(l1s)}')
            loss_txt_road = save_road + 'loss.txt'
            with open(loss_txt_road, 'a+') as file:
                file.write(f'Epoch {epoch},\tL1 = {np.mean(l1s)}\n')       
            return np.mean(l1s), optimizer.param_groups[0]['lr']
        else:
            for d, _ in dataloader:
                local_bs = d.shape[0]
                window = d.permute(1, 0, 2)
                elem = window[-1, :, :].view(1, local_bs, feats)
                window, elem = window.to(device), elem.to(device)
                window[-1, :, :] = 0
                z = model(window, elem)
                if isinstance(z, tuple): z = z[1]
                l1 = l(z, elem) if not isinstance(z, tuple) else (1 / n) * l(z[0], elem) + (1 - 1 / n) * l(z[1], elem)
                # l1s.append(l1[0].squeeze().detach().cpu().numpy())
                if l1s == []:
                    l1s = l1[0].squeeze().detach().cpu().numpy()
                    z1 = z.detach().cpu().numpy()[0]
                    ori_data = elem.detach().cpu().numpy()[0]
                else:
                    l1s = np.concatenate((l1s, l1[0].squeeze().detach().cpu().numpy()), axis=0)
                    z1 = np.concatenate((z1, z.detach().cpu().numpy()[0]), axis=0)
                    ori_data =np.concatenate((ori_data, elem.detach().cpu().numpy()[0]), axis=0)
        return l1s, z1, ori_data

    else:
        y_pred = model(data)
        loss = l(y_pred, data)
        if training:
            tqdm.write(f'Epoch {epoch},\tMSE = {loss}')
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            return loss.item(), optimizer.param_groups[0]['lr']
        else:
            return loss.detach().numpy(), y_pred.detach().numpy()


if __name__ == '__main__':
    sub = [1, 2, 3, 4, 5, 7, 8, 9, 11, 19, 22, 23]
    current_time = str(datetime.now()).split('.')[0] + str(random.randint(1, 100))
    print("The start time is", current_time)
    result = np.zeros((12, 9))
    for sub_i in range(0, 12):
        args.subject = sub[sub_i]
        config = Config('config.yaml')
        ##生成模型、图片以及表格的存储路径
        save_model_road = './checkpoints/' +  args.model + '/' + current_time + '/aEEG_patient_' + str(args.subject) + '/'
        os.makedirs(save_model_road, exist_ok=True)
        train_loader, labels = load_dataset(args.dataset, args.subject)
        model, optimizer, scheduler, epoch, accuracy_list = load_model(args.model, 18)
        model.to(device)
        #
        # ## Prepare data
        trainD = next(iter(train_loader))
        # testD = trainD
        trainO = trainD
        if model.name in ['Attention', 'DAGMM', 'USAD', 'MSCRED', 'CAE_M', 'GDN', 'MTAD_GAT', 'MAD_GAN'] or 'TranAD' in model.name:
            trainD = convert_to_windows(trainD, model) #23925, 10, 18

        ### Testing phase
        torch.zero_grad = True
        model.eval()
        loss, y_hat, y_test = backprop(0, model, trainD, trainO, optimizer, scheduler, save_model_road, training=False)
        for i in range(loss.shape[1]):
            ls = labels[:, i]
            errors = Errors(y_hat[:, i], y_test[:, i], config)
            errors.process_batches(y_test[:, i], )
            if i == 0:
                e_s = [errors.e_s]
            else:
                e_s.append(errors.e_s)
        pred_label = np.zeros_like(ls)
        signal = np.zeros((np.shape(e_s[0])[0] - 1, len(e_s)))
        sample_entropy = np.zeros(len(e_s))
        peak_max_10 = np.zeros((len(e_s), 10))
        wave_min_10 = np.zeros_like(peak_max_10)
        pred_label = np.zeros_like(signal)
        sampling = 2000
        for i in range(len(e_s)):
            signal_temp = Pan_tompkins(e_s[i], sampling).fit()
            signal[:, i] = signal_temp
            sample_entropy[i] = ent.sample_entropy(np.array(signal_temp), 1, np.std(np.array(signal_temp)) * 0.2)[0]
            peaks = scipy.signal.find_peaks(signal_temp, distance=300)
            index_temp = np.argsort(signal_temp[peaks[0]])
            peak_max_10[i, :] = peaks[0][index_temp[-10::]]
            # if i == 9:
            #     print(i)
            for wave_i in range(9, -1, -1):
                if peak_max_10[i, wave_i] < 300:
                    min = np.where(
                        np.diff(signal[int(0): int(peak_max_10[i, wave_i]), i]) < 0)[0]
                else:
                    min = \
                    np.where(np.diff(signal[int(peak_max_10[i, wave_i] - 300): int(peak_max_10[i, wave_i]), i]) < 0)[0]
                fun = lambda x: x[1] - x[0]
                re_n = 0
                re = []
                for k, g in groupby(enumerate(min), fun):
                    a = [v for i, v in g]
                    if re_n == 0:
                        re = [np.array(a)]
                        re_n += 1
                    else:
                        re.append(np.array(a))
                last_t = len(re)
                wave_min_10[i, wave_i] = peak_max_10[i, wave_i] - 300 + re[last_t - 1][-1]
                while re[last_t - 1].shape[0] < 3 and last_t - 2 > -1:
                    wave_min_10[i, wave_i] = peak_max_10[i, wave_i] - 300 + re[last_t - 1][-1]
                    last_t -= 1
                pred_label[int(wave_min_10[i, wave_i]) - 200: int(peak_max_10[i, wave_i]), i] = 1
        if np.min(sample_entropy) > 0.5:
            sample_entropy = sample_entropy - 0.5 
        sample_entropy[np.where(sample_entropy > 0.99)[0]] = 0.99
        Weight = np.zeros(len(e_s))
        SamEN_am = 1 / 2 * np.log((1 - sample_entropy) / sample_entropy)
        SamEN_am[np.where(SamEN_am < 0)[0]] = 0
        SamEN_am = SamEN_am / np.sum(SamEN_am)
        SamEN_am = np.repeat(SamEN_am, pred_label.shape[0]).reshape(pred_label.shape[1], pred_label.shape[0])
        SamEN_am = SamEN_am.transpose()
        pred_label = pred_label * SamEN_am
        pred_label_new = np.sum(pred_label, axis=1)
        pred_label_new_new = np.zeros_like(pred_label_new)
        pred_label_new_new[np.where(pred_label_new > 0.5)[0]] = 1








