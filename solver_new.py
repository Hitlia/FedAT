import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import time
from utils.utils import *
from model.AnomalyTransformer import AnomalyTransformer
from data_factory.data_loader import get_loader_segment
from pyentrp import entropy as ent
import scipy.signal
from Pan_tompkins_algorithms import Pan_tompkins
from sklearn.metrics import *
from itertools import groupby
txtDir_base = '/home/lixinying/anomaly_transformer/code/results/'
def calc_point2point(predict, actual):
    """
    calculate f1 score by predict and actual.
    Args:
        predict (np.ndarray): the predict label
        actual (np.ndarray): np.ndarray
    """
    if predict.shape[0] != actual.shape[0]:
        actual = actual[:predict.shape[0]]
    real_seizure_num = np.where(np.diff(actual) == 1)[0].shape[0]
    pred_seizure_num = 0
    actual = actual.astype(int)
    if np.where(np.diff(actual) == 1)[0].shape[0] == np.where(np.diff(actual) == -1)[0].shape[0]:
        for seizure_index in range(np.where(np.diff(actual) == 1)[0].shape[0]):
            start_seizure = np.where(np.diff(actual) == 1)[0][seizure_index] - 10
            end_seizure = np.where(np.diff(actual) == -1)[0][seizure_index]
            sum_seizure = np.sum(predict[start_seizure:end_seizure])
            if sum_seizure > 0:
                pred_seizure_num += 1
    SDR = pred_seizure_num / real_seizure_num
    if SDR < 1:
        print(SDR)
        print(pred_seizure_num)
        print(real_seizure_num)
    TP = np.sum(predict * actual)
    TN = np.sum((1 - predict) * (1 - actual))
    FP = np.sum(predict * (1 - actual))
    FN = np.sum((1 - predict) * actual)
    SPEC = TN/(FP+TN)
    SEN = TP / (TP + FN)
    acc = (TP+TN)/(TP+TN+FP+FN)
    FA = TP/(FP+TN)
    precision = TP/(TP+FP)
    f1 = f1_score(actual, predict, average='macro')
    # f1 = 2 * precision * recall / (precision + recall + 0.00001)
    try:
        roc_auc = roc_auc_score(actual, predict)
    except:
        roc_auc = 0
    return acc, SEN, SPEC, FA, roc_auc, precision, SDR, pred_seizure_num, real_seizure_num, f1

def my_kl_loss(p, q):
    res = p * (torch.log(p + 0.0001) - torch.log(q + 0.0001))
    return torch.mean(torch.sum(res, dim=-1), dim=1)


def adjust_learning_rate(optimizer, epoch, lr_):
    lr_adjust = {epoch: lr_ * (0.5 ** ((epoch - 1) // 1))}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, dataset_name='', delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.best_score2 = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.val_loss2_min = np.Inf
        self.delta = delta
        self.dataset = dataset_name

    def __call__(self, val_loss, val_loss2, model, path):
        score = -val_loss
        score2 = -val_loss2
        if self.best_score is None:
            self.best_score = score
            self.best_score2 = score2
            self.save_checkpoint(val_loss, val_loss2, model, path)
        elif score < self.best_score + self.delta or score2 < self.best_score2 + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_score2 = score2
            self.save_checkpoint(val_loss, val_loss2, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, val_loss2, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), os.path.join(path, str(self.dataset) + '_checkpoint.pth'))
        self.val_loss_min = val_loss
        self.val_loss2_min = val_loss2


class Solver(object):
    DEFAULTS = {}

    def __init__(self, config):

        self.__dict__.update(Solver.DEFAULTS, **config)

        self.train_loader = get_loader_segment(self.data_path, batch_size=self.batch_size, win_size=self.win_size, step=self.step,
                                               mode='train',
                                               dataset=self.dataset)
        self.vali_loader = get_loader_segment(self.data_path, batch_size=self.batch_size, win_size=self.win_size, step=self.step,
                                              mode='val',
                                              dataset=self.dataset)
        self.test_loader = get_loader_segment(self.data_path, batch_size=self.batch_size, win_size=self.win_size, step=self.step,
                                              mode='test',
                                              dataset=self.dataset)
        self.thre_loader = get_loader_segment(self.data_path, batch_size=self.batch_size, win_size=self.win_size, step=self.step,
                                              mode='thre',
                                              dataset=self.dataset)

        self.build_model()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.criterion = nn.MSELoss()

    def build_model(self):
        self.model = AnomalyTransformer(win_size=self.win_size, enc_in=self.input_c, c_out=1, e_layers=3)
        # self.encoder = nn.Linear(1,self.input_c)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        if torch.cuda.is_available():
            self.model.cuda()
            # self.encoder.cuda()

    def vali(self, vali_loader):
        self.model.eval()
        # self.encoder.eval()

        loss_1 = []
        loss_2 = []
        for i, (input_data, _) in enumerate(vali_loader):
            input = input_data.float().to(self.device)
            series_loss_array = []
            prior_loss_array = []
            rec_loss_array = []
            # print(input.shape)
            for channel_idx in range(1):
                # x = self.encoder(input[:,:,channel_idx:channel_idx+1])
                x = input[:,:,channel_idx:channel_idx+1]
                output, series, prior, _ = self.model(x)
                series_loss = 0.0
                prior_loss = 0.0
                for u in range(len(prior)):
                    series_loss += (torch.mean(my_kl_loss(series[u], (
                            prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                self.win_size)).detach())) + torch.mean(
                        my_kl_loss(
                            (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                    self.win_size)).detach(),
                            series[u])))
                    prior_loss += (torch.mean(
                        my_kl_loss((prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                        self.win_size)),
                                series[u].detach())) + torch.mean(
                        my_kl_loss(series[u].detach(),
                                (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                        self.win_size)))))
                series_loss_array.append(series_loss / len(prior))
                prior_loss_array.append(prior_loss / len(prior))
                rec_loss_array.append(self.criterion(output, x))
            series_loss = torch.stack(prior_loss_array,dim=0).mean(dim=0)
            prior_loss = torch.stack(series_loss_array,dim=0).mean(dim=0)
            rec_loss = torch.stack(rec_loss_array,dim=0).mean(dim=0)
            loss_1.append((rec_loss - self.k * series_loss).item())
            loss_2.append((rec_loss + self.k * prior_loss).item())

        return np.average(loss_1), np.average(loss_2)

    def train(self):

        print("======================TRAIN MODE======================")

        time_now = time.time()
        path = self.model_save_path
        if not os.path.exists(path):
            os.makedirs(path)
        early_stopping = EarlyStopping(patience=3, verbose=True, dataset_name=self.dataset)
        train_steps = len(self.train_loader)

        for epoch in range(self.num_epochs):
            iter_count = 0
            loss1_list = []

            epoch_time = time.time()
            self.model.train()
            for i, (input_data, labels) in enumerate(self.train_loader):

                self.optimizer.zero_grad()
                iter_count += 1
                input = input_data.float().to(self.device)
                series_loss_array = []
                prior_loss_array = []
                rec_loss_array = []
                for channel_idx in range(self.input_c):
                    # x = self.encoder(input[:,:,channel_idx:channel_idx+1])
                    x = input[:,:,channel_idx:channel_idx+1]
                    output, series, prior, _ = self.model(x)
                    # output, series, prior, _ = self.model(input[:,:,channel_idx:channel_idx+1])

                    # calculate Association discrepancy
                    series_loss = 0.0
                    prior_loss = 0.0
                    for u in range(len(prior)):
                        series_loss += (torch.mean(my_kl_loss(series[u], (
                                prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                    self.win_size)).detach())) + torch.mean(
                            my_kl_loss((prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                            self.win_size)).detach(),
                                    series[u])))
                        prior_loss += (torch.mean(my_kl_loss(
                            (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                    self.win_size)),
                            series[u].detach())) + torch.mean(
                            my_kl_loss(series[u].detach(), (
                                    prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                        self.win_size)))))
                    series_loss_array.append(series_loss / len(prior))
                    prior_loss_array.append(prior_loss / len(prior))
                    rec_loss_array.append(self.criterion(output, x))
                series_loss = torch.stack(prior_loss_array,dim=0).mean(dim=0)
                prior_loss = torch.stack(series_loss_array,dim=0).mean(dim=0)
                rec_loss = torch.stack(rec_loss_array,dim=0).mean(dim=0)

                loss1_list.append((rec_loss - self.k * series_loss).item())
                loss1 = rec_loss - self.k * series_loss
                loss2 = rec_loss + self.k * prior_loss

                if (i + 1) % 100 == 0:
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.num_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                # Minimax strategy
                loss1.backward(retain_graph=True)
                loss2.backward()
                self.optimizer.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(loss1_list)

            vali_loss1, vali_loss2 = self.vali(self.test_loader)

            print(
                "Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} ".format(
                    epoch + 1, train_steps, train_loss, vali_loss1))
            early_stopping(vali_loss1, vali_loss2, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            adjust_learning_rate(self.optimizer, epoch + 1, self.lr)

    def test(self):
        self.model.load_state_dict(
            torch.load(
                os.path.join(str(self.model_save_path), str(self.dataset) + '_checkpoint_1channel.pth')))
        self.model.eval()
        temperature = 50

        print("======================TEST MODE======================")

        criterion = nn.MSELoss(reduce=False)

        # # (1) stastic on the train set
        # attens_energy = []
        # for channel_idx in range(1):
        #     attens_energy.append(list())
        # for i, (input_data, labels) in enumerate(self.train_loader):
        #     input = input_data.float().to(self.device)
        #     # attens_energy_array = []
        #     for channel_idx in range(1):
        #         # x = self.encoder(input[:,:,channel_idx:channel_idx+1])
        #         x = input[:,:,channel_idx:channel_idx+1]
        #         output, series, prior, _ = self.model(x)
        #         # output, series, prior, _ = self.model(input[:,:,channel_idx:channel_idx+1])
        #         # output, series, prior, _ = self.model(input)
        #         loss = torch.mean(criterion(x, output), dim=-1)
        #         series_loss = 0.0
        #         prior_loss = 0.0
        #         for u in range(len(prior)):
        #             if u == 0:
        #                 series_loss = my_kl_loss(series[u], (
        #                         prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
        #                                                                                             self.win_size)).detach()) * temperature
        #                 prior_loss = my_kl_loss(
        #                     (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
        #                                                                                             self.win_size)),
        #                     series[u].detach()) * temperature
        #             else:
        #                 series_loss += my_kl_loss(series[u], (
        #                         prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
        #                                                                                             self.win_size)).detach()) * temperature
        #                 prior_loss += my_kl_loss(
        #                     (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
        #                                                                                             self.win_size)),
        #                     series[u].detach()) * temperature

        #         metric = torch.softmax((-series_loss - prior_loss), dim=-1)
        #         cri = metric * loss
        #         cri = cri.detach().cpu().numpy()
        #         attens_energy[channel_idx].append(cri)
        # attens_energy_18 = list()
        # for channel_idx in range(1):
        #     attens_energy_18.append(np.concatenate(attens_energy[channel_idx], axis=0).reshape(-1))
        # attens_energy = np.array(attens_energy_18).transpose(1,0)#np.concatenate(attens_energy, axis=0).reshape(-1)
        # train_energy = np.array(attens_energy)

        # evaluation on the test set
        test_labels = []
        attens_energy = []
        for channel_idx in range(18):
            attens_energy.append(list())

        for i, (input_data, labels) in enumerate(self.thre_loader):
            input = input_data.float().to(self.device)
            for channel_idx in range(18):
                # x = self.encoder(input[:,:,channel_idx:channel_idx+1])
                x = input[:,:,channel_idx:channel_idx+1]
                output, series, prior, _ = self.model(x)
                # output, series, prior, _ = self.model(input[:,:,channel_idx:channel_idx+1])

                loss = torch.mean(criterion(x, output), dim=-1)

                series_loss = 0.0
                prior_loss = 0.0
                for u in range(len(prior)):
                    if u == 0:
                        series_loss = my_kl_loss(series[u], (
                                prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                    self.win_size)).detach()) * temperature
                        prior_loss = my_kl_loss(
                            (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                    self.win_size)),
                            series[u].detach()) * temperature
                    else:
                        series_loss += my_kl_loss(series[u], (
                                prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                    self.win_size)).detach()) * temperature
                        prior_loss += my_kl_loss(
                            (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                    self.win_size)),
                            series[u].detach()) * temperature
                # Metric
                metric = torch.softmax((-series_loss - prior_loss), dim=-1)
                cri = metric * loss
                cri = cri.detach().cpu().numpy()
                attens_energy[channel_idx].append(cri)
            test_labels.append(labels)

        test_labels = np.concatenate(test_labels, axis=0).reshape(-1)
        test_labels = np.array(test_labels)
        
        attens_energy_18 = list()
        for channel_idx in range(18):
            attens_energy_18.append(np.concatenate(attens_energy[channel_idx], axis=0).reshape(-1))
        attens_energy = np.array(attens_energy_18).transpose(1,0)#np.concatenate(attens_energy, axis=0).reshape(-1)
        test_energy = np.array(attens_energy)
        combined_energy = test_energy #np.concatenate([train_energy, test_energy], axis=0)
        # print(combined_energy.shape) #seq_len*channnels

        pred_label = np.zeros_like(combined_energy.shape[0])
        signal = np.zeros((combined_energy.shape[0] - 1, combined_energy.shape[1]))
        sample_entropy = np.zeros(combined_energy.shape[1])
        peak_max_10 = np.zeros((combined_energy.shape[1], 10))
        wave_min_10 = np.zeros_like(peak_max_10)
        pred_label = np.zeros_like(signal)
        sampling = 2000
        for i in range(combined_energy.shape[1]):
            signal_temp = Pan_tompkins(combined_energy[:,i], sampling).fit()
            signal[:, i] = signal_temp
            sample_entropy[i] = ent.sample_entropy(np.array(signal_temp), 1, np.std(np.array(signal_temp)) * 0.2)[0]
            peaks = scipy.signal.find_peaks(signal_temp, distance=300)
            index_temp = np.argsort(signal_temp[peaks[0]])
            peak_max_10[i, :] = peaks[0][index_temp[-10::]]

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
        Weight = np.zeros(combined_energy.shape[1])
        SamEN_am = 1 / 2 * np.log((1 - sample_entropy) / sample_entropy)
        SamEN_am[np.where(SamEN_am < 0)[0]] = 0
        SamEN_am = SamEN_am / np.sum(SamEN_am)
        SamEN_am = np.repeat(SamEN_am, pred_label.shape[0]).reshape(pred_label.shape[1], pred_label.shape[0])
        SamEN_am = SamEN_am.transpose()
        pred_label = pred_label * SamEN_am
        pred_label_new = np.sum(pred_label, axis=1)
        pred_label_new_new = np.zeros_like(pred_label_new)
        pred_label_new_new[np.where(pred_label_new > 0.5)[0]] = 1

        pred = pred_label_new_new.astype(int)
        gt = test_labels.astype(int)

        print("pred:   ", pred.shape)
        print("gt:     ", gt.shape)

        # detection adjustment: please see this issue for more information https://github.com/thuml/Anomaly-Transformer/issues/14
        anomaly_state = False
        for i in range(len(gt)):
            if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
                anomaly_state = True
                for j in range(i, 0, -1):
                    if gt[j] == 0:
                        break
                    else:
                        if pred[j] == 0:
                            pred[j] = 1
                for j in range(i, len(gt)):
                    if gt[j] == 0:
                        break
                    else:
                        if pred[j] == 0:
                            pred[j] = 1
            elif gt[i] == 0:
                anomaly_state = False
            if anomaly_state:
                pred[i] = 1

        pred = np.array(pred)
        gt = np.array(gt)
        print("pred: ", pred.shape)
        print("gt:   ", gt.shape)

        # from sklearn.metrics import precision_recall_fscore_support
        # from sklearn.metrics import accuracy_score
        # accuracy = accuracy_score(gt, pred)
        # precision, recall, f_score, support = precision_recall_fscore_support(gt, pred,
        #                                                                       average='binary')
        precision, recall, f_score, support = precision_recall_fscore_support(gt, pred, average='binary')
        acc, SEN, SPEC, FA, roc_auc, precision, SDR, pred_seizure_num, real_seizure_num, f_score = calc_point2point(pred, gt)
        
        print(
            "Sensitivity : {:0.4f}, Specificity : {:0.4f}, FA : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f} ".format(
                SEN, SPEC, FA, precision, recall, f_score))
        print(
            "Accuracy : {:0.4f}, roc_auc : {:0.4f}, SDR : {:0.4f}, Pred_seizure_num : {:0.4f}, Real_seizure_num : {:0.4f}".format(
                acc, roc_auc, SDR, pred_seizure_num, real_seizure_num))

        txtDir = txtDir_base + self.model_name + '_test_results_11.txt'
        with open(txtDir,'a+') as f:
            f.write("Sensitivity : {:0.4f}, Specificity : {:0.4f}, FA : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f} \t".format(
                SEN, SPEC, FA, precision, recall, f_score))
            f.write("Accuracy : {:0.4f}, roc_auc : {:0.4f}, SDR : {:0.4f}, Pred_seizure_num : {:0.4f}, Real_seizure_num : {:0.4f} \n".format(
                acc, roc_auc, SDR, pred_seizure_num, real_seizure_num))

        return acc, SEN, SPEC, FA, roc_auc, precision, recall, SDR, pred_seizure_num, real_seizure_num, f_score
