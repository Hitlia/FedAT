import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import time
import pickle
import dill
import joblib
from utils.utils import *
from sklearn.metrics import *
from model.AnomalyTransformer import AnomalyTransformer
from model.AnomalyTransformer_freq import AnomalyTransformer_freq
from data_factory.data_loader import get_loader_segment
from spot import SPOT, dSPOT

txtDir_base = '/home/lixinying/anomaly_transformer/code/results/'

def pot_eval(init_score, score, q=1e-3, level=0.02):
    """
    Run POT method on given score.
    Args:
        init_score (np.ndarray): The data to get init threshold.
            For `OmniAnomaly`, it should be the anomaly score of train set.
        score (np.ndarray): The data to run POT method.
            For `OmniAnomaly`, it should be the anomaly score of test set.
        label:
        q (float): Detection level (risk)
        level (float): Probability associated with the initial threshold t

    Returns:
        dict: pot result dict
    """
    s = SPOT(q)  # SPOT object
    s.fit(init_score, score)  # data import
    s.initialize(level=level, min_extrema=False)  # initialization step
    ret = s.run(dynamic=False)  # run
    print(len(ret['alarms']))
    print(len(ret['thresholds']))
    pot_th = ret['thresholds'][-1]#-np.mean(ret['thresholds'])
    return pot_th

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
    FA = FP/(FP+TN)
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
        torch.save(model.state_dict(), path)#os.path.join(path, str(self.dataset) + '_checkpoint.pth'))
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
                                              subject_id=self.subject_id,
                                              dataset=self.dataset)
        self.thre_loader = get_loader_segment(self.data_path, batch_size=self.batch_size, win_size=self.win_size, step=self.step,
                                              mode='thre',
                                              subject_id=self.subject_id,
                                              dataset=self.dataset)

        self.build_model()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.criterion = nn.MSELoss()

    def build_model(self):
        con = self.anormly_ratio/100
        if self.model_name == 'anomaly_transformer':
            self.model = AnomalyTransformer(win_size=self.win_size, enc_in=self.input_c, c_out=self.output_c, e_layers=3)
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
            if torch.cuda.is_available():
                self.model.cuda()
        elif self.model_name == 'anomaly_transformer_freq':
            self.model = AnomalyTransformer_freq(win_size=self.win_size, enc_in=self.input_c, c_out=self.output_c, e_layers=3)
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
            if torch.cuda.is_available():
                self.model.cuda()
        # elif self.model_name == 'ABOD':
        #     self.model = ABODSKI(contamination = con)
        # elif self.model_name == 'AutoEncoder':
        #     self.model = AutoEncoderSKI(contamination=con)
        # elif self.model_name == 'AutoRegODetector':
        #     self.model = AutoRegODetectorSKI(contamination=con)
        # elif self.model_name == 'CBLOF':
        #     self.model = CBLOFSKI(contamination=con)
        # elif self.model_name == 'COF':
        #     self.model = COFSKI(contamination=con)
        # elif self.model_name == 'DeepLog':
        #     self.model = DeepLogSKI(contamination=con)
        # elif self.model_name == 'HBOS':
        #     self.model = HBOSSKI(contamination=con)
        # elif self.model_name == 'IsolationForest':
        #     self.model = IsolationForestSKI(contamination=con)
        # elif self.model_name == 'KDiscordODetector':
        #     self.model = KDiscordODetectorSKI(contamination=con, window_size=256)
        # elif self.model_name == 'KNN':
        #     self.model = KNNSKI(contamination=con)
        # elif self.model_name =='LODA':
        #     self.model = LODASKI(contamination=con)
        # elif self.model_name == 'LOF':
        #     self.model = LOFSKI(contamination=con)
        # elif self.model_name == 'LSTMODetector':
        #     self.model = LSTMODetectorSKI(contamination=con)
        # elif self.model_name == 'MatrixProfile':
        #     self.model = MatrixProfileSKI(contamination=con)
        # elif self.model_name == 'Mo_Gaal':
        #     self.model = Mo_GaalSKI(contamination=con)
        # elif self.model_name == 'OCSVM':
        #     self.model = OCSVMSKI(contamination=con)
        # elif self.model_name == 'PCAODetector':
        #     self.model = PCAODetectorSKI(contamination=con, window_size=256)
        # elif self.model_name == 'SOD':
        #     self.model = SODSKI(contamination=con)
        # elif self.model_name == 'So_Gaal':
        #     self.model = So_GaalSKI(contamination=con)
        # elif self.model_name == 'Telemanom':
        #     self.model = TelemanomSKI(l_s=2, n_predictions=1)
        else:
            raise NotImplementedError

    def vali(self, vali_loader):
        self.model.eval()
        loss_1 = []
        loss_2 = []
        for i, (input_data, _) in enumerate(vali_loader):
            input = input_data.float().to(self.device)
            output, series, prior, _ = self.model(input)
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
            series_loss = series_loss / len(prior)
            prior_loss = prior_loss / len(prior)

            rec_loss = self.criterion(output, input)
            loss_1.append((rec_loss - self.k * series_loss).item())
            loss_2.append((rec_loss + self.k * prior_loss).item())

        return np.average(loss_1), np.average(loss_2)

    def train(self):

        print("======================TRAIN MODE======================")

        time_now = time.time()
        path = self.model_save_path
        # if not os.path.exists(path):
        #     os.makedirs(path)
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

                output, series, prior, _ = self.model(input)

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
                    # series_loss += (torch.mean(my_kl_loss(series[u], (
                    #         prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                    #                                                                                self.win_size)))) + torch.mean(
                    #     my_kl_loss((prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                    #                                                                                        self.win_size)),
                    #                series[u])))
                    prior_loss += (torch.mean(my_kl_loss(
                        (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                self.win_size)),
                        series[u].detach())) + torch.mean(
                        my_kl_loss(series[u].detach(), (
                                prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                       self.win_size)))))
                series_loss = series_loss / len(prior)
                prior_loss = prior_loss / len(prior)

                rec_loss = self.criterion(output, input)

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
        if self.model_name == 'anomaly_transformer' or self.model_name == 'anomaly_transformer_freq':
            self.model.load_state_dict(
                torch.load(self.model_save_path))
                   # os.path.join(str(self.model_save_path), str(self.dataset) + '_checkpoint.pth')))
            self.model.eval()
            temperature = 50

            print("======================TEST MODE======================")

            criterion = nn.MSELoss(reduce=False)

            test_labels = []
            attens_energy = []
            # label_slice = []
            for i, (input_data, labels) in enumerate(self.thre_loader):
                input = input_data.float().to(self.device)
                output, series, prior, _ = self.model(input)

                loss = torch.mean(criterion(input, output), dim=-1)

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
                metric = torch.softmax((-series_loss - prior_loss), dim=-1)

                cri = metric * loss
                # cri = metric
                cri = cri.detach().cpu().numpy()
                attens_energy.append(cri)
                test_labels.append(labels)

            attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
            test_labels = np.concatenate(test_labels, axis=0).reshape(-1)
            test_energy = np.array(attens_energy)
            test_labels = np.array(test_labels)
            
            train_energy = list()

            for i, (input_data, labels) in enumerate(self.train_loader):
                input = input_data.float().to(self.device)
                output, series, prior, _ = self.model(input)

                loss = torch.mean(criterion(input, output), dim=-1)

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
                metric = torch.softmax((-series_loss - prior_loss), dim=-1)

                cri = metric * loss
                cri = cri.detach().cpu().numpy()
                train_energy.append(cri)

            train_energy = np.concatenate(train_energy, axis=0).reshape(-1)
            train_energy = np.array(train_energy)
            # combined_energy = np.concatenate((train_energy, test_energy))
            # # thresh = np.percentile(test_energy, 100 - self.anormly_ratio)
            # spot = SPOT()
            # anomaly_indexes = spot.dspot(combined_energy, 256, 5e-4, int(len(train_energy)), 0.999)
            # anomaly_indexes = np.array(anomaly_indexes) - len(train_energy)
            # anomaly_indexes = anomaly_indexes.astype(int)
            # # print(anomaly_indexes)
            # # print(anomaly_indexes)
            # pred = np.zeros_like(test_labels)
            # pred[anomaly_indexes] = 1
            # pred = pred.astype(int)
            thresh = pot_eval(train_energy, test_energy, q=5e-4, level=0.999)
            print("Threshold :", thresh)

            pred = (test_energy > thresh).astype(int)
            # save_path = txtDir_base + '/anomaly_transformer_scores/'+ str(self.subject_id+1).rjust(2,'0') + '_test_energy.npy'
            # np.save(save_path,np.array(test_energy))

            gt = test_labels.astype(int)

            print("pred:   ", pred.shape)
            print("gt:     ", gt.shape)
        else:
            data_train = []          
            for i, (input_data, labels) in enumerate(self.train_loader):
                data_train.append(input_data.float())
            data_train = np.array(np.concatenate(data_train, axis=0).reshape(-1,18)) # data_length*channels

            self.model.fit(data_train)
            print("Training Done.\n")

            data_test = []
            test_labels = []
            for i, (input_data, labels) in enumerate(self.thre_loader):
                data_test.append(input_data.float())
                test_labels.append(labels)
            data_test = np.array(np.concatenate(data_test, axis=0).reshape(-1,18)) # data_length*channels
            test_labels = np.array(np.concatenate(test_labels, axis=0).reshape(-1))

            pred_score = self.model.predict_score(data_test)
            if len(pred_score.shape)==1:
                score = pred_score
            else:
                score = np.mean(pred_score,axis=-1)
            thresh = np.percentile(score, 100 - self.anormly_ratio)
            print("Threshold :", thresh)
            
            pred = (score > thresh).astype(int)

            gt = test_labels[-len(pred):].astype(int)

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

        precision, recall, f_score, support = precision_recall_fscore_support(gt, pred, average='binary')
        acc, SEN, SPEC, FA, roc_auc, precision, SDR, pred_seizure_num, real_seizure_num, f_score = calc_point2point(pred, gt)
        
        print(
            "Sensitivity : {:0.4f}, Specificity : {:0.4f}, FA : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f} ".format(
                SEN, SPEC, FA, precision, recall, f_score))
        print(
            "Accuracy : {:0.4f}, roc_auc : {:0.4f}, SDR : {:0.4f}, Pred_seizure_num : {:0.4f}, Real_seizure_num : {:0.4f}".format(
                acc, roc_auc, SDR, pred_seizure_num, real_seizure_num))

        txtDir = self.results_save_path #txtDir_base + self.model_name + '_test_results.txt'
        with open(txtDir,'a+') as f:
            f.write("Sensitivity : {:0.4f}, Specificity : {:0.4f}, FA : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f} \t".format(
                SEN, SPEC, FA, precision, recall, f_score))
            f.write("Accuracy : {:0.4f}, roc_auc : {:0.4f}, SDR : {:0.4f}, Pred_seizure_num : {:0.4f}, Real_seizure_num : {:0.4f} \n".format(
                acc, roc_auc, SDR, pred_seizure_num, real_seizure_num))

        return acc, SEN, SPEC, FA, roc_auc, precision, recall, SDR, pred_seizure_num, real_seizure_num, f_score
