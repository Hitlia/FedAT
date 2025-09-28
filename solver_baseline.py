import torch
import torch.nn as nn
import numpy as np
import time
from utils.utils import *
from sklearn.metrics import *
from deepod.models.time_series.couta import COUTA
from deepod.models.time_series.dcdetector import DCdetector
from deepod.models.time_series.dif import DeepIsolationForestTS
from deepod.models.time_series.ncad import NCAD
from deepod.models.time_series.tranad import TranAD
from deepod.models.time_series.tcned import TcnED
from model.STEN import STEN
from data_factory.data_loader import get_loader_segment
from spot import SPOT

def pot_eval(init_score, score, q=1e-3, level=0.02):
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
        # self.vali_loader = get_loader_segment(self.data_path, batch_size=self.batch_size, win_size=self.win_size, step=self.step,
        #                                       mode='val',
        #                                       dataset=self.dataset)
        # self.test_loader = get_loader_segment(self.data_path, batch_size=self.batch_size, win_size=self.win_size, step=self.step,
        #                                       mode='test',
        #                                       subject_id=self.subject_id,
        #                                       dataset=self.dataset)
        # self.thre_loader = get_loader_segment(self.data_path, batch_size=self.batch_size, win_size=self.win_size, step=self.step,
        #                                       mode='thre',
        #                                       subject_id=self.subject_id,
        #                                       dataset=self.dataset)

        self.build_model()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.criterion = nn.MSELoss()

    def build_model(self):
        con = self.anormly_ratio/100
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if self.model_name == 'COUTA':
            self.model = COUTA(seq_len=self.win_size, stride=self.step, epochs=self.num_epochs, hidden_dims=64, device=self.device)
            # self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
            # if torch.cuda.is_available():
            #     self.model.cuda()
        elif self.model_name == 'DCdetector':
            self.model = DCdetector(seq_len=self.win_size, stride=self.step, epochs=self.num_epochs, batch_size=self.batch_size, lr=self.lr, patch_size=[self.step], device=self.device)
        elif self.model_name == 'DIF':
            self.model = DeepIsolationForestTS(seq_len=self.win_size, stride=self.step, epochs=self.num_epochs, hidden_dims='64', device=self.device)
        elif self.model_name == 'NCAD':
            self.model = NCAD(seq_len=self.win_size, stride=self.step, epochs=self.num_epochs, batch_size=self.batch_size, lr=self.lr, device=self.device)
        elif self.model_name == 'TranAD':
            self.model = TranAD(seq_len=self.win_size, stride=self.step, epochs=self.num_epochs, device=self.device)
        elif self.model_name == 'TcnED':
            self.model = TcnED(seq_len=self.win_size, stride=self.step, epochs=self.num_epochs, hidden_dims=64, device=self.device)
        elif self.model_name == 'STEN':
            self.model = STEN(seq_len=int(self.win_size/8), stride=self.step, epoch=self.num_epochs, hidden_dim=64, batch_size=self.batch_size, lr=self.lr, device=self.device)
            # path = '/home/lixinying/anomaly_transformer/code/checkpoints/sten_test.pth'
            # torch.save(self.model,path)
        else:
            raise NotImplementedError

    # def vali(self, vali_loader):
    #     self.model.eval()
    #     loss_1 = []
    #     loss_2 = []
    #     for i, (input_data, _) in enumerate(vali_loader):
    #         input = input_data.float().to(self.device)
    #         output, series, prior, _ = self.model(input)
    #         series_loss = 0.0
    #         prior_loss = 0.0
    #         for u in range(len(prior)):
    #             series_loss += (torch.mean(my_kl_loss(series[u], (
    #                     prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
    #                                                                                            self.win_size)).detach())) + torch.mean(
    #                 my_kl_loss(
    #                     (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
    #                                                                                             self.win_size)).detach(),
    #                     series[u])))
    #             prior_loss += (torch.mean(
    #                 my_kl_loss((prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
    #                                                                                                    self.win_size)),
    #                            series[u].detach())) + torch.mean(
    #                 my_kl_loss(series[u].detach(),
    #                            (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
    #                                                                                                    self.win_size)))))
    #         series_loss = series_loss / len(prior)
    #         prior_loss = prior_loss / len(prior)

    #         rec_loss = self.criterion(output, input)
    #         loss_1.append((rec_loss - self.k * series_loss).item())
    #         loss_2.append((rec_loss + self.k * prior_loss).item())

    #     return np.average(loss_1), np.average(loss_2)

    def train(self):

        print("======================TRAIN MODE======================")
        data_train = []          
        for i, (input_data, labels) in enumerate(self.train_loader):
            data_train.append(input_data.float())
        data_train = np.array(np.concatenate(data_train, axis=0).reshape(-1,18)) # data_length*channels
        self.data_train = data_train
        self.model.fit(data_train)
        path = self.model_save_path
        torch.save(self.model,path,pickle_protocol=4)
        print("Training Done.\n")
        
    def test(self,sub_id):
        self.model = torch.load(self.model_save_path)
        # self.model.eval()
        self.subject_id = sub_id

        self.thre_loader = get_loader_segment(self.data_path, batch_size=self.batch_size, win_size=self.win_size, step=self.step,
                                              mode='thre',
                                              subject_id=self.subject_id,
                                              dataset=self.dataset)
        
        data_test = []
        test_labels = []
        for i, (input_data, labels) in enumerate(self.thre_loader):
            data_test.append(input_data.float())
            test_labels.append(labels)
        data_test = np.array(np.concatenate(data_test, axis=0).reshape(-1,18)) # data_length*channels
        test_labels = np.array(np.concatenate(test_labels, axis=0).reshape(-1))
        if len(test_labels)>self.win_size*1800:
            pred_score = list()
            for i in range(len(test_labels)//(self.win_size*1800)):
                data_test_clip = data_test[i*self.win_size*1800:i*self.win_size*1800+self.win_size*1800,:]
                pred_score_clip = self.model.decision_function(data_test_clip)
                pred_score += pred_score_clip.tolist()
            if len(test_labels)%(self.win_size*1800) != 0:
                data_test_clip = data_test[len(test_labels)//(self.win_size*1800)*self.win_size*1800:,:]
                pred_score_clip = self.model.decision_function(data_test_clip)
                pred_score += pred_score_clip.tolist()
            pred_score = np.array(pred_score)
        else:
            pred_score = self.model.decision_function(data_test)
        # pred_score = self.model.predict_score(data_test)
        if len(pred_score.shape)==1:
            score = pred_score
        else:
            score = np.mean(pred_score,axis=-1)
        
        if self.use_pot:
           
            # train_score = self.model.decision_function(self.data_train)
            data_train = []          
            for i, (input_data, labels) in enumerate(self.train_loader):
                data_train.append(input_data.float())
            data_train = np.array(np.concatenate(data_train, axis=0).reshape(-1,18)) # data_length*channels
            self.data_train = data_train
            
            if len(self.data_train)>self.win_size*1800:
                train_score = list()
                for i in range(len(self.data_train)//(self.win_size*1800)):
                    data_test_clip = self.data_train[i*self.win_size*1800:i*self.win_size*1800+self.win_size*1800,:]
                    pred_score_clip = self.model.decision_function(data_test_clip)
                    train_score += pred_score_clip.tolist()
                if len(self.data_train)%(self.win_size*1800) != 0:
                    data_test_clip = self.data_train[len(self.data_train)//(self.win_size*1800)*self.win_size*1800:,:]
                    pred_score_clip = self.model.decision_function(data_test_clip)
                    train_score += pred_score_clip.tolist()
                train_score = np.array(train_score)
            else:
                train_score = self.model.decision_function(self.data_train)
            
            if len(train_score.shape)==1:
                score_train = train_score
            else:
                score_train = np.mean(train_score,axis=-1)
            thresh = pot_eval(score_train, score, q=5e-3, level=0.999)
        else:
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
