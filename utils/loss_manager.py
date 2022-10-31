import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
import torch.autograd as autograd
import os
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

class LogManager:
    def __init__(self):
        self.log_book=dict()
    def alloc_stat_type(self, stat_type):
        self.log_book[stat_type] = []
    def alloc_stat_type_list(self, stat_type_list):
        for stat_type in stat_type_list:
            self.alloc_stat_type(stat_type)
    def init_stat(self):
        for stat_type in self.log_book.keys():
            self.log_book[stat_type] = []
    def add_stat(self, stat_type, stat):
        assert stat_type in self.log_book, "Wrong stat type"
        self.log_book[stat_type].append(stat)
    def add_torch_stat(self, stat_type, stat):
        assert stat_type in self.log_book, "Wrong stat type"
        self.log_book[stat_type].append(stat.detach().cpu().item())
    def get_stat(self, stat_type):
        result_stat = 0
        stat_list = self.log_book[stat_type]
        if len(stat_list) != 0:
            result_stat = np.mean(stat_list)
            result_stat = np.round(result_stat, 4)
        return result_stat

    def print_stat(self):
        for stat_type in self.log_book.keys():
            if len(self.log_book[stat_type]) == 0:
                continue
            stat = self.get_stat(stat_type)           
            print(stat_type,":",stat, end=' / ')
        print(" ")


# For Hard-label learning
def CE_category(pred, lab):
    celoss = nn.CrossEntropyLoss()
    max_indx = torch.argmax(lab, dim=1)
    ce_loss = celoss(pred, max_indx)
    return ce_loss

def calc_err(pred, lab):
    p = pred.detach()
    t = lab.detach()
    total_num = p.size()[0]
    ans = torch.argmax(p, dim=1)
    tar = torch.argmax(t, dim=1)
    corr = torch.sum((ans==tar).long())
    err = (total_num-corr) / total_num
    return err

def calc_acc(pred, lab):
    err = calc_err(pred, lab)
    return 1.0 - err

def scores(root):

    preds = root + 'y_pred.csv'
    truths =  root + 'y_true.csv'

    df_preds = pd.read_csv(preds)
    df_truths = pd.read_csv(truths)
    
    columns = ['0','1','2','3','4','5']
    test_preds = df_preds[columns].values.tolist()
    test_truth = df_truths[columns].values.tolist()


    predictions, truths = [], []
    for i in range(len(test_preds)):
        x =np.argmax(test_preds[i])
        predictions.append(x)
    
    for i in range(len(test_truth)):
        x =np.argmax(test_truth[i])
        truths.append(x)
        
    
    test_preds = predictions
    test_truth = truths
    test_preds_i = np.array(test_preds)
    test_truth_i = np.array(test_truth)
    f1ma = f1_score(test_truth_i, test_preds_i, average='macro')
    f1mi = f1_score(test_truth_i, test_preds_i, average='micro')
    pre_ma = precision_score(test_truth_i, test_preds_i, average='macro')
    pre_mi = precision_score(test_truth_i, test_preds_i, average='micro')
    re_ma = recall_score(test_truth_i, test_preds_i, average='macro')
    re_mi = recall_score(test_truth_i, test_preds_i, average='micro')


    print('F1-Score Macro = {:5.3f}'.format(f1ma))
    print('F1-Score Micro = {:5.3f}'.format(f1mi))
    print('-------------------------')
    print('Precision Macro = {:5.3f}'.format(pre_ma))
    print('Precision Micro = {:5.3f}'.format(pre_mi))
    print('-------------------------')
    print('Recall Macro = {:5.3f}'.format(re_ma))
    print('Recall Micro = {:5.3f}'.format(re_mi))
    print('-------------------------')



