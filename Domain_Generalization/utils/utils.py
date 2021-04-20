import os
import math

from sklearn import metrics
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
from torchvision import transforms

class AUCMeter(object):
    """Computes and stores AUC"""
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.y_true = []
        self.y_score = []
        #self.image_ids = []
        
    def update(self, y_true, y_score):
        self.y_true.append(y_true)
        self.y_score.append(y_score)
        #self.image_ids.append(image_id)
        
    def calculate(self):
        y_true = torch.cat(self.y_true)
        y_score = torch.cat(self.y_score)
        auc = metrics.roc_auc_score(y_true!=4, 1.-y_score[:,4])
        fpr, tpr, thresholds = metrics.roc_curve(y_true!=4, 1.-y_score[:,4])
        fpr_980 = fpr[np.where(tpr>=0.98)[0][0]]
        fpr_991 = fpr[np.where(tpr>=0.991)[0][0]]
        fpr_993 = fpr[np.where(tpr>=0.993)[0][0]]
        fpr_995 = fpr[np.where(tpr>=0.995)[0][0]]
        fpr_997 = fpr[np.where(tpr>=0.997)[0][0]]
        fpr_999 = fpr[np.where(tpr>=0.999)[0][0]]
        fpr_1 = fpr[np.where(tpr==1.)[0][0]]
        
        return auc, fpr_980, fpr_991, fpr_993, fpr_995, fpr_997, fpr_999, fpr_1, thresholds