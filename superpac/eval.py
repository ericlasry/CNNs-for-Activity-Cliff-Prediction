import numpy as np
import pandas as pd
from math import sqrt
from tqdm import tqdm
import matplotlib.pyplot as plt
import tensorflow as tf


from  base import reverse_onehot, predict

def get_confusion_indices(preds,vals):
    mis_ac = []
    cor_ac = []
    mis_nac = []
    cor_nac = []

    for count, val in enumerate(vals):
        if (val == 0):
            if preds[count] == 0:
                cor_nac.append(count)
            elif preds[count] == 1:
                mis_nac.append(count)
        elif (val == 1):
            if preds[count] == 0:
                mis_ac.append(count)
            elif preds[count] == 1:
                cor_ac.append(count)
    return cor_nac, mis_nac, mis_ac, cor_ac

def get_confusion_matrix(preds, vals):
    cor_nac, mis_nac, mis_ac, cor_ac = get_confusion_indices(preds, vals)
    return np.array([[len(cor_nac), len(mis_nac)],[len(mis_ac),len(cor_ac)]])
            
def get_metrics(model, test_X, test_y):
    test_preds = predict(model, test_X)
    test_preds = reverse_onehot(test_preds)

    c = get_confusion_matrix(test_preds, test_y)
    
    accuracy = (c[0,0]+c[1,1])/(c[0,0]+c[0,1]+c[1,0]+c[1,1])## TP+TN rate ## COR / (COR+MIS)
    precision = c[1,1]/(c[1,1]+c[0,1]) ## TP / TP+FP ## (COR_AC/PRED_AC)
    recall = c[1,1]/(c[1,1]+c[1,0]) ## TP / TP+FN ## (COR_AC/AC)
    specificity = c[0,0]/(c[0,0]+c[0,1]) ## TN / TN+FP ## (COR_NAC/NAC)
    f1 = 2*precision*recall / (precision+recall) 
    
    mcc_denom = sqrt((c[1,1]+c[0,1])*(c[1,0]+c[0,0])*(c[0,1]+c[0,0])*(c[1,1]+c[1,0]))
    mcc = (c[1,1]*c[0,0]-c[1,0]*c[0,1])/mcc_denom
    return [accuracy, precision, recall, specificity, f1, mcc]



def get_ag_metrics(predictor, df_test):
    test_preds = predictor.predict(df_test)

    test_preds = list(test_preds)
    test_y = list(df_test['is_AC'])
    c = get_confusion_matrix(test_preds, test_y)
    
    accuracy = (c[0,0]+c[1,1])/(c[0,0]+c[0,1]+c[1,0]+c[1,1])## TP+TN rate ## COR / (COR+MIS)
    precision = c[1,1]/(c[1,1]+c[0,1]) ## TP / TP+FP ## (COR_AC/PRED_AC)
    recall = c[1,1]/(c[1,1]+c[1,0]) ## TP / TP+FN ## (COR_AC/AC)
    specificity = c[0,0]/(c[0,0]+c[0,1]) ## TN / TN+FP ## (COR_NAC/NAC)
    f1 = 2*precision*recall / (precision+recall) 
    
    mcc_denom = sqrt((c[1,1]+c[0,1])*(c[1,0]+c[0,0])*(c[0,1]+c[0,0])*(c[1,1]+c[1,0]))
    mcc = (c[1,1]*c[0,0]-c[1,0]*c[0,1])/mcc_denom
    return [accuracy, precision, recall, specificity, f1, mcc]

