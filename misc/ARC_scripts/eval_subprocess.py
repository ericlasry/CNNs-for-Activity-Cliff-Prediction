import sys
import subprocess
import numpy as np
import pandas as pd
import tensorflow as tf
from math import sqrt
from tensorflow.keras.models import load_model

from  base_train import onehot, reverse_onehot, predict
from  eval import get_confusion_matrix, get_confusion_indices, get_metrics


def main(args):
    # args is a string of the form "M01 learn_rate dropout"
    split = args.split()
    model_num = split[0]
    lr = split[1]
    dropout = split[2]
    fullness = split[3]
    
    model_name = model_num+"_"+lr+"_"+dropout
    
    test_X = np.load("../split_datasets/test_X_frag_200.npy")
    test_y = np.load("../split_datasets/test_y_frag_200.npy")
    
    # Choose which dataset to test on
    if fullness == "test":
        X_data = test_X
        y_data = test_y   
        label = ""
    elif fullness == "full":
        one_X = np.load("../split_datasets/one_out_X_frag_200.npy")
        one_y = np.load("../split_datasets/one_out_y_frag_200.npy")
        
        X_data = np.concatenate((test_X, one_X))
        y_data = np.concatenate((test_y, one_y))
        label = "_full"
  
    m = load_model("./models/"+model_num+"_"+lr+"_"+dropout)
    
    # If this is the first model of type model_num, we need to create the dataframe.
    try:
        metric_df = pd.read_csv("./metrics/metrics_"+model_num+label+".csv", index_col="Model")  
    except FileNotFoundError:
        metric_df = pd.DataFrame(columns=['Accuracy','Precision','Recall','Specificity','F1', 'MCC'])

    metrics = get_metrics(m, X_data, y_data)
    index_label = "LR:"+lr+", Dropout:"+dropout
    
    metric_df.loc[index_label] = metrics
    
    metric_df.to_csv("./metrics/metrics_"+model_num+label+".csv", index=True, index_label="Model")
    
    return None

if __name__ == '__main__':
    args = sys.stdin.read()
    main(args)