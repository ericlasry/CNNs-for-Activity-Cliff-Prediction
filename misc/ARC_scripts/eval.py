import sys
import subprocess
import numpy as np
import pandas as pd
import tensorflow as tf
from math import sqrt
from tensorflow.keras.models import load_model

from  base_train import onehot, reverse_onehot, predict
from  eval import get_confusion_matrix, get_confusion_indices, get_metrics


for metric_set in ["test", "full"]:
    for m in ["M01", "M02", "M03", "M04"]:
        for lr in [0.0001, 0.001, 0.01]:
            for dropout in [0.5, 0.6, 0.7, 0.8, 0.9]:
                args = bytes(m+" "+str(lr)+" "+str(dropout)+" "+metric_set, "utf-8")
                subprocess.run(["python","/data/stat-cadd/kebl6129/notebooks/eval_subprocess.py"], input=args)    

"""
# Test:
args = bytes("M01"+" "+str(0.001)+" "+str(0.8)+" "+"test", "utf-8")
subprocess.run(["python","/data/stat-cadd/kebl6129/notebooks/eval_subprocess.py"], input=args)    

args = bytes("M01"+" "+str(0.0001)+" "+str(0.8)+" "+"test", "utf-8")
subprocess.run(["python","/data/stat-cadd/kebl6129/notebooks/eval_subprocess.py"], input=args)
"""