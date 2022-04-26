import numpy as np
import pandas as pd
import tensorflow as tf
from math import sqrt
import subprocess
import sys

from tensorflow.keras.layers import Conv2D, Dense, MaxPooling2D, InputLayer, Flatten, Dropout
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.optimizers import Adam

from  base_train import onehot, reverse_onehot, predict
from  eval import get_confusion_matrix, get_confusion_indices, get_metrics
  
    
# Train M3 aug
for m in ["M3w"]:
    for lr in [0.001]:
        for dropout in [0.7, 0.8, 0.9]:
            args = bytes(str(lr)+" "+str(dropout), "utf-8")
            subprocess.run(["python","/data/stat-cadd/kebl6129/notebooks/trainer"+m+".py"], input=args)        

for metric_set in ["test", "full"]:
    for m in ["M3w"]:
        for lr in [0.001]:
            for dropout in [0.7, 0.8, 0.9]:
                args = bytes(m+" "+str(lr)+" "+str(dropout)+" "+metric_set, "utf-8")
                subprocess.run(["python","/data/stat-cadd/kebl6129/notebooks/eval_subprocess.py"], input=args)    