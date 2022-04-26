import sys
import numpy as np
import tensorflow as tf

from tensorflow.keras.layers import Conv2D, Dense, MaxPooling2D, InputLayer, Flatten, Dropout
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam

from  base_train import onehot

def main(args):
    split = args.split()
    lr = float(split[0])
    dropout = float(split[1])
                 
    train_X = np.load("../split_datasets/train_X_frag_200.npy")
    train_y = np.load("../split_datasets/train_y_frag_200.npy")
    test_X = np.load("../split_datasets/test_X_frag_200.npy")
    test_y = np.load("../split_datasets/test_y_frag_200.npy")

    train_y, test_y = np.array(onehot(train_y)), np.array(onehot(test_y))

    num_epochs=40
    IM_HEIGHT=200
    IM_WIDTH=600

    model = Sequential([
                InputLayer(input_shape=(IM_HEIGHT, IM_WIDTH, 3)),
        
                Conv2D(filters=32, kernel_size=3, activation='relu'),
                Conv2D(filters=32, kernel_size=3, activation='relu'),
                Conv2D(filters=32, kernel_size=3, activation='relu'),
                Conv2D(filters=32, kernel_size=3, activation='relu'),
                Conv2D(filters=32, kernel_size=3, activation='relu'),
        
                Dropout(dropout),
                Flatten(),
        
                Dense(32, activation='linear'),
                Dense(2, activation='sigmoid')
                ])
    opt = Adam(learning_rate=lr)
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])

    history = model.fit(
        x = train_X,
        y = train_y,
        verbose = 2,
        validation_data=(test_X, test_y),
        epochs = num_epochs
        )
    model.save('./models/M04_'+str(lr)+'_'+str(dropout))
    return None

if __name__ == '__main__':
    args = sys.stdin.read()
    main(args)