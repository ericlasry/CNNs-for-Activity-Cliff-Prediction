### Imports
import pandas as pd
import numpy as np
import os
from tqdm import tqdm

from  matplotlib import pyplot as plt
from PIL import Image


from tensorflow.keras.layers import Conv2D, Dense, MaxPooling2D, InputLayer, Flatten, Dropout
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam


def create_dataset(im_folder, im_height, im_width, value_df):
    """ Creates arrays of images, class labels and regression values from class folders.

    Parameters
    ----------
    im_folder: A string specifying the path of the data folder
    im_height: An int specifying height of images in the folders
    im_width: An int specifying width of images in the folders
    value_df: A pandas DataFrame containing pKi differences for MMPs

    Returns
    -------
    mmp_ixs: A dict of form MMP_ID:[ixs], where [ixs] is a list of indices corresponding to MMP #MMP_ID
    im_data_array: A numpy array containing MMP images
    class_names: A numpy array containing AC classification for MMPs
    reg_values: A numpy array containing pKi differences for MMPs
    """
    mmp_ixs = {}    
    im_data_array=[]
    class_names=[]
    reg_values=[]
    
    parsed_ids = {mmp:0 for mmp in range(15787)}     # Keeps track of parsed_ids (crucial for augmented)
    ix = 0                                                      # Tracks the indexing arrays
    for dir1 in sorted(os.listdir(im_folder)):                  # Loop through sub-folders
        for file in sorted(os.listdir(os.path.join(im_folder, dir1))):# Loop through image files
            mmp_id = int(file[4:9])                             # Extract MMP_ID from file names
            
            if parsed_ids[mmp_id] == 0:
                parsed_ids[mmp_id] = 1    
                mmp_ixs[mmp_id] = []
                                    
            image_path = os.path.join(im_folder, dir1,  file)
            image = np.array(Image.open(image_path))
            image = np.resize(image,(im_height, im_width,3))
            image = image.astype('float32')
            image /= 255 

            reg_value = (value_df.at[mmp_id,0] - value_df.at[mmp_id,1]).astype('float32') # Extract pKi difference values
            
            class_name = int(dir1)

            mmp_ixs[mmp_id].append(ix)
            im_data_array.append(image)
            class_names.append(class_name)
            reg_values.append(reg_value)

            ix += 1
    return mmp_ixs, np.array(im_data_array) , np.array(class_names), np.array(reg_values)

def get_split_mmp_indices(index_folder_path, set_size=100000):
    """ Gives 0/1/2-out index lists for MMPs.

    Parameters
    ----------
    index_folder_path: A string specifying the path of the folder containing index csvs
    set_size: An int specifying the max index to be returned

    Returns
    -------
    ind_zero_out: A list containing indices of zero-out MMPs
    ind_one_out: A list containing indices of one-out MMPs
    ind_two_out: A list containing indices of two-out MMPs
    """
    # Returns list of indices of zero-out / one-out / two-out MMPs.
    ind_zero_out = pd.read_csv(index_folder_path+"/ind_zero_out_mmps.csv", header=None)
    ind_zero_out = list(ind_zero_out[0])
    ind_zero_out = [int(x) for x in ind_zero_out if x < set_size]        

    ind_one_out = pd.read_csv(index_folder_path+"/ind_one_out_mmps.csv", header=None)
    ind_one_out = list(ind_one_out[0])
    ind_one_out = [int(x) for x in ind_one_out if x < set_size]   

    ind_two_out = pd.read_csv(index_folder_path+"/ind_two_out_mmps.csv", header=None)
    ind_two_out = list(ind_two_out[0])
    ind_two_out = [int(x) for x in ind_two_out if x < set_size] 

    return ind_zero_out, ind_one_out, ind_two_out


def save(data, labels, target_folder, set_size = 0, is_sorted = True):
    """ Saves a set of MMP images.

    Parameters
    ----------
    data: A list of images
    labels: A list of labels for images in data
    target_folder: A string specifying the folder to save images in
    set_size: An int specifying the number of images to save (saves all in data if set_size=0)
    is_sorted: A bool specifying whether to save images sorted by class folders

    Returns
    -------
    None
    """        
    if set_size==0: until = len(labels)
    else: until = set_size

    if is_sorted == True:
        for i in tqdm(range(until)):
            im = data[i]
            name = 'mmp_'+'{:05}'.format(i)+'.jpg'

            im = im.save(target_folder + "/" + str(labels[i]) + "/" + name)
    elif is_sorted == False:
        for i in tqdm(range(until)):
            im = data[i]
            name = 'mmp_'+'{:05}'.format(i)+'.jpg'

            im = im.save(target_folder+'/'+name)
    return None


### SCNN Models + evaluation plots.
class SCNN_Modeller:
    def __init__(self, train_X, train_y, test_X, test_y):
        self.train_X = train_X
        self.train_y = train_y
        self.test_X = test_X
        self.test_y = test_y
    
    def fit_model(self, num_epochs=20, lr=0.000001, eval_plots=True):
        model = Sequential([
            InputLayer(input_shape=(200, 400, 3)),
            Conv2D(filters=32, kernel_size=3, activation='relu'),
            MaxPooling2D((2,2)),
            Conv2D(filters=64, kernel_size=3, activation='relu'),
            MaxPooling2D((2,2)),
            Dropout(0.4),
            Flatten(),
            Dense(32, activation='linear'),
            Dense(2, activation='softmax')
            ])
        opt = Adam(learning_rate= lr)
        model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        history = model.fit(
            x = self.train_X,
            y = self.train_y,
            verbose = 2,
            validation_data = (self.test_X, self.test_y),
            epochs = num_epochs
            )

        if eval_plots:
            acc = history.history['accuracy']
            val_acc = history.history['val_accuracy']
            loss = history.history['loss']
            val_loss = history.history['val_loss']

            epochs_range = range(num_epochs)

            plt.figure(figsize=(15, 15))
            plt.subplot(2, 2, 1)
            plt.plot(epochs_range, acc, label='Training Accuracy')
            plt.plot(epochs_range, val_acc, label='Validation Accuracy')
            plt.legend(loc='lower right')
            plt.title('Training and Validation Accuracy')

            plt.subplot(2, 2, 2)
            plt.plot(epochs_range, loss, label='Training Loss')
            plt.plot(epochs_range, val_loss, label='Validation Loss')
            plt.legend(loc='upper right')
            plt.title('Training and Validation Loss')
            plt.show()
        return history


def hyperparam_eval_plots(history, num_epochs):
    """ Shows evaluation plots (accuracy, loss, precision) for a classification CNN.

    Parameters
    ----------
    history: A tensorflow.keras History object for the model
    num_epochs: An int specifying the number of epochs over which to plot

    Returns
    -------
    None
    """
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    prec = history.history['precision']
    val_prec = history.history['val_precision']

    epochs_range = range(num_epochs)

    plt.figure(figsize=(15, 15))
    plt.subplot(2, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(2, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()
    
    plt.subplot(2, 2, 3)
    plt.plot(epochs_range, prec, label='Training Precision')
    plt.plot(epochs_range, val_prec, label='Validation Precision')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Precision')
    plt.show()
    return None


def onehot(y):
    """ One-hot encodes a binary list.

    Parameters
    ----------
    y: A binary list to be one-hot encoded

    Returns
    -------
    hot_y: A list of [0,1],[1,0]s
    """
    hot_y = []
    for v in y:
        if v in [0,1]:
            hot_y.append([1-v, v])
        else:
            return "Error: Labels not binary."
    return hot_y


def reverse_onehot(hot_y):
    """ Reverses one-hot encoding of a binary list. Accepts Softmax input.

    Parameters
    ----------
    hot_y: A numpy array of two-lists

    Returns
    -------
    y: A binary numpy array
    """
    hot_list = [x for x in hot_y]
    class_list = []
    for v in hot_list:
        if v[0]>v[1]:
            class_list.append(0)
        else:
            class_list.append(1)
    return np.array(class_list)