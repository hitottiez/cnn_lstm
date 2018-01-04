"""
python train.py ../../TSN/data/ucf101_rgb_train_split_1.txt ../../TSN/data/ucf101_rgb_val_split_1.txt
"""
from keras.models import Sequential
from keras.layers.wrappers import TimeDistributed
from keras.preprocessing import sequence
from keras.preprocessing.image import load_img, img_to_array
from keras.layers.core import Dense, Dropout, Flatten, Activation
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.recurrent import LSTM
from keras.layers.pooling import GlobalAveragePooling1D
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.utils import plot_model

import sys
import numpy as np
import matplotlib.pyplot as plt
import random

batch_size = 1
num_classes = 101
epochs = 10
frames = 16
train_num = 500 #the total number of training sequences for training

train_split_file = sys.argv[1]
val_split_file = sys.argv[2]

def load_data(split_file):
    X = []
    Y = []

    split_data = np.genfromtxt(split_file, dtype=None, delimiter=" ")
    total_train_num = len(split_data)

    # Use training data partially
    indices = random.sample(range(total_train_num), train_num)

    for i in indices: #for each sequence
        image_dir = split_data[i][0].decode("UTF-8")
        seq_len = int(split_data[i][1])
        y = int(split_data[i][2])

        seq = []
        for j in range(frames): #for each frame
            # get frames at regular interval. start from frame index 1
            frame = int(seq_len / frames * j) + 1
            image = load_img("%s/img_%05d.jpg" % (image_dir, frame))
            image = img_to_array(image)
            seq.append(image)
        X.append(np.array(seq))
        Y.append(y)

    X = np.array(X)
    #X = X.astype('float32')
    X /= 255

    return X, Y


# Load data
X_train, Y_train = load_data(train_split_file)
X_val, Y_val = load_data(val_split_file)
maxlen = max([len(x) for x in X_train])

print('X_train shape:', X_train.shape)
print('X_val shape:', X_val.shape)
print(X_train.shape[0], 'train samples')
print(X_val.shape[0], 'val samples')


# Model Construction
model=Sequential()

model.add(TimeDistributed(Conv2D(32, 3, 3, border_mode='same'), input_shape=X_train.shape[1:]))
model.add(TimeDistributed(Activation('relu')))
model.add(TimeDistributed(Conv2D(32, 3, 3)))
model.add(TimeDistributed(Activation('relu')))
model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2))))
model.add(TimeDistributed(Dropout(0.25)))

model.add(TimeDistributed(Flatten()))
model.add(TimeDistributed(Dense(512)))

model.add(TimeDistributed(Dense(35, name="first_dense" )))

model.add(LSTM(20, return_sequences=True, name="lstm_layer"));

model.add(TimeDistributed(Dense(1), name="time_distr_dense_one"))
model.add(GlobalAveragePooling1D(name="global_avg"))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# Train
history = model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs,
          verbose=1, validation_data=(X_val, Y_val))
score = model.evaluate(X_val, Y_val, verbose=0)
print('Validation loss:', score[0])
print('Validation accuracy:', score[1])
