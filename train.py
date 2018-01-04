"""
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
from keras.utils import plot_model, np_utils

import sys
import numpy as np
import matplotlib.pyplot as plt
import random
import os

batch_size = 1
num_classes = 101
epochs = 10
train_num = 10 # The total number of training sequences for training
frames = 5 # The number of frames for each sequence


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
    X = X.astype('float32')
    X /= 255
    Y = np_utils.to_categorical(Y, num_classes)

    return X, Y


def build_model():
    model=Sequential()

    model.add(TimeDistributed(Conv2D(32, (3, 3), border_mode='same'), input_shape=X_train.shape[1:]))
    model.add(TimeDistributed(Activation('relu')))
    model.add(TimeDistributed(Conv2D(32, (3, 3))))
    model.add(TimeDistributed(Activation('relu')))
    model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2))))
    model.add(TimeDistributed(Dropout(0.25)))

    model.add(TimeDistributed(Flatten()))
    model.add(TimeDistributed(Dense(512)))

    model.add(TimeDistributed(Dense(35, name="first_dense" )))

    model.add(LSTM(20, return_sequences=True, name="lstm_layer"));

    model.add(TimeDistributed(Dense(num_classes), name="time_distr_dense_one"))
    model.add(GlobalAveragePooling1D(name="global_avg"))

    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])
    plot_model(model, to_file='model/cnn_lstm.png')

    return model


def plot_history(history):
    # Plot the history of accuracy
    plt.plot(history.history['acc'],"o-",label="accuracy")
    plt.plot(history.history['val_acc'],"o-",label="val_acc")
    plt.title('model accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend(loc="lower right")
    plt.savefig("model/model_accuracy.png")

    # Plot the history of loss
    plt.plot(history.history['loss'],"o-",label="loss",)
    plt.plot(history.history['val_loss'],"o-",label="val_loss")
    plt.title('model loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(loc='lower right')
    plt.savefig("model/model_loss.png")


if __name__ == "__main__":
    # Load arguments
    train_split_file = sys.argv[1]
    test_split_file = sys.argv[2]

    # Make directory
    if not os.path.exists("model"):
        os.makedirs("model")

    # Load data
    X_train, Y_train = load_data(train_split_file)
    X_test, Y_test = load_data(test_split_file)
    maxlen = max([len(x) for x in X_train])

    print("Loaded data")
    print('X_train shape:', X_train.shape)
    print('X_test shape:', X_test.shape)
    print(X_train.shape[0], 'train samples')
    print(X_test.shape[0], 'test samples')

    # Build model
    model = build_model()
    model.summary() 
    print("Built model")

    # Train model
    history = model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs,
              verbose=1, validation_data=(X_test, Y_test))
    plot_history(history)
    print("Trained model")

    # Save model and weights
    json_string = model.to_json()
    open('model/cnn_lstm.json', 'w').write(json_string)
    model.save_weights('model/cnn_lstm.hdf5')
    print("Saved model")

    # Evaluate model
    score = model.evaluate(X_test, Y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    # Post processing
    import gc
    gc.collect()
