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
from keras.layers import Merge
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.utils import plot_model, np_utils

import sys
import numpy as np
import matplotlib.pyplot as plt
import random
import os
import argparse

batch_size = 8
num_classes = 101
epochs = 1
frames = 5 # The number of frames for each sequence


def build_rgb_model():
    model=Sequential()

    model.add(TimeDistributed(Conv2D(32, (3, 3), padding='same'), input_shape=(5, 224, 224, 3)))
    model.add(TimeDistributed(Activation('relu')))
    model.add(TimeDistributed(Conv2D(32, (3, 3))))
    model.add(TimeDistributed(Activation('relu')))
    model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2))))
    model.add(TimeDistributed(Dropout(0.25)))

    model.add(TimeDistributed(Flatten()))
    model.add(TimeDistributed(Dense(512)))

    model.add(TimeDistributed(Dense(35, name="first_dense_rgb" )))

    model.add(LSTM(20, return_sequences=True, name="lstm_layer_rgb"));

    model.add(TimeDistributed(Dense(num_classes), name="time_distr_dense_one_rgb"))
    model.add(GlobalAveragePooling1D(name="global_avg_rgb"))

    return model


def build_flow_model():
    model=Sequential()

    model.add(TimeDistributed(Conv2D(32, (3, 3), padding='same'), input_shape=(5, 224, 224, 3)))
    model.add(TimeDistributed(Activation('relu')))
    model.add(TimeDistributed(Conv2D(32, (3, 3))))
    model.add(TimeDistributed(Activation('relu')))
    model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2))))
    model.add(TimeDistributed(Dropout(0.25)))

    model.add(TimeDistributed(Flatten()))
    model.add(TimeDistributed(Dense(512)))

    model.add(TimeDistributed(Dense(35, name="first_dense_flow" )))

    model.add(LSTM(20, return_sequences=True, name="lstm_layer_flow"));

    model.add(TimeDistributed(Dense(num_classes), name="time_distr_dense_one_flow"))
    model.add(GlobalAveragePooling1D(name="global_avg_flow"))

    return model


def build_model():
    rgb_model = build_rgb_model()
    flow_model = build_flow_model()

    model = Sequential()
    model.add(Merge([rgb_model, flow_model], mode='ave'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])
    plot_model(model, to_file='model/cnn_lstm.png')

    return model


def batch_iter(split_file):
    split_data = np.genfromtxt(split_file, dtype=None, delimiter=" ")
    total_seq_num = len(split_data)
    num_batches_per_epoch = int((total_seq_num - 1) / batch_size) + 1

    def data_generator():
        while 1:
            indices = np.random.permutation(np.arange(total_seq_num))

            for batch_num in range(num_batches_per_epoch): # for each batch
                start_index = batch_num * batch_size
                end_index = min((batch_num + 1) * batch_size, total_seq_num)

                X = []
                Y = []
                for i in range(start_index, end_index): # for each sequence
                    image_dir = split_data[indices[i]][0].decode("UTF-8")
                    seq_len = int(split_data[indices[i]][1])
                    y = int(split_data[indices[i]][2])

                    seq = []
                    for j in range(frames): # for each frame
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

                yield (X, Y)

    return num_batches_per_epoch, data_generator()


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
    # Parse arguments
    parser = argparse.ArgumentParser(description="action recognition by cnn and lstm.")
    parser.add_argument("--split_dir", type=str, default='split')
    parser.add_argument("--dataset", type=str, default='ucf101')
    parser.add_argument("--rgb", type=int, default=1)
    parser.add_argument("--flow", type=int, default=1)
    parser.add_argument("--split", type=int, default=1)
    args = parser.parse_args()

    split_dir = args.split_dir
    dataset = args.dataset
    rgb = args.rgb
    flow = args.flow
    split = args.split

    # Make split file path
    rgb_train_split_file = "%s/%s_rgb_train_split_%s" % (split_dir, dataset, split)
    rgb_test_split_file = "%s/%s_rgb_val_split_%s" % (split_dir, dataset, split)
    flow_train_split_file = "%s/%s_flow_train_split_%s" % (split_dir, dataset, split)
    flow_test_split_file = "%s/%s_flow_val_split_%s" % (split_dir, dataset, split)

    # Make directory
    if not os.path.exists("model"):
        os.makedirs("model")

    # Build model
    model = build_model()
    model.summary()
    print("Built model")
    exit()

    # Make batches
    train_steps, train_batches = batch_iter(train_split_file)
    valid_steps, valid_batches = batch_iter(test_split_file)

    # Train model
    history = model.fit_generator(train_batches, steps_per_epoch=train_steps,
                epochs=epochs, verbose=1, validation_data=valid_batches,
                validation_steps=valid_steps)
    plot_history(history)
    print("Trained model")

    # Save model and weights
    json_string = model.to_json()
    open('model/cnn_lstm.json', 'w').write(json_string)
    model.save_weights('model/cnn_lstm.hdf5')
    print("Saved model")

    # Evaluate model
    score = model.evaluate_generator(valid_batches, valid_steps)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    # Clear session
    from keras.backend import tensorflow_backend as backend
    backend.clear_session()
