import csv
import os
import numpy as np
from scipy import ndimage
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, MaxPooling2D, Cropping2D, Dropout
from keras.layers.convolutional import Convolution2D


path_prefix = '/home/markus/Udacity/data/own_bagfiles/images'

samples = []
with open('/home/markus/Udacity/data/own_bagfiles/images/labels.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        # Try to load each image before training, to make sure, they are all valid.
        # We do it here, to avoid slowing down the training.
        image_valid = True
        img_path = os.path.join(path_prefix, line[0])
        if True == os.path.exists(img_path):
            img = ndimage.imread(img_path)
            if (16,16,3) == img.shape:
                samples.append((img_path, line[1]))

print("Overall samples: {}".format(len(samples)))

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1:
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            X_train, y_train = loadTrainingData(batch_samples)

            yield shuffle(X_train, y_train)

def loadTrainingData(batch_samples):
    images = []
    labels = []
    for batch_sample in batch_samples:
        images.append(ndimage.imread(batch_sample[0]))
        labels.append(int(batch_sample[1]))

    X_train = np.array(images)
    y_train = np.array(labels)
    y_train = np_utils.to_categorical(y_train, num_classes=3)
    return X_train, y_train


dropout_rate = 0.5
epochs = 10

model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(16,16,3)))
model.add(Convolution2D(6,(5,5),activation='relu'))
model.add(MaxPooling2D())
model.add(Dropout(dropout_rate))
model.add(Convolution2D(6,(5,5),activation='relu'))
model.add(MaxPooling2D())
model.add(Dropout(dropout_rate))
model.add(Flatten())
model.add(Dense(120, activation='relu'))
model.add(Dropout(dropout_rate))
model.add(Dense(84, activation='relu'))
model.add(Dropout(dropout_rate))
model.add(Dense(3, activation='softmax'))

model.summary()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# True: load whole dataset at once
# False: use generator
use_whole_dataset_at_once = False
if use_whole_dataset_at_once:
    # This section allows to train the model by loading the whole
    # dataset into memory
    print("Using whole training set at once.")
    X_train, y_train = loadTrainingData(samples)
    print("Loaded {} data samples.".format(X_train.shape[0]))
    model.fit(X_train, y_train, batch_size=128, validation_split=0.2, shuffle=True, epochs=epochs)
else:
    # This section uses a generator to provide the data to the model
    print("Using generator")
    train_samples, validation_samples = train_test_split(samples, test_size=0.2)
    train_generator = generator(train_samples, batch_size=16)
    validation_generator = generator(validation_samples, batch_size=16)
    model.fit_generator(train_generator, \
                        steps_per_epoch=len(train_samples), \
                        validation_data=validation_generator, \
                        validation_steps = len(validation_samples), \
                        epochs=epochs)

model.save('model.h5')
print("Model saved!")