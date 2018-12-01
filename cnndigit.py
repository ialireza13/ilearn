import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
from HodaDatasetReader import read_hoda_dataset
import matplotlib.pyplot as plt

K.set_image_dim_ordering('th')

# FARSI Dataset
size = 32
X_train, y_train = read_hoda_dataset('./samples-FA/Train 60000.cdb', images_height=size,
                                     images_width=size,
                                     one_hot=False,
                                     reshape=False)
X_test, y_test = read_hoda_dataset('./samples-FA/Test 20000.cdb', images_height=size,
                                   images_width=size,
                                   one_hot=False,
                                   reshape=False)
# ENGLISH Dataset
# size=28
# (X_train, y_train), (X_test, y_test) = mnist.load_data()
'''
plt.subplot(221)
plt.imshow(X_train[100], cmap=plt.get_cmap('gray'))
plt.subplot(222)
plt.imshow(X_train[205], cmap=plt.get_cmap('gray'))
plt.subplot(223)
plt.imshow(X_train[405], cmap=plt.get_cmap('gray'))
plt.subplot(224)
plt.imshow(X_train[10000], cmap=plt.get_cmap('gray'))
# show the plot
plt.show()
'''
seed = 7
np.random.seed(seed)
X_train = X_train.reshape(X_train.shape[0], 1, size, size).astype('float32')
X_test = X_test.reshape(X_test.shape[0], 1, size, size).astype('float32')

X_train = X_train / 255
X_test = X_test / 255
# one hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]


def baseline_model():
    model = Sequential()
    model.add(Conv2D(32, (5, 5), input_shape=(1, size, size), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


# build the model
model = baseline_model()
# Fit the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=128, verbose=1)
# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("CNN Error: %.2f%%" % (100 - scores[1] * 100))
