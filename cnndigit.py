from keras.layers import Input, Flatten, Dense, Conv2D, MaxPool2D
from keras.models import Model, Sequential
from keras.utils import to_categorical
from keras.datasets import mnist
from HodaDatasetReader import read_hoda_dataset

# FARSI Dataset
x_train_fa, y_train_fa = read_hoda_dataset('./samples-FA/Train 60000.cdb', images_height=32,
                                           images_width=32,
                                           one_hot=False,
                                           reshape=False)
x_test_fa, y_test_fa = read_hoda_dataset('./samples-FA/Test 20000.cdb', images_height=32,
                                         images_width=32,
                                         one_hot=False,
                                         reshape=False)

# ENGLISH Dataset
(x_train_en, y_train_en), (x_test_en, y_test_en) = mnist.load_data()

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
x_train_fa = x_train_fa.reshape(60000, 32, 32, 1)
x_test_fa = x_test_fa.reshape(20000, 32, 32, 1)
x_train_fa = x_train_fa.astype('float32') / 255
x_test_fa = x_test_fa.astype('float32') / 255
y_train_fa = to_categorical(y_train_fa, 10)
y_test_fa = to_categorical(y_test_fa, 10)

x_train_en = x_train_en.reshape(60000, 28, 28, 1)
x_test_en = x_test_en.reshape(10000, 28, 28, 1)
x_train_en = x_train_en.astype('float') / 255
x_test_en = x_test_en.astype('float') / 255
y_train_en = to_categorical(y_train_en, 10)
y_test_en = to_categorical(y_test_en, 10)

size = (28, 28, 1)


def build_model():
    inp = Input(shape=size)
    L1 = Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(size,))(inp)
    L2 = MaxPool2D(pool_size=(2, 2))(L1)
    L3 = Flatten()(L2)
    model = Model(inp, L3)
    model.compile(optimizer='sgd', loss='mse')
    return model


myModel = build_model()

en_model = Sequential()
en_model.add(myModel)
en_model.add(Dense(10, activation='softmax'))
en_model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

en_model.fit(x_train_en, y_train_en, batch_size=128, epochs=5, validation_split=0.2)
scores_en = en_model.evaluate(x_test_en, y_test_en)


# myModel.trainable = False

x_train_fa = x_train_fa[:, 2:30, 2:30, :]
x_test_fa = x_test_fa[:, 2:30, 2:30, :]

fa_model = Sequential()
#fa_model.add(MaxPool2D(pool_size=(1,1), padding='same'))
fa_model.add(myModel)
fa_model.add(Dense(10, activation='softmax'))
fa_model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

fa_model.fit(x_train_fa, y_train_fa, batch_size=128, epochs=5, validation_split=0.2)
scores_fa = fa_model.evaluate(x_test_fa, y_test_fa)