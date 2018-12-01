import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from keras.models import model_from_json
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler

'''X=np.linspace(-10,10,num=10000)
Y=X*X
X=X.reshape(-1,1)
Y=Y.reshape(-1,1)'''

dataset = np.loadtxt('T', delimiter=',')
# np.random.shuffle(dataset)
qq = 9191
Y_ = dataset[0:qq, 2]
X1_ = dataset[0:qq, 0]
X2_ = dataset[0:qq, 1]
X1 = X1_.reshape(-1, 1)
X2 = X2_.reshape(-1, 1)
Y = Y_.reshape(-1, 1)


# create model
def base_model():
    model = Sequential()
    model.add(Dense(600, input_dim=2, kernel_initializer='normal', activation='relu'))
    model.add(Dense(300, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))
    model.compile(loss='mean_squared_error', optimizer='sgd')
    return model


scale_x1 = StandardScaler()
scale_x2 = StandardScaler()
scale_y = StandardScaler()
X_train1 = scale_x1.fit_transform(X1)
X_train2 = scale_x2.fit_transform(X2)
X_train = np.concatenate((X_train1, X_train2), axis=1)
Y_train = scale_y.fit_transform(Y)
# Compile model
model = base_model()
model.fit(X_train, Y_train, validation_split=0.2, epochs=200, batch_size=10, verbose=2, shuffle=True)
# plt.scatter(scale_x1.inverse_transform(X_train[:,0]),scale_x2.inverse_transform(X_train[:,1]),c=scale_y.inverse_transform(model.predict(X_train)),s=2)
fig1 = pyplot.figure()
ax = Axes3D(fig1)
ax.scatter(X1_, X2_, Y_, s=3)
ax.scatter(scale_x1.inverse_transform(X_train1), scale_x2.inverse_transform(X_train2),
           scale_y.inverse_transform(model.predict(X_train)), s=1)
'''
q1=np.ndarray(shape=1)
q2=np.ndarray(shape=1)
q1[0]=1.0025
q2[0]=35.2
q1 = q1.reshape(-1, 1)
q2 = q2.reshape(-1, 1)
q1=scale_x1.transform(q1)
q2=scale_x2.transform(q2)
q=np.concatenate((q1,q2),axis=1)
ax.scatter(scale_x1.inverse_transform(q1),scale_x2.inverse_transform(q2),
           scale_y.inverse_transform(model.predict(q)),s=5)
'''
# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")

# later...
'''
# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")

# evaluate loaded model on test data
loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
score = loaded_model.evaluate(X, Y, verbose=0)
'''
