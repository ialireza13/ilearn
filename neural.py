import keras as K
from keras import backend as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import numpy as np

def create_dataset(loc):
    dataset = np.loadtxt(loc,delimiter=',')
    Y = dataset[:,2]
    X = dataset[:,0:2]
    print(X.shape)
    return X, Y

def create_neuralNet(input_size=2):
    model = Sequential()
    model.add(Dense(16,input_dim=input_size,kernel_initializer='normal',activation='relu'))
    model.add(Dense(64, kernel_initializer='normal',activation='relu'))
    model.add(Dense(1024,kernel_initializer='normal'))
    model.add(Dense(64, kernel_initializer='normal'))
    model.add(Dense(1,kernel_initializer='normal'))
    model.compile(loss='squared_hinge', optimizer='sgd')
    return model

def main():
    loc = 'T.txt'
    X , Y = create_dataset(loc)
    input_dim = 2
    model = create_neuralNet(input_dim)
    #estimator = KerasRegressor(build_fn=model, epochs=100, batch_size=32, verbose=1)
    #kfold = KFold(n_splits=10)
    #results = cross_val_score(estimator, X, Y, cv=kfold)
    #print("Results: %.2f (%.2f) MSE" % (results.mean(), results.std()))
    model.fit(X,Y,validation_split=0.2,epochs=100,batch_size=10,verbose=1,shuffle=True)
main()