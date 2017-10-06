from keras.layers import Dense, Activation
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential,Model
from first import X,y,X_train,X_test,y_train,y_test
from first import scoring_SN,X_scoring,X,y,write_to_csv
from sklearn.metrics import mean_squared_error
import numpy as np
#X_shape = X_train.shape
model = Sequential()
model.add(Dense(units=200, input_shape=(21,)))
model.add(Activation('linear'))
model.add(BatchNormalization())

model.add(Dense(units=100))
model.add(Activation('linear'))
model.add(BatchNormalization())

model.add(Dense(units=75))
model.add(Activation('linear'))
model.add(BatchNormalization())

model.add(Dense(units=50))
model.add(Activation('linear'))
model.add(BatchNormalization())

model.add(Dense(units=100))
model.add(Activation('linear'))
model.add(BatchNormalization())

model.add(Dense(units=200))
model.add(Activation('linear'))
model.add(BatchNormalization())

model.add(Dense(units=150))
model.add(Activation('linear'))
model.add(BatchNormalization())

model.add(Dense(units=50))
model.add(Activation('relu'))
model.add(BatchNormalization())

model.add((Dense(units=1)))


# For a mean squared error regression problem
model.compile(optimizer='adagrad',
              loss='mse')

# For custom metrics
import keras.backend as K

def mean_pred(y_true, y_pred):
    return K.mean(y_pred)

model.compile(optimizer='rmsprop',
              loss='mse',
              metrics=['mse'])

model.fit(np.array(X),np.array(y),epochs=600)

#RMSE:11.7721391065 #RMSE:12.2106810917


y_pred = model.predict(np.array(X_scoring))

#print ("RMSE:{}".format(np.sqrt(mean_squared_error(y_test,y_pred))))
#print type(y_pred)

np.savetxt("save.txt",y_pred)
y_pred2 = np.array(y_pred,dtype=float)

write_to_csv('csvs/neural_net.csv',y_pred2)