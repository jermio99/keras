from keras.models import Sequential
from keras.layers import Dense

import numpy as np
x_train = np.array([1,2,3,4,5,6,7,8,9,10])
y_train = np.array([1,2,3,4,5,6,7,8,9,10])
x_test = np.array([11,12,13,14,15,16,17,18,19,20])
y_test = np.array([11,12,13,14,15,16,17,18,19,20])
x_predict = np.array([21,22,23,24,25])

model = Sequential()
# model.add(Dense(500, input_dim=(1, activation='relu'))
model.add(Dense(500, input_shape=(1, ), activation='relu')) #input_dim=1, input_shape=(1, )컬럼이 하나만들어간다는말
model.add(Dense(30))
model.add(Dense(30))
model.add(Dense(1))

# model.summary()


model.compile(loss='mse', optimizer='adam',
            metrics=['mse'])

from keras.callbacks import EarlyStopping

early_stopping = EarlyStopping(monitor='val_loss', patience=1000, mode='auto')

model.fit(x_train,y_train, epochs=100, batch_size=1)


loss, acc = model.evaluate(x_test, y_test)
print("acc :", acc)
print("loss:", loss)

y_predict = model.predict(x_predict)
print(y_predict)












keras15_lstm_dnn.py
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM

#1. 데이터
a = np.array(range(1,11))

size = 5
def split_x(seq, size):
    aaa = []
    for i in range(len(seq) - size + 1):
        subset = seq[i:(i+size)]
        aaa.append([item for item in subset])
    print(type(aaa))
    return np.array(aaa)

dataset = split_x(a,size)
print("=====================")
print(dataset)

x_train = dataset[:,0:-1]
y_train = dataset[:, -1]

print(x_train.shape) #{6,4)
print(y_train.shape) #(6,)

model = Sequential()
model.add(Dense(333, input_shape=(4, )))  #input_dim=4
model.add(Dense(1))
# model.summary()

model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=1)

# 모델을 마무리해서 완성하시오. fit까지

x2 = np.array([7,8,9,10]) #(4, ) -> (1,4)
x2 = x2.reshape((1,4))  #(1,4)

y_pred = model.predict(x2)
print(y_pred)

# y_pred를 구하시오.
