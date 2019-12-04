from keras.models import Sequential
from keras.layers import Dense

#1. 데이터
import numpy as np

x = np.array(range(1,101))
y = np.array(range(1,101))
print(x)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x,y, random_state=66, test_size=0.4, shuffle=False
)
x_val, x_test, y_val, y_test = train_test_split(
    x_test,y_test, random_state=66, test_size=0.5, shuffle=False
)  #6:2:2

#2. 모델구성
from keras.models import Sequential, Model
from keras.layers import Dense, Input
# model = Sequential()

'''
input1 = Input(shape=(1,))
dense1 = Dense(10, activation='relu')(input1)
dense2 = Dense(9)(dense1)
dense3 = Dense(8)(dense2)
dense4 = Dense(7)(dense3)
dense5 = Dense(6)(dense4)
dense6 = Dense(5)(dense5)
dense7 = Dense(4)(dense6)
dense8 = Dense(3)(dense7)
dense9 = Dense(2)(dense8)
output1 = Dense(1)(dense9)
#앞라인에 있던것이 뒷라인으로
'''

input1 = Input(shape=(1,))
xx = Dense(10, activation='relu')(input1)
xx = Dense(9)(xx)
xx = Dense(8)(xx)
xx = Dense(7)(xx)
xx = Dense(6)(xx)
xx = Dense(5)(xx)
xx = Dense(4)(xx)
xx = Dense(3)(xx)
xx = Dense(2)(xx)
output1 = Dense(1)(xx)

model = Model(inputs = input1, outputs=output1) 
#시작은 input1 끝은 output1으로 하겠다는말
model.summary()


#3. 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mse'])
model.fit(x_train,y_train, epochs=100, batch_size=1,
validation_data=(x_val, y_val))

#4. 평가예측
loss, mse = model.evaluate(x_test, y_test)
print("mse :", mse)

y_predict = model.predict(x_test)
print(y_predict)

# rmse : mse에다가 root
# RMSE 구하기
from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
print("RMSE:", RMSE(y_test, y_predict))

from sklearn.metrics import r2_score
r2_y_predict = r2_score(y_test, y_predict)
print("R2 : ", r2_y_predict)


#레이어를 10개 이상늘리기
'''
mse : 7.858034362095978e-11
[[ 81.00001 ]
 [ 82.00001 ]
 [ 82.99999 ]
 [ 84.00001 ]
 [ 85.00001 ]
 [ 85.999985]
 [ 87.00001 ]
 [ 88.      ]
 [ 89.00001 ]
 [ 90.000015]
 [ 91.000015]
 [ 92.      ]
 [ 93.00001 ]
 [ 94.000015]
 [ 95.      ]
 [ 96.      ]
 [ 97.00001 ]
 [ 97.99999 ]
 [ 99.      ]
 [100.00001 ]]
RMSE: 8.864555388353158e-06
R2 :  0.9999999999976367
'''
