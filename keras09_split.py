from keras.models import Sequential
from keras.layers import Dense

#1. 데이터
import numpy as np

x = np.array(range(1,101))
y = np.array(range(1,101))
print(x)

x_train = x[:60]
x_val = x[60:80]
x_test = x[80:]
y_train = y[:60]
y_val = y[60:80]
y_test = y[80:]

#2. 모델구성
from keras.models import Sequential
from keras.layers import Dense
model = Sequential()

model.add(Dense(10, input_shape=(1, ), activation='relu'))
model.add(Dense(8))
model.add(Dense(5))
model.add(Dense(3))
model.add(Dense(1))

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

'''
결과
mse : 1.3096723705530167e-10
[[80.999985]
 [82.      ]
 [82.99999 ]
 [83.99999 ]
 [84.99999 ]
 [85.999985]
 [86.99999 ]
 [87.99999 ]
 [88.99999 ]
 [89.999985]
 [90.999985]
 [91.999985]
 [93.      ]
 [93.999985]
 [94.99999 ]
 [95.999985]
 [96.999985]
 [97.99999 ]
 [98.99999 ]
 [99.999985]]
RMSE: 1.1444091796875e-05
R2 :  0.9999999999960612
'''
