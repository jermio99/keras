from keras.models import Sequential
from keras.layers import Dense

import numpy as np
x_train = np.array([1,2,3,4,5,6,7,8,9,10])
y_train = np.array([1,2,3,4,5,6,7,8,9,10])
x_test = np.array([11,12,13,14,15,16,17,18,19,20])
y_test = np.array([11,12,13,14,15,16,17,18,19,20])
# x_predict = np.array([21,22,23,24,25])

model = Sequential()
# model.add(Dense(500, input_dim=(1, activation='relu'))
model.add(Dense(100, input_shape=(1, ), activation='relu')) #input_dim=1, input_shape=(1, )컬럼이 하나만들어간다는말
model.add(Dense(50))
model.add(Dense(40))
model.add(Dense(30))
model.add(Dense(20))
model.add(Dense(10))
model.add(Dense(1))

model.summary()

#loss의 최소화를 위해서 optimizer이 있다
model.compile(loss='mse', optimizer='adam',
           # metrics=['accuracy'])
            metrics=['mse']) # metrics는 훈련할때 우리한테 보이는 공간
model.fit(x_train,y_train, epochs=100, batch_size=1)

loss, mse = model.evaluate(x_test, y_test) # a[0], a[1]
# evaluate를 반환하면 loss와 acc 값이 나온다
print("mse :", mse)   # 1.0 // metrics=['mse']로 바꿨을때 결과값: 6.02910432689896e-08
print("loss:", loss)  # 3.6167995176583645e-07 // 6.02910432689896e-08


y_predict = model.predict(x_test)
print(y_predict)

# rmse : mse에다가 root
# RMSE 구하기
from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict)) #sqrt는 root를 뜻함
print("RMSE:", RMSE(y_test, y_predict))

# R2구하기 (대략 1이면 좋고 0이면 안좋은거, 정확하지는 않음)
from sklearn.metrics import r2_score
r2_y_predict = r2_score(y_test, y_predict)
print("R2 : ", r2_y_predict)
#R2값이 올라가고 내려갈때 RMSE값다 올라가고 내려감
