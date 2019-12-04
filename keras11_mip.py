from keras.models import Sequential
from keras.layers import Dense

#1. 데이터
import numpy as np

x = np.array([range(1,101), range(101,201)])
y = np.array([range(1,101), range(101,201)])
# print(x)

print(x.shape)

x = np.transpose(x)
y = np.transpose(y)

print(x.shape)

# x_train = x[:60]
# x_val = x[60:80]
# x_test = x[80:]
# y_train = y[:60]
# y_val = y[60:80]
# y_test = y[80:]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x,y, random_state=66, test_size=0.4, shuffle=False
)
x_val, x_test, y_val, y_test = train_test_split(
    x_test,y_test, random_state=66, test_size=0.5, shuffle=False
)  #6:2:2

#2. 모델구성
from keras.models import Sequential
from keras.layers import Dense
model = Sequential()

model.add(Dense(10, input_shape=(2, ), activation='relu'))
#컬럼이 하나라는것은 면밀이 따지면 틀린말
model.add(Dense(8))
model.add(Dense(5))
model.add(Dense(3))
model.add(Dense(2))
#input과 output이 2개

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
