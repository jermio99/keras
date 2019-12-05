# keras11_mlp2.py 카피
# R2 0.5 이하로 만들기
# 레이어는 5개 이상 (히든만)
# 노드는 10개 이상
# epochs 는 100개 이상
# batch_size 는 1

from keras.models import Sequential
from keras.layers import Dense

#1. 데이터
import numpy as np

x = np.array([range(1,101), range(101,201)])
y = np.array([range(201,301)])
# print(x)

print(x.shape)  # (2, 100)

x = np.transpose(x) #행은 무시한다
y = np.transpose(y)

print(x.shape)

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

model.add(Dense(210, input_shape=(2, ), activation='relu'))
#컬럼이 하나라는것은 면밀이 따지면 틀린말
model.add(Dense(300))
model.add(Dense(450))
model.add(Dense(100))
model.add(Dense(80))
model.add(Dense(79))
model.add(Dense(81))
model.add(Dense(58))
model.add(Dense(34))
model.add(Dense(1))

model.summary()

#3. 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mse'])
model.fit(x_train,y_train, epochs=100, batch_size=1,
validation_data=(x_val, y_val))

#4. 평가예측
loss, acc = model.evaluate(x_test, y_test, batch_size=1)
print("acc :", acc)

# aaa = np.array([[101,102,103],[201,202,203]])  # 2, 3
# aaa = np.transpose(aaa)
# y_predict = model.predict(aaa)

y_predict = model.predict(x_test)
print(y_predict)


# RMSE 구하기
from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
print("RMSE:", RMSE(y_test, y_predict))

# R2 구하기
from sklearn.metrics import r2_score
r2_y_predict = r2_score(y_test, y_predict)
print("R2 : ", r2_y_predict)
