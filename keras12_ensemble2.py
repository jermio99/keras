from keras.models import Sequential
from keras.layers import Dense

#1. 데이터
import numpy as np

x1 = np.array([range(100), range(311,411), range(100)])
x2 = np.array([range(100, 200), range(311,411), range(100)])

y1 = np.array([range(501,601), range(711,811), range(100)])
y2 = np.array([range(501,601), range(711,811), range(100)])
y3 = np.array([range(401,501), range(211,311), range(100)])


x1 = np.transpose(x1)
x2 = np.transpose(x2)

y1 = np.transpose(y1)
y2 = np.transpose(y2)
y3 = np.transpose(y3)

# print(x1.shape) # (100, 3)
# print(x2.shape) # (100, 3)
# print(y1.shape) # (100, 3)
# print(y2.shape) # (100, 3)

from sklearn.model_selection import train_test_split
x1_train, x1_test, y1_train, y1_test = train_test_split(
    x1,y1, random_state=66, test_size=0.4, shuffle=False
)
x1_val, x1_test, y1_val, y1_test = train_test_split(
    x1_test,y1_test, random_state=66, test_size=0.5, shuffle=False
)


x2_train, x2_test, y2_train, y2_test = train_test_split(
    x2,y2, random_state=66, test_size=0.4, shuffle=False
)
x2_val, x2_test, y2_val, y2_test = train_test_split(
    x2_test,y2_test, random_state=66, test_size=0.5, shuffle=False
)

y3_train, y3_test = train_test_split(
    y3, random_state=66, test_size=0.4, shuffle=False
)
y3_val, y3_test= train_test_split(
   y3_test, random_state=66, test_size=0.5, shuffle=False
)

# print(x2_test.shape)  #(20, 3)


#2. 모델구성
from keras.models import Sequential, Model
from keras.layers import Dense, Input
# model = Sequential()

input1 = Input(shape=(3,))
dense1 = Dense(10, activation='relu')(input1)
dense2 = Dense(9)(dense1)
dense3 = Dense(8)(dense2)
dense4 = Dense(7)(dense3)
dense5 = Dense(6)(dense4)
dense6 = Dense(5)(dense5)
dense7 = Dense(4)(dense6)
dense8 = Dense(3)(dense7)
dense9 = Dense(2)(dense8)
middle1 = Dense(3)(dense9)


input2 = Input(shape=(3,))
xx = Dense(10, activation='relu')(input1)
xx = Dense(9)(xx)
xx = Dense(8)(xx)
xx = Dense(7)(xx)
xx = Dense(6)(xx)
xx = Dense(5)(xx)
xx = Dense(4)(xx)
xx = Dense(3)(xx)
xx = Dense(2)(xx)
middle2 = Dense(3)(xx)


from keras.layers.merge import concatenate
merge1 = concatenate([middle1, middle2])  #2개 이상의 input이면 리스트로 표현

output1 = Dense(30)(merge1)
output1 = Dense(13)(output1)
output1 = Dense(3)(output1)

output2 = Dense(15)(merge1)
output2 = Dense(32)(output2)
output2 = Dense(3)(output2)

output3 = Dense(15)(merge1)
output3 = Dense(32)(output3)
output3 = Dense(3)(output3)

model = Model(inputs = [input1,input2], outputs=[output1,output2,output3]) 
#시작은 input1 끝은 output1으로 하겠다는말
model.summary()




'''
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
