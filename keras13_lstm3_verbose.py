from numpy import array
from keras.models import Sequential
from keras.layers import Dense, LSTM

#1. 데이터
x = array([[1,2,3],[2,3,4],[3,4,5],[4,5,6]])
y = array([4,5,6,7])
print(x)
print("x.shape :", x.shape)  #(4,3)
print("y.shape :", y.shape)  #(4, )

x = x.reshape((x.shape[0], x.shape[1], 1)) 

print(x)
print("x.shape: ", x.shape) #(4, 3, 1)

#2. 모델 구성
model = Sequential()
model.add(LSTM(36, activation='relu', input_shape=(3,1)))
model.add(Dense(10))
model.add(Dense(5))
model.add(Dense(1))
# model.summary()

#3. 실행
model.compile(optimizer='adam', loss='mse')
model.fit(x,y,epochs=1000, verbose=0)

x_input = array([6,7,8])
x_input = x_input.reshape((1,3,1))

yhat = model.predict(x_input, verbose=2)
print(yhat)
