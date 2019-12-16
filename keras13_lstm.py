from numpy import array
from keras.models import Sequential
from keras.layers import Dense, LSTM

#1. 데이터
x = array([[1,2,3],[2,3,4],[3,4,5],[4,5,6]])
y = array([4,5,6,7])
print(x)
print("x.shape :", x.shape)  #(4,3)
print("y.shape :", y.shape)  #(4, )

'''
  x  y
 123 4
 234 5
 345 6
 456 7
'''

x = x.reshape((x.shape[0], x.shape[1], 1)) 

print(x)
print("x.shape: ", x.shape) #(4, 3, 1)

#2. 모델 구성
model = Sequential()
model.add(LSTM(36, activation='relu', input_shape=(3,1)))
model.add(Dense(30))
model.add(Dense(28))
model.add(Dense(25))
model.add(Dense(23))
model.add(Dense(20))
model.add(Dense(18))
model.add(Dense(15))
model.add(Dense(12))
model.add(Dense(8))
model.add(Dense(5))
model.add(Dense(1))
# model.summary()

#3. 실행
model.compile(optimizer='adam', loss='mse')
model.fit(x,y,epochs=100)

x_input = array([6,7,8])
x_input = x_input.reshape((1,3,1))

yhat = model.predict(x_input)
print(yhat)
