import numpy as np
import pandas as pd
from sklearn import linear_model
import matplotlib.pyplot as plt


dataframe = pd.read_csv('Advertising.csv')
X = dataframe.values[:, 2]
y = dataframe.values[:, 4]

plt.plot(X,y, "go")
plt.title("Dataset")
plt.xlabel("X_value")
plt.ylabel("y_value")
#plt.show()


def predict(new_radio, w1, w0):
	return new_radio*w1 + w0


def loss_function(X, y, w1, w0):
	n = len(X)
	sum_loss = 0
	for i in range(n):
		sum_loss += (y[i] - (w1*X[i] + w0))**2
	return sum_loss/n


def update(X, y, w1, w0, learning_rate):
	n = len(X)
	w0_temp = 0.0
	w1_temp = 0.0
	for i in range(n):
		w1_temp += -2*X[i]*(y[i] - (w1*X[i] + w0))
		w0_temp += -2*(y[i]-(w1*X[i] + w0))
	w1 -= (w1_temp/n) * learning_rate
	w0 -= (w0_temp/n) * learning_rate

	return w1,w0

def train(X, y, w1, w0, learning_rate, epochs):
	for i in range(epochs):
		w1,w0 = update(X, y, w1, w0, learning_rate)
		loss_value = loss_function(X, y, w1, w0)
		print('epoch={} loss={}'.format(i,loss_value))

	return w1, w0


w1 = np.random.rand()
w0 = np.random.rand()
learning_rate = 0.001

print('w1 = ',w1,' w0 = ',w0)
print('Train model')

w1,w0 = train(X,y,w1,w0, learning_rate, 100)
print('w1=', w1, 'w0= ', w0)
print('Find out model with Y = {} + {} * X'.format(w0, w1))

plt.plot(X,y, "go")
plt.title("Trained model")
plt.xlabel("X_value")
plt.ylabel("y_value")
x = np.linspace(0, 50, num=1000)
plt.plot(x, w0 + w1*x, color='yellow')
plt.show()
