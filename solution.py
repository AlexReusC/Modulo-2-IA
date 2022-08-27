import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def calculate_loss(df, b0, b1):
	n = len(df.index)
	temp = ( df["y"] - ( b0 + b1*df["x"] ) ) ** 2
	return (1/n)*temp.sum()

def calculate_db0(df, b0, b1):
	n = len(df.index)
	temp = df["y"] - ( b0 + ( b1 * df["x"] ) )
	return (-2/n)*temp.sum()

def calculate_db1(df, b0, b1):
	n = len(df.index)
	temp = df["x"] * ( df["y"] - ( b0 + b1*df["x"] ) )
	return ( -2/n ) * temp.sum()

df = pd.read_csv('./score.csv')
df.rename(columns = {'Hours':'x', 'Scores':'y'}, inplace = True)

b0 = 0
b1 = 0
x = np.linspace(0, 10, 20)

epochs = 100
lr = 0.001

for i in range(epochs):
	db0 =	calculate_db0(df, b0, b1)
	db1 = calculate_db1(df, b0, b1)

	b0 = b0 - (lr * db0)
	b1 = b1 - (lr * db1)

	loss = calculate_loss(df, b0, b1)

	print(f"b0: {b0}, b1: {b1}, loss: {loss}")
	
	y = b0 + (b1*x)
	plt.clf()
	plt.scatter(df["x"], df["y"])
	plt.plot(x, y)
	plt.pause(0.001)

plt.show()

print(f'x: 1, expected y: {b0 + (b1*1)}')
print(f'x: 3, expected y: {b0 + (b1*3)}')
print(f'x: 5, expected y: {b0 + (b1*5)}')
print(f'x: 7, expected y: {b0 + (b1*7)}')
print(f'x: 9.5, expected y: {b0 + (b1*9)}')

