"""
Linear regression model based on medical costs
Análisis y mejora
Data from: https://www.kaggle.com/datasets/mirichoi0218/insurance
Author: Alejandro Castro Reus
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

"""
FIRST DELIVERY
"""
print("")
print("FIRST DELIVERY")
print("")

def calculate_loss(df, b0, b1):
	n = len(df.index)
	temp = ( df["y"] - ( b0 + b1*df["x1"] ) ) ** 2
	return (1/n)*temp.sum()

def calculate_db0(df, b0, b1):
	n = len(df.index)
	temp = df["y"] - ( b0 + ( b1 * df["x1"] ) )
	return (-2/n)*temp.sum()

def calculate_db1(df, b0, b1):
	n = len(df.index)
	temp = df["x1"] * ( df["y"] - ( b0 + b1*df["x1"] ) )
	return ( -2/n ) * temp.sum()

df = pd.read_csv('./insurance.csv')
df = df[["bmi", "age", "charges"]]
df.rename(columns = {'bmi':'x1', 'age': 'x2', 'charges':'y'}, inplace = True)

b0 = 0
b1 = 0
x = np.linspace(10, 50, 80)

epochs = 100
lr = 0.001

#we improve the model every epoch
for i in range(epochs):
	db0 =	calculate_db0(df, b0, b1)
	db1 = calculate_db1(df, b0, b1)

	b0 = b0 - (lr * db0)
	b1 = b1 - (lr * db1)

	loss = calculate_loss(df, b0, b1)
	y = b0 + (b1*x)
	

#final model
loss = calculate_loss(df, b0, b1)
print(f"b0: {b0}, b1: {b1}, loss: {loss}")

#we make predictions with our best model
print(f'x: 1, expected y: {b0 + (b1*1)}')
print(f'x: 3, expected y: {b0 + (b1*3)}')
print(f'x: 5, expected y: {b0 + (b1*5)}')
print(f'x: 7, expected y: {b0 + (b1*7)}')
print(f'x: 9.5, expected y: {b0 + (b1*9)}')

plt.scatter(df["x1"], df["y"])
plt.plot(x, y, color="red")
plt.show()