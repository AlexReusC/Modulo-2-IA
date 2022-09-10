"""
Linear regression model based on medical costs
Data from: https://www.kaggle.com/datasets/mirichoi0218/insurance
Author: Alejandro Castro Reus
"""


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression 
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error 
import numpy as np

"""
FIRST DELIVERY
"""

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
print(b0, b1, loss)

#we make predictions with our best model
print(f'x: 1, expected y: {b0 + (b1*1)}')
print(f'x: 3, expected y: {b0 + (b1*3)}')
print(f'x: 5, expected y: {b0 + (b1*5)}')
print(f'x: 7, expected y: {b0 + (b1*7)}')
print(f'x: 9.5, expected y: {b0 + (b1*9)}')



"""
SECOND DELIVERY
"""

#Linear regression with one independent variable
#Separation in train/test
x_train, x_test, y_train, y_test = train_test_split(df["x1"], df["y"], test_size= 0.2, random_state=0)

reg = LinearRegression().fit(x_train.values.reshape(-1, 1), y_train.values.reshape(-1, 1))
print(reg.intercept_, reg.coef_)
y_train_pred = reg.predict(x_train.values.reshape(-1,1))
y_test_pred = reg.predict(x_test.values.reshape(-1,1))

print(f"Coeffs: {reg.coef_}, Intercept: {reg.intercept_}")

mse_train = mean_absolute_error(y_train, y_train_pred)
mse_test = mean_absolute_error(y_test, y_test_pred)

print(f"mse_train: {mse_train}, mse_test: {mse_test}")


#Plots of 
figure, axis = plt.subplots(2,2)

#Bias in models
axis[0, 0].scatter(x_train, y_train, alpha=0.5)
axis[0, 0].plot(x_train, y_train_pred, color="red")
axis[0, 0].set_title("Correlation (Train)")

axis[1, 0].scatter(x_test, y_test, alpha=0.5)
axis[1, 0].plot(x_test, y_test_pred, color="red")
axis[1, 0].set_title("Correlation (Test)")

#Variation in model
axis[0, 1].scatter(x_train, y_train, alpha=0.5)
axis[0, 1].scatter(x_train, y_train - mse_train, color="red", alpha=0.5)
axis[0, 1].set_title("Real and predicted data (Train)")

axis[1, 1].scatter(x_test, y_test, alpha=0.5)
axis[1, 1].scatter(x_test, y_test - mse_test, color="red", alpha=0.5)
axis[1, 1].set_title("Real and predicted data (Test)")

plt.show()

"""
We can see underfitting: low bias and high variance
"""


#Linear regression with two independent variables
x_train, x_test, y_train, y_test = train_test_split(df[["x1", "x2"]], df["y"], test_size= 0.2, random_state=0)
mult_reg = LinearRegression().fit(x_train, y_train)

y_mult_train_pred = mult_reg.predict(x_train)
y_mult_test_pred = mult_reg.predict(x_test)

mse_mult_train = mean_absolute_error(y_train, y_mult_train_pred)
mse_mult_test = mean_absolute_error(y_test, y_mult_test_pred)

print(f"mse_mult_train: {mse_mult_train}, mse_mult_test: {mse_mult_test}")