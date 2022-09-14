"""
Linear regression model based on medical costs
Uso de framework
Data from: https://www.kaggle.com/datasets/mirichoi0218/insurance
Author: Alejandro Castro Reus
"""


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression 
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error 


"""
SECOND DELIVERY
"""
print("")
print("SECOND DELIVERY")
print("")

df = pd.read_csv('./insurance.csv')
df = df[["bmi", "age", "charges"]]
df.rename(columns = {'bmi':'x1', 'age': 'x2', 'charges':'y'}, inplace = True)


#Linear regression with one independent variable
#Separation in train/test
x_train, x_test, y_train, y_test = train_test_split(df["x1"], df["y"], test_size= 0.2, random_state=0)

reg = LinearRegression().fit(x_train.values.reshape(-1, 1), y_train.values.reshape(-1, 1))
y_train_pred = reg.predict(x_train.values.reshape(-1,1))
y_test_pred = reg.predict(x_test.values.reshape(-1,1))

print(f"Coeffs: {reg.coef_}, Intercept: {reg.intercept_}")

mse_train = mean_absolute_error(y_train, y_train_pred)
mse_test = mean_absolute_error(y_test, y_test_pred)

print(f"mse_train: {mse_train}, mse_test: {mse_test}")

validation_score = cross_val_score(reg, x_train.values.reshape(-1,1), y_train.values.reshape(-1,1), cv=5)
print(f"Accuracy: {validation_score.mean()}, std: {validation_score.std()}")

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