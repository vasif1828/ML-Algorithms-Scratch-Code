import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import  datasets
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error


X,y =  datasets.make_regression(n_samples=100, n_features=1,noise=20, random_state=4)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

# fig = plt.figure(figsize = (8,6))
# plt.scatter(X[:,0], y, color = "b", marker = "o", s= 30)
# plt.show()

from Linear_Regression_Scratch import LinearRegression
regressor = LinearRegression(lr = 0.1)
regressor.fit(X_train,y_train)
predictions = regressor.predict(X_test)

mse = mean_squared_error(y_test, predictions)
print("MSE ifS lr = 0.1:", mse)

preds = regressor.predict(X)
fig = plt.figure(figsize=(8,5))
m1 = plt.scatter(X_train, y_train, color = "blue")
m2 = plt.scatter(X_test, y_test, color = "blue")
plt.plot(X, preds, color = "red", linewidth = 5, label = "prediction")
plt.show()