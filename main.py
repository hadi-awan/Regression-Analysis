# Library to load data in dataframes
import pandas as pd

# Library for arrays
import numpy as np

# Library for regression analysis
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, cross_val_score

# Data visualization library
from matplotlib import pyplot as plt
import seaborn as sns

# Load in the data into pandas
diabetes = pd.read_csv("Dataset heart-disease-dataset.csv")

# Logistic Regression
X_age = diabetes['age']
y_thalach = diabetes['thalach']

# Let's normalize the X and Y between -1 and 1
X_age = np.array(X_age).reshape(-1,1)
y_thalach = np.array(y_thalach).reshape(-1,1)

# Build the regression model
# Split the dataset 80% for training and 20% for validation
X_train_1, X_test_1, Y_train_1, Y_test_1 = train_test_split(X_age, y_thalach.ravel(), test_size=0.2, random_state=5)

# Take non-linear regression
lr = linear_model.LogisticRegression(max_iter=10000)

# Fit the data
lr.fit(X_train_1, Y_train_1)

# Get prediction for the test data
y_pred_1 = lr.predict(X_test_1)

# Measure the performance MSE.R-squared
mse = mean_squared_error(Y_test_1, y_pred_1)
r = round(lr.score(X_test_1, Y_test_1), 2)

print("The model performance")
print("-----------------------------")
print("MSE is {}".format(mse))
print("R-squared score is {}".format(r))
print("\n")

# Plot the regression graph
prediction_space = np.linspace(min(X_age), max(X_age)).reshape(-1,1)
plt.scatter(X_age, y_thalach)
plt.plot(prediction_space, lr.predict(prediction_space), color='red', linewidth=3)
plt.ylabel("Thalach")
plt.xlabel("Age")
plt.show()