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

# Load the data in the pandas dataframe
dp = pd.DataFrame(diabetes, columns=['age', 'sex', 'cp', 'tresbps', 'chol', 'fbs', 'restecg', 'thalach',
                                     'exang', 'oldpeak'])

"""
# Generate heatmap to understand the correlation between the variables
correlation_matrix = dp.corr().round(2)
sns.set(rc={'figure.figsize': (11.7, 8.27)})
sns.heatmap(data=correlation_matrix, annot=True)
"""

# The heatmap shows that the variable age has strong negative correlation with thalach
# Let's take age as X variable for simple linear regression
X_age = dp.age
y_thalach = dp.thalach

# Let's normalize the X and Y between -1 and 1
X_age = np.array(X_age).reshape(-1, 1)
y_thalach = np.array(y_thalach).reshape(-1, 1)

# Build the regression model
# Split the dataset 80% for training and 20% for validation
X_train_1, X_test_1, Y_train_1, Y_test_1 = train_test_split(X_age, y_thalach, test_size=0.2, random_state=5)

# Take linear regression
lr = linear_model.LinearRegression()

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
prediction_space = np.linspace(min(X_age), max(X_age)).reshape(-1, 1)
plt.scatter(X_age, y_thalach)
plt.plot(prediction_space, lr.predict(prediction_space), color='red', linewidth=3)
plt.ylabel("Thalach")
plt.xlabel("Age")
plt.title("Thalach vs Age")
plt.show()
