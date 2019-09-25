# Multiple Linear Regression

# Importing the libraries
import numpy as np
import statsmodels.formula.api as smf
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

# Encoding categorical data
from sklearn.preprocessing import OneHotEncoder#, LabelEncoder
from sklearn.compose import make_column_transformer
# labelencoder = LabelEncoder()
# X[:, 3] = labelencoder.fit_transform(X[:, 3])
# onehotencoder = OneHotEncoder(categorical_features = [3])
# X = onehotencoder.fit_transform(X).toarray()
preprocessor_X = make_column_transformer((OneHotEncoder(), [3]), remainder="passthrough")
X = preprocessor_X.fit_transform(X)

# Avoiding the Dummy Variable Trap
X = X[:, 1:]

#Adding x_0 term to dataset
X = np.append(arr = np.ones((50, 1)).astype(int), values = X, axis = 1)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

X_opt = X_train[:, [0, 1, 2, 3, 4, 5]]
regressor_OLS = smf.OLS(endog = y_train, exog = X_opt.astype(np.float64)).fit()
regressor_OLS.summary()

X_opt = X_train[:, [0, 1, 3, 4, 5]]
regressor_OLS = smf.OLS(endog = y_train, exog = X_opt.astype(np.float64)).fit()
regressor_OLS.summary()

X_opt = X_train[:, [0, 3, 4, 5]]
regressor_OLS = smf.OLS(endog = y_train, exog = X_opt.astype(np.float64)).fit()
regressor_OLS.summary()

X_opt = X_train[:, [0, 3, 5]]
regressor_OLS = smf.OLS(endog = y_train, exog = X_opt.astype(np.float64)).fit()
regressor_OLS.summary()

X_opt = X_train[:, [0, 3]]
regressor_OLS = smf.OLS(endog = y_train, exog = X_opt.astype(np.float64)).fit()
regressor_OLS.summary()

# Predicting the Test set results
y_pred = regressor_OLS.predict(X_test[:, [0, 3]])
print(y_test - y_pred)
