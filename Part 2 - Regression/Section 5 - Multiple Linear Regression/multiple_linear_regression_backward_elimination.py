# Multiple Linear Regression

# Importing the libraries
import numpy as np
import statsmodels.formula.api as sm
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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 43)

def backwardElimination(x, y, sl):
    numVars = len(x[0])
    #indexes to keep track of optimal indexes for test set
    indexes = list(range(0, numVars))
    for i in range(0, numVars):
        regressor_OLS = sm.OLS(y, x).fit()
        maxVar = max(regressor_OLS.pvalues).astype(float)
        if maxVar > sl:
            for j in range(0, numVars - i):
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                    x = np.delete(x, j, 1)
#                   Since we will have to remove the column in full test matrix
                    del indexes[j]
    regressor_OLS.summary()
    return x, indexes

def backwardElimination_rsquared_adj(x, y, SL):
    numVars = len(x[0])
    numRecords = len(x)
    #indexes to keep track of optimal indexes for test set
    indexes = list(range(0, numVars))
    temp = np.zeros((numRecords,numVars)).astype(int)
    for i in range(0, numVars):
        regressor_OLS = sm.OLS(y, x).fit()
        maxVar = max(regressor_OLS.pvalues).astype(float)
        adjR_before = regressor_OLS.rsquared_adj.astype(float)
        if maxVar > SL:
            for j in range(0, numVars - i):
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                    temp[:,j] = x[:, j]
                    x = np.delete(x, j, 1)
                    tmp_regressor = sm.OLS(y, x).fit()
                    adjR_after = tmp_regressor.rsquared_adj.astype(float)
                    if (adjR_before >= adjR_after):
                        x_rollback = np.hstack((x, temp[:,[0,j]]))
                        x_rollback = np.delete(x_rollback, j, 1)
                        print (regressor_OLS.summary())
                        return x_rollback, indexes
                    else:
#                       Since we will have to remove the column in full test matrix
                        del indexes[j]
    regressor_OLS.summary()
    return x, indexes
 
SL = 0.05
X_opt = X_train[:, [0, 1, 2, 3, 4, 5]]

X_Modeled, indexes_opt = backwardElimination(X_opt.astype(np.float64), y_train, SL)
regressor_OLS = sm.OLS(endog = y_train, exog = X_Modeled).fit()

X_Modeled_rsquared_adj, indexes_opt_rsquared_adj = backwardElimination_rsquared_adj(X_opt.astype(np.float64), y_train, SL)
regressor_OLS_rsquared_adj = sm.OLS(endog = y_train, exog = X_Modeled_rsquared_adj).fit()

# Predicting the Test set results
y_pred = regressor_OLS.predict(X_test[:, indexes_opt])
y_pred_rsquared_adj = regressor_OLS_rsquared_adj.predict(X_test[:, indexes_opt_rsquared_adj])
print(y_test - y_pred)
print(y_test - y_pred_rsquared_adj)
print(y_train - regressor_OLS_rsquared_adj.predict(X_train[:, indexes_opt_rsquared_adj]))


'''I have split the data-set into train and test data and then modeled using train data. I also returned the indexes of remaining columns after elimination to use while testing.

Then I tested the model with the corresponding remaining columns in test set. I tested with different combinations of test-train data and the model works well. But for the split with random_state = 40, while Backward Elimination with p-values only model works well,  Backward Elimination with p-values and Adjusted R Squared performs horribly. Could it be the case of underfitting or something else?
'''