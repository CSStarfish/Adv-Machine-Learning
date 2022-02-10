# Comparison of Locally Weighted and Random Forest Regression

## Concept of Locally Weighted Regression
The main idea of locally weighted regression is that the predictions are a linear combination of the observed/actual values from the dataset (i.e., the predicted y is a linear combination of the actual values of y).  It applies weights to each value proportional to the distance between the local datapoints, with weights closer to one the closer they are together and closer to zero the further away they are, thus accounting for variability in the dataset.  Linear regression stems from the following equation:
![](linearregression.jpg).  If we multiply the equation by a matrix of weights consisting of the transpose of X (i.e., the independent observations of the dataset) and solve for the beta coefficient, we will find that the expected value of beta is equal to the inverse of the product of X and the transpose of X, multiplied by the product of the transpose of X and y.  If we then plug this value into beta, we will find that each predicted value of y is a linear combination of each actual value of y.  As a result, locally weighted regression tends to be more accurate than random forest regression

(I will be adding further information about the hyperparameter tau and its effects on the model's variability, as well as the differences between parametric and non-parametric models next.)

## Concept of Random Forest Regression
The main idea of random forest regression is that the predictions are made by running a collection of several decision trees at once and then averaging the results of their decisions together.


## Code
Note: I uploaded an ipynb file and I will be including the code directly into the markdown soon.

```
import numpy as np
import pandas as pd
from scipy import linalg
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, train_test_split as tts
from sklearn.metrics import mean_squared_error as mse
from sklearn.preprocessing import StandardScaler

def tricubic(x):
  return np.where(np.abs(x)>1,0,70/81*(1-np.abs(x)**3)**3)
def Epanechnikov(x):
  return np.where(np.abs(x)>1,0,3/4*(1-np.abs(x)**2)) 
def Quartic(x):
  return np.where(np.abs(x)>1,0,15/16*(1-np.abs(x)**2)**2) 

def lowess_reg(x, y, xnew, kern, tau):
    n = len(x)
    yest = np.zeros(n)
    
    w = np.array([kern((x - x[i])/(2*tau)) for i in range(n)])     
    
    for i in range(n):
        weights = w[:, i]
        b = np.array([np.sum(weights * y), np.sum(weights * y * x)])
        A = np.array([[np.sum(weights), np.sum(weights * x)],
                    [np.sum(weights * x), np.sum(weights * x * x)]])
        theta, res, rnk, s = linalg.lstsq(A, b)
        yest[i] = theta[0] + theta[1] * x[i] 
    f = interp1d(x, yest, fill_value='extrapolate')
    return f(xnew)
    
def lowess_kern(x, y, kern, tau):

    n = len(x)
    yest = np.zeros(n)
   
    w = np.array([kern((x - x[i])/(2*tau)) for i in range(n)])     
    
    for i in range(n):
        weights = w[:, i]
        b = np.array([np.sum(weights * y), np.sum(weights * y * x)])
        A = np.array([[np.sum(weights), np.sum(weights * x)],
                    [np.sum(weights * x), np.sum(weights * x * x)]])
        theta, res, rnk, s = linalg.lstsq(A, b)
        yest[i] = theta[0] + theta[1] * x[i] 

    return yest
    
rf = RandomForestRegressor(n_estimators=100,max_depth=3,random_state=123) 
rf2 = RandomForestRegressor(n_estimators=1000,max_depth=3,random_state=123)
rf3 = RandomForestRegressor(n_estimators=1000,max_depth=5,random_state=123)
rf4 = RandomForestRegressor(n_estimators=100,max_depth=5,random_state=123)

scale = StandardScaler()



# Boston Housing Prices Data
boston_data = pd.read_csv("Boston Housing Prices.csv")

x = boston_data["rooms"].values
y = boston_data["cmedv"].values

# Testing Kernels

# Tau = 0.01, Best kernel was Epanechnikov and an MSE of 43.809283109365026
kf = KFold(n_splits=10,shuffle=True,random_state=410)
mse_tri_b1 = []
mse_epa_b1 = []
mse_quart_b1 = []

for idxtrain, idxtest in kf.split(x):
  ytrain = y[idxtrain]
  xtrain = x[idxtrain]
  xtrain = scale.fit_transform(xtrain.reshape(-1,1))
  ytest = y[idxtest]
  xtest = x[idxtest]
  xtest = scale.transform(xtest.reshape(-1,1))
  yhat_tri = lowess_reg(xtrain.ravel(), ytrain, xtest.ravel(), tricubic, 0.01)
  yhat_epa = lowess_reg(xtrain.ravel(), ytrain, xtest.ravel(), Epanechnikov, 0.01)
  yhat_quart = lowess_reg(xtrain.ravel(), ytrain, xtest.ravel(), Quartic, 0.01)
  mse_tri_b1.append(mse(ytest,yhat_tri))
  mse_epa_b1.append(mse(ytest,yhat_epa))
  mse_quart_b1.append(mse(ytest,yhat_quart))
print("The MSE for LOWESS with the tricubic kernel is: " + str(np.mean(mse_tri_b1)))
print("The MSE for LOWESS with the Epanechnikov kernel is: " + str(np.mean(mse_epa_b1)))
print("The MSE for LOWESS with the quartic kernel is: " + str(np.mean(mse_quart_b1)))

# Tau = 0.05, Best kernel was Epanechnikov and an MSE of 36.44728316493236
kf = KFold(n_splits=10,shuffle=True,random_state=410)
mse_tri_b2 = []
mse_epa_b2 = []
mse_quart_b2 = []

for idxtrain, idxtest in kf.split(x):
  ytrain = y[idxtrain]
  xtrain = x[idxtrain]
  xtrain = scale.fit_transform(xtrain.reshape(-1,1))
  ytest = y[idxtest]
  xtest = x[idxtest]
  xtest = scale.transform(xtest.reshape(-1,1))
  yhat_tri = lowess_reg(xtrain.ravel(), ytrain, xtest.ravel(), tricubic, 0.05)
  yhat_epa = lowess_reg(xtrain.ravel(), ytrain, xtest.ravel(), Epanechnikov, 0.05)
  yhat_quart = lowess_reg(xtrain.ravel(), ytrain, xtest.ravel(), Quartic, 0.05)
  mse_tri_b2.append(mse(ytest,yhat_tri))
  mse_epa_b2.append(mse(ytest,yhat_epa))
  mse_quart_b2.append(mse(ytest,yhat_quart))
print("The MSE for LOWESS with the tricubic kernel is: " + str(np.mean(mse_tri_b2)))
print("The MSE for LOWESS with the Epanechnikov kernel is: " + str(np.mean(mse_epa_b2)))
print("The MSE for LOWESS with the quartic kernel is: " + str(np.mean(mse_quart_b2)))

# Tau = 0.1, Best kernel was tricubic and an MSE of 35.44061356471881
kf = KFold(n_splits=10,shuffle=True,random_state=410)
mse_tri_b3 = []
mse_epa_b3 = []
mse_quart_b3 = []

for idxtrain, idxtest in kf.split(x):
  ytrain = y[idxtrain]
  xtrain = x[idxtrain]
  xtrain = scale.fit_transform(xtrain.reshape(-1,1))
  ytest = y[idxtest]
  xtest = x[idxtest]
  xtest = scale.transform(xtest.reshape(-1,1))
  yhat_tri = lowess_reg(xtrain.ravel(), ytrain, xtest.ravel(), tricubic, 0.1)
  yhat_epa = lowess_reg(xtrain.ravel(), ytrain, xtest.ravel(), Epanechnikov, 0.1)
  yhat_quart = lowess_reg(xtrain.ravel(), ytrain, xtest.ravel(), Quartic, 0.1)
  mse_tri_b3.append(mse(ytest,yhat_tri))
  mse_epa_b3.append(mse(ytest,yhat_epa))
  mse_quart_b3.append(mse(ytest,yhat_quart))
print("The MSE for LOWESS with the tricubic kernel is: " + str(np.mean(mse_tri_b3)))
print("The MSE for LOWESS with the Epanechnikov kernel is: " + str(np.mean(mse_epa_b3)))
print("The MSE for LOWESS with the quartic kernel is: " + str(np.mean(mse_quart_b3)))

# Tau = 0.5, Best kernel was quartic with an MSE of 36.924548944943005
kf = KFold(n_splits=10,shuffle=True,random_state=410)
mse_tri_b4 = []
mse_epa_b4 = []
mse_quart_b4 = []

for idxtrain, idxtest in kf.split(x):
  ytrain = y[idxtrain]
  xtrain = x[idxtrain]
  xtrain = scale.fit_transform(xtrain.reshape(-1,1))
  ytest = y[idxtest]
  xtest = x[idxtest]
  xtest = scale.transform(xtest.reshape(-1,1))
  yhat_tri = lowess_reg(xtrain.ravel(), ytrain, xtest.ravel(), tricubic, 0.5)
  yhat_epa = lowess_reg(xtrain.ravel(), ytrain, xtest.ravel(), Epanechnikov, 0.5)
  yhat_quart = lowess_reg(xtrain.ravel(), ytrain, xtest.ravel(), Quartic, 0.5)
  mse_tri_b4.append(mse(ytest,yhat_tri))
  mse_epa_b4.append(mse(ytest,yhat_epa))
  mse_quart_b4.append(mse(ytest,yhat_quart))
print("The MSE for LOWESS with the tricubic kernel is: " + str(np.mean(mse_tri_b4)))
print("The MSE for LOWESS with the Epanechnikov kernel is: " + str(np.mean(mse_epa_b4)))
print("The MSE for LOWESS with the quartic kernel is: " + str(np.mean(mse_quart_b4)))

# Testing Random Forests, Best was a forest of 1000 trees with a depth of 3 and an MSE of 35.87895228710361
kf = KFold(n_splits=10,shuffle=True,random_state=410)
mse_100_3 = []
mse_100_5 = []
mse_1000_3 = []
mse_1000_5 = []

for idxtrain,idxtest in kf.split(x):
  ytrain = y[idxtrain]
  xtrain = x[idxtrain]
  xtrain = scale.fit_transform(xtrain.reshape(-1,1))
  ytest = y[idxtest]
  xtest = x[idxtest]
  xtest = scale.transform(xtest.reshape(-1,1))
  rf.fit(xtrain,ytrain)
  rf2.fit(xtrain,ytrain)
  rf3.fit(xtrain,ytrain)
  rf4.fit(xtrain,ytrain)
  yhat_rf_1003 = rf.predict(xtest)
  yhat_rf_10003 = rf2.predict(xtest)
  yhat_rf_10005 = rf3.predict(xtest)
  yhat_rf_1005 = rf4.predict(xtest)
  mse_100_3.append(mse(ytest,yhat_rf_1003))
  mse_1000_3.append(mse(ytest,yhat_rf_10003))
  mse_1000_5.append(mse(ytest,yhat_rf_10005))
  mse_100_5.append(mse(ytest,yhat_rf_1005))
print("The MSE for Random Forest with 100 trees and depth of 3: " + str(np.mean(mse_100_3)))
print("The MSE for Random Forest with 1000 trees and depth of 3: " + str(np.mean(mse_1000_3)))
print("The MSE for Random Forest with 100 trees and depth of 5: " + str(np.mean(mse_100_5)))
print("The MSE for Random Forest with 1000 trees and depth of 5: " + str(np.mean(mse_1000_5)))

# Testing Kernel vs. Random Forest, Best was Locally Weighted Regression and an MSE of 35.44061356471881
kf = KFold(n_splits=10,shuffle=True,random_state=410)
mse_lwr = []
mse_rf = []

for idxtrain,idxtest in kf.split(x):
  ytrain = y[idxtrain]
  xtrain = x[idxtrain]
  xtrain = scale.fit_transform(xtrain.reshape(-1,1))
  ytest = y[idxtest]
  xtest = x[idxtest]
  xtest = scale.transform(xtest.reshape(-1,1))
  yhat_lwr = lowess_reg(xtrain.ravel(), ytrain, xtest.ravel(), tricubic, 0.1)
  rf2.fit(xtrain,ytrain)
  yhat_rf = rf2.predict(xtest)
  mse_lwr.append(mse(ytest,yhat_lwr))
  mse_rf.append(mse(ytest,yhat_rf))
print("The MSE for Random Forest is: " + str(np.mean(mse_rf)))
print("The MSE for Locally Weighted Regression is: " + str(np.mean(mse_lwr)))



# Cars Data



# Walmart Sales Data
```
