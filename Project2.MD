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
```
