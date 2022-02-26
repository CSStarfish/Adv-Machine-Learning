# Analyzing The Effects of Gradient Boosting on Multivariate Regression Analysis

### Concept of Multivariate Regression Analysis
similar univariate locally weighted (LOWESS) regression, we want to use a non-parametric linear model to predict the y-values by applying weights to each observation row in X.  As discussed in Project 2, this process is performed by multiplying the nxn, diagonal matrix W by y and multiplying the same nxn, diagonal matrix W by the nxp matrix X and the beta vector of coefficients, n different times for each element of y (which is a column vector with shape nx1).  Each of the n matrices, W, are different from the others based on the Euclidean distances between the X observations they are being applied upon.  After calculating the Euclidean distance between each observation of X via the chosen kernel (i.e, tricubic, Epanechnikov, quartic, etc.), weights are appropriately applied closer to a value of 1 when the observations are closer together and closer to a value of 0 when the observations are farther apart.  This is calculated by subtracting each observation of X from X and then dividing by twice the value of tau.  Tau is the smoothing constant provided, and it creates more smoothly-shaped curves at higher values and more volatile/reactive curves at lower values.  As such, it is important to test out a variety of taus to determine an optimal tau for the data that minimizes the mean squared error (MSE) and mean absolute error (MAE).  The beta coefficients are determined by projecting the rows from the dot product of matrix W and y onto the column space of the dot product of W and X.  When you solve such a linear system in which the product of A and beta equals b, the column vector b is a linear combination of A, which is an orthogonal projection of b onto the vector subspace spanned by A's columns. The predicted values of y (which are the dot products of each observation in X and beta) will be used alongside X to interpolate additional X-values designated for testing that fall inside the convex hull of the training X-values to validate the accuracy of the model.  The convex hull is determined by the convex combination of the coordiantes of a point A and a point B, such that the convex combination is equal to the product of t and the coordinates of A, plus the product of the coordinates of B and the quantity (1-t), where t is any number between 0 and 1.  For multivariate regression, we will be determining how multiple features of X contribute to the dependent variable y, as opposed to a single X feature's contribution in univariate regression.  As a result, we will need to use a multi-dimensional interpolator.  In addition, our convex hull will contain all convex combinations of the k points in the n-dimensional Euclidean space, where k is the number of training observations we implemented and n is the number of features in our data.  A key point to make is that we must not extrapolate (and ensure that we only make predictions  within the range of X-values provided during the training of the model.  If we choose to force the model to have an intercept, then we will append a column of 1's and proceed with the remainder of the algorithm.

### Concept of Gradient Boosting


### Concept of Extreme Gradient Boosting (XGBoost)


## Analysis and Visualization of Differences Between Multivariate Regression, Gradient Boosted Multivariate Regression, and Extreme Gradient Boosted Multivariate Regression

### Boston Housing Prices Dataset


### Cars Dataset


### Walmart Sales Dataset


### Concrete Strength Dataset


### Conclusions


### Code - Making Mathematical Comparisons and Plotting Results

Note: Please see my main Github repository (entitled Adv-Machine-Learning) for the code below saved as a .ipynb file named "Project_3.ipynb"). Thank you!

```
# Import Statements
import numpy as np
import pandas as pd
from scipy.linalg import lstsq
from scipy.sparse.linalg import lsmr
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d, griddata, LinearNDInterpolator, NearestNDInterpolator
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, train_test_split as tts
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
from matplotlib import pyplot
import xgboost as xgb

# Setting up kernels and model definitions
# Tricubic Kernel
def Tricubic(x):
  if len(x.shape) == 1:
    x = x.reshape(-1,1)
  d = np.sqrt(np.sum(x**2,axis=1))
  return np.where(d>1,0,70/81*(1-d**3)**3)

# Quartic Kernel
def Quartic(x):
  if len(x.shape) == 1:
    x = x.reshape(-1,1)
  d = np.sqrt(np.sum(x**2,axis=1))
  return np.where(d>1,0,15/16*(1-d**2)**2)

# Epanechnikov Kernel
def Epanechnikov(x):
  if len(x.shape) == 1:
    x = x.reshape(-1,1)
  d = np.sqrt(np.sum(x**2,axis=1))
  return np.where(d>1,0,3/4*(1-d**2)) 
  
# Implementing new kernels that we have not previously used in class
# Triangular Kernel
def Triangular(x):
  if len(x.shape) == 1:
    x = x.reshape(-1,1)
  d = np.sqrt(np.sum(x**2,axis=1))
  return np.where(d>1,0,(1-np.abs(d))) 

# Triweight Kernel
def Triweight(x):
  if len(x.shape) == 1:
    x = x.reshape(-1,1)
  d = np.sqrt(np.sum(x**2,axis=1))
  return np.where(d>1,0,35/32*(1-d**2)**3) 
  
# Defining the locally weighted regression model

def lw_reg(X, y, xnew, kern, tau, intercept):
    # tau is called bandwidth K((x-x[i])/(2*tau))
    n = len(X) # the number of observations
    yest = np.zeros(n)

    if len(y.shape)==1: # here we make column vectors
      y = y.reshape(-1,1)

    if len(X.shape)==1:
      X = X.reshape(-1,1)
    
    if intercept:
      X1 = np.column_stack([np.ones((len(X),1)),X])
    else:
      X1 = X

    w = np.array([kern((X - X[i])/(2*tau)) for i in range(n)])

    #Looping through all X-points
    for i in range(n):          
        W = np.diag(w[:,i])
        b = np.transpose(X1).dot(W).dot(y)
        A = np.transpose(X1).dot(W).dot(X1)
        beta, res, rnk, s = lstsq(A, b)
        yest[i] = np.dot(X1[i],beta)
    if X.shape[1]==1:
      f = interp1d(X.flatten(),yest,fill_value='extrapolate')
    else:
      f = LinearNDInterpolator(X, yest)
    output = f(xnew)
    if sum(np.isnan(output))>0:
      g = NearestNDInterpolator(X,y.ravel()) 
      output[np.isnan(output)] = g(xnew[np.isnan(output)])
    return output
 
# Defining the Gradient Boosted locally weighted regression model
def boosted_lwr(X, y, xnew, kern, tau, intercept):
  # we need decision trees
  # for training the boosted method we use X and y
  Fx = lw_reg(X,y,X,kern,tau,intercept) # we need this for training the Decision Tree
  # Now train the Decision Tree on y_i - F(x_i)
  new_y = y - Fx
  tree_model = DecisionTreeRegressor(max_depth=2, random_state=123)
  tree_model.fit(X,new_y) # new_y is the difference between y and the initial predictions
  output = tree_model.predict(xnew) + lw_reg(X,y,xnew,kern,tau,intercept) # output = h_i + F(x_i)
  return output 
  
# Boston Housing Prices Dataset

# Will be updating soon

# Making plots

# Cars Dataset

# Will be updating soon

# Making plots

# Walmart Sales Dataset

# Making plots

# Will be updating soon

# Making plots

```
