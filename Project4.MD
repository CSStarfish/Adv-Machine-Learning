### Multiple Boosting Algorithm and Application
```
# Import Statements
!pip install xlrd==1.2.0
import numpy as np
import pandas as pd
from scipy.linalg import lstsq
from scipy.sparse.linalg import lsmr
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d, griddata, LinearNDInterpolator, NearestNDInterpolator
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, train_test_split as tts
from sklearn.metrics import mean_squared_error as mse
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from matplotlib import pyplot
import lightgbm as lgb
import xgboost as xgb

# Establish kernels for locally-weighted regression
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
  
# Implementing new kernels
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

#Defining the kernel local regression model

def lw_reg(X, y, xnew, kern, tau, intercept):
    n = len(X)
    yest = np.zeros(n)

    if len(y.shape)==1:
      y = y.reshape(-1,1)

    if len(X.shape)==1:
      X = X.reshape(-1,1)
    
    if intercept:
      X1 = np.column_stack([np.ones((len(X),1)),X])
    else:
      X1 = X

    w = np.array([kern((X - X[i])/(2*tau)) for i in range(n)])

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
    
# Multiple boosting algorithm methods
def booster_lwrandchoice(X,y,xnew,kern,tau,model_boosting,nboost):
  Fx = lw_reg(X,y,X,kern,tau,True)
  Fx_new = lw_reg(X,y,xnew,kern,tau,True)
  new_y = y - Fx
  output = Fx
  output_new = Fx_new
  for i in range(nboost):
    model_boosting.fit(X,new_y)
    output += model_boosting.predict(X)
    output_new += model_boosting.predict(xnew)
    new_y = y - output
  return output_new
  
# The following multiple boosting algorithm didn't work nearly as well as booster_lwrandchoice
def booster_lwr(X,y,xnew,kern,tau,nboost):
  Fx = lw_reg(X,y,X,kern,tau,True)
  Fx_new = lw_reg(X,y,xnew,kern,tau,True)
  new_y = y - Fx
  output = Fx
  output_new = Fx_new
  for i in range(nboost):
    output += lw_reg(X,new_y,X,kern,tau,True)
    output_new += lw_reg(X,y,xnew,kern,tau,True)
    new_y = y - output
  return output_new
  
scale = StandardScaler()
# Load data
concrete = pd.read_excel("Concrete_Data.xls")

# Used less than all features because the models were taking several hours to run and I couldn't tell if they were ever going to finish
# because they seemed to be stuck in an infinite loop, so I reduced the number of X features to three.
X = concrete[['Cement (component 1)(kg in a m^3 mixture)', 'Coarse Aggregate  (component 6)(kg in a m^3 mixture)', 'Age (day)']].values
y = concrete["Concrete compressive strength(MPa, megapascals) "].values

# Implement first set of random forest models with which to boost
model_boosting1 = RandomForestRegressor(n_estimators=5, max_depth=2)
model_boosting2 = RandomForestRegressor(n_estimators=10, max_depth=2)
model_boosting3 = RandomForestRegressor(n_estimators=5, max_depth=3)
model_boosting4 = RandomForestRegressor(n_estimators=10, max_depth=3)
model_boosting5 = RandomForestRegressor(n_estimators=50, max_depth=2)
model_boosting6 = RandomForestRegressor(n_estimators=10, max_depth=2)
model_boosting7 = RandomForestRegressor(n_estimators=50, max_depth=3)
model_boosting8 = RandomForestRegressor(n_estimators=100, max_depth=3)

mse_blwr_tric1 = []
mse_blwr_quart1 = []
mse_blwr_epa1 = []
mse_blwr_triw1 = []
mse_blwr_trian1 = []

mse_blwr_tric2 = []
mse_blwr_quart2 = []
mse_blwr_epa2 = []
mse_blwr_triw2 = []
mse_blwr_trian2 = []

mse_blwr_tric3 = []
mse_blwr_quart3 = []
mse_blwr_epa3 = []
mse_blwr_triw3 = []
mse_blwr_trian3 = []

mse_blwr_tric4 = []
mse_blwr_quart4 = []
mse_blwr_epa4 = []
mse_blwr_triw4 = []
mse_blwr_trian4 = []

mse_blwr_tric5 = []
mse_blwr_quart5 = []
mse_blwr_epa5 = []
mse_blwr_triw5 = []
mse_blwr_trian5 = []

mse_blwr_tric6 = []
mse_blwr_quart6 = []
mse_blwr_epa6 = []
mse_blwr_triw6 = []
mse_blwr_trian6 = []

mse_blwr_tric7 = []
mse_blwr_quart7 = []
mse_blwr_epa7 = []
mse_blwr_triw7 = []
mse_blwr_trian7 = []

mse_blwr_tric8 = []
mse_blwr_quart8 = []
mse_blwr_epa8 = []
mse_blwr_triw8 = []
mse_blwr_trian8 = []


kf = KFold(n_splits=5,shuffle=True,random_state=1234)
for idxtrain, idxtest in kf.split(X):
  xtrain = X[idxtrain]
  ytrain = y[idxtrain]
  ytest = y[idxtest]
  xtest = X[idxtest]
  xtrain = scale.fit_transform(xtrain)
  xtest = scale.transform(xtest)
    
  yhat_blwr_1 = booster_lwrandchoice(xtrain,ytrain,xtest, Tricubic,1,model_boosting1,2)
  yhat_blwr_2 = booster_lwrandchoice(xtrain,ytrain,xtest, Quartic,1,model_boosting1,2)
  yhat_blwr_3 = booster_lwrandchoice(xtrain,ytrain,xtest, Epanechnikov,1,model_boosting1,2)
  yhat_blwr_4 = booster_lwrandchoice(xtrain,ytrain,xtest, Triweight,1,model_boosting1,2)
  yhat_blwr_5 = booster_lwrandchoice(xtrain,ytrain,xtest, Triangular,1,model_boosting1,2)

  mse_blwr_tric1.append(mse(ytest, yhat_blwr_1))
  mse_blwr_quart1.append(mse(ytest, yhat_blwr_2))
  mse_blwr_epa1.append(mse(ytest, yhat_blwr_3))
  mse_blwr_triw1.append(mse(ytest, yhat_blwr_4))
  mse_blwr_trian1.append(mse(ytest, yhat_blwr_5))

  yhat_blwr_1 = booster_lwrandchoice(xtrain,ytrain,xtest, Tricubic,1,model_boosting2,2)
  yhat_blwr_2 = booster_lwrandchoice(xtrain,ytrain,xtest, Quartic,1,model_boosting2,2)
  yhat_blwr_3 = booster_lwrandchoice(xtrain,ytrain,xtest, Epanechnikov,1,model_boosting2,2)
  yhat_blwr_4 = booster_lwrandchoice(xtrain,ytrain,xtest, Triweight,1,model_boosting2,2)
  yhat_blwr_5 = booster_lwrandchoice(xtrain,ytrain,xtest, Triangular,1,model_boosting2,2)

  mse_blwr_tric2.append(mse(ytest, yhat_blwr_1))
  mse_blwr_quart2.append(mse(ytest, yhat_blwr_2))
  mse_blwr_epa2.append(mse(ytest, yhat_blwr_3))
  mse_blwr_triw2.append(mse(ytest, yhat_blwr_4))
  mse_blwr_trian2.append(mse(ytest, yhat_blwr_5))

  yhat_blwr_1 = booster_lwrandchoice(xtrain,ytrain,xtest, Tricubic,1,model_boosting3,2)
  yhat_blwr_2 = booster_lwrandchoice(xtrain,ytrain,xtest, Quartic,1,model_boosting3,2)
  yhat_blwr_3 = booster_lwrandchoice(xtrain,ytrain,xtest, Epanechnikov,1,model_boosting3,2)
  yhat_blwr_4 = booster_lwrandchoice(xtrain,ytrain,xtest, Triweight,1,model_boosting3,2)
  yhat_blwr_5 = booster_lwrandchoice(xtrain,ytrain,xtest, Triangular,1,model_boosting3,2)

  mse_blwr_tric3.append(mse(ytest, yhat_blwr_1))
  mse_blwr_quart3.append(mse(ytest, yhat_blwr_2))
  mse_blwr_epa3.append(mse(ytest, yhat_blwr_3))
  mse_blwr_triw3.append(mse(ytest, yhat_blwr_4))
  mse_blwr_trian3.append(mse(ytest, yhat_blwr_5))


  yhat_blwr_1 = booster_lwrandchoice(xtrain,ytrain,xtest, Tricubic,1,model_boosting4,2)
  yhat_blwr_2 = booster_lwrandchoice(xtrain,ytrain,xtest, Quartic,1,model_boosting4,2)
  yhat_blwr_3 = booster_lwrandchoice(xtrain,ytrain,xtest, Epanechnikov,1,model_boosting4,2)
  yhat_blwr_4 = booster_lwrandchoice(xtrain,ytrain,xtest, Triweight,1,model_boosting4,2)
  yhat_blwr_5 = booster_lwrandchoice(xtrain,ytrain,xtest, Triangular,1,model_boosting4,2)

  mse_blwr_tric4.append(mse(ytest, yhat_blwr_1))
  mse_blwr_quart4.append(mse(ytest, yhat_blwr_2))
  mse_blwr_epa4.append(mse(ytest, yhat_blwr_3))
  mse_blwr_triw4.append(mse(ytest, yhat_blwr_4))
  mse_blwr_trian4.append(mse(ytest, yhat_blwr_5))

  yhat_blwr_1 = booster_lwrandchoice(xtrain,ytrain,xtest, Tricubic,1,model_boosting5,2)
  yhat_blwr_2 = booster_lwrandchoice(xtrain,ytrain,xtest, Quartic,1,model_boosting5,2)
  yhat_blwr_3 = booster_lwrandchoice(xtrain,ytrain,xtest, Epanechnikov,1,model_boosting5,2)
  yhat_blwr_4 = booster_lwrandchoice(xtrain,ytrain,xtest, Triweight,1,model_boosting5,2)
  yhat_blwr_5 = booster_lwrandchoice(xtrain,ytrain,xtest, Triangular,1,model_boosting5,2)

  mse_blwr_tric5.append(mse(ytest, yhat_blwr_1))
  mse_blwr_quart5.append(mse(ytest, yhat_blwr_2))
  mse_blwr_epa5.append(mse(ytest, yhat_blwr_3))
  mse_blwr_triw5.append(mse(ytest, yhat_blwr_4))
  mse_blwr_trian5.append(mse(ytest, yhat_blwr_5))

  yhat_blwr_1 = booster_lwrandchoice(xtrain,ytrain,xtest, Tricubic,1,model_boosting6,2)
  yhat_blwr_2 = booster_lwrandchoice(xtrain,ytrain,xtest, Quartic,1,model_boosting6,2)
  yhat_blwr_3 = booster_lwrandchoice(xtrain,ytrain,xtest, Epanechnikov,1,model_boosting6,2)
  yhat_blwr_4 = booster_lwrandchoice(xtrain,ytrain,xtest, Triweight,1,model_boosting6,2)
  yhat_blwr_5 = booster_lwrandchoice(xtrain,ytrain,xtest, Triangular,1,model_boosting6,2)

  mse_blwr_tric6.append(mse(ytest, yhat_blwr_1))
  mse_blwr_quart6.append(mse(ytest, yhat_blwr_2))
  mse_blwr_epa6.append(mse(ytest, yhat_blwr_3))
  mse_blwr_triw6.append(mse(ytest, yhat_blwr_4))
  mse_blwr_trian6.append(mse(ytest, yhat_blwr_5))

  yhat_blwr_1 = booster_lwrandchoice(xtrain,ytrain,xtest, Tricubic,1,model_boosting7,2)
  yhat_blwr_2 = booster_lwrandchoice(xtrain,ytrain,xtest, Quartic,1,model_boosting7,2)
  yhat_blwr_3 = booster_lwrandchoice(xtrain,ytrain,xtest, Epanechnikov,1,model_boosting7,2)
  yhat_blwr_4 = booster_lwrandchoice(xtrain,ytrain,xtest, Triweight,1,model_boosting7,2)
  yhat_blwr_5 = booster_lwrandchoice(xtrain,ytrain,xtest, Triangular,1,model_boosting7,2)

  mse_blwr_tric7.append(mse(ytest, yhat_blwr_1))
  mse_blwr_quart7.append(mse(ytest, yhat_blwr_2))
  mse_blwr_epa7.append(mse(ytest, yhat_blwr_3))
  mse_blwr_triw7.append(mse(ytest, yhat_blwr_4))
  mse_blwr_trian7.append(mse(ytest, yhat_blwr_5))

  yhat_blwr_1 = booster_lwrandchoice(xtrain,ytrain,xtest, Tricubic,1,model_boosting8,2)
  yhat_blwr_2 = booster_lwrandchoice(xtrain,ytrain,xtest, Quartic,1,model_boosting8,2)
  yhat_blwr_3 = booster_lwrandchoice(xtrain,ytrain,xtest, Epanechnikov,1,model_boosting8,2)
  yhat_blwr_4 = booster_lwrandchoice(xtrain,ytrain,xtest, Triweight,1,model_boosting8,2)
  yhat_blwr_5 = booster_lwrandchoice(xtrain,ytrain,xtest, Triangular,1,model_boosting8,2)

  mse_blwr_tric8.append(mse(ytest, yhat_blwr_1))
  mse_blwr_quart8.append(mse(ytest, yhat_blwr_2))
  mse_blwr_epa8.append(mse(ytest, yhat_blwr_3))
  mse_blwr_triw8.append(mse(ytest, yhat_blwr_4))
  mse_blwr_trian8.append(mse(ytest, yhat_blwr_5))

print('The Cross-validated Mean Squared Error for Boosted LWR with tricubic is : '+str(np.mean(mse_blwr_tric1)))
print('The Cross-validated Mean Squared Error for Boosted LWR with quartic is : '+str(np.mean(mse_blwr_quart1)))
print('The Cross-validated Mean Squared Error for Boosted LWR with Epanechnikov is : '+str(np.mean(mse_blwr_epa1)))
print('The Cross-validated Mean Squared Error for Boosted LWR with triweight is : '+str(np.mean(mse_blwr_triw1)))
print('The Cross-validated Mean Squared Error for Boosted LWR with triangular is : '+str(np.mean(mse_blwr_trian1)))
print('\n')
print('The Cross-validated Mean Squared Error for Boosted LWR with tricubic is  : '+str(np.mean(mse_blwr_tric2)))
print('The Cross-validated Mean Squared Error for Boosted LWR with quartic is : '+str(np.mean(mse_blwr_quart2)))
print('The Cross-validated Mean Squared Error for Boosted LWR with Epanechnikov is : '+str(np.mean(mse_blwr_epa2)))
print('The Cross-validated Mean Squared Error for Boosted LWR with triweight is : '+str(np.mean(mse_blwr_triw2)))
print('The Cross-validated Mean Squared Error for Boosted LWR with triangular is : '+str(np.mean(mse_blwr_trian2)))
print('\n')
print('The Cross-validated Mean Squared Error for Boosted LWR with tricubic is : '+str(np.mean(mse_blwr_tric3)))
print('The Cross-validated Mean Squared Error for Boosted LWR with quartic is : '+str(np.mean(mse_blwr_quart3)))
print('The Cross-validated Mean Squared Error for Boosted LWR with Epanechnikov is : '+str(np.mean(mse_blwr_epa3)))
print('The Cross-validated Mean Squared Error for Boosted LWR with triweight is : '+str(np.mean(mse_blwr_triw3)))
print('The Cross-validated Mean Squared Error for Boosted LWR with triangular is : '+str(np.mean(mse_blwr_trian3)))
print('\n')
print('The Cross-validated Mean Squared Error for Boosted LWR with tricubic is : '+str(np.mean(mse_blwr_tric4)))
print('The Cross-validated Mean Squared Error for Boosted LWR with quartic is : '+str(np.mean(mse_blwr_quart4)))
print('The Cross-validated Mean Squared Error for Boosted LWR with Epanechnikov is : '+str(np.mean(mse_blwr_epa4)))
print('The Cross-validated Mean Squared Error for Boosted LWR with triweight is : '+str(np.mean(mse_blwr_triw4)))
print('The Cross-validated Mean Squared Error for Boosted LWR with triangular is : '+str(np.mean(mse_blwr_trian4)))
print('\n')
print('The Cross-validated Mean Squared Error for Boosted LWR with tricubic is : '+str(np.mean(mse_blwr_tric5)))
print('The Cross-validated Mean Squared Error for Boosted LWR with quartic is : '+str(np.mean(mse_blwr_quart5)))
print('The Cross-validated Mean Squared Error for Boosted LWR with Epanechnikov is : '+str(np.mean(mse_blwr_epa5)))
print('The Cross-validated Mean Squared Error for Boosted LWR with triweight is : '+str(np.mean(mse_blwr_triw5)))
print('The Cross-validated Mean Squared Error for Boosted LWR with triangular is : '+str(np.mean(mse_blwr_trian5)))
print('\n')
print('The Cross-validated Mean Squared Error for Boosted LWR with tricubic is : '+str(np.mean(mse_blwr_tric6)))
print('The Cross-validated Mean Squared Error for Boosted LWR with quartic is : '+str(np.mean(mse_blwr_quart6)))
print('The Cross-validated Mean Squared Error for Boosted LWR with Epanechnikov is : '+str(np.mean(mse_blwr_epa6)))
print('The Cross-validated Mean Squared Error for Boosted LWR with triweight is : '+str(np.mean(mse_blwr_triw6)))
print('The Cross-validated Mean Squared Error for Boosted LWR with triangular is : '+str(np.mean(mse_blwr_trian6)))
print('\n')
print('The Cross-validated Mean Squared Error for Boosted LWR with tricubic is : '+str(np.mean(mse_blwr_tric7)))
print('The Cross-validated Mean Squared Error for Boosted LWR with quartic is : '+str(np.mean(mse_blwr_quart7)))
print('The Cross-validated Mean Squared Error for Boosted LWR with Epanechnikov is : '+str(np.mean(mse_blwr_epa7)))
print('The Cross-validated Mean Squared Error for Boosted LWR with triweight is : '+str(np.mean(mse_blwr_triw7)))
print('The Cross-validated Mean Squared Error for Boosted LWR with triangular is : '+str(np.mean(mse_blwr_trian7)))
print('\n')
print('The Cross-validated Mean Squared Error for Boosted LWR with tricubic is : '+str(np.mean(mse_blwr_tric8)))
print('The Cross-validated Mean Squared Error for Boosted LWR with quartic is : '+str(np.mean(mse_blwr_quart8)))
print('The Cross-validated Mean Squared Error for Boosted LWR with Epanechikov is : '+str(np.mean(mse_blwr_epa8)))
print('The Cross-validated Mean Squared Error for Boosted LWR with triweight is : '+str(np.mean(mse_blwr_triw8)))
print('The Cross-validated Mean Squared Error for Boosted LWR with triangular is : '+str(np.mean(mse_blwr_trian8)))

# The seventh random forest model had the best results and the triweight/quartic kernels tended to outperform the others
# Tau = 0.01
mse_blwr_triw7 = []
mse_blwr_quart7 = []

mse_blwr_triw8 = []
mse_blwr_quart8 = []

kf = KFold(n_splits=5,shuffle=True,random_state=1234)
for idxtrain, idxtest in kf.split(X):
  xtrain = X[idxtrain]
  ytrain = y[idxtrain]
  ytest = y[idxtest]
  xtest = X[idxtest]
  xtrain = scale.fit_transform(xtrain)
  xtest = scale.transform(xtest)
    
  yhat_blwr_1 = booster_lwrandchoice(xtrain,ytrain,xtest, Triweight,0.01,model_boosting7,2)
  yhat_blwr_2 = booster_lwrandchoice(xtrain,ytrain,xtest, Quartic,0.01,model_boosting7,2)

  mse_blwr_triw7.append(mse(ytest, yhat_blwr_1))
  mse_blwr_quart7.append(mse(ytest, yhat_blwr_2))

  yhat_blwr_1 = booster_lwrandchoice(xtrain,ytrain,xtest, Triweight,0.01,model_boosting8,2)
  yhat_blwr_2 = booster_lwrandchoice(xtrain,ytrain,xtest, Quartic,0.01,model_boosting8,2)

  mse_blwr_triw8.append(mse(ytest, yhat_blwr_1))
  mse_blwr_quart8.append(mse(ytest, yhat_blwr_2))

print('The Cross-validated Mean Squared Error for Boosted LWR with triweight is : '+str(np.mean(mse_blwr_triw7)))
print('The Cross-validated Mean Squared Error for Boosted LWR with quartic is : '+str(np.mean(mse_blwr_quart7)))
print('\n')
print('The Cross-validated Mean Squared Error for Boosted LWR with triweight is  : '+str(np.mean(mse_blwr_triw8)))
print('The Cross-validated Mean Squared Error for Boosted LWR with quartic is : '+str(np.mean(mse_blwr_quart8)))

# Tau = 0.05
mse_blwr_tric7 = []
mse_blwr_quart7 = []

mse_blwr_tric8 = []
mse_blwr_quart8 = []

kf = KFold(n_splits=5,shuffle=True,random_state=1234)
for idxtrain, idxtest in kf.split(X):
  xtrain = X[idxtrain]
  ytrain = y[idxtrain]
  ytest = y[idxtest]
  xtest = X[idxtest]
  xtrain = scale.fit_transform(xtrain)
  xtest = scale.transform(xtest)
  
  yhat_blwr_1 = booster_lwrandchoice(xtrain,ytrain,xtest, Triweight,0.05,model_boosting7,2)
  yhat_blwr_2 = booster_lwrandchoice(xtrain,ytrain,xtest, Quartic,0.05,model_boosting7,2)

  mse_blwr_triw7.append(mse(ytest, yhat_blwr_1))
  mse_blwr_quart7.append(mse(ytest, yhat_blwr_2))

  yhat_blwr_1 = booster_lwrandchoice(xtrain,ytrain,xtest, Triweight,0.05,model_boosting8,2)
  yhat_blwr_2 = booster_lwrandchoice(xtrain,ytrain,xtest, Quartic,0.05,model_boosting8,2)

  mse_blwr_triw8.append(mse(ytest, yhat_blwr_1))
  mse_blwr_quart8.append(mse(ytest, yhat_blwr_2))

print('The Cross-validated Mean Squared Error for Boosted LWR with triweight is : '+str(np.mean(mse_blwr_triw7)))
print('The Cross-validated Mean Squared Error for Boosted LWR with quartic is : '+str(np.mean(mse_blwr_quart7)))
print('\n')
print('The Cross-validated Mean Squared Error for Boosted LWR with triweight is  : '+str(np.mean(mse_blwr_triw8)))
print('The Cross-validated Mean Squared Error for Boosted LWR with quartic is : '+str(np.mean(mse_blwr_quart8)))

# Tau = 0.1
mse_blwr_tric7 = []
mse_blwr_quart7 = []


mse_blwr_tric8 = []
mse_blwr_quart8 = []



kf = KFold(n_splits=5,shuffle=True,random_state=1234)
for idxtrain, idxtest in kf.split(X):
  xtrain = X[idxtrain]
  ytrain = y[idxtrain]
  ytest = y[idxtest]
  xtest = X[idxtest]
  xtrain = scale.fit_transform(xtrain)
  xtest = scale.transform(xtest)
  
  yhat_blwr_1 = booster_lwrandchoice(xtrain,ytrain,xtest, Triweight,0.1,model_boosting7,2)
  yhat_blwr_2 = booster_lwrandchoice(xtrain,ytrain,xtest, Quartic,0.1,model_boosting7,2)

  mse_blwr_triw7.append(mse(ytest, yhat_blwr_1))
  mse_blwr_quart7.append(mse(ytest, yhat_blwr_2))

  yhat_blwr_1 = booster_lwrandchoice(xtrain,ytrain,xtest, Triweight,0.1,model_boosting8,2)
  yhat_blwr_2 = booster_lwrandchoice(xtrain,ytrain,xtest, Quartic,0.1,model_boosting8,2)

  mse_blwr_triw8.append(mse(ytest, yhat_blwr_1))
  mse_blwr_quart8.append(mse(ytest, yhat_blwr_2))

print('The Cross-validated Mean Squared Error for Boosted LWR with triweight is : '+str(np.mean(mse_blwr_triw7)))
print('The Cross-validated Mean Squared Error for Boosted LWR with quartic is : '+str(np.mean(mse_blwr_quart7)))
print('\n')
print('The Cross-validated Mean Squared Error for Boosted LWR with triweight is  : '+str(np.mean(mse_blwr_triw8)))
print('The Cross-validated Mean Squared Error for Boosted LWR with quartic is : '+str(np.mean(mse_blwr_quart8)))

# Tau = 0.5
mse_blwr_tric7 = []
mse_blwr_quart7 = []


mse_blwr_tric8 = []
mse_blwr_quart8 = []



kf = KFold(n_splits=5,shuffle=True,random_state=1234)
for idxtrain, idxtest in kf.split(X):
  xtrain = X[idxtrain]
  ytrain = y[idxtrain]
  ytest = y[idxtest]
  xtest = X[idxtest]
  xtrain = scale.fit_transform(xtrain)
  xtest = scale.transform(xtest)
  
  yhat_blwr_1 = booster_lwrandchoice(xtrain,ytrain,xtest, Triweight,0.5,model_boosting7,2)
  yhat_blwr_2 = booster_lwrandchoice(xtrain,ytrain,xtest, Quartic,0.5,model_boosting7,2)

  mse_blwr_triw7.append(mse(ytest, yhat_blwr_1))
  mse_blwr_quart7.append(mse(ytest, yhat_blwr_2))

  yhat_blwr_1 = booster_lwrandchoice(xtrain,ytrain,xtest, Triweight,0.5,model_boosting8,2)
  yhat_blwr_2 = booster_lwrandchoice(xtrain,ytrain,xtest, Quartic,0.5,model_boosting8,2)

  mse_blwr_triw8.append(mse(ytest, yhat_blwr_1))
  mse_blwr_quart8.append(mse(ytest, yhat_blwr_2))

print('The Cross-validated Mean Squared Error for Boosted LWR with triweight is : '+str(np.mean(mse_blwr_triw7)))
print('The Cross-validated Mean Squared Error for Boosted LWR with quartic is : '+str(np.mean(mse_blwr_quart7)))
print('\n')
print('The Cross-validated Mean Squared Error for Boosted LWR with triweight is  : '+str(np.mean(mse_blwr_triw8)))
print('The Cross-validated Mean Squared Error for Boosted LWR with quartic is : '+str(np.mean(mse_blwr_quart8)))

Tau = 1
mse_blwr_tric7 = []
mse_blwr_quart7 = []


mse_blwr_tric8 = []
mse_blwr_quart8 = []



kf = KFold(n_splits=5,shuffle=True,random_state=1234)
for idxtrain, idxtest in kf.split(X):
  xtrain = X[idxtrain]
  ytrain = y[idxtrain]
  ytest = y[idxtest]
  xtest = X[idxtest]
  xtrain = scale.fit_transform(xtrain)
  xtest = scale.transform(xtest)
  
  yhat_blwr_1 = booster_lwrandchoice(xtrain,ytrain,xtest, Triweight,1,model_boosting7,2)
  yhat_blwr_2 = booster_lwrandchoice(xtrain,ytrain,xtest, Quartic,1,model_boosting7,2)

  mse_blwr_triw7.append(mse(ytest, yhat_blwr_1))
  mse_blwr_quart7.append(mse(ytest, yhat_blwr_2))

  yhat_blwr_1 = booster_lwrandchoice(xtrain,ytrain,xtest, Triweight,1,model_boosting8,2)
  yhat_blwr_2 = booster_lwrandchoice(xtrain,ytrain,xtest, Quartic,1,model_boosting8,2)

  mse_blwr_triw8.append(mse(ytest, yhat_blwr_1))
  mse_blwr_quart8.append(mse(ytest, yhat_blwr_2))

print('The Cross-validated Mean Squared Error for Boosted LWR with triweight is : '+str(np.mean(mse_blwr_triw7)))
print('The Cross-validated Mean Squared Error for Boosted LWR with quartic is : '+str(np.mean(mse_blwr_quart7)))
print('\n')
print('The Cross-validated Mean Squared Error for Boosted LWR with triweight is  : '+str(np.mean(mse_blwr_triw8)))
print('The Cross-validated Mean Squared Error for Boosted LWR with quartic is : '+str(np.mean(mse_blwr_quart8)))


# Alternate multiple boosting method (didn't work as well)
mse_blwr_tric1 = []
mse_blwr_quart1 = []
mse_blwr_epa1 = []
mse_blwr_triw1 = []
mse_blwr_trian1 = []

mse_blwr_tric2 = []
mse_blwr_quart2 = []
mse_blwr_epa2 = []
mse_blwr_triw2 = []
mse_blwr_trian2 = []

mse_blwr_tric3 = []
mse_blwr_quart3 = []
mse_blwr_epa3 = []
mse_blwr_triw3 = []
mse_blwr_trian3 = []

mse_blwr_tric4 = []
mse_blwr_quart4 = []
mse_blwr_epa4 = []
mse_blwr_triw4 = []
mse_blwr_trian4 = []

mse_blwr_tric5 = []
mse_blwr_quart5 = []
mse_blwr_epa5 = []
mse_blwr_triw5 = []
mse_blwr_trian5 = []

mse_blwr_tric6 = []
mse_blwr_quart6 = []
mse_blwr_epa6 = []
mse_blwr_triw6 = []
mse_blwr_trian6 = []

mse_blwr_tric7 = []
mse_blwr_quart7 = []
mse_blwr_epa7 = []
mse_blwr_triw7 = []
mse_blwr_trian7 = []

mse_blwr_tric8 = []
mse_blwr_quart8 = []
mse_blwr_epa8 = []
mse_blwr_triw8 = []
mse_blwr_trian8 = []


kf = KFold(n_splits=3,shuffle=True,random_state=1234)
for idxtrain, idxtest in kf.split(X):
  xtrain = X[idxtrain]
  ytrain = y[idxtrain]
  ytest = y[idxtest]
  xtest = X[idxtest]
  xtrain = scale.fit_transform(xtrain)
  xtest = scale.transform(xtest)
    
  yhat_blwr_1 = booster_lwr(xtrain,ytrain,xtest, Tricubic,1,2)
  yhat_blwr_2 = booster_lwr(xtrain,ytrain,xtest, Quartic,1,2)
  yhat_blwr_3 = booster_lwr(xtrain,ytrain,xtest, Epanechnikov,1,2)
  yhat_blwr_4 = booster_lwr(xtrain,ytrain,xtest, Triweight,1,2)
  yhat_blwr_5 = booster_lwr(xtrain,ytrain,xtest, Triangular,1,2)

  mse_blwr_tric1.append(mse(ytest, yhat_blwr_1))
  mse_blwr_quart1.append(mse(ytest, yhat_blwr_2))
  mse_blwr_epa1.append(mse(ytest, yhat_blwr_3))
  mse_blwr_triw1.append(mse(ytest, yhat_blwr_4))
  mse_blwr_trian1.append(mse(ytest, yhat_blwr_5))

  yhat_blwr_1 = booster_lwr(xtrain,ytrain,xtest, Tricubic,1,2)
  yhat_blwr_2 = booster_lwr(xtrain,ytrain,xtest, Quartic,1,2)
  yhat_blwr_3 = booster_lwr(xtrain,ytrain,xtest, Epanechnikov,1,2)
  yhat_blwr_4 = booster_lwr(xtrain,ytrain,xtest, Triweight,1,2)
  yhat_blwr_5 = booster_lwr(xtrain,ytrain,xtest, Triangular,1,2)

  mse_blwr_tric2.append(mse(ytest, yhat_blwr_1))
  mse_blwr_quart2.append(mse(ytest, yhat_blwr_2))
  mse_blwr_epa2.append(mse(ytest, yhat_blwr_3))
  mse_blwr_triw2.append(mse(ytest, yhat_blwr_4))
  mse_blwr_trian2.append(mse(ytest, yhat_blwr_5))

  yhat_blwr_1 = booster_lwr(xtrain,ytrain,xtest, Tricubic,1,2)
  yhat_blwr_2 = booster_lwr(xtrain,ytrain,xtest, Quartic,1,2)
  yhat_blwr_3 = booster_lwr(xtrain,ytrain,xtest, Epanechnikov,1,2)
  yhat_blwr_4 = booster_lwr(xtrain,ytrain,xtest, Triweight,1,2)
  yhat_blwr_5 = booster_lwr(xtrain,ytrain,xtest, Triangular,1,2)

  mse_blwr_tric3.append(mse(ytest, yhat_blwr_1))
  mse_blwr_quart3.append(mse(ytest, yhat_blwr_2))
  mse_blwr_epa3.append(mse(ytest, yhat_blwr_3))
  mse_blwr_triw3.append(mse(ytest, yhat_blwr_4))
  mse_blwr_trian3.append(mse(ytest, yhat_blwr_5))

  yhat_blwr_1 = booster_lwr(xtrain,ytrain,xtest, Tricubic,1,2)
  yhat_blwr_2 = booster_lwr(xtrain,ytrain,xtest, Quartic,1,2)
  yhat_blwr_3 = booster_lwr(xtrain,ytrain,xtest, Epanechnikov,1,2)
  yhat_blwr_4 = booster_lwr(xtrain,ytrain,xtest, Triweight,1,2)
  yhat_blwr_5 = booster_lwr(xtrain,ytrain,xtest, Triangular,1,2)

  mse_blwr_tric4.append(mse(ytest, yhat_blwr_1))
  mse_blwr_quart4.append(mse(ytest, yhat_blwr_2))
  mse_blwr_epa4.append(mse(ytest, yhat_blwr_3))
  mse_blwr_triw4.append(mse(ytest, yhat_blwr_4))
  mse_blwr_trian4.append(mse(ytest, yhat_blwr_5))

  yhat_blwr_1 = booster_lwr(xtrain,ytrain,xtest, Tricubic,1,2)
  yhat_blwr_2 = booster_lwr(xtrain,ytrain,xtest, Quartic,1,2)
  yhat_blwr_3 = booster_lwr(xtrain,ytrain,xtest, Epanechnikov,1,2)
  yhat_blwr_4 = booster_lwr(xtrain,ytrain,xtest, Triweight,1,2)
  yhat_blwr_5 = booster_lwr(xtrain,ytrain,xtest, Triangular,1,2)

  mse_blwr_tric5.append(mse(ytest, yhat_blwr_1))
  mse_blwr_quart5.append(mse(ytest, yhat_blwr_2))
  mse_blwr_epa5.append(mse(ytest, yhat_blwr_3))
  mse_blwr_triw5.append(mse(ytest, yhat_blwr_4))
  mse_blwr_trian5.append(mse(ytest, yhat_blwr_5))

  yhat_blwr_1 = booster_lwr(xtrain,ytrain,xtest, Tricubic,1,2)
  yhat_blwr_2 = booster_lwr(xtrain,ytrain,xtest, Quartic,1,2)
  yhat_blwr_3 = booster_lwr(xtrain,ytrain,xtest, Epanechnikov,1,2)
  yhat_blwr_4 = booster_lwr(xtrain,ytrain,xtest, Triweight,1,2)
  yhat_blwr_5 = booster_lwr(xtrain,ytrain,xtest, Triangular,1,2)

  mse_blwr_tric6.append(mse(ytest, yhat_blwr_1))
  mse_blwr_quart6.append(mse(ytest, yhat_blwr_2))
  mse_blwr_epa6.append(mse(ytest, yhat_blwr_3))
  mse_blwr_triw6.append(mse(ytest, yhat_blwr_4))
  mse_blwr_trian6.append(mse(ytest, yhat_blwr_5))

  yhat_blwr_1 = booster_lwr(xtrain,ytrain,xtest, Tricubic,1,2)
  yhat_blwr_2 = booster_lwr(xtrain,ytrain,xtest, Quartic,1,2)
  yhat_blwr_3 = booster_lwr(xtrain,ytrain,xtest, Epanechnikov,1,2)
  yhat_blwr_4 = booster_lwr(xtrain,ytrain,xtest, Triweight,1,2)
  yhat_blwr_5 = booster_lwr(xtrain,ytrain,xtest, Triangular,1,2)

  mse_blwr_tric7.append(mse(ytest, yhat_blwr_1))
  mse_blwr_quart7.append(mse(ytest, yhat_blwr_2))
  mse_blwr_epa7.append(mse(ytest, yhat_blwr_3))
  mse_blwr_triw7.append(mse(ytest, yhat_blwr_4))
  mse_blwr_trian7.append(mse(ytest, yhat_blwr_5))

  yhat_blwr_1 = booster_lwr(xtrain,ytrain,xtest, Tricubic,1,2)
  yhat_blwr_2 = booster_lwr(xtrain,ytrain,xtest, Quartic,1,2)
  yhat_blwr_3 = booster_lwr(xtrain,ytrain,xtest, Epanechnikov,1,2)
  yhat_blwr_4 = booster_lwr(xtrain,ytrain,xtest, Triweight,1,2)
  yhat_blwr_5 = booster_lwr(xtrain,ytrain,xtest, Triangular,1,2)

  mse_blwr_tric8.append(mse(ytest, yhat_blwr_1))
  mse_blwr_quart8.append(mse(ytest, yhat_blwr_2))
  mse_blwr_epa8.append(mse(ytest, yhat_blwr_3))
  mse_blwr_triw8.append(mse(ytest, yhat_blwr_4))
  mse_blwr_trian8.append(mse(ytest, yhat_blwr_5))


print('The Cross-validated Mean Squared Error for Boosted LWR with tricubic is : '+str(np.mean(mse_blwr_tric1)))
print('The Cross-validated Mean Squared Error for Boosted LWR with quartic is : '+str(np.mean(mse_blwr_quart1)))
print('The Cross-validated Mean Squared Error for Boosted LWR with Epanechnikov is : '+str(np.mean(mse_blwr_epa1)))
print('The Cross-validated Mean Squared Error for Boosted LWR with triweight is : '+str(np.mean(mse_blwr_triw1)))
print('The Cross-validated Mean Squared Error for Boosted LWR with triangular is : '+str(np.mean(mse_blwr_trian1)))
print('\n')
print('The Cross-validated Mean Squared Error for Boosted LWR with tricubic is  : '+str(np.mean(mse_blwr_tric2)))
print('The Cross-validated Mean Squared Error for Boosted LWR with quartic is : '+str(np.mean(mse_blwr_quart2)))
print('The Cross-validated Mean Squared Error for Boosted LWR with Epanechnikov is : '+str(np.mean(mse_blwr_epa2)))
print('The Cross-validated Mean Squared Error for Boosted LWR with triweight is : '+str(np.mean(mse_blwr_triw2)))
print('The Cross-validated Mean Squared Error for Boosted LWR with triangular is : '+str(np.mean(mse_blwr_trian2)))
print('\n')
print('The Cross-validated Mean Squared Error for Boosted LWR with tricubic is : '+str(np.mean(mse_blwr_tric3)))
print('The Cross-validated Mean Squared Error for Boosted LWR with quartic is : '+str(np.mean(mse_blwr_quart3)))
print('The Cross-validated Mean Squared Error for Boosted LWR with Epanechnikov is : '+str(np.mean(mse_blwr_epa3)))
print('The Cross-validated Mean Squared Error for Boosted LWR with triweight is : '+str(np.mean(mse_blwr_triw3)))
print('The Cross-validated Mean Squared Error for Boosted LWR with triangular is : '+str(np.mean(mse_blwr_trian3)))
print('\n')
print('The Cross-validated Mean Squared Error for Boosted LWR with tricubic is : '+str(np.mean(mse_blwr_tric4)))
print('The Cross-validated Mean Squared Error for Boosted LWR with quartic is : '+str(np.mean(mse_blwr_quart4)))
print('The Cross-validated Mean Squared Error for Boosted LWR with Epanechnikov is : '+str(np.mean(mse_blwr_epa4)))
print('The Cross-validated Mean Squared Error for Boosted LWR with triweight is : '+str(np.mean(mse_blwr_triw4)))
print('The Cross-validated Mean Squared Error for Boosted LWR with triangular is : '+str(np.mean(mse_blwr_trian4)))
print('\n')
print('The Cross-validated Mean Squared Error for Boosted LWR with tricubic is : '+str(np.mean(mse_blwr_tric5)))
print('The Cross-validated Mean Squared Error for Boosted LWR with quartic is : '+str(np.mean(mse_blwr_quart5)))
print('The Cross-validated Mean Squared Error for Boosted LWR with Epanechnikov is : '+str(np.mean(mse_blwr_epa5)))
print('The Cross-validated Mean Squared Error for Boosted LWR with triweight is : '+str(np.mean(mse_blwr_triw5)))
print('The Cross-validated Mean Squared Error for Boosted LWR with triangular is : '+str(np.mean(mse_blwr_trian5)))
print('\n')
print('The Cross-validated Mean Squared Error for Boosted LWR with tricubic is : '+str(np.mean(mse_blwr_tric6)))
print('The Cross-validated Mean Squared Error for Boosted LWR with quartic is : '+str(np.mean(mse_blwr_quart6)))
print('The Cross-validated Mean Squared Error for Boosted LWR with Epanechnikov is : '+str(np.mean(mse_blwr_epa6)))
print('The Cross-validated Mean Squared Error for Boosted LWR with triweight is : '+str(np.mean(mse_blwr_triw6)))
print('The Cross-validated Mean Squared Error for Boosted LWR with triangular is : '+str(np.mean(mse_blwr_trian6)))
print('\n')
print('The Cross-validated Mean Squared Error for Boosted LWR with tricubic is : '+str(np.mean(mse_blwr_tric7)))
print('The Cross-validated Mean Squared Error for Boosted LWR with quartic is : '+str(np.mean(mse_blwr_quart7)))
print('The Cross-validated Mean Squared Error for Boosted LWR with Epanechnikov is : '+str(np.mean(mse_blwr_epa7)))
print('The Cross-validated Mean Squared Error for Boosted LWR with triweight is : '+str(np.mean(mse_blwr_triw7)))
print('The Cross-validated Mean Squared Error for Boosted LWR with triangular is : '+str(np.mean(mse_blwr_trian7)))
print('\n')
print('The Cross-validated Mean Squared Error for Boosted LWR with tricubic is : '+str(np.mean(mse_blwr_tric8)))
print('The Cross-validated Mean Squared Error for Boosted LWR with quartic is : '+str(np.mean(mse_blwr_quart8)))
print('The Cross-validated Mean Squared Error for Boosted LWR with Epanechikov is : '+str(np.mean(mse_blwr_epa8)))
print('The Cross-validated Mean Squared Error for Boosted LWR with triweight is : '+str(np.mean(mse_blwr_triw8)))
print('The Cross-validated Mean Squared Error for Boosted LWR with triangular is : '+str(np.mean(mse_blwr_trian8)))

# New set of XGBoost models to boost
model_boosting1 = xgb.XGBRegressor(max_depth=2, n_estimators=10, objective ='reg:squarederror')
model_boosting2 = xgb.XGBRegressor(max_depth=3, n_estimators=10, objective ='reg:squarederror')
model_boosting3 = xgb.XGBRegressor(max_depth=2, n_estimators=50, objective ='reg:squarederror')
model_boosting4 = xgb.XGBRegressor(max_depth=3, n_estimators=50, objective ='reg:squarederror')
model_boosting5 = xgb.XGBRegressor(max_depth=2, n_estimators=100, objective ='reg:squarederror')
model_boosting6 = xgb.XGBRegressor(max_depth=3, n_estimators=100, objective ='reg:squarederror')

# we want more nested cross-validations

mse_blwr_tric1 = []
mse_blwr_quart1 = []
mse_blwr_epa1 = []
mse_blwr_triw1 = []
mse_blwr_trian1 = []

mse_blwr_tric2 = []
mse_blwr_quart2 = []
mse_blwr_epa2 = []
mse_blwr_triw2 = []
mse_blwr_trian2 = []

mse_blwr_tric3 = []
mse_blwr_quart3 = []
mse_blwr_epa3 = []
mse_blwr_triw3 = []
mse_blwr_trian3 = []

mse_blwr_tric4 = []
mse_blwr_quart4 = []
mse_blwr_epa4 = []
mse_blwr_triw4 = []
mse_blwr_trian4 = []

mse_blwr_tric5 = []
mse_blwr_quart5 = []
mse_blwr_epa5 = []
mse_blwr_triw5 = []
mse_blwr_trian5 = []

mse_blwr_tric6 = []
mse_blwr_quart6 = []
mse_blwr_epa6 = []
mse_blwr_triw6 = []
mse_blwr_trian6 = []


kf = KFold(n_splits=5,shuffle=True,random_state=1234)
for idxtrain, idxtest in kf.split(X):
  xtrain = X[idxtrain]
  ytrain = y[idxtrain]
  ytest = y[idxtest]
  xtest = X[idxtest]
  xtrain = scale.fit_transform(xtrain)
  xtest = scale.transform(xtest)
    
  yhat_blwr_1 = booster_lwrandchoice(xtrain,ytrain,xtest, Tricubic,1,model_boosting1,2)
  yhat_blwr_2 = booster_lwrandchoice(xtrain,ytrain,xtest, Quartic,1,model_boosting1,2)
  yhat_blwr_3 = booster_lwrandchoice(xtrain,ytrain,xtest, Epanechnikov,1,model_boosting1,2)
  yhat_blwr_4 = booster_lwrandchoice(xtrain,ytrain,xtest, Triweight,1,model_boosting1,2)
  yhat_blwr_5 = booster_lwrandchoice(xtrain,ytrain,xtest, Triangular,1,model_boosting1,2)

  mse_blwr_tric1.append(mse(ytest, yhat_blwr_1))
  mse_blwr_quart1.append(mse(ytest, yhat_blwr_2))
  mse_blwr_epa1.append(mse(ytest, yhat_blwr_3))
  mse_blwr_triw1.append(mse(ytest, yhat_blwr_4))
  mse_blwr_trian1.append(mse(ytest, yhat_blwr_5))

  yhat_blwr_1 = booster_lwrandchoice(xtrain,ytrain,xtest, Tricubic,1,model_boosting2,2)
  yhat_blwr_2 = booster_lwrandchoice(xtrain,ytrain,xtest, Quartic,1,model_boosting2,2)
  yhat_blwr_3 = booster_lwrandchoice(xtrain,ytrain,xtest, Epanechnikov,1,model_boosting2,2)
  yhat_blwr_4 = booster_lwrandchoice(xtrain,ytrain,xtest, Triweight,1,model_boosting2,2)
  yhat_blwr_5 = booster_lwrandchoice(xtrain,ytrain,xtest, Triangular,1,model_boosting2,2)

  mse_blwr_tric2.append(mse(ytest, yhat_blwr_1))
  mse_blwr_quart2.append(mse(ytest, yhat_blwr_2))
  mse_blwr_epa2.append(mse(ytest, yhat_blwr_3))
  mse_blwr_triw2.append(mse(ytest, yhat_blwr_4))
  mse_blwr_trian2.append(mse(ytest, yhat_blwr_5))

  yhat_blwr_1 = booster_lwrandchoice(xtrain,ytrain,xtest, Tricubic,1,model_boosting3,2)
  yhat_blwr_2 = booster_lwrandchoice(xtrain,ytrain,xtest, Quartic,1,model_boosting3,2)
  yhat_blwr_3 = booster_lwrandchoice(xtrain,ytrain,xtest, Epanechnikov,1,model_boosting3,2)
  yhat_blwr_4 = booster_lwrandchoice(xtrain,ytrain,xtest, Triweight,1,model_boosting3,2)
  yhat_blwr_5 = booster_lwrandchoice(xtrain,ytrain,xtest, Triangular,1,model_boosting3,2)

  mse_blwr_tric3.append(mse(ytest, yhat_blwr_1))
  mse_blwr_quart3.append(mse(ytest, yhat_blwr_2))
  mse_blwr_epa3.append(mse(ytest, yhat_blwr_3))
  mse_blwr_triw3.append(mse(ytest, yhat_blwr_4))
  mse_blwr_trian3.append(mse(ytest, yhat_blwr_5))


  yhat_blwr_1 = booster_lwrandchoice(xtrain,ytrain,xtest, Tricubic,1,model_boosting4,2)
  yhat_blwr_2 = booster_lwrandchoice(xtrain,ytrain,xtest, Quartic,1,model_boosting4,2)
  yhat_blwr_3 = booster_lwrandchoice(xtrain,ytrain,xtest, Epanechnikov,1,model_boosting4,2)
  yhat_blwr_4 = booster_lwrandchoice(xtrain,ytrain,xtest, Triweight,1,model_boosting4,2)
  yhat_blwr_5 = booster_lwrandchoice(xtrain,ytrain,xtest, Triangular,1,model_boosting4,2)

  mse_blwr_tric4.append(mse(ytest, yhat_blwr_1))
  mse_blwr_quart4.append(mse(ytest, yhat_blwr_2))
  mse_blwr_epa4.append(mse(ytest, yhat_blwr_3))
  mse_blwr_triw4.append(mse(ytest, yhat_blwr_4))
  mse_blwr_trian4.append(mse(ytest, yhat_blwr_5))

  yhat_blwr_1 = booster_lwrandchoice(xtrain,ytrain,xtest, Tricubic,1,model_boosting5,2)
  yhat_blwr_2 = booster_lwrandchoice(xtrain,ytrain,xtest, Quartic,1,model_boosting5,2)
  yhat_blwr_3 = booster_lwrandchoice(xtrain,ytrain,xtest, Epanechnikov,1,model_boosting5,2)
  yhat_blwr_4 = booster_lwrandchoice(xtrain,ytrain,xtest, Triweight,1,model_boosting5,2)
  yhat_blwr_5 = booster_lwrandchoice(xtrain,ytrain,xtest, Triangular,1,model_boosting5,2)

  mse_blwr_tric5.append(mse(ytest, yhat_blwr_1))
  mse_blwr_quart5.append(mse(ytest, yhat_blwr_2))
  mse_blwr_epa5.append(mse(ytest, yhat_blwr_3))
  mse_blwr_triw5.append(mse(ytest, yhat_blwr_4))
  mse_blwr_trian5.append(mse(ytest, yhat_blwr_5))

  yhat_blwr_1 = booster_lwrandchoice(xtrain,ytrain,xtest, Tricubic,1,model_boosting6,2)
  yhat_blwr_2 = booster_lwrandchoice(xtrain,ytrain,xtest, Quartic,1,model_boosting6,2)
  yhat_blwr_3 = booster_lwrandchoice(xtrain,ytrain,xtest, Epanechnikov,1,model_boosting6,2)
  yhat_blwr_4 = booster_lwrandchoice(xtrain,ytrain,xtest, Triweight,1,model_boosting6,2)
  yhat_blwr_5 = booster_lwrandchoice(xtrain,ytrain,xtest, Triangular,1,model_boosting6,2)

  mse_blwr_tric6.append(mse(ytest, yhat_blwr_1))
  mse_blwr_quart6.append(mse(ytest, yhat_blwr_2))
  mse_blwr_epa6.append(mse(ytest, yhat_blwr_3))
  mse_blwr_triw6.append(mse(ytest, yhat_blwr_4))
  mse_blwr_trian6.append(mse(ytest, yhat_blwr_5))


print('The Cross-validated Mean Squared Error for Boosted LWR with tricubic is : '+str(np.mean(mse_blwr_tric1)))
print('The Cross-validated Mean Squared Error for Boosted LWR with quartic is : '+str(np.mean(mse_blwr_quart1)))
print('The Cross-validated Mean Squared Error for Boosted LWR with Epanechnikov is : '+str(np.mean(mse_blwr_epa1)))
print('The Cross-validated Mean Squared Error for Boosted LWR with triweight is : '+str(np.mean(mse_blwr_triw1)))
print('The Cross-validated Mean Squared Error for Boosted LWR with triangular is : '+str(np.mean(mse_blwr_trian1)))
print('\n')
print('The Cross-validated Mean Squared Error for Boosted LWR with tricubic is  : '+str(np.mean(mse_blwr_tric2)))
print('The Cross-validated Mean Squared Error for Boosted LWR with quartic is : '+str(np.mean(mse_blwr_quart2)))
print('The Cross-validated Mean Squared Error for Boosted LWR with Epanechnikov is : '+str(np.mean(mse_blwr_epa2)))
print('The Cross-validated Mean Squared Error for Boosted LWR with triweight is : '+str(np.mean(mse_blwr_triw2)))
print('The Cross-validated Mean Squared Error for Boosted LWR with triangular is : '+str(np.mean(mse_blwr_trian2)))
print('\n')
print('The Cross-validated Mean Squared Error for Boosted LWR with tricubic is : '+str(np.mean(mse_blwr_tric3)))
print('The Cross-validated Mean Squared Error for Boosted LWR with quartic is : '+str(np.mean(mse_blwr_quart3)))
print('The Cross-validated Mean Squared Error for Boosted LWR with Epanechnikov is : '+str(np.mean(mse_blwr_epa3)))
print('The Cross-validated Mean Squared Error for Boosted LWR with triweight is : '+str(np.mean(mse_blwr_triw3)))
print('The Cross-validated Mean Squared Error for Boosted LWR with triangular is : '+str(np.mean(mse_blwr_trian3)))
print('\n')
print('The Cross-validated Mean Squared Error for Boosted LWR with tricubic is : '+str(np.mean(mse_blwr_tric4)))
print('The Cross-validated Mean Squared Error for Boosted LWR with quartic is : '+str(np.mean(mse_blwr_quart4)))
print('The Cross-validated Mean Squared Error for Boosted LWR with Epanechnikov is : '+str(np.mean(mse_blwr_epa4)))
print('The Cross-validated Mean Squared Error for Boosted LWR with triweight is : '+str(np.mean(mse_blwr_triw4)))
print('The Cross-validated Mean Squared Error for Boosted LWR with triangular is : '+str(np.mean(mse_blwr_trian4)))
print('\n')
print('The Cross-validated Mean Squared Error for Boosted LWR with tricubic is : '+str(np.mean(mse_blwr_tric5)))
print('The Cross-validated Mean Squared Error for Boosted LWR with quartic is : '+str(np.mean(mse_blwr_quart5)))
print('The Cross-validated Mean Squared Error for Boosted LWR with Epanechnikov is : '+str(np.mean(mse_blwr_epa5)))
print('The Cross-validated Mean Squared Error for Boosted LWR with triweight is : '+str(np.mean(mse_blwr_triw5)))
print('The Cross-validated Mean Squared Error for Boosted LWR with triangular is : '+str(np.mean(mse_blwr_trian5)))
print('\n')
print('The Cross-validated Mean Squared Error for Boosted LWR with tricubic is : '+str(np.mean(mse_blwr_tric6)))
print('The Cross-validated Mean Squared Error for Boosted LWR with quartic is : '+str(np.mean(mse_blwr_quart6)))
print('The Cross-validated Mean Squared Error for Boosted LWR with Epanechnikov is : '+str(np.mean(mse_blwr_epa6)))
print('The Cross-validated Mean Squared Error for Boosted LWR with triweight is : '+str(np.mean(mse_blwr_triw6)))
print('The Cross-validated Mean Squared Error for Boosted LWR with triangular is : '+str(np.mean(mse_blwr_trian6)))

mse_blwr_triw4 = []
mse_blwr_epa4 = []
mse_blwr_quart4 = []
mse_blwr_trian4 = []

mse_blwr_triw5 = []
mse_blwr_epa5 = []
mse_blwr_quart5 = []
mse_blwr_trian5 = []


mse_blwr_triw6 = []
mse_blwr_epa6 = []
mse_blwr_quart6 = []
mse_blwr_trian6 = []




kf = KFold(n_splits=5,shuffle=True,random_state=1234)
for idxtrain, idxtest in kf.split(X):
  xtrain = X[idxtrain]
  ytrain = y[idxtrain]
  ytest = y[idxtest]
  xtest = X[idxtest]
  xtrain = scale.fit_transform(xtrain)
  xtest = scale.transform(xtest)
    

  yhat_blwr_1 = booster_lwrandchoice(xtrain,ytrain,xtest, Triweight,0.01,model_boosting4,2)
  yhat_blwr_2 = booster_lwrandchoice(xtrain,ytrain,xtest, Quartic,0.01,model_boosting4,2)
  yhat_blwr_3 = booster_lwrandchoice(xtrain,ytrain,xtest, Epanechnikov,0.01,model_boosting4,2)
  yhat_blwr_4 = booster_lwrandchoice(xtrain,ytrain,xtest, Triangular,0.01,model_boosting4,2)

  mse_blwr_triw4.append(mse(ytest, yhat_blwr_1))
  mse_blwr_quart4.append(mse(ytest, yhat_blwr_2))
  mse_blwr_epa4.append(mse(ytest, yhat_blwr_3))
  mse_blwr_trian4.append(mse(ytest, yhat_blwr_4))

  yhat_blwr_1 = booster_lwrandchoice(xtrain,ytrain,xtest, Triweight,0.01,model_boosting5,2)
  yhat_blwr_2 = booster_lwrandchoice(xtrain,ytrain,xtest, Quartic,0.01,model_boosting5,2)
  yhat_blwr_3 = booster_lwrandchoice(xtrain,ytrain,xtest, Epanechnikov,0.01,model_boosting5,2)
  yhat_blwr_4 = booster_lwrandchoice(xtrain,ytrain,xtest, Triangular,0.01,model_boosting5,2)

  mse_blwr_triw5.append(mse(ytest, yhat_blwr_1))
  mse_blwr_quart5.append(mse(ytest, yhat_blwr_2))
  mse_blwr_epa5.append(mse(ytest, yhat_blwr_3))
  mse_blwr_trian5.append(mse(ytest, yhat_blwr_4))

  yhat_blwr_1 = booster_lwrandchoice(xtrain,ytrain,xtest, Triweight,0.01,model_boosting6,2)
  yhat_blwr_2 = booster_lwrandchoice(xtrain,ytrain,xtest, Quartic,0.01,model_boosting6,2)
  yhat_blwr_3 = booster_lwrandchoice(xtrain,ytrain,xtest, Epanechnikov,0.01,model_boosting6,2)
  yhat_blwr_4 = booster_lwrandchoice(xtrain,ytrain,xtest, Triangular,0.01,model_boosting6,2)

  mse_blwr_triw6.append(mse(ytest, yhat_blwr_1))
  mse_blwr_quart6.append(mse(ytest, yhat_blwr_2))
  mse_blwr_epa6.append(mse(ytest, yhat_blwr_3))
  mse_blwr_trian6.append(mse(ytest, yhat_blwr_4))

print('The Cross-validated Mean Squared Error for Boosted LWR with quartic is : '+str(np.mean(mse_blwr_quart4)))
print('The Cross-validated Mean Squared Error for Boosted LWR with Epanechnikov is : '+str(np.mean(mse_blwr_epa4)))
print('The Cross-validated Mean Squared Error for Boosted LWR with triweight is : '+str(np.mean(mse_blwr_triw4)))
print('The Cross-validated Mean Squared Error for Boosted LWR with triangular is : '+str(np.mean(mse_blwr_trian4)))
print('\n')
print('The Cross-validated Mean Squared Error for Boosted LWR with quartic is : '+str(np.mean(mse_blwr_quart5)))
print('The Cross-validated Mean Squared Error for Boosted LWR with Epanechnikov is : '+str(np.mean(mse_blwr_epa5)))
print('The Cross-validated Mean Squared Error for Boosted LWR with triweight is : '+str(np.mean(mse_blwr_triw5)))
print('The Cross-validated Mean Squared Error for Boosted LWR with triangular is : '+str(np.mean(mse_blwr_trian5)))
print('\n')
print('The Cross-validated Mean Squared Error for Boosted LWR with quartic is : '+str(np.mean(mse_blwr_quart6)))
print('The Cross-validated Mean Squared Error for Boosted LWR with Epanechnikov is : '+str(np.mean(mse_blwr_epa6)))
print('The Cross-validated Mean Squared Error for Boosted LWR with triweight is : '+str(np.mean(mse_blwr_triw6)))
print('The Cross-validated Mean Squared Error for Boosted LWR with triangular is : '+str(np.mean(mse_blwr_trian6)))


mse_blwr_triw4 = []
mse_blwr_epa4 = []
mse_blwr_quart4 = []
mse_blwr_trian4 = []

mse_blwr_triw5 = []
mse_blwr_epa5 = []
mse_blwr_quart5 = []
mse_blwr_trian5 = []


mse_blwr_triw6 = []
mse_blwr_epa6 = []
mse_blwr_quart6 = []
mse_blwr_trian6 = []




kf = KFold(n_splits=5,shuffle=True,random_state=1234)
for idxtrain, idxtest in kf.split(X):
  xtrain = X[idxtrain]
  ytrain = y[idxtrain]
  ytest = y[idxtest]
  xtest = X[idxtest]
  xtrain = scale.fit_transform(xtrain)
  xtest = scale.transform(xtest)
    

  yhat_blwr_1 = booster_lwrandchoice(xtrain,ytrain,xtest, Triweight,0.05,model_boosting4,2)
  yhat_blwr_2 = booster_lwrandchoice(xtrain,ytrain,xtest, Quartic,0.05,model_boosting4,2)
  yhat_blwr_3 = booster_lwrandchoice(xtrain,ytrain,xtest, Epanechnikov,0.05,model_boosting4,2)
  yhat_blwr_4 = booster_lwrandchoice(xtrain,ytrain,xtest, Triangular,0.05,model_boosting4,2)

  mse_blwr_triw4.append(mse(ytest, yhat_blwr_1))
  mse_blwr_quart4.append(mse(ytest, yhat_blwr_2))
  mse_blwr_epa4.append(mse(ytest, yhat_blwr_3))
  mse_blwr_trian4.append(mse(ytest, yhat_blwr_4))

  yhat_blwr_1 = booster_lwrandchoice(xtrain,ytrain,xtest, Triweight,0.05,model_boosting5,2)
  yhat_blwr_2 = booster_lwrandchoice(xtrain,ytrain,xtest, Quartic,0.05,model_boosting5,2)
  yhat_blwr_3 = booster_lwrandchoice(xtrain,ytrain,xtest, Epanechnikov,0.05,model_boosting5,2)
  yhat_blwr_4 = booster_lwrandchoice(xtrain,ytrain,xtest, Triangular,0.05,model_boosting5,2)

  mse_blwr_triw5.append(mse(ytest, yhat_blwr_1))
  mse_blwr_quart5.append(mse(ytest, yhat_blwr_2))
  mse_blwr_epa5.append(mse(ytest, yhat_blwr_3))
  mse_blwr_trian5.append(mse(ytest, yhat_blwr_4))

  yhat_blwr_1 = booster_lwrandchoice(xtrain,ytrain,xtest, Triweight,0.05,model_boosting6,2)
  yhat_blwr_2 = booster_lwrandchoice(xtrain,ytrain,xtest, Quartic,0.05,model_boosting6,2)
  yhat_blwr_3 = booster_lwrandchoice(xtrain,ytrain,xtest, Epanechnikov,0.05,model_boosting6,2)
  yhat_blwr_4 = booster_lwrandchoice(xtrain,ytrain,xtest, Triangular,0.05,model_boosting6,2)

  mse_blwr_triw6.append(mse(ytest, yhat_blwr_1))
  mse_blwr_quart6.append(mse(ytest, yhat_blwr_2))
  mse_blwr_epa6.append(mse(ytest, yhat_blwr_3))
  mse_blwr_trian6.append(mse(ytest, yhat_blwr_4))

print('The Cross-validated Mean Squared Error for Boosted LWR with quartic is : '+str(np.mean(mse_blwr_quart4)))
print('The Cross-validated Mean Squared Error for Boosted LWR with Epanechnikov is : '+str(np.mean(mse_blwr_epa4)))
print('The Cross-validated Mean Squared Error for Boosted LWR with triweight is : '+str(np.mean(mse_blwr_triw4)))
print('The Cross-validated Mean Squared Error for Boosted LWR with triangular is : '+str(np.mean(mse_blwr_trian4)))
print('\n')
print('The Cross-validated Mean Squared Error for Boosted LWR with quartic is : '+str(np.mean(mse_blwr_quart5)))
print('The Cross-validated Mean Squared Error for Boosted LWR with Epanechnikov is : '+str(np.mean(mse_blwr_epa5)))
print('The Cross-validated Mean Squared Error for Boosted LWR with triweight is : '+str(np.mean(mse_blwr_triw5)))
print('The Cross-validated Mean Squared Error for Boosted LWR with triangular is : '+str(np.mean(mse_blwr_trian5)))
print('\n')
print('The Cross-validated Mean Squared Error for Boosted LWR with quartic is : '+str(np.mean(mse_blwr_quart6)))
print('The Cross-validated Mean Squared Error for Boosted LWR with Epanechnikov is : '+str(np.mean(mse_blwr_epa6)))
print('The Cross-validated Mean Squared Error for Boosted LWR with triweight is : '+str(np.mean(mse_blwr_triw6)))
print('The Cross-validated Mean Squared Error for Boosted LWR with triangular is : '+str(np.mean(mse_blwr_trian6)))


mse_blwr_triw4 = []
mse_blwr_epa4 = []
mse_blwr_quart4 = []
mse_blwr_trian4 = []

mse_blwr_triw5 = []
mse_blwr_epa5 = []
mse_blwr_quart5 = []
mse_blwr_trian5 = []


mse_blwr_triw6 = []
mse_blwr_epa6 = []
mse_blwr_quart6 = []
mse_blwr_trian6 = []




kf = KFold(n_splits=5,shuffle=True,random_state=1234)
for idxtrain, idxtest in kf.split(X):
  xtrain = X[idxtrain]
  ytrain = y[idxtrain]
  ytest = y[idxtest]
  xtest = X[idxtest]
  xtrain = scale.fit_transform(xtrain)
  xtest = scale.transform(xtest)
    

  yhat_blwr_1 = booster_lwrandchoice(xtrain,ytrain,xtest, Triweight,0.1,model_boosting4,2)
  yhat_blwr_2 = booster_lwrandchoice(xtrain,ytrain,xtest, Quartic,0.1,model_boosting4,2)
  yhat_blwr_3 = booster_lwrandchoice(xtrain,ytrain,xtest, Epanechnikov,0.1,model_boosting4,2)
  yhat_blwr_4 = booster_lwrandchoice(xtrain,ytrain,xtest, Triangular,0.1,model_boosting4,2)

  mse_blwr_triw4.append(mse(ytest, yhat_blwr_1))
  mse_blwr_quart4.append(mse(ytest, yhat_blwr_2))
  mse_blwr_epa4.append(mse(ytest, yhat_blwr_3))
  mse_blwr_trian4.append(mse(ytest, yhat_blwr_4))

  yhat_blwr_1 = booster_lwrandchoice(xtrain,ytrain,xtest, Triweight,0.1,model_boosting5,2)
  yhat_blwr_2 = booster_lwrandchoice(xtrain,ytrain,xtest, Quartic,0.1,model_boosting5,2)
  yhat_blwr_3 = booster_lwrandchoice(xtrain,ytrain,xtest, Epanechnikov,0.1,model_boosting5,2)
  yhat_blwr_4 = booster_lwrandchoice(xtrain,ytrain,xtest, Triangular,0.1,model_boosting5,2)

  mse_blwr_triw5.append(mse(ytest, yhat_blwr_1))
  mse_blwr_quart5.append(mse(ytest, yhat_blwr_2))
  mse_blwr_epa5.append(mse(ytest, yhat_blwr_3))
  mse_blwr_trian5.append(mse(ytest, yhat_blwr_4))

  yhat_blwr_1 = booster_lwrandchoice(xtrain,ytrain,xtest, Triweight,0.1,model_boosting6,2)
  yhat_blwr_2 = booster_lwrandchoice(xtrain,ytrain,xtest, Quartic,0.1,model_boosting6,2)
  yhat_blwr_3 = booster_lwrandchoice(xtrain,ytrain,xtest, Epanechnikov,0.1,model_boosting6,2)
  yhat_blwr_4 = booster_lwrandchoice(xtrain,ytrain,xtest, Triangular,0.1,model_boosting6,2)

  mse_blwr_triw6.append(mse(ytest, yhat_blwr_1))
  mse_blwr_quart6.append(mse(ytest, yhat_blwr_2))
  mse_blwr_epa6.append(mse(ytest, yhat_blwr_3))
  mse_blwr_trian6.append(mse(ytest, yhat_blwr_4))

print('The Cross-validated Mean Squared Error for Boosted LWR with quartic is : '+str(np.mean(mse_blwr_quart4)))
print('The Cross-validated Mean Squared Error for Boosted LWR with Epanechnikov is : '+str(np.mean(mse_blwr_epa4)))
print('The Cross-validated Mean Squared Error for Boosted LWR with triweight is : '+str(np.mean(mse_blwr_triw4)))
print('The Cross-validated Mean Squared Error for Boosted LWR with triangular is : '+str(np.mean(mse_blwr_trian4)))
print('\n')
print('The Cross-validated Mean Squared Error for Boosted LWR with quartic is : '+str(np.mean(mse_blwr_quart5)))
print('The Cross-validated Mean Squared Error for Boosted LWR with Epanechnikov is : '+str(np.mean(mse_blwr_epa5)))
print('The Cross-validated Mean Squared Error for Boosted LWR with triweight is : '+str(np.mean(mse_blwr_triw5)))
print('The Cross-validated Mean Squared Error for Boosted LWR with triangular is : '+str(np.mean(mse_blwr_trian5)))
print('\n')
print('The Cross-validated Mean Squared Error for Boosted LWR with quartic is : '+str(np.mean(mse_blwr_quart6)))
print('The Cross-validated Mean Squared Error for Boosted LWR with Epanechnikov is : '+str(np.mean(mse_blwr_epa6)))
print('The Cross-validated Mean Squared Error for Boosted LWR with triweight is : '+str(np.mean(mse_blwr_triw6)))
print('The Cross-validated Mean Squared Error for Boosted LWR with triangular is : '+str(np.mean(mse_blwr_trian6)))



mse_blwr_triw4 = []
mse_blwr_epa4 = []
mse_blwr_quart4 = []
mse_blwr_trian4 = []

mse_blwr_triw5 = []
mse_blwr_epa5 = []
mse_blwr_quart5 = []
mse_blwr_trian5 = []


mse_blwr_triw6 = []
mse_blwr_epa6 = []
mse_blwr_quart6 = []
mse_blwr_trian6 = []




kf = KFold(n_splits=5,shuffle=True,random_state=1234)
for idxtrain, idxtest in kf.split(X):
  xtrain = X[idxtrain]
  ytrain = y[idxtrain]
  ytest = y[idxtest]
  xtest = X[idxtest]
  xtrain = scale.fit_transform(xtrain)
  xtest = scale.transform(xtest)
    

  yhat_blwr_1 = booster_lwrandchoice(xtrain,ytrain,xtest, Triweight,0.5,model_boosting4,2)
  yhat_blwr_2 = booster_lwrandchoice(xtrain,ytrain,xtest, Quartic,0.5,model_boosting4,2)
  yhat_blwr_3 = booster_lwrandchoice(xtrain,ytrain,xtest, Epanechnikov,0.5,model_boosting4,2)
  yhat_blwr_4 = booster_lwrandchoice(xtrain,ytrain,xtest, Triangular,0.5,model_boosting4,2)

  mse_blwr_triw4.append(mse(ytest, yhat_blwr_1))
  mse_blwr_quart4.append(mse(ytest, yhat_blwr_2))
  mse_blwr_epa4.append(mse(ytest, yhat_blwr_3))
  mse_blwr_trian4.append(mse(ytest, yhat_blwr_4))

  yhat_blwr_1 = booster_lwrandchoice(xtrain,ytrain,xtest, Triweight,0.5,model_boosting5,2)
  yhat_blwr_2 = booster_lwrandchoice(xtrain,ytrain,xtest, Quartic,0.5,model_boosting5,2)
  yhat_blwr_3 = booster_lwrandchoice(xtrain,ytrain,xtest, Epanechnikov,0.5,model_boosting5,2)
  yhat_blwr_4 = booster_lwrandchoice(xtrain,ytrain,xtest, Triangular,0.5,model_boosting5,2)

  mse_blwr_triw5.append(mse(ytest, yhat_blwr_1))
  mse_blwr_quart5.append(mse(ytest, yhat_blwr_2))
  mse_blwr_epa5.append(mse(ytest, yhat_blwr_3))
  mse_blwr_trian5.append(mse(ytest, yhat_blwr_4))

  yhat_blwr_1 = booster_lwrandchoice(xtrain,ytrain,xtest, Triweight,0.5,model_boosting6,2)
  yhat_blwr_2 = booster_lwrandchoice(xtrain,ytrain,xtest, Quartic,0.5,model_boosting6,2)
  yhat_blwr_3 = booster_lwrandchoice(xtrain,ytrain,xtest, Epanechnikov,0.5,model_boosting6,2)
  yhat_blwr_4 = booster_lwrandchoice(xtrain,ytrain,xtest, Triangular,0.5,model_boosting6,2)

  mse_blwr_triw6.append(mse(ytest, yhat_blwr_1))
  mse_blwr_quart6.append(mse(ytest, yhat_blwr_2))
  mse_blwr_epa6.append(mse(ytest, yhat_blwr_3))
  mse_blwr_trian6.append(mse(ytest, yhat_blwr_4))

print('The Cross-validated Mean Squared Error for Boosted LWR with quartic is : '+str(np.mean(mse_blwr_quart4)))
print('The Cross-validated Mean Squared Error for Boosted LWR with Epanechnikov is : '+str(np.mean(mse_blwr_epa4)))
print('The Cross-validated Mean Squared Error for Boosted LWR with triweight is : '+str(np.mean(mse_blwr_triw4)))
print('The Cross-validated Mean Squared Error for Boosted LWR with triangular is : '+str(np.mean(mse_blwr_trian4)))
print('\n')
print('The Cross-validated Mean Squared Error for Boosted LWR with quartic is : '+str(np.mean(mse_blwr_quart5)))
print('The Cross-validated Mean Squared Error for Boosted LWR with Epanechnikov is : '+str(np.mean(mse_blwr_epa5)))
print('The Cross-validated Mean Squared Error for Boosted LWR with triweight is : '+str(np.mean(mse_blwr_triw5)))
print('The Cross-validated Mean Squared Error for Boosted LWR with triangular is : '+str(np.mean(mse_blwr_trian5)))
print('\n')
print('The Cross-validated Mean Squared Error for Boosted LWR with quartic is : '+str(np.mean(mse_blwr_quart6)))
print('The Cross-validated Mean Squared Error for Boosted LWR with Epanechnikov is : '+str(np.mean(mse_blwr_epa6)))
print('The Cross-validated Mean Squared Error for Boosted LWR with triweight is : '+str(np.mean(mse_blwr_triw6)))
print('The Cross-validated Mean Squared Error for Boosted LWR with triangular is : '+str(np.mean(mse_blwr_trian6)))



# New set of LightGBM models to boost
model_boosting1 = lgb.LGBMRegressor(learning_rate=0.05, n_estimators=5, max_depth=2)
model_boosting2 = lgb.LGBMRegressor(learning_rate=0.05, n_estimators=5, max_depth=3)
model_boosting3 = lgb.LGBMRegressor(learning_rate=0.05, n_estimators=10, max_depth=2)
model_boosting4 = lgb.LGBMRegressor(learning_rate=0.05, n_estimators=10, max_depth=3)
model_boosting5 = lgb.LGBMRegressor(learning_rate=0.1, n_estimators=5, max_depth=2)
model_boosting6 = lgb.LGBMRegressor(learning_rate=0.1, n_estimators=5, max_depth=3)
model_boosting7 = lgb.LGBMRegressor(learning_rate=0.1, n_estimators=10, max_depth=2)
model_boosting8 = lgb.LGBMRegressor(learning_rate=0.1, n_estimators=10, max_depth=3)

mse_blwr_tric1 = []
mse_blwr_quart1 = []
mse_blwr_epa1 = []
mse_blwr_triw1 = []
mse_blwr_trian1 = []

mse_blwr_tric2 = []
mse_blwr_quart2 = []
mse_blwr_epa2 = []
mse_blwr_triw2 = []
mse_blwr_trian2 = []

mse_blwr_tric3 = []
mse_blwr_quart3 = []
mse_blwr_epa3 = []
mse_blwr_triw3 = []
mse_blwr_trian3 = []

mse_blwr_tric4 = []
mse_blwr_quart4 = []
mse_blwr_epa4 = []
mse_blwr_triw4 = []
mse_blwr_trian4 = []

mse_blwr_tric5 = []
mse_blwr_quart5 = []
mse_blwr_epa5 = []
mse_blwr_triw5 = []
mse_blwr_trian5 = []

mse_blwr_tric6 = []
mse_blwr_quart6 = []
mse_blwr_epa6 = []
mse_blwr_triw6 = []
mse_blwr_trian6 = []

mse_blwr_tric7 = []
mse_blwr_quart7 = []
mse_blwr_epa7 = []
mse_blwr_triw7 = []
mse_blwr_trian7 = []

mse_blwr_tric8 = []
mse_blwr_quart8 = []
mse_blwr_epa8 = []
mse_blwr_triw8 = []
mse_blwr_trian8 = []


kf = KFold(n_splits=3,shuffle=True,random_state=1234)
for idxtrain, idxtest in kf.split(X):
  xtrain = X[idxtrain]
  ytrain = y[idxtrain]
  ytest = y[idxtest]
  xtest = X[idxtest]
  xtrain = scale.fit_transform(xtrain)
  xtest = scale.transform(xtest)
    
  yhat_blwr_1 = booster_lwrandchoice(xtrain,ytrain,xtest, Tricubic,1,model_boosting1,2)
  yhat_blwr_2 = booster_lwrandchoice(xtrain,ytrain,xtest, Quartic,1,model_boosting1,2)
  yhat_blwr_3 = booster_lwrandchoice(xtrain,ytrain,xtest, Epanechnikov,1,model_boosting1,2)
  yhat_blwr_4 = booster_lwrandchoice(xtrain,ytrain,xtest, Triweight,1,model_boosting1,2)
  yhat_blwr_5 = booster_lwrandchoice(xtrain,ytrain,xtest, Triangular,1,model_boosting1,2)

  mse_blwr_tric1.append(mse(ytest, yhat_blwr_1))
  mse_blwr_quart1.append(mse(ytest, yhat_blwr_2))
  mse_blwr_epa1.append(mse(ytest, yhat_blwr_3))
  mse_blwr_triw1.append(mse(ytest, yhat_blwr_4))
  mse_blwr_trian1.append(mse(ytest, yhat_blwr_5))

  yhat_blwr_1 = booster_lwrandchoice(xtrain,ytrain,xtest, Tricubic,1,model_boosting2,2)
  yhat_blwr_2 = booster_lwrandchoice(xtrain,ytrain,xtest, Quartic,1,model_boosting2,2)
  yhat_blwr_3 = booster_lwrandchoice(xtrain,ytrain,xtest, Epanechnikov,1,model_boosting2,2)
  yhat_blwr_4 = booster_lwrandchoice(xtrain,ytrain,xtest, Triweight,1,model_boosting2,2)
  yhat_blwr_5 = booster_lwrandchoice(xtrain,ytrain,xtest, Triangular,1,model_boosting2,2)

  mse_blwr_tric2.append(mse(ytest, yhat_blwr_1))
  mse_blwr_quart2.append(mse(ytest, yhat_blwr_2))
  mse_blwr_epa2.append(mse(ytest, yhat_blwr_3))
  mse_blwr_triw2.append(mse(ytest, yhat_blwr_4))
  mse_blwr_trian2.append(mse(ytest, yhat_blwr_5))

  yhat_blwr_1 = booster_lwrandchoice(xtrain,ytrain,xtest, Tricubic,1,model_boosting3,2)
  yhat_blwr_2 = booster_lwrandchoice(xtrain,ytrain,xtest, Quartic,1,model_boosting3,2)
  yhat_blwr_3 = booster_lwrandchoice(xtrain,ytrain,xtest, Epanechnikov,1,model_boosting3,2)
  yhat_blwr_4 = booster_lwrandchoice(xtrain,ytrain,xtest, Triweight,1,model_boosting3,2)
  yhat_blwr_5 = booster_lwrandchoice(xtrain,ytrain,xtest, Triangular,1,model_boosting3,2)

  mse_blwr_tric3.append(mse(ytest, yhat_blwr_1))
  mse_blwr_quart3.append(mse(ytest, yhat_blwr_2))
  mse_blwr_epa3.append(mse(ytest, yhat_blwr_3))
  mse_blwr_triw3.append(mse(ytest, yhat_blwr_4))
  mse_blwr_trian3.append(mse(ytest, yhat_blwr_5))


  yhat_blwr_1 = booster_lwrandchoice(xtrain,ytrain,xtest, Tricubic,1,model_boosting4,2)
  yhat_blwr_2 = booster_lwrandchoice(xtrain,ytrain,xtest, Quartic,1,model_boosting4,2)
  yhat_blwr_3 = booster_lwrandchoice(xtrain,ytrain,xtest, Epanechnikov,1,model_boosting4,2)
  yhat_blwr_4 = booster_lwrandchoice(xtrain,ytrain,xtest, Triweight,1,model_boosting4,2)
  yhat_blwr_5 = booster_lwrandchoice(xtrain,ytrain,xtest, Triangular,1,model_boosting4,2)

  mse_blwr_tric4.append(mse(ytest, yhat_blwr_1))
  mse_blwr_quart4.append(mse(ytest, yhat_blwr_2))
  mse_blwr_epa4.append(mse(ytest, yhat_blwr_3))
  mse_blwr_triw4.append(mse(ytest, yhat_blwr_4))
  mse_blwr_trian4.append(mse(ytest, yhat_blwr_5))

  yhat_blwr_1 = booster_lwrandchoice(xtrain,ytrain,xtest, Tricubic,1,model_boosting5,2)
  yhat_blwr_2 = booster_lwrandchoice(xtrain,ytrain,xtest, Quartic,1,model_boosting5,2)
  yhat_blwr_3 = booster_lwrandchoice(xtrain,ytrain,xtest, Epanechnikov,1,model_boosting5,2)
  yhat_blwr_4 = booster_lwrandchoice(xtrain,ytrain,xtest, Triweight,1,model_boosting5,2)
  yhat_blwr_5 = booster_lwrandchoice(xtrain,ytrain,xtest, Triangular,1,model_boosting5,2)

  mse_blwr_tric5.append(mse(ytest, yhat_blwr_1))
  mse_blwr_quart5.append(mse(ytest, yhat_blwr_2))
  mse_blwr_epa5.append(mse(ytest, yhat_blwr_3))
  mse_blwr_triw5.append(mse(ytest, yhat_blwr_4))
  mse_blwr_trian5.append(mse(ytest, yhat_blwr_5))

  yhat_blwr_1 = booster_lwrandchoice(xtrain,ytrain,xtest, Tricubic,1,model_boosting6,2)
  yhat_blwr_2 = booster_lwrandchoice(xtrain,ytrain,xtest, Quartic,1,model_boosting6,2)
  yhat_blwr_3 = booster_lwrandchoice(xtrain,ytrain,xtest, Epanechnikov,1,model_boosting6,2)
  yhat_blwr_4 = booster_lwrandchoice(xtrain,ytrain,xtest, Triweight,1,model_boosting6,2)
  yhat_blwr_5 = booster_lwrandchoice(xtrain,ytrain,xtest, Triangular,1,model_boosting6,2)

  mse_blwr_tric6.append(mse(ytest, yhat_blwr_1))
  mse_blwr_quart6.append(mse(ytest, yhat_blwr_2))
  mse_blwr_epa6.append(mse(ytest, yhat_blwr_3))
  mse_blwr_triw6.append(mse(ytest, yhat_blwr_4))
  mse_blwr_trian6.append(mse(ytest, yhat_blwr_5))

  yhat_blwr_1 = booster_lwrandchoice(xtrain,ytrain,xtest, Tricubic,1,model_boosting7,2)
  yhat_blwr_2 = booster_lwrandchoice(xtrain,ytrain,xtest, Quartic,1,model_boosting7,2)
  yhat_blwr_3 = booster_lwrandchoice(xtrain,ytrain,xtest, Epanechnikov,1,model_boosting7,2)
  yhat_blwr_4 = booster_lwrandchoice(xtrain,ytrain,xtest, Triweight,1,model_boosting7,2)
  yhat_blwr_5 = booster_lwrandchoice(xtrain,ytrain,xtest, Triangular,1,model_boosting7,2)

  mse_blwr_tric7.append(mse(ytest, yhat_blwr_1))
  mse_blwr_quart7.append(mse(ytest, yhat_blwr_2))
  mse_blwr_epa7.append(mse(ytest, yhat_blwr_3))
  mse_blwr_triw7.append(mse(ytest, yhat_blwr_4))
  mse_blwr_trian7.append(mse(ytest, yhat_blwr_5))

  yhat_blwr_1 = booster_lwrandchoice(xtrain,ytrain,xtest, Tricubic,1,model_boosting8,2)
  yhat_blwr_2 = booster_lwrandchoice(xtrain,ytrain,xtest, Quartic,1,model_boosting8,2)
  yhat_blwr_3 = booster_lwrandchoice(xtrain,ytrain,xtest, Epanechnikov,1,model_boosting8,2)
  yhat_blwr_4 = booster_lwrandchoice(xtrain,ytrain,xtest, Triweight,1,model_boosting8,2)
  yhat_blwr_5 = booster_lwrandchoice(xtrain,ytrain,xtest, Triangular,1,model_boosting8,2)

  mse_blwr_tric8.append(mse(ytest, yhat_blwr_1))
  mse_blwr_quart8.append(mse(ytest, yhat_blwr_2))
  mse_blwr_epa8.append(mse(ytest, yhat_blwr_3))
  mse_blwr_triw8.append(mse(ytest, yhat_blwr_4))
  mse_blwr_trian8.append(mse(ytest, yhat_blwr_5))

print('The Cross-validated Mean Squared Error for Boosted LWR with tricubic is : '+str(np.mean(mse_blwr_tric1)))
print('The Cross-validated Mean Squared Error for Boosted LWR with quartic is : '+str(np.mean(mse_blwr_quart1)))
print('The Cross-validated Mean Squared Error for Boosted LWR with Epanechnikov is : '+str(np.mean(mse_blwr_epa1)))
print('The Cross-validated Mean Squared Error for Boosted LWR with triweight is : '+str(np.mean(mse_blwr_triw1)))
print('The Cross-validated Mean Squared Error for Boosted LWR with triangular is : '+str(np.mean(mse_blwr_trian1)))
print('\n')
print('The Cross-validated Mean Squared Error for Boosted LWR with tricubic is  : '+str(np.mean(mse_blwr_tric2)))
print('The Cross-validated Mean Squared Error for Boosted LWR with quartic is : '+str(np.mean(mse_blwr_quart2)))
print('The Cross-validated Mean Squared Error for Boosted LWR with Epanechnikov is : '+str(np.mean(mse_blwr_epa2)))
print('The Cross-validated Mean Squared Error for Boosted LWR with triweight is : '+str(np.mean(mse_blwr_triw2)))
print('The Cross-validated Mean Squared Error for Boosted LWR with triangular is : '+str(np.mean(mse_blwr_trian2)))
print('\n')
print('The Cross-validated Mean Squared Error for Boosted LWR with tricubic is : '+str(np.mean(mse_blwr_tric3)))
print('The Cross-validated Mean Squared Error for Boosted LWR with quartic is : '+str(np.mean(mse_blwr_quart3)))
print('The Cross-validated Mean Squared Error for Boosted LWR with Epanechnikov is : '+str(np.mean(mse_blwr_epa3)))
print('The Cross-validated Mean Squared Error for Boosted LWR with triweight is : '+str(np.mean(mse_blwr_triw3)))
print('The Cross-validated Mean Squared Error for Boosted LWR with triangular is : '+str(np.mean(mse_blwr_trian3)))
print('\n')
print('The Cross-validated Mean Squared Error for Boosted LWR with tricubic is : '+str(np.mean(mse_blwr_tric4)))
print('The Cross-validated Mean Squared Error for Boosted LWR with quartic is : '+str(np.mean(mse_blwr_quart4)))
print('The Cross-validated Mean Squared Error for Boosted LWR with Epanechnikov is : '+str(np.mean(mse_blwr_epa4)))
print('The Cross-validated Mean Squared Error for Boosted LWR with triweight is : '+str(np.mean(mse_blwr_triw4)))
print('The Cross-validated Mean Squared Error for Boosted LWR with triangular is : '+str(np.mean(mse_blwr_trian4)))
print('\n')
print('The Cross-validated Mean Squared Error for Boosted LWR with tricubic is : '+str(np.mean(mse_blwr_tric5)))
print('The Cross-validated Mean Squared Error for Boosted LWR with quartic is : '+str(np.mean(mse_blwr_quart5)))
print('The Cross-validated Mean Squared Error for Boosted LWR with Epanechnikov is : '+str(np.mean(mse_blwr_epa5)))
print('The Cross-validated Mean Squared Error for Boosted LWR with triweight is : '+str(np.mean(mse_blwr_triw5)))
print('The Cross-validated Mean Squared Error for Boosted LWR with triangular is : '+str(np.mean(mse_blwr_trian5)))
print('\n')
print('The Cross-validated Mean Squared Error for Boosted LWR with tricubic is : '+str(np.mean(mse_blwr_tric6)))
print('The Cross-validated Mean Squared Error for Boosted LWR with quartic is : '+str(np.mean(mse_blwr_quart6)))
print('The Cross-validated Mean Squared Error for Boosted LWR with Epanechnikov is : '+str(np.mean(mse_blwr_epa6)))
print('The Cross-validated Mean Squared Error for Boosted LWR with triweight is : '+str(np.mean(mse_blwr_triw6)))
print('The Cross-validated Mean Squared Error for Boosted LWR with triangular is : '+str(np.mean(mse_blwr_trian6)))
print('\n')
print('The Cross-validated Mean Squared Error for Boosted LWR with tricubic is : '+str(np.mean(mse_blwr_tric7)))
print('The Cross-validated Mean Squared Error for Boosted LWR with quartic is : '+str(np.mean(mse_blwr_quart7)))
print('The Cross-validated Mean Squared Error for Boosted LWR with Epanechnikov is : '+str(np.mean(mse_blwr_epa7)))
print('The Cross-validated Mean Squared Error for Boosted LWR with triweight is : '+str(np.mean(mse_blwr_triw7)))
print('The Cross-validated Mean Squared Error for Boosted LWR with triangular is : '+str(np.mean(mse_blwr_trian7)))
print('\n')
print('The Cross-validated Mean Squared Error for Boosted LWR with tricubic is : '+str(np.mean(mse_blwr_tric8)))
print('The Cross-validated Mean Squared Error for Boosted LWR with quartic is : '+str(np.mean(mse_blwr_quart8)))
print('The Cross-validated Mean Squared Error for Boosted LWR with Epanechikov is : '+str(np.mean(mse_blwr_epa8)))
print('The Cross-validated Mean Squared Error for Boosted LWR with triweight is : '+str(np.mean(mse_blwr_triw8)))
print('The Cross-validated Mean Squared Error for Boosted LWR with triangular is : '+str(np.mean(mse_blwr_trian8)))

mse_blwr_tric4 = []
mse_blwr_quart4 = []


mse_blwr_tric6 = []
mse_blwr_tric6 = []


mse_blwr_tric8 = []
mse_blwr_quart8 = []



kf = KFold(n_splits=5,shuffle=True,random_state=1234)
for idxtrain, idxtest in kf.split(X):
  xtrain = X[idxtrain]
  ytrain = y[idxtrain]
  ytest = y[idxtest]
  xtest = X[idxtest]
  xtrain = scale.fit_transform(xtrain)
  xtest = scale.transform(xtest)

  yhat_blwr_1 = booster_lwrandchoice(xtrain,ytrain,xtest, Tricubic,0.01,model_boosting4,2)
  yhat_blwr_2 = booster_lwrandchoice(xtrain,ytrain,xtest, Quartic,0.01,model_boosting4,2)

  mse_blwr_tric4.append(mse(ytest, yhat_blwr_1))
  mse_blwr_quart4.append(mse(ytest, yhat_blwr_2))

  yhat_blwr_1 = booster_lwrandchoice(xtrain,ytrain,xtest, Tricubic,0.01,model_boosting6,2)
  yhat_blwr_2 = booster_lwrandchoice(xtrain,ytrain,xtest, Quartic,0.01,model_boosting6,2)

  mse_blwr_tric6.append(mse(ytest, yhat_blwr_1))
  mse_blwr_quart6.append(mse(ytest, yhat_blwr_2))

  yhat_blwr_1 = booster_lwrandchoice(xtrain,ytrain,xtest, Tricubic,0.01,model_boosting8,2)
  yhat_blwr_2 = booster_lwrandchoice(xtrain,ytrain,xtest, Quartic,0.01,model_boosting8,2)

  mse_blwr_tric8.append(mse(ytest, yhat_blwr_1))
  mse_blwr_quart8.append(mse(ytest, yhat_blwr_2))

print('The Cross-validated Mean Squared Error for Boosted LWR with tricubic is : '+str(np.mean(mse_blwr_tric4)))
print('The Cross-validated Mean Squared Error for Boosted LWR with quartic is : '+str(np.mean(mse_blwr_quart4)))
print('\n')
print('The Cross-validated Mean Squared Error for Boosted with tricubic LWR is : '+str(np.mean(mse_blwr_tric6)))
print('The Cross-validated Mean Squared Error for Boosted with quartic LWR is : '+str(np.mean(mse_blwr_quart6)))
print('\n')
print('The Cross-validated Mean Squared Error for Boosted LWR tricubic is : '+str(np.mean(mse_blwr_tric6)))
print('The Cross-validated Mean Squared Error for Boosted LWR quartic is : '+str(np.mean(mse_blwr_quart6)))

mse_blwr_tric4 = []
mse_blwr_quart4 = []


mse_blwr_tric6 = []
mse_blwr_tric6 = []


mse_blwr_tric8 = []
mse_blwr_quart8 = []



kf = KFold(n_splits=5,shuffle=True,random_state=1234)
for idxtrain, idxtest in kf.split(X):
  xtrain = X[idxtrain]
  ytrain = y[idxtrain]
  ytest = y[idxtest]
  xtest = X[idxtest]
  xtrain = scale.fit_transform(xtrain)
  xtest = scale.transform(xtest)

  yhat_blwr_1 = booster_lwrandchoice(xtrain,ytrain,xtest, Tricubic,0.05,model_boosting4,2)
  yhat_blwr_2 = booster_lwrandchoice(xtrain,ytrain,xtest, Quartic,0.05,model_boosting4,2)

  mse_blwr_tric4.append(mse(ytest, yhat_blwr_1))
  mse_blwr_quart4.append(mse(ytest, yhat_blwr_2))

  yhat_blwr_1 = booster_lwrandchoice(xtrain,ytrain,xtest, Tricubic,0.05,model_boosting6,2)
  yhat_blwr_2 = booster_lwrandchoice(xtrain,ytrain,xtest, Quartic,0.05,model_boosting6,2)

  mse_blwr_tric6.append(mse(ytest, yhat_blwr_1))
  mse_blwr_quart6.append(mse(ytest, yhat_blwr_2))

  yhat_blwr_1 = booster_lwrandchoice(xtrain,ytrain,xtest, Tricubic,0.05,model_boosting8,2)
  yhat_blwr_2 = booster_lwrandchoice(xtrain,ytrain,xtest, Quartic,0.05,model_boosting8,2)

  mse_blwr_tric8.append(mse(ytest, yhat_blwr_1))
  mse_blwr_quart8.append(mse(ytest, yhat_blwr_2))

print('The Cross-validated Mean Squared Error for Boosted LWR with tricubic is : '+str(np.mean(mse_blwr_tric4)))
print('The Cross-validated Mean Squared Error for Boosted LWR with quartic is : '+str(np.mean(mse_blwr_quart4)))
print('\n')
print('The Cross-validated Mean Squared Error for Boosted with tricubic LWR is : '+str(np.mean(mse_blwr_tric6)))
print('The Cross-validated Mean Squared Error for Boosted with quartic LWR is : '+str(np.mean(mse_blwr_quart6)))
print('\n')
print('The Cross-validated Mean Squared Error for Boosted LWR tricubic is : '+str(np.mean(mse_blwr_tric6)))
print('The Cross-validated Mean Squared Error for Boosted LWR quartic is : '+str(np.mean(mse_blwr_quart6)))

mse_blwr_tric4 = []
mse_blwr_quart4 = []


mse_blwr_tric6 = []
mse_blwr_tric6 = []


mse_blwr_tric8 = []
mse_blwr_quart8 = []



kf = KFold(n_splits=5,shuffle=True,random_state=1234)
for idxtrain, idxtest in kf.split(X):
  xtrain = X[idxtrain]
  ytrain = y[idxtrain]
  ytest = y[idxtest]
  xtest = X[idxtest]
  xtrain = scale.fit_transform(xtrain)
  xtest = scale.transform(xtest)

  yhat_blwr_1 = booster_lwrandchoice(xtrain,ytrain,xtest, Tricubic,0.1,model_boosting4,2)
  yhat_blwr_2 = booster_lwrandchoice(xtrain,ytrain,xtest, Quartic,0.1,model_boosting4,2)

  mse_blwr_tric4.append(mse(ytest, yhat_blwr_1))
  mse_blwr_quart4.append(mse(ytest, yhat_blwr_2))

  yhat_blwr_1 = booster_lwrandchoice(xtrain,ytrain,xtest, Tricubic,0.1,model_boosting6,2)
  yhat_blwr_2 = booster_lwrandchoice(xtrain,ytrain,xtest, Quartic,0.1,model_boosting6,2)

  mse_blwr_tric6.append(mse(ytest, yhat_blwr_1))
  mse_blwr_quart6.append(mse(ytest, yhat_blwr_2))

  yhat_blwr_1 = booster_lwrandchoice(xtrain,ytrain,xtest, Tricubic,0.1,model_boosting8,2)
  yhat_blwr_2 = booster_lwrandchoice(xtrain,ytrain,xtest, Quartic,0.1,model_boosting8,2)

  mse_blwr_tric8.append(mse(ytest, yhat_blwr_1))
  mse_blwr_quart8.append(mse(ytest, yhat_blwr_2))

print('The Cross-validated Mean Squared Error for Boosted LWR with tricubic is : '+str(np.mean(mse_blwr_tric4)))
print('The Cross-validated Mean Squared Error for Boosted LWR with quartic is : '+str(np.mean(mse_blwr_quart4)))
print('\n')
print('The Cross-validated Mean Squared Error for Boosted with tricubic LWR is : '+str(np.mean(mse_blwr_tric6)))
print('The Cross-validated Mean Squared Error for Boosted with quartic LWR is : '+str(np.mean(mse_blwr_quart6)))
print('\n')
print('The Cross-validated Mean Squared Error for Boosted LWR tricubic is : '+str(np.mean(mse_blwr_tric6)))
print('The Cross-validated Mean Squared Error for Boosted LWR quartic is : '+str(np.mean(mse_blwr_quart6)))

mse_blwr_tric4 = []
mse_blwr_quart4 = []


mse_blwr_tric6 = []
mse_blwr_tric6 = []


mse_blwr_tric8 = []
mse_blwr_quart8 = []



kf = KFold(n_splits=5,shuffle=True,random_state=1234)
for idxtrain, idxtest in kf.split(X):
  xtrain = X[idxtrain]
  ytrain = y[idxtrain]
  ytest = y[idxtest]
  xtest = X[idxtest]
  xtrain = scale.fit_transform(xtrain)
  xtest = scale.transform(xtest)

  yhat_blwr_1 = booster_lwrandchoice(xtrain,ytrain,xtest, Tricubic,0.5,model_boosting4,2)
  yhat_blwr_2 = booster_lwrandchoice(xtrain,ytrain,xtest, Quartic,0.5,model_boosting4,2)

  mse_blwr_tric4.append(mse(ytest, yhat_blwr_1))
  mse_blwr_quart4.append(mse(ytest, yhat_blwr_2))

  yhat_blwr_1 = booster_lwrandchoice(xtrain,ytrain,xtest, Tricubic,0.5,model_boosting6,2)
  yhat_blwr_2 = booster_lwrandchoice(xtrain,ytrain,xtest, Quartic,0.5,model_boosting6,2)

  mse_blwr_tric6.append(mse(ytest, yhat_blwr_1))
  mse_blwr_quart6.append(mse(ytest, yhat_blwr_2))

  yhat_blwr_1 = booster_lwrandchoice(xtrain,ytrain,xtest, Tricubic,0.5,model_boosting8,2)
  yhat_blwr_2 = booster_lwrandchoice(xtrain,ytrain,xtest, Quartic,0.5,model_boosting8,2)

  mse_blwr_tric8.append(mse(ytest, yhat_blwr_1))
  mse_blwr_quart8.append(mse(ytest, yhat_blwr_2))

print('The Cross-validated Mean Squared Error for Boosted LWR with tricubic is : '+str(np.mean(mse_blwr_tric4)))
print('The Cross-validated Mean Squared Error for Boosted LWR with quartic is : '+str(np.mean(mse_blwr_quart4)))
print('\n')
print('The Cross-validated Mean Squared Error for Boosted with tricubic LWR is : '+str(np.mean(mse_blwr_tric6)))
print('The Cross-validated Mean Squared Error for Boosted with quartic LWR is : '+str(np.mean(mse_blwr_quart6)))
print('\n')
print('The Cross-validated Mean Squared Error for Boosted LWR tricubic is : '+str(np.mean(mse_blwr_tric6)))
print('The Cross-validated Mean Squared Error for Boosted LWR quartic is : '+str(np.mean(mse_blwr_quart6)))

mse_blwr_tric4 = []
mse_blwr_quart4 = []


mse_blwr_tric6 = []
mse_blwr_tric6 = []


mse_blwr_tric8 = []
mse_blwr_quart8 = []



kf = KFold(n_splits=5,shuffle=True,random_state=1234)
for idxtrain, idxtest in kf.split(X):
  xtrain = X[idxtrain]
  ytrain = y[idxtrain]
  ytest = y[idxtest]
  xtest = X[idxtest]
  xtrain = scale.fit_transform(xtrain)
  xtest = scale.transform(xtest)

  yhat_blwr_1 = booster_lwrandchoice(xtrain,ytrain,xtest, Tricubic,1,model_boosting4,2)
  yhat_blwr_2 = booster_lwrandchoice(xtrain,ytrain,xtest, Quartic,1,model_boosting4,2)

  mse_blwr_tric4.append(mse(ytest, yhat_blwr_1))
  mse_blwr_quart4.append(mse(ytest, yhat_blwr_2))

  yhat_blwr_1 = booster_lwrandchoice(xtrain,ytrain,xtest, Tricubic,1,model_boosting6,2)
  yhat_blwr_2 = booster_lwrandchoice(xtrain,ytrain,xtest, Quartic,1,model_boosting6,2)

  mse_blwr_tric6.append(mse(ytest, yhat_blwr_1))
  mse_blwr_quart6.append(mse(ytest, yhat_blwr_2))

  yhat_blwr_1 = booster_lwrandchoice(xtrain,ytrain,xtest, Tricubic,1,model_boosting8,2)
  yhat_blwr_2 = booster_lwrandchoice(xtrain,ytrain,xtest, Quartic,1,model_boosting8,2)

  mse_blwr_tric8.append(mse(ytest, yhat_blwr_1))
  mse_blwr_quart8.append(mse(ytest, yhat_blwr_2))

print('The Cross-validated Mean Squared Error for Boosted LWR with tricubic is : '+str(np.mean(mse_blwr_tric4)))
print('The Cross-validated Mean Squared Error for Boosted LWR with quartic is : '+str(np.mean(mse_blwr_quart4)))
print('\n')
print('The Cross-validated Mean Squared Error for Boosted with tricubic LWR is : '+str(np.mean(mse_blwr_tric6)))
print('The Cross-validated Mean Squared Error for Boosted with quartic LWR is : '+str(np.mean(mse_blwr_quart6)))
print('\n')
print('The Cross-validated Mean Squared Error for Boosted LWR tricubic is : '+str(np.mean(mse_blwr_tric6)))
print('The Cross-validated Mean Squared Error for Boosted LWR quartic is : '+str(np.mean(mse_blwr_quart6)))

# Next set of XGBoost models to boost
model_boosting1 = xgb.XGBRegressor(max_depth=2, learning_rate=0.05, n_estimators=5, objective='reg:squarederror')
model_boosting2 = xgb.XGBRegressor(max_depth=3, learning_rate=0.05, n_estimators=5, objective='reg:squarederror')
model_boosting3 = xgb.XGBRegressor(max_depth=2, learning_rate=0.05, n_estimators=10, objective='reg:squarederror')
model_boosting4 = xgb.XGBRegressor(max_depth=3, learning_rate=0.05, n_estimators=10, objective='reg:squarederror')
model_boosting5 = xgb.XGBRegressor(max_depth=2, learning_rate=0.1, n_estimators=5, objective='reg:squarederror')
model_boosting6 = xgb.XGBRegressor(max_depth=3, learning_rate=0.1, n_estimators=5, objective='reg:squarederror')
model_boosting7 = xgb.XGBRegressor(max_depth=2, learning_rate=0.1, n_estimators=10, objective='reg:squarederror')
model_boosting8 = xgb.XGBRegressor(max_depth=3, learning_rate=0.1, n_estimators=10, objective='reg:squarederror')

mse_blwr_tric1 = []
mse_blwr_quart1 = []
mse_blwr_epa1 = []
mse_blwr_triw1 = []
mse_blwr_trian1 = []

mse_blwr_tric2 = []
mse_blwr_quart2 = []
mse_blwr_epa2 = []
mse_blwr_triw2 = []
mse_blwr_trian2 = []

mse_blwr_tric3 = []
mse_blwr_quart3 = []
mse_blwr_epa3 = []
mse_blwr_triw3 = []
mse_blwr_trian3 = []

mse_blwr_tric4 = []
mse_blwr_quart4 = []
mse_blwr_epa4 = []
mse_blwr_triw4 = []
mse_blwr_trian4 = []

mse_blwr_tric5 = []
mse_blwr_quart5 = []
mse_blwr_epa5 = []
mse_blwr_triw5 = []
mse_blwr_trian5 = []

mse_blwr_tric6 = []
mse_blwr_quart6 = []
mse_blwr_epa6 = []
mse_blwr_triw6 = []
mse_blwr_trian6 = []

mse_blwr_tric7 = []
mse_blwr_quart7 = []
mse_blwr_epa7 = []
mse_blwr_triw7 = []
mse_blwr_trian7 = []

mse_blwr_tric8 = []
mse_blwr_quart8 = []
mse_blwr_epa8 = []
mse_blwr_triw8 = []
mse_blwr_trian8 = []


kf = KFold(n_splits=3,shuffle=True,random_state=1234)
for idxtrain, idxtest in kf.split(X):
  xtrain = X[idxtrain]
  ytrain = y[idxtrain]
  ytest = y[idxtest]
  xtest = X[idxtest]
  xtrain = scale.fit_transform(xtrain)
  xtest = scale.transform(xtest)
    
  yhat_blwr_1 = booster_lwrandchoice(xtrain,ytrain,xtest, Tricubic,1,model_boosting1,2)
  yhat_blwr_2 = booster_lwrandchoice(xtrain,ytrain,xtest, Quartic,1,model_boosting1,2)
  yhat_blwr_3 = booster_lwrandchoice(xtrain,ytrain,xtest, Epanechnikov,1,model_boosting1,2)
  yhat_blwr_4 = booster_lwrandchoice(xtrain,ytrain,xtest, Triweight,1,model_boosting1,2)
  yhat_blwr_5 = booster_lwrandchoice(xtrain,ytrain,xtest, Triangular,1,model_boosting1,2)

  mse_blwr_tric1.append(mse(ytest, yhat_blwr_1))
  mse_blwr_quart1.append(mse(ytest, yhat_blwr_2))
  mse_blwr_epa1.append(mse(ytest, yhat_blwr_3))
  mse_blwr_triw1.append(mse(ytest, yhat_blwr_4))
  mse_blwr_trian1.append(mse(ytest, yhat_blwr_5))

  yhat_blwr_1 = booster_lwrandchoice(xtrain,ytrain,xtest, Tricubic,1,model_boosting2,2)
  yhat_blwr_2 = booster_lwrandchoice(xtrain,ytrain,xtest, Quartic,1,model_boosting2,2)
  yhat_blwr_3 = booster_lwrandchoice(xtrain,ytrain,xtest, Epanechnikov,1,model_boosting2,2)
  yhat_blwr_4 = booster_lwrandchoice(xtrain,ytrain,xtest, Triweight,1,model_boosting2,2)
  yhat_blwr_5 = booster_lwrandchoice(xtrain,ytrain,xtest, Triangular,1,model_boosting2,2)

  mse_blwr_tric2.append(mse(ytest, yhat_blwr_1))
  mse_blwr_quart2.append(mse(ytest, yhat_blwr_2))
  mse_blwr_epa2.append(mse(ytest, yhat_blwr_3))
  mse_blwr_triw2.append(mse(ytest, yhat_blwr_4))
  mse_blwr_trian2.append(mse(ytest, yhat_blwr_5))

  yhat_blwr_1 = booster_lwrandchoice(xtrain,ytrain,xtest, Tricubic,1,model_boosting3,2)
  yhat_blwr_2 = booster_lwrandchoice(xtrain,ytrain,xtest, Quartic,1,model_boosting3,2)
  yhat_blwr_3 = booster_lwrandchoice(xtrain,ytrain,xtest, Epanechnikov,1,model_boosting3,2)
  yhat_blwr_4 = booster_lwrandchoice(xtrain,ytrain,xtest, Triweight,1,model_boosting3,2)
  yhat_blwr_5 = booster_lwrandchoice(xtrain,ytrain,xtest, Triangular,1,model_boosting3,2)

  mse_blwr_tric3.append(mse(ytest, yhat_blwr_1))
  mse_blwr_quart3.append(mse(ytest, yhat_blwr_2))
  mse_blwr_epa3.append(mse(ytest, yhat_blwr_3))
  mse_blwr_triw3.append(mse(ytest, yhat_blwr_4))
  mse_blwr_trian3.append(mse(ytest, yhat_blwr_5))


  yhat_blwr_1 = booster_lwrandchoice(xtrain,ytrain,xtest, Tricubic,1,model_boosting4,2)
  yhat_blwr_2 = booster_lwrandchoice(xtrain,ytrain,xtest, Quartic,1,model_boosting4,2)
  yhat_blwr_3 = booster_lwrandchoice(xtrain,ytrain,xtest, Epanechnikov,1,model_boosting4,2)
  yhat_blwr_4 = booster_lwrandchoice(xtrain,ytrain,xtest, Triweight,1,model_boosting4,2)
  yhat_blwr_5 = booster_lwrandchoice(xtrain,ytrain,xtest, Triangular,1,model_boosting4,2)

  mse_blwr_tric4.append(mse(ytest, yhat_blwr_1))
  mse_blwr_quart4.append(mse(ytest, yhat_blwr_2))
  mse_blwr_epa4.append(mse(ytest, yhat_blwr_3))
  mse_blwr_triw4.append(mse(ytest, yhat_blwr_4))
  mse_blwr_trian4.append(mse(ytest, yhat_blwr_5))

  yhat_blwr_1 = booster_lwrandchoice(xtrain,ytrain,xtest, Tricubic,1,model_boosting5,2)
  yhat_blwr_2 = booster_lwrandchoice(xtrain,ytrain,xtest, Quartic,1,model_boosting5,2)
  yhat_blwr_3 = booster_lwrandchoice(xtrain,ytrain,xtest, Epanechnikov,1,model_boosting5,2)
  yhat_blwr_4 = booster_lwrandchoice(xtrain,ytrain,xtest, Triweight,1,model_boosting5,2)
  yhat_blwr_5 = booster_lwrandchoice(xtrain,ytrain,xtest, Triangular,1,model_boosting5,2)

  mse_blwr_tric5.append(mse(ytest, yhat_blwr_1))
  mse_blwr_quart5.append(mse(ytest, yhat_blwr_2))
  mse_blwr_epa5.append(mse(ytest, yhat_blwr_3))
  mse_blwr_triw5.append(mse(ytest, yhat_blwr_4))
  mse_blwr_trian5.append(mse(ytest, yhat_blwr_5))

  yhat_blwr_1 = booster_lwrandchoice(xtrain,ytrain,xtest, Tricubic,1,model_boosting6,2)
  yhat_blwr_2 = booster_lwrandchoice(xtrain,ytrain,xtest, Quartic,1,model_boosting6,2)
  yhat_blwr_3 = booster_lwrandchoice(xtrain,ytrain,xtest, Epanechnikov,1,model_boosting6,2)
  yhat_blwr_4 = booster_lwrandchoice(xtrain,ytrain,xtest, Triweight,1,model_boosting6,2)
  yhat_blwr_5 = booster_lwrandchoice(xtrain,ytrain,xtest, Triangular,1,model_boosting6,2)

  mse_blwr_tric6.append(mse(ytest, yhat_blwr_1))
  mse_blwr_quart6.append(mse(ytest, yhat_blwr_2))
  mse_blwr_epa6.append(mse(ytest, yhat_blwr_3))
  mse_blwr_triw6.append(mse(ytest, yhat_blwr_4))
  mse_blwr_trian6.append(mse(ytest, yhat_blwr_5))

  yhat_blwr_1 = booster_lwrandchoice(xtrain,ytrain,xtest, Tricubic,1,model_boosting7,2)
  yhat_blwr_2 = booster_lwrandchoice(xtrain,ytrain,xtest, Quartic,1,model_boosting7,2)
  yhat_blwr_3 = booster_lwrandchoice(xtrain,ytrain,xtest, Epanechnikov,1,model_boosting7,2)
  yhat_blwr_4 = booster_lwrandchoice(xtrain,ytrain,xtest, Triweight,1,model_boosting7,2)
  yhat_blwr_5 = booster_lwrandchoice(xtrain,ytrain,xtest, Triangular,1,model_boosting7,2)

  mse_blwr_tric7.append(mse(ytest, yhat_blwr_1))
  mse_blwr_quart7.append(mse(ytest, yhat_blwr_2))
  mse_blwr_epa7.append(mse(ytest, yhat_blwr_3))
  mse_blwr_triw7.append(mse(ytest, yhat_blwr_4))
  mse_blwr_trian7.append(mse(ytest, yhat_blwr_5))

  yhat_blwr_1 = booster_lwrandchoice(xtrain,ytrain,xtest, Tricubic,1,model_boosting8,2)
  yhat_blwr_2 = booster_lwrandchoice(xtrain,ytrain,xtest, Quartic,1,model_boosting8,2)
  yhat_blwr_3 = booster_lwrandchoice(xtrain,ytrain,xtest, Epanechnikov,1,model_boosting8,2)
  yhat_blwr_4 = booster_lwrandchoice(xtrain,ytrain,xtest, Triweight,1,model_boosting8,2)
  yhat_blwr_5 = booster_lwrandchoice(xtrain,ytrain,xtest, Triangular,1,model_boosting8,2)

  mse_blwr_tric8.append(mse(ytest, yhat_blwr_1))
  mse_blwr_quart8.append(mse(ytest, yhat_blwr_2))
  mse_blwr_epa8.append(mse(ytest, yhat_blwr_3))
  mse_blwr_triw8.append(mse(ytest, yhat_blwr_4))
  mse_blwr_trian8.append(mse(ytest, yhat_blwr_5))


print('The Cross-validated Mean Squared Error for Boosted LWR with tricubic is : '+str(np.mean(mse_blwr_tric1)))
print('The Cross-validated Mean Squared Error for Boosted LWR with quartic is : '+str(np.mean(mse_blwr_quart1)))
print('The Cross-validated Mean Squared Error for Boosted LWR with Epanechnikov is : '+str(np.mean(mse_blwr_epa1)))
print('The Cross-validated Mean Squared Error for Boosted LWR with triweight is : '+str(np.mean(mse_blwr_triw1)))
print('The Cross-validated Mean Squared Error for Boosted LWR with triangular is : '+str(np.mean(mse_blwr_trian1)))
print('\n')
print('The Cross-validated Mean Squared Error for Boosted LWR with tricubic is  : '+str(np.mean(mse_blwr_tric2)))
print('The Cross-validated Mean Squared Error for Boosted LWR with quartic is : '+str(np.mean(mse_blwr_quart2)))
print('The Cross-validated Mean Squared Error for Boosted LWR with Epanechnikov is : '+str(np.mean(mse_blwr_epa2)))
print('The Cross-validated Mean Squared Error for Boosted LWR with triweight is : '+str(np.mean(mse_blwr_triw2)))
print('The Cross-validated Mean Squared Error for Boosted LWR with triangular is : '+str(np.mean(mse_blwr_trian2)))
print('\n')
print('The Cross-validated Mean Squared Error for Boosted LWR with tricubic is : '+str(np.mean(mse_blwr_tric3)))
print('The Cross-validated Mean Squared Error for Boosted LWR with quartic is : '+str(np.mean(mse_blwr_quart3)))
print('The Cross-validated Mean Squared Error for Boosted LWR with Epanechnikov is : '+str(np.mean(mse_blwr_epa3)))
print('The Cross-validated Mean Squared Error for Boosted LWR with triweight is : '+str(np.mean(mse_blwr_triw3)))
print('The Cross-validated Mean Squared Error for Boosted LWR with triangular is : '+str(np.mean(mse_blwr_trian3)))
print('\n')
print('The Cross-validated Mean Squared Error for Boosted LWR with tricubic is : '+str(np.mean(mse_blwr_tric4)))
print('The Cross-validated Mean Squared Error for Boosted LWR with quartic is : '+str(np.mean(mse_blwr_quart4)))
print('The Cross-validated Mean Squared Error for Boosted LWR with Epanechnikov is : '+str(np.mean(mse_blwr_epa4)))
print('The Cross-validated Mean Squared Error for Boosted LWR with triweight is : '+str(np.mean(mse_blwr_triw4)))
print('The Cross-validated Mean Squared Error for Boosted LWR with triangular is : '+str(np.mean(mse_blwr_trian4)))
print('\n')
print('The Cross-validated Mean Squared Error for Boosted LWR with tricubic is : '+str(np.mean(mse_blwr_tric5)))
print('The Cross-validated Mean Squared Error for Boosted LWR with quartic is : '+str(np.mean(mse_blwr_quart5)))
print('The Cross-validated Mean Squared Error for Boosted LWR with Epanechnikov is : '+str(np.mean(mse_blwr_epa5)))
print('The Cross-validated Mean Squared Error for Boosted LWR with triweight is : '+str(np.mean(mse_blwr_triw5)))
print('The Cross-validated Mean Squared Error for Boosted LWR with triangular is : '+str(np.mean(mse_blwr_trian5)))
print('\n')
print('The Cross-validated Mean Squared Error for Boosted LWR with tricubic is : '+str(np.mean(mse_blwr_tric6)))
print('The Cross-validated Mean Squared Error for Boosted LWR with quartic is : '+str(np.mean(mse_blwr_quart6)))
print('The Cross-validated Mean Squared Error for Boosted LWR with Epanechnikov is : '+str(np.mean(mse_blwr_epa6)))
print('The Cross-validated Mean Squared Error for Boosted LWR with triweight is : '+str(np.mean(mse_blwr_triw6)))
print('The Cross-validated Mean Squared Error for Boosted LWR with triangular is : '+str(np.mean(mse_blwr_trian6)))
print('\n')
print('The Cross-validated Mean Squared Error for Boosted LWR with tricubic is : '+str(np.mean(mse_blwr_tric7)))
print('The Cross-validated Mean Squared Error for Boosted LWR with quartic is : '+str(np.mean(mse_blwr_quart7)))
print('The Cross-validated Mean Squared Error for Boosted LWR with Epanechnikov is : '+str(np.mean(mse_blwr_epa7)))
print('The Cross-validated Mean Squared Error for Boosted LWR with triweight is : '+str(np.mean(mse_blwr_triw7)))
print('The Cross-validated Mean Squared Error for Boosted LWR with triangular is : '+str(np.mean(mse_blwr_trian7)))
print('\n')
print('The Cross-validated Mean Squared Error for Boosted LWR with tricubic is : '+str(np.mean(mse_blwr_tric8)))
print('The Cross-validated Mean Squared Error for Boosted LWR with quartic is : '+str(np.mean(mse_blwr_quart8)))
print('The Cross-validated Mean Squared Error for Boosted LWR with Epanechikov is : '+str(np.mean(mse_blwr_epa8)))
print('The Cross-validated Mean Squared Error for Boosted LWR with triweight is : '+str(np.mean(mse_blwr_triw8)))
print('The Cross-validated Mean Squared Error for Boosted LWR with triangular is : '+str(np.mean(mse_blwr_trian8)))


mse_blwr_tric8 = []
mse_blwr_quart8 = []
mse_blwr_epa8 = []
mse_blwr_triw8 = []
mse_blwr_trian8 = []

kf = KFold(n_splits=5,shuffle=True,random_state=1234)
for idxtrain, idxtest in kf.split(X):
  xtrain = X[idxtrain]
  ytrain = y[idxtrain]
  ytest = y[idxtest]
  xtest = X[idxtest]
  xtrain = scale.fit_transform(xtrain)
  xtest = scale.transform(xtest)
    
  yhat_blwr_1 = booster_lwrandchoice(xtrain,ytrain,xtest, Tricubic,0.01,model_boosting8,2)
  yhat_blwr_2 = booster_lwrandchoice(xtrain,ytrain,xtest, Quartic,0.01,model_boosting8,2)
  yhat_blwr_3 = booster_lwrandchoice(xtrain,ytrain,xtest, Epanechnikov,0.01,model_boosting8,2)
  yhat_blwr_4 = booster_lwrandchoice(xtrain,ytrain,xtest, Triweight,0.01,model_boosting8,2)
  yhat_blwr_5 = booster_lwrandchoice(xtrain,ytrain,xtest, Triangular,0.01,model_boosting8,2)

  mse_blwr_tric8.append(mse(ytest, yhat_blwr_1))
  mse_blwr_quart8.append(mse(ytest, yhat_blwr_2))
  mse_blwr_epa8.append(mse(ytest, yhat_blwr_3))
  mse_blwr_triw8.append(mse(ytest, yhat_blwr_4))
  mse_blwr_trian8.append(mse(ytest, yhat_blwr_5))

 
print('The Cross-validated Mean Squared Error for Boosted LWR with tricubic is : '+str(np.mean(mse_blwr_tric8)))
print('The Cross-validated Mean Squared Error for Boosted LWR with quartic is : '+str(np.mean(mse_blwr_quart8)))
print('The Cross-validated Mean Squared Error for Boosted LWR with Epanechikov is : '+str(np.mean(mse_blwr_epa8)))
print('The Cross-validated Mean Squared Error for Boosted LWR with triweight is : '+str(np.mean(mse_blwr_triw8)))
print('The Cross-validated Mean Squared Error for Boosted LWR with triangular is : '+str(np.mean(mse_blwr_trian8)))


mse_blwr_tric8 = []
mse_blwr_quart8 = []
mse_blwr_epa8 = []
mse_blwr_triw8 = []
mse_blwr_trian8 = []

kf = KFold(n_splits=5,shuffle=True,random_state=1234)
for idxtrain, idxtest in kf.split(X):
  xtrain = X[idxtrain]
  ytrain = y[idxtrain]
  ytest = y[idxtest]
  xtest = X[idxtest]
  xtrain = scale.fit_transform(xtrain)
  xtest = scale.transform(xtest)
    
  yhat_blwr_1 = booster_lwrandchoice(xtrain,ytrain,xtest, Tricubic,0.05,model_boosting8,2)
  yhat_blwr_2 = booster_lwrandchoice(xtrain,ytrain,xtest, Quartic,0.05,model_boosting8,2)
  yhat_blwr_3 = booster_lwrandchoice(xtrain,ytrain,xtest, Epanechnikov,0.05,model_boosting8,2)
  yhat_blwr_4 = booster_lwrandchoice(xtrain,ytrain,xtest, Triweight,0.05,model_boosting8,2)
  yhat_blwr_5 = booster_lwrandchoice(xtrain,ytrain,xtest, Triangular,0.05,model_boosting8,2)

  mse_blwr_tric8.append(mse(ytest, yhat_blwr_1))
  mse_blwr_quart8.append(mse(ytest, yhat_blwr_2))
  mse_blwr_epa8.append(mse(ytest, yhat_blwr_3))
  mse_blwr_triw8.append(mse(ytest, yhat_blwr_4))
  mse_blwr_trian8.append(mse(ytest, yhat_blwr_5))

 
print('The Cross-validated Mean Squared Error for Boosted LWR with tricubic is : '+str(np.mean(mse_blwr_tric8)))
print('The Cross-validated Mean Squared Error for Boosted LWR with quartic is : '+str(np.mean(mse_blwr_quart8)))
print('The Cross-validated Mean Squared Error for Boosted LWR with Epanechikov is : '+str(np.mean(mse_blwr_epa8)))
print('The Cross-validated Mean Squared Error for Boosted LWR with triweight is : '+str(np.mean(mse_blwr_triw8)))
print('The Cross-validated Mean Squared Error for Boosted LWR with triangular is : '+str(np.mean(mse_blwr_trian8)))


mse_blwr_tric8 = []
mse_blwr_quart8 = []
mse_blwr_epa8 = []
mse_blwr_triw8 = []
mse_blwr_trian8 = []

kf = KFold(n_splits=5,shuffle=True,random_state=1234)
for idxtrain, idxtest in kf.split(X):
  xtrain = X[idxtrain]
  ytrain = y[idxtrain]
  ytest = y[idxtest]
  xtest = X[idxtest]
  xtrain = scale.fit_transform(xtrain)
  xtest = scale.transform(xtest)
    
  yhat_blwr_1 = booster_lwrandchoice(xtrain,ytrain,xtest, Tricubic,0.1,model_boosting8,2)
  yhat_blwr_2 = booster_lwrandchoice(xtrain,ytrain,xtest, Quartic,0.1,model_boosting8,2)
  yhat_blwr_3 = booster_lwrandchoice(xtrain,ytrain,xtest, Epanechnikov,0.1,model_boosting8,2)
  yhat_blwr_4 = booster_lwrandchoice(xtrain,ytrain,xtest, Triweight,0.1,model_boosting8,2)
  yhat_blwr_5 = booster_lwrandchoice(xtrain,ytrain,xtest, Triangular,0.1,model_boosting8,2)

  mse_blwr_tric8.append(mse(ytest, yhat_blwr_1))
  mse_blwr_quart8.append(mse(ytest, yhat_blwr_2))
  mse_blwr_epa8.append(mse(ytest, yhat_blwr_3))
  mse_blwr_triw8.append(mse(ytest, yhat_blwr_4))
  mse_blwr_trian8.append(mse(ytest, yhat_blwr_5))

 
print('The Cross-validated Mean Squared Error for Boosted LWR with tricubic is : '+str(np.mean(mse_blwr_tric8)))
print('The Cross-validated Mean Squared Error for Boosted LWR with quartic is : '+str(np.mean(mse_blwr_quart8)))
print('The Cross-validated Mean Squared Error for Boosted LWR with Epanechikov is : '+str(np.mean(mse_blwr_epa8)))
print('The Cross-validated Mean Squared Error for Boosted LWR with triweight is : '+str(np.mean(mse_blwr_triw8)))
print('The Cross-validated Mean Squared Error for Boosted LWR with triangular is : '+str(np.mean(mse_blwr_trian8)))


mse_blwr_tric8 = []
mse_blwr_quart8 = []
mse_blwr_epa8 = []
mse_blwr_triw8 = []
mse_blwr_trian8 = []

kf = KFold(n_splits=5,shuffle=True,random_state=1234)
for idxtrain, idxtest in kf.split(X):
  xtrain = X[idxtrain]
  ytrain = y[idxtrain]
  ytest = y[idxtest]
  xtest = X[idxtest]
  xtrain = scale.fit_transform(xtrain)
  xtest = scale.transform(xtest)
    
  yhat_blwr_1 = booster_lwrandchoice(xtrain,ytrain,xtest, Tricubic,0.5,model_boosting8,2)
  yhat_blwr_2 = booster_lwrandchoice(xtrain,ytrain,xtest, Quartic,0.5,model_boosting8,2)
  yhat_blwr_3 = booster_lwrandchoice(xtrain,ytrain,xtest, Epanechnikov,0.5,model_boosting8,2)
  yhat_blwr_4 = booster_lwrandchoice(xtrain,ytrain,xtest, Triweight,0.5,model_boosting8,2)
  yhat_blwr_5 = booster_lwrandchoice(xtrain,ytrain,xtest, Triangular,0.5,model_boosting8,2)

  mse_blwr_tric8.append(mse(ytest, yhat_blwr_1))
  mse_blwr_quart8.append(mse(ytest, yhat_blwr_2))
  mse_blwr_epa8.append(mse(ytest, yhat_blwr_3))
  mse_blwr_triw8.append(mse(ytest, yhat_blwr_4))
  mse_blwr_trian8.append(mse(ytest, yhat_blwr_5))

 
print('The Cross-validated Mean Squared Error for Boosted LWR with tricubic is : '+str(np.mean(mse_blwr_tric8)))
print('The Cross-validated Mean Squared Error for Boosted LWR with quartic is : '+str(np.mean(mse_blwr_quart8)))
print('The Cross-validated Mean Squared Error for Boosted LWR with Epanechikov is : '+str(np.mean(mse_blwr_epa8)))
print('The Cross-validated Mean Squared Error for Boosted LWR with triweight is : '+str(np.mean(mse_blwr_triw8)))
print('The Cross-validated Mean Squared Error for Boosted LWR with triangular is : '+str(np.mean(mse_blwr_trian8)))


mse_blwr_tric8 = []
mse_blwr_quart8 = []
mse_blwr_epa8 = []
mse_blwr_triw8 = []
mse_blwr_trian8 = []

kf = KFold(n_splits=5,shuffle=True,random_state=1234)
for idxtrain, idxtest in kf.split(X):
  xtrain = X[idxtrain]
  ytrain = y[idxtrain]
  ytest = y[idxtest]
  xtest = X[idxtest]
  xtrain = scale.fit_transform(xtrain)
  xtest = scale.transform(xtest)
    
  yhat_blwr_1 = booster_lwrandchoice(xtrain,ytrain,xtest, Tricubic,1,model_boosting8,2)
  yhat_blwr_2 = booster_lwrandchoice(xtrain,ytrain,xtest, Quartic,1,model_boosting8,2)
  yhat_blwr_3 = booster_lwrandchoice(xtrain,ytrain,xtest, Epanechnikov,1,model_boosting8,2)
  yhat_blwr_4 = booster_lwrandchoice(xtrain,ytrain,xtest, Triweight,1,model_boosting8,2)
  yhat_blwr_5 = booster_lwrandchoice(xtrain,ytrain,xtest, Triangular,1,model_boosting8,2)

  mse_blwr_tric8.append(mse(ytest, yhat_blwr_1))
  mse_blwr_quart8.append(mse(ytest, yhat_blwr_2))
  mse_blwr_epa8.append(mse(ytest, yhat_blwr_3))
  mse_blwr_triw8.append(mse(ytest, yhat_blwr_4))
  mse_blwr_trian8.append(mse(ytest, yhat_blwr_5))

 
print('The Cross-validated Mean Squared Error for Boosted LWR with tricubic is : '+str(np.mean(mse_blwr_tric8)))
print('The Cross-validated Mean Squared Error for Boosted LWR with quartic is : '+str(np.mean(mse_blwr_quart8)))
print('The Cross-validated Mean Squared Error for Boosted LWR with Epanechikov is : '+str(np.mean(mse_blwr_epa8)))
print('The Cross-validated Mean Squared Error for Boosted LWR with triweight is : '+str(np.mean(mse_blwr_triw8)))
print('The Cross-validated Mean Squared Error for Boosted LWR with triangular is : '+str(np.mean(mse_blwr_trian8)))
```

### Description and Application of LightGBM Algorithm

The LightGBM boosting algorithm was developed by Microsoft as a less-intensive alternative to other boosting algorithms, like extreme gradient boosting (XGB).  According to the documentation for the algorithm provided [here](https://lightgbm.readthedocs.io/en/latest/), it is faster and more efficient than other tree-based methods while still maintaining high accuracy.  The main feature that distinguishes LightGBM from XGB and other tree-based boosting algorithms is that it implements leaf-wise splitting, as opposed to level-wise splitting.  As such, it searches for the leaf with the greatest change in loss (i.e., the greatest decline in loss) and splits the data at that particular node, as opposed to splitting at a full level of the tree (Reference: [https://www.geeksforgeeks.org/lightgbm-light-gradient-boosting-machine/](https://www.geeksforgeeks.org/lightgbm-light-gradient-boosting-machine/)).  Because this generates a less complex decision tree, the LightGBM model is faster and requires less memory usage.  Please see the two diagrams below for an illustrated comparison of leaf-splitting and level-splitting, respectively.

# Leaf Splitting
![](https://i.stack.imgur.com/YOE9y.png)


# Level Splitting
![](https://i.stack.imgur.com/e1FWe.png)


The LightGBM documentation also states that this algorithm is quick and efficient because it is histogram-based, which means it clusters continuous features into bins.  By contrast, XGB uses pre-sort algorithms.  Furthermore, if there are categorical features in the data, LightGBM splits such features into two subsets per category based on "accumulated values".  LightGBM also mplements Gradient-based One-sided Sampling (GOSS) and exclusive feature bundling (EFB) to more heavily weight the data observations that contribute the most to predicting the dependent variable, y, and grouping "exclusive" features into one feature, respectively.  Something interesting to note from the reference previously mentioned is that one should be cautious when using LightGBM on smaller datasets because it has a tendency to be overfit.  Among its many hyperparameters, the LightGBM regressor algorithm includes a hyperparameter for its learning rate, which tells the model at what rate it should adjust its weight predictions given the current level of error - higher learning rates tend to execute more quickly, but may be less accurate than smaller learning rates (Reference: [https://machinelearningmastery.com/learning-rate-for-deep-learning-neural-networks/](https://machinelearningmastery.com/learning-rate-for-deep-learning-neural-networks/).  Due to the learning rate's seemingly significant impact on the accuracy of the model, I chose to experiment with a variety of learning rate settings in my analysis/application of the LightGBM algorithm below.  In addition, I tuned its "max_depth" hyperparameter to help determine an optimal level at which to discontinue splitting the tree further.



## Code for LightGBM Application to Concrete Strength Dataset
```
# Import Statements
!pip install xlrd==1.2.0
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.model_selection import KFold, train_test_split as tts
from sklearn.metrics import mean_squared_log_error as le
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae
from sklearn.preprocessing import StandardScaler

scale = StandardScaler()

# Load data
concrete = pd.read_excel("Concrete_Data.xls")

X = concrete[['Cement (component 1)(kg in a m^3 mixture)', 'Blast Furnace Slag (component 2)(kg in a m^3 mixture)',
              'Fly Ash (component 3)(kg in a m^3 mixture)', 'Coarse Aggregate  (component 6)(kg in a m^3 mixture)',
              'Water  (component 4)(kg in a m^3 mixture)', 'Superplasticizer (component 5)(kg in a m^3 mixture)',
              'Fine Aggregate (component 7)(kg in a m^3 mixture)', 'Age (day)']].values
y = concrete["Concrete compressive strength(MPa, megapascals) "].values

kf = KFold(n_splits=10,shuffle=True,random_state=410)
error_depth2l005 = []
error_depth3l005 = []
error_depth5l005 = []
error_depth10l005 = []

error_depth2l01 = []
error_depth3l01 = []
error_depth5l01 = []
error_depth10l01 = []

error_depth2l001 = []
error_depth3l001 = []
error_depth5l001 = []
error_depth10l001 = []

error_depth2l05 = []
error_depth3l05 = []
error_depth5l05 = []
error_depth10l05 = []

error_depth2l1 = []
error_depth3l1 = []
error_depth5l1 = []
error_depth10l1 = []

error_depth2l5 = []
error_depth3l5 = []
error_depth5l5 = []
error_depth10l5 = []

error_depth2l005_500 = []
error_depth3l005_500 = []
error_depth5l005_500 = []
error_depth10l005_500 = []

error_depth2l01_500 = []
error_depth3l01_500 = []
error_depth5l01_500 = []
error_depth10l01_500 = []

error_depth2l001_500 = []
error_depth3l001_500 = []
error_depth5l001_500 = []
error_depth10l001_500 = []

error_depth2l05_500 = []
error_depth3l05_500 = []
error_depth5l05_500 = []
error_depth10l05_500 = []

error_depth2l1_500 = []
error_depth3l1_500 = []
error_depth5l1_500 = []
error_depth10l1_500 = []

error_depth2l5_500 = []
error_depth3l5_500 = []
error_depth5l5_500 = []
error_depth10l5_500 = []

error_depth2l005_1000 = []
error_depth3l005_1000 = []
error_depth5l005_1000 = []
error_depth10l005_1000 = []

error_depth2l01_1000 = []
error_depth3l01_1000 = []
error_depth5l01_1000 = []
error_depth10l01_1000 = []

error_depth2l001_1000 = []
error_depth3l001_1000 = []
error_depth5l001_1000 = []
error_depth10l001_1000 = []

error_depth2l05_1000 = []
error_depth3l05_1000 = []
error_depth5l05_1000 = []
error_depth10l05_1000 = []

error_depth2l1_1000 = []
error_depth3l1_1000 = []
error_depth5l1_1000 = []
error_depth10l1_1000 = []

error_depth2l5_1000 = []
error_depth3l5_1000 = []
error_depth5l5_1000 = []
error_depth10l5_1000 = []

for idxtrain, idxtest in kf.split(X):
  ytrain = y[idxtrain]
  xtrain = X[idxtrain]
  ytest = y[idxtest]
  xtest = X[idxtest]
  xtrain = scale.fit_transform(xtrain)
  xtest = scale.transform(xtest)

  gbm = lgb.LGBMRegressor(learning_rate=0.005, max_depth=2, n_estimators=100)
  gbm.fit(xtrain,ytrain)
  yhat1 = gbm.predict(xtest)

  gbm = lgb.LGBMRegressor(learning_rate=0.005, max_depth=3, n_estimators=100)
  gbm.fit(xtrain,ytrain)
  yhat2 = gbm.predict(xtest)

  gbm = lgb.LGBMRegressor(learning_rate=0.005, max_depth=5, n_estimators=100)
  gbm.fit(xtrain,ytrain)
  yhat3 = gbm.predict(xtest)

  gbm = lgb.LGBMRegressor(learning_rate=0.005, max_depth=10, n_estimators=100)
  gbm.fit(xtrain,ytrain)
  yhat4 = gbm.predict(xtest)


  gbm = lgb.LGBMRegressor(learning_rate=0.001, max_depth=2, n_estimators=100)
  gbm.fit(xtrain,ytrain)
  yhat5 = gbm.predict(xtest)

  gbm = lgb.LGBMRegressor(learning_rate=0.001, max_depth=3, n_estimators=100)
  gbm.fit(xtrain,ytrain)
  yhat6 = gbm.predict(xtest)

  gbm = lgb.LGBMRegressor(learning_rate=0.001, max_depth=5, n_estimators=100)
  gbm.fit(xtrain,ytrain)
  yhat7 = gbm.predict(xtest)

  gbm = lgb.LGBMRegressor(learning_rate=0.001, max_depth=10, n_estimators=100)
  gbm.fit(xtrain,ytrain)
  yhat8 = gbm.predict(xtest)


  gbm = lgb.LGBMRegressor(learning_rate=0.05, max_depth=2, n_estimators=100)
  gbm.fit(xtrain,ytrain)
  yhat9 = gbm.predict(xtest)

  gbm = lgb.LGBMRegressor(learning_rate=0.05, max_depth=3, n_estimators=100)
  gbm.fit(xtrain,ytrain)
  yhat10 = gbm.predict(xtest)

  gbm = lgb.LGBMRegressor(learning_rate=0.05, max_depth=5, n_estimators=100)
  gbm.fit(xtrain,ytrain)
  yhat11 = gbm.predict(xtest)

  gbm = lgb.LGBMRegressor(learning_rate=0.05, max_depth=10, n_estimators=100)
  gbm.fit(xtrain,ytrain)
  yhat12 = gbm.predict(xtest)


  gbm = lgb.LGBMRegressor(learning_rate=0.01, max_depth=2, n_estimators=100)
  gbm.fit(xtrain,ytrain)
  yhat13 = gbm.predict(xtest)

  gbm = lgb.LGBMRegressor(learning_rate=0.01, max_depth=3, n_estimators=100)
  gbm.fit(xtrain,ytrain)
  yhat14 = gbm.predict(xtest)

  gbm = lgb.LGBMRegressor(learning_rate=0.01, max_depth=5, n_estimators=100)
  gbm.fit(xtrain,ytrain)
  yhat15 = gbm.predict(xtest)

  gbm = lgb.LGBMRegressor(learning_rate=0.01, max_depth=10, n_estimators=100)
  gbm.fit(xtrain,ytrain)
  yhat16 = gbm.predict(xtest)


  gbm = lgb.LGBMRegressor(learning_rate=0.5, max_depth=2, n_estimators=100)
  gbm.fit(xtrain,ytrain)
  yhat17 = gbm.predict(xtest)

  gbm = lgb.LGBMRegressor(learning_rate=0.5, max_depth=3, n_estimators=100)
  gbm.fit(xtrain,ytrain)
  yhat18 = gbm.predict(xtest)

  gbm = lgb.LGBMRegressor(learning_rate=0.5, max_depth=5, n_estimators=100)
  gbm.fit(xtrain,ytrain)
  yhat19 = gbm.predict(xtest)

  gbm = lgb.LGBMRegressor(learning_rate=0.5, max_depth=10, n_estimators=100)
  gbm.fit(xtrain,ytrain)
  yhat20 = gbm.predict(xtest)


  gbm = lgb.LGBMRegressor(learning_rate=0.1, max_depth=2, n_estimators=100)
  gbm.fit(xtrain,ytrain)
  yhat21 = gbm.predict(xtest)

  gbm = lgb.LGBMRegressor(learning_rate=0.1, max_depth=3, n_estimators=100)
  gbm.fit(xtrain,ytrain)
  yhat22 = gbm.predict(xtest)

  gbm = lgb.LGBMRegressor(learning_rate=0.1, max_depth=5, n_estimators=100)
  gbm.fit(xtrain,ytrain)
  yhat23 = gbm.predict(xtest)

  gbm = lgb.LGBMRegressor(learning_rate=0.1, max_depth=10, n_estimators=100)
  gbm.fit(xtrain,ytrain)
  yhat24 = gbm.predict(xtest)


  gbm = lgb.LGBMRegressor(learning_rate=0.005, max_depth=2, n_estimators=500)
  gbm.fit(xtrain,ytrain)
  yhat25 = gbm.predict(xtest)

  gbm = lgb.LGBMRegressor(learning_rate=0.005, max_depth=3, n_estimators=500)
  gbm.fit(xtrain,ytrain)
  yhat26 = gbm.predict(xtest)

  gbm = lgb.LGBMRegressor(learning_rate=0.005, max_depth=5, n_estimators=500)
  gbm.fit(xtrain,ytrain)
  yhat27 = gbm.predict(xtest)

  gbm = lgb.LGBMRegressor(learning_rate=0.005, max_depth=10, n_estimators=500)
  gbm.fit(xtrain,ytrain)
  yhat28 = gbm.predict(xtest)


  gbm = lgb.LGBMRegressor(learning_rate=0.001, max_depth=2, n_estimators=500)
  gbm.fit(xtrain,ytrain)
  yhat29 = gbm.predict(xtest)

  gbm = lgb.LGBMRegressor(learning_rate=0.001, max_depth=3, n_estimators=500)
  gbm.fit(xtrain,ytrain)
  yhat30 = gbm.predict(xtest)

  gbm = lgb.LGBMRegressor(learning_rate=0.001, max_depth=5, n_estimators=500)
  gbm.fit(xtrain,ytrain)
  yhat31 = gbm.predict(xtest)

  gbm = lgb.LGBMRegressor(learning_rate=0.001, max_depth=10, n_estimators=500)
  gbm.fit(xtrain,ytrain)
  yhat32 = gbm.predict(xtest)


  gbm = lgb.LGBMRegressor(learning_rate=0.05, max_depth=2, n_estimators=500)
  gbm.fit(xtrain,ytrain)
  yhat33 = gbm.predict(xtest)

  gbm = lgb.LGBMRegressor(learning_rate=0.05, max_depth=3, n_estimators=500)
  gbm.fit(xtrain,ytrain)
  yhat34 = gbm.predict(xtest)

  gbm = lgb.LGBMRegressor(learning_rate=0.05, max_depth=5, n_estimators=500)
  gbm.fit(xtrain,ytrain)
  yhat35 = gbm.predict(xtest)

  gbm = lgb.LGBMRegressor(learning_rate=0.05, max_depth=10, n_estimators=500)
  gbm.fit(xtrain,ytrain)
  yhat36 = gbm.predict(xtest)


  gbm = lgb.LGBMRegressor(learning_rate=0.01, max_depth=2, n_estimators=500)
  gbm.fit(xtrain,ytrain)
  yhat37 = gbm.predict(xtest)

  gbm = lgb.LGBMRegressor(learning_rate=0.01, max_depth=3, n_estimators=500)
  gbm.fit(xtrain,ytrain)
  yhat38 = gbm.predict(xtest)

  gbm = lgb.LGBMRegressor(learning_rate=0.01, max_depth=5, n_estimators=500)
  gbm.fit(xtrain,ytrain)
  yhat39 = gbm.predict(xtest)

  gbm = lgb.LGBMRegressor(learning_rate=0.01, max_depth=10, n_estimators=500)
  gbm.fit(xtrain,ytrain)
  yhat40 = gbm.predict(xtest)


  gbm = lgb.LGBMRegressor(learning_rate=0.5, max_depth=2, n_estimators=500)
  gbm.fit(xtrain,ytrain)
  yhat41 = gbm.predict(xtest)

  gbm = lgb.LGBMRegressor(learning_rate=0.5, max_depth=3, n_estimators=500)
  gbm.fit(xtrain,ytrain)
  yhat42 = gbm.predict(xtest)

  gbm = lgb.LGBMRegressor(learning_rate=0.5, max_depth=5, n_estimators=500)
  gbm.fit(xtrain,ytrain)
  yhat43 = gbm.predict(xtest)

  gbm = lgb.LGBMRegressor(learning_rate=0.5, max_depth=10, n_estimators=500)
  gbm.fit(xtrain,ytrain)
  yhat44 = gbm.predict(xtest)


  gbm = lgb.LGBMRegressor(learning_rate=0.1, max_depth=2, n_estimators=500)
  gbm.fit(xtrain,ytrain)
  yhat45 = gbm.predict(xtest)

  gbm = lgb.LGBMRegressor(learning_rate=0.1, max_depth=3, n_estimators=500)
  gbm.fit(xtrain,ytrain)
  yhat46 = gbm.predict(xtest)

  gbm = lgb.LGBMRegressor(learning_rate=0.1, max_depth=5, n_estimators=500)
  gbm.fit(xtrain,ytrain)
  yhat47 = gbm.predict(xtest)

  gbm = lgb.LGBMRegressor(learning_rate=0.1, max_depth=10, n_estimators=500)
  gbm.fit(xtrain,ytrain)
  yhat48 = gbm.predict(xtest)



  gbm = lgb.LGBMRegressor(learning_rate=0.005, max_depth=2, n_estimators=1000)
  gbm.fit(xtrain,ytrain)
  yhat49 = gbm.predict(xtest)

  gbm = lgb.LGBMRegressor(learning_rate=0.005, max_depth=3, n_estimators=1000)
  gbm.fit(xtrain,ytrain)
  yhat50 = gbm.predict(xtest)

  gbm = lgb.LGBMRegressor(learning_rate=0.005, max_depth=5, n_estimators=1000)
  gbm.fit(xtrain,ytrain)
  yhat51 = gbm.predict(xtest)

  gbm = lgb.LGBMRegressor(learning_rate=0.005, max_depth=10, n_estimators=1000)
  gbm.fit(xtrain,ytrain)
  yhat52 = gbm.predict(xtest)


  gbm = lgb.LGBMRegressor(learning_rate=0.001, max_depth=2, n_estimators=1000)
  gbm.fit(xtrain,ytrain)
  yhat53 = gbm.predict(xtest)

  gbm = lgb.LGBMRegressor(learning_rate=0.001, max_depth=3, n_estimators=1000)
  gbm.fit(xtrain,ytrain)
  yhat54 = gbm.predict(xtest)

  gbm = lgb.LGBMRegressor(learning_rate=0.001, max_depth=5, n_estimators=1000)
  gbm.fit(xtrain,ytrain)
  yhat55 = gbm.predict(xtest)

  gbm = lgb.LGBMRegressor(learning_rate=0.001, max_depth=10, n_estimators=1000)
  gbm.fit(xtrain,ytrain)
  yhat56 = gbm.predict(xtest)


  gbm = lgb.LGBMRegressor(learning_rate=0.05, max_depth=2, n_estimators=1000)
  gbm.fit(xtrain,ytrain)
  yhat57 = gbm.predict(xtest)

  gbm = lgb.LGBMRegressor(learning_rate=0.05, max_depth=3, n_estimators=1000)
  gbm.fit(xtrain,ytrain)
  yhat58 = gbm.predict(xtest)

  gbm = lgb.LGBMRegressor(learning_rate=0.05, max_depth=5, n_estimators=1000)
  gbm.fit(xtrain,ytrain)
  yhat59 = gbm.predict(xtest)

  gbm = lgb.LGBMRegressor(learning_rate=0.05, max_depth=10, n_estimators=1000)
  gbm.fit(xtrain,ytrain)
  yhat60 = gbm.predict(xtest)


  gbm = lgb.LGBMRegressor(learning_rate=0.01, max_depth=2, n_estimators=1000)
  gbm.fit(xtrain,ytrain)
  yhat61 = gbm.predict(xtest)

  gbm = lgb.LGBMRegressor(learning_rate=0.01, max_depth=3, n_estimators=1000)
  gbm.fit(xtrain,ytrain)
  yhat62 = gbm.predict(xtest)

  gbm = lgb.LGBMRegressor(learning_rate=0.01, max_depth=5, n_estimators=1000)
  gbm.fit(xtrain,ytrain)
  yhat63 = gbm.predict(xtest)

  gbm = lgb.LGBMRegressor(learning_rate=0.01, max_depth=10, n_estimators=1000)
  gbm.fit(xtrain,ytrain)
  yhat64 = gbm.predict(xtest)


  gbm = lgb.LGBMRegressor(learning_rate=0.5, max_depth=2, n_estimators=1000)
  gbm.fit(xtrain,ytrain)
  yhat65 = gbm.predict(xtest)

  gbm = lgb.LGBMRegressor(learning_rate=0.5, max_depth=3, n_estimators=1000)
  gbm.fit(xtrain,ytrain)
  yhat66 = gbm.predict(xtest)

  gbm = lgb.LGBMRegressor(learning_rate=0.5, max_depth=5, n_estimators=1000)
  gbm.fit(xtrain,ytrain)
  yhat67 = gbm.predict(xtest)

  gbm = lgb.LGBMRegressor(learning_rate=0.5, max_depth=10, n_estimators=1000)
  gbm.fit(xtrain,ytrain)
  yhat68 = gbm.predict(xtest)


  gbm = lgb.LGBMRegressor(learning_rate=0.1, max_depth=2, n_estimators=1000)
  gbm.fit(xtrain,ytrain)
  yhat69 = gbm.predict(xtest)

  gbm = lgb.LGBMRegressor(learning_rate=0.1, max_depth=3, n_estimators=1000)
  gbm.fit(xtrain,ytrain)
  yhat70 = gbm.predict(xtest)

  gbm = lgb.LGBMRegressor(learning_rate=0.1, max_depth=5, n_estimators=1000)
  gbm.fit(xtrain,ytrain)
  yhat71 = gbm.predict(xtest)

  gbm = lgb.LGBMRegressor(learning_rate=0.1, max_depth=10, n_estimators=1000)
  gbm.fit(xtrain,ytrain)
  yhat72 = gbm.predict(xtest)

error_depth2l005.append(mse(ytest, yhat1))
error_depth3l005.append(mse(ytest, yhat2))
error_depth5l005.append(mse(ytest, yhat3))
error_depth10l005.append(mse(ytest, yhat4))

error_depth2l001.append(mse(ytest, yhat5))
error_depth3l001.append(mse(ytest, yhat6))
error_depth5l001.append(mse(ytest, yhat7))
error_depth10l001.append(mse(ytest, yhat8))

error_depth2l05.append(mse(ytest, yhat9))
error_depth3l05.append(mse(ytest, yhat10))
error_depth5l05.append(mse(ytest, yhat11))
error_depth10l05.append(mse(ytest, yhat12))

error_depth2l01.append(mse(ytest, yhat13))
error_depth3l01.append(mse(ytest, yhat14))
error_depth5l01.append(mse(ytest, yhat15))
error_depth10l01.append(mse(ytest, yhat16))

error_depth2l5.append(mse(ytest, yhat17))
error_depth3l5.append(mse(ytest, yhat18))
error_depth5l5.append(mse(ytest, yhat19))
error_depth10l5.append(mse(ytest, yhat20))

error_depth2l1.append(mse(ytest, yhat21))
error_depth3l1.append(mse(ytest, yhat22))
error_depth5l1.append(mse(ytest, yhat23))
error_depth10l1.append(mse(ytest, yhat24))


error_depth2l005_500.append(mse(ytest, yhat25))
error_depth3l005_500.append(mse(ytest, yhat26))
error_depth5l005_500.append(mse(ytest, yhat27))
error_depth10l005_500.append(mse(ytest, yhat28))

error_depth2l001_500.append(mse(ytest, yhat29))
error_depth3l001_500.append(mse(ytest, yhat30))
error_depth5l001_500.append(mse(ytest, yhat31))
error_depth10l001_500.append(mse(ytest, yhat32))

error_depth2l05_500.append(mse(ytest, yhat33))
error_depth3l05_500.append(mse(ytest, yhat34))
error_depth5l05_500.append(mse(ytest, yhat35))
error_depth10l05_500.append(mse(ytest, yhat36))

error_depth2l01_500.append(mse(ytest, yhat37))
error_depth3l01_500.append(mse(ytest, yhat38))
error_depth5l01_500.append(mse(ytest, yhat49))
error_depth10l01_500.append(mse(ytest, yhat40))

error_depth2l5_500.append(mse(ytest, yhat41))
error_depth3l5_500.append(mse(ytest, yhat42))
error_depth5l5_500.append(mse(ytest, yhat43))
error_depth10l5_500.append(mse(ytest, yhat44))

error_depth2l1_500.append(mse(ytest, yhat45))
error_depth3l1_500.append(mse(ytest, yhat46))
error_depth5l1_500.append(mse(ytest, yhat47))
error_depth10l1_500.append(mse(ytest, yhat48))



error_depth2l005_1000.append(mse(ytest, yhat49))
error_depth3l005_1000.append(mse(ytest, yhat50))
error_depth5l005_1000.append(mse(ytest, yhat51))
error_depth10l005_1000.append(mse(ytest, yhat52))

error_depth2l001_1000.append(mse(ytest, yhat53))
error_depth3l001_1000.append(mse(ytest, yhat54))
error_depth5l001_1000.append(mse(ytest, yhat55))
error_depth10l001_1000.append(mse(ytest, yhat56))

error_depth2l05_1000.append(mse(ytest, yhat57))
error_depth3l05_1000.append(mse(ytest, yhat58))
error_depth5l05_1000.append(mse(ytest, yhat59))
error_depth10l05_1000.append(mse(ytest, yhat60))

error_depth2l01_1000.append(mse(ytest, yhat61))
error_depth3l01_1000.append(mse(ytest, yhat62))
error_depth5l01_1000.append(mse(ytest, yhat63))
error_depth10l01_1000.append(mse(ytest, yhat64))

error_depth2l5_1000.append(mse(ytest, yhat65))
error_depth3l5_1000.append(mse(ytest, yhat66))
error_depth5l5_1000.append(mse(ytest, yhat67))
error_depth10l5_1000.append(mse(ytest, yhat68))

error_depth2l1_1000.append(mse(ytest, yhat69))
error_depth3l1_1000.append(mse(ytest, yhat70))
error_depth5l1_1000.append(mse(ytest, yhat71))
error_depth10l1_1000.append(mse(ytest, yhat72))





error_depth2l005.append(mae(ytest, yhat1))
error_depth3l005.append(mae(ytest, yhat2))
error_depth5l005.append(mae(ytest, yhat3))
error_depth10l005.append(mae(ytest, yhat4))

error_depth2l001.append(mae(ytest, yhat5))
error_depth3l001.append(mae(ytest, yhat6))
error_depth5l001.append(mae(ytest, yhat7))
error_depth10l001.append(mae(ytest, yhat8))

error_depth2l05.append(mae(ytest, yhat9))
error_depth3l05.append(mae(ytest, yhat10))
error_depth5l05.append(mae(ytest, yhat11))
error_depth10l05.append(mae(ytest, yhat12))

error_depth2l01.append(mae(ytest, yhat13))
error_depth3l01.append(mae(ytest, yhat14))
error_depth5l01.append(mae(ytest, yhat15))
error_depth10l01.append(mae(ytest, yhat16))

error_depth2l5.append(mae(ytest, yhat17))
error_depth3l5.append(mae(ytest, yhat18))
error_depth5l5.append(mae(ytest, yhat19))
error_depth10l5.append(mae(ytest, yhat20))

error_depth2l1.append(mae(ytest, yhat21))
error_depth3l1.append(mae(ytest, yhat22))
error_depth5l1.append(mae(ytest, yhat23))
error_depth10l1.append(mae(ytest, yhat24))


error_depth2l005_500.append(mae(ytest, yhat25))
error_depth3l005_500.append(mae(ytest, yhat26))
error_depth5l005_500.append(mae(ytest, yhat27))
error_depth10l005_500.append(mae(ytest, yhat28))

error_depth2l001_500.append(mae(ytest, yhat29))
error_depth3l001_500.append(mae(ytest, yhat30))
error_depth5l001_500.append(mae(ytest, yhat31))
error_depth10l001_500.append(mae(ytest, yhat32))

error_depth2l05_500.append(mae(ytest, yhat33))
error_depth3l05_500.append(mae(ytest, yhat34))
error_depth5l05_500.append(mae(ytest, yhat35))
error_depth10l05_500.append(mae(ytest, yhat36))

error_depth2l01_500.append(mae(ytest, yhat37))
error_depth3l01_500.append(mae(ytest, yhat38))
error_depth5l01_500.append(mae(ytest, yhat49))
error_depth10l01_500.append(mae(ytest, yhat40))

error_depth2l5_500.append(mae(ytest, yhat41))
error_depth3l5_500.append(mae(ytest, yhat42))
error_depth5l5_500.append(mae(ytest, yhat43))
error_depth10l5_500.append(mae(ytest, yhat44))

error_depth2l1_500.append(mae(ytest, yhat45))
error_depth3l1_500.append(mae(ytest, yhat46))
error_depth5l1_500.append(mae(ytest, yhat47))
error_depth10l1_500.append(mae(ytest, yhat48))



error_depth2l005_1000.append(mae(ytest, yhat49))
error_depth3l005_1000.append(mae(ytest, yhat50))
error_depth5l005_1000.append(mae(ytest, yhat51))
error_depth10l005_1000.append(mae(ytest, yhat52))

error_depth2l001_1000.append(mae(ytest, yhat53))
error_depth3l001_1000.append(mae(ytest, yhat54))
error_depth5l001_1000.append(mae(ytest, yhat55))
error_depth10l001_1000.append(mae(ytest, yhat56))

error_depth2l05_1000.append(mae(ytest, yhat57))
error_depth3l05_1000.append(mae(ytest, yhat58))
error_depth5l05_1000.append(mae(ytest, yhat59))
error_depth10l05_1000.append(mae(ytest, yhat60))

error_depth2l01_1000.append(mae(ytest, yhat61))
error_depth3l01_1000.append(mae(ytest, yhat62))
error_depth5l01_1000.append(mae(ytest, yhat63))
error_depth10l01_1000.append(mae(ytest, yhat64))

error_depth2l5_1000.append(mae(ytest, yhat65))
error_depth3l5_1000.append(mae(ytest, yhat66))
error_depth5l5_1000.append(mae(ytest, yhat67))
error_depth10l5_1000.append(mae(ytest, yhat68))

error_depth2l1_1000.append(mae(ytest, yhat69))
error_depth3l1_1000.append(mae(ytest, yhat70))
error_depth5l1_1000.append(mae(ytest, yhat71))
error_depth10l1_1000.append(mae(ytest, yhat72))





error_depth2l005.append(le(ytest, yhat1))
error_depth3l005.append(le(ytest, yhat2))
error_depth5l005.append(le(ytest, yhat3))
error_depth10l005.append(le(ytest, yhat4))

error_depth2l001.append(le(ytest, yhat5))
error_depth3l001.append(le(ytest, yhat6))
error_depth5l001.append(le(ytest, yhat7))
error_depth10l001.append(le(ytest, yhat8))

error_depth2l05.append(le(ytest, yhat9))
error_depth3l05.append(le(ytest, yhat10))
error_depth5l05.append(le(ytest, yhat11))
error_depth10l05.append(le(ytest, yhat12))

error_depth2l01.append(le(ytest, yhat13))
error_depth3l01.append(le(ytest, yhat14))
error_depth5l01.append(le(ytest, yhat15))
error_depth10l01.append(le(ytest, yhat16))

error_depth2l5.append(le(ytest, yhat17))
error_depth3l5.append(le(ytest, yhat18))
error_depth5l5.append(le(ytest, yhat19))
error_depth10l5.append(le(ytest, yhat20))

error_depth2l1.append(le(ytest, yhat21))
error_depth3l1.append(le(ytest, yhat22))
error_depth5l1.append(le(ytest, yhat23))
error_depth10l1.append(le(ytest, yhat24))


error_depth2l005_500.append(le(ytest, yhat25))
error_depth3l005_500.append(le(ytest, yhat26))
error_depth5l005_500.append(le(ytest, yhat27))
error_depth10l005_500.append(le(ytest, yhat28))

error_depth2l001_500.append(le(ytest, yhat29))
error_depth3l001_500.append(le(ytest, yhat30))
error_depth5l001_500.append(le(ytest, yhat31))
error_depth10l001_500.append(le(ytest, yhat32))

error_depth2l05_500.append(le(ytest, yhat33))
error_depth3l05_500.append(le(ytest, yhat34))
error_depth5l05_500.append(le(ytest, yhat35))
error_depth10l05_500.append(le(ytest, yhat36))

error_depth2l01_500.append(le(ytest, yhat37))
error_depth3l01_500.append(le(ytest, yhat38))
error_depth5l01_500.append(le(ytest, yhat49))
error_depth10l01_500.append(le(ytest, yhat40))

error_depth2l5_500.append(le(ytest, yhat41))
error_depth3l5_500.append(le(ytest, yhat42))
error_depth5l5_500.append(le(ytest, yhat43))
error_depth10l5_500.append(le(ytest, yhat44))

error_depth2l1_500.append(le(ytest, yhat45))
error_depth3l1_500.append(le(ytest, yhat46))
error_depth5l1_500.append(le(ytest, yhat47))
error_depth10l1_500.append(le(ytest, yhat48))



error_depth2l005_1000.append(le(ytest, yhat49))
error_depth3l005_1000.append(le(ytest, yhat50))
error_depth5l005_1000.append(le(ytest, yhat51))
error_depth10l005_1000.append(le(ytest, yhat52))

error_depth2l001_1000.append(le(ytest, yhat53))
error_depth3l001_1000.append(le(ytest, yhat54))
error_depth5l001_1000.append(le(ytest, yhat55))
error_depth10l001_1000.append(le(ytest, yhat56))

error_depth2l05_1000.append(le(ytest, yhat57))
error_depth3l05_1000.append(le(ytest, yhat58))
error_depth5l05_1000.append(le(ytest, yhat59))
error_depth10l05_1000.append(le(ytest, yhat60))

error_depth2l01_1000.append(le(ytest, yhat61))
error_depth3l01_1000.append(le(ytest, yhat62))
error_depth5l01_1000.append(le(ytest, yhat63))
error_depth10l01_1000.append(le(ytest, yhat64))

error_depth2l5_1000.append(le(ytest, yhat65))
error_depth3l5_1000.append(le(ytest, yhat66))
error_depth5l5_1000.append(le(ytest, yhat67))
error_depth10l5_1000.append(le(ytest, yhat68))

error_depth2l1_1000.append(le(ytest, yhat69))
error_depth3l1_1000.append(le(ytest, yhat70))
error_depth5l1_1000.append(le(ytest, yhat71))
error_depth10l1_1000.append(le(ytest, yhat72))




print("The MAE for 100 trees, depth of 2, and 0.005 learning rate is: " + str(np.mean(error_depth2l005[1])))
print("The MAE for 100 trees, depth of 3, and 0.005 learning rate is: " + str(np.mean(error_depth3l005[1])))
print("The MAE for 100 trees, depth of 5, and 0.005 learning rate is: " + str(np.mean(error_depth5l005[1])))
print("The MAE for 100 trees, depth of 10, and 0.005 learning rate is: " + str(np.mean(error_depth10l005[1])))

print("The MAE for 100 trees, depth of 2, and 0.01 learning rate is: " + str(np.mean(error_depth2l01[1])))
print("The MAE for 100 trees, depth of 3, and 0.01 learning rate is: " + str(np.mean(error_depth3l01[1])))
print("The MAE for 100 trees, depth of 5, and 0.01 learning rate is: " + str(np.mean(error_depth5l01[1])))
print("The MAE for 100 trees, depth of 10, and 0.01 learning rate is: " + str(np.mean(error_depth10l01[1])))

print("The MAE for 100 trees, depth of 2, and 0.001 learning rate is: " + str(np.mean(error_depth2l001[1])))
print("The MAE for 100 trees, depth of 3, and 0.001 learning rate is: " + str(np.mean(error_depth3l001[1])))
print("The MAE for 100 trees, depth of 5, and 0.001 learning rate is: " + str(np.mean(error_depth5l001[1])))
print("The MAE for 100 trees, depth of 10, and 0.001 learning rate is: " + str(np.mean(error_depth10l001[1])))

print("The MAE for 100 trees, depth of 2, and 0.05 learning rate is: " + str(np.mean(error_depth2l05[1])))
print("The MAE for 100 trees, depth of 3, and 0.05 learning rate is: " + str(np.mean(error_depth3l05[1])))
print("The MAE for 100 trees, depth of 5, and 0.05 learning rate is: " + str(np.mean(error_depth5l05[1])))
print("The MAE for 100 trees, depth of 10, and 0.05 learning rate is: " + str(np.mean(error_depth10l05[1])))

print("The MAE for 100 trees, depth of 2, and 0.1 learning rate is: " + str(np.mean(error_depth2l1[1])))
print("The MAE for 100 trees, depth of 3, and 0.1 learning rate is: " + str(np.mean(error_depth3l1[1])))
print("The MAE for 100 trees, depth of 5, and 0.1 learning rate is: " + str(np.mean(error_depth5l1[1])))
print("The MAE for 100 trees, depth of 10, and 0.1 learning rate is: " + str(np.mean(error_depth10l1[1])))

print("The MAE for 100 trees, depth of 2, and 0.5 learning rate is: " + str(np.mean(error_depth2l5[1])))
print("The MAE for 100 trees, depth of 3, and 0.5 learning rate is: " + str(np.mean(error_depth3l5[1])))
print("The MAE for 100 trees, depth of 5, and 0.5 learning rate is: " + str(np.mean(error_depth5l5[1])))
print("The MAE for 100 trees, depth of 10, and 0.5 learning rate is: " + str(np.mean(error_depth10l5[1])))


print("The MAE for 500 trees, depth of 2, and 0.005 learning rate is: " + str(np.mean(error_depth2l005_500[1])))
print("The MAE for 500 trees, depth of 3, and 0.005 learning rate is: " + str(np.mean(error_depth3l005_500[1])))
print("The MAE for 500 trees, depth of 5, and 0.005 learning rate is: " + str(np.mean(error_depth5l005_500[1])))
print("The MAE for 500 trees, depth of 10, and 0.005 learning rate is: " + str(np.mean(error_depth10l005_500[1])))

print("The MAE for 500 trees, depth of 2, and 0.01 learning rate is: " + str(np.mean(error_depth2l01_500[1])))
print("The MAE for 500 trees, depth of 3, and 0.01 learning rate is: " + str(np.mean(error_depth3l01_500[1])))
print("The MAE for 500 trees, depth of 5, and 0.01 learning rate is: " + str(np.mean(error_depth5l01_500[1])))
print("The MAE for 500 trees, depth of 10, and 0.01 learning rate is: " + str(np.mean(error_depth10l01_500[1])))

print("The MAE for 500 trees, depth of 2, and 0.001 learning rate is: " + str(np.mean(error_depth2l001_500[1])))
print("The MAE for 500 trees, depth of 3, and 0.001 learning rate is: " + str(np.mean(error_depth3l001_500[1])))
print("The MAE for 500 trees, depth of 5, and 0.001 learning rate is: " + str(np.mean(error_depth5l001_500[1])))
print("The MAE for 500 trees, depth of 10, and 0.001 learning rate is: " + str(np.mean(error_depth10l001_500[1])))

print("The MAE for 500 trees, depth of 2, and 0.05 learning rate is: " + str(np.mean(error_depth2l05_500[1])))
print("The MAE for 500 trees, depth of 3, and 0.05 learning rate is: " + str(np.mean(error_depth3l05_500[1])))
print("The MAE for 500 trees, depth of 5, and 0.05 learning rate is: " + str(np.mean(error_depth5l05_500[1])))
print("The MAE for 500 trees, depth of 10, and 0.05 learning rate is: " + str(np.mean(error_depth10l05_500[1])))

print("The MAE for 500 trees, depth of 2, and 0.1 learning rate is: " + str(np.mean(error_depth2l1_500[1])))
print("The MAE for 500 trees, depth of 3, and 0.1 learning rate is: " + str(np.mean(error_depth3l1_500[1])))
print("The MAE for 500 trees, depth of 5, and 0.1 learning rate is: " + str(np.mean(error_depth5l1_500[1])))
print("The MAE for 500 trees, depth of 10, and 0.1 learning rate is: " + str(np.mean(error_depth10l1_500[1])))

print("The MAE for 500 trees, depth of 2, and 0.5 learning rate is: " + str(np.mean(error_depth2l5_500[1])))
print("The MAE for 500 trees, depth of 3, and 0.5 learning rate is: " + str(np.mean(error_depth3l5_500[1])))
print("The MAE for 500 trees, depth of 5, and 0.5 learning rate is: " + str(np.mean(error_depth5l5_500[1])))
print("The MAE for 500 trees, depth of 5, and 0.5 learning rate is: " + str(np.mean(error_depth10l5_500[1])))



print("The MAE for 1000 trees, depth of 2, and 0.005 learning rate is: " + str(np.mean(error_depth2l005_1000[1])))
print("The MAE for 1000 trees, depth of 3, and 0.005 learning rate is: " + str(np.mean(error_depth3l005_1000[1])))
print("The MAE for 1000 trees, depth of 5, and 0.005 learning rate is: " + str(np.mean(error_depth5l005_1000[1])))
print("The MAE for 1000 trees, depth of 10, and 0.005 learning rate is: " + str(np.mean(error_depth10l005_1000[1])))

print("The MAE for 1000 trees, depth of 2, and 0.01 learning rate is: " + str(np.mean(error_depth2l01_1000[1])))
print("The MAE for 1000 trees, depth of 3, and 0.01 learning rate is: " + str(np.mean(error_depth3l01_1000[1])))
print("The MAE for 1000 trees, depth of 5, and 0.01 learning rate is: " + str(np.mean(error_depth5l01_1000[1])))
print("The MAE for 1000 trees, depth of 10, and 0.01 learning rate is: " + str(np.mean(error_depth10l01_1000[1])))

print("The MAE for 1000 trees, depth of 2, and 0.001 learning rate is: " + str(np.mean(error_depth2l001_1000[1])))
print("The MAE for 1000 trees, depth of 3, and 0.001 learning rate is: " + str(np.mean(error_depth3l001_1000[1])))
print("The MAE for 1000 trees, depth of 5, and 0.001 learning rate is: " + str(np.mean(error_depth5l001_1000[1])))
print("The MAE for 1000 trees, depth of 10, and 0.001 learning rate is: " + str(np.mean(error_depth10l001_1000[1])))

print("The MAE for 1000 trees, depth of 2, and 0.05 learning rate is: " + str(np.mean(error_depth2l05_1000[1])))
print("The MAE for 1000 trees, depth of 3, and 0.05 learning rate is: " + str(np.mean(error_depth3l05_1000[1])))
print("The MAE for 1000 trees, depth of 5, and 0.05 learning rate is: " + str(np.mean(error_depth5l05_1000[1])))
print("The MAE for 1000 trees, depth of 10, and 0.05 learning rate is: " + str(np.mean(error_depth10l05_1000[1])))

print("The MAE for 1000 trees, depth of 2, and 0.1 learning rate is: " + str(np.mean(error_depth2l1_1000[1])))
print("The MAE for 1000 trees, depth of 3, and 0.1 learning rate is: " + str(np.mean(error_depth3l1_1000[1])))
print("The MAE for 1000 trees, depth of 5, and 0.1 learning rate is: " + str(np.mean(error_depth5l1_1000[1])))
print("The MAE for 1000 trees, depth of 10, and 0.1 learning rate is: " + str(np.mean(error_depth10l1_1000[1])))

print("The MAE for 1000 trees, depth of 2, and 0.5 learning rate is: " + str(np.mean(error_depth2l5_1000[1])))
print("The MAE for 1000 trees, depth of 3, and 0.5 learning rate is: " + str(np.mean(error_depth3l5_1000[1])))
print("The MAE for 1000 trees, depth of 5, and 0.5 learning rate is: " + str(np.mean(error_depth5l5_1000[1])))
print("The MAE for 1000 trees, depth of 10, and 0.5 learning rate is: " + str(np.mean(error_depth10l5_1000[1])))


print("\n")


print("The log error for 100 trees, depth of 2, and 0.005 learning rate is: " + str(np.mean(error_depth2l005[2])))
print("The log error for 100 trees, depth of 3, and 0.005 learning rate is: " + str(np.mean(error_depth3l005[2])))
print("The log error for 100 trees, depth of 5, and 0.005 learning rate is: " + str(np.mean(error_depth5l005[2])))
print("The log error for 100 trees, depth of 10, and 0.005 learning rate is: " + str(np.mean(error_depth10l005[2])))

print("The log error for 100 trees, depth of 2, and 0.01 learning rate is: " + str(np.mean(error_depth2l01[2])))
print("The log error for 100 trees, depth of 3, and 0.01 learning rate is: " + str(np.mean(error_depth3l01[2])))
print("The log error for 100 trees, depth of 5, and 0.01 learning rate is: " + str(np.mean(error_depth5l01[2])))
print("The log error for 100 trees, depth of 10, and 0.01 learning rate is: " + str(np.mean(error_depth10l01[2])))

print("The log error for 100 trees, depth of 2, and 0.001 learning rate is: " + str(np.mean(error_depth2l001[2])))
print("The log error for 100 trees, depth of 3, and 0.001 learning rate is: " + str(np.mean(error_depth3l001[2])))
print("The log error for 100 trees, depth of 5, and 0.001 learning rate is: " + str(np.mean(error_depth5l001[2])))
print("The log error for 100 trees, depth of 10, and 0.001 learning rate is:  " + str(np.mean(error_depth10l001[2])))

print("The log error for 100 trees, depth of 2, and 0.05 learning rate is: " + str(np.mean(error_depth2l05[2])))
print("The log error for 100 trees, depth of 3, and 0.05 learning rate is: " + str(np.mean(error_depth3l05[2])))
print("The log error for 100 trees, depth of 5, and 0.05 learning rate is: " + str(np.mean(error_depth5l05[2])))
print("The log error for 100 trees, depth of 10, and 0.05 learning rate is: " + str(np.mean(error_depth10l05[2])))

print("The log error for 100 trees, depth of 2, and 0.1 learning rate is: " + str(np.mean(error_depth2l1[2])))
print("The log error for 100 trees, depth of 3, and 0.1 learning rate is: " + str(np.mean(error_depth3l1[2])))
print("The log error for 100 trees, depth of 4, and 0.1 learning rate is: " + str(np.mean(error_depth5l1[2])))
print("The log error for 100 trees, depth of 5, and 0.1 learning rate is: " + str(np.mean(error_depth10l1[2])))

print("The log error for 100 trees, depth of 2, and 0.5 learning rate is: " + str(np.mean(error_depth2l5[2])))
print("The log error for 100 trees, depth of 3, and 0.5 learning rate is: " + str(np.mean(error_depth3l5[2])))
print("The log error for 100 trees, depth of 5, and 0.5 learning rate is: " + str(np.mean(error_depth5l5[2])))
print("The log error for 100 trees, depth of 10, and 0.5 learning rate is: " + str(np.mean(error_depth10l5[2])))


print("The log error for 500 trees, depth of 2, and 0.005 learning rate is: " + str(np.mean(error_depth2l005_500[2])))
print("The log error for 500 trees, depth of 3, and 0.005 learning rate is: " + str(np.mean(error_depth3l005_500[2])))
print("The log error for 500 trees, depth of 5, and 0.005 learning rate is: " + str(np.mean(error_depth5l005_500[2])))
print("The log error for 500 trees, depth of 10, and 0.005 learning rate is: " + str(np.mean(error_depth10l005_500[2])))

print("The log error for 500 trees, depth of 2, and 0.01 learning rate is:  " + str(np.mean(error_depth2l01_500[2])))
print("The log error for 500 trees, depth of 3, and 0.01 learning rate is:  " + str(np.mean(error_depth3l01_500[2])))
print("The log error for 500 trees, depth of 5, and 0.01 learning rate is: " + str(np.mean(error_depth5l01_500[2])))
print("The log error for 500 trees, depth of 10, and 0.01 learning rate is: " + str(np.mean(error_depth10l01_500[2])))

print("The log error for 500 trees, depth of 2, and 0.001 learning rate is: " + str(np.mean(error_depth2l001_500[2])))
print("The log error for 500 trees, depth of 3, and 0.001 learning rate is: " + str(np.mean(error_depth3l001_500[2])))
print("The log error for 500 trees, depth of 5, and 0.001 learning rate is: " + str(np.mean(error_depth5l001_500[2])))
print("The log error for 500 trees, depth of 10, and 0.001 learning rate is: " + str(np.mean(error_depth10l001_500[2])))

print("The log error for 500 trees, depth of 2, and 0.05 learning rate is: " + str(np.mean(error_depth2l05_500[2])))
print("The log error for 500 trees, depth of 3, and 0.05 learning rate is: " + str(np.mean(error_depth3l05_500[2])))
print("The log error for 500 trees, depth of 5, and 0.05 learning rate is: " + str(np.mean(error_depth5l05_500[2])))
print("The log error for 500 trees, depth of 10, and 0.05 learning rate is: " + str(np.mean(error_depth10l05_500[2])))

print("The log error for 500 trees, depth of 2, and 0.1 learning rate is: " + str(np.mean(error_depth2l1_500[2])))
print("The log error for 500 trees, depth of 3, and 0.1 learning rate is: " + str(np.mean(error_depth3l1_500[2])))
print("The log error for 500 trees, depth of 5, and 0.1 learning rate is: " + str(np.mean(error_depth5l1_500[2])))
print("The log error for 500 trees, depth of 10, and 0.1 learning rate is: " + str(np.mean(error_depth10l1_500[2])))

print("The log error for 500 trees, depth of 2, and 0.5 learning rate is: " + str(np.mean(error_depth2l5_500[2])))
print("The log error for 500 trees, depth of 3, and 0.5 learning rate is: " + str(np.mean(error_depth3l5_500[2])))
print("The log error for 500 trees, depth of 5, and 0.5 learning rate is: " + str(np.mean(error_depth5l5_500[2])))
print("The log error for 500 trees, depth of 10 , and 0.5 learning rate is: " + str(np.mean(error_depth10l5_500[2])))



print("The log error for 1000 trees, depth of 2, and 0.005 learning rate is: " + str(np.mean(error_depth2l005_1000[2])))
print("The log error for 1000 trees, depth of 3, and 0.005 learning rate is: " + str(np.mean(error_depth3l005_1000[2])))
print("The log error for 1000 trees, depth of 5, and 0.005 learning rate is: " + str(np.mean(error_depth5l005_1000[2])))
print("The log error for 1000 trees, depth of 10, and 0.005 learning rate is: " + str(np.mean(error_depth10l005_1000[2])))

print("The log error for 1000 trees, depth of 2, and 0.01 learning rate is: " + str(np.mean(error_depth2l01_1000[2])))
print("The log error for 1000 trees, depth of 3, and 0.01 learning rate is: " + str(np.mean(error_depth3l01_1000[2])))
print("The log error for 1000 trees, depth of 5, and 0.01 learning rate is: " + str(np.mean(error_depth5l01_1000[2])))
print("The log error for 1000 trees, depth of 10, and 0.01 learning rate is: " + str(np.mean(error_depth10l01_1000[2])))

print("The log error for 1000 trees, depth of 2, and 0.001 learning rate is: " + str(np.mean(error_depth2l001_1000[2])))
print("The log error for 1000 trees, depth of 3, and 0.001 learning rate is: " + str(np.mean(error_depth3l001_1000[2])))
print("The log error for 1000 trees, depth of 5, and 0.001 learning rate is: " + str(np.mean(error_depth5l001_1000[2])))
print("The log error for 1000 trees, depth of 10, and 0.001 learning rate is: " + str(np.mean(error_depth10l001_1000[2])))

print("The log error for 1000 trees, depth of 2, and 0.05 learning rate is: " + str(np.mean(error_depth2l05_1000[2])))
print("The log error for 1000 trees, depth of 3, and 0.05 learning rate is: " + str(np.mean(error_depth3l05_1000[2])))
print("The log error for 1000 trees, depth of 5, and 0.05 learning rate is: " + str(np.mean(error_depth5l05_1000[2])))
print("The log error for 1000 trees, depth of 10, and 0.05 learning rate is: " + str(np.mean(error_depth10l05_1000[2])))

print("The log error for 1000 trees, depth of 2, and 0.1 learning rate is: " + str(np.mean(error_depth2l1_1000[2])))
print("The log error for 1000 trees, depth of 3, and 0.1 learning rate is: " + str(np.mean(error_depth3l1_1000[2])))
print("The log error for 1000 trees, depth of 5, and 0.1 learning rate is: " + str(np.mean(error_depth5l1_1000[2])))
print("The log error for 1000 trees, depth of 10, and 0.1 learning rate is: " + str(np.mean(error_depth10l1_1000[2])))

print("The log error for 1000 trees, depth of 2, and 0.5 learning rate is: " + str(np.mean(error_depth2l5_1000[2])))
print("The log error for 1000 trees, depth of 3, and 0.5 learning rate is: " + str(np.mean(error_depth3l5_1000[2])))
print("The log error for 1000 trees, depth of 5, and 0.5 learning rate is: " + str(np.mean(error_depth5l5_1000[2])))
print("The log error for 1000 trees, depth of 10, and 0.5 learning rate is: " + str(np.mean(error_depth10l5_1000[2])))


print("\n")


print("The MSE for 100 trees, depth of 2, and 0.005 learning rate is: " + str(np.mean(error_depth2l005[0])))
print("The MSE for 100 trees, depth of 3, and 0.005 learning rate is: " + str(np.mean(error_depth3l005[0])))
print("The MSE for 100 trees, depth of 5, and 0.005 learning rate is: " + str(np.mean(error_depth5l005[0])))
print("The MSE for 100 trees, depth of 10, and 0.005 learning rate is: " + str(np.mean(error_depth10l005[0])))

print("The MSE for 100 trees, depth of 2, and 0.01 learning rate is: " + str(np.mean(error_depth2l01[0])))
print("The MSE for 100 trees, depth of 3, and 0.01 learning rate is: " + str(np.mean(error_depth3l01[0])))
print("The MSE for 100 trees, depth of 5, and 0.01 learning rate is: " + str(np.mean(error_depth5l01[0])))
print("The MSE for 100 trees, depth of 19, and 0.01 learning rate is: " + str(np.mean(error_depth10l01[0])))

print("The MSE for 100 trees, depth of 2, and 0.001 learning rate is: " + str(np.mean(error_depth2l001[0])))
print("The MSE for 100 trees, depth of 3, and 0.001 learning rate is: " + str(np.mean(error_depth3l001[0])))
print("The MSE for 100 trees, depth of 5, and 0.001 learning rate is: " + str(np.mean(error_depth5l001[0])))
print("The MSE for 100 trees, depth of 10, and 0.001 learning rate is: " + str(np.mean(error_depth10l001[0])))

print("The MSE for 100 trees, depth of 2, and 0.05 learning rate is: " + str(np.mean(error_depth2l05[0])))
print("The MSE for 100 trees, depth of 3, and 0.05 learning rate is: " + str(np.mean(error_depth3l05[0])))
print("The MSE for 100 trees, depth of 5, and 0.05 learning rate is: " + str(np.mean(error_depth5l05[0])))
print("The MSE for 100 trees, depth of 10, and 0.05 learning rate is: " + str(np.mean(error_depth10l05[0])))

print("The MSE for 100 trees, depth of 2, and 0.1 learning rate is: " + str(np.mean(error_depth2l1[0])))
print("The MSE for 100 trees, depth of 3, and 0.1 learning rate is: " + str(np.mean(error_depth3l1[0])))
print("The MSE for 100 trees, depth of 5, and 0.1 learning rate is: " + str(np.mean(error_depth5l1[0])))
print("The MSE for 100 trees, depth of 10, and 0.1 learning rate is: " + str(np.mean(error_depth10l1[0])))

print("The MSE for 100 trees, depth of 2, and 0.5 learning rate is: " + str(np.mean(error_depth2l5[0])))
print("The MSE for 100 trees, depth of 3, and 0.5 learning rate is: " + str(np.mean(error_depth3l5[0])))
print("The MSE for 100 trees, depth of 5, and 0.5 learning rate is: " + str(np.mean(error_depth5l5[0])))
print("The MSE for 100 trees, depth of 10, and 0.5 learning rate is: " + str(np.mean(error_depth10l5[0])))


print("The MSE for 500 trees, depth of 2, and 0.005 learning rate is: " + str(np.mean(error_depth2l005_500[0])))
print("The MSE for 500 trees, depth of 3, and 0.005 learning rate is: " + str(np.mean(error_depth3l005_500[0])))
print("The MSE for 500 trees, depth of 5, and 0.005 learning rate is: " + str(np.mean(error_depth5l005_500[0])))
print("The MSE for 500 trees, depth of 10, and 0.005 learning rate is: " + str(np.mean(error_depth10l005_500[0])))

print("The MSE for 500 trees, depth of 2, and 0.01 learning rate is:  " + str(np.mean(error_depth2l01_500[0])))
print("The MSE for 500 trees, depth of 3, and 0.01 learning rate is:  " + str(np.mean(error_depth3l01_500[0])))
print("The MSE for 500 trees, depth of 5, and 0.01 learning rate is:  " + str(np.mean(error_depth5l01_500[0])))
print("The MSE for 500 trees, depth of 10, and 0.01 learning rate is:  " + str(np.mean(error_depth10l01_500[0])))

print("The MSE for 500 trees, depth of 2, and 0.001 learning rate is: " + str(np.mean(error_depth2l001_500[0])))
print("The MSE for 500 trees, depth of 3, and 0.001 learning rate is: " + str(np.mean(error_depth3l001_500[0])))
print("The MSE for 500 trees, depth of 5, and 0.001 learning rate is: " + str(np.mean(error_depth5l001_500[0])))
print("The MSE for 500 trees, depth of 10, and 0.001 learning rate is: " + str(np.mean(error_depth10l001_500[0])))

print("The MSE for 500 trees, depth of 2, and 0.05 learning rate is: " + str(np.mean(error_depth2l05_500[0])))
print("The MSE for 500 trees, depth of 3, and 0.05 learning rate is: " + str(np.mean(error_depth3l05_500[0])))
print("The MSE for 500 trees, depth of 5, and 0.05 learning rate is: " + str(np.mean(error_depth5l05_500[0])))
print("The MSE for 500 trees, depth of 10, and 0.05 learning rate is: " + str(np.mean(error_depth10l05_500[0])))

print("The MSE for 500 trees, depth of 2, and 0.1 learning rate is: " + str(np.mean(error_depth2l1_500[0])))
print("The MSE for 500 trees, depth of 3, and 0.1 learning rate is: " + str(np.mean(error_depth3l1_500[0])))
print("The MSE for 500 trees, depth of 5, and 0.1 learning rate is: " + str(np.mean(error_depth5l1_500[0])))
print("The MSE for 500 trees, depth of 10, and 0.1 learning rate is: " + str(np.mean(error_depth10l1_500[0])))

print("The MSE for 500 trees, depth of 2, and 0.5 learning rate is: " + str(np.mean(error_depth2l5_500[0])))
print("The MSE for 500 trees, depth of 3, and 0.5 learning rate is: " + str(np.mean(error_depth3l5_500[0])))
print("The MSE for 500 trees, depth of 5, and 0.5 learning rate is: " + str(np.mean(error_depth5l5_500[0])))
print("The MSE for 500 trees, depth of 10, and 0.5 learning rate is: " + str(np.mean(error_depth10l5_500[0])))



print("The MSE for 1000 trees, depth of 2, and 0.005 learning rate is: " + str(np.mean(error_depth2l005_1000[0])))
print("The MSE for 1000 trees, depth of 3, and 0.005 learning rate is: " + str(np.mean(error_depth3l005_1000[0])))
print("The MSE for 1000 trees, depth of 5, and 0.005 learning rate is: " + str(np.mean(error_depth5l005_1000[0])))
print("The MSE for 1000 trees, depth of 10, and 0.005 learning rate is: " + str(np.mean(error_depth10l005_1000[0])))

print("The MSE for 1000 trees, depth of 2, and 0.01 learning rate is: " + str(np.mean(error_depth2l01_1000[0])))
print("The MSE for 1000 trees, depth of 3, and 0.01 learning rate is: " + str(np.mean(error_depth3l01_1000[0])))
print("The MSE for 1000 trees, depth of 5, and 0.01 learning rate is: " + str(np.mean(error_depth5l01_1000[0])))
print("The MSE for 1000 trees, depth of 10, and 0.01 learning rate is: " + str(np.mean(error_depth10l01_1000[0])))

print("The MSE for 1000 trees, depth of 2, and 0.001 learning rate is: " + str(np.mean(error_depth2l001_1000[0])))
print("The MSE for 1000 trees, depth of 3, and 0.001 learning rate is: " + str(np.mean(error_depth3l001_1000[0])))
print("The MSE for 1000 trees, depth of 5, and 0.001 learning rate is: " + str(np.mean(error_depth5l001_1000[0])))
print("The MSE for 1000 trees, depth of 10, and 0.001 learning rate is: " + str(np.mean(error_depth10l001_1000[0])))

print("The MSE for 1000 trees, depth of 2, and 0.05 learning rate is: " + str(np.mean(error_depth2l05_1000[0])))
print("The MSE for 1000 trees, depth of 3, and 0.05 learning rate is: " + str(np.mean(error_depth3l05_1000[0])))
print("The MSE for 1000 trees, depth of 5, and 0.05 learning rate is: " + str(np.mean(error_depth5l05_1000[0])))
print("The MSE for 1000 trees, depth of 10, and 0.05 learning rate is: " + str(np.mean(error_depth10l05_1000[0])))

print("The MSE for 1000 trees, depth of 2, and 0.1 learning rate is: " + str(np.mean(error_depth2l1_1000[0])))
print("The MSE for 1000 trees, depth of 3, and 0.1 learning rate is: " + str(np.mean(error_depth3l1_1000[0])))
print("The MSE for 1000 trees, depth of 5, and 0.1 learning rate is: " + str(np.mean(error_depth5l1_1000[0])))
print("The MSE for 1000 trees, depth of 10, and 0.1 learning rate is: " + str(np.mean(error_depth10l1_1000[0])))

print("The MSE for 1000 trees, depth of 2, and 0.5 learning rate is: " + str(np.mean(error_depth2l5_1000[0])))
print("The MSE for 1000 trees, depth of 3, and 0.5 learning rate is: " + str(np.mean(error_depth3l5_1000[0])))
print("The MSE for 1000 trees, depth of 5, and 0.5 learning rate is: " + str(np.mean(error_depth5l5_1000[0])))
print("The MSE for 1000 trees, depth of 10, and 0.5 learning rate is: " + str(np.mean(error_depth10l5_1000[0])))

```

