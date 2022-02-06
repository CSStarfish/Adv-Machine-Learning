# Comparison of Locally Weighted and Random Forest Regression

## Concept of Locally Weighted Regression



## Concept of Random Forest Regression



### Locally Weighted Regression Applied To A Walmart Dataset

(Editing of code below in progress, I have uploaded its current state, so far:)
```
import numpy as np
import pandas as pd
from scipy import linalg
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
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
    f = interp1d(x, yest,fill_value='extrapolate')
    return f(xnew)
    
data = pd.read_csv("Walmart.csv")

x = data['Temperature'].values
y = data['Weekly_Sales'].values

xtrain, xtest, ytrain, ytest = tts(x, y, test_size = 0.25, random_state = 123)

scale = StandardScaler()
ytrain_scaled = scale.fit_transform(ytrain.reshape(-1,1))
ytest_scale = scale.transform(ytest.reshape(-1,1))

yhat_test_tri = lowess_reg(xtrain, ytrain_scaled, xtest, tricubic, 0.1)
print(mse(yhat_test_tri, ytest_scale))

yhat_test_epa = lowess_reg(xtrain, ytrain_scaled, xtest), Epanechnikov, 0.1)
print(mse(yhat_test_epa, ytest_scale))

yhat_test_quart = lowess_reg(xtrain, ytrain_scaled, xtest, Quartic, 0.1)
print(mse(yhat_test_quart, ytest_scale))

# Select the kernel with the lowest MSE for comparison

# Plot a bar chart with each method's MSE 
```

**MSE - Tricubic: 1.015528366246407**

The Tricubic kernel proved to have the highest accuracy on the Walmart dataset.  I also performed loess regression with a tricubic kernel and a lower tau of 0.01, but the MSE was negatively impacted and slightly increased to 1.0155283663056112.  As such, I decided to continue my analysis on the Walmart dataset with a tau of 0.1.


**MSE - Epanechnikov: 1.0155283662485854**

Similar to the Tricubic kernel with a smaller tau of 0.01, the MSE slightly increased to 1.0155283662961123.


**MSE - Quartic: 1.0155283662492287**
Unlike the Tricubic and Epanechnikov kernels, utilizing the smaller tau of 0.01 decreased the MSE to 1.015528366238946.  As such, a smaller tau improved the accuracy of the model when using a Quartic kernel.

### Random Forest Regression Applied To A Walmart Dataset

(Editing of code below in progress, I have uploaded its current state, so far:)
```
import numpy as np
import pandas as pd
from scipy import linalg
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, train_test_split as tts
from sklearn.metrics import mean_squared_error as mse
from sklearn.preprocessing import StandardScaler

data = pd.read_csv("Walmart.csv")

x = data['Temperature'].values
y = data['Weekly_Sales'].values

xtrain, xtest, ytrain, ytest = tts(x, y, test_size = 0.25, random_state = 123)

scale = StandardScaler()
ytrain_scaled = scale.fit_transform(ytrain.reshape(-1,1))
ytest_scale = scale.transform(ytest.reshape(-1,1))

rf = RandomForestRegressor(n_estimators=1000,max_depth=3)

rf.fit(xtrain.reshape(-1,1), ytrain_scaled.reshape(-1,1)
yhat_rf = rf.predict(xtest.reshape(-1,1))

print(mse(yhat_rf, ytest_scale))

# Plot a bar chart with each method's MSE 
```

**MSE: 1.0026378217724825**

### Locally Weighted Regression Applied To The Boston Housing Dataset

(Editing of code below in progress, I have uploaded its current state, so far:)
```
import numpy as np
import pandas as pd
from scipy import linalg
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
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
    f = interp1d(x.ravel(), yest.ravel(),fill_value='extrapolate')
    return f(xnew)
    
data = pd.read_csv("Boston Housing Prices.csv")

x = data['rooms'].values
y = data['cmedv'].values

xtrain, xtest, ytrain, ytest = tts(x, y, test_size = 0.25, random_state = 123)
scale = StandardScaler()
xtrain_scaled = scale.fit_transform(xtrain)
xtest_scale = scale.transform(xtest)

yhat_test_tri = lowess_reg(xtrain_scaled.ravel(), ytrain, xtest_scale.ravel(), tricubic, 0.1)
print(mse(yhat_test_tri, ytest))

yhat_test_epa = lowess_reg(xtrain_scaled.ravel(), ytrain, xtest_scale.ravel(), Epanechnikov, 0.1)
print(mse(yhat_test_epa, ytest))

yhat_test_quart = lowess_reg(xtrain_scaled.ravel(), ytrain, xtest_scale.ravel(), Quartic, 0.1)
print(mse(yhat_test_quart, ytest))

# Select the kernel with the lowest MSE for comparison

# Plot a bar chart with each method's MSE 
```

**MSE:**

### Random Forest Regression Applied To The Boston Housing Dataset

(Editing of code below in progress, I have uploaded its current state, so far:)
```
import numpy as np
import pandas as pd
from scipy import linalg
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, train_test_split as tts
from sklearn.metrics import mean_squared_error as mse
from sklearn.preprocessing import StandardScaler

data = pd.read_csv("Boston Housing Prices.csv")

x = data['rooms'].values
y = data['cmedv'].values

xtrain, xtest, ytrain, ytest = tts(x, y, test_size = 0.25, random_state = 123)
scale = StandardScaler()
xtrain_scaled = scale.fit_transform(xtrain)
xtest_scale = scale.transform(xtest)

rf = RandomForestRegressor(n_estimators=1000,max_depth=3)

rf.fit(xtrain_scaled, ytrain)
yhat_rf = rf.predict(xtest_scale)

print(mse(yhat_rf, ytest))

# Plot a bar chart with each method's MSE 
```

**MSE:**

### Locally Weighted Regression Applied To The Cars Dataset

(Editing of code below in progress, I have uploaded its current state, so far:)
```
import numpy as np
import pandas as pd
from scipy import linalg
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
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
    f = interp1d(x.ravel(), yest.ravel(),fill_value='extrapolate')
    return f(xnew)
    
data = pd.read_csv("cars.csv")

x = data['WGT'].values
y = data['MPG'].values

xtrain, xtest, ytrain, ytest = tts(x, y, test_size = 0.25, random_state = 123)
scale = StandardScaler()
xtrain_scaled = scale.fit_transform(xtrain)
xtest_scale = scale.transform(xtest)

yhat_test_tri = lowess_reg(xtrain_scaled.ravel(), ytrain, xtest_scale.ravel(), tricubic, 0.1)
print(mse(yhat_test_tri, ytest))

yhat_test_epa = lowess_reg(xtrain_scaled.ravel(), ytrain, xtest_scale.ravel(), Epanechnikov, 0.1)
print(mse(yhat_test_epa, ytest))

yhat_test_quart = lowess_reg(xtrain_scaled.ravel(), ytrain, xtest_scale.ravel(), Quartic, 0.1)
print(mse(yhat_test_quart, ytest))

# Select the kernel with the lowest MSE for comparison

# Plot a bar chart with each method's MSE 
```

**MSE:**

### Random Forest Regression Applied To The Cars Dataset

(Editing of code below in progress, I have uploaded its current state, so far:)
```
import numpy as np
import pandas as pd
from scipy import linalg
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, train_test_split as tts
from sklearn.metrics import mean_squared_error as mse
from sklearn.preprocessing import StandardScaler

data = pd.read_csv("cars.csv")

x = data['WGT'].values
y = data['MPG'].values

xtrain, xtest, ytrain, ytest = tts(x, y, test_size = 0.25, random_state = 123)
scale = StandardScaler()
xtrain_scaled = scale.fit_transform(xtrain)
xtest_scale = scale.transform(xtest)

rf = RandomForestRegressor(n_estimators=1000,max_depth=3)

rf.fit(xtrain_scaled, ytrain)
yhat_rf = rf.predict(xtest_scale)

print(mse(yhat_rf, ytest))

# Plot a bar chart with each method's MSE 
```

**MSE:**

