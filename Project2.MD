# Comparison of Locally Weighted and Random Forest Regression

## Concept of Locally Weighted Regression
The main idea of locally weighted regression is that the predictions are a linear combination of the observed/actual values from the dataset (i.e., the predicted y is a linear combination of the actual values of y).  It applies weights to each value proportional to the distance between the local datapoints, with weights closer to one the closer they are together and closer to zero the further away they are, thus accounting for variability in the dataset.  Linear regression stems from the following equation:
![](linearregression.jpg).  If we multiply the equation by a matrix of weights consisting of the transpose of X (i.e., the independent observations of the dataset) and solve for the beta coefficient, we will find that the expected value of beta is equal to the inverse of the product of X and the transpose of X, multiplied by the product of the transpose of X and y.  If we then plug this value into beta, we will find that each predicted value of y is a linear combination of each actual value of y.  As a result, locally weighted regression tends to be more accurate than random forest regression

(I will be adding further information about the hyperparameter tau and its effects on the model's variability, as well as the differences between parametric and non-parametric models next.)

## Concept of Random Forest Regression
The main idea of random forest regression is that the predictions are made by running a collection of several decision trees at once and then averaging the results of their decisions together.


### Locally Weighted Regression Applied To A Walmart Dataset

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

The Tricubic kernel proved to have the highest accuracy on the Walmart dataset.  I also performed loess regression with a tricubic kernel and a lower tau of 0.01, but the MSE was negatively impacted and slightly increased to 1.0155283663056112.  As such, I decided to try increasing the tau to 0.5.  This led to an improvement in accuracy with the lowest MSE at just 1.0155283662460473.


**MSE - Epanechnikov: 1.0155283662485854**

Similar to the Tricubic kernel with a smaller tau of 0.01, the MSE slightly increased to 1.0155283662961123, and decreased to just 1.01552836624602 with a higher tau of 0.5.


**MSE - Quartic: 1.0155283662492287**
Unlike the Tricubic and Epanechnikov kernels, utilizing the smaller tau of 0.01 decreased the MSE to 1.015528366238946.  As such, a smaller tau improved the accuracy of the model when using a Quartic kernel.  While increasing the tau to 0.5 did improve the MSE from the initial tau of 0.1, it did not improve it past that of the 0.01 tau.  With tau = 0.5, the MSE was 1.0155283662461692.


### Random Forest Regression Applied To A Walmart Dataset

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

**MSE (1,000 trees and depth of 3): 1.0026378217724825**

**MSE (100 trees and depth of 3): 1.002540902426993**

**MSE (1,000 trees and depth of 5): 1.0054284300015228**

**MSE (100 trees and depth of 5): 1.0046603734933888**

With the number of trees decreased to 100, the MSE improved to just 1.002540902426993.  I then experimented with extending the depth of each tree to 5 levels, but this greatly increased the MSE to 1.0054284300015228 with a forest of 1000 trees and 1.0046603734933888 with a forest of 100 trees.


### Locally Weighted Regression Applied To The Boston Housing Dataset

```
import numpy as np
import pandas as pd
from scipy import linalg
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from sklearn.model_selection import KFold, train_test_split as tts
from sklearn.metrics import mean_squared_error as mse

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
    
data2 = pd.read_csv("Boston Housing Prices.csv")

x2 = data2['rooms'].values
y2 = data2['cmedv'].values

xtrain2, xtest2, ytrain2, ytest2 = tts(x2, y2, test_size = 0.25, random_state = 123)

yhat2_test_tri = lowess_reg(xtrain2, ytrain2, xtest2, tricubic, 0.1)
print(mse(yhat2_test_tri, ytest2))

yhat2_test_epa = lowess_reg(xtrain2, ytrain2, xtest2, Epanechnikov, 0.1)
print(mse(yhat2_test_epa, ytest2))

yhat2_test_quart = lowess_reg(xtrain2, ytrain2, xtest2, Quartic, 0.1)
print(mse(yhat2_test_quart, ytest2))

# Select the kernel with the lowest MSE for comparison

# Plot a bar chart with each method's MSE 
```

**MSE - Tricubic: 44.16006090451543**

**MSE - Epanechnikov: 44.358249572483196**

**MSE - Quartic: 44.10804045673317**


### Random Forest Regression Applied To The Boston Housing Dataset

```
import numpy as np
import pandas as pd
from scipy import linalg
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, train_test_split as tts
from sklearn.metrics import mean_squared_error as mse

data2 = pd.read_csv("Boston Housing Prices.csv")

x2 = data2['rooms'].values
y2 = data2['cmedv'].values

xtrain2, xtest2, ytrain2, ytest2 = tts(x2, y2, test_size = 0.25, random_state = 123)

rf = RandomForestRegressor(n_estimators=1000,max_depth=3)

rf.fit(xtrain2.reshape(-1,1), ytrain2.reshape(-1,1))
yhat2_rf = rf.predict(xtest2.reshape(-1,1))

print(mse(yhat2_rf, ytest2))

# Plot a bar chart with each method's MSE 
```

**MSE (1,000 trees and depth of 3): 43.85795072968915**

**MSE (100 trees and depth of 3): 43.96618831213318**

**MSE (1,000 trees and depth of 5): 45.008688679120844**

**MSE (100 trees and depth of 5): 45.01038118242601**


### Locally Weighted Regression Applied To The Cars Dataset

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
    
data3 = pd.read_csv("cars.csv")

x3 = data3['WGT'].values
y3 = data3['MPG'].values

xtrain3, xtest3, ytrain3, ytest3 = tts(x3, y3, test_size = 0.25, random_state = 123)
scale = StandardScaler()
xtrain3_scaled = scale.fit_transform(xtrain3)
xtest3_scale = scale.transform(xtest3)

yhat3_test_tri = lowess_reg(xtrain3_scaled.ravel(), ytrain3, xtest3_scale.ravel(), tricubic, 0.1)
print(mse(yhat3_test_tri, ytest3))

yhat3_test_epa = lowess_reg(xtrain3_scaled.ravel(), ytrain3, xtest3_scale.ravel(), Epanechnikov, 0.1)
print(mse(yhat3_test_epa, ytest3))

yhat3_test_quart = lowess_reg(xtrain3_scaled.ravel(), ytrain3, xtest3_scale.ravel(), Quartic, 0.1)
print(mse(yhat3_test_quart, ytest3))

# Select the kernel with the lowest MSE for comparison

# Plot a bar chart with each method's MSE 
```

**MSE - Tricubic: 15.961885966790936**

**MSE - Epanechnikov: 16.04457105493449**

**MSE - Quartic: 15.993683601316075**


### Random Forest Regression Applied To The Cars Dataset

```
import numpy as np
import pandas as pd
from scipy import linalg
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, train_test_split as tts
from sklearn.metrics import mean_squared_error as mse
from sklearn.preprocessing import StandardScaler

data3 = pd.read_csv("cars.csv")

x3 = data3['WGT'].values
y3 = data3['MPG'].values

xtrain3, xtest3, ytrain3, ytest3 = tts(x3, y3, test_size = 0.25, random_state = 123)
scale = StandardScaler()
xtrain3_scaled = scale.fit_transform(xtrain3)
xtest3_scale = scale.transform(xtest3)

rf = RandomForestRegressor(n_estimators=1000,max_depth=3)

rf.fit(xtrain3_scaled.reshape(-1,1), ytrain3.reshape(-1,1))
yhat3_rf = rf.predict(xtest3_scale.reshape(-1,1))

print(mse(yhat3_rf, ytest3))

# Plot a bar chart with each method's MSE 
```

**MSE (1,000 trees and depth of 3): 15.764035906060245**

**MSE (100 trees and depth of 3): 15.759496652558436**

**MSE (1,000 trees and depth of 5): 16.372545103908113**

**MSE (100 trees and depth of 5): 16.236897436066638**


## Charts

### Walmart Data
```
import matplotlib.pyplot as plt

%matplotlib inline
%config InlineBackend.figure_format = 'retina'

fig, ax = plt.subplots(figsize=(10,8))
ax.set_xlim(3, 9)
ax.set_ylim(0, 51)
ax.scatter(x=data['Temperature'], y=data['Weekly_Sales'],s=25)
ax.set_xlabel('Temperature (degrees Celsius)',fontsize=16,color='darkgreen')
ax.set_ylabel('Weekly Sales (Thousands of Dollars)',fontsize=16,color='darkgreen')
ax.set_title('Walmart Sales Data',fontsize=18,color='purple')
ax.grid(b=True,which='major', color ='grey', linestyle='-', alpha=0.8)
ax.grid(b=True,which='minor', color ='grey', linestyle='--', alpha=0.2)
ax.minorticks_on()
plt.show()
```


### Boston Housing Prices Data

```
%matplotlib inline
%config InlineBackend.figure_format = 'retina'

fig, ax = plt.subplots(figsize=(10,8))
ax.set_xlim(3, 9)
ax.set_ylim(0, 51)
ax.scatter(x=data2['rooms'], y=data2['cmedv'],s=25)
ax.set_xlabel('Number of Rooms',fontsize=16,color='darkgreen')
ax.set_ylabel('House Price (Thousands of Dollars)',fontsize=16,color='darkgreen')
ax.set_title('Boston Housing Prices',fontsize=18,color='purple')
ax.grid(b=True,which='major', color ='grey', linestyle='-', alpha=0.8)
ax.grid(b=True,which='minor', color ='grey', linestyle='--', alpha=0.2)
ax.minorticks_on()
plt.show()
```


### Cars Data

```
%matplotlib inline
%config InlineBackend.figure_format = 'retina'

fig, ax = plt.subplots(figsize=(10,8))
ax.set_xlim(3, 9)
ax.set_ylim(0, 51)
ax.scatter(x=data3['WGT'], y=data3['MPG'],s=25)
ax.set_xlabel('Weight (lbs.)',fontsize=16,color='darkgreen')
ax.set_ylabel('Miles Per Gallon',fontsize=16,color='darkgreen')
ax.set_title('Car Mileage',fontsize=18,color='purple')
ax.grid(b=True,which='major', color ='grey', linestyle='-', alpha=0.8)
ax.grid(b=True,which='minor', color ='grey', linestyle='--', alpha=0.2)
ax.minorticks_on()
plt.show()
```
