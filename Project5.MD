# Comparison of Different Regularization and Variable Selection Techniques

Note: Please see my Python notebook file uploaded in the Adv-Machine-Learning repository entitled, "Project_5.ipynb", for my full code for the project.

## Part 1. Description of Technique
To complete this portion of the project, I first completed the data simulation required for Part 2 (please see my description of technique for Part 2. below for a description of how these data were simulated) and performed model fitting and prediction for the sklearn-compliant Square Root Lasso and SCAD penalty regression functions on this simulated data.  Initially, I tested my classes using an alpha of 0.1 and a lambda of 0.1 on one sample dataset of the full 100 datasets generated; in Part 3, I then used the GridSearchCV function to test out varying levels of alpha and lambda betwen 0.001 and 1 (as well as L1 ratios between 0.5 and 1), and then calculated the average number of correctly identified significant variables/non-zero coefficients, the average L2 distance between estimated and ground truth betas, and the average root mean square error (RMSE) for each regression model applied across all 100 simulated datasets.  Per the suggestion in class, I also timed how long it took for my computer to complete the regression fittings - it took approximately 6.05 seconds and  seconds for Square Root Lasso and SCAD penalty, respectively.  Since both Square Root Lasso and SCAD initially identified all variables as signficiant via non-zero coefficients for each feature, I also incorporated soft-thresholding to set any coefficients whose absolute value fell below 0.05 equal to zero.  As such, Square Root Lasso identified 106 non-zero beta coefficients/significant variables and SCAD identified 735 significant variables.  I then compared which of these identified variables overlapped with the ground truth significant variables - Square Root Lasso correctly identified 24 of these and SCAD correctly identified all 27.  Lastly, I computed the L2 distance between the ground truth and the beta coefficients estimated by these two models and determined that the distance was approximately 2.125 for Square Root Lasso and about 3.778 for the SCAD penalty regression model.

### Part 1. Code

```
# Create Sklearn compliant function for Square Root Lasso Regression
class SQRTLasso(BaseEstimator, RegressorMixin):
    def __init__(self, alpha=0.01):
        self.alpha = alpha
    def fit(self, x, y):
        alpha=self.alpha
        def f_obj(x,y,beta,alpha):
          n =len(x)
          beta = beta.flatten()
          beta = beta.reshape(-1,1)
          output = np.sqrt(1/n*np.sum((y-x.dot(beta))**2)) + alpha*np.sum(np.abs(beta))
          return output
        def f_grad(x,y,beta,alpha):
          n=x.shape[0]
          p=x.shape[1]
          beta = beta.flatten()
          beta = beta.reshape(-1,1)
          output = np.array((-1/np.sqrt(n))*np.transpose(x).dot(y-x.dot(beta))/np.sqrt(np.sum((y-x.dot(beta))**2))+alpha*np.sign(beta)).flatten()
          return output
        def objective(beta):
          return(f_obj(x,y,beta,alpha))
        def gradient(beta):
          return(f_grad(x,y,beta,alpha))
        
        beta0 = np.ones((x.shape[1],1))
        output = minimize(objective, beta0, method='L-BFGS-B', jac=gradient,options={'gtol': 1e-8, 'maxiter': 50000,'maxls': 25,'disp': True})
        beta = output.x
        self.coef_ = beta
        
    def predict(self, x):
        return x.dot(self.coef_)
        
# Test out an example model
model = SQRTLasso(alpha = 0.1)
%%time
model.fit(x,y.reshape(-1,1))
beta_hat = model.coef_
print(beta_hat)

# Implementing soft-thresholding by setting any absolute values less than 0.05 equal to 0
beta_hat[abs(beta_hat)<0.05] = 0
print(beta_hat)

# Detect the position of the estimated non-zero beta coefficients/significant variables
pos_sqrtlasso = np.where(beta_hat!=0)
print(pos_sqrtlasso)

print(np.array(pos_sqrtlasso).shape[1])
# My square root lasso regression model identified 106 important variables

# Check how many of these important variables were actually important
print(np.intersect1d(pos,pos_sqrtlasso))
print(np.intersect1d(pos,pos_sqrtlasso).shape)
# We were able to reconstruct 24 ground truths


# Compute the L2 distance
print(np.linalg.norm(model.coef_-beta,ord=2))


# Create Sklearn compliant function for SCAD Penalty Regression
class SCADRegression(BaseEstimator, RegressorMixin):
    def __init__(self, a=2,lam=1):
        self.a, self.lam = a, lam
  
    def fit(self, x, y):
        a = self.a
        lam   = self.lam

        def scad_penalty(beta_hat, lambda_val, a_val):
          is_linear = (np.abs(beta_hat) <= lambda_val)
          is_quadratic = np.logical_and(lambda_val < np.abs(beta_hat), np.abs(beta_hat) <= a_val * lambda_val)
          is_constant = (a_val * lambda_val) < np.abs(beta_hat)
    
          linear_part = lambda_val * np.abs(beta_hat) * is_linear
          quadratic_part = (2 * a_val * lambda_val * np.abs(beta_hat) - beta_hat**2 - lambda_val**2) / (2 * (a_val - 1)) * is_quadratic
          constant_part = (lambda_val**2 * (a_val + 1)) / 2 * is_constant
          return linear_part + quadratic_part + constant_part

        def scad_derivative(beta_hat, lambda_val, a_val):
          return lambda_val * ((beta_hat <= lambda_val) + (a_val * lambda_val - beta_hat)*((a_val * lambda_val - beta_hat) > 0) / ((a_val - 1) * lambda_val) * (beta_hat > lambda_val))

        
        def scad(beta):
          beta = beta.flatten()
          beta = beta.reshape(-1,1)
          n = len(y)
          return 1/n*np.sum((y-x.dot(beta))**2) + np.sum(scad_penalty(beta,lam,a))

         
        def dscad(beta):
          beta = beta.flatten()
          beta = beta.reshape(-1,1)
          n = len(y)
          output = -2/n*np.transpose(x).dot(y-x.dot(beta))+scad_derivative(beta,lam,a)
          return output.flatten()
        
        p = x.shape[1]
        beta0 = np.zeros(p)
        output = minimize(scad, beta0, method='L-BFGS-B', jac=dscad,options={'gtol': 1e-8, 'maxiter': 50000,'maxls': 50,'disp': False})
        beta = output.x
        self.coef_ = beta
        
    def predict(self, x):
        return x.dot(self.coef_)
   
model = SCADRegression(a = 0.01, lam=0.1)

%%time
model.fit(x,y.reshape(-1,1))

beta_hat = model.coef_
print(beta_hat)

# Incorporate soft-thresholding by setting any absolute values less than 0.05 equal to 0
beta_hat[abs(beta_hat)<0.05] = 0
print(beta_hat)

# Detect the position of the estimated non-zero beta coefficients/significant variables
pos_SCAD = np.where(beta_hat!=0)
print(pos_SCAD)

print(np.array(pos_SCAD).shape[1])
# My SCAD penalty regression model identified 855 important variables

# Check how many of these important variables are actually important
print(np.intersect1d(pos,pos_SCAD))
print(np.intersect1d(pos,pos_SCAD).shape)
# We were able to reconstruct all 27 ground truths

# Compute the L2 distance
print(np.linalg.norm(model.coef_-beta,ord=2))

```


## Part 2. Description of Technique
To simulate a dataset consisting of 100 dataframes with 200 rows and 1200 columns each, I first established the size of the dataframes and then created the "ground truth" beta coefficients as designated by the assignment to contain a sequence of seven 1's, 25 0's, five 0.25's, fifty 0's, 15 0.7's, and 1098 more 0's, thus yielding a sequence of 27 important variablies/non-zero coefficients.  I then incorporated the designated correlation structure using the Toeplitz function with rho = 0.8.  Lastly, I generated the x values by generating a random selection of numbers from a normal distribution with a mean of zero, a standard deviation of 1, and the aforementioned correlation structure.  The y values were generated by multiplying the x values by the ground truth beta coefficients and adding the "noise term" that included the designated standard deviation of 3.5 units.  Finally, I created a for loop to generate all 100 datasets with this similar style of randomly selecting x values from a normal distribution and adding noise to the product of these values and the previously determined beta coefficients to generate the respective y-values.

### Part 2. Code

```
# Simulate data with 200 observations/rows and 1200 features/columns
n = 200 
p = 1200

# Create ground truth coefficients
beta = 0
beta = np.concatenate((([1]*7), ([0]*25), ([0.25]*5), ([0]*50), ([0.7]*15), ([0]*1098)))

# Detect the position of the non-zero coefficients/significant variables
pos = np.where(beta!=0)
print(pos)

print(np.array(pos).shape)
# There are 27 important variables

# Incorporating our toeplitz correlation structure
vctr = []
for i in range(p):
  vctr.append(0.8**i)
  
mu = [0]*p
sigma = 3.5

# Generate the random samples
np.random.seed(123)

x = np.random.multivariate_normal(mu, toeplitz(vctr), size=n)
y = np.matmul(x,beta) + sigma*np.random.normal(0,1,n)

# Making 100 datasets
X = []
Y = []
for i in range(0,100):
  xnew = np.random.multivariate_normal(mu, toeplitz(vctr), size=n)
  X.append(xnew)
  Y.append(np.matmul(xnew,beta) + sigma*np.random.normal(0,1,n))

```

## Part 3. Description of Technique
For the final portion of the project, I used the GridSearchCV function to determine the best set of alpha and, when applicable, L1 ratio and/or lambda parameters to minimize the root mean-squared error (RMSE) of the respective regression models applied to one of the simulated datasets.  I then calculated the L2 distance between the ground truth beta coefficients and those estimated by the regression model being tested.  In the case of ridge regression, the best alpha was determined to be approximately 0.001 with an RMSE of approximately 5.9033 (or a mean-squared error of about 34.8486).  The L2 distance for ridge regression with this alpha was about 3.0028.  Lasso regression's best alpha was calculated to be about 0.1641 with an RMSE of approximately 3.936 (or a mean-squared error of 15.4909).  The L2 distance for lasso regression with this alpha was about 3.80797.  Elastic Net regression, which is essentially a convex combination of ridge and lasso regression, had the lowest RMSE at about 3.9004 (or a mean-squared error of 15.2135) using an alpha of approximately 0.0418 and an L1 ratio of 0.60204.  The L2 distance for elastic net regression with this comination of alpha and L1 ratio was approximately 3.1211.  The best alpha value for square root lasso regression was found to be 0.1437, yielding an RMSE of approximately 3.8686 and an MSE of about 14.9664.  This regression model's L2 distance was notably smaller than the preceding regression models' L2 distances at  approximately 1.2019.  Lastly, the best alpha for SCAD penalty regression was determined to be approximately 0.1641, similar to the best alpha for lasso regression, and the best lambda value for SCAD penalty was 0.001.  This pair of arguments yielded an RMSE of approximately 5.9201 (or a mean-squared error of 35.0478) and an L2 distance of 3.0188 units. 

Finally, I applied the best sets of arguments to each regression model, respectively, and fit these models to each of the 100 datasets to determine the average number of ground truth beta coefficients identified, the average L2 distance between the estimated and ground truth coefficients, and the average RMSE of the models across all 100 datasets.  For ridge regression, the average number of ground truth beta coefficients correctly identified was all 27, the average L2 distance was about 3.14995, and the average RMSE was quite small at approximately 1.0124e-5.  For lasso regression, the average number of ground truth betas correctly identified was about 20.88, the average L2 distance was 3.1569, and the average RMSE was approxiamtely 1.7096.  For elastic net regression, the average number of ground truth coefficients correctly identified was approximately 24.25, the average L2 distance between estimated and true betas was about 3.1936, and the average RMSE was approximately 0.3912.  For square root lasso regression, the average number of correctly identified ground truths was also all 27 (similar to ridge regression), the average L2 distance was calculated to be about 1.3437, and finally the average RMSE was about 3.1854.  Lastly, for SCAD penalty regression, the average number of ground truth beta coefficients correctly identified was again all 27, the average L2 distance was determined to be approximately 3.1633, and its average RMSE was approximately 0.00665897.  

In determining which regression method appeared to perform the best at predicting the ground truth beta coefficients and thus would be the most accurate model, I focused on the three that were able to recreate all 27 ground truths: ridge, square root lasso, and SCAD.  From there, I looked at which of these had the lowest average RMSE and determined that this would be the ridge regression model at just 0.0000101.

### Part 3. Code

```
# Ridge Regression
model = Ridge(alpha=0.1, fit_intercept=False)
model.fit(x,y)
beta_hat = model.coef_
print(beta_hat)

# Implement soft-thresholding by setting any absolute values less than 0.05 equal to 0
beta_hat[abs(beta_hat)<0.05] = 0
print(beta_hat)

# Detect the position of the estimated non-zero beta coefficients/significant variables
pos_ridge = np.where(beta_hat!=0)
print(pos_ridge)

print(np.array(pos_ridge).shape[1])
# My ridge regression model identified 627 important variables

# Check how many of these important variables were actually important
print(np.intersect1d(pos,pos_ridge))
print(np.intersect1d(pos,pos_ridge).shape)
# We were able to reconstruct all 27 ground truths


# Lasso Regression
model = Lasso(alpha=0.1, fit_intercept=False, max_iter=5000) # no intercept because the mean is 0
model.fit(x,y)
beta_hat = model.coef_
print(beta_hat)

# Detect the position of the estimated non-zero beta coefficients/significant variables
pos_lasso = np.where(beta_hat!=0)
print(pos_lasso)

print(np.array(pos_lasso).shape[1])
# My lasso regression model identified 144 important variables

# Check how many of these important variables were actually important
print(np.intersect1d(pos,pos_lasso))
print(np.intersect1d(pos,pos_lasso).shape)
# We were able to reconstruct 21 ground truths


# Elastic Net Regression
model = ElasticNet(alpha=0.1, fit_intercept=False)
model.fit(x,y)
beta_hat = model.coef_
print(beta_hat)

# Detect the position of the estimated non-zero beta coefficients/significant variables
pos_elasticnet = np.where(beta_hat!=0)
print(pos_elasticnet)

print(np.array(pos_elasticnet).shape[1])
# My elastic net regression model identified 238 important variables

# Detect the position of the estimated non-zero beta coefficients/significant variables
print(np.intersect1d(pos,pos_elasticnet))
print(np.intersect1d(pos,pos_elasticnet).shape)
# We  were able to reconstruct 25 ground truths


# RMSE, MSE, and L2 Distance Calculations
# Ridge Regression calculations
%%time
model = Ridge(fit_intercept=False,max_iter=1000)
params = [{'alpha':np.linspace(0.001,1,num=50)}]
gs = GridSearchCV(estimator=model,cv=10,scoring='neg_mean_squared_error',param_grid=params)
gs_results = gs.fit(x,y)
print(gs_results.best_params_)
print('The root mean square error is: ', np.sqrt(np.abs(gs_results.best_score_)))

%%time
model = Ridge(fit_intercept=False,max_iter=1000)
params = [{'alpha':np.linspace(0.001,1,num=50)}]
gs = GridSearchCV(estimator=model,cv=10,scoring='neg_mean_squared_error',param_grid=params)
gs_results = gs.fit(x,y)
print(gs_results.best_params_)
print('The mean square error is: ', np.abs(gs_results.best_score_))

# Testing out the results using the best parameters found by GridSearchCV
model = Ridge(alpha=0.001, fit_intercept=False, max_iter=1000)
model.fit(x,y)

# Compute the L2 distance
print(np.linalg.norm(model.coef_-beta,ord=2))

true_coeffs = []
L2Distance = []
RMSE = []
model = Ridge(alpha=0.001, fit_intercept=False, max_iter=1000)
for i in range(0,100):
  model.fit(X[i], Y[i])
  true_coeffs.append(np.intersect1d(pos,np.where(model.coef_!=0)).shape[0])
  L2Distance.append(np.linalg.norm(model.coef_-beta,ord=2))
  RMSE.append(np.sqrt(MSE(Y[i],model.predict(X[i]))))

print(np.mean(true_coeffs))
print(np.mean(L2Distance))
print(np.mean(RMSE))



# Lasso Regression calculations
%%time
model = Lasso(fit_intercept=False,max_iter=2500)
params = [{'alpha':np.linspace(0.001,1,num=50)}]
gs = GridSearchCV(estimator=model,cv=10,scoring='neg_mean_squared_error',param_grid=params)
gs_results = gs.fit(x,y)
print(gs_results.best_params_)
print('The root mean square error is: ', np.sqrt(np.abs(gs_results.best_score_)))

%%time
model = Lasso(fit_intercept=False,max_iter=2500)
params = [{'alpha':np.linspace(0.001,1,num=50)}]
gs = GridSearchCV(estimator=model,cv=10,scoring='neg_mean_squared_error',param_grid=params)
gs_results = gs.fit(x,y)
print(gs_results.best_params_)
print('The mean square error is: ', np.abs(gs_results.best_score_))

# Testing out the results using the best parameters found by GridSearchCV
model = Lasso(alpha=0.16410204081632654, fit_intercept=False, max_iter=2500)
model.fit(x,y)

# Compute the L2 distance
print(np.linalg.norm(model.coef_-beta,ord=2))

# Compute the average true coefficients, L2 distances, and RMSEs
true_coeffs = []
L2Distance = []
RMSE = []
model = Lasso(alpha=0.16410204081632654, fit_intercept=False, max_iter=2500)
for i in range(0,100):
  model.fit(X[i], Y[i])
  true_coeffs.append(np.intersect1d(pos,np.where(model.coef_!=0)).shape[0])
  L2Distance.append(np.linalg.norm(model.coef_-beta,ord=2))
  RMSE.append(np.sqrt(MSE(Y[i],model.predict(X[i]))))
  
print(np.mean(true_coeffs))
print(np.mean(L2Distance))
print(np.mean(RMSE))


# Elastic Net Regression calculations
%%time
model = ElasticNet(fit_intercept=False, max_iter=500000)
params = [{'alpha':np.linspace(0.001,1,num=50),'l1_ratio':np.linspace(0.5,1,num=50)}]
gs = GridSearchCV(estimator=model,cv=10,scoring='neg_mean_squared_error',param_grid=params)
gs_results = gs.fit(x,y)
print(gs_results.best_params_)
print('The root mean square error is: ', np.sqrt(np.abs(gs_results.best_score_)))

%%time
model = ElasticNet(fit_intercept=False, max_iter=500000)
params = [{'alpha':np.linspace(0.001,1,num=50),'l1_ratio':np.linspace(0.5,1,num=50)}]
gs = GridSearchCV(estimator=model,cv=10,scoring='neg_mean_squared_error',param_grid=params)
gs_results = gs.fit(x,y)
print(gs_results.best_params_)
print('The mean square error is: ', np.abs(gs_results.best_score_))

# Testing out the results using the best parameters found by GridSearchCV
model = ElasticNet(alpha=0.041775510204081635, l1_ratio=0.6020408163265306, fit_intercept=False, max_iter=500000)
model.fit(x,y)

# Compute the L2 distance
np.linalg.norm(model.coef_-beta,ord=2)

true_coeffs = []
L2Distance = []
RMSE = []
model = ElasticNet(alpha=0.041775510204081635, l1_ratio=0.6020408163265306, fit_intercept=False, max_iter=500000)
for i in range(0,100):
  model.fit(X[i], Y[i])
  true_coeffs.append(np.intersect1d(pos,np.where(model.coef_!=0)).shape[0])
  L2Distance.append(np.linalg.norm(model.coef_-beta,ord=2))
  RMSE.append(np.sqrt(MSE(Y[i],model.predict(X[i]))))
  
print(np.mean(true_coeffs))
print(np.mean(L2Distance))
print(np.mean(RMSE))


# SCAD Penalty Regression calculations
%%time
model = SCADRegression()
params = [{'a':np.linspace(0.001,1,num=50), 'lam':np.linspace(0.001,3,num=100)}]
gs = GridSearchCV(estimator=model,cv=10,scoring='neg_mean_squared_error',param_grid=params)
gs_results = gs.fit(x,y.reshape(-1,1))
print(gs_results.best_params_)
print('The root mean square error is: ', np.sqrt(np.abs(gs_results.best_score_)))

%%time
model = SCADRegression()
params = [{'a':np.linspace(0.001,1,num=50), 'lam':np.linspace(0.001,3,num=100)}]
gs = GridSearchCV(estimator=model,cv=10,scoring='neg_mean_squared_error',param_grid=params)
gs_results = gs.fit(x,y.reshape(-1,1))
print(gs_results.best_params_)
print('The mean square error is: ', np.abs(gs_results.best_score_))

# Testing out the results using the best parameters found by GridSearchCV
model = SCADRegression(a=0.16410204081632654, lam=0.001)
model.fit(x,y.reshape(-1,1))

# Compute the L2 distance
np.linalg.norm(model.coef_-beta,ord=2)

true_coeffs = []
L2Distance = []
RMSE = []
model = SCADRegression(a=0.16410204081632654, lam=0.001)
for i in range(0,100):
  model.fit(X[i], Y[i].reshape(-1,1))
  true_coeffs.append(np.intersect1d(pos,np.where(model.coef_!=0)).shape[0])
  L2Distance.append(np.linalg.norm(model.coef_-beta,ord=2))
  RMSE.append(np.sqrt(MSE(Y[i],model.predict(X[i]))))

print(np.mean(true_coeffs))
print(np.mean(L2Distance))
print(np.mean(RMSE))


# Square Root Lasso Regression calculations
%%time
model = SQRTLasso()
params = [{'alpha':np.linspace(0.001,1,num=50)}]
gs = GridSearchCV(estimator=model,cv=10,scoring='neg_mean_squared_error',param_grid=params)
gs_results = gs.fit(x,y)
print(gs_results.best_params_)
print('The root mean square error is: ', np.sqrt(np.abs(gs_results.best_score_)))

%%time
model = SQRTLasso()
params = [{'alpha':np.linspace(0.001,1,num=50)}]
gs = GridSearchCV(estimator=model,cv=10,scoring='neg_mean_squared_error',param_grid=params)
gs_results = gs.fit(x,y)
print(gs_results.best_params_)
print('The mean square error is: ', np.abs(gs_results.best_score_))

# Testing out the results using the best parameters found by GridSearchCV


# Compute the L2 distance
np.linalg.norm(model.coef_-beta,ord=2)

true_coeffs = []
L2Distance = []
RMSE = []
model = SQRTLasso(alpha=0.1437142857142857)
for i in range(0,100):
  model.fit(X[i], Y[i].reshape(-1,1))
  true_coeffs.append(np.intersect1d(pos,np.where(model.coef_!=0)).shape[0])
  L2Distance.append(np.linalg.norm(model.coef_-beta,ord=2))
  RMSE.append(np.sqrt(MSE(Y[i],model.predict(X[i]))))

print(np.mean(true_coeffs))
print(np.mean(L2Distance))
print(np.mean(RMSE))
```
