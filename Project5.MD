# Comparison of Different Regularization and Variable Selection Techniques

## Part 1. Description of Technique

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

```

## Part 3. Description of Technique

### Part 3. Code
Note: I will be adding the SCAD penalty regression model and Square Root Lasso regression model results next

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

# Compute the L2 distance
print(np.linalg.norm(model.coef_-beta,ord=2))



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

# Compute the L2 distance
print(np.linalg.norm(model.coef_-beta,ord=2))



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

# Compute the L2 distance
np.linalg.norm(model.coef_-beta,ord=2)


# SCAD Penalty Regression calculations

```
