# Comparison of Different Regularization and Variable Selection Techniques

## Part 1. Description of Technique

### Part 1. Code


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
Note: I will be adding the L2 distance to the ideal solution next

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


# RMSE and L2 Distance Calculations
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

```
