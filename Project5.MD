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
