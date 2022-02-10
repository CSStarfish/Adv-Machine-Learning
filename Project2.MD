# Comparison of Locally Weighted and Random Forest Regression

## Concept of Locally Weighted Regression
The main idea of locally weighted regression is that the predictions are a linear combination of the observed/actual values from the dataset (i.e., the predicted y is a linear combination of the actual values of y).  It applies weights to each value proportional to the distance between the local datapoints, with weights closer to one the closer they are together and closer to zero the further away they are, thus accounting for variability in the dataset.  Linear regression stems from the following equation:
![](linearregression.jpg).  If we multiply the equation by a matrix of weights consisting of the transpose of X (i.e., the independent observations of the dataset) and solve for the beta coefficient, we will find that the expected value of beta is equal to the inverse of the product of X and the transpose of X, multiplied by the product of the transpose of X and y.  If we then plug this value into beta, we will find that each predicted value of y is a linear combination of each actual value of y.  As a result, locally weighted regression tends to be more accurate than random forest regression

(I will be adding further information about the hyperparameter tau and its effects on the model's variability, as well as the differences between parametric and non-parametric models next.)

## Concept of Random Forest Regression
The main idea of random forest regression is that the predictions are made by running a collection of several decision trees at once and then averaging the results of their decisions together.


## Code
Note: I uploaded an ipynb file and I will be including the code directly into the markdown soon.
