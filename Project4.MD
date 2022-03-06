### Multiple Boosting Algorithm and Application


### Description and Application of LightGBM Algorithm

The LightGBM boosting algorithm was developed by Microsoft as a less-intensive alternative to other boosting algorithms, like extreme gradient boosting (XGB).  According to the documentation for the algorithm provided [here](https://lightgbm.readthedocs.io/en/latest/), it is faster and more efficient than other tree-based methods while still maintaining high accuracy.  The main feature that distinguishes LightGBM from XGB and other tree-based boosting algorithms is that it implements leaf-wise splitting, as opposed to level-wise splitting.  As such, it searches for the leaf with the greatest change in loss (i.e., the greatest decline in loss) and splits the data at that particular node, as opposed to splitting at a full level of the tree (Reference: [https://www.geeksforgeeks.org/lightgbm-light-gradient-boosting-machine/](https://www.geeksforgeeks.org/lightgbm-light-gradient-boosting-machine/)).  Because this generates a less complex decision tree, the LightGBM model is faster and requires less memory usage.  Please see the two diagrams below for an illustrated comparison of leaf-splitting and level-splitting, respectively.

# Leaf Splitting
![](https://i.stack.imgur.com/YOE9y.png)


# Level Splitting
![](https://i.stack.imgur.com/e1FWe.png)


The LightGBM documentation also states that this algorithm is quick and efficient because it is histogram-based, which means it clusters continuous features into bins.  By contrast, XGB uses pre-sort algorithms.  Furthermore, if there are categorical features in the data, LightGBM splits such features into two subsets per category based on "accumulated values".  LightGBM also mplements Gradient-based One-sided Sampling (GOSS) and exclusive feature bundling (EFB) to more heavily weight the data observations that contribute the most to predicting the dependent variable, y, and grouping "exclusive" features into one feature, respectively.  Something interesting to note from the reference previously mentioned is that one should be cautious when using LightGBM on smaller datasets because it has a tendency to be overfit.  Among its many hyperparameters, the LightGBM regressor algorithm includes a hyperparameter for its learning rate, which tells the model at what rate it should adjust its weight predictions given the current level of error - higher learning rates tend to execute more quickly, but may be less accurate than smaller learning rates (Reference: [https://machinelearningmastery.com/learning-rate-for-deep-learning-neural-networks/](https://machinelearningmastery.com/learning-rate-for-deep-learning-neural-networks/).  Due to the learning rate's seemingly significant impact on the accuracy of the model, I chose to experiment with a variety of learning rate settings in my analysis/application of the LightGBM algorithm below.  In addition, I tuned its "max_depth" hyperparameter to help determine an optimal level at which to discontinue splitting the tree further.



## Code
