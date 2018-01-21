### 1. What are the advantages and disadvantages of KNN?

Advantages: K-Nearest Neighbors have a nice intuitive explanation, and then tend to work very well for problems where comparables are inherently indicative. For example, you could build a kNN housing price model by modeling on other houses in the area with similar number of bedrooms, floor space, etc.

Disadvantages: They are memory-intensive.They also do not have built-in feature selection or regularization, so they do not handle high dimensionality well.

### 2. What’s the difference between a generative and discriminative model?

A generative model will learn categories of data while a discriminative model will simply learn the distinction between different categories of data. Discriminative models will generally outperform generative models on classification tasks.

Reference:
https://stackoverflow.com/questions/879432/what-is-the-difference-between-a-generative-and-discriminative-algorithm

### 3. You are working on a time series data set. You manager has asked you to build a high accuracy model. You start with the decision tree algorithm, since you know it works fairly well on all kinds of data. Later, you tried a time series regression model and got higher accuracy than decision tree model. 
Can this happen? Why?

Time series data is known to possess *linearity*. On the other hand, a decision tree algorithm is known to work best to detect *non – linear* interactions. The reason why decision tree failed to provide robust predictions because it couldn’t map the linear relationship as good as a regression model did. Therefore, we learned that, a linear regression model can provide robust prediction given the data set satisfies its linearity assumptions.

### 4. While working on a data set, how do you select important variables? Explain your methods.

- Remove the correlated variables prior to selecting important variables
- Use linear regression and select variables based on p values
- Use Forward Selection, Backward Selection, Stepwise Selection
- Use Random Forest, Xgboost and plot variable importance chart
- Measure information gain for the available set of features and select top n features accordingly.

### 5. What’re the differences between logistic regression and linear regression?

***- Linear regression output as probabilities***

It's tempting to use the linear regression output as probabilities but it's a mistake because the output can be negative, and greater than 1 whereas probability cannot. As regression might actually produce probabilities that could be less than 0, or even bigger than 1, logistic regression was introduced.

***- Outcome***

In linear regression, the outcome (dependent variable) is continuous. It can have any one of an infinite number of possible values. In logistic regression, the outcome (dependent variable) has only a limited number of possible values.

***- The dependent variable***

Logistic Regression is used when response variable is categorical in nature. Linear Regression is used when your response variable is continuous.

***- Equation***

Linear Regression gives an equation which is of the form Y = mX + C, means equation with degree 1.However, Logistic Regression gives an equation which is of the form Y = e^X/1 + e^-X (there is a logistic function)

***- Coefficient interpretation***

In linear regression, the coefficient interpretation of independent variables are quite straight forward. However in logistic regression, depends on the family (binomial, poisson, etc.) and link (log, logit, inverse-log, etc.) you use, the interpretation is different.

***- Error Minimization Technique***

Linear Regression uses Ordinary Least Squares method to minimize the errors and arrive at a best possible fit. While logistic regression uses maximum likelihood method to arrive at the solution.

### 6. Is dimensionality reduction the same as feature selection? Why?
While feature selection and dimensionality reduction both have the effect of feeding a lower number of features, feature selection techniques (like regularization) are an exercise in finding (and selecting) those features that are significant when it comes to signaling a given target variable.  Dimensionality reduction on the other hand, blindly reduces the count of features (dimensions) that are being used, without paying attention to their effect in predicting a given target variable.  

It is entirely possible that in some circumstances one might use SVD to identify and drop information captured by certain lower variance features that could have been very significant when it came to classifying a certain target variable. 

### 7. Explain bagging.
Bagging, or Bootstrap Aggregating, is an ensemble method in which the dataset is first divided into multiple subsets through resampling. Then, each subset is used to train a model, and the final predictions are made through voting or averaging the component models. Bagging is performed in parallel.

### 8. How can you choose a classifier based on training set size?
If training set is small, high bias / low variance models (e.g. Naive Bayes) tend to perform better because they are less likely to be overfit.

If training set is large, low bias / high variance models (e.g. Logistic Regression) tend to perform better because they can reflect more complex relationships.

### 9. How does the Linear Regression algorithm figure out what are the best coefficient values? (This was a question asked in C3 Energy’s Data Scientist interview)
At the highest level, the coefficients are a function of ***minimizing the sum of square of the residuals***. Next, write down these equations while paying careful attention to what is a residual. To go further, consider the following:

1. Write the minimization goal (ideally in linear algebraic (matrix) notation) of minimizing the sum of squares of the residuals given a linear regression model.

2. Solve the minimization equation by illustrating that the sum of square of the residuals is a convex function, which can be differentiated and the coefficients can be derived by setting the differentiation to 0 and solving that equation.

3. Describe that the complexity of solving the linear algebra based solution in #2 is of **polynomial time** and a more common solution is by observing that the equation is **convex** and hence **numerical algorithms** such as gradient descent may be much more efficient.

### 10. (EZ) What does P-value signify about the statistical data?
P-values is used to determine the significance of results after a hypothesis test in statistics. 

定义：the probability of finding the observed, or more extreme, results when the null hypothesis of a study question is true. 适用情形：when you want to know whether a model result that appears interesting and useful is **within the range of normal chance variability**.

- P- Value > 0.05 denotes weak evidence against the null hypothesis which means the null hypothesis cannot be rejected.
- P-value <= 0.05 denotes strong evidence against the null hypothesis which means the null hypothesis can be rejected.
- P-value=0.05 is the marginal value indicating it is possible to go either way.

### 11. Are expected value and mean value different?期望和均值有区别吗？
They are not different but the terms are used in **different contexts**. Mean is generally referred when talking **distribution or sample population** while expected value is generally referred in a **random variable context**.

- For sampling data
  - Mean is the only value that comes from the sampling data
  - Expected value is the mean of all the means, i.e., the value that is built from multiple samples. Expected value is the population mean.

- For distributions: Mean value and expected value are samoe irrespective of the distribution, under the condition that the distribution is in the same population.

### 12. (EZ) How do you understand the term Normal Distribution?如何理解正态分布？
Data is usually distributed in different ways with a bias to the left or to the right or it can all be jumbled up. However, there are chances that the data is distributed **around a central value** without any bias to the left or right, and reaches normal distribution in **the form of a bell shaped curve**. The random variables are distributed in the form of **an symmetrical bell shaped curve**.

### 13. (EZ & Important) What is the Central Limit Theorem? Explain it. Why is it important? 解释中心极限定理
The CLT states that the arithmetic mean of a sufficiently large number of iterates of independent random variables will be approximately normally distributed regardless of the underlying distribution. i.e., the sampling distribution of the sample mean is normally distributed.

用处：

- Used in hypothesis testing
- Used for confidence intervals
- Random variables must be iid distributed
- Finite variance

### 13. (ML) What is the advantages of ReLU over sigmoid function?
Sigmoid function has the problem of vanish（突然变为零） gradient because the gradient of sigmoid becomes increasingly small as the absolute value of x increases. But ReLU can **reduce the likelihood** of the gradient **to vanish** and the constant gradient of ReLU when x>0 will **result in faster learning**.

Another advantage of ReLU is **sparsity**, which arises when x<=0. The more such units that exist in a layer, the more sparse the result representation. Sigmoids on the other hand are always likely to generate some non-zero value resulting in dense representations. Sparse representations seem to be more beneficial than dense representations.


### 14. How will you define the number of clusters in a clustering algorithm?
The objective of clustering is to group similar entities in a way that entities within a group are similar to each other, but the groups are different from each other.

Within Sum of Squares is generally used to explain the homogeneity within a cluster. If you plot WSS versus the number of clusters, you will get an Elbow curve. We can then choose k after which we don't see any decrements in WSS.

Some data scientists also use hierarchical clustering first to create dendrograms and identify the distinct groups from there.

### 15. How do you treat missing values?
First we need to identify patterns. If any patterns are identified with some variables, it could lead to interesting and meaningful business insights.

- **Remove rows with missing values** - this works if 
  - the values are missing randomly
  - if you don't lose too much of the dataset
 
- **Build another predictive model to predict the missing values**
- **Use a model that can incorporate missing data** - Like a random forest, or any tree-based method


### 16. How do you understand by statistical power of sensitivity and how do you calculate it?
**Sensitivity (Recall)**: TP/ (TP + FN)

Sensitivity is used to validate the accuracy of a classifier, which is the "Predicted TRUE events"/ Total events.

**Specificity**: TN / (TN + FP)

**Precision**: TP/ (TP + FP)

### 17. What is a point estimate?
In statistics, point estimate involves the use of sample data to calculate a single statistic which is to serve as a "best guess" or "best estimate" of an unknown parameter. There are a variety of point estimators, each with different properties.
- Minimum-variance mean-unbiased estimator (MVUE)
- Best linear unbiased estimator (BLUE)
- Minimum mean square error (MMSE)
- Median-unbiased estimator
- Maximum likelihood (ML)
- Method of moments, generalized method of moments

### 18. (DL) Describe what is the Artificial Neural Network?
An ANN is a computational model. It is based on the structure and functions of biological neural networks. It works like the way human brain processes information. It includes a large number of connected processing units that work together to process information, which can generate meaningful results.

### 19. (Dimensionality Reduction) What is Random Projection?
- It is an unsupervised machine learning method for reducing the dimensions
- It creates a minimum reduced dimension k which can make the new pairwise data distance preserved within an accepted error compared to the original pairwise data distance.

### 20. (Dimensionality Reduction) Basic Steps to do PCA.
- Standardize the data
- Obtain the eigenvectors and eigenvalues from the covariance matrix or correlation matrix, or perform Singular Vector Decomposition.
- Sort eigenvalues in descending order and choose the k eigenvalues where k is the number of dimensions of the new feature subspace (k << d)
- Construct the projection matrix W from the selected k eigenvectors.
- Transform the original dataset X via W to obtain a k-dimensional feature subspace Y

### 21. (Dimensionality Reduction) Random Projection vs. PCA
相同: Both are defining a projection from a high-dimensional space into a low-dimensional space -- picking a small set of basis vectors in the high-dimensional space that can be used as a basis (to "explain") the data in the low-dimensional space. 

区别：The major difference is that PCA is trying hard to pick the "best" basis vectors by looking for directions in which the original data varies most. Random projection is picking the directions randomly!
- Random Projection runs much faster than PCA with very high dimensions
- In general PCA works well on relatively low dimensional data

### 22. (ML算法) Give some classification situations where you will use an SVM over a RF and vice versa.
- When the data is outlier free and clean then go for SVM. If your data might contain outliers then RF would be the best choice.
- Generally, SVM consumes more computational power than Random Forest, so if you're constrained with memory go for Random Forest.
- Random Forest gives you a very good idea of **variable importance** in your data. If you want to have variable importance, then go for RF.
- RF is preferred for multiclass problems.
- SVM is preferred in high-dimensional problem set - like text classification.

### 23. 基本定义Bagging
Bagging, or Bootstrap Aggregating, is an ensemble method in which the dataset is first divided into multiple subsets through resampling. Then, each subset is used to train a model, and the final predictions are made through voting or averaging the component models. Bagging is performed in parallel.

### 24. How can you choose a classifier based on training set size?
- If training set is **small**, high bias/ low variance models (e.g. Naive Bayes) tend to perform better because they are less likely to overfit.
- If training set is **large**, low bias/ high variance models （e.g. Logistic Regression) tend to perform better because they can reflect more complex relationships.
