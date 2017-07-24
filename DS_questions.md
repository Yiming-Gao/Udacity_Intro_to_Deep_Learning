### 1. What are the advantages and disadvantages of neural networks?

Advantages: K-Nearest Neighbors have a nice intuitive explanation, and then tend to work very well for problems where comparables are inherently indicative. For example, you could build a kNN housing price model by modeling on other houses in the area with similar number of bedrooms, floor space, etc.

Disadvantages: They are memory-intensive.They also do not have built-in feature selection or regularization, so they do not handle high dimensionality well.

### 2. What’s the difference between a generative and discriminative model?

A generative model will learn categories of data while a discriminative model will simply learn the distinction between different categories of data. Discriminative models will generally outperform generative models on classification tasks.

Reference:
https://stackoverflow.com/questions/879432/what-is-the-difference-between-a-generative-and-discriminative-algorithm

### 3. You are working on a time series data set. You manager has asked you to build a high accuracy model. You start with the decision tree algorithm, since you know it works fairly well on all kinds of data. Later, you tried a time series regression model and got higher accuracy than decision tree model. 
Can this happen? Why?

Time series data is known to posses *linearity*. On the other hand, a decision tree algorithm is known to work best to detect *non – linear* interactions. The reason why decision tree failed to provide robust predictions because it couldn’t map the linear relationship as good as a regression model did. Therefore, we learned that, a linear regression model can provide robust prediction given the data set satisfies its linearity assumptions.

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

