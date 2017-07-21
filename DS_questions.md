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
