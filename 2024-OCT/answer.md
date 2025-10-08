# BACS3013 OCT 2024 Answers

[Link to the paper](https://eprints.tarc.edu.my/30305/1/BACS3013.pdf)

- [Question 1](#question-1)
- [Question 2](#question-2)
- [Question 3](#question-3)
- [Question 4](#question-4)

## Answers

### Question 1

a)

- **Missing Value Treatment**: Handling missing data by techniques such as mean/mode/median imputation, or using algorithms that can handle missing values.
- **Feature Scaling**: Normalizing or standardizing features to ensure they are on a similar scale, which can improve the performance of many machine learning algorithms.
- **Feature Engineering**: Encoding categorical variables, creating new features from existing ones, or transforming features to better represent the underlying patterns in the data.
- **Outlier Detection and Removal**: Identifying and handling outliers that may skew the results of the analysis.

b)

- **Data Collection**: Gathering e-commerce data from the platform's database.
- **Data Selection**: Relevant data such as sales figures, customer-item interactions, and customer demographics are selected.
- **Data Cleaning**: Cleaning the data by handling missing values, normalizing features, and encoding categorical variables, such as binning age groups.
- **Data Transformation**: Converting raw data into a suitable format for analysis, such as creating user-item interaction matrices.
- **Data Analytics**: Applying machine learning algorithms such as collaborative filtering or content-based filtering to generate personalized recommendations.
- **Interpretation and Visualization**: Generating reports and visualizations to present the recommendation results to stakeholders.

### Question 2

a)

i)

| Method  | Description                                                                                                                                                                                                         |
| ------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| IQR     | Calculate the first (Q1) and third quartiles (Q3) of the data, then determining the interquartile range (IQR = Q3 - Q1). Outliers are defined as data points that fall below Q1 - 1.5*IQR or above Q3 + 1.5*IQR.    |
| Z-Score | Calculate the mean and standard deviation of the data. The Z-score for each data point is computed as (X - mean) / standard deviation. Data points with an absolute Z-score greater than 3 are considered outliers. |

ii) Given the presence of outliers, identify 2 methods for feature engineering and briefly describe each.

| Method             | Description                                                                                                            |
| ------------------ | ---------------------------------------------------------------------------------------------------------------------- |
| Log Transformation | Apply a logarithmic transformation to reduce the impact of extreme values and make the data more normally distributed. |
| Binning            | Group continuous data into discrete bins or intervals, which can help to mitigate the                                  |

b)

- Set of 1-itemsets

$Support(\{MILK\}) = 3$

$Support(\{BREAD\}) = 5$

$Support(\{BUTTER\}) = 3$

$Support(\{JAM\}) = 1$

- JAM is removed

- Set of 2-itemsets

$Support(\{MILK, BREAD\}) = 3$

$Support(\{MILK, BUTTER\}) = 2$

$Support(\{BREAD, BUTTER\}) = 3$

- Set of 3-itemsets

$Support(\{MILK, BREAD, BUTTER\}) = 2$

- Frequent itemsets with minimum support of 2:
- 1-itemsets: {MILK}, {BREAD}, {BUTTER}
- 2-itemsets: {MILK, BREAD}, {MILK, BUTTER}, {BREAD, BUTTER}
- 3-itemsets: {MILK, BREAD, BUTTER}

### Question 3

a)

| Supervised Learning                    | Unsupervised Learning                                                        |
| -------------------------------------- | ---------------------------------------------------------------------------- |
| Involves labeled data                  | Involves unlabeled data                                                      |
| Used to predict values for unseen data | Used to cluster data points based on similarity and discover hidden patterns |

b)

| Key Decisions              | Description                                                                                                             |
| -------------------------- | ----------------------------------------------------------------------------------------------------------------------- |
| Splitting Decision         | Choosing the feature that best separates the classes at each node using metrics like Gini impurity or information gain. |
| Stopping Criteria          | Deciding when to stop splitting nodes, which can be based on maximum tree depth, minimum samples per leaf, or purity.   |
| Assignment of Class Labels | Assigning class labels to leaf nodes based on the majority class of the training samples that reach that node.          |

c)

i) (Not in current scope) SVM works by finding the optimal hyperplane that separates different classes in the feature space. It maximizes the margin between the closest points of each class, known as support vectors. A kernel function can be used to transform the data into a higher-dimensional space to handle non-linear separations.

ii)

| SVM                                                                  | KNN                                          |
| -------------------------------------------------------------------- | -------------------------------------------- |
| Best for high complexity data                                        | Best for low complexity data                 |
| Can classify data with non-linear boundaries through kernel function | Can classify data with non-linear boundaries |

iii) A kernel in SVM is a function that transforms the input data into a higher-dimensional space, allowing the algorithm to find a linear separation between classes that are not linearly separable in the original space. The kernel trick enables SVM to compute the inner products of the transformed data points without explicitly performing the transformation, which saves computational resources and allows for efficient handling of complex datasets.

### Question 4

a) T-Test, where null hypothesis states that there is no significant difference between the means of two groups.

b) Chi-Square Test, where null hypothesis states that there is no significant association between two categorical variables.

c) Recall-oriented model is preferred in defect detection because it prioritizes minimizing false negatives, ensuring that most actual defects are identified, which is crucial in quality control scenarios.

d) F1-Score is the appropriate performance metric for the model in this scenario because it ensures the model does not solely focus on maximising recall, but also has a good precision. This is important in defect detection to avoid overwhelming the inspection process with lots of false positives, which can waste resources and time.

e)

- The model is overfitting the training data, as it memorizes the training examples to perform well on them but fails to generalize to unseen data. This can be solved by using techniques such as K-Fold Cross Validation to ensure the model is robust and generalizes well to new data.
- The training data may be biased or not representative of the overall population, leading to poor performance on the test data which is less represented in the training set. This can be solved by using stratified sampling to ensure that the sample is representative of the population.

f) Deep learning is suitable for defect detection because it can automatically learn hierarchical feature representations from raw data, making it effective for complex tasks like image recognition. Deep learning models, such as convolutional neural networks (CNNs), can capture intricate patterns and variations in defect images, leading to higher accuracy and robustness compared to traditional machine learning methods that rely on manual feature extraction.
