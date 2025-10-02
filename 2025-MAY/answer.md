# BACS3013 MAY 2025 Answers

[Link to the paper](https://eprints.tarc.edu.my/33102/1/BACS3013.pdf)

- [Question 1](#question-1)
- [Question 2](#question-2)
- [Question 3](#question-3)
- [Question 4](#question-4)

## Answers

### Question 1

a)

i) **Understanding Data**: The schema and relationship of the collected data are studied through descriptive statistics and data visualization techniques. Central tendencies and dispersions can be performed on numerical data such as age and income.

ii) **Data Selection**: The relevant features for analyse customer purchasing behaviour are selected. Unnecessary or redundant features such as customer ID and name are removed, while important features like age, income, and customer-item interactions are retained.

iii) **Data Cleaning**: Missing values in features like income are handled by imputing with the median value. Outliers in features like age are detected using the Z-score method and can be treated by capping depending on the context.

iv) **Data Transformation**: Numerical features like age and income are normalized using Z-Score normalization to ensure they are on a similar scale. Categorical features like item categories are encoded using one-hot encoding to convert them into a numerical format suitable for analysis.

v) **Data Analytics**: K-Means clustering is applied to group customers based on their purchasing behaviour. The optimal number of clusters is determined using the Elbow Method.

vi) **Interpretation and** Evaluation: The clustering results are interpreted by business experts to uncover insights about customer segments. The clustering quality is evaluated using the Silhouette Score to assess how well the clusters are formed.

b)

- **Effective Communication**: The data scientist must be able to communicate complex technical findings in a clear and concise manner to non-technical stakeholders, as they lack technical background. This includes creating visualizations and reports that are easy to understand.
- **Team Collaboration**: The data scientist must work effectively within a multidisciplinary team, collaborating with data engineers, business analysts, and domain experts. This involves sharing knowledge, listening to others' perspectives, and contributing to a positive team dynamic because data science projects often require input from various fields.

c)

i) Stratified Sampling is a sampling technique where the population is divided into distinct subgroups or strata based on a specific characteristic which is relevant to the study. A random sample is then taken from each stratum in proportion to its size in the overall population to ensure that each subgroup is adequately represented in the sample.

ii) Real-world scenario: Statified sampling can be used in a healthcare study to ensure the sample is representative across different age groups. This is important because different age groups may have varying health conditions and responses to treatments. Having accurate representation of all age groups ensures that the study's findings are generalizable to the entire population.

d)

- Human Error: Mistakes made by individuals during data entry
- Undisclosed Information: Missing data due to non-response or refusal to provide information
- Nonapplicable information: Attributes that do not apply to certain records

### Question 2

a)

i) Value: 24, Q1 is the 25th percentile of the values in the dataset.

ii) Value: 32, Median is the 50th percentile of the values in the dataset.

iii) Value: 40, Q3 is the 75th percentile of the values in the dataset.

iv) Value: 10, Minimum is the smallest value in the dataset.

v) Value: 50, Maximum is the largest value in the dataset.

vi) Value: 40 - 24 = 16, IQR is the difference between the third quartile (Q3) and the first quartile (Q1).

b) 150, 155, 155, 160, 165, 170, 180, 190

i) Equal Interval Binning segregates the data range into bins of equal width, which is the difference of the boundaries of the bin. Equal Frequency Binning divides the data into bins that contain an equal number of data points.

ii)

- 190 - 150 = 40 (Range)
- 40 / 4 = 10 (Bin Width)

| Bin | Values        |
| --- | ------------- |
| 1   | 150, 155, 155 |
| 2   | 160, 165      |
| 3   | 170           |
| 4   | 180, 190      |

iii)

- 8 / 4 = 2 (Number of values per bin)

| Bin | Values   |
| --- | -------- |
| 1   | 150, 155 |
| 2   | 155, 160 |
| 3   | 165, 170 |
| 4   | 180, 190 |

### Question 3

a)

i) Precision, Recall

ii)

- Precision = True Positives / (True Positives + False Positives)
  - This metric is suitable in scenarios where the cost of false positives is high, such as in email spam detection, where misclassifying a legitimate email as spam can lead to important communications being missed.
- Recall = True Positives / (True Positives + False Negatives)
  - This metric is suitable in scenarios where the cost of false negatives is high, such as in medical diagnostics, where failing to identify a disease can have serious health consequences.

iii)

- It ensures that the model are reliable and accurate enough for deployment in real-world applications, which maximizes the benefits and minimizes the risks associated with its use.
- It helps in identifying areas for improvement in the model, guiding further development and refinement to enhance its performance over time. Issues such as overfitting, underfitting, or bias can be detected and addressed through performance evaluation.

b)

i) Association rule, this is because the order of buying products is not relevant, only the co-occurrence of the items across transactions matters. The insight such as most frequently bought together items can be derived and used for bundled recommendations.

ii) Sequence rule, this is because the order of buying products matters, as it captures the temporal aspect of customer purchasing behaviour. The insight can be used to predict future purchases based on past buying patterns.

c)

i) Linear regression is more suitable it is because the model has a higher interpretability than KNN, which provides clear insights into the significance and impact of each feature on the next month's sales revenue. As KNN is a lazy learning algorithm, it does not provide explicit coefficients and equations for interpretation.

ii) Fraud detection in financial transactions can use KNN as it has more complex nonlinear relationships between features and the target variable. KNN can capture these complex patterns by considering the proximity of data points in the feature space, making it effective for identifying fraudulent transactions based on similarities to known fraud cases.

### Question 4

a)

- **Accuracy**: This dimension ensures that the patient information stored in the EMR system is correct and free from errors. Accurate data is crucial for making informed clinical decisions, as any inaccuracies could lead to misdiagnosis or inappropriate treatment plans.
- **Completeness**: This dimension ensures that all necessary patient information is recorded in the EMR system. Incomplete data can hinder healthcare providers' ability to make comprehensive assessments of a patient's health status, potentially leading to suboptimal care.

b)

i)

- SVM is **effective in high-dimensional spaces**, where the feature space can be complex due to the various extracted features from the images. SVM can handle a large number of features and find the optimal hyperplane that separates the classes effectively.
- SVM is **robust to overfitting**, especially in cases where the number of features exceeds the number of samples. This is particularly important in this scenario, as the dataset consists of a limited number of product images, and SVM can generalize well even with small datasets.

ii) Features of image data are often complex and not linearly separable in the original feature space. Kernel functions allow SVM to project the data into a higher-dimensional space where a linear separation is possible, enabling the model to capture intricate patterns and relationship in the image data.

c)

i)

- Bar Chart: A bar chart is suitable for representing monthly sales revenue per region. It allows for easy comparison of sales figures across different regions and months, making it straightforward to identify trends and patterns in sales performance over time.
- Histogram: A histogram is appropriate for displaying the central tendency and dispersion of customer age. It allows for the visualization of the distribution of ages, helping to identify common age groups and any potential outliers in the customer base.

ii) 2 common mistakes in data visualization, and how they can mislead analysis:

- Choosing less appropriate chart types such as using a pie chart for features with many categories can make it difficult to compare the sizes of the slices accurately. This can mislead the analysis by obscuring important differences between categories.
- Too much information in a single chart, such as overcrowding a line chart with too many lines representing different categories. This can lead to confusion and make it hard to discern individual trends, potentially leading to incorrect conclusions about the data.

