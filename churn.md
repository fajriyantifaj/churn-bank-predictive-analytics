# Data Analytics Churn Costumer In Banking Sector
 
## Project Domain

Customer churning [1, 2] is the estimate or analysis of degree of customers who turn to shift to an alternative. It is the most common problem witnessed in any industry. Banking is one such industry that focuses a lot on customer’s behavior by tracking their activities

As we know, customers are one of the most valuable assets in the business domain, especially in the banking sector. When customers use the product, they contribute to the product's growth rate. However, many would unsubscribe from the product if it no longer had any advantages.

Many competitive companies have noticed that a critical approach for survival within the industry is to retain existing customers. This leads to the importance of churn management in organizations such as banks.

  

## Business Understanding

For the banking sector, customers churn predictions are a severe issue and have an immense impact on the proﬁt line of bankers. Thus, customer retention schemes can target high-risk customers who wish to discontinue their custom and switch to another competitor. To minimize the cost of bank sectors' customer retention marketing scheme, an accurate and prior identiﬁcation of these customers is hypercritical.  The Problem is based on the domain of the banking sector, where the bank wants to predict a customer's Churn depending upon the customer's data. By Churn, it is meant that the bank wants to predict if a customer will be leaving the bank or not. 

## Goals

In order to create a comprehensive classification model that can reliably predict whether or not a bank's customers will leave, this project aims to succinctly describe how a banking dataset was used to explore and cluster data using visualization, statistical analysis, and principal component analysis.
 
This project uses machine learning models such as SVM, Logistic Regression, Gradient Boosting, Decision Tree, KNeighbors, and Random Forest to achieve its goal of predicting whether or not a bank's clients would quit.

## Data Understanding

The data utilized in this project to undertake analysis and predictive modeling of bank customer churn was obtained from [kaggle](https://www.kaggle.com/datasets/mathchi/churn-for-bank-customers). The target dataset is a list of bank customers comprising information about 10,000 customers and 14 attributes for each client. The variables included in the dataset are listed in Table 1.
There is a significant imbalance between the fraction of churners and non-churners in the dataset. Additionally, in order to ascertain the percentages across genders, age groups, and other categories, we performed an exploratory data analysis. To prevent the classifiers from favoring the majority class of non-churners while forecasting the future, the data must be balanced before  inputting to the classifier. To accomplish the balance, we used oversampling methods (SMOTE).

About dataset

<figcaption align="center">Table 1 - The Description of Variables  </figcaption>  


|Variable | Description|
|--|--|
|RowNumber  | corresponds to the record (row) number and has no effect on the output. |
|CustomerId   |contains random values and has no effect on customer leaving the bank. |
| Surname|the surname of a customer has no impact on their decision to leave the bank.|
|CreditScore   | can have an effect on customer churn, since a customer with a higher credit score is less likely to leave the bank. |
| Geography  |  a customer’s location can affect their decision to leave the bank |
|Gender|it’s interesting to explore whether gender plays a role in a customer leaving the bank|
|Age  |  this is certainly relevant, since older customers are less likely to leave their bank than younger ones |
|Tenure |refers to the number of years that the customer has been a client of the bank.Normally, older clients are more loyal and less likely to leave a bank|
|Balance  | also a very good indicator of customer churn, as people with a higher balance in their accounts are less likely to leave the bank compared to those with lower balances. |
|NumOfProducts|refers to the number of products that a customer has purchased through the bank|
|HasCrCard  |denotes whether or not a customer has a credit card. This column is also relevant, since people with a credit card are less likely to leave the bank.  |
|IsActiveMember|active customers are less likely to leave the bank.  |
|Exited| whether or not the customer left the bank|
| EstimatedSalary |as with balance, people with lower salaries are more likely to leave the bank compared to those with higher salaries  |



> Exploratory data analysis (EDA)


<figure>  
<img  src="https://drive.google.com/uc?export=view&id=17vBlrc81FfduxD49mZngqTslLPl3PEq7">
<figcaption align="center">Fig.1 - EDA Geography, Gender, HasCrCard, and IsActiveMember  </figcaption>  
</figure>
  
  
From the visualization Fig.1 , we can observe: 

- Clients who have been with the bank for a short period of time or a long period of time are more likely to churn than those who have been with the bank for a longer period of time.

-   Female clients churn at a higher rate than male customers.
  
* The majority of clients that churned used credit cards. 

*  The vast majority of the information comes from French citizens. However, the proportion of churned customers is inversely linked to the population of consumers.

## Data Preparation


* check null data from dataset
* To get the statistical overview of the data, I used .describe()
* Since "RowNumber," "CustomerId," and "Surname" columns are not required in this analysis project because they have no bearing on the issue to be solved. We can eliminate that column.
* Exploring data using data visualization method.
* Applied correlation matrix, Encoding Categorical Data, and handling imbalance data using SMOTE.
* 

### Correlation Matrix

A correlation matrix is a table showing correlation coefficients between variables. Each cell in the table shows the correlation between two variables. A correlation matrix is used to summarize data, as an input into a more advanced analysis, and as a diagnostic for advanced analyses.
<figure>  
<img  src="https://drive.google.com/uc?export=view&id=15COG59Q9e95kcjrlKpFDtdGwZdd21whR">
<figcaption align="center">Fig.2 - Correlation Matrix </figcaption>  
</figure>

>From the visualization Fig.2 we can see that "Age" feature was found to have the strongest relationship with Exited. Here, we can infer that the rate of client loss grows as customer age increases, positive relationship.

* Plotting corelation between age and exited

  

Used a boxplot from the seaborn library to conduct basic visualization in order to find outliers in the datasets. to identify outliers in age column, look at the visualization below.

  
<figure>  
<img src="https://drive.google.com/uc?export=view&id=1mI1vJPUjbjnIP1UVs2ALgiaO7CK6povc">
<figcaption align="center">Fig.3 - Age Outlier  </figcaption>  
</figure>


in the visualization Fig.3  above we can identify some outliers. To drop the outliers I used the pandas library to handle the data in order to remove and clean the outliers.

 
Remove the outlier

  

### Encoding Categorical Data

To generate categorical data accessible to the various models, categorical data must first be translated into integer representation through the process of encoding.

  

the "geography" column contains categorical data, it should be one-hot encoded using the pandas library (pd.get dummies) to provide more features. We also transform the categorical information in "gender" into numerical information. Male is equal to 0, while female is equal to 1.


### [Imbalance data](https://developers.google.com/machine-learning/data-prep/construct/sampling-splitting/imbalanced-data)
A classification dataset with skewed class proportions is called imbalanced. Classes that make up a large proportion of the data set are called majority classes. Those that make up a smaller proportion are minority classes.

#### Handling imbalance data using SMOTE

SMOTE(Synthetic Minority Oversampling Technique) was utilized in this project to manage imbalance data. SMOTE is an oversampling technique where the synthetic samples are generated for the minority class. This algorithm helps to overcome the overfitting problem posed by random oversampling. It focuses on the feature space to generate new instances with the help of interpolation between the positive instances that lie together.

  
<figure>  
<img src="https://drive.google.com/uc?export=view&id=1ObVFa3TwvHce3tR8G1WSfJBhwzc2mdF1">
<figcaption align="center">Fig.4 - Imbalance Class </figcaption>  
</figure>
  
Fig. 4 shows that class 0 (not exited) is more dominant than class 1 (exited)
 
We can oversample the minority class using SMOTE and plot the transformed dataset.

The SMOTE class acts like a data transform object from scikit-learn in that it must be defined and configured, fit on a dataset, then applied to create a new transformed version of the dataset.

For example, we can define a SMOTE instance with default parameters that will balance the minority class and then fit and apply it in one step to create a transformed version of our dataset.

Once transformed, we can summarize the class distribution of the new transformed dataset, which would expect to now be balanced through the creation of many new synthetic examples in the minority class.


### Standar Scaler

Data scaling is a data preparation step for numerical features. Many machine learning algorithms, such as gradient descent methods, the KNN algorithm, linear and logistic regression, and so on, require data scaling to get good results. For this, various scalers are defined. [Standard Scaler](https://www.geeksforgeeks.org/data-pre-processing-wit-sklearn-using-standard-and-minmax-scaler/) assists in obtaining a standardized distribution with a mean of zero and a standard deviation of one (unit variance). By removing the feature's mean value and dividing the result by the feature's standard deviation, it standardizes the feature.

 

## Modeling

This project developed some models using a variety of modelling techniques. Some of models using hyperparameters tuning to improve the accuracy.
To build the best models, this project established a variety of modeling techniques.  Several models applied hyperparameter tuning to improve accuracy.

GridSearchCV to automate the tuning of hyperparameters.

>GridSearchCV is the process of performing hyperparameter tuning in order to determine the optimal values for a given model. As mentioned above, the performance of a model significantly depends on the value of hyperparameters. Note that there is no way to know in advance the best values for hyperparameters so ideally, we need to try all possible values to know the optimal values.


 ### 1.Logistic Regression


>Logistic regression analysis is one kind of regression analysis which response variable is categorical and predictor variables are either categorical or numerical. If the response variable consists of two categories called binary logistic regression. Whereas, if the response variable consists of more than two categories and the category is a level called ordinal logistic regression. The probability model between predictor variables X1i, X2i, ..., Xpi with response variables (π) is as follows[3]. 

We applied best estimator parameter C=0.03359818286283781, penalty='l1', solver='saga' . 

**C**: float, default=1.0
Inverse of regularization strength; must be a positive float. Like in support vector machines, smaller values specify stronger regularization.

**penalty**: {‘l1’, ‘l2’, ‘elasticnet’, ‘none’}, default=’l2’

Specify the norm of the penalty:

-   `'none'`: no penalty is added;
    
-   `'l2'`: add a L2 penalty term and it is the default choice;
    
-   `'l1'`: add a L1 penalty term;
    
-   `'elasticnet'`: both L1 and L2 penalty terms are added.

**solver**{‘newton-cg’, ‘lbfgs’, ‘liblinear’, ‘sag’, ‘saga’}, default=’lbfgs’

Algorithm to use in the optimization problem. Default is ‘lbfgs’. To choose a solver, you might want to consider the following aspects:

-   For small datasets, ‘liblinear’ is a good choice, whereas ‘sag’ and ‘saga’ are faster for large ones;
    
 -   For multiclass problems, only ‘newton-cg’, ‘sag’, ‘saga’ and ‘lbfgs’ handle multinomial loss;
     
 -   ‘liblinear’ is limited to one-versus-rest schemes.  

Accuracy score : 0.785624607658506



 ### 2.SVM

>SVM algorithm is one of supervised machine learning algorithms based on statistical learning theory [4]. It selects from the training samples a set of characteristic subsets so that the classification of the character subset is equivalent to the division of the entire dataset. the SVM has been used to solve different classification problems successfully in many applications [5,6]. So, the application scope is very broad due to its excellent learning ability. For example, intrusion detection, classification of facial expression, prediction of time series, speech recognition, image recognition, signal processing, detection of genes, text classification, recognition of fonts, diagnosis of faults, chemical analysis, recognition of images and other fields. In solving classification problems, the SVM algorithm has certain obvious advantages.  It's got a shorter prediction time [7][8]. The accuracy of the target detection classifier can be guaranteed by the global optimal solution. But there are some drawbacks, such as the model of detection is long-established. When processing large-scale data, time complexity and space complexity increase linearly with the increase in data [9,10]. In comparison, SVM is more capable of solving smaller samples, nonlinearity and high dimensionality problems compared with other classification algorithms [11, 12].

To optimize the accuracy we used 'C': 10, 'gamma': 0.1, 'kernel': 'rbf' as parameter

**C**: float, default=1.0

Regularization parameter. The strength of the regularization is inversely proportional to C. Must be strictly positive. The penalty is a squared l2 penalty.

**kernel**: {‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’} or callable, default=’rbf’

Specifies the kernel type to be used in the algorithm. If none is given, ‘rbf’ will be used. If a callable is given it is used to pre-compute the kernel matrix from data matrices; that matrix should be an array of shape  `(n_samples,  n_samples)`.
Accuracy score : 0.844632768361582

### 3. Random Forest

> Random Forest developed by Leo Breiman [13] is a group
of un-pruned classification or regression trees made from the random selection of samples of the training data. Random features are selected in the induction process. Prediction is made by aggregating (majority vote for classification or averaging for regression) the predictions of the ensemble. 

We found that the best accuracy was achieved using the parameters n_estimators=50, max_depth=16, random_state=55, n_jobs=-1. 

* max_features:

These are the maximum number of features Random Forest is allowed to try in individual tree. There are multiple options available in Python to assign maximum features. Here are a few of them :

1.  _Auto/None_  : This will simply take all the features which make sense in every tree. Here we simply do not put any restrictions on the individual tree.
2.  _sqrt_  : This option will take square root of the total number of features in individual run. For instance, if the total number of variables are 100, we can only take 10 of them in individual tree.”log2″ is another similar type of option for max_features.
3.  _0.2_  : This option allows the random forest to take 20% of variables in individual run. We can assign and value in a format “0.x” where we want x% of features to be considered.

*  n_estimators :

This is the number of trees you want to build before taking the maximum voting or averages of predictions. Higher number of trees give you better performance but makes your code slower.

* max_depth: int, default=None

The maximum depth of the tree. If None, then nodes are expanded until all leaves are pure or until all leaves contain less than min_samples_split samples.

* random_state: int, RandomState instance or None, default=None

Controls both the randomness of the bootstrapping of the samples used when building trees (if  `bootstrap=True`) and the sampling of the features to consider when looking for the best split at each node (if  `max_features  <  n_features`).


### 4. Gradient Boosting

> [Gradient Boosting](https://www.geeksforgeeks.org/ml-gradient-boosting/) is a popular boosting algorithm. In gradient boosting, each predictor corrects its predecessor’s error. In contrast to Adaboost, the weights of the training instances are not tweaked, instead, each predictor is trained using the residual errors of predecessor as labels.

We discovered that utilizing the parameters learning_rate=0.03, max_depth=8, n_estimators= 1000, subsample= 0.9 resulted in the highest accuracy.

**learning_rate**: float, default=0.1

Learning rate shrinks the contribution of each tree by  `learning_rate`. There is a trade-off between learning_rate and n_estimators. Values must be in the range  `(0.0,  inf)`.

**n_estimators**: int, default=100

The number of boosting stages to perform. Gradient boosting is fairly robust to over-fitting so a large number usually results in better performance. Values must be in the range  `[1,  inf)`.

**subsample**: float, default=1.0

The fraction of samples to be used for fitting the individual base learners. If smaller than 1.0 this results in Stochastic Gradient Boosting.  `subsample`  interacts with the parameter  `n_estimators`. Choosing  `subsample  <  1.0`  leads to a reduction of variance and an increase in bias. Values must be in the range  `(0.0,  1.0]`.

**max_depth**: int, default=3

The maximum depth of the individual regression estimators. The maximum depth limits the number of nodes in the tree. Tune this parameter for best performance; the best value depends on the interaction of the input variables. Values must be in the range  `[1,  inf)`.

### 5.KNN

>The k-Nearest-Neighbours (kNN) is a non-parametric classification method, which is simple but effective in many cases [14]. For a data record t to be classified, its k nearest neighbours are retrieved, and this forms a neighbourhood of t. Majority voting among the data records in the neighbourhood is usually used to decide the classification for t with or without consideration of distance-based weighting. However, to apply kNN we need to choose an appropriate value for k, and the success of classification is very much dependent on this value. In a sense, the kNN method is biased by k. There are many ways of choosing the k value, but a simple one is to run the algorithm many times with different k values and choose the one with the best performance.

We found that the parameters n_neighbors=10  produced the better accuracy.

**n_neighbors**: int, default=5

Number of neighbors to use by default for  [`kneighbors`](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html#sklearn.neighbors.KNeighborsClassifier.kneighbors "sklearn.neighbors.KNeighborsClassifier.kneighbors")  queries.


### 6.Desicion Tree

>A  decision  tree  is  a  classiﬁer  expressed  as  a  recursive  partition  of  the  instance  space.  The  decision  tree  consists  of  nodes  that  form  a  rooted  tree, meaning  it  is  a  directed  tree  with  a  node  called  “root”  that  has  no  incoming edges.  All other nodes  have exactly one incoming  edge.  A node with  outgoing edges is called  an internal or  test  node.  All  other  nodes  are  called  leaves (also known  as  terminal  or  decision  nodes).  In  a  decision  tree,  each  internal  node splits  the  instance  space  into  two  or  more  sub-spaces  according  to  a  certain discrete  function  of  the  input  attributes  values. 







## Evaluation

 
This project used Accuracy, Precision, Recall, and F1 to evaluate the model as Accuracy, Precision, Recall, and F1  are the best metrics for assessing classification problems. 


Accuracy
Accuracy is a ratio of the true detected cases to the total cases, and it has been utilized to evaluate models on a balanced dataset [24]. Accordingly, it can be calculated as:
>Accuracy = TP+TN/TP+FP+FN+TN

Accuracy is the most straightforward intuitive performance metric because it is just a ratio of correctly predicted observations to total observations. One can believe that our model is the best if we have high accuracy. Accuracy is a helpful indicator, but only when the values of false positives and false negatives are almost equal. As a result, you must consider additional criteria while evaluating the model's performance. Our model received the most excellent accuracy of 0.88, indicating that it is about 88% correct.

Precision

Precision is the proportion of accurately predicted positive observations to all expected positive statements. High accuracy is associated with a low false positive rate. We have 0.77 accuracy, which is rather good.

>Precision = TP/TP+FP


Recall and F1-score

Recall: calculates the ratio of retrieved relevant churns over the total number of a relevant customer churning [25]. F1-score allows combining both precisions and recall into a single measure that captures both properties.

>Recall = TP/TP+FN
>F1 Score = 2*(Recall * Precision) / (Recall + Precision)

 
True Positives (TP) - These are the correctly predicted positive values which means that the value of actual class is yes and the value of predicted class is also yes. E.g. if actual class value indicates that the costumers is churn/exited and predicted class tells the same thing.

True Negatives (TN) - These are the correctly predicted negative values which means that the value of actual class is no and value of predicted class is also no. E.g. if actual class says that the costumers is not churn and predicted class tells the same thing.

False positives and false negatives, these values occur when your actual class contradicts with the predicted class.

False Positives (FP) – When actual class is no and predicted class is yes. E.g. if actual class says the customers not exited/churn but predicted class tells you that the costumers indicate exited.

False Negatives (FN) – When actual class is yes but predicted class in no. E.g. if actual class value indicates that the costumers exited and predicted class tells that the costumers are nit exited.


## Conclusion

 - Using machine learning  techniques, we aimed to develop a churn prediction model for this project.
* The data set had 10000 rows, and there were no missing values. The dataset had 14 variables.

- Clients who have been with the bank for a short period of time or a long period of time are more likely to churn than those who have been with the bank for a longer period of time.

- Customers between the ages of 40 and 60 were more likely to leave the bank.

- The bank's target market or retention strategy for various age groups may need to be reconsidered.

-   Female clients churn at a higher rate than male customers.
  
* The majority of clients that churned used credit cards. 

*  The vast majority of the information comes from French citizens. However, the proportion of churned customers is inversely linked to the population of consumers.
* Predictions were made with a total of 6 classification models. 
* Utilized Random Forest and Gradient Boosting to predict customer churn
-   Both models have good accuracies score, with the Gradient Boosting model performing slightly better than the Random Forest model (88% versus 85%)

#### Summary evaluation
model |accuracy|recall|precision|f1\_score|
|---|---|---|---|---|
|Logistic Regression|0\.78|0\.79|0\.77|0\.78|
|SVM|0\.84|0\.84|0\.82|0\.84|
|Decision Tree|0\.84|0\.84|0\.82|0\.79|
|GradientBoosting|0\.88|0\.88|0\.86|0\.88|
|KNeighbors|0\.82|0\.80|0\.82|0\.82|
|Random Forest|0\.85|0\.86|0\.84|0\.85|
  


> Six models were used to make predictions. The Gradient Boosting approach was used to choose the highest accuracy score.

#
[1] S.A. Qureshi, A.S. Rehman, A.M. Qamar, A. Kamal, _Telecommunication Subscribers’ Churn Prediction Model Using Machine Learning_ (IEEE, 2013), pp. 131–136

[2] B. Mishachandar, K.A. Kumar, Predicting customer churn using targeted proactive retention.Int. J. Eng. Technol. 7(2.27), 69–76 (2018)

[3] Hosmer, David W., and Stanley Lemeshow. Special topics. John Wiley & Sons, Inc., 2000.
89

[4] Wendong, Y., Zhengzheng, L., & Bo, J. (2017, June). A multi-factor analysis model of quantitative investment based on GA and SVM. In 2017 2nd International Conference on Image, Vision and Computing (ICIVC) (pp. 1152-1155). IEEE.

[5] Kareem, F. Q., & Abdulazeez, A. M. Ultrasound Medical Images
Classification Based on Deep Learning Algorithms: A Review.

[6]- Zeebaree, Diyar Qader, Habibollah Haron, and Adnan Mohsin
Abdulazeez. "Gene selection and classification of microarray data
using convolutional neural network." In 2018 International Conference on Advanced Science and Engineering (ICOASE), pp.145-150. IEEE, 2018.

[7] Abdulqader, Dildar Masood, Adnan Mohsin Abdulazeez, and Diyar Qader Zeebaree. "Machine Learning Supervised Algorithms of Gene Selection: A Review." Machine Learning 62, no. 03 (2020).

[8] Zebari, R., Abdulazeez, A., Zeebaree, D., Zebari, D., & Saeed, J.
(2020). A Comprehensive Review of Dimensionality Reduction
Techniques for Feature Selection and Feature Extraction. Journal of
Applied Science and Technology Trends, 1(2), 56-70.

[9] Dai, H. (2018, March). Research on SVM improved algorithm for
large data classification. In 2018 IEEE 3rd International Conference
on Big Data Analysis (ICBDA) (pp. 181-185). IEEE.

[10] Haji, S. H., & Abdulazeez, A. M. (2021). COMPARISON OF
OPTIMIZATION TECHNIQUES BASED ON GRADIENT
DESCENT ALGORITHM: A REVIEW. PalArch's Journal of
Archaeology of Egypt/Egyptology, 18(4), 2715-2743.

[11] Tao, P., Sun, Z., & Sun, Z. (2018). An improved intrusion detection algorithm based on GA and SVM. Ieee Access, 6, 13624-13631.

[12] Chauhan, V. K., Dahiya, K., & Sharma, A. (2019). Problem formulations and solvers in linear SVM: a review. Artificial Intelligence Review, 52(2), 803-855.

[13] Breiman, L., Random Forests, Machine Learning 45(1), 5-32, 2001.

[14] D. Hand, H. Mannila, P. Smyth.: Principles of Data Mining. The MIT Press. (2001)
