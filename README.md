# Ensemble Learning
---


In this project, we are going to evaluate logistic regression classifiers using the resampling on the LendingClub data in which more than 80 different features what could be related to assessing the credit risk. Credit risk is an inherently imbalanced classification problem as the number of good loans is much larger than the number of at-risk lonas.

What we are going to do is to put the *'loan_status'* value as an output(y) while taking all other features as independent variables(X). We don't know what feature weighs over the others in determining the importances of features that make one a bad loan. The idea to find them is to run different ensemble learners, comparing the results from each algorithm, determining which one gives the best performance.

Before doing that, we need a data preprocessing process to make the data suitable for the models. As mentioned, the number of outputs could be highly biased to good loans(or classifiead as 'low-risk'). In the Jupyter notebook, you may find that the number of *'low_risk'* is 68470 against that of *'high_risk'* 347. This imbalanced data could be leading the model to render a biased result. For avoiding the misleading, we applied several resampling methods on them; the naive random oversampling, the SMOTE oversampling, the Cluster Centroids undersampling and the combination(over and under) sampling using the **SMOTEENN** algorithm.

After splitting the data into train and test, we needed at least two steps - encoding and scaling - for preprocessing before going into the resampling. We found there were 9 columns containing categorical data as below.

```column_list = ['home_ownership', 'verification_status', 'issue_d', 'pymnt_plan', 'initial_list_status', 'next_pymnt_d', 'application_type', 'hardship_flag', 'debt_settlement_flag']```

**LabelEncoder** could be a simple tool for encoding categorical data into numerical one, but there is still a issue that Machine Learnig algotrithms will assume that two nearby values are more similar tha two distant values. Definitely this is not the case. To fix this issue, we tried to create a binary encoding by using pd.get_dummies function. 

Next, we might have encounter another issue unless we do concern of feature scaling for the input data differently ranging in different scales. We standardized and transformed it by using the **StandardScaler** tool.

## Resampling Evaluation

As a result, we got each balanced accuracy scores and classification report

### 1. Naive Random Oversampling   
   Accuracy Score: 0.832780705572896

![Naive Random Oversampling](images\naive_random_oversampling.png)      

### 2. SMOTE Oversampling   
   Accuracy Score: 0.8388510243681058

![SMOTE Oversampling](images/SMOTE.png)   

### 3. Undersampling   
   Accuracy Score: 0.8215575767118339

![Undersampling](images/undersampling.png)   

### 4. Combination Sampling(SMOTEENN)   
   Accuracy Score: 0.8388319216626994

![Combination sampling](images/combination_sampling.png)   


#### Which model had the best balanced accuracy score?   

> SMOTE Oversampling model shows the highest score with a very slight difference with the others.

#### Which model had the best recall score?   
> Again, SMOTE has the best overall recall score at 0.87. However, we are more concerning about detecting high_risk(the bad loans) in the project, the Undersampling shows the highest recall score in terms of predicting high_risk, regardless of its overall poor score at 0.76.

#### Which model had the best geometric mean score?    

> SMOTE and SMOTEENN both scored at 0.84, the highest.

## Ensemble Model Evaluation

After the resampling process, we trained and tested two different ensemble classifers - the **Balanced Random Forest Classifier** and the **Adaboost Classifier**- to predict loan risk and evaluated each model. 

Below is the result.

### 1. Balanced Random Forest   
   Accuracy Score: 0.7887512850910909

![Balanced Random Forest](images/Randomforest.png)      

### 2. AdaBoost   
   Accuracy Score: 0.931601605553446

![Adaboost](images/adaboost.png)   


#### Which model had the best balanced accuracy score?    
> AdaBoost got much higher score at 0.931 while Random Forest was just at 0.788. This score make us interesting.

#### Which model had the best recall score?    
> AdaBoost. It outperforms Random Forest both in overall and each segement of recall. Especially, we'd like to focus on the recall score at 0.92 for AdaBoost, which is far better than that of Random Forest at 0.70.

#### Which model had the best geometric mean score?   
> Again, Adaboost takes the lead at 0.93

#### What are the top three features?   
> According to the result of importances shown as below,
![Feature Importances](images/feature_importance.png)    

> 'total_rec_prncp', 'total_pymt' and 'total_pymnt_inv' are the top three features. However, we need to be discreet about the accuracy of the result because the Random Forest model does not perform well enough to take these things into account seriously. There might have been other features actually found to be more important than those top three if we improved the model with fine-tuning.
 


