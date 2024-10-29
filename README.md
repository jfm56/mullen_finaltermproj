

Classification Algorithm Comparison using Student Performance Dataset


1. Introduction
This project evaluates the performance of three machine learning algorithms: Decision Tree, Random Forest, and Support Vector Machine (SVM). Additionally, an LSTM model was developed to explore a neural network-based approach to classification. The dataset used is the Student Performance dataset, which contains demographic data, school-related variables, and personal information.
The primary goal is to predict whether a student is at risk of failing based on a range of features. Such predictions can help educators target at-risk students with specific interventions.


2. Algorithms and Dataset
Dataset
The dataset used is the Student Performance dataset (student-mat.csv). The target variable G3 (final grade) was transformed into a binary class, where students who scored less than 9 are classified as 'F' (Fail), and all others are classified as 'Not Failed'.
Dataset Source: The dataset can be downloaded from the UCI Machine Learning Repository at this link.
Purpose: The primary objective is to predict whether a student will fail or pass based on the given attributes, such as school-related features, demographic data, and socio-economic factors.
Algorithms
1.	Decision Tree: A decision tree classifier that splits data into branches based on feature values to create a tree-like structure. It uses entropy to guide splitting decisions and attempts to maximize the separation between classes ('F' vs. 'Not Failed').
2.	Random Forest: An ensemble method that aggregates the predictions from multiple decision trees. This helps improve robustness and reduces the risk of overfitting.
3.	Support Vector Machine (SVM): SVM separates classes using an optimal hyperplane that maximizes the margin between class labels.
4.	LSTM Neural Network: An LSTM model trained on the standardized dataset to capture sequential relationships between the features.
Data Preprocessing
•	One-Hot Encoding: Categorical variables were converted into numerical values using one-hot encoding. This transformation helps classifiers interpret non-numeric features like school type, family support, etc.
•	Standard Scaling: All features were scaled using StandardScaler to ensure that the data’s distribution is centered and has a standard deviation of 1.


3. Methodology
•	10-Fold Cross-Validation: Each algorithm was evaluated using 10-fold cross-validation to ensure robust performance and minimize the effects of random data splits. This technique divides the dataset into 10 equal parts, training on 9 parts and validating on 1, repeating the process 10 times.
•	Evaluation Metrics:
o	Confusion Matrix: A summary of correct and incorrect predictions. It includes:
	True Positives (TP): Students correctly predicted as failing.
	True Negatives (TN): Students correctly predicted as not failing.
	False Positives (FP): Students incorrectly predicted as failing.
	False Negatives (FN): Students incorrectly predicted as not failing.
o	ROC-AUC (Receiver Operating Characteristic - Area Under the Curve): Measures the trade-off between True Positive Rate and False Positive Rate. A higher AUC indicates a better-performing model in distinguishing between classes.
•	Feature Importance Analysis: The Decision Tree model identifies the most influential features in predicting whether a student fails.

4. Results
Model Performance:
Decision Tree:
•	Training Accuracy: 1.000
•	Test Accuracy: 0.886
•	Most Important Feature: failures
•	Second Most Important Feature: travel time
Interpretation: The Decision Tree model perfectly fit the training data, indicating possible overfitting. Its test accuracy of 88.6% suggests that it captures the data patterns but may struggle to generalize to unseen samples.
Scatter Plot Analysis
The scatter plot visualizes the decision surface using the two most important features, failures (past failures) and travel time. The plot is color-coded to indicate predicted classes:
•	Red points represent students predicted to fail (class 'F').
•	Blue points represent students predicted as not failing (class 'Not Failed').
This visualization illustrates how the Decision Tree classifier uses these two key features to distinguish between students at risk of failing and those who are not.
  
Random Forest:
•	Test Accuracy: 0.886
•	Confusion Matrix:
[[ 6  5]  # TN, FP
 [ 1 67]] # FN, TP
 
 Interpretation: The Random Forest performed well in predicting students who did not fail while minimizing false predictions. It achieved a high accuracy and demonstrated strong generalization across test data.
Support Vector Machine (SVM):
•	Test Accuracy: 0.862
•	Confusion Matrix:
[[ 0 11]  # TN, FP
 [ 0 68]] # FN, TP
Interpretation: The SVM classifier correctly predicted all failing students, but it struggled with predicting students who did not fail, resulting in higher false positives. This discrepancy suggests that the SVM hyperplane may have been too focused on separating the failing class.
LSTM Model:
•	Training Accuracy after Epoch 10: 91.89%
•	Validation Accuracy: 89.87%
Interpretation: The LSTM model consistently improved its performance over the 10 epochs, achieving a validation accuracy of 89.87%. This suggests that the neural network captured sequential relationships effectively.
10-Fold Cross-Validation Results:
•	Random Forest: 10-fold cross-validation average accuracy: 0.911
•	SVM: 10-fold cross-validation average accuracy: 0.843
Interpretation: The Random Forest model achieved the highest average accuracy during cross-validation, indicating its effectiveness in generalizing across multiple data subsets.
Receiver Operating Characteristic (ROC) Curves:
The ROC curves illustrate the performance of the Random Forest and SVM classifiers. Each curve plots the True Positive Rate (TPR) against the False Positive Rate (FPR) at different thresholds.
•	Random Forest: AUC = 0.83
•	SVM: AUC = 0.87
Interpretation: The SVM model achieved a slightly higher AUC, indicating that it was more effective in distinguishing between the two classes. However, the differences in confusion matrices suggest that Random Forest was more balanced in its predictions.
ROC Curve Plot
  
5. Comparison and Discussion
Accuracy Comparison:
•	Random Forest consistently achieved the highest average accuracy, demonstrating its robustness in generalizing well across different data subsets.
•	The confusion matrices indicate that SVM struggled with correctly identifying non-failing students, while Random Forest showed better overall sensitivity and specificity.
Feature Importance and Scatter Plot Analysis:
•	The Decision Tree identified failures as the most significant feature in determining a student’s likelihood of failing. The second most important feature was travel time. A scatter plot using these two features provides a visual representation of how these factors contribute to the classification.
Overfitting:
•	The Decision Tree model’s perfect accuracy on training data indicates that it overfitted the training set, resulting in slightly lower test accuracy. Overfitting is common in Decision Trees due to their propensity to create highly specific branches.
Justification:
•	Random Forest: By averaging the results from multiple decision trees, Random Forest reduces the risk of overfitting and improves overall robustness. This allowed it to achieve high test accuracy and cross-validation scores.
•	SVM: Although SVM achieved a higher AUC score, it struggled with false positives, indicating that its decision boundary was not well-suited to all data points in this particular dataset.


6. Conclusion
The Random Forest classifier emerged as the most effective model for this binary classification task, providing higher accuracy and balanced performance metrics compared to Decision Tree and SVM models. It demonstrated robustness in handling various student-related features and produced accurate classifications for predicting failing students.
Educators can use these findings to identify students at risk of failing and develop interventions targeting key features influencing failure rates. Future improvements could involve hyperparameter tuning, more sophisticated ensemble techniques, or a focus on further understanding feature relationships within the dataset.




