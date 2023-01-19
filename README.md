# Breast-Cancer-Prediction-using-Machine-Learning
This project aims to predict whether a tumor is cancerous (malignant) or non-cancerous (benign) using machine learning algorithms.

**Description**<br>
Breast cancer is a disease in which the cells of breast tissue undergo uncontrolled division, resulting in a lump or mass in that region. Early diagnosis of breast cancer is crucial for effective treatment and can save lives. The goal of this project is to develop a model that can accurately predict whether a tumor is cancerous or not.

The project uses the Breast Cancer Wisconsin (Diagnostic) dataset, which contains 30 features and 569 instances with 357 benign and 212 malignant records. The dataset is used to train and test different machine learning algorithms, including:
- Logistic Regression
- K-Nearest Neighbours Classifier
- Naive Bayes Classifier
- Support Vector Machine Classifier
- Light Gradient Boosted Machine Classifier

For each algorithm, the following performance metrics are obtained:
- Accuracy
- Precision
- Recall
- F1 Score
- AUC-ROC Curve

We also obtain the confusion matrix, which gives the number of true positives, true negatives, false positives and false negatives. This can be used to understand the performance of the model on different classes.

The Receiver Operating Characteristic (ROC) curve is a graphical representation of the performance of a binary classifier system as the discrimination threshold is varied. The area under the curve (AUC) is a measure of the overall performance of the model.

In addition, the importance of each feature is obtained, which can be used to identify the most important features for the prediction of breast cancer.

**Libraries**<br>
The project is implemented in Python 3.6 environment using Jupyter Notebook and the following libraries:
- Numpy v1.19.1
- Matplotlib v3.3.1
- Plotly v4.9.0
- Seaborn v0.10.1
- Sci-kit learn v0.23.2

**Implementation**<br>
To run the project, open the 'Breast Cancer Prediction using Machine Learning.ipynb' file using Jupyter Notebook and click on the Run button |>>|

**Conclusion**<br>
This project demonstrates how machine learning algorithms can be applied to predict breast cancer with high accuracy. It also shows the importance of evaluating the model's performance using various metrics, such as accuracy, precision, recall, F1 score, and AUC-ROC curve, as well as identifying the most important features that contribute to the prediction. The implementation of the project uses the Breast Cancer Wisconsin (Diagnostic) dataset and the machine learning libraries in Python to train and test the models.
