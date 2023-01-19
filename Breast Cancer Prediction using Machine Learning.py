#!/usr/bin/env python
# coding: utf-8

# # Breast Cancer Prediction using Machine Learning

# In[1]:


#Importing the required libraries.
import numpy as np 
import pandas as pd
import os
import matplotlib.pyplot as plt
from PIL import  Image
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import itertools
import warnings
warnings.filterwarnings("ignore")
import io
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
import plotly.figure_factory as ff


# In[2]:


#Reading the dataset stored as a .csv file.
dataset = pd.read_csv("breast-cancer-wisconsin.csv")

##Displaying the first five rows of the dataset.
dataset.head()


# # Data Preprocessing

# In[3]:


#Data overview
print ("Rows     : " ,dataset.shape[0])
print ("Columns  : " ,dataset.shape[1])
print ("\nFeatures : \n" ,dataset.columns.tolist())
print ("\nMissing values :  ", dataset.isnull().sum().values.sum())
print ("Unique values  :  \n",dataset.nunique())


# In[4]:


#Separating benign and malignant patients.
benign     = dataset[dataset["diagnosis"] == "B"]
malignant = dataset[dataset["diagnosis"] == "M"]

#Assigning numercial values to catagorical target column.
dataset["diagnosis"] = dataset["diagnosis"].replace({1:"M",0:"B"})


# In[5]:


#Separating catagorical and numerical columns.
Id_col     = ['id']
target_col = ["diagnosis"]
cat_cols   = dataset.nunique()[dataset.nunique() < 6].keys().tolist()
cat_cols   = [x for x in cat_cols if x not in target_col]
num_cols   = [x for x in dataset.columns if x not in cat_cols + target_col + Id_col]


# In[6]:


#Exploratory Data Analysis.

l = ['Benign', 'Malignant']
v = dataset["diagnosis"].value_counts().values.tolist()

trace = go.Pie(labels = l, values = v,
               marker = dict(colors =  [ 'black' ,'#FF96A7'], line = dict(color = "white", width =  1.3)),
               rotation = 90,
               hoverinfo = "label+value+text",
               hole = .5
              )
layout = go.Layout(dict(title = "Patient Classification in data",
                        plot_bgcolor  = "rgb(243,243,243)",
                        paper_bgcolor = "rgb(243,243,243)",
                       )
                  )

data = [trace]
fig = go.Figure(data = data,layout = layout)
py.iplot(fig)


# In[7]:


#Data preprocessing.
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

#customer id col
Id_col     = ['id']
#Target columns
target_col = ["diagnosis"]
#categorical columns
cat_cols   = dataset.nunique()[dataset.nunique() < 6].keys().tolist()
cat_cols   = [x for x in cat_cols if x not in target_col]
#numerical columns
num_cols   = [x for x in dataset.columns if x not in cat_cols + target_col + Id_col]
#Binary columns with 2 values
bin_cols   = dataset.nunique()[dataset.nunique() == 2].keys().tolist()
#Columns more than 2 values
multi_cols = [i for i in cat_cols if i not in bin_cols]

#Label encoding Binary columns.
le = LabelEncoder()
for i in bin_cols :
    dataset[i] = le.fit_transform(dataset[i])
    
#Duplicating columns for multi value columns.
dataset = pd.get_dummies(data = dataset, columns = multi_cols )

#Scaling Numerical columns.
std = StandardScaler()
scaled = std.fit_transform(dataset[num_cols])
scaled = pd.DataFrame(scaled,columns=num_cols)

#Dropping original values merging scaled values for numerical columns.
df_dataset_og = dataset.copy()
dataset = dataset.drop(columns = num_cols,axis = 1)
dataset = dataset.merge(scaled,left_index=True,right_index=True,how = "left")


# In[8]:


#Features Summary.
summary = (df_dataset_og[[i for i in df_dataset_og.columns if i not in Id_col]].
           describe().transpose().reset_index())

summary = summary.rename(columns = {"index" : "feature"})
summary = np.around(summary,3)

val_lst = [summary['feature'], summary['count'], summary['mean'], summary['std'], summary['min'], summary['25%'],
           summary['50%'], summary['75%'], summary['max']]

trace  = go.Table(header = dict(values = summary.columns.tolist(), 
                                line = dict(color = ['#506784']), 
                                fill = dict(color = ['#119DFF']),
                               ),
                  cells  = dict(values = val_lst,
                                line = dict(color = ['#506784']),
                                fill = dict(color = ["lightgrey",'#F5F8FF'])
                               ),
                  columnwidth = [180,60,100,100,60,60,80,80,80])
layout = go.Layout(dict(title = "Features Summary"))
figure = go.Figure(data=[trace],layout=layout)
py.iplot(figure)


# In[9]:


#Correlation Matrix.
correlation = dataset.corr()

#tick labels
matrix_cols = correlation.columns.tolist()

#convert to array
corr_array  = np.array(correlation)

#Plotting
trace = go.Heatmap(z = corr_array, x = matrix_cols, y = matrix_cols,
                   colorscale = "burg",
                   colorbar   = dict(title = "Pearson Correlation coefficient",
                                     titleside = "right"
                                    ) ,
                  )

layout = go.Layout(dict(title = "Correlation Matrix for Features",
                        autosize = False,
                        height  = 720, width   = 800,
                        margin  = dict(r = 0 ,l = 210,
                                       t = 25,b = 210,
                                      ),
                        yaxis   = dict(tickfont = dict(size = 9)),
                        xaxis   = dict(tickfont = dict(size = 9))
                       )
                  )

data = [trace]
fig = go.Figure(data=data,layout=layout)
py.iplot(fig)


# # Model Building

# # 1. Logistic Regression

# In[10]:


#Baseline Model
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
from sklearn.metrics import roc_auc_score,roc_curve,scorer
from sklearn.metrics import f1_score
import statsmodels.api as sm
from sklearn.metrics import precision_score,recall_score

#splitting train and test data 
train,test = train_test_split(dataset,test_size = .25 ,random_state = 111)
    
##seperating dependent and independent variables
cols    = [i for i in dataset.columns if i not in Id_col + target_col]
train_X = train[cols]
train_Y = train[target_col]
test_X  = test[cols]
test_Y  = test[target_col]

#Function attributes
#dataframe     - processed dataframe
#Algorithm     - Algorithm used 
#training_x    - predictor variables dataframe(training)
#testing_x     - predictor variables dataframe(testing)
#training_y    - target variable(training)
#training_y    - target variable(testing)
#cf - ["coefficients","features"](cooefficients for logistic regression, features for tree based models)

def breast_cancer_prediction(algorithm,training_x,testing_x,training_y,testing_y,cols,cf) :
    
    #model
    algorithm.fit(training_x,training_y)
    predictions   = algorithm.predict(testing_x)
    probabilities = algorithm.predict_proba(testing_x)
    
    #coeffs
    if   cf == "coefficients" :
        coefficients  = pd.DataFrame(algorithm.coef_.ravel())
    elif cf == "features" :
        coefficients  = pd.DataFrame(algorithm.feature_importances_)
        
    column_df     = pd.DataFrame(cols)
    coef_sumry    = (pd.merge(coefficients,column_df,left_index= True, right_index= True, how = "left"))
    coef_sumry.columns = ["coefficients","features"]
    coef_sumry    = coef_sumry.sort_values(by = "coefficients",ascending = False)
    
    print (algorithm)
    print ("\n Classification report : \n",classification_report(testing_y,predictions))
    print ("Accuracy   Score : ",accuracy_score(testing_y,predictions))
    
    #confusion matrix
    conf_matrix = confusion_matrix(testing_y,predictions)
    
    #roc_auc_score
    model_roc_auc = roc_auc_score(testing_y,predictions) 
    print ("Area under curve : ",model_roc_auc,"\n")
    fpr,tpr,thresholds = roc_curve(testing_y,probabilities[:,1])
    
    #Plot Confusion Matrix
    trace1 = go.Heatmap(z = conf_matrix , x = ["Benign","Malignant"], y = ["Benign","Malignant"],
                        showscale  = False, colorscale = "burg", name = "matrix")
    
    #Plot ROC Curve
    trace2 = go.Scatter(x = fpr,y = tpr,
                        name = "Roc : " + str(model_roc_auc),
                        line = dict(color = ('rgb(22, 96, 167)'),width = 2))
    trace3 = go.Scatter(x = [0,1],y=[0,1],
                        line = dict(color = ('rgb(205, 12, 24)'),width = 2,
                        dash = 'dot'))
    
    #plot coeffs
    trace4 = go.Bar(x = coef_sumry["features"],y = coef_sumry["coefficients"],
                    name = "coefficients",
                    marker = dict(color = coef_sumry["coefficients"],
                                  colorscale = "burg",
                                  line = dict(width = .6,color = "black")))
    
    #subplots
    fig = tls.make_subplots(rows=2, cols=2, specs=[[{}, {}], [{'colspan': 2}, None]],
                            subplot_titles=('Confusion Matrix',
                                            'Receiver operating characteristic',
                                            'Feature Importances'))
    
    fig.append_trace(trace1,1,1)
    fig.append_trace(trace2,1,2)
    fig.append_trace(trace3,1,2)
    fig.append_trace(trace4,2,1)
    
    fig['layout'].update(showlegend=False, title="Model performance" ,
                         autosize = False,height = 900,width = 800,
                         plot_bgcolor = 'rgba(240,240,240, 0.95)',
                         paper_bgcolor = 'rgba(240,240,240, 0.95)',
                         margin = dict(b = 195))
    fig["layout"]["xaxis2"].update(dict(title = "false positive rate"))
    fig["layout"]["yaxis2"].update(dict(title = "true positive rate"))
    fig["layout"]["xaxis3"].update(dict(showgrid = True,tickfont = dict(size = 10),
                                        tickangle = 90))
    py.iplot(fig)
    
logit  = LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
          intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
          penalty='l2', random_state=None, solver='liblinear', tol=0.0001,
          verbose=0, warm_start=False)

breast_cancer_prediction(logit,train_X,test_X,train_Y,test_Y,
                         cols,"coefficients")


# <b>Inferences</b><br>
# <b>Confusion Matrix</b><br>
# From the confusion matrix, we observe that the algorithm classifies 53 Malignant (True Positives) and 85 Benign instances (True Negatives) correctly while it classifies 4 benign tumors as malignant (False Positives) and 1 malignant tumor as benign (False Negative).
# 
# <b>Receiver operating characteristic(ROC) Curve</b><br>
# The ROC Area under the curve(AUC) score of 0.959 and the curve, both demonstrate how well the algorithm was able to classify the 
# records.
# 
# <b>Feature Importance</b><br>
# From the Feature Importance graph, we observe that features like Compactness, Symmetry and Fractal Dimension contribute towards the tumor being detected as malignant whereas features such as Radius, Area and Concavity contribute towards the tumor being detected as benign.  

# # 2. Logistic Regression - Synthetic Minority Oversampling Technique (SMOTE)

# In[11]:


#Synthetic Minority Oversampling Technique (SMOTE)
from imblearn.over_sampling import SMOTE

cols    = [i for i in dataset.columns if i not in Id_col+target_col]

smote_X = dataset[cols]
smote_Y = dataset[target_col]

#Split train and test data
smote_train_X,smote_test_X,smote_train_Y,smote_test_Y = train_test_split(smote_X,smote_Y,
                                                                         test_size = .25 ,
                                                                         random_state = 111)

#oversampling minority class using smote
os = SMOTE(random_state = 0)
os_smote_X,os_smote_Y = os.fit_sample(smote_train_X,smote_train_Y)
os_smote_X = pd.DataFrame(data = os_smote_X,columns=cols)
os_smote_Y = pd.DataFrame(data = os_smote_Y,columns=target_col)


logit_smote = LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
          intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
          penalty='l2', random_state=None, solver='liblinear', tol=0.0001,
          verbose=0, warm_start=False)

breast_cancer_prediction(logit_smote,os_smote_X,test_X,os_smote_Y,test_Y,cols,"coefficients")


# <b>Inferences</b><br>
# The Modifications made to the Logistic Regression Algorithm using SMOTE helped the classifier better classify the tumores which is evident from the higher accuracy<br>
# <b>Confusion Matrix</b><br>
# From the confusion matrix, we observe that the algorithm classifies 55 Malignant (True Positives) and 84 Benign instances (True Negatives) correctly while it classifies 2 benign tumors as malignant (False Positives) and 2 malignant tumor as benign (False Negative).
# 
# <b>Receiver operating characteristic(ROC) Curve</b><br>
# The ROC Area under the curve(AUC) score of 0.97 and the curve, both demonstrate how well the algorithm was able to classify the 
# records.
# 
# <b>Feature Importance</b><br>
# From the Feature Importance graph, we observe that features like Compactness, Symmetry and Fractal Dimension contribute towards the tumor being detected as malignant whereas features such as Radius, Area and Concavity contribute towards the tumor being detected as benign.  

# # 3. K - Nearest Neighbors Classifier

# In[12]:


#KNN Classifier
#Applying knn algorithm to SMOTE oversampled data.
def breast_cancer_prediction_alg(algorithm,training_x,testing_x,training_y,testing_y) :
    
    #model
    algorithm.fit(training_x,training_y)
    predictions   = algorithm.predict(testing_x)
    probabilities = algorithm.predict_proba(testing_x)
    
    print (algorithm)
    print ("\n Classification report : \n",classification_report(testing_y,predictions))
    print ("Accuracy Score   : ",accuracy_score(testing_y,predictions))
    
    #confusion matrix
    conf_matrix = confusion_matrix(testing_y,predictions)
    
    #roc_auc_score
    model_roc_auc = roc_auc_score(testing_y,predictions) 
    print ("Area under curve : ",model_roc_auc)
    fpr,tpr,thresholds = roc_curve(testing_y,probabilities[:,1])
     
    #plot roc curve
    trace1 = go.Scatter(x = fpr,y = tpr,
                        name = "Roc : " + str(model_roc_auc),
                        line = dict(color = ('rgb(22, 96, 167)'),width = 2),
                       )
    trace2 = go.Scatter(x = [0,1],y=[0,1],
                        line = dict(color = ('rgb(205, 12, 24)'),width = 2,
                        dash = 'dot'))
    
    #plot confusion matrix
    trace3 = go.Heatmap(z = conf_matrix ,x = ["Benign","Malignant"],
                        y = ["Benign","Malignant"],
                        showscale  = False,colorscale = "Burg",name = "matrix",
                        xaxis = "x2",yaxis = "y2"
                       )
    
    layout = go.Layout(dict(title="Model performance" ,
                            autosize = False,height = 500,width = 800,
                            showlegend = False,
                            plot_bgcolor  = "rgb(243,243,243)",
                            paper_bgcolor = "rgb(243,243,243)",
                            xaxis = dict(title = "false positive rate",
                                         gridcolor = 'rgb(255, 255, 255)',
                                         domain=[0, 0.6],
                                         ticklen=5,gridwidth=2),
                            yaxis = dict(title = "true positive rate",
                                         gridcolor = 'rgb(255, 255, 255)',
                                         zerolinewidth=1,
                                         ticklen=5,gridwidth=2),
                            margin = dict(b=200),
                            xaxis2=dict(domain=[0.7, 1],tickangle = 90,
                                        gridcolor = 'rgb(255, 255, 255)'),
                            yaxis2=dict(anchor='x2',gridcolor = 'rgb(255, 255, 255)')
                           )
                  )
    data = [trace1,trace2,trace3]
    fig = go.Figure(data=data,layout=layout)
    py.iplot(fig)
    
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
           metric_params=None, n_jobs=1, n_neighbors=5, p=2,
           weights='uniform')

breast_cancer_prediction_alg(knn,os_smote_X,test_X,os_smote_Y,test_Y)


# <b>Inferences</b><br>
# <b>Confusion Matrix</b><br>
# From the confusion matrix, we observe that the algorithm classifies 56 Malignant (True Positives) and 83 Benign instances (True Negatives) correctly while it classifies 1 benign tumors as malignant (False Positives) and 3 malignant tumor as benign (False Negative).
# 
# <b>Receiver operating characteristic(ROC) Curve</b><br>
# The ROC Area under the curve(AUC) score of 0.973 and the curve, both demonstrate how well the algorithm was able to classify the 
# records.

# # 4. Naive Bayes Classifier

# In[13]:


#Gaussian Naive Bayes.
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB(priors=None)

breast_cancer_prediction_alg(gnb,os_smote_X,test_X,os_smote_Y,test_Y)


# <b>Inferences</b><br>
# <b>Confusion Matrix</b><br>
# From the confusion matrix, we observe that the algorithm classifies 55 Malignant (True Positives) and 82 Benign instances (True Negatives) correctly while it classifies 2 benign tumors as malignant (False Positives) and 4 malignant tumor as benign (False Negative).
# 
# <b>Receiver operating characteristic(ROC) Curve</b><br>
# The ROC Area under the curve(AUC) score of 0.959 and the curve, both demonstrate how well the algorithm was able to classify the 
# records.

# # 5. Support Vector Machine

# In[14]:


#Support Vector Machine
from sklearn.svm import SVC

#Using linear hyper plane
svc_lin  = SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
               decision_function_shape='ovr', degree=3, gamma=1.0, kernel='linear',
               max_iter=-1, probability=True, random_state=None, shrinking=True,
               tol=0.001, verbose=False)

cols = [i for i in dataset.columns if i not in Id_col + target_col]
breast_cancer_prediction(svc_lin,os_smote_X,test_X,os_smote_Y,test_Y,cols,"coefficients")


# <b>Inferences</b><br>
# <b>Confusion Matrix</b><br>
# From the confusion matrix, we observe that the algorithm classifies 53 Malignant (True Positives) and 85 Benign instances (True Negatives) correctly while it classifies 4 benign tumors as malignant (False Positives) and 1 malignant tumor as benign (False Negative).
# 
# <b>Receiver operating characteristic(ROC) Curve</b><br>
# The ROC Area under the curve(AUC) score of 0.959 and the curve, both demonstrate how well the algorithm was able to classify the 
# records.
# 
# <b>Feature Importance</b><br>
# From the Feature Importance graph, we observe that features like Compactness, Symmetry and Fractal Dimension contribute towards the tumor being detected as malignant whereas features such as Radius, Area and Concavity contribute towards the tumor being detected as benign.  

# # 6. LightGBM Classifier

# In[15]:


#LightGBMClassifier
from lightgbm import LGBMClassifier

lgbm_c = LGBMClassifier(boosting_type='gbdt', class_weight=None, colsample_bytree=1.0,
                        learning_rate=0.5, max_depth=7, min_child_samples=20,
                        min_child_weight=0.001, min_split_gain=0.0, n_estimators=100,
                        n_jobs=-1, num_leaves=500, objective='binary', random_state=None,
                        reg_alpha=0.0, reg_lambda=0.0, silent=True, subsample=1.0,
                        subsample_for_bin=200000, subsample_freq=0)

cols = [i for i in dataset.columns if i not in Id_col + target_col]
breast_cancer_prediction(lgbm_c,os_smote_X,test_X,os_smote_Y,test_Y,cols,"features")


# <b>Inferences</b><br>
# <b>Confusion Matrix</b><br>
# From the confusion matrix, we observe that the algorithm classifies 56 Malignant (True Positives) and 84 Benign instances (True Negatives) correctly while it classifies 1 benign tumor as malignant (False Positives) and 2 malignant tumor as benign (False Negative).
# 
# <b>Receiver operating characteristic(ROC) Curve</b><br>
# The ROC Area under the curve(AUC) score of 0.979 and the curve, both demonstrate how well the algorithm was able to classify the 
# records.
# 
# <b>Feature Importance</b><br>
# From the Feature Importance graph, we observe that features like Compactness, Symmetry and Fractal Dimension contribute towards the tumor being detected as malignant whereas features such as Radius, Area and Concavity contribute towards the tumor being detected as benign.  

# # Model Performances

# In[16]:


#Model Performances
from sklearn.metrics import f1_score
from sklearn.metrics import cohen_kappa_score

#gives model report in dataframe
def model_report(model,training_x,testing_x,training_y,testing_y,name) :
    model.fit(training_x,training_y)
    predictions  = model.predict(testing_x)
    accuracy     = accuracy_score(testing_y,predictions)
    recallscore  = recall_score(testing_y,predictions)
    precision    = precision_score(testing_y,predictions)
    roc_auc      = roc_auc_score(testing_y,predictions)
    f1score      = f1_score(testing_y,predictions) 
    kappa_metric = cohen_kappa_score(testing_y,predictions)
    
    df = pd.DataFrame({"Model"           : [name],
                       "Accuracy_score"  : [accuracy],
                       "Recall_score"    : [recallscore],
                       "Precision"       : [precision],
                       "f1_score"        : [f1score],
                       "Area_under_curve": [roc_auc],
                       "Kappa_metric"    : [kappa_metric],
                      })
    return df

#outputs for every model
model1 = model_report(logit,train_X,test_X,train_Y,test_Y,
                      "Logistic Regression  ")
model2 = model_report(logit_smote,os_smote_X,test_X,os_smote_Y,test_Y,
                      "Log.R - SMOTE  ")
model3 = model_report(knn,os_smote_X,test_X,os_smote_Y,test_Y,
                      "KNN Classifier  ")
model4 = model_report(gnb,os_smote_X,test_X,os_smote_Y,test_Y,
                     "Naive Bayes Classifier  ")
model5 = model_report(svc_lin,os_smote_X,test_X,os_smote_Y,test_Y,
                      "SVM Classifier  ")
model6 = model_report(lgbm_c,os_smote_X,test_X,os_smote_Y,test_Y,
                      "LGBM Classifier  ")

#concat all models
model_performances = pd.concat([model1,model2,model3,model4,model5,model6],axis = 0).reset_index()
model_performances = model_performances.drop(columns = "index",axis =1)

table  = ff.create_table(np.round(model_performances,3))
py.iplot(table)


# <b>Metric - Highest - Lowest</b>
# <br>Accuracy - LGBM Classifier - Naive Bayes Classifier
# <br>Recall - LGBM Classifier - Logistic Regression / SVM Classifier
# <br>Precision - Logistic Regression & SVM Classifier - Naive Bayes Classifier
# <br>F1 Score - LGBM Classifier - Naive Bayes Classifier
# <br>Area Under Curve - LGBM Classifier - Logistic Regression / Naive Bayes / SVM Classifier
# <br>Kappa Metric - LGBM Classifier - Naive Bayes Classifier

# In[17]:


#Compare model metrics
def output_tracer(metric,color) :
    tracer = go.Bar(y = model_performances["Model"] ,
                    x = model_performances[metric],
                    orientation = "h", name = metric ,
                    marker = dict(line = dict(width =.7),
                                  color = color)
                   )
    return tracer

layout = go.Layout(dict(title = "Model performances",
                        plot_bgcolor  = "rgb(243,243,243)",
                        paper_bgcolor = "rgb(243,243,243)",
                        xaxis = dict(gridcolor = 'rgb(255, 255, 255)',
                                     title = "metric",
                                     zerolinewidth=1,
                                     ticklen=5,gridwidth=2),
                        yaxis = dict(gridcolor = 'rgb(255, 255, 255)',
                                     zerolinewidth=1,ticklen=5,gridwidth=2),
                        margin = dict(l = 250),
                        height = 780
                       )
                  )

trace1  = output_tracer("Accuracy_score","#FF96A7")
trace2  = output_tracer('Recall_score',"#FFD5DC")
trace3  = output_tracer('Precision',"#8D8284")
trace4  = output_tracer('f1_score',"#663C43")
trace5  = output_tracer('Kappa_metric',"#1A0F11")

data = [trace1,trace2,trace3,trace4,trace5]
fig = go.Figure(data=data,layout=layout)
py.iplot(fig)

