#!/usr/bin/env python
# coding: utf-8

# # Heart Disease Prediction

# ## 1.Import Essential Libraries

# In[1]:


# Import the needed lib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
np.random.seed(42)
import warnings
warnings.filterwarnings("ignore")


# ## 2.Importing our dataset

# In[2]:


## Load the dataset using read_csv
heart_disease=pd.read_csv("heart-disease.csv")


# In[3]:


## See wheather the dataset has  been loaded
heart_disease #Note: If the data is not in numeric form convert them to number before using it


# In[4]:


type(heart_disease) # Veriying it as  dataframe object in pandas


# In[5]:


heart_disease.shape # shape of dataset


# In[6]:


## For our convinient we are going to view first 5 rows of our dataset
heart_disease.head(5)


# In[7]:


heart_disease.describe() ## This will give us the statistical summary


# In[8]:


## check wheather our data has missing values
heart_disease.isna().sum()


# In[9]:


heart_disease.info() # information about heart_disease


# # 3.Checking the correlation

# In[10]:


heart_disease.corr()["target"].abs().sort_values(ascending=False)


# In[11]:


# "fbs" is less correlated


# In[12]:


correlation=heart_disease.corr()
correlation


# In[13]:


import seaborn as sns
sns.heatmap(correlation)


# In[14]:


## Customizing Correlation Matrix
fig,ax=plt.subplots(figsize=(10,10))
ax=sns.heatmap(correlation,
               annot=True,
                 linewidths=0.5,
                 fmt=".2f",
                 cmap="RdBu_r")


# In[16]:


heart_disease.hist(figsize=(10,10));


# # 4. Exploratory Data Analysis

# ## First Analysing the target value

# In[17]:


## Before analysing it split the dataset into x and y
x=heart_disease.drop("target",axis=1)
y=heart_disease["target"]


# In[18]:


x


# In[19]:


y


# In[20]:


# now we can analyse the data
import seaborn as sns
sns.countplot(y)
target_temp=heart_disease.target.value_counts() # to find the percentage
print(target_temp)


# In[21]:


print(f"Percentage of patient without heart disease:{target_temp[0]*100/303:.2f}%")
print(f"Percentage of patient with heart disease:{target_temp[1]*100/303:.2f}%")


# ### Analysing Each Feature Separately

# In[22]:


#Analysis of sex
heart_disease["sex"].unique()


# In[23]:


sns.barplot(heart_disease["sex"],y)


# #### We notice, that female are more likely to have heart problems than male

# In[24]:


## Analysing Chest Pain 
heart_disease["cp"].unique()


# In[25]:


sns.barplot(heart_disease["cp"],y)


# #### We notice, that chest pain of '0', i.e. the ones with typical angina are much less likely to have heart problems
# 

# In[26]:


# Analysing FBS
heart_disease["fbs"].unique()


# In[27]:


sns.barplot(heart_disease["fbs"],y)


# #### Nothing Extraordinary

# In[28]:


## Analysing CA
heart_disease["ca"].unique()


# In[29]:


sns.barplot(heart_disease["ca"],y)


# In[30]:


# Analysing  thal
heart_disease["thal"].unique()


# In[31]:


sns.barplot(heart_disease["thal"],y)


# In[32]:


#Analysing Restecg
heart_disease["restecg"].unique()


# In[33]:


sns.barplot(heart_disease["restecg"],y)


# 
# #### We realize that people with restecg '1' and '0' are much more likely to have a heart disease than with restecg '2

# In[34]:


##Analysing exang
heart_disease["exang"].unique()


# In[35]:


sns.barplot(heart_disease["exang"],y)


# People with exang=1 i.e. Exercise induced angina are much less likely to have heart problems

# In[36]:


## Analysing slope
heart_disease["slope"].unique()


# In[37]:


sns.barplot(heart_disease["slope"],y)


# We observe that people with slope 2  causes heart pain more than slope 1 and 2 

# # 5. Train Test Split

# In[40]:


## Spliting the data into training and testing sets
from sklearn.model_selection import train_test_split,GridSearchCV
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)


# In[41]:


x_train


# In[42]:


x_test


# In[43]:


y_train


# In[44]:


y_test


# In[45]:


x_train.shape,y_train.shape,x_test.shape,y_test.shape


# # 6. Model Fitting

# In[46]:


from sklearn.metrics import accuracy_score


# ## Logistic Regression

# In[47]:


from sklearn.linear_model import LogisticRegression


lr=LogisticRegression()

lr.fit(x_train,y_train)

y_preds_lr=lr.predict(x_test)


# In[48]:


y_preds_lr.shape


# In[49]:


y_preds_lr


# In[50]:


score_lr = round(accuracy_score(y_preds_lr,y_test)*100,2)

print("The accuracy score achieved using Logistic Regression is: "+str(score_lr)+" %")


# #### The accuracy is not that good so we can either tune our model or use another algorithm

# ## SVM

# In[51]:


from sklearn import svm

sv=svm.SVC(kernel="linear")

sv.fit(x_train,y_train)

y_preds_svm=sv.predict(x_test)


# In[52]:


y_preds_svm.shape


# In[53]:


score_svm = round(accuracy_score(y_preds_svm,y_test)*100,2)

print("The accuracy score achieved using Linear SVM is: "+str(score_svm)+" %")


# #### Not satisfied with this accuracy 

# ## Random Forest Classifier

# In[54]:


from sklearn.ensemble import RandomForestClassifier
np.random.seed(42)
clf=RandomForestClassifier()
clf.fit(x_train,y_train)


# In[55]:


y_preds=clf.predict(x_test)
clf.score(x_test,y_test)


# In[56]:


y_preds.shape


# In[57]:


score_clf = round(accuracy_score(y_preds,y_test)*100,2)

print("The accuracy score achieved using Random forest  is: "+str(score_svm)+" %")


# ### Evaluating RandomForestClassifier

# In[58]:


from sklearn.metrics import confusion_matrix

y_preds=clf.predict(x_test)

confusion_matrix(y_test,y_preds)

pd.crosstab(y_test,
           y_preds,
           rownames=["Actual Labels"],
           colnames=["Predicted Labels"])


# In[62]:


# Customizing the confusion matrix
import matplotlib.pyplot as plt
conf=confusion_matrix(y_test,y_preds)

def plot_conf_mat(conf_mat):
    """
    Plot a confusion matrix using seaborn heatmap().

    """
    fig,ax=plt.subplots(figsize=(3,3))
    ax=sns.heatmap(conf,
                  annot=True, # Annotate the boxes with conf_mat info
                  cbar=False)
    plt.xlabel("True Label")
    plt.ylabel("Predicted Label");
    
plot_conf_mat(conf)


# In[63]:


from sklearn.model_selection import cross_val_score

np.random.seed(42)
cv_acc=cross_val_score(clf,x,y,scoring="accuracy")
print(f"The cross-validate accuracy is: {np.mean(cv_acc)*100:.2f}%")


# In[64]:


from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score


# In[65]:



# Evaluation
print("Classifier  Metrics on the test set")
print(f"Accuracy:{accuracy_score(y_test,y_preds)*100:.2f}")
print(f"Precision:{precision_score(y_test,y_preds)}")
print(f"Recall:{recall_score(y_test,y_preds)}")
print(f"F1 score:{f1_score(y_test,y_preds)}")


# In[67]:


from sklearn.model_selection import RandomizedSearchCV

grid={"n_estimators":[10,100,200,500,1000,1200],
     "max_depth":[ None,5,10,20,30],
     "max_features":["auto","sqrt"],
     "min_samples_split":[2,4,6],
     "min_samples_leaf":[1,2,4]}

np.random.seed(43)




clf=RandomForestClassifier(n_jobs=1)

#Set up Randomized
rs_clf=RandomizedSearchCV(estimator=clf,
                         param_distributions=grid,
                         n_iter=10,
                         cv=5,
                         verbose=2)

rs_clf.fit(x_train,y_train)


# In[69]:


rs_clf.best_params_


# In[70]:


def evaluate_preds(y_true,y_preds):
    """
    Performs evaluation comparision on y_true labels vs y_preds labels.
    on a classification model.
    """
    accuracy=accuracy_score(y_true,y_preds)
    precision=precision_score(y_true,y_preds)
    recall=recall_score(y_true,y_preds)
    f1=f1_score(y_true,y_preds)
    metric_dict={"accuracy":round(accuracy,2),
               "precision":round(precision,2),
               "recall":round(recall,2),
               "f1":round(f1,2)}
    print(f"Accuracy:{accuracy*100:2f}%")
    print(f"Precision:{precision:2f}")
    print(f"Recall:{recall:2f}")
    print(f"F1:{f1:2f}")
    
    return metric_dict


# In[71]:


# Make baseline predictions
y_preds_1=rs_clf.predict(x_test)

#Evaluation 
rs_metrics=evaluate_preds(y_test,y_preds_1)
rs_metrics


# In[72]:


grid1={"n_estimators":[10,200,500],
     "max_depth":[ None,20],
     "max_features":["auto","sqrt"],
     "min_samples_split":[2],
     "min_samples_leaf":[4]}


# In[73]:


from sklearn.model_selection import GridSearchCV,train_test_split


np.random.seed(43)


clf=RandomForestClassifier(n_jobs=1)

#Set up GridSearch
gs_clf=GridSearchCV(estimator=clf,
                         param_grid=grid1,
                         cv=5,
                         verbose=2)

gs_clf.fit(x_train,y_train)


# In[74]:


gs_clf.best_params_


# In[75]:


gs_y_preds=gs_clf.predict(x_test)

gs_metrics=evaluate_preds(y_test,gs_y_preds)


# In[76]:


score_gs_clf = round(accuracy_score(gs_y_preds,y_test)*100,2)

print("The accuracy score achieved using Grid Search Random forest  is: "+str(score_svm)+" %")


# In[77]:


scores = [score_lr,score_svm,score_clf,score_gs_clf]
algorithms = ["Logistic Regression","Support Vector Machine","Random Forest","GrisSearch Random Forest"]   

for i in range(len(algorithms)):
    print("The accuracy score achieved using "+algorithms[i]+" is: "+str(scores[i])+" %")


# In[78]:


sns.set(rc={'figure.figsize':(10,10)})
plt.xlabel("Algorithms")
plt.ylabel("Accuracy score")

sns.barplot(algorithms,scores)


# ## Logistic Regression has a good result

# In[ ]:




