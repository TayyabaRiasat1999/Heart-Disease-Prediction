#!/usr/bin/env python
# coding: utf-8

# #### Import the required libraries

# In[4]:


import pandas as pd
import numpy as np
import seaborn as sns
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn import tree


# #### Read the data from CSV

# In[5]:


dummy_df= pd.read_csv('Heart_Dataset.csv')
dummy_df.head()


# ## EDA 

# In[6]:


# number of columns and rows
dummy_df.shape


# In[7]:


columns_names = dummy_df.columns.values
columns_names


# In[8]:


# getting information about the dataset
dummy_df.info()


# In[9]:


# finding missing values
dummy_df.isnull().sum()


# In[10]:


# plotting a heatmap for null values in our case since we don't have any null values can't see any thing
sns.heatmap(dummy_df.isnull(),yticklabels =False, cbar=False, cmap='viridis')


# In[ ]:





# ## UniVariant Analysis

# In[11]:


sns.pairplot(data=dummy_df)
plt.show()


# In[13]:


plt.figure(figsize = (8,4))
ax1 = sns.countplot(x='HeartDisease', hue='chestPainType',data=dummy_df, palette = 'Paired')
# lengend_lebels,_ = ax1.get_legend_handles_labels()
# ax1.legend(bbox_to_anchor(1,1))
plt.title('heart diease status in states ofchest Pain' , size=20)
plt.xlabel('heart diease')
plt.ylabel('number of patients')


# In[14]:


plt.figure(figsize = (8,4))
ax1 = sns.countplot(x='HeartDisease', hue='RestingElectrocardiographicResults',data=dummy_df, palette = 'Paired')
# lengend_lebels,_ = ax1.get_legend_handles_labels()
# ax1.legend(bbox_to_anchor(1,1))
plt.title('heart diease status in states of RestingElectrocardiographicResults' , size=20)
plt.xlabel('heart diease')
plt.ylabel('number of patients')


# In[50]:


plt.figure(figsize = (8,4))
ax1 = sns.countplot(x='HeartDisease', hue='exerciseInducedAgngina',data=dummy_df, palette = 'Paired')
# lengend_lebels,_ = ax1.get_legend_handles_labels()
# ax1.legend(bbox_to_anchor(1,1))
plt.title('heart diease status in states of exerciseInducedAgngina' , size=20)
plt.xlabel('heart diease')
plt.ylabel('number of patients')


# In[77]:


f, axes = plt.subplots(1, 2)
sns.boxplot(dummy_df.age, ax=axes[0])



sns.set_style('darkgrid')
sns.histplot(dummy_df.age,bins=20, kde=True, ax=axes[1])
plt.xlabel("Age")
plt.ylabel("Count of Patinets")



plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# ## Data Pre-Processing

# In[4]:


contact_df = dummy_df


# In[5]:


# creating preprocessing function
def preprocessor(df):
    res_df = df.copy()
    le = preprocessing.LabelEncoder()
    
    res_df['chestPainType'] = le.fit_transform(res_df['chestPainType'])
    res_df['RestingElectrocardiographicResults'] = le.fit_transform(res_df['RestingElectrocardiographicResults'])
    res_df['slopePeakExerciseSTsegment'] = le.fit_transform(res_df['slopePeakExerciseSTsegment'])
    res_df['majorColoredVessels'] = le.fit_transform(res_df['majorColoredVessels'])
    res_df['thal'] = le.fit_transform(res_df['thal'])
    res_df['HeartDisease'] = le.fit_transform(res_df['HeartDisease'])
    return res_df
    


# In[6]:


# getting preprocessed data from the above function
encoded_df = preprocessor(contact_df)

# separating independent features
x= encoded_df.drop(['HeartDisease'],axis = 1).values

# separating class labels
y= encoded_df['HeartDisease'].values


# In[7]:


# 0== Absence , 1 == Presence
Yes_Disease = encoded_df['HeartDisease'].value_counts(normalize = True)
print(Yes_Disease)

plt.figure(figsize=(5,4))
plt.title('Heart Disease Status Count')
plt.bar(['Absence','Presence'],encoded_df['HeartDisease'].value_counts(), edgecolor = 'k', width = 0.7)
plt.show()


# In[ ]:





# In[16]:


plt.figure(figsize=(25,15))
ax1 = sns.countplot(x='age', hue='HeartDisease', data=encoded_df )
legend_labels,_ = ax1.get_legend_handles_labels()
ax1.legend(bbox_to_anchor=(1,1))
plt.xlabel('Age')
plt.ylabel('Counts')
plt.title('Disease Status on each age group', size = 20)
# plt.legend(loc='lower right')
plt.show()


# In[17]:


encoded_df.describe()


# In[ ]:





# ## Split the data into Train-Test

# In[57]:


#splitting the dataset into training and test   
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)


# # Build Decision Tree Model

# In[58]:


# decision tree with depth = 2
# applying decision tree algorithm
model_dt_2 = DecisionTreeClassifier(random_state=1, max_depth=2)
model_dt_2.fit(x_train,y_train)

# getting accuray for train dataset
model_dt_2_score_train = model_dt_2.score(x_train,y_train)
print("training score: ", model_dt_2_score_train)

# getting accuray for test dataset
model_dt_2_score_test = model_dt_2.score(x_test,y_test)
print("testing score: ", model_dt_2_score_test)


# In[59]:


# decision tree with depth = 4
# applying decision tree algorithm
model_dt_4 = DecisionTreeClassifier(random_state=1, max_depth=4)
model_dt_4.fit(x_train,y_train)

# getting accuray for train dataset
model_dt_4_score_train = model_dt_4.score(x_train,y_train)
print("training score: ", model_dt_4_score_train)

# getting accuray for test dataset
model_dt_4_score_test = model_dt_4.score(x_test,y_test)
print("testing score: ", model_dt_4_score_test)


# In[60]:


# decision tree with depth = 8
# applying decision tree algorithm
model_dt_8 = DecisionTreeClassifier(random_state=1, max_depth=8, criterion="entropy")
model_dt_8.fit(x_train,y_train)

# getting accuray for train dataset
model_dt_8_score_train = model_dt_8.score(x_train,y_train)
print("training score: ", model_dt_8_score_train)

# getting accuray for test dataset
model_dt_8_score_test = model_dt_8.score(x_test,y_test)
print("testing score: ", model_dt_8_score_test)


# In[61]:


# decision tree with depth = 8
# applying decision tree algorithm
model_dt = DecisionTreeClassifier(random_state=1, max_depth=8, criterion="gini")
model_dt.fit(x_train,y_train)

# getting accuray for train dataset
model_dt_score_train = model_dt.score(x_train,y_train)
print("training score: ", model_dt_score_train)

# getting accuray for test dataset
model_dt_score_test = model_dt.score(x_test,y_test)
print("testing score: ", model_dt_score_test)


# In[62]:


# getting predicted probabilities 
y_pred_dt = model_dt.predict_proba(x_test)[:, 1]


# In[63]:


# plotting decision tree graph
plt.figure("Decision Tree" , figsize=[25,15])
tree.plot_tree(model_dt,fontsize=10,filled=True, feature_names = columns_names, class_names=['Absence','Presence'])
plt.tight_layout()
plt.show()


# ### Performance  Metrics

# In[64]:


# finding FPR => False Positive Rate and TPR => True Positive Rate
fpr_dt, tpr_dt, threshold = roc_curve(y_test, model_dt.predict(x_test))
roc_auc_dt = auc(fpr_dt, tpr_dt)


# In[65]:


# getting predicted values
predictions = model_dt.predict(x_test)


# Model Accuracy
print(model_dt.score(x_test,y_test))
y_actual_result = y_test[0]


# In[66]:


for i in range(len(predictions)):
    if(predictions[i]==1):
        y_actual_result = np.vstack((y_actual_result, y_test[i]))


# ### Recall 

# In[67]:


# recall value => results equal to equal to 1
y_actual_result = y_actual_result.flatten()
count = 0
for result in y_actual_result:
    if(result == 1):
        count =count+1
print("true yes | predicted yes :")
print(count/float(len(y_actual_result)))


# ### Area Under the Curve

# In[68]:


plt.figure(1)

plt.plot(fpr_dt, tpr_dt, color ='green',lw = 2, label='Decision Tree( AUC = %0.2f )'% roc_auc_dt)
plt.plot([0,1],[0,1], color='navy',lw = 2, linestyle='--' )

plt.xlim([0.0,1.0])
plt.ylim([0.0,1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Area Under Curve')
plt.legend(loc='lower right')
plt.show


# ### Confusion Matrix 

# In[69]:


confusion_matrix(y_test, predictions)


# In[70]:


accuracy_score(y_test, predictions)


# In[71]:


cm = confusion_matrix(y_test, predictions, labels=model_dt.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model_dt.classes_)
disp.plot()


# ## Bayes Classification Method

# In[72]:


encoded_df.groupby('HeartDisease').describe()


# In[73]:


from sklearn.naive_bayes import BernoulliNB
bnb = BernoulliNB(force_alpha=True)
bnb.fit(x_train, y_train)

y_pred_bnb = bnb.predict(x_test)

print("Accuracy of the BernoulliNB: ",accuracy_score(y_test,y_pred_bnb))

# getting accuray for train dataset
bnb_score_train = bnb.score(x_train,y_train)
print("training score: ", bnb_score_train)

# getting accuray for test dataset
bnb_score_test = bnb.score(x_test,y_test)
print("testing score: ", bnb_score_test)


# In[74]:


from sklearn.naive_bayes import GaussianNB


gnb = GaussianNB()
gnb.fit(x_train, y_train)
y_pred_gnb = gnb.predict(x_test)
print("Accuracy of the GaussianNB: ",accuracy_score(y_test,y_pred_gnb))

# getting accuray for train dataset
gnb_score_train = gnb.score(x_train,y_train)
print("training score: ", gnb_score_train)

# getting accuray for test dataset
gnb_score_test = gnb.score(x_test,y_test)
print("testing score: ", gnb_score_test)


# In[75]:


# getting predicted probabilities 
y_pred_prob_gnb = gnb.predict_proba(x_test)[:, 1]


# ### Performance Metrics

# In[76]:


# finding FPR => False Positive Rate and TPR => True Positive Rate
fpr_dt, tpr_dt, threshold = roc_curve(y_test, gnb.predict(x_test))
roc_auc_dt = auc(fpr_dt, tpr_dt)


# In[77]:


# getting predicted values
predictions = gnb.predict(x_test)


# Model Accuracy
print(gnb.score(x_test,y_test))
y_actual_result = y_test[0]


# In[78]:


for i in range(len(predictions)):
    if(predictions[i]==1):
        y_actual_result = np.vstack((y_actual_result, y_test[i]))


# ### Recall 

# In[79]:


# recall value => results equal to equal to 1
y_actual_result = y_actual_result.flatten()
count = 0
for result in y_actual_result:
    if(result == 1):
        count =count+1
print("true yes | predicted yes :")
print(count/float(len(y_actual_result)))


# ### Area Under the Curve

# In[80]:


plt.figure(1)

plt.plot(fpr_dt, tpr_dt, color ='green',lw = 2, label='GaussianNB( AUC = %0.2f )'% roc_auc_dt)
plt.plot([0,1],[0,1], color='navy',lw = 2, linestyle='--' )

plt.xlim([0.0,1.0])
plt.ylim([0.0,1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Area Under Curve')
plt.legend(loc='lower right')
plt.show


# ### Confusion Matrix 

# In[81]:


confusion_matrix(y_test, predictions)


# In[82]:


accuracy_score(y_test, predictions)


# In[83]:


cm = confusion_matrix(y_test, predictions, labels=model_dt.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model_dt.classes_)
disp.plot()


# In[ ]:




