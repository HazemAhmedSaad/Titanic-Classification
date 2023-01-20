#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import random as rnd

# visualization
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


# In[2]:


import os
print("printing dir")
for dirname, _, filenames in os.walk('E:\Faculty\Selected_1\PROJECT'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[3]:


train = pd.read_csv('E:\\Faculty\\Selected_1\\PROJECT\\train.csv',index_col='PassengerId')
test = pd.read_csv('E:\\Faculty\\Selected_1\\PROJECT\\test.csv',index_col='PassengerId')
sample_sub_test = pd.read_csv('E:\Faculty\\Selected_1\\PROJECT\\gender_submission.csv')
combine = [train, test]


# In[4]:



# Collect information about the data set


# In[5]:


train.shape


# In[6]:


test.shape


# In[7]:


train.head()


# In[8]:


train.tail()


# In[9]:


test.head(5)


# In[10]:


test.tail()


# In[11]:


train.info()


# In[12]:


test.info()


# In[13]:


# check missing values in train data
train.isnull().sum()


# In[14]:


# check missing values in train data
test.isnull().sum()


# In[15]:


print(train.columns.values)


# In[16]:


train.describe()


# In[17]:


train.describe(include=['O'])


# In[18]:


train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# In[19]:


train[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# In[20]:


train[["SibSp", "Survived"]].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# In[21]:


train[["Parch", "Survived"]].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# In[22]:


g = sns.FacetGrid(train, col='Survived')
g.map(plt.hist, 'Age', bins=20)


# In[23]:


# # grid = sns.FacetGrid(train, col='Pclass', hue='Survived')
# grid = sns.FacetGrid(train, col='Survived', row='Pclass', size=2.2, aspect=1.6)
# grid.map(plt.hist, 'Age', alpha=.5, bins=20)
# grid.add_legend();


# In[24]:


# # grid = sns.FacetGrid(train, col='Embarked')
# grid = sns.FacetGrid(train, row='Embarked', size=2.2, aspect=1.6)
# grid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette='deep')
# grid.add_legend()


# In[25]:


grid = sns.FacetGrid(train, row='Embarked', col='Survived', height=2.2, aspect=1.6)
grid.map(sns.barplot, 'Sex', 'Fare', alpha=.5, ci=None)
grid.add_legend()


# In[26]:




# ------------------------------>> Start in "procssing" the data set <<----------------------------


# In[27]:


for dataset in combine:
    dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} ).astype(int)

train.head()


# In[28]:


freq_port = train.Embarked.dropna().mode()[0]
freq_port


# In[29]:


for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)
    
train[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# In[30]:


test['Fare'].fillna(test['Fare'].dropna().median(), inplace=True)
test.head()


# In[31]:


for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

train.head()


# In[32]:


test['Age'].fillna(test['Age'].dropna().median(), inplace=True)
test.head()


# In[33]:


train['Age'].fillna(train['Age'].dropna().median(), inplace=True)
train.head()


# In[34]:


print(train.Cabin)


# In[35]:


from sklearn.preprocessing import LabelEncoder
lbl = LabelEncoder()
train.Cabin = lbl.fit_transform(train.Cabin)
print(train.Cabin)


# In[36]:


from sklearn.preprocessing import LabelEncoder
lbl = LabelEncoder()
test.Cabin = lbl.fit_transform(test.Cabin)
print(test.Cabin)


# In[37]:


freq_port = train.Cabin.dropna().mode()
freq_port


# In[38]:


freq_port = test.Cabin.dropna().mode()
freq_port


# In[39]:


for dataset in combine:
    dataset['Cabin'] = dataset['Cabin'].fillna(freq_port)


# In[40]:


from sklearn.preprocessing import LabelEncoder
lbl = LabelEncoder()
train.Ticket = lbl.fit_transform(train.Ticket)
print(train.Ticket)


# In[41]:


from sklearn.preprocessing import LabelEncoder
lbl = LabelEncoder()
test.Ticket = lbl.fit_transform(test.Ticket)
print(test.Ticket)


# In[42]:


# check missing values in train data
train.isnull().sum()


# In[43]:


# check missing values in train data
test.isnull().sum()


# In[44]:


train.head()


# In[45]:


train = train.drop(['Name'], axis=1)


# In[46]:


test = test.drop(['Name'], axis=1)


# In[47]:


Y_test = sample_sub_test.drop("PassengerId", axis=1)
Y_test.head()


# In[48]:


# ---------------------------> START MODEL <---------------------------------

X_train = train.drop("Survived", axis=1)
Y_train = train["Survived"]
X_test  = test#.drop("PassengerId", axis=1).copy()
Y_test  = sample_sub_test["Survived"]
X_train.shape, Y_train.shape, X_test.shape, Y_test.shape


# In[49]:


# --------------------> Logistic Regression <---------------------

logreg = LogisticRegression(max_iter = 150 ,solver = 'liblinear')
logreg.fit(X_train, Y_train)
Y_pred = logreg.predict(X_test)
# Y_pred = Y_pred
# print(Y_pred)
print(Y_pred.shape)
# print(len(Y_pred))
# print("========================================================================")
# print(Y_test)
print(Y_pred.shape)
# print(len(Y_test))
print("=========================================")

acc_log = round(logreg.score(X_test, Y_test) * 100, 2)
print ("The Accuracy of Logistic Regressioin Before Optimization is:", acc_log)


# In[50]:


print(classification_report(Y_test, Y_pred))


# In[51]:


# ----------------------> confusion matrix <-------------------------
confusion = confusion_matrix(Y_test, Y_pred)
print(confusion)
# print(confusion.shape)
print("True positive is:", confusion[0,0] )
print("True negative is:", confusion[1,1] )
print("False positive is:", confusion[0,1] )
print("False negative is:",confusion[1,0] )


# In[52]:


print("Specificity is:", confusion[1,1] / (confusion[1,1] + confusion[0,1]))


# In[53]:


from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
logreg_pred_proba = logreg.predict_proba(X_test)
fpr, tpr, thresholds = roc_curve(Y_test, logreg_pred_proba[:,1])


# In[54]:


# ----------------------> ROC Curve <-------------------------
plt.plot([0,1],[0,1],"k--")
plt.plot(fpr, tpr, label = "Logistic Regression")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Logistic Regression ROC Curve")
plt.show()


# In[55]:


# --------------------------------->>> After Optimization <<<------------------------------


# In[56]:


# --------------------> Logistic Regression <---------------------

logreg = LogisticRegression(max_iter = 150, solver = 'liblinear', penalty = 'l1', C = 0.11)
logreg.fit(X_train, Y_train)
Y_pred = logreg.predict(X_test)
# Y_pred = Y_pred
# print(Y_pred)
print(Y_pred.shape)
# print(len(Y_pred))
# print("========================================================================")
# print(Y_test)
print(Y_pred.shape)
# print(len(Y_test))
print("=========================================")

acc_log = round(logreg.score(X_test, Y_test) * 100, 2)
print ("The Accuracy of Logistic Regressioin After Optimization is:", acc_log)


# In[57]:


print(classification_report(Y_test, Y_pred))


# In[58]:


from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
logreg_pred_proba = logreg.predict_proba(X_test)
fpr, tpr, thresholds = roc_curve(Y_test, logreg_pred_proba[:,1])


# In[59]:


print("False Positive:")
print( fpr)
print("True Positive:")
print(tpr)
print("Thresholds:")
print(thresholds) 


# In[60]:


# ----------------------> confusion matrix <-------------------------
confusion = confusion_matrix(Y_test, Y_pred)
print(confusion)
# print(confusion.shape)
print("True positive is:", confusion[0,0] )
print("True negative is:", confusion[1,1] )
print("False positive is:", confusion[0,1] )
print("False negative is:",confusion[1,0] )


# In[61]:


# ----------------------> Specificity <-------------------------
print("Specificity is:", confusion[1,1] / (confusion[1,1] + confusion[0,1]))


# In[62]:


# ----------------------> ROC Curve <-------------------------
plt.plot([0,1],[0,1],"k--")
plt.plot(fpr, tpr, label = "Logistic Regression")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Logistic Regression ROC Curve")
plt.show()


# In[75]:


# # Support Vector Machines

# svc = SVC(kernel='rbf',C=0.1,gamma='auto')
# svc.fit(X_train, Y_train)
# Y_pred = svc.predict(X_test)
# acc_svc = round(svc.score(X_test, Y_test) * 100, 2)
# print ("the accuracy of Support Vector Machines is:", acc_svc)


# In[74]:


# confusion = confusion_matrix(Y_test, Y_pred)
# print(confusion)
# # print(confusion.shape)
# print("True positive is:", confusion[0,0] )
# print("True negative is:", confusion[1,1] )
# print("False positive is:", confusion[0,1] )
# print("False negative is:",confusion[1,0] )


# In[73]:


# print(classification_report(Y_test, Y_pred))

