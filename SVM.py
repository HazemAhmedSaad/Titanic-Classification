from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, accuracy_score,roc_curve
from sklearn.model_selection import cross_val_score
import pandas as pd
import numpy as np
# visualization
import seaborn as sns
import matplotlib.pyplot as plt



# Data Importing
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
test_survived = pd.read_csv("gender_submission.csv")


print("Shape of train_df :", train_df.shape)
print("Shape of test_df :", test_df.shape)
print("Shape of test_survived :", test_survived.shape)

#  I will add a new column which called train/test in order not to confuse the train and test datasets.
# I will split them later.
train_df["train/test"] = "Train"
train_df.head()

# Adding "Survived" column to test dataset.
test_df = test_df.merge(test_survived)
#test_df.head()
test_df.insert(1, "survived", value=test_df["Survived"])
#test_df.head()
test_df.drop(columns="Survived", inplace=True)
#test_df.head()
test_df.rename(columns={"survived": "Survived"}, inplace=True)
test_df.head()

#  I will add a new column which called train/test in order not to confuse the train and test datasets. I will split them later.
test_df["train/test"] = "Test"
test_df.head()

# Creating general dataframe that includes train and test datasets.
df = pd.concat([train_df, test_df])
print("Shape of df :", df.shape)
df.head()

df.info()

# I want to take Passenger ID as an index.
df.set_index("PassengerId", inplace=True)
df.head()

# Statistical informations about dataset
df.describe()

# Rate of missing values by columns.
df.isna().sum()

# Converting the categorical features to numeric
df["Sex"] = [1 if each == "male" else 0 for each in df.Sex]
df.replace({"C": 0, "Q": 1, "S": 2}, inplace=True)
df.head()

# Dropping the columns we have talked about above.
df.drop(columns=["Name", "SibSp", "Ticket","Fare", "Cabin"], inplace=True)

# Missing values analysis
df.isna().sum()

# Dropping the rows including NaN values.
df.dropna(subset=["Age"], inplace=True)
df.dropna(subset=["Embarked"], inplace=True)

df.head()
df.isna().sum()

categorical_features = ["Pclass", "Sex","Age", "Parch", "Embarked"]
df = pd.get_dummies(df, columns=categorical_features, drop_first=True)

df.head()

df_train = df[df["train/test"] == "Train"]
df_test = df[df["train/test"] == "Test"]


# Determine the train_x, train_y, test_x, test_y

train_x = df_train.drop(columns=["Survived", "train/test"])
train_y = df_train["Survived"]

test_x = df_test.drop(columns=["Survived", "train/test"])
test_y = df_test["Survived"]

print("train_x shape :", train_x.shape)
print("train_y shape :", train_y.shape)
print("test_X shape :", test_x.shape)
print("test_y shape :", test_y.shape)


# Determine the x and y
x = df.drop(columns=["Survived", "train/test"])
y = df["Survived"]


#grid = {"C": np.arange(1, 7, 1), 'gamma': [
#    0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5]}
#svm0 = SVC(random_state=42)
#svm_cv = GridSearchCV(svm0, grid, cv=10)
#svm_cv.fit(x, y)
#print("Best parameters of SVC :", svm_cv.best_params_)

#svm = SVC(C=svm_cv.best_params_["C"], gamma=svm_cv.best_params_[
 #         "gamma"], random_state=42)
#svm.fit(train_x, train_y)
#print("SVC Accuracy :", svm.score(test_x, test_y))
#print(svm.score(train_x, train_y))


clf= SVC(kernel='rbf' ,random_state = 42, C=1.0, gamma='auto')
clf.fit(train_x,train_y)

print(clf.score(train_x, train_y))
print(clf.score(test_x, test_y))

y_pred=clf.predict(test_x)
y_pred

cm = confusion_matrix(test_y, y_pred)
print("confugin matrix is :",cm)

confusion = confusion_matrix(test_y, y_pred)
print(confusion)
# print(confusion.shape)
print("True positive is:", confusion[0,0] )
print("True negative is:", confusion[1,1] )
print("False positive is:", confusion[0,1] )
print("False negative is:",confusion[1,0] )

print(classification_report(test_y, y_pred))

g = sns.FacetGrid(train_df, col='Survived')
g.map(plt.hist, 'Age', bins=20)

# grid = sns.FacetGrid(train, col='Pclass', hue='Survived')
grid = sns.FacetGrid(train_df, col='Survived', row='Pclass', height=2.2, aspect=1.6)
grid.map(plt.hist, 'Age', alpha=.5, bins=20)
grid.add_legend();

grid = sns.FacetGrid(train_df, row='Embarked', col='Survived', height=2.2, aspect=1.6)
grid.map(sns.barplot, 'Sex', 'Fare', alpha=.5, ci=None)
grid.add_legend()

#fpr, tpr, thresholds = roc_curve(test_y, y_pred[:,1])
#y_pred = clf.predict_proba(test_x)
#fpr, tpr, thresholds = roc_curve(test_y, y_pred[:,1])
#plt.plot([0,1],[0,1],"k--")
#plt.plot(fpr, tpr, label = "Logistic Regression")
#plt.xlabel("False Positive Rate")
#plt.ylabel("True Positive Rate")
#plt.title("Logistic Regression ROC Curve")
#plt.show()
#AccScore = accuracy_score(test_x, y_pred, normalize=False)
#print('Accuracy Score is : ', AccScore)
