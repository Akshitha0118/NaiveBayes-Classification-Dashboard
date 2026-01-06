# import the libraries
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score,roc_curve
from matplotlib.colors import ListedColormap


# read the dataset
dataset=pd.read_csv(r'C:\Users\ADMIN\Downloads\3rd - KNN\3rd - KNN\Social_Network_Ads.csv')  


# separate the features and target variable
x= dataset.iloc[:,[2,3]].values
y = dataset.iloc[: , -1].values


# split the dataset into training and testing sets
x_train , x_test , y_train , y_test=train_test_split(x,y,test_size=0.20,random_state=0)


# feature scaling
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)


# train the model
classifier = GaussianNB(var_smoothing=0.15)
classifier.fit(x_train,y_train)


# predict the test set results
y_pred = classifier.predict(x_test)
y_pred


# confusion matrix
cm = confusion_matrix(y_test,y_pred)
cm

# accuracy score
ac = accuracy_score(y_test,y_pred)
ac

# bias-variance 
variance = classifier.score(x_test,y_test)
bias = classifier.score(x_train,y_train)

bias 
variance

# AUC-ROC Curve
y_pred_prob = classifier.predict_proba(x_test)[:,1]

auc_score=roc_auc_score(y_test,y_pred_prob)
auc_score


# Visualising the Training set results
X_set, y_set = x_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Naive bayes (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

# Visualising the Test set results
X_set, y_set = x_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Naive bayes (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()


import pickle

# Save model
with open("Naive_bayes.pkl", "wb") as f:
    pickle.dump(classifier, f)

# Save scaler
with open("scaler.pkl", "wb") as f:
    pickle.dump(sc, f)

print("Model and scaler saved successfully")