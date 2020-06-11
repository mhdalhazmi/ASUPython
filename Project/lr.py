# Logistic Regression example of Iris data set
# author: d updated by sdm

from pml53 import plot_decision_regions                # plotting function
import matplotlib.pyplot as plt                        # so we can add to plot
from sklearn import datasets                           # read the data sets
from sklearn.metrics import accuracy_score
import numpy as np                                     # needed for arrays
from sklearn.model_selection import train_test_split   # splits database
from sklearn.preprocessing import StandardScaler       # standardize data
from sklearn.linear_model import LogisticRegression    # the algorithm
import pandas as pd

iris = pd.read_csv('data.txt')                 # load the data set
X = iris.iloc[:,0:4]                      # separate the features we want
print('first 5 observations',X.head(5))
y = iris.iloc[:,4]                             # extract the classifications
print("\n")
print('first 5 observations',y.head(5))

# split the probl0:em into train and test
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=0)

sc = StandardScaler()                       # create the standard scalar
sc.fit(X_train)                             # compute the required transformation
X_train_std = sc.transform(X_train)         # apply to the training data
X_test_std = sc.transform(X_test)           # and SAME transformation of test data!!!

# create logistic regression component.
# C is the inverse of the regularization strength. Smaller -> stronger!
#    C is used to penalize extreme parameter weights.
# solver is the particular algorithm to use
# multi_class determines how loss is computed - ovr -> binary problem for each label

lr = LogisticRegression(C=1, solver='lbfgs', multi_class='ovr', random_state=0)
lr.fit(X_train_std, y_train)                # apply the algorithm to training data

# combine the train and test data
X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))

y_pred = lr.predict(X_test_std)                   # work on the test data

# show the results
print('Number in test ',len(y_test))
print('Misclassified samples: %d' % (y_test != y_pred).sum())
print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))

# combine the train and test sets
X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))

# and analyze the combined sets
print('Number in combined ',len(y_combined))
y_combined_pred = lr.predict(X_combined_std)
print('Misclassified combined samples: %d' % (y_combined != y_combined_pred).sum())
print('Combined Accuracy: %.2f' % accuracy_score(y_combined, y_combined_pred))

# plot the results
plot_decision_regions(X=X_combined_std, y=y_combined, classifier=lr, test_idx=range(105,150))
plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.legend(loc='upper left')
plt.show()
