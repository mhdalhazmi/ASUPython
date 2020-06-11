# K-Nearest Neighbors example
# author: Allee, Hartin updated by sdm

from pml53 import plot_decision_regions                # visualize results
import matplotlib.pyplot as plt                        # update the plot
from sklearn import datasets                           # read the data
import numpy as np                                     # for arrays
from sklearn.model_selection import train_test_split   # split the data
from sklearn.preprocessing import StandardScaler       # scale the data
from sklearn.neighbors import KNeighborsClassifier     # the algorithm
from sklearn.metrics import accuracy_score             # grade the results
import pandas as pd

iris = pd.read_csv('data.txt')                 # load the data set
X = iris.iloc[:,0:2]                      # separate the features we want
y = iris.iloc[:,4]                             # extract the classifications

# split the data
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=0)

sc = StandardScaler()                  # create the standard scaler
sc.fit(X_train)                        # fit to the training data
X_train_std = sc.transform(X_train)    # transform the training data
X_test_std = sc.transform(X_test)      # do same transformation on test data

# create the classifier and fit it
# using 10 neighbors
# since only 2 features, minkowski is same as euclidean distance
# where p=2 specifies sqrt(sum of squares). (p=1 is Manhattan distance)
knn = KNeighborsClassifier(n_neighbors=1,p=2,metric='minkowski')
knn.fit(X_train_std,y_train)

# run on the test data and print results and check accuracy
y_pred = knn.predict(X_test_std)
print('Number in test ',len(y_test))
print('Misclassified samples: %d' % (y_test != y_pred).sum())
print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))

# combine the train and test data
X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))
print('Number in combined ',len(y_combined))

# check results on combined data
y_combined_pred = knn.predict(X_combined_std)
print('Misclassified combined samples: %d' % (y_combined != y_combined_pred).sum())
print('Combined Accuracy: %.2f' % accuracy_score(y_combined, y_combined_pred))

# visualize the results
plot_decision_regions(X=X_combined_std, y=y_combined, classifier=knn,
                      test_idx=range(105,150))
plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.legend(loc='upper left')
plt.show()
