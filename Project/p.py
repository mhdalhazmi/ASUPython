import matplotlib.pyplot as plt                        # so we can add to plot
from sklearn.metrics import accuracy_score
import numpy as np                                     # needed for arrays
from sklearn.model_selection import train_test_split   # splits database
from sklearn.preprocessing import StandardScaler       # standardize data
from sklearn.linear_model import LogisticRegression    # the algorithm
from sklearn.neighbors import KNeighborsClassifier     # the algorithm
from sklearn.ensemble import RandomForestClassifier    # the algorithm
from sklearn.svm import SVC                             # the algorithm
from sklearn.linear_model import Perceptron            # the algorithm
from sklearn.tree import DecisionTreeClassifier         # the algorithm
from sklearn.tree import export_graphviz                # a cool graph
from sklearn.metrics import accuracy_score             # grade the results
import pandas as pd


def knn(X_train, X_test, y_train, y_test, X_train_std, X_test_std,results):
    # create the classifier and fit it
    # using 10 neighbors
    # since only 2 features, minkowski is same as euclidean distance
    # where p=2 specifies sqrt(sum of squares). (p=1 is Manhattan distance)
    knn = KNeighborsClassifier(n_neighbors=1,p=3,metric='minkowski')
    knn.fit(X_train_std,y_train)

    # run on the test data and print results and check accuracy
    y_pred = knn.predict(X_test_std)
    #print('Number in test ',len(y_test))
    #print('Misclassified samples: %d' % (y_test != y_pred).sum())
    #print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))

    # combine the train and test data
    X_combined_std = np.vstack((X_train_std, X_test_std))
    y_combined = np.hstack((y_train, y_test))
    #print('Number in combined ',len(y_combined))

    # check results on combined data
    y_combined_pred = knn.predict(X_combined_std)
    #print('Misclassified combined samples: %d' % (y_combined != y_combined_pred).sum())
    #print("#########################")
    #print("Algorithm: K-Nearest Neighbor")
    #print('Combined Accuracy: %.2f' % accuracy_score(y_combined, y_combined_pred))
    #print("######################### \n")
    results.append(['K-Nearest Neighbor',round(accuracy_score(y_combined, y_combined_pred),2)])

def lr(X_train, X_test, y_train, y_test, X_train_std, X_test_std,results):

    lr = LogisticRegression(C=1, solver='lbfgs', multi_class='ovr', random_state=0)
    lr.fit(X_train_std, y_train)                # apply the algorithm to training data
    # combine the train and test data
    X_combined_std = np.vstack((X_train_std, X_test_std))
    y_combined = np.hstack((y_train, y_test))
    y_pred = lr.predict(X_test_std)                   # work on the test data
    # show the results
    #print('Number in test ',len(y_test))
    #print('Misclassified samples: %d' % (y_test != y_pred).sum())
    #print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))
    # combine the train and test sets
    X_combined_std = np.vstack((X_train_std, X_test_std))
    y_combined = np.hstack((y_train, y_test))

    # and analyze the combined sets
    #print('Number in combined ',len(y_combined))
    y_combined_pred = lr.predict(X_combined_std)
    #print('Misclassified combined samples: %d' % (y_combined != y_combined_pred).sum())
    #print("#########################")
    #print("Algorithm: Logistic Regression")
    #print('Combined Accuracy: %.2f' % accuracy_score(y_combined, y_combined_pred))
    #print("######################### \n")
    results.append(['Logistic Regression',round(accuracy_score(y_combined, y_combined_pred),2)])

def rf(X_train, X_test, y_train, y_test,results):
    forest = RandomForestClassifier(criterion='entropy', n_estimators=3,
                                    random_state=1, n_jobs=2)
    forest.fit(X_train,y_train)

    y_pred = forest.predict(X_test)         # see how we do on the test data
    #print('Number in test ',len(y_test))
    #print('Misclassified samples: %d' % (y_test != y_pred).sum())

    #print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))  # check accuracy

    # combine the train and test data
    X_combined = np.vstack((X_train, X_test))
    y_combined = np.hstack((y_train, y_test))
    #print('Number in combined ',len(y_combined))

    # see how we do on the combined data
    y_combined_pred = forest.predict(X_combined)
    #print('Misclassified combined samples: %d' % (y_combined != y_combined_pred).sum())
    #print("#########################")
    #print("Algorithm: Random Forest")
    #print('Combined Accuracy: %.2f' % accuracy_score(y_combined, y_combined_pred))
    #print("######################### \n")
    results.append(['Random Forest',round(accuracy_score(y_combined, y_combined_pred),2)])

def svm(X_train, X_test, y_train, y_test, X_train_std, X_test_std,results):
    svm = SVC(kernel='rbf', C=1.0, random_state=0)
    svm.fit(X_train_std, y_train)                      # do the training

    y_pred = svm.predict(X_test_std)                   # work on the test data

    # show the results
    #print('Number in test ',len(y_test))
    #print('Misclassified samples: %d' % (y_test != y_pred).sum())
    #print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))

    # combine the train and test sets
    X_combined_std = np.vstack((X_train_std, X_test_std))
    y_combined = np.hstack((y_train, y_test))

    # and analyze the combined sets
    #print('Number in combined ',len(y_combined))
    y_combined_pred = svm.predict(X_combined_std)
    #print('Misclassified combined samples: %d' % (y_combined != y_combined_pred).sum())
    #print("#########################")
    #print("Algorithm: Support Vector Machine")
    #print('Combined Accuracy: %.2f ' % accuracy_score(y_combined, y_combined_pred))
    #print("######################### \n")
    results.append(['Support Vector Machine',round(accuracy_score(y_combined, y_combined_pred),2)])


def pr(X_train, X_test, y_train, y_test, X_train_std, X_test_std,results):
    ppn = Perceptron(max_iter=10, tol=1e-3, eta0=0.001, fit_intercept=True, random_state=100, verbose=False)
    ppn.fit(X_train_std, y_train)              # do the training

    #print('Number in test ',len(y_test))
    y_pred = ppn.predict(X_test_std)           # now try with the test data

    # Note that this only counts the samples where the predicted value was wrong
    #print('Misclassified samples: %d' % (y_test != y_pred).sum())  # how'd we do?
    #print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))

    # vstack puts first array above the second in a vertical stack
    # hstack puts first array to left of the second in a horizontal stack
    # NOTE the double parens!
    X_combined_std = np.vstack((X_train_std, X_test_std))
    y_combined = np.hstack((y_train, y_test))
    #print('Number in combined ',len(y_combined))

    # we did the stack so we can see how the combination of test and train data did
    y_combined_pred = ppn.predict(X_combined_std)
    #print('Misclassified combined samples: %d' % (y_combined != y_combined_pred).sum())
    #print("#########################")
    #print("Algorithm: Perceptron")
    #print('Combined Accuracy: %.2f ' % accuracy_score(y_combined, y_combined_pred))
    #print("######################### \n")
    results.append(['Perceptron',round(accuracy_score(y_combined, y_combined_pred),2)])

def dt(X_train, X_test, y_train, y_test,features_label,results):
    tree = DecisionTreeClassifier(criterion='entropy',max_depth=5 ,random_state=0)
    tree.fit(X_train,y_train)

    # combine the train and test data
    X_combined = np.vstack((X_train, X_test))
    y_combined = np.hstack((y_train, y_test))

    # see how we do on the combined data
    y_combined_pred = tree.predict(X_combined)
    #print('Misclassified combined samples: %d' % (y_combined != y_combined_pred).sum())
    #print("#########################")
    #print("Algorithm: Decision Tree")
    #print('Combined Accuracy: %.2f' % accuracy_score(y_combined, y_combined_pred))
    #print("######################### \n")
    results.append(['Decision Tree',round(accuracy_score(y_combined, y_combined_pred),2)])


results = []
#features_label = [' variance of Wavelet Transformed image' ,' skewness of Wavelet Transformed image' ,'curtosis of Wavelet Transformed image','entropy of image', 'class']
features_label = [' variance' ,' skewness' ,'curtosis','entropy', 'class']
data = pd.read_csv('data.txt', names= features_label)

X = data.iloc[:,0:4]                      # separate the features we want

y = data.iloc[:,4]                             # extract the classifications


# split the data
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=20)

sc = StandardScaler()                  # create the standard scaler
sc.fit(X_train)                        # fit to the training data
X_train_std = sc.transform(X_train)    # transform the training data
X_test_std = sc.transform(X_test)      # do same transformation on test data

knn(X_train, X_test, y_train, y_test, X_train_std, X_test_std,results)
lr(X_train, X_test, y_train, y_test, X_train_std, X_test_std,results)
rf(X_train, X_test, y_train, y_test,results)
svm(X_train, X_test, y_train, y_test, X_train_std, X_test_std,results)
pr(X_train, X_test, y_train, y_test, X_train_std, X_test_std,results)
dt(X_train, X_test, y_train, y_test,features_label,results)

comparison = pd.DataFrame(results,columns=['Algorithm','Combined Accuracy'],dtype=float)
print (comparison)
