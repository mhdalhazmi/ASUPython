#################################################################################
# Program that will apply different supervised machine learning algorithems     #
# and find the accuracy of the models and summaries them                        #
#################################################################################

import matplotlib.pyplot as plt                        # so we can add to plot
import numpy as np                                     # needed for arrays
from sklearn.model_selection import train_test_split   # splits database
from sklearn.preprocessing import StandardScaler       # standardize data
from sklearn.linear_model import LogisticRegression    # the algorithm
from sklearn.neighbors import KNeighborsClassifier     # the algorithm
from sklearn.ensemble import RandomForestClassifier    # the algorithm
from sklearn.svm import SVC                            # the algorithm
from sklearn.linear_model import Perceptron            # the algorithm
from sklearn.tree import DecisionTreeClassifier        # the algorithm
from sklearn.metrics import accuracy_score             # check the accuracy of the models
import pandas as pd                                    # needed for datasets

####################################################################################
# Funtion that will implement KNN algorithm and printout the accuracy of the model #
####################################################################################

def knn(X_train, X_test, y_train, y_test, X_train_std, X_test_std,results):

    # create the classifier and fit it
    # using 2 neighbors
    knn = KNeighborsClassifier(n_neighbors=2,p=1,metric='minkowski')
    knn.fit(X_train_std,y_train)                                                # apply the algorithm to training data

    y_pred = knn.predict(X_test_std)                                            # run on the test data

    # combine the train and test data
    X_combined_std = np.vstack((X_train_std, X_test_std))
    y_combined = np.hstack((y_train, y_test))

    # check results on combined data and print all the stats for the model and data
    y_combined_pred = knn.predict(X_combined_std)
    print("########################")
    print("Algorithm: K-Nearest Neighbor")
    print('Number in test ',len(y_test))
    print('Number in combined ',len(y_combined))
    print('Misclassified samples: %d' % (y_test != y_pred).sum())
    print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))
    print('Misclassified combined samples: %d' % (y_combined != y_combined_pred).sum())
    print('Combined Accuracy: %.2f' % accuracy_score(y_combined, y_combined_pred))
    print("######################### \n")

    results.append(['K-Nearest Neighbor',round(accuracy_score(y_combined, y_combined_pred),2)]) #store the accurasy into results

####################################################################################################
# Funtion that will implement Logistic Regression algorithm and printout the accuracy of the model #
####################################################################################################
def lr(X_train, X_test, y_train, y_test, X_train_std, X_test_std,results):

    # create the classifier and fit it
    lr = LogisticRegression(C=10, solver='lbfgs', multi_class='ovr', random_state=0)
    lr.fit(X_train_std, y_train)                                                # apply the algorithm to training data

    # run on the test data
    y_pred = lr.predict(X_test_std)
    # combine the train and test data
    X_combined_std = np.vstack((X_train_std, X_test_std))
    y_combined = np.hstack((y_train, y_test))

    # analyze the combined sets and print all the stats for the model and data
    y_combined_pred = lr.predict(X_combined_std)
    print("########################")
    print("Algorithm: Logistic Regression")
    print('Number in test ',len(y_test))
    print('Number in combined ',len(y_combined))
    print('Misclassified samples: %d' % (y_test != y_pred).sum())
    print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))
    print('Misclassified combined samples: %d' % (y_combined != y_combined_pred).sum())
    print('Combined Accuracy: %.2f' % accuracy_score(y_combined, y_combined_pred))
    print("######################### \n")
    results.append(['Logistic Regression',round(accuracy_score(y_combined, y_combined_pred),2)]) #store the accurasy into results

##############################################################################################
# Funtion that will implement Random Forest algorithm and printout the accuracy of the model #
##############################################################################################
def rf(X_train, X_test, y_train, y_test,results):
    # create the classifier and fit it
    forest = RandomForestClassifier(criterion='entropy', n_estimators=3,
                                    random_state=1, n_jobs=2)
    forest.fit(X_train,y_train)

    y_pred = forest.predict(X_test)         # see how we do on the test data

    # combine the train and test data
    X_combined = np.vstack((X_train, X_test))
    y_combined = np.hstack((y_train, y_test))

    # analyze the combined sets and print all the stats for the model and data
    y_combined_pred = forest.predict(X_combined)

    print("#########################")
    print("Algorithm: Random Forest")
    print('Number in test ',len(y_test))
    print('Misclassified samples: %d' % (y_test != y_pred).sum())
    print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))  # check accuracy
    print('Number in combined ',len(y_combined))
    print('Misclassified combined samples: %d' % (y_combined != y_combined_pred).sum())
    print('Combined Accuracy: %.2f' % accuracy_score(y_combined, y_combined_pred))
    print("######################### \n")

    results.append(['Random Forest',round(accuracy_score(y_combined, y_combined_pred),2)]) #store the accurasy into results

#######################################################################################################
# Funtion that will implement Support Vector Machine algorithm and printout the accuracy of the model #
#######################################################################################################
def svm(X_train, X_test, y_train, y_test, X_train_std, X_test_std,results):
    svm = SVC(kernel='rbf', C=1.0, random_state=0)
    svm.fit(X_train_std, y_train)                      # do the training

    y_pred = svm.predict(X_test_std)                   # work on the test data

    # combine the train and test sets
    X_combined_std = np.vstack((X_train_std, X_test_std))
    y_combined = np.hstack((y_train, y_test))

    # analyze the combined sets and print all the stats for the model and data
    y_combined_pred = svm.predict(X_combined_std)

    print("########################")
    print("Algorithm: Support Vector Machine")
    print('Number in test ',len(y_test))
    print('Number in combined ',len(y_combined))
    print('Misclassified samples: %d' % (y_test != y_pred).sum())
    print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))
    print('Misclassified combined samples: %d' % (y_combined != y_combined_pred).sum())
    print('Combined Accuracy: %.2f' % accuracy_score(y_combined, y_combined_pred))
    print("######################### \n")

    results.append(['Support Vector Machine',round(accuracy_score(y_combined, y_combined_pred),2)])

###########################################################################################
# Funtion that will implement Perceptron algorithm and printout the accuracy of the model #
###########################################################################################
def pr(X_train, X_test, y_train, y_test, X_train_std, X_test_std,results):
    # create the classifier and fit it
    ppn = Perceptron(max_iter=10, tol=1e-3, eta0=0.01, fit_intercept=True, random_state=0, verbose=False)
    ppn.fit(X_train_std, y_train)              # do the training

    y_pred = ppn.predict(X_test_std)           # now try with the test data

    # combine the train and test sets
    X_combined_std = np.vstack((X_train_std, X_test_std))
    y_combined = np.hstack((y_train, y_test))

    # analyze the combined sets and print all the stats for the model and data
    y_combined_pred = ppn.predict(X_combined_std)

    print("########################")
    print("Algorithm: Perceptron")
    print('Number in test ',len(y_test))
    print('Number in combined ',len(y_combined))
    print('Misclassified samples: %d' % (y_test != y_pred).sum())
    print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))
    print('Misclassified combined samples: %d' % (y_combined != y_combined_pred).sum())
    print('Combined Accuracy: %.2f' % accuracy_score(y_combined, y_combined_pred))
    print("######################### \n")

    results.append(['Perceptron',round(accuracy_score(y_combined, y_combined_pred),2)])

##############################################################################################
# Funtion that will implement Decision Tree algorithm and printout the accuracy of the model #
##############################################################################################
def dt(X_train, X_test, y_train, y_test,features_label,results):
    # create the classifier and fit it
    tree = DecisionTreeClassifier(criterion='entropy',max_depth=6 ,random_state=0)
    tree.fit(X_train,y_train)

    y_pred = tree.predict(X_test)         # see how we do on the test data

    # combine the train and test data
    X_combined = np.vstack((X_train, X_test))
    y_combined = np.hstack((y_train, y_test))

    # analyze the combined sets and print all the stats for the model and data
    y_combined_pred = tree.predict(X_combined)
    print("#########################")
    print("Algorithm: Decision Tree")
    print('Number in test ',len(y_test))
    print('Misclassified samples: %d' % (y_test != y_pred).sum())
    print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))
    print('Number in combined ',len(y_combined))
    print('Misclassified combined samples: %d' % (y_combined != y_combined_pred).sum())
    print('Combined Accuracy: %.2f' % accuracy_score(y_combined, y_combined_pred))
    print("######################### \n")

    results.append(['Decision Tree',round(accuracy_score(y_combined, y_combined_pred),2)]) #store the accurasy into results


results = []                                                                    # list to store all of our combined accuracies
features_label = [' variance' ,' skewness' ,'curtosis','entropy', 'class']
data = pd.read_csv('data.txt', names= features_label)                           # read the data to a dataframe

X = data.iloc[:,0:4]                                                            # extract the features
y = data.iloc[:,4]                                                              # extract the classifications

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=0) # split the data into 30% test and 70% train

sc = StandardScaler()                                                           # create the standard scaler
sc.fit(X_train)                                                                 # fit to the training data
X_train_std = sc.transform(X_train)                                             # transform the training data
X_test_std = sc.transform(X_test)                                               # do same transformation on test data


knn(X_train, X_test, y_train, y_test, X_train_std, X_test_std,results)          # call the algorithm
lr(X_train, X_test, y_train, y_test, X_train_std, X_test_std,results)           # call the algorithm
rf(X_train, X_test, y_train, y_test,results)                                    # call the algorithm
svm(X_train, X_test, y_train, y_test, X_train_std, X_test_std,results)          # call the algorithm
pr(X_train, X_test, y_train, y_test, X_train_std, X_test_std,results)           # call the algorithm
dt(X_train, X_test, y_train, y_test,features_label,results)                     # call the algorithm

comparison = pd.DataFrame(results,columns=['Algorithm','Combined Accuracy'],dtype=float) # create dataframe for the summary of the accuracies
print("Accuracy Summary: ")
print (comparison)                                                              # print the summaries
