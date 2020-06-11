
#### Import the libraries
import pandas as pd                                                             # Deal with dataframe
import matplotlib.pyplot as plt                                                 # Plot
from sklearn.neural_network import MLPClassifier                                # ML algorithm
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score # confusion matrix, analysis report and score
from sklearn.preprocessing import StandardScaler                                # Scaler
from sklearn.model_selection import train_test_split                            # Train and Split the data
from sklearn.decomposition import PCA                                           # PCA
from warnings import filterwarnings                                             # Deal with warnings
filterwarnings("ignore")

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import minmax_scale
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import PowerTransformer


################################################
# Multiple Perceptron classifier
# Input :
#   - X_train_pca : Reduced Training Features
#   - y_train: Training Set for Dependent Variable
#   - X_train_pca : Reduced Test Features
# Output:
#   - y_predect: Predected Classification
#################################################

def classifier(X_train_pca, y_train, X_test_pca):
    mlpc = MLPClassifier(hidden_layer_sizes=(1000), activation ="logistic", max_iter=2000, alpha=0.00001, solver="adam", tol=0.0001, random_state = 49) # Prepare the model
    mlpc.fit(X_train_pca, y_train)                                              # Fit the model
    y_predect = mlpc.predict(X_test_pca)                                        # Predect the target value
    return y_predect

################################################
# Principle Component Analysis For Feature Reduction
# Input :
#   - trial : Number of components needed
#   - X_train: Training Set for Independent Variables
#   - X_test : Testing Set for Independent Variables
# Output:
#   - y_predect: Predected Classification
#################################################

def feature_reduction(trial, X_train, X_test):
    pca = PCA(n_components = trial)                                             # Prepare the model
    X_train_pca = pca.fit_transform(X_train)                                    # Fit model
    X_test_pca = pca.transform(X_test)                                          # Transform the test features
    return X_train_pca, X_test_pca

#### Main Porgram Starts Here

data = pd.read_csv("sonar_all_data_2.csv")                                      # Read dataset
print(data.shape)                                                               # How many columns and rows in dataset
X = data.iloc[:,:60]                                                            # Features
y = data.iloc[:,60]                                                             # Target
print(y.value_counts())                                                         # Count the number of unique targets


sc= StandardScaler()                                                            # Standard Scaler
X_train,X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state = 0) #Split the dataset into training and testing set as 70% and 30%
X_train = sc.fit_transform(X_train)                                             # Prepare the parameter for the model based on the training feature
X_test = sc.transform(X_test)                                                   # Do the same transformation on the testing features


accuracy =[]                                                                    # Prepare the accuracy
components = []                                                                 # Prepare the number of components
comp = range(1,61)
for trial in comp:                                                              # For each number fo feature find the accuracy of the model
    X_train_pca, X_test_pca = feature_reduction(trial, X_train, X_test)         # Use PCA to reduce the features
    y_predect = classifier(X_train_pca, y_train, X_test_pca)                    # Apply th model and predect the target value
    score = accuracy_score(y_test, y_predect)                                   # Find the score of the predection compared to the actual target
    print("accuracy : {:.2f} and number of components is : {:.2f}" .format(score,trial))    # Print the score and the number of components
    accuracy.append(score)                                                      # Save that accuracy
    components.append(trial)                                                    # Save the number of component
    trial +=1                                                                   # Increase the number of component


max_accuracy = max(accuracy)                                                    # Find the maximum accuracy
max_index = accuracy.index(max(accuracy))                                       # Find the index of maximum accuracy
print("Maximum Accuracy : {:.2f}, and number of components is : {} ".format(max_accuracy,max_index+1)) # Print them

X_train_pca, X_test_pca = feature_reduction(max_index+1, X_train, X_test)       # For the best accuracy perform the algorithm again
y_predect = classifier(X_train_pca, y_train, X_test_pca)                        # For the best accuracy perform the algorithm again

print(classification_report(y_test,y_predect))                                  # For the best accuracy do an analysis report
print("confusion matrix :\n", confusion_matrix(y_test, y_predect))              # For the best accuracy find the confusion matrix

plt.plot(components,accuracy)                                                   # Plot components vs accuracy
plt.title("Accuracy of MLPC using PCA for Feature Reduction")
plt.xlabel("Number of Components")
plt.ylabel("Accuracy")
plt.grid(True)
plt.show()
