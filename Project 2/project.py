import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from matplotlib import cm as cm
import operator
from warnings import filterwarnings
filterwarnings("ignore")

label=[]
for i in range(62):
    label.append("X"+str(i))
    i +=1

data = pd.read_csv("sonar_all_data_2.csv", names=label)
print(data.shape)
X = data.drop(["X60", "X61"] ,axis=1)
y = data["X60"]
print(data["X61"].head())
print(data["X61"].value_counts())
#sns.countplot(data=data, x="X61")
#plt.show()

sc= StandardScaler()
X_train,X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state = 0)
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

dct={}
accuracy =[]
components = []
comp = range(1,61)
for trial in comp:
    pca = PCA(n_components = trial)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    mlpc = MLPClassifier(hidden_layer_sizes=(1000), activation ="logistic", max_iter=20000, alpha=0.00001, solver="adam", tol=0.0001)
    mlpc.fit(X_train_pca, y_train)
    y_predect = mlpc.predict(X_test_pca)
    print("accuracy : {:.2f} and number of components is : {:.2f}" .format(accuracy_score(y_test, y_predect),trial))
    dct[trial]= accuracy_score(y_test, y_predect)
    accuracy.append(accuracy_score(y_test, y_predect))
    components.append(trial)
    trial +=1
    #cm = confusion_matrix(y_test, y_predect)
    #print(cm)
    #sns.heatmap(cm, center=True)
    #plt.show()


print(max(dct, key=dct.get))
print("%0.2f" %dct[max(dct, key=dct.get)])
plt.plot(components,accuracy)
plt.show()
#print(accuracy)
