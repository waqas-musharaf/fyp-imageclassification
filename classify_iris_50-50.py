import numpy as np
import pandas as pd
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss
from sklearn.metrics import f1_score

data = "C:\\Users\\Waqas\\Desktop\\New Datasets\\iris_dataset.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pd.read_csv(data, names=names)
array = dataset.values

X = array[:,0:4]
Y = array[:,4]
X_train, X_valid, Y_train, Y_valid = model_selection.train_test_split(X, Y, test_size=0.50, random_state=0)

mlr = LogisticRegression(solver='newton-cg', multi_class='multinomial')
mnb = MultinomialNB()
svcp = SVC(kernel = 'poly', gamma='auto', probability=True)
svcs = SVC(kernel = 'sigmoid', gamma='auto', probability=True)
svcr = SVC(kernel = 'rbf', gamma='auto', probability=True)
knn5 = KNeighborsClassifier(n_neighbors=5)
knn8 = KNeighborsClassifier(n_neighbors=8)
knn12 = KNeighborsClassifier(n_neighbors=12)
knn16 = KNeighborsClassifier(n_neighbors=16)
knn30 = KNeighborsClassifier(n_neighbors=30)
dt = DecisionTreeClassifier()
	
mlr.fit(X_train, Y_train)
mnb.fit(X_train, Y_train)
svcp.fit(X_train, Y_train)
svcs.fit(X_train, Y_train)
svcr.fit(X_train, Y_train)
knn5.fit(X_train, Y_train)
knn8.fit(X_train, Y_train)
knn12.fit(X_train, Y_train)
knn16.fit(X_train, Y_train)
knn30.fit(X_train, Y_train)
dt.fit(X_train, Y_train)
	
mlr_pred = mlr.predict(X_valid)
mnb_pred = mnb.predict(X_valid)
svcp_pred = svcp.predict(X_valid)
svcs_pred = svcs.predict(X_valid)
svcr_pred = svcr.predict(X_valid)
knn5_pred = knn5.predict(X_valid)
knn8_pred = knn8.predict(X_valid)
knn12_pred = knn12.predict(X_valid)
knn16_pred = knn16.predict(X_valid)
knn30_pred = knn30.predict(X_valid)
dt_pred = dt.predict(X_valid)

print("----------------------------------------------------------")
print("Predictor Accuracy Scores:\n")
print("MLR: %s" % accuracy_score(Y_valid, mlr_pred))
print("MNB: %s" % accuracy_score(Y_valid, mnb_pred))
print("SVC (Kernel=Poly): %s" % accuracy_score(Y_valid, svcp_pred))
print("SVC (Kernel=Sigmoid): %s" % accuracy_score(Y_valid, svcs_pred))
print("SVC (Kernel=RBF): %s" % accuracy_score(Y_valid, svcr_pred))
print("KNN (K=5): %s" % accuracy_score(Y_valid, knn5_pred))
print("KNN (K=8): %s" % accuracy_score(Y_valid, knn8_pred))
print("KNN (K=12): %s" % accuracy_score(Y_valid, knn12_pred))
print("KNN (K=16): %s" % accuracy_score(Y_valid, knn16_pred))
print("KNN (K=30): %s" % accuracy_score(Y_valid, knn30_pred))
print("DT: %s" % accuracy_score(Y_valid, dt_pred))
print("----------------------------------------------------------")
print("Logarithmic Loss:\n")
print("MLR: %s" % log_loss(Y_valid, mlr.predict_proba(X_valid)))
print("MNB: %s" % log_loss(Y_valid, mnb.predict_proba(X_valid)))
print("SVC (Kernel=Poly): %s" % log_loss(Y_valid, svcp.predict_proba(X_valid)))
print("SVC (Kernel=Sigmoid): %s" % log_loss(Y_valid, svcs.predict_proba(X_valid)))
print("SVC (Kernel=RBF): %s" % log_loss(Y_valid, svcr.predict_proba(X_valid)))
print("KNN (K=5): %s" % log_loss(Y_valid, knn5.predict_proba(X_valid)))
print("KNN (K=8): %s" % log_loss(Y_valid, knn8.predict_proba(X_valid)))
print("KNN (K=12): %s" % log_loss(Y_valid, knn12.predict_proba(X_valid)))
print("KNN (K=16): %s" % log_loss(Y_valid, knn16.predict_proba(X_valid)))
print("KNN (K=30): %s" % log_loss(Y_valid, knn30.predict_proba(X_valid)))
print("DT: %s" % log_loss(Y_valid, dt.predict_proba(X_valid)))
print("----------------------------------------------------------")
print("F1 Score:\n")
print("MLR: %s" % f1_score(Y_valid, mlr_pred, average='micro'))
print("MNB: %s" % f1_score(Y_valid, mnb_pred, average='micro'))
print("SVC (Kernel=Poly): %s" % f1_score(Y_valid, svcp_pred, average='micro'))
print("SVC (Kernel=Sigmoid): %s" % f1_score(Y_valid, svcs_pred, average='micro'))
print("SVC (Kernel=RBF): %s" % f1_score(Y_valid, svcr_pred, average='micro'))
print("KNN (K=5): %s" % f1_score(Y_valid, knn5_pred, average='micro'))
print("KNN (K=8): %s" % f1_score(Y_valid, knn8_pred, average='micro'))
print("KNN (K=12): %s" % f1_score(Y_valid, knn12_pred, average='micro'))
print("KNN (K=16): %s" % f1_score(Y_valid, knn16_pred, average='micro'))
print("KNN (K=30): %s" % f1_score(Y_valid, knn30_pred, average='micro'))
print("DT: %s" % f1_score(Y_valid, dt_pred, average='micro'))
print("----------------------------------------------------------")