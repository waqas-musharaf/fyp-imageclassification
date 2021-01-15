import numpy as np
import pandas as pd
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss
from sklearn.metrics import f1_score

data = "C:\\Users\\Waqas\\Desktop\\New Datasets\\banknote_dataset.csv"
names = ['variance', 'skewness', 'kurtosis', 'entropy', 'class']
dataset = pd.read_csv(data, names=names)
array = dataset.values

X = array[:,0:4]
Y = array[:,4]
X_train, X_valid, Y_train, Y_valid = model_selection.train_test_split(X, Y, test_size=0.35, random_state=0)

lr = LogisticRegression(solver='liblinear', multi_class='ovr')
nb = GaussianNB()
svcp = SVC(kernel = 'poly', gamma='auto')
svcs = SVC(kernel = 'sigmoid', gamma='auto')
svcr = SVC(kernel = 'rbf', gamma='auto')
knn5 = KNeighborsClassifier(n_neighbors=5)
knn25 = KNeighborsClassifier(n_neighbors=25)
knn37 = KNeighborsClassifier(n_neighbors=37)
knn49 = KNeighborsClassifier(n_neighbors=49)
knn100 = KNeighborsClassifier(n_neighbors=100)
dt = DecisionTreeClassifier()
	
lr.fit(X_train, Y_train)
nb.fit(X_train, Y_train)
svcp.fit(X_train, Y_train)
svcs.fit(X_train, Y_train)
svcr.fit(X_train, Y_train)
knn5.fit(X_train, Y_train)
knn25.fit(X_train, Y_train)
knn37.fit(X_train, Y_train)
knn49.fit(X_train, Y_train)
knn100.fit(X_train, Y_train)
dt.fit(X_train, Y_train)
	
lr_pred = lr.predict(X_valid)
nb_pred = nb.predict(X_valid)
svcp_pred = svcp.predict(X_valid)
svcs_pred = svcs.predict(X_valid)
svcr_pred = svcr.predict(X_valid)
knn5_pred = knn5.predict(X_valid)
knn25_pred = knn25.predict(X_valid)
knn37_pred = knn37.predict(X_valid)
knn49_pred = knn49.predict(X_valid)
knn100_pred = knn100.predict(X_valid)
dt_pred = dt.predict(X_valid)

print("----------------------------------------------------------")
print("Predictor Accuracy Scores:\n")
print("LR: %s" % accuracy_score(Y_valid, lr_pred))
print("NB: %s" % accuracy_score(Y_valid, nb_pred))
print("SVC (Kernel=Poly): %s" % accuracy_score(Y_valid, svcp_pred))
print("SVC (Kernel=Sigmoid): %s" % accuracy_score(Y_valid, svcs_pred))
print("SVC (Kernel=RBF): %s" % accuracy_score(Y_valid, svcr_pred))
print("KNN (K=5): %s" % accuracy_score(Y_valid, knn5_pred))
print("KNN (K=25): %s" % accuracy_score(Y_valid, knn25_pred))
print("KNN (K=37): %s" % accuracy_score(Y_valid, knn37_pred))
print("KNN (K=49): %s" % accuracy_score(Y_valid, knn49_pred))
print("KNN (K=100): %s" % accuracy_score(Y_valid, knn100_pred))
print("DT: %s" % accuracy_score(Y_valid, dt_pred))
print("----------------------------------------------------------")
print("Logarithmic Loss:\n")
print("LR: %s" % log_loss(Y_valid, lr_pred))
print("NB: %s" % log_loss(Y_valid, nb_pred))
print("SVC (Kernel=Poly): %s" % log_loss(Y_valid, svcp_pred))
print("SVC (Kernel=Sigmoid): %s" % log_loss(Y_valid, svcs_pred))
print("SVC (Kernel=RBF): %s" % log_loss(Y_valid, svcr_pred))
print("KNN (K=5): %s" % log_loss(Y_valid, knn5_pred))
print("KNN (K=25): %s" % log_loss(Y_valid, knn25_pred))
print("KNN (K=37): %s" % log_loss(Y_valid, knn37_pred))
print("KNN (K=49): %s" % log_loss(Y_valid, knn49_pred))
print("KNN (K=100): %s" % log_loss(Y_valid, knn100_pred))
print("DT: %s" % log_loss(Y_valid, dt_pred))
print("----------------------------------------------------------")
print("F1 Score:\n")
print("LR: %s" % f1_score(Y_valid, lr_pred, average='micro'))
print("NB: %s" % f1_score(Y_valid, nb_pred, average='micro'))
print("SVC (Kernel=Poly): %s" % f1_score(Y_valid, svcp_pred, average='micro'))
print("SVC (Kernel=Sigmoid): %s" % f1_score(Y_valid, svcs_pred, average='micro'))
print("SVC (Kernel=RBF): %s" % f1_score(Y_valid, svcr_pred, average='micro'))
print("KNN (K=5): %s" % f1_score(Y_valid, knn5_pred, average='micro'))
print("KNN (K=25): %s" % f1_score(Y_valid, knn25_pred, average='micro'))
print("KNN (K=37): %s" % f1_score(Y_valid, knn37_pred, average='micro'))
print("KNN (K=49): %s" % f1_score(Y_valid, knn49_pred, average='micro'))
print("KNN (K=100): %s" % f1_score(Y_valid, knn100_pred, average='micro'))
print("DT: %s" % f1_score(Y_valid, dt_pred, average='micro'))
print("----------------------------------------------------------")