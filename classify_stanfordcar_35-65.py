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

classes = "C:\\Users\\Waqas\\Desktop\\New Datasets\\stanfordcar_dataset\\names.csv"
labels = pd.read_csv(classes)
data = "C:\\Users\\Waqas\\Desktop\\New Datasets\\stanfordcar_dataset\\anno_all_vl.csv"
names = "'filename', 'feature1', 'feature2', 'feature3', 'feature4', 'class'"
dataset = pd.read_csv(data, names=names)
array = dataset.values

X = array[:,1:5]
Y = array[:,5]
X_train, X_valid, Y_train, Y_valid = model_selection.train_test_split(X, Y, test_size=0.35, random_state=0)

## Training and fitting models:

# MLR:
mlr = LogisticRegression(solver='newton-cg', multi_class='multinomial')
mlr.fit(X_train, Y_train)
mlr_pred = mlr.predict(X_valid)

# MNB:
mnb = MultinomialNB()
mnb.fit(X_train, Y_train)
mnb_pred = mnb.predict(X_valid)

# SVCP (Dropped due to indefinite execution time):
# svcp = SVC(kernel = 'poly', gamma='auto')
# svcp.fit(X_train, Y_train)
# svcp_pred = svcp.predict(X_valid)

# SVCS:
svcs = SVC(kernel = 'sigmoid', gamma='auto')
svcs.fit(X_train, Y_train)
svcs_pred = svcs.predict(X_valid)

# SVCR:
svcr = SVC(kernel = 'rbf', gamma='auto')
svcr.fit(X_train, Y_train)
svcr_pred = svcr.predict(X_valid)

# KNN5:
knn5 = KNeighborsClassifier(n_neighbors=5)
knn5.fit(X_train, Y_train)
knn5_pred = knn5.predict(X_valid)

# KNN46:
knn46 = KNeighborsClassifier(n_neighbors=46)
knn46.fit(X_train, Y_train)
knn46_pred = knn46.predict(X_valid)

# KNN68:
knn68 = KNeighborsClassifier(n_neighbors=68)
knn68.fit(X_train, Y_train)
knn68_pred = knn68.predict(X_valid)

# KNN91:
knn91 = KNeighborsClassifier(n_neighbors=91)
knn91.fit(X_train, Y_train)
knn91_pred = knn91.predict(X_valid)

# KNN182:
knn182 = KNeighborsClassifier(n_neighbors=182)
knn182.fit(X_train, Y_train)
knn182_pred = knn182.predict(X_valid)

# DT:
dt = DecisionTreeClassifier()
dt.fit(X_train, Y_train)
dt_pred = dt.predict(X_valid)

print("----------------------------------------------------------")
print("Predictor Accuracy Scores:\n")
print("MLR: %s" % accuracy_score(Y_valid, mlr_pred))
print("MNB: %s" % accuracy_score(Y_valid, mnb_pred))
#print("SVC (Kernel=Poly): %s" % accuracy_score(Y_valid, svcp_pred))
print("SVC (Kernel=Sigmoid): %s" % accuracy_score(Y_valid, svcs_pred))
print("SVC (Kernel=RBF): %s" % accuracy_score(Y_valid, svcr_pred))
print("KNN (K=5): %s" % accuracy_score(Y_valid, knn5_pred))
print("KNN (K=46): %s" % accuracy_score(Y_valid, knn46_pred))
print("KNN (K=68): %s" % accuracy_score(Y_valid, knn68_pred))
print("KNN (K=91): %s" % accuracy_score(Y_valid, knn91_pred))
print("KNN (K=182): %s" % accuracy_score(Y_valid, knn182_pred))
print("DT: %s" % accuracy_score(Y_valid, dt_pred))
print("----------------------------------------------------------")
# Logarithmic Loss
print("Logarithmic Loss:\n")
print("Due to a known bug with metrics.log_loss and model_selection.train_test_split, where train/test datasets " +
	"are split with uneven number of classes, resulting in logarithmic loss being un-calculatable, this metric " +
	"has been dropped for this dataset.")
#print("MLR: %s" % log_loss(Y_valid, mlr.predict_proba(X_valid)))
#print("MNB: %s" % log_loss(Y_valid, mnb.predict_proba(X_valid)))
#print("SVC (Kernel=Poly): %s" % log_loss(Y_valid, svcp.predict_proba(X_valid)))
#print("SVC (Kernel=Sigmoid): %s" % log_loss(Y_valid, svcs.predict_proba(X_valid)))
#print("SVC (Kernel=RBF): %s" % log_loss(Y_valid, svcr.predict_proba(X_valid)))
#print("KNN (K=5): %s" % log_loss(Y_valid, knn5.predict_proba(X_valid)))
#print("KNN (K=46): %s" % log_loss(Y_valid, knn46.predict_proba(X_valid)))
#print("KNN (K=68): %s" % log_loss(Y_valid, knn68.predict_proba(X_valid)))
#print("KNN (K=91): %s" % log_loss(Y_valid, knn91.predict_proba(X_valid)))
#print("KNN (K=182): %s" % log_loss(Y_valid, knn182.predict_proba(X_valid)))
#print("DT: %s" % log_loss(Y_valid, dt.predict_proba(X_valid)))
print("----------------------------------------------------------")
print("F1 Score:\n")
print("MLR: %s" % f1_score(Y_valid, mlr_pred, average='micro'))
print("MNB: %s" % f1_score(Y_valid, mnb_pred, average='micro'))
#print("SVC (Kernel=Poly): %s" % f1_score(Y_valid, svcp_pred, average='micro'))
print("SVC (Kernel=Sigmoid): %s" % f1_score(Y_valid, svcs_pred, average='micro'))
print("SVC (Kernel=RBF): %s" % f1_score(Y_valid, svcr_pred, average='micro'))
print("KNN (K=5): %s" % f1_score(Y_valid, knn5_pred, average='micro'))
print("KNN (K=46): %s" % f1_score(Y_valid, knn46_pred, average='micro'))
print("KNN (K=68): %s" % f1_score(Y_valid, knn68_pred, average='micro'))
print("KNN (K=91): %s" % f1_score(Y_valid, knn91_pred, average='micro'))
print("KNN (K=182): %s" % f1_score(Y_valid, knn182_pred, average='micro'))
print("DT: %s" % f1_score(Y_valid, dt_pred, average='micro'))
print("----------------------------------------------------------")