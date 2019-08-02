import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import csv
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv("Caracteristicas.csv")
labels = np.asarray(df.Clasificacion)
le = LabelEncoder()
le.fit(labels)
# apply encoding to labels
labels = le.transform(labels)
df_selected = df.drop(['Filename', 'Clasificacion',"HMean", "SMean","Vmean","MeanGray","Eccentricity","Ellipcity","Orientation","Hue std","CentroideY","Perimeter",
"CentroideX","Areacara","Amount of touching pixels","Number of touched corners"], axis=1)

df_f = df_selected.to_dict(orient='records')

vec = DictVectorizer()
features = vec.fit_transform(df_f).toarray()
features_train, features_test, labels_train, labels_test = train_test_split(
    features, labels, 
    test_size=0.25, random_state=42)

#clf = RandomForestClassifier()
clf = SVC(C=100, cache_size=200, class_weight='balanced', coef0=0.0,
  decision_function_shape='ovr', degree=3, gamma=0.00001, kernel='linear',
  max_iter=-1, probability=False, random_state=None, shrinking=True,
  tol=0.001, verbose=False)
clf.fit(features_train,labels_train)
acc_test = clf.score(features_test,labels_test)
print("TEST ACCURACY WITH SVC(Support Vector Classifier):",acc_test)
clf = RandomForestClassifier()
clf.fit(features_train,labels_train)
acc_test = clf.score(features_test,labels_test)
print("TEST ACCURACY WITH Random Forest Classifier:",acc_test)
clf = GaussianNB()
clf.fit(features_train,labels_train)
acc_test = clf.score(features_test,labels_test)
print("TEST ACCURACY WITH Naive Bayes:",acc_test)