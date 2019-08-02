import pandas as pd
from sklearn.feature_extraction import DictVectorizer
dfTest = pd.read_csv('CaracteristicasTEST.csv') 

df = pd.read_csv('Caracteristicas.csv')
X = df.drop("Filename",axis=1)
X = X.drop("Clasificacion",axis=1)
y = df["Clasificacion"]

XTest = dfTest.drop("Filename",axis=1)
XTest = XTest.drop("Clasificacion",axis=1)
yTest = dfTest["Clasificacion"]
vec = DictVectorizer()
features = vec.fit_transform(XTest).toarray()


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

from sklearn.ensemble import RandomForestClassifier

random_forest = RandomForestClassifier(n_estimators=30, max_depth=10, random_state=1)

random_forest.fit(X_train, y_train)

from sklearn.metrics import accuracy_score

y_predict = random_forest.predict(X_test)
print(accuracy_score(y_test, y_predict))


yTest_predict = random_forest.predict(XTest)
print("TEST 2",accuracy_score(yTest, yTest_predict))


from sklearn.metrics import confusion_matrix

pd.DataFrame(
    confusion_matrix(y_test, y_predict),
    columns=['Fotos Pornograficas', 'Fotos No Pornograficas'],
    index=['Verdadero porno', 'Verdadero No porno']
)