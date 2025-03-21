import pandas as pd

df = pd.read_csv('data/processed/train.csv')

X = df.drop(columns=['PassengerId', 'Survived'])
y = df['Survived']

print(X.head())

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=16)

from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression(random_state=16)

logreg.fit(X_train, y_train)

y_pred = logreg.predict(X_test)

from sklearn import metrics

cnf_matrix = metrics.confusion_matrix(y_test, y_pred)

print(cnf_matrix)
print(metrics.classification_report(y_test, y_pred))