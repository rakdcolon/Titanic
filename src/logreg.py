import pandas as pd

df = pd.read_csv('data/processed/train.csv')

X = df.drop(columns=['PassengerId', 'Survived'])
y = df['Survived']

print(X.head())

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=16)

# import the class
from sklearn.linear_model import LogisticRegression

# instantiate the model (using the default parameters)
logreg = LogisticRegression(random_state=16)

# fit the model with data
logreg.fit(X_train, y_train)

y_pred = logreg.predict(X_test)

# import the metrics class
from sklearn import metrics

cnf_matrix = metrics.confusion_matrix(y_test, y_pred)

print(cnf_matrix)
print(metrics.classification_report(y_test, y_pred))