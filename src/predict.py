import pandas as pd
from sklearn.linear_model import LogisticRegression

df_train = pd.read_csv('data/processed/train.csv')
df_test = pd.read_csv('data/processed/test.csv')

X_train = df_train.drop(columns=['PassengerId', 'Survived'])
y_train = df_train['Survived']

X_test = df_test.drop(columns=['PassengerId'])
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

logreg = LogisticRegression(random_state=16, max_iter=1000)
logreg.fit(X_train, y_train)

y_pred = logreg.predict(X_test)

submission = pd.DataFrame({
    'PassengerId': df_test['PassengerId'],
    'Survived': y_pred
})

submission.to_csv('submission.csv', index=False)