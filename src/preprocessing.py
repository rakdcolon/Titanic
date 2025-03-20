import pandas as pd
import re
import numpy as np
from sklearn.decomposition import PCA

def cabin_letter_to_int(letter: str) -> int:
    mapping = {
        'A': 8, 'B': 7, 'C': 6,
        'D': 5, 'E': 4, 'F': 3,
        'G': 2, 'T': 1, 'N': 0,
    }
    return mapping.get(letter, 0)
		
def extract_int_from_string(s: str) -> int:
    digits = re.sub(r'\D', '', s)
    return int(digits) if digits else 0

def preprocess(df, pca_components: int = 0):
    df['Title'] = df['Name'].str.split(',').str[1].str.split().str[0]
    df = pd.get_dummies(df, columns=['Title'], prefix='is')
    df = df.rename(columns={'is_the': 'is_Cts.'})
    df['is_Male'] = (df['Sex'] == 'male')
    df['Embarked'] = df['Embarked'].fillna('N')
    df = pd.get_dummies(df, columns=['Embarked'], prefix='from')
    df['Cabin'] = df['Cabin'].fillna('N').str[0].apply(cabin_letter_to_int)
    df['Ticket'] = df['Ticket'].apply(extract_int_from_string)
    df = df.drop(columns=['Sex', 'Name'])
    df['Age'] = df['Age'].fillna(df['Age'].mean())
    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    cols_to_skip = ['PassengerId', 'Survived']
    cols_to_normalize = [col for col in numeric_cols if col not in cols_to_skip]
    norms = np.linalg.norm(df[cols_to_normalize].values.astype(float), axis=0)
    norms = np.where(norms == 0, 1, norms)
    df[cols_to_normalize] = df[cols_to_normalize] / norms
    if pca_components and pca_components > 0:
        df_normalized = df[cols_to_normalize].fillna(0)
        pca = PCA(n_components=pca_components, whiten=True, random_state=42)
        pca_features = pca.fit_transform(df_normalized)
        df_pca = pd.DataFrame(pca_features, 
                                columns=pd.Index([f'PCA_{i+1}' for i in range(pca_components)]), 
                                index=df.index)
        df = df.drop(columns=cols_to_normalize)
        df = pd.concat([df, df_pca], axis=1)
    
    return df


if __name__ == '__main__':
    df_train = pd.read_csv('data/raw/train.csv')
    df_train_cleaned = preprocess(df_train, pca_components=6)
    df_train_cleaned.to_csv('data/processed/train.csv', index=False)
    
    df_test = pd.read_csv('data/raw/test.csv')
    df_test_cleaned = preprocess(df_test, pca_components=6)
    df_test_cleaned.to_csv('data/processed/test.csv', index=False)
