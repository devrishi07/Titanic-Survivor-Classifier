import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split




def fillna_age(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill missing values in the Age feature using the median grouped by Sex and Pclass.
    """
    df = df.copy()
    df["Age"] = (
        df.groupby(["Sex", "Pclass"])["Age"]
          .transform(lambda x: x.fillna(x.median()))
    )
    return df



def add_age_group(df: pd.DataFrame) -> pd.DataFrame:

    """
    Add a feature Age_group that categorises Age
    """

    bins = [0, 10, 20, 30, 40, 50, 60, 70, 80]
    labels = [
        "0-10", "10-20", "20-30", "30-40",
        "40-50", "50-60", "60-70", "70-80"
    ]

    df = df.copy()
    df["Age_group"] = pd.cut(df["Age"], bins=bins, labels=labels)
    return df



# def select_useful_features(extra_features=None) -> list:

#     features = ['Age', 'Age_group', 'Pclass', 'Sibsp', 'Parch']

#     if extra_features:
#         features.extend(extra_features)
    
#     return features


def split(df: pd.DataFrame, num_features, cat_features):
    """
    Split the dataframe into training and validation set
    """

    X = df[num_features + cat_features]
    y = df['Survived']

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    return (X_train, X_val, y_train, y_val)



    
def build_preprocessor(num_features: list, cat_features: list):
    """
    Create a ColumnTransformer for numeric and categorical features.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", num_features),
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), cat_features),
        ]
    )
    return preprocessor



