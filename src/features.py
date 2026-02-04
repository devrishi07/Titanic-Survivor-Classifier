import pandas as pd

def add_family_size(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add family_size feature as SibSp + Parch + 1
    """

    df = df.copy()
    df['Family_size'] = df['Parch'] + df['SibSp'] + 1

    return df

def add_is_child(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add is_child feature if a passanger's age is less than 10
    """

    df = df.copy()
    df['Is_Child'] = (df['Age'] < 10).astype(int)

    return df

def add_male_child(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add male_child feature if a passanger's age is less than 10 and they are male
    """

    df = df.copy()
    df['Male_Child'] = ((df['Sex'] == 'male') & (df['Age'] < 10)).astype(int)

    return df


def add_rich_male(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add rich_male feature if they has a Pclass of 1 
    """

    df = df.copy()

    df["Rich_Male"] = (
    (df["Sex"] == "male") &
    (df["Pclass"] == 1)
    ).astype(int)

    return df
