"""
Module implementing functions to preprocess data
"""

# Dependencies
## Data manipulation
import pandas as pd

## Format
from typing import List

def handle_missing_vals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handles missing values in the provided DataFrame for specific columns.

    Currently, this function supports imputing missing values for the "3P%" column:
    - For rows where "3P%" is missing or the player attempted zero 3-point shots ("3PA" == 0),
        "3P%" is set to 0 if "3PA" is 0, otherwise it is calculated as ("3P Made" / "3PA") * 100.

    Parameters:
            df (pd.DataFrame): The input DataFrame containing basketball statistics.

    Returns:
            pd.DataFrame: The DataFrame with missing values handled for supported columns.

    Raises:
            NotImplementedError: If missing values are found in columns other than "3P%".
    """
    print("Handling missing values ...")
    null_columns = df.columns[df.isnull().any()].to_list()
    for null_column in null_columns:
        if null_column == "3P%":
            mask = (df[null_column].isnull()) | (df["3PA"] == 0)  # player without attempted 3P shoot get 3P% = 0%
            df.loc[mask, '3P%'] = df.loc[mask].apply(
                lambda row: (row['3P Made'] / row['3PA'] * 100) if row['3PA'] > 0 else 0, axis=1
            )
        else:
            raise NotImplementedError(f"Missing values for {null_column} is not supported")
    return df

def cap_outliers(df: pd.DataFrame, target) -> pd.DataFrame:
    numeric_cols = df.select_dtypes(include=[int, float]).columns.difference([target])

    for col in numeric_cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        upper_bound = q3 + 1.5 * iqr
        lower_bound = q1 - 1.5 * iqr
        mask_capped = (df[col] < lower_bound) | (df[col] > upper_bound)
        df[f"{col}_capped"] = mask_capped.astype(int)
        df[col] = df[col].clip(lower=lower_bound,upper=upper_bound)
    
    return df

def handle_duplicates(df: pd.DataFrame, target: str):
    l1 = len(df)
    df = df.drop_duplicates()
    l2 = len(df)
    print("Removing {} duplicates".format(l1 - l2))
    mask_duplicated = df.duplicated(subset=df.columns.difference([target]), keep=False)
    df = df[~mask_duplicated]
    l1 = len(df)
    print("Removing {} quasi-duplicates".format(l2 - l1))
    mask_duplicated = df.duplicated(subset=df.columns.difference([target]), keep=False)
    assert not mask_duplicated.any().any(), "Duplicates remaining"

    return df

def preprocess(df: pd.DataFrame, drop_col: List[str], target: str, mode: str) -> pd.DataFrame:
    # Dropping irrelevant feature
    df = df.drop(columns=drop_col)

    # Handling missing values
    df = handle_missing_vals(df)

    # Remove normal / quasi duplicates
    df = handle_duplicates(df, target)

    if mode == "cap":
        # Cap outliers
        df = cap_outliers(df,target)
    
    return df