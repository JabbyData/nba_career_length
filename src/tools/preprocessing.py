"""
Module implementing functions to preprocess data
"""

# Dependencies
## Data manipulation
import pandas as pd

def handle_missing_vals(df: pd.DataFrame):
    null_columns = df.columns[df.isnull().any()].to_list()
    for null_column in null_columns:
        if null_column == "3P%":
            mask = (df[null_column].isnull()) | (df["3PA"] == 0)  # player without attempted 3P shoot get 3P% = 0%
            df.loc[mask, '3P%'] = df.loc[mask].apply(
                lambda row: (row['3P Made'] / row['3PA'] * 100) if row['3PA'] > 0 else 0, axis=1
            )
            mask = df
        else:
            raise NotImplementedError(f"Missing values for {null_column} is not supported")
    return df

