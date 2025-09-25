"""
Module implementing functions to preprocess data
"""

# Dependencies
## Data manipulation
import pandas as pd

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

def check_conform_values(df: pd.DataFrame) -> bool:
    """
    Checks for data consistency in basketball statistics DataFrame.

    This function asserts that for each player:
    - The number of 3-point shots made ("3P Made") does not exceed the number of 3-point attempts ("3PA").
    - The number of field goals made ("FGM") does not exceed the number of field goal attempts ("FGA").
    - The number of free throws made ("FTM") does not exceed the number of free throw attempts ("FTA").

    Raises:
        AssertionError: If any player has more made shots than attempted in any category.

    Args:
        df (pd.DataFrame): DataFrame containing basketball statistics with columns "3P Made", "3PA", "FGM", "FGA", "FTM", "FTA".

    Returns:
        bool: True if all values are consistent.
    """
    assert len(df[df["3P Made"]>df["3PA"]]) == 0, "Incoherent values in 3P : player with more shoot succeeded than attempted"
    assert len(df[df["FGM"]>df["FGA"]]) == 0, "Incoherent values in FG : player with more shoot succeeded than attempted"
    assert len(df[df["FTM"]>df["FTA"]]) == 0, "Incoherent values in FT : player with more shoot succeeded than attempted"
    return True