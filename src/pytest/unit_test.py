"""
Module to implement unit tests
"""
# Function to test
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..')) # add repo path to python path
from src.tools.preprocessing import handle_missing_vals, cap_outliers, handle_duplicates

# Test framework
import unittest
import pandas as pd

path_data = os.path.join("src", "data", "nba_filtered.csv")
target = "TARGET_5Yrs"

class TestHandleMissing(unittest.TestCase):

    def setUp(self):
        self.df = pd.read_csv(path_data)

    def test_missing_3p_percent(self):
        df = self.df.copy()
        df = handle_missing_vals(df)
        self.assertFalse(df.isnull().any().any())
        mask = (df["3PA"] == 0) & (df["3P%"] != 0)
        self.assertTrue(len(df[mask]) == 0)

    def test_non_implemented(self):
        df = self.df.copy()
        df.at[0, "3PA"] = None
        with self.assertRaises(NotImplementedError):
            handle_missing_vals(df)

class TestCapOutliers(unittest.TestCase):

    def setUp(self):
        self.df = pd.read_csv(path_data)

    def test_outlier_capping(self):
        df_capped = cap_outliers(self.df.copy(), "target")

        for col in self.df.columns:
            if col != "target":
                capped_col = f"{col}_capped"
                self.assertIn(capped_col, df_capped.columns)
                self.assertTrue((df_capped[col] <= self.df[col].max()).all())

class TestHandleDuplicates(unittest.TestCase):

    def setUp(self):
        self.df = pd.read_csv(path_data)
        self.df_with_dups = pd.concat([self.df, self.df.iloc[[0]]], ignore_index=True)
        quasi_dup = self.df.iloc[1].copy()
        quasi_dup[target] = quasi_dup[target] + 1
        self.df_with_quasi_dups = pd.concat([self.df, pd.DataFrame([quasi_dup])], ignore_index=True)

    def test_remove_duplicates(self):
        df_cleaned = handle_duplicates(self.df_with_dups.copy(), target)
        self.assertEqual(len(df_cleaned), len(self.df))

if __name__=="__main__":
    unittest.main()


