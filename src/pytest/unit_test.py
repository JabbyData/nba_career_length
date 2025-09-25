"""
Module to implement unit tests
"""
# Function to test
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..')) # required path to access lib
from src.tools.preprocessing import handle_missing_vals

# Test framework
import unittest
import pandas as pd

path_data = os.path.join("src", "data", "nba_logreg.csv")

class TestHandleMissing(unittest.TestCase):

    def test_missing_3p_percent(self):
        df = pd.read_csv(path_data)
        df = handle_missing_vals(df)
        self.assertFalse(df.isnull().any().any())
        mask = (df["3PA"]==0) & (df["3P%"] != 0)
        self.assertTrue(len(df[mask])==0)

    def test_non_implemented(self):
        df = pd.read_csv(path_data)
        df.at[0, "3PA"] = None
        with self.assertRaises(NotImplementedError):
            handle_missing_vals(df)

if __name__=="__main__":
    unittest.main()

