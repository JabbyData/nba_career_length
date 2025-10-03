"""
Module to parse arguments
"""

import argparse

def init_parser(description: str="default parser") -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=description
    )
    parser.add_argument(
        "--train_data_path",
        type=str,
        required=True,
        help="Path to the trainig dataset."
    )
    parser.add_argument(
        "--target",
        type=str,
        required=True,
        help="Target to predict."
    )
    parser.add_argument(
        "--drop_col",
        type=list[str],
        required=True,
        help="Columns to drop"
    )
    return parser