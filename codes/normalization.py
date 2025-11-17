"""
This script is an illustrative example based on the preprocessing steps used in the
thesis on detecting fake TikTok accounts. It shows how to:
  - load a (synthetic) CSV file of TikTok account features,
  - apply Min–Max normalization,
  - apply Z-score (Standard) normalization,
  - save the normalized versions to new CSV files.

The script does NOT contain real datasets. Filenames, paths, and column names are
generic and should be adapted to the user's own data.

Usage (example):
    python normalization.py

Expected input:
    data/tiktok_accounts_example.csv
    with at least the following numeric columns:
        - follower_count
        - following_count
        - like_count
        - video_count
        - jaro_similarity
        - nickname_complexity
"""

from pathlib import Path
from typing import List

import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler


# --------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------

BASE_DIR = Path("data")

# Example input file (synthetic / anonymised in the public repo)
INPUT_FILE = BASE_DIR / "tiktok_accounts_example.csv"

# Output files for normalized data
MINMAX_OUTPUT_FILE = BASE_DIR / "tiktok_accounts_minmax.csv"
ZSCORE_OUTPUT_FILE = BASE_DIR / "tiktok_accounts_zscore.csv"

# Columns that will be normalized (adapt to your dataset)
NUMERIC_COLUMNS: List[str] = [
    "follower_count",
    "following_count",
    "like_count",
    "video_count",
    "jaro_similarity",
    "nickname_complexity",
]


# --------------------------------------------------------------------
# Normalization helpers
# --------------------------------------------------------------------

def minmax_normalize(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """
    Apply Min–Max normalization to the selected columns.

    New values are scaled to [0, 1]:
        x_scaled = (x - min) / (max - min)
    """
    scaler = MinMaxScaler()
    df = df.copy()
    df[columns] = scaler.fit_transform(df[columns])
    return df


def zscore_normalize(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """
    Apply Z-score (Standard) normalization to the selected columns.

    New values have mean 0 and variance 1:
        z = (x - mean) / std
    """
    scaler = StandardScaler()
    df = df.copy()
    df[columns] = scaler.fit_transform(df[columns])
    return df


# --------------------------------------------------------------------
# Main script
# --------------------------------------------------------------------

def main() -> None:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_FILE}\n"
            "Create 'data/tiktok_accounts_example.csv' with the required columns "
            f"({', '.join(NUMERIC_COLUMNS)}) or adjust INPUT_FILE and NUMERIC_COLUMNS."
        )

    # Load anonymised / synthetic TikTok account data
    df = pd.read_csv(INPUT_FILE)

    # Ensure columns exist
    missing = [c for c in NUMERIC_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"The following required columns are missing from {INPUT_FILE}: {missing}"
        )

    # --- Min–Max normalization ---
    df_minmax = minmax_normalize(df, NUMERIC_COLUMNS)
    MINMAX_OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df_minmax.to_csv(MINMAX_OUTPUT_FILE, index=False)
    # --- Z-score normalization ---
    df_zscore = zscore_normalize(df, NUMERIC_COLUMNS)
    ZSCORE_OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df_zscore.to_csv(ZSCORE_OUTPUT_FILE, index=False)

if __name__ == "__main__":
    main()
