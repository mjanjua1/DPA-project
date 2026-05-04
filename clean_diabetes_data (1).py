"""
clean_diabetes_data.py
Cleans the diabetes_012_health_indicators_BRFSS2015.csv dataset.

Steps:
  1. Load the raw CSV
  2. Remove duplicate rows
  3. Cap extreme BMI outliers at 70
  4. Convert float columns to int
  5. Save the cleaned CSV

Usage:
  python clean_diabetes_data.py
  python clean_diabetes_data.py --input my_file.csv --output cleaned.csv
"""

import argparse
import pandas as pd


# ── Config ────────────────────────────────────────────────────────────────────

DEFAULT_INPUT  = "diabetes_012_health_indicators_BRFSS2015.csv"
DEFAULT_OUTPUT = "diabetes_012_health_indicators_BRFSS2015_cleaned.csv"
BMI_CAP        = 70   # BMI values above this are treated as data-entry errors


# ── Cleaning steps ────────────────────────────────────────────────────────────

def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.drop_duplicates()
    removed = before - len(df)
    print(f"  [duplicates]  removed {removed:,} rows  →  {len(df):,} remaining")
    return df


def cap_bmi(df: pd.DataFrame, cap: int = BMI_CAP) -> pd.DataFrame:
    if "BMI" not in df.columns:
        print("  [BMI cap]     column 'BMI' not found, skipping")
        return df
    outliers = (df["BMI"] > cap).sum()
    df = df.copy()
    df.loc[df["BMI"] > cap, "BMI"] = cap
    print(f"  [BMI cap]     capped {outliers:,} values above {cap} to {cap}")
    return df


def fix_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    float_cols = df.select_dtypes(include="float64").columns.tolist()
    df[float_cols] = df[float_cols].astype(int)
    print(f"  [dtypes]      converted {len(float_cols)} float64 columns to int64")
    return df


# ── Main ──────────────────────────────────────────────────────────────────────

def clean(input_path: str, output_path: str) -> pd.DataFrame:
    print(f"\nLoading  →  {input_path}")
    df = pd.read_csv(input_path)
    print(f"  shape: {df.shape[0]:,} rows × {df.shape[1]} columns")

    print("\nCleaning …")
    df = remove_duplicates(df)
    df = cap_bmi(df)
    df = fix_dtypes(df)

    # Sanity checks
    missing = df.isnull().sum().sum()
    dupes   = df.duplicated().sum()
    print(f"\nValidation")
    print(f"  missing values : {missing}")
    print(f"  duplicate rows : {dupes}")

    print(f"\nSaving   →  {output_path}")
    df.to_csv(output_path, index=False)
    print(f"  final shape: {df.shape[0]:,} rows × {df.shape[1]} columns\n")

    return df


def main():
    parser = argparse.ArgumentParser(description="Clean the BRFSS 2015 diabetes dataset.")
    parser.add_argument("--input",  default=DEFAULT_INPUT,  help="Path to raw CSV")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Path for cleaned CSV")
    args = parser.parse_args()

    clean(args.input, args.output)


if __name__ == "__main__":
    main()
