import pandas as pd
import numpy as np
import os

RAW_DATA_DIR = "data/raw"
PROCESSED_DATA_PATH = "data/processed/clean_data.csv"


def load_all_csvs(data_dir):
    """Load and concatenate all CSV files from a directory."""
    all_files = [f for f in os.listdir(data_dir) if f.endswith(".csv")]
    print(f"Found {len(all_files)} files")

    df_list = []
    for file in all_files:
        file_path = os.path.join(data_dir, file)
        print(f"Loading {file_path}")
        df = pd.read_csv(file_path)
        df_list.append(df)

    combined_df = pd.concat(df_list, ignore_index=True)
    return combined_df


def clean_data(df):
    """Clean dataset: handle missing, infinite values."""
    print("Initial shape:", df.shape)

    # Replace infinite values with NaN
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Drop rows with any NaN
    df.dropna(inplace=True)

    print("Shape after cleaning:", df.shape)
    return df


def normalize_labels(df):
    """
    Normalize labels for:
    - Binary classification: Benign vs Attack
    - Multi-class classification: DoS, DDoS, PortScan, Bot, etc.
    """
    # Standardize label column name
    df.columns = df.columns.str.strip()

    if "Label" not in df.columns:
        raise ValueError("Label column not found in dataset")

    # Binary labels
    df["BinaryLabel"] = df["Label"].apply(lambda x: 0 if x == "BENIGN" else 1)

    print("Label distribution (original):")
    print(df["Label"].value_counts())

    print("Binary label distribution:")
    print(df["BinaryLabel"].value_counts())

    return df


def main():
    print("Starting ETL pipeline...")

    # 1. Load
    df = load_all_csvs(RAW_DATA_DIR)

    # 2. Clean
    df = clean_data(df)

    # 3. Normalize labels
    df = normalize_labels(df)

    # 4. Save processed data
    os.makedirs("data/processed", exist_ok=True)
    df.to_csv(PROCESSED_DATA_PATH, index=False)

    print(f"ETL complete. Clean data saved to {PROCESSED_DATA_PATH}")


if __name__ == "__main__":
    main()
