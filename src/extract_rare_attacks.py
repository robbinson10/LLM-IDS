import pandas as pd
import os

DATA_PATH = "data/processed/clean_data.csv"
OUTPUT_PATH = "data/genai"

# Rare attack labels from CIC-IDS2017
RARE_ATTACKS = [
    "Web Attack – XSS",
    "Web Attack – Sql Injection",
    "Infiltration",
    "Heartbleed"
]

def main():
    print("Loading processed dataset...")
    df = pd.read_csv(DATA_PATH)
    df.columns = df.columns.str.strip()

    print("Original dataset shape:", df.shape)

    # Filter only rare attack samples
    rare_df = df[df["Label"].isin(RARE_ATTACKS)]
    print("Rare attack samples shape:", rare_df.shape)

    print("\nClass distribution:")
    print(rare_df["Label"].value_counts())

    # Drop non-feature columns
    drop_cols = ["Flow ID", "Source IP", "Destination IP", "Timestamp"]
    for col in drop_cols:
        if col in rare_df.columns:
            rare_df = rare_df.drop(columns=[col])

    # Create output directory
    os.makedirs(OUTPUT_PATH, exist_ok=True)

    # Save dataset for GenAI training
    output_file = os.path.join(OUTPUT_PATH, "rare_attacks.csv")
    rare_df.to_csv(output_file, index=False)

    print(f"\nRare attack dataset saved to: {output_file}")
    print("Ready for GenAI synthetic data generation.")

if __name__ == "__main__":
    main()
