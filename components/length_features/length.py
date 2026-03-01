import argparse
import os
import glob
import pandas as pd

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    files = glob.glob(os.path.join(args.data, "*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet found in: {args.data}")

    df = pd.read_parquet(files[0])

    if "reviewText" not in df.columns:
        raise ValueError("Expected column 'reviewText' not found.")

    out_df = pd.DataFrame({"review_length": df["reviewText"].astype(str).str.len()})

    os.makedirs(args.out, exist_ok=True)
    out_df.to_parquet(os.path.join(args.out, "length.parquet"), index=False)

    print("✅ length_features done")

if __name__ == "__main__":
    main()
