import argparse
import os
import glob
import pandas as pd
from textblob import TextBlob

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    files = glob.glob(os.path.join(args.data, "*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet file found in {args.data}")

    df = pd.read_parquet(files[0])

    if "reviewText" not in df.columns:
        raise ValueError("Expected column 'reviewText' not found")

    sentiment = df["reviewText"].astype(str).apply(
        lambda x: TextBlob(x).sentiment.polarity
    )

    out_df = pd.DataFrame({"sentiment": sentiment})

    os.makedirs(args.out, exist_ok=True)
    out_df.to_parquet(os.path.join(args.out, "sentiment.parquet"), index=False)

    print("✅ sentiment_features completed")

if __name__ == "__main__":
    main()
