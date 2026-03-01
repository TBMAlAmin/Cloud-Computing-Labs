import argparse
import os
import glob
import pandas as pd

def load_one_parquet(folder):
    files = glob.glob(os.path.join(folder, "*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet file found in {folder}")
    return pd.read_parquet(files[0])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--length", required=True)
    parser.add_argument("--sentiment", required=True)
    parser.add_argument("--tfidf", required=True)
    parser.add_argument("--sbert", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    length_df = load_one_parquet(args.length)
    sentiment_df = load_one_parquet(args.sentiment)
    tfidf_df = load_one_parquet(args.tfidf)
    sbert_df = load_one_parquet(args.sbert)

    # Ensure all columns are strings (parquet-safe)
    length_df.columns = length_df.columns.map(str)
    sentiment_df.columns = sentiment_df.columns.map(str)
    tfidf_df.columns = tfidf_df.columns.map(str)
    sbert_df.columns = sbert_df.columns.map(str)

    merged = pd.concat([length_df, sentiment_df, tfidf_df, sbert_df], axis=1)

    os.makedirs(args.out, exist_ok=True)
    merged.to_parquet(os.path.join(args.out, "all_features.parquet"), index=False)

    print("✅ merge_features completed")

if __name__ == "__main__":
    main()
