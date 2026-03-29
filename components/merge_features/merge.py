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
    parser.add_argument("--base", required=True)
    parser.add_argument("--length", required=True)
    parser.add_argument("--sentiment", required=True)
    parser.add_argument("--tfidf", required=True)
    parser.add_argument("--sbert", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    base_df = load_one_parquet(args.base)
    length_df = load_one_parquet(args.length)
    sentiment_df = load_one_parquet(args.sentiment)
    tfidf_df = load_one_parquet(args.tfidf)
    sbert_df = load_one_parquet(args.sbert)

    keep_cols = [c for c in ["asin", "reviewerID", "overall"] if c in base_df.columns]
    if not keep_cols:
        raise ValueError("Expected asin/reviewerID/overall in base input but none were found.")

    base_keep = base_df[keep_cols].reset_index(drop=True)
    length_df = length_df.reset_index(drop=True)
    sentiment_df = sentiment_df.reset_index(drop=True)
    tfidf_df = tfidf_df.reset_index(drop=True)
    sbert_df = sbert_df.reset_index(drop=True)

    if not (
        len(base_keep) == len(length_df) == len(sentiment_df) == len(tfidf_df) == len(sbert_df)
    ):
        raise ValueError("Feature row counts do not match across component outputs.")

    sbert_df.columns = [f"sbert_{c}" for c in sbert_df.columns.map(str)]
    tfidf_df.columns = tfidf_df.columns.map(str)
    length_df.columns = length_df.columns.map(str)
    sentiment_df.columns = sentiment_df.columns.map(str)

    merged = pd.concat(
        [base_keep, length_df, sentiment_df, tfidf_df, sbert_df],
        axis=1
    )

    os.makedirs(args.out, exist_ok=True)
    merged.to_parquet(os.path.join(args.out, "data.parquet"), index=False)

    print("merge_features completed")
    print("Columns:", list(merged.columns))
    print("Rows:", len(merged))


if __name__ == "__main__":
    main()
