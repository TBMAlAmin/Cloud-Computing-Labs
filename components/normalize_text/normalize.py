import argparse
import os
import glob
import re
import string
import pandas as pd

def normalize(s: str) -> str:
    if s is None:
        return ""
    s = str(s).lower().strip()
    s = re.sub(r"https?://\S+|www\.\S+", " ", s)
    s = re.sub(r"\d+", " ", s)
    s = s.translate(str.maketrans("", "", string.punctuation))
    s = re.sub(r"\s+", " ", s).strip()
    return s

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

    df["reviewText"] = df["reviewText"].apply(normalize)
    df = df[df["reviewText"].str.len() >= 10].copy()

    os.makedirs(args.out, exist_ok=True)
    df.to_parquet(os.path.join(args.out, "normalized.parquet"), index=False)
    print("✅ normalize_text done")

if __name__ == "__main__":
    main()
