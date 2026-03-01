import argparse
import os
import glob
import pandas as pd
from sklearn.model_selection import train_test_split

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--train_out", type=str, required=True)
    parser.add_argument("--val_out", type=str, required=True)
    parser.add_argument("--test_out", type=str, required=True)
    args = parser.parse_args()

    files = glob.glob(os.path.join(args.data, "*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet found in: {args.data}")

    df = pd.read_parquet(files[0])

    train_ratio = args.train_ratio
    val_ratio = args.val_ratio
    test_ratio = 1.0 - train_ratio - val_ratio
    if train_ratio <= 0 or val_ratio <= 0 or test_ratio <= 0:
        raise ValueError("train_ratio + val_ratio must be < 1.0 and each ratio > 0")

    temp_ratio = 1.0 - train_ratio
    train_df, temp_df = train_test_split(df, test_size=temp_ratio, random_state=args.seed)

    val_within_temp = val_ratio / temp_ratio
    val_df, test_df = train_test_split(
        temp_df, test_size=(1.0 - val_within_temp), random_state=args.seed
    )

    os.makedirs(args.train_out, exist_ok=True)
    os.makedirs(args.val_out, exist_ok=True)
    os.makedirs(args.test_out, exist_ok=True)

    train_df.to_parquet(os.path.join(args.train_out, "train.parquet"), index=False)
    val_df.to_parquet(os.path.join(args.val_out, "val.parquet"), index=False)
    test_df.to_parquet(os.path.join(args.test_out, "test.parquet"), index=False)

    print("✅ split_dataset done")
    print(f"train={len(train_df)} val={len(val_df)} test={len(test_df)}")

if __name__ == "__main__":
    main()
