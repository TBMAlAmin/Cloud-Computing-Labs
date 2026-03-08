import pandas as pd
from tsfresh import extract_features

def main():
    train_path = "../../data/train_FD001.txt"

    cols = ["engine_id", "cycle"]
    cols += [f"op_setting_{i}" for i in range(1, 4)]
    cols += [f"sensor_{i}" for i in range(1, 22)]

    df = pd.read_csv(train_path, sep=r"\s+", header=None)
    df = df.iloc[:, :26]
    df.columns = cols

    features = extract_features(
        df,
        column_id="engine_id",
        column_sort="cycle"
    )

    print(features.shape)
    print(features.head())

if __name__ == "__main__":
    main()
