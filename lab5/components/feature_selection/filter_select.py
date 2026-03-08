import numpy as np
import pandas as pd
from tsfresh import extract_features
from sklearn.feature_selection import VarianceThreshold, mutual_info_regression

def main():
    train_path = "../../data/train_FD001.txt"

    cols = ["engine_id", "cycle"]
    cols += [f"op_setting_{i}" for i in range(1, 4)]
    cols += [f"sensor_{i}" for i in range(1, 22)]

    df = pd.read_csv(train_path, sep=r"\s+", header=None)
    df = df.iloc[:, :26]
    df.columns = cols

    max_cycle = df.groupby("engine_id")["cycle"].max()
    df["RUL"] = df["engine_id"].map(max_cycle) - df["cycle"]

    features = extract_features(
        df.drop(columns=["RUL"]),
        column_id="engine_id",
        column_sort="cycle"
    )

    y = df.groupby("engine_id")["RUL"].last()
    X = features.fillna(0)

    vt = VarianceThreshold()
    X_vt = pd.DataFrame(
        vt.fit_transform(X),
        index=X.index,
        columns=X.columns[vt.get_support()]
    )

    corr = X_vt.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))

    drop_cols = [col for col in upper.columns if any(upper[col] > 0.95)]
    X_corr = X_vt.drop(columns=drop_cols)

    mi = mutual_info_regression(X_corr, y)
    mi_scores = pd.Series(mi, index=X_corr.columns).sort_values(ascending=False)

    selected_features = mi_scores.head(20)

    print("After variance threshold:", X_vt.shape)
    print("After correlation filter:", X_corr.shape)
    print(selected_features)

if __name__ == "__main__":
    main()