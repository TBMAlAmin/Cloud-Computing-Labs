import pandas as pd

train_path = "../../data/train_FD001.txt"

cols = ["engine_id", "cycle"]
cols += [f"op_setting_{i}" for i in range(1,4)]
cols += [f"sensor_{i}" for i in range(1,22)]

df = pd.read_csv(train_path, sep=r"\s+", header=None)
df = df.iloc[:, :26]
df.columns = cols

max_cycle = df.groupby("engine_id")["cycle"].max()

df["RUL"] = df["engine_id"].map(max_cycle) - df["cycle"]

print(df.head())
print(df.shape)
