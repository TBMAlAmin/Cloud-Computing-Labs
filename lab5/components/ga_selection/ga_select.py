import random
import numpy as np
import pandas as pd

from tsfresh import extract_features
from sklearn.feature_selection import VarianceThreshold, mutual_info_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

from deap import base, creator, tools, algorithms


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

    mi = mutual_info_regression(X_corr, y, random_state=42)
    mi_scores = pd.Series(mi, index=X_corr.columns).sort_values(ascending=False)

    X_final = X_corr[mi_scores.head(20).index]

    n_features = X_final.shape[1]

    if "FitnessMin" not in creator.__dict__:
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    if "Individual" not in creator.__dict__:
        creator.create("Individual", list, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()
    toolbox.register("attr_bool", random.randint, 0, 1)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=n_features)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    def evaluate(individual):
        selected = [i for i, bit in enumerate(individual) if bit == 1]

        if len(selected) == 0:
            return (999999,)

        X_sel = X_final.iloc[:, selected]

        model = RandomForestRegressor(random_state=42, n_estimators=100)
        scores = cross_val_score(
            model,
            X_sel,
            y,
            cv=3,
            scoring="neg_root_mean_squared_error"
        )

        rmse = -scores.mean()
        penalty = 0.01 * len(selected)

        return (rmse + penalty,)

    toolbox.register("evaluate", evaluate)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
    toolbox.register("select", tools.selTournament, tournsize=3)

    random.seed(42)

    population = toolbox.population(n=20)
    ngen = 10
    cxpb = 0.5
    mutpb = 0.2

    algorithms.eaSimple(population, toolbox, cxpb=cxpb, mutpb=mutpb, ngen=ngen, verbose=False)

    best = tools.selBest(population, k=1)[0]

    selected_cols = [X_final.columns[i] for i, bit in enumerate(best) if bit == 1]

    print("Top MI features:", X_final.shape[1])
    print("Selected by GA:", len(selected_cols))
    print(selected_cols)


if __name__ == "__main__":
    main()
