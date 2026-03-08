import random
import time
import numpy as np
import pandas as pd

from tsfresh import extract_features
from sklearn.feature_selection import VarianceThreshold, mutual_info_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from deap import base, creator, tools, algorithms


def main():
    start_time = time.time()

    train_path = "../../data/train_FD001.txt"
    window_size = 30
    random.seed(42)
    np.random.seed(42)

    cols = ["engine_id", "cycle"]
    cols += [f"op_setting_{i}" for i in range(1, 4)]
    cols += [f"sensor_{i}" for i in range(1, 22)]

    df = pd.read_csv(train_path, sep=r"\s+", header=None)
    df = df.iloc[:, :26]
    df.columns = cols

    max_cycle = df.groupby("engine_id")["cycle"].max()
    df["max_cycle"] = df["engine_id"].map(max_cycle)
    df["RUL"] = df["max_cycle"] - df["cycle"]

    samples = []
    targets = {}

    for engine_id, group in df.groupby("engine_id"):
        group = group.sort_values("cycle").copy()
        engine_max = int(group["max_cycle"].iloc[0])

        min_cutoff = window_size
        max_cutoff = engine_max - 1

        if max_cutoff < min_cutoff:
            continue

        cutoff = np.random.randint(min_cutoff, max_cutoff + 1)

        window = group[(group["cycle"] > cutoff - window_size) & (group["cycle"] <= cutoff)].copy()

        if len(window) != window_size:
            continue

        samples.append(window.drop(columns=["max_cycle", "RUL"]))
        targets[engine_id] = engine_max - cutoff

    X_input = pd.concat(samples, ignore_index=True)
    y = pd.Series(targets).sort_index()

    features = extract_features(
        X_input,
        column_id="engine_id",
        column_sort="cycle"
    )

    features = features.sort_index()
    X = features.replace([np.inf, -np.inf], np.nan).fillna(0)

    y = y.loc[X.index]

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

    X_top = X_corr[mi_scores.head(20).index]

    n_features = X_top.shape[1]

    if not hasattr(creator, "FitnessMin"):
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    if not hasattr(creator, "Individual"):
        creator.create("Individual", list, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()
    toolbox.register("attr_bool", random.randint, 0, 1)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=n_features)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    def evaluate(individual):
        selected = [i for i, bit in enumerate(individual) if bit == 1]

        if len(selected) == 0:
            return (999999,)

        X_sel = X_top.iloc[:, selected]

        model = RandomForestRegressor(random_state=42, n_estimators=100)
        X_train_cv, X_test_cv, y_train_cv, y_test_cv = train_test_split(
            X_sel, y, test_size=0.2, random_state=42
        )

        model.fit(X_train_cv, y_train_cv)
        preds_cv = model.predict(X_test_cv)
        rmse_cv = np.sqrt(mean_squared_error(y_test_cv, preds_cv))
        penalty = 0.01 * len(selected)

        return (rmse_cv + penalty,)

    toolbox.register("evaluate", evaluate)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
    toolbox.register("select", tools.selTournament, tournsize=3)

    population = toolbox.population(n=20)
    algorithms.eaSimple(population, toolbox, cxpb=0.5, mutpb=0.2, ngen=10, verbose=False)

    best = tools.selBest(population, k=1)[0]
    selected_cols = [X_top.columns[i] for i, bit in enumerate(best) if bit == 1]

    X_final = X_top[selected_cols]

    X_train, X_test, y_train, y_test = train_test_split(
        X_final, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(random_state=42, n_estimators=100)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))

    end_time = time.time()

    print("Samples used:", len(X_final))
    print("Selected features:", len(selected_cols))
    print("RMSE:", rmse)
    print("Runtime (seconds):", round(end_time - start_time, 2))


if __name__ == "__main__":
    main()