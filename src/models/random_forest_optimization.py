import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "processed" / "students_performance_processed.csv"
METRICS_DIR = ROOT / "results" / "metrics"

TARGET = "final_class"
SEEDS = (42, 7, 123)
N_FOLDS = 10

PAPER = {"accuracy": 0.746, "precision": 0.752, "recall": 0.746, "f1": 0.721, "auc": 0.860}

PARAM_GRID = {
    "min_samples_leaf": [5, 10, 15],
    "max_features": ["sqrt", 0.5],
}


def load_data(path):
    df = pd.read_csv(path)
    g = df.groupby("department_encoded")["mid-term"]
    df["dept_n"] = df["department_encoded"].map(df["department_encoded"].value_counts())
    df["dept_mid_mean"] = g.transform("mean")
    df["mid_rel"] = df["mid-term"] - df["dept_mid_mean"]
    df["mid_rank_dept"] = g.rank(pct=True)
    return df


def target_encoding(train_df, apply_df, col="department_encoded", alpha=10):
    """Média da nota final por departamento (suavizada), calculada só com o treino."""
    mu = train_df["final"].mean()
    g = train_df.groupby(col)["final"].agg(["sum", "count"])
    mapping = ((g["sum"] + alpha * mu) / (g["count"] + alpha)).to_dict()
    return apply_df[col].map(mapping).fillna(mu).values


def build_features_baseline(train_df, test_df):
    cols = ["mid_term_class", "faculty_encoded", "department_encoded"]
    return train_df[cols], test_df[cols]


def build_features_improved(train_df, test_df):
    cols = ["mid-term", "dept_mid_mean", "mid_rel", "mid_rank_dept", "dept_n", "faculty_encoded"]
    Xa, Xb = train_df[cols].copy(), test_df[cols].copy()

    te = np.zeros(len(train_df))
    inner = StratifiedKFold(5, shuffle=True, random_state=0)
    for a, b in inner.split(train_df, train_df[TARGET]):
        te[b] = target_encoding(train_df.iloc[a], train_df.iloc[b])
    Xa["dept_te"] = te
    Xb["dept_te"] = target_encoding(train_df, test_df)
    return Xa, Xb


def model_baseline():
    cats = ["mid_term_class", "faculty_encoded", "department_encoded"]
    prep = ColumnTransformer([("oh", OneHotEncoder(handle_unknown="ignore"), cats)])
    rf = RandomForestClassifier(n_estimators=100, min_samples_split=5, random_state=42)
    return Pipeline([("prep", prep), ("rf", rf)])


def model_improved():
    return RandomForestClassifier(n_estimators=100, min_samples_split=5, random_state=42)


def model_tuned():
    """Random Forest com busca de hiperparâmetros (usada apenas no treino de cada fold)."""
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    return GridSearchCV(
        rf,
        PARAM_GRID,
        cv=StratifiedKFold(3, shuffle=True, random_state=0),
        scoring="accuracy",
        n_jobs=1,
    )


def cross_val(df, make_model, build_features, seed=42):
    y = df[TARGET].values
    proba = np.zeros((len(df), 4))
    for tr, te in StratifiedKFold(N_FOLDS, shuffle=True, random_state=seed).split(df, y):
        Xa, Xb = build_features(df.iloc[tr], df.iloc[te])
        proba[te] = make_model().fit(Xa, y[tr]).predict_proba(Xb)
    pred = proba.argmax(axis=1) + 1
    return {
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred, average="weighted", zero_division=0),
        "recall": recall_score(y, pred, average="weighted"),
        "f1": f1_score(y, pred, average="weighted"),
        "auc": roc_auc_score(y, proba, multi_class="ovr", average="weighted"),
    }


def mean_over_seeds(df, *args):
    runs = [cross_val(df, *args, seed=s) for s in SEEDS]
    return {k: round(float(np.mean([r[k] for r in runs])), 3) for k in runs[0]}


def show(title, res):
    print(f"\n{title}")
    print("  " + "  ".join(f"{k}={v:.3f}" for k, v in res.items()))


def main(path=DATA_PATH):
    df = load_data(path)
    print(f"Dados: {df.shape[0]} alunos | média de {len(SEEDS)} divisões de folds ({N_FOLDS}-fold estratificado)")
    print("Artigo (Random Forest):", PAPER)

    results = {
        "1) Replicação (entradas do artigo, One-Hot)": mean_over_seeds(df, model_baseline, build_features_baseline),
        "2) Features melhoradas": mean_over_seeds(df, model_improved, build_features_improved),
        "3) Features melhoradas + hiperparâmetros ajustados": mean_over_seeds(df, model_tuned, build_features_improved),
    }
    for title, res in results.items():
        show(title, res)

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    out = {"seeds": list(SEEDS), "n_folds": N_FOLDS, "param_grid": PARAM_GRID,
           "article_reference": PAPER, "results": results}
    with open(METRICS_DIR / "random_forest_optimization.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\nResultados salvos em {METRICS_DIR}")


if __name__ == "__main__":
    main()
