import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore")

DATA_PATH = "data/processed/students_performance_processed.csv"
TARGET = "final_class"
SEEDS = (42, 7, 123)

PAPER = {"accuracy": 0.735, "precision": 0.735, "recall": 0.735, "f1": 0.704, "auc": 0.804}


def load_data(path):
    df = pd.read_csv(path)

    g = df.groupby("department_encoded")["mid-term"]
    df["dept_n"] = df["department_encoded"].map(df["department_encoded"].value_counts())
    df["dept_mid_mean"] = g.transform("mean")
    df["mid_rel"] = df["mid-term"] - df["dept_mid_mean"]
    df["mid_rank_dept"] = g.rank(pct=True)
    return df


def target_encoding(train_df, apply_df, col="department_encoded", alpha=10):
    mu = train_df["final"].mean()
    g = train_df.groupby(col)["final"].agg(["sum", "count"])
    mapping = ((g["sum"] + alpha * mu) / (g["count"] + alpha)).to_dict()
    return apply_df[col].map(mapping).fillna(mu).values


def build_features_improved(train_df, test_df):
    cols = ["mid-term", "dept_mid_mean", "mid_rel", "mid_rank_dept", "dept_n"]
    Xa, Xb = train_df[cols].copy(), test_df[cols].copy()

    te = np.zeros(len(train_df))
    inner = StratifiedKFold(5, shuffle=True, random_state=0)
    for a, b in inner.split(train_df, train_df[TARGET]):
        te[b] = target_encoding(train_df.iloc[a], train_df.iloc[b])
    Xa["dept_te"] = te
    Xb["dept_te"] = target_encoding(train_df, test_df)
    return Xa, Xb


def build_features_baseline(train_df, test_df):
    cols = ["mid_term_class", "faculty_encoded", "department_encoded"]
    return train_df[cols], test_df[cols]


def model_baseline():
    cats = ["mid_term_class", "faculty_encoded", "department_encoded"]
    prep = ColumnTransformer(
        [("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cats)]
    )
    return Pipeline([("prep", prep), ("scale", StandardScaler()),
                     ("svm", SVC(probability=True, random_state=42))])


def model_improved():
    return make_pipeline(StandardScaler(), SVC(probability=True, random_state=42))


def cross_val(df, make_model, build_features, seed=42):
    y = df[TARGET].values
    proba = np.zeros((len(df), 4))
    for tr, te in StratifiedKFold(10, shuffle=True, random_state=seed).split(df, y):
        Xa, Xb = build_features(df.iloc[tr], df.iloc[te])
        proba[te] = make_model().fit(Xa, y[tr]).predict_proba(Xb)
    pred = proba.argmax(axis=1) + 1
    return {
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred, average="weighted", zero_division=0),
        "recall": recall_score(y, pred, average="weighted"),
        "f1": f1_score(y, pred, average="weighted"),
        "auc": roc_auc_score(y, proba, multi_class="ovr", average="weighted"),
        "recall_macro": recall_score(y, pred, average="macro"),
    }


def mean_over_seeds(df, *args):
    runs = [cross_val(df, *args, seed=s) for s in SEEDS]
    return {k: float(np.mean([r[k] for r in runs])) for k in runs[0]}


def show(title, res):
    print(f"\n{title}")
    print("  " + "  ".join(f"{k}={v:.3f}" for k, v in res.items()))


def main(path=DATA_PATH):
    df = load_data(path)
    print(f"Dados: {df.shape[0]} alunos | média de {len(SEEDS)} divisões de folds (10-fold estratificado)")
    print("Artigo (SVM):", PAPER)

    show("1) Replicação (equivalente ao Orange)",
         mean_over_seeds(df, model_baseline, build_features_baseline))
    show("2) SVM com features melhoradas",
         mean_over_seeds(df, model_improved, build_features_improved))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else DATA_PATH)