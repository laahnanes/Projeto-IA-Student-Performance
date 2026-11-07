import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore")

DATA_PATH = "data/processed/students_performance_processed.csv"


FEATURES = ["mid_term_class", "faculty_encoded", "department_encoded"]
TARGET = "final_class"


PAPER = {"accuracy": 0.735, "precision": 0.735, "recall": 0.735, "f1": 0.704, "auc": 0.804}

SCORING = {
    "accuracy": "accuracy",
    "precision": "precision_weighted",
    "recall": "recall_weighted",
    "f1": "f1_weighted",
    "auc": "roc_auc_ovr_weighted",
}


def load_data(path):
    df = pd.read_csv(path)
    return df[FEATURES], df[TARGET]


def create_model():
    prep = ColumnTransformer(
        [("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False), FEATURES)]
    )
    return Pipeline([
        ("prep", prep),
        ("scale", StandardScaler()),
        ("svm", SVC(C=1.0, kernel="rbf", probability=True, random_state=42)),
    ])


def make_cv():
    return StratifiedKFold(n_splits=10, shuffle=True, random_state=42)


def evaluate(model, X, y):
    r = cross_validate(model, X, y, cv=make_cv(), scoring=SCORING, n_jobs=-1)
    return {k: r[f"test_{k}"].mean() for k in SCORING}


def main(path=DATA_PATH):
    X, y = load_data(path)
    print(f"X: {X.shape} | y: {y.shape}")
    print("Distribuição do alvo:", y.value_counts().sort_index().to_dict())

    model = create_model()
    res = evaluate(model, X, y)

    print("\n--- RESULTADOS (validação cruzada, 10 folds) ---")
    print(f"{'':10s}{'accuracy':>10s}{'precision':>10s}{'recall':>10s}{'f1':>10s}{'auc':>10s}")
    print(f"{'Artigo':10s}" + "".join(f"{PAPER[k]:10.3f}" for k in SCORING))
    print(f"{'Python':10s}" + "".join(f"{res[k]:10.3f}" for k in SCORING))

    pred = cross_val_predict(model, X, y, cv=make_cv(), n_jobs=-1)
    cm = confusion_matrix(y, pred)
    pct = cm / cm.sum(axis=0, keepdims=True) * 100
    print("\nMatriz de confusão (linhas = real, colunas = previsto):")
    print(cm)
    print("Total previsto por classe:", cm.sum(axis=0).tolist())
    print("Proporção do previsto (%),(Tabela 10 do artigo):")
    print(np.round(pct, 1))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else DATA_PATH)