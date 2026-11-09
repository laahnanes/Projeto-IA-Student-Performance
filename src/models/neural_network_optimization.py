import json
import warnings
from pathlib import Path

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
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "processed" / "students_performance_processed.csv"
METRICS_DIR = ROOT / "results" / "metrics"

FEATURES = ["mid_term_class", "faculty", "department"]
TARGET = "final_class"
RANDOM_STATE = 42
N_FOLDS = 10

PAPER = {"accuracy": 0.746, "precision": 0.748, "recall": 0.746, "f1": 0.723, "auc": 0.863}

PARAM_GRID = {
    "nn__hidden_layer_sizes": [(50,), (100,)],
    "nn__alpha": [1e-4, 1.0],
    "nn__learning_rate_init": [1e-3, 1e-2],
}


def build_pipeline(**nn_params):
    encoder = ColumnTransformer(
        [("onehot", OneHotEncoder(handle_unknown="ignore"), FEATURES)]
    )
    nn = MLPClassifier(random_state=RANDOM_STATE, max_iter=500, **nn_params)
    return Pipeline([("encoder", encoder), ("nn", nn)])


def model_baseline():
    """Replicação: configuração padrão do Orange."""
    return build_pipeline(hidden_layer_sizes=(100,), alpha=1e-4)


def model_tuned():
    """Mesma rede, com busca de hiperparâmetros usando apenas o treino do fold."""
    return GridSearchCV(
        build_pipeline(),
        PARAM_GRID,
        cv=StratifiedKFold(3, shuffle=True, random_state=0),
        scoring="accuracy",
        n_jobs=-1,
    )


def cross_val(df, make_model):
    X, y = df[FEATURES], df[TARGET].values
    proba = np.zeros((len(df), 4))
    chosen = []
    cv = StratifiedKFold(N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    for tr, te in cv.split(X, y):
        model = make_model().fit(X.iloc[tr], y[tr])
        proba[te] = model.predict_proba(X.iloc[te])
        if hasattr(model, "best_params_"):
            chosen.append({k.replace("nn__", ""): str(v) for k, v in model.best_params_.items()})
    pred = proba.argmax(axis=1) + 1
    res = {
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred, average="weighted", zero_division=0),
        "recall": recall_score(y, pred, average="weighted"),
        "f1": f1_score(y, pred, average="weighted"),
        "auc": roc_auc_score(y, proba, multi_class="ovr", average="weighted"),
    }
    return {k: round(float(v), 3) for k, v in res.items()}, chosen


def show(title, res):
    print(f"\n{title}")
    print(" " + " ".join(f"{k}={v:.3f}" for k, v in res.items()))


def main(path=DATA_PATH):
    df = pd.read_csv(path)
    print(f"Dados: {df.shape[0]} alunos | {N_FOLDS}-fold estratificado (busca aninhada)")
    print("Artigo (Neural Network):", PAPER)

    base, _ = cross_val(df, model_baseline)
    tuned, chosen = cross_val(df, model_tuned)
    show("1) Replicação (padrão do Orange)", base)
    show("2) Hiperparâmetros ajustados (busca dentro de cada fold)", tuned)

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    out = {
        "n_folds": N_FOLDS,
        "param_grid": {k: [str(v) for v in vs] for k, vs in PARAM_GRID.items()},
        "article_reference": PAPER,
        "results": {"replicacao": base, "otimizada": tuned},
        "best_params_per_fold": chosen,
    }
    with open(METRICS_DIR / "neural_network_optimization.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\nResultados salvos em {METRICS_DIR}")


if __name__ == "__main__":
    main()