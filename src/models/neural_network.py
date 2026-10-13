"""Modelagem — Neural Network (MLP).

Replica o experimento de Yağcı (2022) com uma rede neural:
- Entradas: mid_term_class, faculty e department (todas categóricas -> One-Hot).
- Alvo: final_class (4 classes).
- Avaliação: validação cruzada estratificada com 10 folds.
- Métricas: acurácia, precisão, recall, F1 (média ponderada) e AUC (one-vs-rest ponderado).

Uso, a partir da raiz do projeto:
    python src/models/neural_network.py
"""

import json
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "processed" / "students_performance_processed.csv"
METRICS_DIR = ROOT / "results" / "metrics"

FEATURES = ["mid_term_class", "faculty", "department"]
TARGET = "final_class"

RANDOM_STATE = 42
N_FOLDS = 10

# Valores reportados no artigo (Tabela 14) para a Neural Network.
ARTICLE_NN = {"AUC": 0.863, "CA": 0.746, "F1": 0.723, "Precision": 0.748, "Recall": 0.746}


def build_model():
    """Pipeline: One-Hot das variáveis de entrada + rede neural (MLP)."""
    encoder = ColumnTransformer(
        [("onehot", OneHotEncoder(handle_unknown="ignore"), FEATURES)]
    )
    # O artigo não informa hiperparâmetros (usou o software Orange).
    # Usamos uma configuração simples e fixa, registrada em hyperparameters.
    mlp = MLPClassifier(
        hidden_layer_sizes=(100,),
        activation="relu",
        solver="adam",
        alpha=0.0001,
        max_iter=500,
        random_state=RANDOM_STATE,
    )
    return Pipeline([("encoder", encoder), ("mlp", mlp)])


def evaluate(model, X, y):
    """Validação cruzada estratificada; métricas calculadas sobre as predições de todos os folds."""
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    proba = cross_val_predict(model, X, y, cv=cv, method="predict_proba")
    classes = sorted(y.unique())
    pred = pd.Series(proba.argmax(axis=1)).map(dict(enumerate(classes))).to_numpy()

    metrics = {
        "AUC": roc_auc_score(y, proba, multi_class="ovr", average="weighted"),
        "CA": accuracy_score(y, pred),
        "F1": f1_score(y, pred, average="weighted"),
        "Precision": precision_score(y, pred, average="weighted", zero_division=0),
        "Recall": recall_score(y, pred, average="weighted"),
    }
    cm = confusion_matrix(y, pred, labels=classes)
    return {k: round(float(v), 3) for k, v in metrics.items()}, cm, classes


def main():
    df = pd.read_csv(DATA_PATH)
    print(f"Dataset carregado: {df.shape[0]} registros")

    X = df[FEATURES]
    y = df[TARGET]

    model = build_model()
    metrics, cm, classes = evaluate(model, X, y)

    print(f"\nNeural Network — validação cruzada com {N_FOLDS} folds")
    print(f"{'Métrica':<10} {'Obtido':>8} {'Artigo':>8}")
    for name, value in metrics.items():
        print(f"{name:<10} {value:>8.3f} {ARTICLE_NN[name]:>8.3f}")

    print("\nMatriz de confusão (linhas = real, colunas = previsto):")
    cm_df = pd.DataFrame(
        cm,
        index=[f"real_{c}" for c in classes],
        columns=[f"prev_{c}" for c in classes],
    )
    print(cm_df)

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    result = {
        "model": "Neural Network (MLPClassifier)",
        "n_folds": N_FOLDS,
        "features": FEATURES,
        "target": TARGET,
        "hyperparameters": model.named_steps["mlp"].get_params(),
        "metrics": metrics,
        "article_reference": ARTICLE_NN,
    }
    with open(METRICS_DIR / "neural_network_metrics.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2, default=str)
    cm_df.to_csv(METRICS_DIR / "neural_network_confusion_matrix.csv")
    print(f"\nResultados salvos em {METRICS_DIR}")


if __name__ == "__main__":
    main()
