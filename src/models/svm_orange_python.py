import json
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore")

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "src").is_dir() and (p / "data").is_dir())
DATA_PATH = ROOT / "data" / "processed" / "students_performance_processed.csv"
METRICS_DIR = ROOT / "results" / "metrics"
FIGURES_DIR = ROOT / "results" / "figures"

FEATURES = ["mid_term_class", "faculty_encoded", "department_encoded"]
TARGET = "final_class"

N_FOLDS = 10
RANDOM_STATE = 42

PAPER = {"accuracy": 0.735, "precision": 0.735, "recall": 0.735, "f1": 0.704, "auc": 0.804}
ARTICLE_SVM = {"AUC": 0.804, "CA": 0.735, "F1": 0.704, "Precision": 0.735, "Recall": 0.735}

SCORING = {
    "accuracy": "accuracy",
    "precision": "precision_weighted",
    "recall": "recall_weighted",
    "f1": "f1_weighted",
    "auc": "roc_auc_ovr_weighted",
}

METRIC_NAMES = {
    "auc": "AUC",
    "accuracy": "CA",
    "f1": "F1",
    "precision": "Precision",
    "recall": "Recall",
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
        ("svm", SVC(C=1.0, kernel="rbf", probability=True, random_state=RANDOM_STATE)),
    ])


def make_cv():
    return StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)


def evaluate(model, X, y):
    r = cross_validate(model, X, y, cv=make_cv(), scoring=SCORING, n_jobs=-1)
    return {k: r[f"test_{k}"].mean() for k in SCORING}


def save_confusion_figure(cm, classes, path):
    pct = cm / cm.sum(axis=0, keepdims=True) * 100
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    ax.imshow(pct, cmap="Blues", vmin=0, vmax=100)
    ticks = range(len(classes))
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)
    ax.set_xlabel("Classe prevista")
    ax.set_ylabel("Classe real")
    ax.set_title("SVM: matriz de confusão (proporção do previsto)")
    for i in ticks:
        for j in ticks:
            color = "white" if pct[i, j] > 55 else "black"
            ax.text(j, i, f"{pct[i, j]:.1f}%\n({cm[i, j]})", ha="center", va="center", color=color, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_metrics_figure(obtained, path):
    names = list(ARTICLE_SVM)
    x = np.arange(len(names))
    width = 0.38
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    bars_a = ax.bar(x - width / 2, [ARTICLE_SVM[n] for n in names], width, label="Artigo", color="#9aa5bb")
    bars_b = ax.bar(x + width / 2, [obtained[n] for n in names], width, label="Replicação (Python)", color="#1f5f84")
    for bars in (bars_a, bars_b):
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01, f"{bar.get_height():.3f}", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Valor da métrica")
    ax.set_title(f"SVM: artigo x replicação ({N_FOLDS}-fold estratificado)")
    ax.legend(loc="upper right", ncol=3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


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
    classes = sorted(y.unique())
    cm = confusion_matrix(y, pred, labels=classes)
    pct = cm / cm.sum(axis=0, keepdims=True) * 100
    print("\nMatriz de confusão (linhas = real, colunas = previsto):")
    print(cm)
    print("Total previsto por classe:", cm.sum(axis=0).tolist())
    print("Proporção do previsto (%),(Tabela 10 do artigo):")
    print(np.round(pct, 1))

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    obtained = {METRIC_NAMES[k]: round(float(v), 3) for k, v in res.items()}
    result = {
        "model": "SVM (SVC, kernel RBF, C=1)",
        "n_folds": N_FOLDS,
        "metric_aggregation": "média das métricas dos folds",
        "features": FEATURES,
        "target": TARGET,
        "hyperparameters": model.named_steps["svm"].get_params(),
        "metrics": obtained,
        "article_reference": ARTICLE_SVM,
    }
    with open(METRICS_DIR / "svm_metrics.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2, default=str)

    cm_df = pd.DataFrame(
        cm,
        index=[f"real_{c}" for c in classes],
        columns=[f"prev_{c}" for c in classes],
    )
    cm_df.to_csv(METRICS_DIR / "svm_confusion_matrix.csv")

    save_confusion_figure(cm, classes, FIGURES_DIR / "svm_confusion_matrix.png")
    save_metrics_figure(obtained, FIGURES_DIR / "svm_metrics_vs_article.png")
    print(f"\nResultados salvos em {METRICS_DIR}")
    print(f"Gráficos salvos em {FIGURES_DIR}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else DATA_PATH)