import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import cross_validate, StratifiedKFold


DATA_PATH = "data/processed/students_performance_processed.csv"


def load_data(path):
    """Carrega o dataset resultante do pré-processamento."""
    return pd.read_csv(path)


def prepare_data(df):
    """Define as variáveis de entrada e a variável alvo do modelo."""

    features = [
        "mid_term_class",
        "faculty_encoded",
        "department_encoded",
    ]

    target = "final_class"

    X = df[features]
    y = df[target]

    return X, y


def create_model():
    """Cria o modelo Support Vector Machine."""
    return SVC(probability=True)


def evaluate_model(model, X, y):
    """Avalia o modelo utilizando validação cruzada estratificada com 10 folds."""

    cv = StratifiedKFold(
        n_splits=10,
        shuffle=True,
        random_state=42
    )

    scoring = {
        "accuracy": "accuracy",
        "precision": "precision_weighted",
        "recall": "recall_weighted",
        "f1": "f1_weighted",
        "auc": "roc_auc_ovr_weighted",
    }

    results = cross_validate(
        model,
        X,
        y,
        cv=cv,
        scoring=scoring
    )

    return results


def main():
    df = load_data(DATA_PATH)
    X, y = prepare_data(df)

    print("Dataset processado carregado com sucesso!")

    print("\n--- VARIÁVEIS DE ENTRADA (X) ---")
    print(X.head())

    print("\n--- VARIÁVEL ALVO (y) ---")
    print(y.head())

    print("\n--- DIMENSÕES ---")
    print(f"X: {X.shape}")
    print(f"y: {y.shape}")

    model = create_model()

    print("\nModelo SVM criado com sucesso!")
    print(model)

    results = evaluate_model(model, X, y)

    print("\n--- VALIDAÇÃO CRUZADA (10 FOLDS) ---")
    print(f"Acurácia média: {results['test_accuracy'].mean():.3f}")
    print(f"Precisão média: {results['test_precision'].mean():.3f}")
    print(f"Recall médio: {results['test_recall'].mean():.3f}")
    print(f"F1-score médio: {results['test_f1'].mean():.3f}")
    print(f"AUC médio: {results['test_auc'].mean():.3f}")


if __name__ == "__main__":
    main()