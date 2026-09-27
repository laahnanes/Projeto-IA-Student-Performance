import pandas as pd
from sklearn.preprocessing import LabelEncoder

GRADE_COLUMNS = ["mid-term", "final"]

CATEGORICAL_COLUMNS = ["faculty", "department"]

RAW_PATH = "data/raw/students_performance.csv"
FINAL_CLASS_COLUMN = "final_class"
PROCESSED_PATH = "data/processed/students_performance_processed.csv"

BIN_EDGES = [-float("inf"), 32.5, 55, 77.5, float("inf")]
BIN_LABELS = [1, 2, 3, 4]


def discretize_grades(df, columns=GRADE_COLUMNS, edges=BIN_EDGES, labels=BIN_LABELS):
    """
    Discretiza as colunas de nota informadas segundo as faixas do artigo:

        Classe 1: nota < 32.5
        Classe 2: 32.5 <= nota < 55
        Classe 3: 55 <= nota < 77.5
        Classe 4: nota >= 77.5

    """
    df = df.copy()

    for col in columns:
        if col not in df.columns:
            raise KeyError(
                f"Coluna '{col}' não encontrada no dataset. "
                f"Colunas disponíveis: {list(df.columns)}"
            )

        new_col = f"{col.replace('-', '_')}_class"
        df[new_col] = pd.cut(
            df[col],
            bins=edges,
            labels=labels,
            right=False,  
        ).astype("Int64")  

    return df


def encode_categoricals_label(df, columns=CATEGORICAL_COLUMNS):

    """Aplica Label Encoding em `faculty` e `department`, criando colunas
    "<coluna>_encoded" com códigos inteiros."""

    df = df.copy()
    encoders = {}

    for col in columns:
        if col not in df.columns:
            raise KeyError(
                f"Coluna '{col}' não encontrada no dataset. "
                f"Colunas disponíveis: {list(df.columns)}"
            )

        le = LabelEncoder()
        df[f"{col}_encoded"] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

    return df, encoders


def encode_categoricals_onehot(df, columns=CATEGORICAL_COLUMNS):
    
    """Aplica One-Hot Encoding em `faculty` e `department`."""
    
    for col in columns:
        if col not in df.columns:
            raise KeyError(
                f"Coluna '{col}' não encontrada no dataset. "
                f"Colunas disponíveis: {list(df.columns)}"
            )

    return pd.get_dummies(df, columns=columns, prefix=columns)



def build_processed_dataset(
    df,
    grade_columns=GRADE_COLUMNS,
    categorical_columns=CATEGORICAL_COLUMNS,
    categorical_strategy="label",
):
    
    """Executa o pipeline completo de transformação"""

    df_out = discretize_grades(df, columns=grade_columns)

    if categorical_strategy == "label":
        df_out, _ = encode_categoricals_label(df_out, columns=categorical_columns)
    elif categorical_strategy == "onehot":
        df_out = encode_categoricals_onehot(df_out, columns=categorical_columns)
    else:
        raise ValueError(
            "categorical_strategy deve ser 'label' ou 'onehot', "
            f"recebido: {categorical_strategy!r}"
        )

    return df_out


def save_processed_dataset(df, path=PROCESSED_PATH):
    """Salva o dataset processado em CSV, criando a pasta se necessário."""
    import os

    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Dataset processado salvo em: {path}")


def main():
    from src.preprocessing.data_analysis import load_data

    df = load_data(RAW_PATH)
    print(f"Dataset original carregado: {df.shape[0]} registros, {df.shape[1]} colunas.")

    df_processed = build_processed_dataset(df, categorical_strategy="label")
    print(
        f"Dataset processado gerado: {df_processed.shape[0]} registros, "
        f"{df_processed.shape[1]} colunas."
    )

    print(f"\n--- Distribuição das classes ({FINAL_CLASS_COLUMN}) ---")
    print(df_processed[FINAL_CLASS_COLUMN].value_counts().sort_index())

    save_processed_dataset(df_processed)


if __name__ == "__main__":
    main()