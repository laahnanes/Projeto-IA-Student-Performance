import pandas as pd


DATA_PATH = "data/raw/students_performance.csv"


def load_data(path):
    return pd.read_csv(path)


def inspect_dataset(df):

    print("\n--- DIMENSÕES DO DATASET ---")
    print(f"Quantidade de registros: {df.shape[0]}")
    print(f"Quantidade de colunas: {df.shape[1]}")

    print("\n--- COLUNAS ---")
    print(df.columns.tolist())

    print("\n--- TIPOS DAS VARIÁVEIS ---")
    print(df.dtypes)


def main():
    df = load_data(DATA_PATH)

    print("Dataset carregado com sucesso!")

    inspect_dataset(df)


if __name__ == "__main__":
    main()