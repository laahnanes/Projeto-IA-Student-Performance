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

def check_missing_values(df):
    """Verifica a existência de valores ausentes no dataset."""

    print("\n--- VALORES AUSENTES ---")
    print(df.isnull().sum())


def check_duplicates(df):
    """Verifica a existência de registros duplicados no dataset."""

    print("\n--- REGISTROS DUPLICADOS ---")
    print(f"Quantidade de registros duplicados: {df.duplicated().sum()}")

def check_grade_ranges(df):
    print("\n--- VERIFICAÇÃO DOS INTERVALOS DAS NOTAS ---")

    for column in ["mid-term", "final"]:
        minimum = df[column].min()
        maximum = df[column].max()
        invalid_values = df[(df[column] < 0) | (df[column] > 100)]

        print(f"\n{column}:")
        print(f"Menor valor observado: {minimum}")
        print(f"Maior valor observado: {maximum}")
        print(f"Valores fora do intervalo 0-100: {len(invalid_values)}")


def analyze_grades(df):

    print("\n--- ESTATÍSTICAS DESCRITIVAS DAS NOTAS ---")
    print(df[["mid-term", "final"]].describe())


def analyze_categorical_variables(df):

    print("\n--- DISTRIBUIÇÃO POR FACULTY ---")
    print(df["faculty"].value_counts())

    print("\n--- DISTRIBUIÇÃO POR DEPARTMENT ---")
    print(df["department"].value_counts())

def main():
    df = load_data(DATA_PATH)

    print("Dataset carregado com sucesso!")

    inspect_dataset(df)
    check_missing_values(df)
    check_duplicates(df)
    check_grade_ranges(df)
    analyze_grades(df)
    analyze_categorical_variables(df)

if __name__ == "__main__":
    main()