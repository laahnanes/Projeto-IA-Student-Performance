# Validação e Documentação do Pré-processamento
**Projeto:** Replicação — Predição de desempenho acadêmico (Yağcı, 2022)
**Responsável por esta etapa:** Beatriz
**Baseado em:** artigo original + dataset `40561_2022_192_MOESM1_ESM.xlsx` + PRs #2 e #4 (Larah) e PR #3 (Júlia)

---

## 1. Objetivo desta etapa

Conferir se as etapas de análise inicial (Larah) e transformação dos dados (Júlia) seguem
exatamente a metodologia descrita no artigo, validar o dataset processado e documentar
os resultados para uso confiável nas etapas seguintes (modelagem com Random Forest e
Neural Network).

## 2. Dataset original — características confirmadas

Verificação feita diretamente sobre `40561_2022_192_MOESM1_ESM.xlsx`:

| Item | Resultado | Bate com o artigo? |
|---|---|---|
| Nº de registros | 1854 | Sim (Tabela 2 e Tabela 3 do artigo) |
| Nº de colunas | 5 (`stdID`, `mid-term`, `final`, `faculty`, `department`) | Sim |
| Valores ausentes | 0 em todas as colunas | Sim |
| Registros duplicados (linha completa) | 0 | Sim |
| Registros duplicados (`stdID`) | 0 | Sim |
| Faixa de `mid-term` | 10 a 100 | Artigo cita "0 a 100" — mínimo real é 10 |
| Faixa de `final` | 10 a 100 | Mesmo ponto acima |
| Distribuição por faculdade (`faculty`) | 404 / 319 / 296 / 221 / 192 / 116 / 92 / 88 / 68 / 30 / 28 | ✅ Idêntico à Tabela 2 |
| Nº de departamentos distintos (`department`) | 44 | Não informado no artigo, mas consistente internamente |

## 3. Revisão da análise e limpeza dos dados — Larah (Issue #1)

**PRs:**
- PR #2 (versão inicial): https://github.com/laahnanes/Projeto-IA-Student-Performance/pull/2
- PR #4 (ajustes solicitados na revisão): https://github.com/laahnanes/Projeto-IA-Student-Performance/pull/4

Arquivo revisado: `src/preprocessing/data_analysis.py`

| Tarefa da issue | Implementada no código? | PR |
|---|---|---|
| Carregar e inspecionar o dataset | Sim `load_data()` / `inspect_dataset()` | #2 |
| Conferir quantidade de registros e colunas | Sim `inspect_dataset()` | #2 |
| Verificar tipos das variáveis | Sim `inspect_dataset()` (`df.dtypes`) | #2 |
| Verificar valores ausentes | Sim `check_missing_values()` | #2 |
| Verificar registros duplicados | Sim `check_duplicates()` | #2 |
| Verificar valores inconsistentes / fora do intervalo esperado | Sim `check_grade_ranges()` | #4 |
| Análise básica das notas e variáveis | Sim `analyze_grades()` e `analyze_categorical_variables()` | #4 |

**Histórico da revisão:** no PR #2, foram identificados dois itens faltantes (checagem de
intervalo das notas e análise descritiva). Larah os implementou no PR #4, que também
fecha a Issue #1 (`Closes #1`).

**Conferência dos resultados do PR #4** (mesma lógica executada sobre o dataset original):

| Verificação | Resultado |
|---|---|
| `mid-term`: mínimo / máximo / fora de 0–100 | 10 / 100 / 0 |
| `final`: mínimo / máximo / fora de 0–100 | 10 / 100 / 0 |
| `mid-term`: média / desvio padrão | 72,50 / 19,65 |
| `final`: média / desvio padrão | 72,10 / 15,90 |
| Categorias distintas em `faculty` / `department` | 11 / 44 |

Todos os valores conferem com o dataset original e com o artigo (ver seção 2).

**Ponto menor (não bloqueante):** o script não verifica variações de escrita nas colunas
de texto (ex.: o mesmo departamento escrito de duas formas). No dataset atual isso não
ocorre, pois os 44 departamentos são distintos e consistentes.

## 4. Validação da transformação dos dados — Júlia (PR #3)

**Link:** https://github.com/laahnanes/Projeto-IA-Student-Performance/pull/3
**Arquivo:** `src/preprocessing/transform.py`
**Status:** Validado. Resultados batem exatamente com o gabarito e com o artigo.

### 4.1 Discretização das notas

O código usa `pd.cut(df[col], bins=[-inf, 32.5, 55, 77.5, inf], labels=[1,2,3,4], right=False)`,
o que corresponde exatamente à regra do artigo (limite inferior incluso, superior exclusivo):

- Classe 1: nota < 32,5
- Classe 2: 32,5 ≤ nota < 55
- Classe 3: 55 ≤ nota < 77,5
- Classe 4: nota ≥ 77,5

Reexecutei essa mesma lógica sobre o dataset original para conferir:

| Classe | Faixa | `final_class` (esperado / obtido) | `mid_term_class` (esperado / obtido) |
|---|---|---|---|
| 1 | < 32,5 | 38 / **38** | 56 / **56** |
| 2 | 32,5 – 55 | 154 / **154** | 277 / **277** |
| 3 | 55 – 77,5 | 1016 / **1016** | 734 / **734** |
| 4 | ≥ 77,5 | 646 / **646** | 787 / **787** |
| **Total** | | 1854 / **1854** | 1854 / **1854** |

Todos os valores batem exatamente. Não há registros nulos gerados pela discretização, e o
número de registros (1854) foi preservado.

### 4.2 Tratamento das variáveis categóricas (`faculty`, `department`)

O código implementa duas estratégias intercambiáveis:

- **Label Encoding** (padrão em `main()`): converte cada categoria em um número inteiro
  arbitrário (`LabelEncoder` do scikit-learn).
- **One-Hot Encoding** (disponível via parâmetro `categorical_strategy="onehot"`).

**Observação para o grupo:** Label Encoding é adequado para Random Forest (baseado em
árvores, que não assume ordem entre os valores), mas pode ser problemático para a Neural
Network, que pode interpretar a numeração como uma relação de ordem/distância entre
faculdades/departamentos que não existe de fato. Recomenda-se usar **One-Hot Encoding**
como entrada da Neural Network (a função já suporta essa troca via parâmetro).

### 4.3 Pontos menores (não bloqueiam a aprovação)

- O `main()` imprime apenas a distribuição de `final_class`, não a de `mid_term_class`
  (apenas um log a mais, não afeta o resultado).
- Vale documentar explicitamente, no relatório final, qual estratégia de encoding
  (label ou one-hot) foi usada para cada um dos dois modelos (RF e NN).

### 4.4 Conclusão da validação desta etapa

A transformação implementada por Júlia está **correta e fiel à metodologia do artigo**.
Não foram encontradas inconsistências que impeçam a aprovação do PR #3.

## 5. Conclusão geral

- O dataset original está íntegro, sem nulos ou duplicatas, e consistente com o artigo.
- A etapa de análise e limpeza (Larah, PRs #2 e #4) está **completa e validada**: todas as
  tarefas da Issue #1 foram implementadas e os resultados conferem com o dataset original.
- A etapa de transformação (Júlia, PR #3) foi **totalmente validada**: a discretização
  das notas está correta e idêntica à metodologia do artigo, os 1854 registros foram
  preservados, e as variáveis categóricas foram devidamente preparadas para os modelos.
- Única observação geral: a diferença entre a faixa de notas descrita no artigo (0–100)
  e a faixa real dos dados (10–100).

## 6. Situação final e próximos passos

Com o pré-processamento validado, o projeto está pronto para a etapa de modelagem.

1. Gerar o dataset processado localmente com o script de transformação (os arquivos de
   dados não são versionados no repositório, conforme o `.gitignore`).
2. Treinar e avaliar **Random Forest** e **Neural Network** sobre o dataset processado.
3. Para a Neural Network, considerar o uso de One-Hot Encoding em `faculty` e `department`
   (ver seção 4.2), documentando qual estratégia foi usada em cada modelo.
4. Comparar as métricas obtidas (acurácia, F1, precisão, recall, AUC) com as do artigo
   (RF: CA 0,746 / AUC 0,860; NN: CA 0,746 / AUC 0,863).
5. Incorporar esta documentação à seção de metodologia do relatório final do grupo.
