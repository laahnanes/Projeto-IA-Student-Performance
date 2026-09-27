# Projeto de Inteligência Artificial — Predição de Desempenho Acadêmico
 
Projeto acadêmico da disciplina de Inteligência Artificial. Este repositório contém a
replicação do estudo de Yağcı (2022), *"Educational data mining: prediction of students'
academic performance using machine learning algorithms"* (Smart Learning Environments),
utilizando o dataset original disponibilizado como material suplementar do artigo.
 
## Equipe
 
| Integrante | Conta no GitHub | Responsabilidade inicial |
| --- | --- | --- |
| Larah | `@laahnanes` | Pré-processamento — Parte 1: análise e limpeza dos dados |
| Júlia | `@ailuj97` | Pré-processamento — Parte 2: transformação e discretização dos dados |
| Beatriz | `@bibisouza` | Validação e documentação do pré-processamento |
 
## Descrição geral
 
O projeto consiste na replicação do experimento proposto por Yağcı (2022), que utiliza
algoritmos de aprendizado de máquina para prever a nota final de estudantes universitários
a partir de três variáveis: a nota da prova intermediária (*mid-term*), a Faculdade
(*faculty*) e o Departamento (*department*) do aluno.
 
**Problema estudado:** prever se um estudante terá desempenho satisfatório ao final do
semestre, a partir de informações disponíveis ainda na metade do curso, permitindo
intervenções pedagógicas antecipadas (Educational Data Mining / Learning Analytics).
 
**Relevância:** identificar precocemente estudantes em risco de reprovação permite que a
instituição e os professores atuem preventivamente, antes do exame final, reduzindo a
evasão e o número de reprovações.
 
**Objetivos:**
- Replicar a metodologia original de discretização das notas e preparação dos dados.
- Treinar e comparar o desempenho dos dois algoritmos que obtiveram os melhores resultados
  no artigo original: **Random Forest** e **Neural Network (Rede Neural)**.
- Comparar as métricas obtidas (acurácia, F1, precisão, recall e AUC) com os resultados
  reportados no artigo original.
**Conjunto de dados:** dataset original do artigo (material suplementar), contendo os
registros de 1854 estudantes que cursaram a disciplina Turkish Language-I na Kırşehir Ahi
Evran University (Turquia), no semestre letivo de 2019–2020. As variáveis disponíveis são:
identificador do aluno (`stdID`), nota da prova intermediária (`mid-term`), nota final
(`final`), faculdade (`faculty`) e departamento (`department`).
 
## Estrutura do repositório
 
```text
.
├── .github/
│   ├── ISSUE_TEMPLATE/
│   │   └── task.md
│   └── pull_request_template.md
├── data/
│   ├── raw/
│   │   └── .gitkeep
│   ├── processed/
│   │   └── .gitkeep
│   └── README.md
├── notebooks/
│   └── .gitkeep
├── src/
│   ├── data/
│   │   └── __init__.py
│   ├── preprocessing/
│   │   └── __init__.py
│   ├── models/
│   │   └── __init__.py
│   ├── evaluation/
│   │   └── __init__.py
│   ├── utils/
│   │   └── __init__.py
│   └── __init__.py
├── tests/
│   ├── .gitkeep
│   └── __init__.py
├── results/
│   ├── figures/
│   │   └── .gitkeep
│   └── metrics/
│       └── .gitkeep
├── .gitignore
├── README.md
└── requirements.txt
```
 
### Finalidade dos diretórios
 
- `data/raw/`: datasets originais, sem alterações. Os arquivos de dados não devem ser versionados por padrão.
- `data/processed/`: dados gerados por limpeza, transformação ou preparação. Esses arquivos também não devem ser versionados por padrão.
- `notebooks/`: análises exploratórias, experimentos e registros de investigação em Jupyter.
- `src/data/`: código reutilizável de carregamento e manipulação inicial dos dados.
- `src/preprocessing/`: código reutilizável de pré-processamento.
- `src/models/`: código de modelos e treinamento a ser desenvolvido pelos alunos.
- `src/evaluation/`: código de avaliação e métricas a ser desenvolvido pelos alunos.
- `src/utils/`: funções auxiliares compartilhadas pelo projeto.
- `tests/`: testes automatizados do código produzido pela equipe.
- `results/figures/`: figuras e gráficos gerados pelos experimentos.
- `results/metrics/`: resultados de métricas gerados pelos experimentos.
- `.github/`: modelos para Issues e Pull Requests.
Os notebooks devem apoiar a exploração e a comunicação dos experimentos. Coloque código reutilizável em `src/` para evitar concentrar toda a implementação nos notebooks.
 
## Configuração do ambiente
 
É recomendado utilizar Python 3.10 ou uma versão mais recente compatível com as dependências do projeto.
 
1. Clone o repositório e acesse sua pasta:
```bash
   git clone <URL_DO_REPOSITORIO>
   cd <NOME_DO_REPOSITORIO>
```
 
2. Crie um ambiente virtual:
```bash
   python -m venv .venv
```
 
3. Ative o ambiente virtual:
   No Linux ou macOS:
```bash
   source .venv/bin/activate
```
 
   No Windows (PowerShell):
 
```powershell
   .venv\Scripts\Activate.ps1
```
 
4. Instale as dependências:
```bash
   pip install -r requirements.txt
```
 
## Organização dos dados
 
Armazene os datasets originais em `data/raw/` e preserve-os sem modificações. Salve em `data/processed/` apenas os dados resultantes de limpeza, transformação ou preparação.
 
Datasets e artefatos gerados podem ser grandes ou conter informações que não devem ser publicadas. Por isso, o `.gitignore` impede o versionamento desses arquivos por padrão. Os arquivos `.gitkeep` mantêm a estrutura das pastas no Git. Se a equipe precisar compartilhar dados, deve combinar um meio apropriado e documentar como obtê-los em `data/README.md`.
 
## Fluxo de desenvolvimento
 
O fluxo esperado é:
 
```text
Issue → Branch → Commits → Pull Request → Code Review → Approval → Merge
```
 
Regras do projeto:
 
1. Toda alteração de código deve estar associada a uma Issue.
2. Cada atividade deve ser desenvolvida em uma branch própria.
3. Não devem ser realizados commits diretamente na `main`.
4. Toda alteração deve entrar na `main` por meio de Pull Request.
5. O Pull Request deve estar associado à Issue correspondente. Quando o merge concluir a atividade, use `Closes #numero_da_issue` na descrição do PR.
6. O autor do Pull Request não pode aprovar o próprio PR.
7. Todo PR precisa de pelo menos uma aprovação de outro integrante da equipe antes do merge.
8. Cada integrante deve utilizar sua própria conta GitHub e realizar seus próprios commits.
9. O histórico do GitHub será utilizado para acompanhar a participação individual dos integrantes.
### Exemplo de início de uma atividade
 
Depois de criar ou escolher uma Issue, atualize a `main` local e crie uma branch com um nome descritivo:
 
```bash
git switch main
git pull
git switch -c issue-12-descricao-curta
```
 
Faça commits pequenos e claros. Ao concluir a atividade, envie a branch e abra um Pull Request utilizando o modelo do repositório.
 
## Observação
 
Este template não contém uma solução de Machine Learning. A escolha das técnicas, a implementação, os testes, os experimentos e a análise dos resultados são responsabilidades dos alunos.
