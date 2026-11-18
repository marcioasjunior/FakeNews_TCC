# FakeNews_TCC

Classificação de fake news em português com Machine Learning (Regressão Logística, Random Forest e SVM) e explicabilidade (SHAP e LIME), usando o Fake.br-Corpus.

## Estrutura

```
FakeNews_TCC/
├── preparacao.py              # leitura, divisão treino/teste, TF-IDF e modelos (comum a todos)
├── main.py                    # treina e avalia os modelos: métricas, matrizes de confusão, gráfico
├── shap_explicabilidade.py    # explicações globais e locais com SHAP
├── lime_explicabilidade.py    # explicações locais com LIME (e visão global agregada)
├── Dataset/
│   └── pre-processed.csv      # baixar do Fake.br-Corpus (ver abaixo)
└── resultados/                # gerada ao rodar os scripts
    ├── modelos/               # métricas, matrizes de confusão e gráfico de F1 (main.py)
    ├── shap/                  # explicações globais e locais (shap_explicabilidade.py)
    └── lime/                  # explicações locais, visão global e fidelidade (lime_explicabilidade.py)
```

Os parâmetros do experimento (tamanho do teste, semente, tamanho do vocabulário) ficam apenas em `preparacao.py`, para que todos os scripts usem exatamente os mesmos dados e modelos.

## Dataset

Baixar o arquivo `pre-processed.csv` da pasta `preprocessed/` do repositório
[Fake.br-Corpus](https://github.com/roneysco/Fake.br-Corpus) e salvar em `Dataset/pre-processed.csv`.

## Como rodar

```
pip install pandas scikit-learn matplotlib seaborn shap lime
python main.py                    # métricas e gráficos dos modelos
python shap_explicabilidade.py    # explicações SHAP (pasta resultados/shap/)
python lime_explicabilidade.py    # explicações LIME (pasta resultados/lime/), leva alguns minutos
```
