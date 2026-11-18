# FakeNews_TCC

Classificação de notícias falsas em português com Machine Learning (Regressão Logística, Random Forest e SVM) e explicabilidade (SHAP e LIME), usando o [Fake.br-Corpus](https://github.com/roneysco/Fake.br-Corpus).

## Estrutura

```
TCC/
├── Dados/
│   └── Fake.br-Corpus-master/      # repositório do corpus (só para o experimento de tamanho igualado)
└── Code/                           # esta pasta
    ├── executar.py                 # menu: roda cada etapa ou tudo de uma vez e mostra os resultados
    ├── preparacao.py               # parte comum: leitura, divisão 80/20, TF-IDF, modelos e amostra estratificada
    ├── main.py                     # avaliação na divisão 80/20: métricas, matrizes de confusão e gráfico
    ├── avaliacao_cv.py             # validação cruzada estratificada de 5 partes
    ├── shap_explicabilidade.py     # SHAP: importância global, notícias selecionadas e palavras ausentes
    ├── lime_explicabilidade.py     # LIME: notícias selecionadas, visão global e fidelidade
    ├── corpus_original.py          # leitura dos textos originais, pré-processamento reconstruído e pares
    ├── tamanho_igualado.py         # desempenho com texto completo x tamanho igualado
    ├── shap_tamanho_igualado.py    # SHAP com texto completo x tamanho igualado
    ├── requirements.txt            # versões das bibliotecas
    ├── Dataset/
    │   └── pre-processed.csv       # baixar do Fake.br-Corpus (ver abaixo)
    └── resultados/                 # criada ao rodar os scripts
```

Os parâmetros do experimento (tamanho do teste, semente e tamanho do vocabulário) ficam só em `preparacao.py`, para que todos os scripts usem exatamente os mesmos dados e modelos.

## Preparação

1. **Bibliotecas.** Com Python 3.13 (de preferência dentro de um ambiente virtual):

   ```
   pip install -r requirements.txt
   ```

2. **Dados principais.** Baixe o arquivo `pre-processed.csv` da pasta `preprocessed/` do [Fake.br-Corpus](https://github.com/roneysco/Fake.br-Corpus) e salve em `Dataset/pre-processed.csv`.

3. **Textos originais** (necessários só para as opções 5 e 8 do menu). Baixe o repositório completo do Fake.br-Corpus (botão *Code → Download ZIP*), descompacte e coloque a pasta `Fake.br-Corpus-master` dentro de uma pasta `Dados`, **ao lado** da pasta do código, como na estrutura acima.

## Como rodar

O jeito mais simples é pelo menu:

```
python executar.py
```

| Opção | O que faz | Tempo aproximado |
|---|---|---|
| 1 | Treino e teste dos modelos (`main.py`) | 1 min |
| 2 | Validação cruzada de 5 partes (`avaliacao_cv.py`) | 3 min |
| 3 | SHAP (`shap_explicabilidade.py`) | 2 min |
| 4 | LIME (`lime_explicabilidade.py`) | 12 min |
| 5 | Texto completo x tamanho igualado (`tamanho_igualado.py`) | 5 min |
| 6 | **Rodar tudo na configuração do TCC** (opções 1 a 5, em ordem) | 25 min |
| 7 | Mostrar os resultados principais | — |
| 8 | SHAP com texto completo x tamanho igualado (`shap_tamanho_igualado.py`) | 5 min |
| 0 | Sair | — |

Os tempos dependem do computador.

Nas opções 3 e 4, o menu pergunta qual conjunto usar na visão global:

- **Amostra de 150 notícias** (padrão): amostra estratificada de 75 falsas e 75 verdadeiras do conjunto de teste. É a configuração usada no TCC.
- **Teste inteiro, 1.440 notícias:** verificação complementar. Leva cerca de 10 minutos no SHAP e cerca de 2 horas no LIME, e os resultados vão para pastas separadas (`shap_teste_inteiro/` e `lime_teste_inteiro/`).

Os scripts também podem ser rodados diretamente, por exemplo:

```
python main.py
python shap_explicabilidade.py completo    # SHAP do Random Forest no teste inteiro
python lime_explicabilidade.py completo    # LIME no teste inteiro
```

## Reprodutibilidade

- Todos os sorteios usam a semente 42. No mesmo ambiente, rodar de novo produz exatamente os mesmos números.
- Os números do TCC foram gerados com Python 3.13 e as versões do `requirements.txt`. Com outras versões do Python ou das bibliotecas, o Random Forest e a SVM podem dar resultados um pouco diferentes (a Regressão Logística não muda), sem alterar as conclusões.
- Para não misturar resultados de execuções diferentes, renomeie a pasta `resultados/` antes de rodar tudo de novo.
