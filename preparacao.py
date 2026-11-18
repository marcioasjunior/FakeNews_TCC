import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC


PASTA_DO_CODIGO = os.path.dirname(os.path.abspath(__file__))
CAMINHO_CSV = os.path.join(PASTA_DO_CODIGO, 'Dataset', 'pre-processed.csv')
PASTA_RESULTADOS = os.path.join(PASTA_DO_CODIGO, 'resultados')

TAMANHO_TESTE = 0.2
SEMENTE = 42
MAX_PALAVRAS = 10000


def carregarDados(caminho=CAMINHO_CSV):
    return pd.read_csv(caminho, index_col='index')


def separar_treino(dados):
    return train_test_split(
        dados['preprocessed_news'],
        dados['label'],
        test_size=TAMANHO_TESTE,
        random_state=SEMENTE,
        stratify=dados['label']
    )


def vet_textos(textos_treino, textos_teste):
    vetorizador = TfidfVectorizer(max_features=MAX_PALAVRAS)
    matriz_treino = vetorizador.fit_transform(textos_treino)
    matriz_teste = vetorizador.transform(textos_teste)
    return vetorizador, matriz_treino, matriz_teste


def criarModelos():
    return {
        'Regressão Logística': LogisticRegression(max_iter=1000, random_state=SEMENTE),
        'Random Forest':       RandomForestClassifier(n_estimators=100, random_state=SEMENTE),
        'SVM':                 LinearSVC(random_state=SEMENTE, max_iter=2000)
    }


def treinarModelos(modelos, matriz_treino, rotulos_treino, mostrar=True):
    for nome, modelo in modelos.items():
        if mostrar:
            print(f"Treinando: {nome}")
        modelo.fit(matriz_treino, rotulos_treino)
    return modelos


def preparar_experimento(mostrar=True):
    dados = carregarDados()
    if mostrar:
        print(f"\n--- Leitura concluída! {len(dados)} documentos. ---")
        print(dados['label'].value_counts().to_string())

    textos_treino, textos_teste, rotulos_treino, rotulos_teste = separar_treino(dados)
    if mostrar:
        print(f"\nTreino: {len(textos_treino)} amostras | Teste: {len(textos_teste)} amostras")

    vetorizador, matriz_treino, matriz_teste = vet_textos(textos_treino, textos_teste)
    if mostrar:
        print(f"Matriz TF-IDF de treino: {matriz_treino.shape}\n")

    modelos = treinarModelos(criarModelos(), matriz_treino, rotulos_treino, mostrar)

    return {
        'dados': dados,
        'textos_treino': textos_treino,
        'textos_teste': textos_teste,
        'rotulos_treino': rotulos_treino,
        'rotulos_teste': rotulos_teste,
        'vetorizador': vetorizador,
        'matriz_treino': matriz_treino,
        'matriz_teste': matriz_teste,
        'modelos': modelos,
    }


def pegar_amostra(rotulos_teste, n_por_classe, excluir=()):
    """Mesmo n_por_classe -> mesmas notícias no SHAP e no LIME."""
    sorteador = np.random.default_rng(SEMENTE)
    rotulos = np.asarray(rotulos_teste)
    posicoes = np.arange(len(rotulos))

    sorteadas = []
    for classe in ['fake', 'true']:
        indices_classe = posicoes[rotulos == classe]
        candidatas = np.setdiff1d(indices_classe, list(excluir))
        if n_por_classe > len(candidatas):
            raise ValueError(f"Só há {len(candidatas)} notícias '{classe}' disponíveis.")
        embaralhadas = sorteador.permutation(candidatas)
        sorteadas.append(embaralhadas[:n_por_classe])

    return np.concatenate(sorteadas)


def nome_arquivo(texto):
    trocas = {'ã': 'a', 'á': 'a', 'â': 'a', 'é': 'e', 'ê': 'e', 'í': 'i',
              'ó': 'o', 'ô': 'o', 'õ': 'o', 'ú': 'u', 'ç': 'c'}
    texto = texto.lower().replace(' ', '_')
    for letra_com_acento, letra_sem_acento in trocas.items():
        texto = texto.replace(letra_com_acento, letra_sem_acento)
    return texto

print(f"\n Preparação Realizada \n")
