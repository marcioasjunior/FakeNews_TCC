import os
import re
import unicodedata
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from preparacao import PASTA_DO_CODIGO, carregarDados, TAMANHO_TESTE, SEMENTE

PASTA_CORPUS = os.path.join(os.path.dirname(PASTA_DO_CODIGO), 'Dados', 'Fake.br-Corpus-master')
PASTA_TEXTOS_COMPLETOS = os.path.join(PASTA_CORPUS, 'full_texts')
PASTA_TAMANHO_IGUALADO = os.path.join(PASTA_CORPUS, 'size_normalized_texts')


def conferirPastas():
    for pasta in (PASTA_TEXTOS_COMPLETOS, PASTA_TAMANHO_IGUALADO):
        if not os.path.isdir(pasta):
            raise FileNotFoundError(f"Pasta do corpus não encontrada: {pasta}")


# Regras deduzidas comparando os textos completos com o pre-processed.csv:
# minúsculas; "R$" e "/" viram espaço; tira a pontuação das pontas; descarta a
# palavra com número ou hífen; tira acentos; tira stopwords do NLTK (sem acento).


STOPWORDS = set("""
a ao aos aquela aquelas aquele aqueles aquilo as com como da das de dela delas
dele deles depois do dos e ela elas ele eles em entre era eram essa essas esse
esses esta estamos estas estava estavam este esteja estejam estejamos estes
esteve estive estivemos estiver estivera estiveram estiverem estivermos
estivesse estivessem estou eu foi fomos for fora foram forem formos fosse
fossem fui haja hajam havemos hei houve houver houvera houveram houvesse
houvessem isso isto lhe lhes mais mas me mesmo meu meus minha minhas muito na
nas nem no nos nossa nossas nosso nossos num numa o os ou para pela pelas pelo
pelos por qual quando que quem se seja sejam sejamos sem serei seremos seria
seriam seu seus somos sou sua suas te tem temos tenha tenham tenhamos tenho
terei teremos teria teriam teu teus teve tinha tinham tive tivemos tiver tivera
tiveram tiverem tivermos tivesse tivessem tu tua tuas um uma vos
""".split())

PONTUACAO_DAS_PONTAS = '.,;:!?"\'()[]{}“”‘’«»…*–—'


def limpar_acentos(palavra):
    decomposta = unicodedata.normalize('NFKD', palavra)
    sem_acento = ''
    for letra in decomposta:
        if not unicodedata.combining(letra):
            sem_acento += letra
    return sem_acento


def preparar_texto(texto):
    texto = re.sub(r'[$/]', ' ', texto.lower())
    palavras = []
    for palavra in texto.split():
        palavra = palavra.strip(PONTUACAO_DAS_PONTAS)
        if not palavra.isalpha():
            continue
        palavra = limpar_acentos(palavra)
        if palavra in STOPWORDS:
            continue
        palavras.append(palavra)
    return ' '.join(palavras)


def lerTexto(caminho):
    with open(caminho, 'rb') as arquivo:
        conteudo = arquivo.read()
    try:
        return conteudo.decode('utf-8')
    except UnicodeDecodeError:
        return conteudo.decode('latin-1')


def numero_arquivo(nome):
    return int(os.path.splitext(nome)[0])


def listar_textos():
    """Ordem alfabética ("1.txt", "10.txt"...), a mesma das linhas do pre-processed.csv."""
    conferirPastas()
    arquivos = {}
    for classe in ('fake', 'true'):
        arquivos[classe] = sorted(os.listdir(os.path.join(PASTA_TEXTOS_COMPLETOS, classe)))
    return arquivos


def carregar_pares():
    dados = carregarDados()
    arquivos = listar_textos()
    pares = []
    for classe in ('fake', 'true'):
        for nome in arquivos[classe]:
            pares.append(numero_arquivo(nome))
    dados['par'] = pares
    return dados


def igualar_tamanho():
    conferirPastas()
    linhas = []
    for classe in ('fake', 'true'):
        pasta = os.path.join(PASTA_TAMANHO_IGUALADO, classe)
        for nome in sorted(os.listdir(pasta), key=numero_arquivo):
            texto = lerTexto(os.path.join(pasta, nome))
            linhas.append({
                'par': numero_arquivo(nome),
                'label': classe,
                'texto_original': texto,
                'preprocessed_news': preparar_texto(texto),
            })
    dados = pd.DataFrame(linhas)

    pares_fake = set(dados.loc[dados['label'] == 'fake', 'par'])
    pares_true = set(dados.loc[dados['label'] == 'true', 'par'])
    pares_completos = pares_fake & pares_true
    return dados[dados['par'].isin(pares_completos)].reset_index(drop=True)


def separar_pares(dados):
    """As duas notícias de um par vão juntas para o treino ou para o teste."""
    divisor = GroupShuffleSplit(n_splits=1, test_size=TAMANHO_TESTE, random_state=SEMENTE)
    posicoes_treino, posicoes_teste = next(divisor.split(dados, dados['label'], groups=dados['par']))

    pares_treino = set(dados['par'].iloc[posicoes_treino])
    pares_teste = set(dados['par'].iloc[posicoes_teste])
    if pares_treino & pares_teste:
        raise RuntimeError("Há pares divididos entre treino e teste.")

    textos_treino = dados['preprocessed_news'].iloc[posicoes_treino]
    textos_teste = dados['preprocessed_news'].iloc[posicoes_teste]
    rotulos_treino = dados['label'].iloc[posicoes_treino]
    rotulos_teste = dados['label'].iloc[posicoes_teste]
    return textos_treino, textos_teste, rotulos_treino, rotulos_teste
