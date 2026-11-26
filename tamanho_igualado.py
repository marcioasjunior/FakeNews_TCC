import os
import difflib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupKFold, cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression  
from sklearn.metrics import accuracy_score, f1_score

from preparacao import (vet_textos, criarModelos, treinarModelos, separar_treino,
                        SEMENTE, MAX_PALAVRAS, PASTA_RESULTADOS)
from corpus_original import (PASTA_TEXTOS_COMPLETOS, preparar_texto, lerTexto,
                             listar_textos, carregar_pares,
                             igualar_tamanho, separar_pares)


N_PARTES = 5
VALIDAR_PREPROCESSAMENTO = True

pasta_saida = os.path.join(PASTA_RESULTADOS, 'tamanho_igualado')
os.makedirs(pasta_saida, exist_ok=True)


# Confere o preparar_texto() contra o pre-processed.csv (difflib).


dados_completo = carregar_pares()
arquivos_completos = listar_textos()

if VALIDAR_PREPROCESSAMENTO:
    print("\nConferindo o pré-processamento contra o pre-processed.csv (cerca de 1 minuto)...")
    palavras_iguais, palavras_total, noticiasIguais = 0, 0, 0
    linha_do_csv = 0
    for classe in ('fake', 'true'):
        for nome in arquivos_completos[classe]:
            original = dados_completo['preprocessed_news'].iloc[linha_do_csv].split()
            nosso = preparar_texto(lerTexto(os.path.join(PASTA_TEXTOS_COMPLETOS, classe, nome))).split()
            comparador = difflib.SequenceMatcher(None, original, nosso, autojunk=False)
            for bloco in comparador.get_matching_blocks():
                palavras_iguais += bloco.size
            palavras_total += max(len(original), len(nosso))
            if original == nosso:
                noticiasIguais += 1
            linha_do_csv += 1

    concordancia = palavras_iguais / palavras_total
    texto_conferencia = (
        "Conferência do pré-processamento (full_texts -> preparar_texto() x pre-processed.csv)\n"
        f"  Palavras que coincidem, na mesma ordem: {concordancia:.2%}\n"
        f"  Notícias idênticas palavra por palavra: {noticiasIguais} de {len(dados_completo)}\n"
        "  As diferenças que sobram vêm de abreviações (ex.: 'Sr.', 'J.F.') que o\n"
        "  separador de palavras original tratou de outro jeito.\n"
    )
    print(texto_conferencia)
    with open(os.path.join(pasta_saida, 'conferencia_preprocessamento.txt'), 'w', encoding='utf-8') as arquivo:
        arquivo.write(texto_conferencia)


dados_igualado = igualar_tamanho()

print(f"\nTexto completo:   {len(dados_completo)} notícias, {dados_completo['par'].nunique()} pares")
print(f"Tamanho igualado: {len(dados_igualado)} notícias, {dados_igualado['par'].nunique()} pares")


palavras_completo_original = []
for classe in ('fake', 'true'):
    for nome in arquivos_completos[classe]:
        texto = lerTexto(os.path.join(PASTA_TEXTOS_COMPLETOS, classe, nome))
        palavras_completo_original.append(len(texto.split()))
dados_completo['palavras_original'] = palavras_completo_original
dados_completo['palavras_preprocessado'] = dados_completo['preprocessed_news'].str.split().str.len()
dados_igualado['palavras_original'] = dados_igualado['texto_original'].str.split().str.len()
dados_igualado['palavras_preprocessado'] = dados_igualado['preprocessed_news'].str.split().str.len()

versoes = (('Texto completo', dados_completo), ('Tamanho igualado', dados_igualado))

dados_tamanho = []
for versao, dados in versoes:
    for classe in ('fake', 'true'):
        da_classe = dados[dados['label'] == classe]
        dados_tamanho.append({
            'Versão': versao,
            'Classe': classe,
            'Palavras no texto original (média)': round(da_classe['palavras_original'].mean(), 1),
            'Palavras no texto original (mediana)': int(da_classe['palavras_original'].median()),
            'Palavras após pré-processamento (média)': round(da_classe['palavras_preprocessado'].mean(), 1),
            'Palavras após pré-processamento (mediana)': int(da_classe['palavras_preprocessado'].median()),
        })
tabTamanho = pd.DataFrame(dados_tamanho)
tabTamanho.to_csv(os.path.join(pasta_saida, 'tamanho_dos_textos.csv'), index=False)
print("\nTamanho dos textos:")
print(tabTamanho.to_string(index=False))


dados_divisao = []
for versao, dados in versoes:
    textos_treino, textos_teste, rotulos_treino, rotulos_teste = separar_pares(dados)

    print(f"\n{versao}: treino {len(textos_treino)} | teste {len(textos_teste)} "
          f"({(rotulos_teste == 'fake').sum()} fake / {(rotulos_teste == 'true').sum()} true)")

    vetorizador, matriz_treino, matriz_teste = vet_textos(textos_treino, textos_teste)
    modelos = treinarModelos(criarModelos(), matriz_treino, rotulos_treino, mostrar=False)
    for nome, modelo in modelos.items():
        previsto = modelo.predict(matriz_teste)
        dados_divisao.append({
            'Versão': versao,
            'Modelo': nome,
            'Acurácia': round(accuracy_score(rotulos_teste, previsto), 4),
            'F1-Score': round(f1_score(rotulos_teste, previsto, average='weighted'), 4),
        })

resumo_divisao = pd.DataFrame(dados_divisao)
resumo_divisao.to_csv(os.path.join(pasta_saida, 'divisao_unica_por_pares.csv'), index=False)


# TF-IDF refeito em cada rodada (Pipeline).


metricas = {'Acurácia': 'accuracy', 'F1-Score': 'f1_weighted'}
linhas_rodadas, dados_resumo = [], []

for versao, dados in versoes:
    divisor = GroupKFold(n_splits=N_PARTES, shuffle=True, random_state=SEMENTE)
    for nome, modelo in criarModelos().items():
        print(f"Validação cruzada por pares: {versao} - {nome} ({N_PARTES} rodadas)")
        pipeline = make_pipeline(TfidfVectorizer(max_features=MAX_PALAVRAS), modelo)
        resultado = cross_validate(pipeline, dados['preprocessed_news'], dados['label'],
                                   groups=dados['par'], cv=divisor, scoring=metricas)
        for rodada in range(N_PARTES):
            linhas_rodadas.append({
                'Versão': versao, 'Modelo': nome, 'Rodada': rodada + 1,
                'Acurácia': round(resultado['test_Acurácia'][rodada], 4),
                'F1-Score': round(resultado['test_F1-Score'][rodada], 4),
            })
        f1_das_rodadas = resultado['test_F1-Score']
        dados_resumo.append({
            'Versão': versao, 'Modelo': nome,
            'Acurácia média': round(resultado['test_Acurácia'].mean(), 4),
            'Acurácia desvio': round(resultado['test_Acurácia'].std(), 4),
            'F1 média': round(f1_das_rodadas.mean(), 4),
            'F1 desvio': round(f1_das_rodadas.std(), 4),
            'F1 mínimo': round(f1_das_rodadas.min(), 4),
            'F1 máximo': round(f1_das_rodadas.max(), 4),
        })

tabela_rodadas = pd.DataFrame(linhas_rodadas)
tabResumo = pd.DataFrame(dados_resumo)
tabela_rodadas.to_csv(os.path.join(pasta_saida, 'cv_por_pares_por_rodada.csv'), index=False)
tabResumo.to_csv(os.path.join(pasta_saida, 'cv_por_pares_resumo.csv'), index=False)


####-----------------------------------------------------------------------------------------------------
####Foi implementado para exemplificar como ficaria o treino apenas contando a qtd de palavras

treino_texto, teste_texto, treino_rotulo, teste_rotulo = separar_treino(dados_completo)

def log_palavras(textos):
    return np.log(textos.str.split().str.len().values).reshape(-1, 1)

modelo_tamanho = LogisticRegression().fit(log_palavras(treino_texto), treino_rotulo)
previsto_tamanho = modelo_tamanho.predict(log_palavras(teste_texto))
corte = np.exp(-modelo_tamanho.intercept_[0] / modelo_tamanho.coef_[0][0])

tabSoTamanho = pd.DataFrame([{
    'Acurácia': round(accuracy_score(teste_rotulo, previsto_tamanho), 4),
    'F1': round(f1_score(teste_rotulo, previsto_tamanho, average='weighted'), 4),
    'Corte (palavras)': round(corte),
}])
tabSoTamanho.to_csv(os.path.join(pasta_saida, 'acerto_so_tamanho.csv'), index=False)
print("\nSó o tamanho (Regressão Logística com o log do número de palavras):")
print(tabSoTamanho.to_string(index=False))
print(f"  Abaixo de {corte:.0f} palavras -> fake; acima -> true")
####-----------------------------------------------------------------------------------------------------

def buscar(tabela, versao, nome, coluna):
    linha = tabela[(tabela['Versão'] == versao) & (tabela['Modelo'] == nome)]
    return linha[coluna].iloc[0]


nomes_modelos = list(criarModelos())
dados_texto = []
for nome in nomes_modelos:
    media_completo = buscar(tabResumo, 'Texto completo', nome, 'F1 média')
    desvio_completo = buscar(tabResumo, 'Texto completo', nome, 'F1 desvio')
    media_igualado = buscar(tabResumo, 'Tamanho igualado', nome, 'F1 média')
    desvio_igualado = buscar(tabResumo, 'Tamanho igualado', nome, 'F1 desvio')
    dados_texto.append({
        'Modelo': nome,
        'F1 completo (divisão única)': f"{buscar(resumo_divisao, 'Texto completo', nome, 'F1-Score'):.4f}",
        'F1 igualado (divisão única)': f"{buscar(resumo_divisao, 'Tamanho igualado', nome, 'F1-Score'):.4f}",
        'F1 completo (CV, média ± desvio)': f"{media_completo:.4f} ± {desvio_completo:.4f}",
        'F1 igualado (CV, média ± desvio)': f"{media_igualado:.4f} ± {desvio_igualado:.4f}",
        'Queda do F1 (CV, pontos percentuais)': f"{100 * (media_completo - media_igualado):.1f}",
    })
tabTextos = pd.DataFrame(dados_texto)
tabTextos.to_csv(os.path.join(pasta_saida, 'tabela_para_texto.csv'), index=False)

print("\n" + "=" * 70)
print("TEXTO COMPLETO x TAMANHO IGUALADO (divisão por pares)")
print("=" * 70)
print(tabTextos.to_string(index=False))

figura, eixo = plt.subplots(figsize=(8, 5))
largura = 0.35
posicoes_x = np.arange(len(nomes_modelos))
cores = {'Texto completo': '#4C72B0', 'Tamanho igualado': '#C44E52'}
for i, versao in enumerate(cores):
    medias, desvios = [], []
    for nome in nomes_modelos:
        medias.append(buscar(tabResumo, versao, nome, 'F1 média'))
        desvios.append(buscar(tabResumo, versao, nome, 'F1 desvio'))
    barras = eixo.bar(posicoes_x + (i - 0.5) * largura, medias, largura, yerr=desvios,
                      capsize=5, color=cores[versao], label=versao)
    for barra, media in zip(barras, medias):
        eixo.text(barra.get_x() + barra.get_width() / 2, media + 0.01, f'{media:.3f}',
                  ha='center', fontsize=9)
eixo.set_xticks(posicoes_x)
eixo.set_xticklabels(nomes_modelos)
eixo.set_ylim(0.5, 1.0)
eixo.set_ylabel('F1-Score (weighted), média de 5 rodadas')
eixo.set_title('Texto completo x tamanho igualado (validação cruzada por pares)')
eixo.legend(loc='lower right')
eixo.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(pasta_saida, 'f1_completo_x_igualado.png'), dpi=150)
plt.close()

print(f"\nResultados salvos em: {pasta_saida}")
