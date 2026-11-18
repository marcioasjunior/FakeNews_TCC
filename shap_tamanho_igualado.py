import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from sklearn.metrics import f1_score

from preparacao import (vet_textos, criarModelos, treinarModelos, pegar_amostra,
                        nome_arquivo, PASTA_RESULTADOS, SEMENTE)
from corpus_original import carregar_pares, igualar_tamanho, separar_pares


pasta_saida = os.path.join(PASTA_RESULTADOS, 'shap_tamanho_igualado')
os.makedirs(pasta_saida, exist_ok=True)

N_PALAVRAS_GRAFICO = 20
N_AMOSTRA_RF = 150
N_REFERENCIA = 100


versoes = {
    'Texto completo': carregar_pares(),
    'Tamanho igualado': igualar_tamanho(),
}


def tabela_global(valores, X, nomes_palavras):
    importancia = np.abs(valores).mean(axis=0)
    presente = X > 0
    with np.errstate(invalid='ignore'):   # palavra sem ocorrência: divisão por 0
        media_presente = np.where(presente, valores, 0).sum(axis=0) / presente.sum(axis=0)
    tabela = pd.DataFrame({
        'palavra': nomes_palavras,
        'importancia_media': importancia,
        'shap_medio_quando_presente': media_presente,
        'noticias_com_a_palavra': presente.sum(axis=0),
    })
    tabela['direcao'] = np.where(tabela['shap_medio_quando_presente'] > 0, 'true', 'fake')
    tabela = tabela.sort_values('importancia_media', ascending=False).reset_index(drop=True)
    tabela.insert(0, 'posicao', tabela.index + 1)
    return tabela


def fracao_ausentes(valores, X, rotulos):
    total = np.abs(valores).sum(axis=1)
    ausente = np.abs(np.where(X == 0, valores, 0)).sum(axis=1)
    fracao = ausente / total
    return {
        'todas': round(fracao.mean(), 3),
        'fakes': round(fracao[rotulos == 'fake'].mean(), 3),
        'trues': round(fracao[rotulos == 'true'].mean(), 3),
    }


tabelas = {}

dados_ausentes = []
dados_f1 = []

for versao, dados in versoes.items():
    print(f"\n{'=' * 70}\n{versao}\n{'=' * 70}")
    textos_treino, textos_teste, rotulos_treino, rotulos_teste = separar_pares(dados)
    vetorizador, matriz_treino, matriz_teste = vet_textos(textos_treino, textos_teste)
    modelos = treinarModelos(criarModelos(), matriz_treino, rotulos_treino)
    nomes_palavras = vetorizador.get_feature_names_out()
    teste_denso = matriz_teste.toarray()
    real = rotulos_teste.values

    for nome, modelo in modelos.items():
        previsto = modelo.predict(matriz_teste)
        dados_f1.append({'Versão': versao, 'Modelo': nome,
                          'F1-Score': round(f1_score(rotulos_teste, previsto, average='weighted'), 4)})

    sorteador = np.random.default_rng(SEMENTE)
    referencias = sorteador.choice(matriz_treino.shape[0], N_REFERENCIA, replace=False)
    referencia = matriz_treino[referencias]

    resultados = {}
    for nome in ['Regressão Logística', 'SVM']:
        print(f"Calculando SHAP: {nome}")
        explicador = shap.LinearExplainer(modelos[nome], referencia)
        valores = explicador.shap_values(matriz_teste)
        if hasattr(valores, 'toarray'):
            valores = valores.toarray()
        else:
            valores = np.asarray(valores)
        resultados[nome] = (valores, teste_denso, real)

    amostra = pegar_amostra(rotulos_teste, N_AMOSTRA_RF // 2)
    print(f"Calculando SHAP: Random Forest ({len(amostra)} notícias estratificadas)")
    explicador_rf = shap.TreeExplainer(
        modelos['Random Forest'],
        data=referencia.toarray(),
        feature_perturbation='interventional',
        model_output='probability'
    )
    valores_rf = np.asarray(explicador_rf.shap_values(teste_denso[amostra], check_additivity=True))
    if valores_rf.shape[-1] == 2:
        valores_rf = valores_rf[:, :, 1]
    else:
        valores_rf = valores_rf[1]
    resultados['Random Forest'] = (valores_rf, teste_denso[amostra], real[amostra])

    for nome, (valores, X, rotulos) in resultados.items():
        tabela = tabela_global(valores, X, nomes_palavras)
        tabelas[(versao, nome)] = tabela
        arquivo = f"{nome_arquivo(versao)}_{nome_arquivo(nome)}"
        tabela.head(100).to_csv(os.path.join(pasta_saida, f'importancia_{arquivo}.csv'), index=False)

        fracoes = fracao_ausentes(valores, X, rotulos)
        dados_ausentes.append({
            'Versão': versao, 'Modelo': nome,
            'Fração média (todas)': fracoes['todas'],
            'Fração média (fakes)': fracoes['fakes'],
            'Fração média (trues)': fracoes['trues'],
        })

        shap.summary_plot(valores, X, feature_names=nomes_palavras, max_display=N_PALAVRAS_GRAFICO,
                          plot_type='bar', show=False)
        plt.title(f'Importância global (média do |SHAP|) - {nome}\n{versao}')
        plt.tight_layout()
        plt.savefig(os.path.join(pasta_saida, f'barras_{arquivo}.png'), dpi=150)
        plt.close()


nomes_modelos = list(criarModelos())

tabResultados_f1 = pd.DataFrame(dados_f1)
tabResultados_f1.to_csv(os.path.join(pasta_saida, 'f1_modelos_explicados.csv'), index=False)

tabAusentes = pd.DataFrame(dados_ausentes)
tabAusentes.to_csv(os.path.join(pasta_saida, 'fracao_palavras_ausentes.csv'), index=False)
print("\nFração do SHAP que vem de palavras AUSENTES:")
print(tabAusentes.to_string(index=False))

colunas = {}
for nome in nomes_modelos:
    for versao in versoes:
        tabela = tabelas[(versao, nome)]
        coluna = []
        for palavra, direcao in zip(tabela['palavra'].head(20), tabela['direcao'].head(20)):
            coluna.append(f"{palavra} ({direcao})")
        colunas[f"{nome} - {versao}"] = coluna
comparacao = pd.DataFrame(colunas)
comparacao.index = comparacao.index + 1
comparacao.to_csv(os.path.join(pasta_saida, 'comparacao_top20_completo_x_igualado.csv'), index_label='posicao')

dados_resumo = []
print("\nTop 20: texto completo x tamanho igualado")
for nome in nomes_modelos:
    top_completo = tabelas[('Texto completo', nome)].head(20)
    top_igualado = tabelas[('Tamanho igualado', nome)].head(20)
    palavras_completo = list(top_completo['palavra'])
    palavras_igualado = list(top_igualado['palavra'])
    em_comum = [p for p in palavras_completo if p in palavras_igualado]
    sairam = [p for p in palavras_completo if p not in palavras_igualado]
    entraram = [p for p in palavras_igualado if p not in palavras_completo]
    dados_resumo.append({
        'Modelo': nome,
        'Em comum no top 20': len(em_comum),
        'Saíram do top 20': ', '.join(sairam),
        'Entraram no top 20': ', '.join(entraram),
        'Palavras "fake" no top 20 (completo)': int((top_completo['direcao'] == 'fake').sum()),
        'Palavras "fake" no top 20 (igualado)': int((top_igualado['direcao'] == 'fake').sum()),
    })
    print(f"  {nome}: {len(em_comum)} de 20 em comum")
    print(f"    saíram:   {', '.join(sairam)}")
    print(f"    entraram: {', '.join(entraram)}")
pd.DataFrame(dados_resumo).to_csv(os.path.join(pasta_saida, 'resumo_top20.csv'), index=False)

print(f"\nArquivos salvos em: {pasta_saida}")
