import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

from preparacao import (preparar_experimento, nome_arquivo, pegar_amostra,
                        PASTA_RESULTADOS, SEMENTE)


TESTE_INTEIRO = 'completo' in sys.argv

if TESTE_INTEIRO:
    pasta_saida = os.path.join(PASTA_RESULTADOS, 'shap_teste_inteiro')
else:
    pasta_saida = os.path.join(PASTA_RESULTADOS, 'shap')
pasta_global = os.path.join(pasta_saida, 'global')
pasta_local = os.path.join(pasta_saida, 'local')
os.makedirs(pasta_global, exist_ok=True)
os.makedirs(pasta_local, exist_ok=True)

N_PALAVRAS_GRAFICO = 20
N_AMOSTRA_RF = 150
N_REFERENCIA = 100
NOTICIA_EXEMPLO = 2693


experimento = preparar_experimento()
textos_teste = experimento['textos_teste']
rotulos_teste = experimento['rotulos_teste']
matriz_treino = experimento['matriz_treino']
matriz_teste = experimento['matriz_teste']
modelos = experimento['modelos']

nomes_palavras = experimento['vetorizador'].get_feature_names_out()

teste_denso = matriz_teste.toarray()

previsoes = {}
for nome, modelo in modelos.items():
    previsoes[nome] = modelo.predict(matriz_teste)


prob_reglog = modelos['Regressão Logística'].predict_proba(matriz_teste)[:, 1]
real = rotulos_teste.values
classe_reglog = previsoes['Regressão Logística']
posicoes = np.arange(len(real))

escolhidas = {}

acertos_fake = posicoes[(real == 'fake') & (classe_reglog == 'fake')]
mais_certa = acertos_fake[np.argmin(prob_reglog[acertos_fake])]
escolhidas[mais_certa] = 'fake_acerto_com_certeza'

acertos_true = posicoes[(real == 'true') & (classe_reglog == 'true')]
mais_certa = acertos_true[np.argmax(prob_reglog[acertos_true])]
escolhidas[mais_certa] = 'true_acerto_com_certeza'

erros_fake = posicoes[(real == 'fake') & (classe_reglog == 'true')]
ordem = np.argsort(prob_reglog[erros_fake])[::-1]
for i, posicao in enumerate(erros_fake[ordem][:2]):
    escolhidas[posicao] = f'fake_que_escapou_{i + 1}'

erros_true = posicoes[(real == 'true') & (classe_reglog == 'fake')]
ordem = np.argsort(prob_reglog[erros_true])
for i, posicao in enumerate(erros_true[ordem][:2]):
    escolhidas[posicao] = f'true_acusada_de_fake_{i + 1}'

if NOTICIA_EXEMPLO in textos_teste.index:
    escolhidas[list(textos_teste.index).index(NOTICIA_EXEMPLO)] = 'exemplo_2693'

print("\nNotícias escolhidas para a análise local:")
for posicao, motivo in escolhidas.items():
    print(f"  nº {textos_teste.index[posicao]:>5} | real: {real[posicao]:<4} | "
          f"P(true) RegLog: {prob_reglog[posicao]:.3f} | {motivo}")


# RegLog e SVM: LinearExplainer no teste inteiro (saída em pontuação).
# Random Forest: TreeExplainer na amostra de 150 + escolhidas (saída em P(true)).


sorteador = np.random.default_rng(SEMENTE)
referencias = sorteador.choice(matriz_treino.shape[0], N_REFERENCIA, replace=False)
referencia = matriz_treino[referencias]

shap_resultados = {}

for nome in ['Regressão Logística', 'SVM']:
    print(f"\nCalculando SHAP: {nome}")
    explicador = shap.LinearExplainer(modelos[nome], referencia)
    valores = explicador.shap_values(matriz_teste)
    if hasattr(valores, 'toarray'):
        valores = valores.toarray()
    else:
        valores = np.asarray(valores)
    shap_resultados[nome] = {
        'valores': valores,
        'base': float(np.ravel(explicador.expected_value)[0]),
        'X': teste_denso,
        'posicoes': posicoes,
        'na_global': np.ones(len(posicoes), dtype=bool),
    }

if TESTE_INTEIRO:
    amostra_global_rf = posicoes
    amostra_rf = posicoes
else:
    amostra_global_rf = pegar_amostra(rotulos_teste, N_AMOSTRA_RF // 2, excluir=escolhidas)
    amostra_rf = np.concatenate([list(escolhidas), amostra_global_rf])

pd.DataFrame({
    'noticia': textos_teste.index[amostra_global_rf],
    'real': real[amostra_global_rf],
}).to_csv(os.path.join(pasta_global, 'amostra_global_rf.csv'), index=False)

if TESTE_INTEIRO:
    print(f"\nCalculando SHAP: Random Forest (teste inteiro, {len(amostra_rf)} notícias; leva alguns minutos)")
else:
    print(f"\nCalculando SHAP: Random Forest ({len(amostra_rf)} notícias: {len(amostra_global_rf)} da "
          f"amostra estratificada + {len(escolhidas)} escolhidas)")
explicador_rf = shap.TreeExplainer(
    modelos['Random Forest'],
    data=referencia.toarray(),
    feature_perturbation='interventional',   # o modo padrão dava erro no Windows
    model_output='probability'
)
valores_rf = np.asarray(explicador_rf.shap_values(teste_denso[amostra_rf], check_additivity=True))
# Conforme a versão do SHAP, o formato muda; ficamos com a classe "true" (índice 1)
if valores_rf.shape[-1] == 2:
    valores_rf = valores_rf[:, :, 1]
else:
    valores_rf = valores_rf[1]

if TESTE_INTEIRO:
    na_global_rf = np.ones(len(amostra_rf), dtype=bool)
else:
    na_global_rf = ~np.isin(amostra_rf, list(escolhidas))
shap_resultados['Random Forest'] = {
    'valores': valores_rf,
    'base': float(np.ravel(explicador_rf.expected_value)[1]),
    'X': teste_denso[amostra_rf],
    'posicoes': amostra_rf,
    'na_global': na_global_rf,
}


# Importância = média do |SHAP| da palavra nas notícias.
# Direção = sinal da média do SHAP nas notícias que TÊM a palavra.

def tabela_global(valores, X):
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


def salvar_compTop20(tabelas, arquivo):
    colunas = {}
    for nome, tabela in tabelas.items():
        coluna = []
        for palavra, direcao in zip(tabela['palavra'].head(20), tabela['direcao'].head(20)):
            coluna.append(f"{palavra} ({direcao})")
        colunas[nome] = coluna
    comparacao = pd.DataFrame(colunas)
    comparacao.index = comparacao.index + 1
    comparacao.to_csv(os.path.join(pasta_global, arquivo), index_label='posicao')


tabelasGlobais = {}

for nome, resultado in shap_resultados.items():
    linhas_globais = resultado['na_global']
    valores = resultado['valores'][linhas_globais]
    X = resultado['X'][linhas_globais]
    tabela = tabela_global(valores, X)
    tabelasGlobais[nome] = tabela

    arquivo = nome_arquivo(nome)
    tabela.head(100).to_csv(os.path.join(pasta_global, f'importancia_{arquivo}.csv'), index=False)

    shap.summary_plot(valores, X, feature_names=nomes_palavras, max_display=N_PALAVRAS_GRAFICO,
                      plot_type='bar', show=False)
    plt.title(f'Importância global (média do |SHAP|) - {nome}')
    plt.tight_layout()
    plt.savefig(os.path.join(pasta_global, f'barras_{arquivo}.png'), dpi=150)
    plt.close()

    shap.summary_plot(valores, X, feature_names=nomes_palavras, max_display=N_PALAVRAS_GRAFICO, show=False)
    plt.title(f'Impacto de cada palavra por notícia - {nome}\n(direita = true, esquerda = fake)')
    plt.tight_layout()
    plt.savefig(os.path.join(pasta_global, f'beeswarm_{arquivo}.png'), dpi=150)
    plt.close()

    print(f"\nTop 10 palavras - {nome}:")
    for _, linha in tabela.head(10).iterrows():
        print(f"  {linha['posicao']:>2}. {linha['palavra']:<15} importância {linha['importancia_media']:.4f} -> {linha['direcao']}")

salvar_compTop20(tabelasGlobais, 'comparacao_top20_modelos.csv')

print("\nPalavras em comum no top 20:")
nomes_modelos = list(tabelasGlobais)
for i in range(len(nomes_modelos)):
    for j in range(i + 1, len(nomes_modelos)):
        modelo_a, modelo_b = nomes_modelos[i], nomes_modelos[j]
        top20_a = set(tabelasGlobais[modelo_a]['palavra'].head(20))
        top20_b = set(tabelasGlobais[modelo_b]['palavra'].head(20))
        print(f"  {modelo_a} x {modelo_b}: {len(top20_a & top20_b)} de 20")

# RegLog e SVM nas mesmas 150 notícias do RF, só para comparar com o LIME
# (não roda no modo completo).

if not TESTE_INTEIRO:
    tabelas_amostra = {}
    for nome, resultado in shap_resultados.items():
        dados_amostra = np.isin(resultado['posicoes'], amostra_global_rf)
        tabelas_amostra[nome] = tabela_global(resultado['valores'][dados_amostra],
                                                     resultado['X'][dados_amostra])
        if nome != 'Random Forest':
            arquivo = f'importancia_{nome_arquivo(nome)}_amostra{N_AMOSTRA_RF}.csv'
            tabelas_amostra[nome].head(100).to_csv(os.path.join(pasta_global, arquivo), index=False)

    salvar_compTop20(tabelas_amostra, f'comparacao_top20_modelos_amostra{N_AMOSTRA_RF}.csv')
    print(f"\nImportância da RL e do SVM nas {N_AMOSTRA_RF} notícias da amostra salva (para comparar com o LIME).")


# Parte do |SHAP| que vem de palavras AUSENTES na notícia.


dados_ausentes = []
for nome, resultado in shap_resultados.items():
    linhas_globais = resultado['na_global']
    valores = resultado['valores'][linhas_globais]
    X = resultado['X'][linhas_globais]
    rotulos = real[resultado['posicoes'][linhas_globais]]

    total = np.abs(valores).sum(axis=1)
    ausente = np.abs(np.where(X == 0, valores, 0)).sum(axis=1)
    fracao = ausente / total
    dados_ausentes.append({
        'Modelo': nome,
        'Fração média (todas)': round(fracao.mean(), 3),
        'Fração média (fakes)': round(fracao[rotulos == 'fake'].mean(), 3),
        'Fração média (trues)': round(fracao[rotulos == 'true'].mean(), 3),
    })
tabAusentes = pd.DataFrame(dados_ausentes)
tabAusentes.to_csv(os.path.join(pasta_global, 'fracao_shap_palavras_ausentes.csv'), index=False)
print("\nFração do SHAP que vem de palavras AUSENTES na notícia:")
print(tabAusentes.to_string(index=False))


linhas_locais = []
for posicao, motivo in escolhidas.items():
    numero = textos_teste.index[posicao]
    for nome, resultado in shap_resultados.items():
        linha_encontrada = np.where(resultado['posicoes'] == posicao)[0]
        if len(linha_encontrada) == 0:
            continue
        linha = linha_encontrada[0]
        val_noticia = resultado['valores'][linha]
        tfidf_noticia = resultado['X'][linha]

        explicacao = shap.Explanation(
            values=val_noticia,
            base_values=resultado['base'],
            data=tfidf_noticia,
            feature_names=list(nomes_palavras),
        )
        shap.plots.waterfall(explicacao, max_display=15, show=False)
        plt.title(f'Notícia {numero} ({motivo}) - {nome}\nreal: {real[posicao]} | previsto: {previsoes[nome][posicao]}')
        plt.tight_layout()
        plt.savefig(os.path.join(pasta_local, f'noticia_{numero}_{nome_arquivo(nome)}.png'),
                    dpi=150, bbox_inches='tight')
        plt.close()

        ordem = np.argsort(np.abs(val_noticia))[::-1]
        for indice_palavra in ordem[:15]:
            linhas_locais.append({
                'noticia': numero,
                'motivo': motivo,
                'real': real[posicao],
                'previsto': previsoes[nome][posicao],
                'modelo': nome,
                'palavra': nomes_palavras[indice_palavra],
                'valor_tfidf': round(tfidf_noticia[indice_palavra], 4),
                'presente_na_noticia': 'sim' if tfidf_noticia[indice_palavra] > 0 else 'não',
                'shap': round(val_noticia[indice_palavra], 4),
                'empurra_para': 'true' if val_noticia[indice_palavra] > 0 else 'fake',
            })

tabela_locais = pd.DataFrame(linhas_locais)
tabela_locais.to_csv(os.path.join(pasta_local, 'contribuicoes_noticias_escolhidas.csv'), index=False)

linhas_textos = []
for posicao, motivo in escolhidas.items():
    linhas_textos.append({
        'noticia': textos_teste.index[posicao],
        'motivo': motivo,
        'real': real[posicao],
        'texto': textos_teste.iloc[posicao],
    })
pd.DataFrame(linhas_textos).to_csv(os.path.join(pasta_local, 'textos_noticias_escolhidas.csv'), index=False)

print(f"\nArquivos salvos em: {pasta_saida}")
print("  global/  -> gráficos de barras e beeswarm, importâncias, comparação e palavras ausentes")
print("  local/   -> cascatas (waterfall) por notícia e modelo, contribuições e textos")
