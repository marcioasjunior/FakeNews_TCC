import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lime.lime_text import LimeTextExplainer

from preparacao import (preparar_experimento, nome_arquivo, pegar_amostra,
                        PASTA_RESULTADOS, SEMENTE)


TESTE_INTEIRO = 'completo' in sys.argv

if TESTE_INTEIRO:
    pasta_saida = os.path.join(PASTA_RESULTADOS, 'lime_teste_inteiro')
else:
    pasta_saida = os.path.join(PASTA_RESULTADOS, 'lime')
pasta_global = os.path.join(pasta_saida, 'global')
pasta_local = os.path.join(pasta_saida, 'local')
os.makedirs(pasta_global, exist_ok=True)
os.makedirs(pasta_local, exist_ok=True)

N_PALAVRAS_EXPLICACAO = 15
N_PERTURBACOES = 5000
N_AMOSTRA_GLOBAL = 150
N_PALAVRAS_GRAFICO = 20
NOTICIA_EXEMPLO = 2693

COR_TRUE = '#4C72B0'
COR_FAKE = '#C44E52'


experimento = preparar_experimento()
textos_teste = experimento['textos_teste']
rotulos_teste = experimento['rotulos_teste']
matriz_teste = experimento['matriz_teste']
vetorizador = experimento['vetorizador']
modelos = experimento['modelos']

real = rotulos_teste.values
posicoes = np.arange(len(real))

previsoes = {}
for nome, modelo in modelos.items():
    previsoes[nome] = modelo.predict(matriz_teste)


# O LIME espera [P(fake), P(true)]. O SVM não tem probabilidade: a sigmoide da
# pontuação vira P(true) (mesma classe; pesos menores, comparar só ordem e direção).


def chances_svm(matriz):
    pontuacao = modelos['SVM'].decision_function(matriz)
    p_true = 1 / (1 + np.exp(-pontuacao))
    p_fake = 1 - p_true
    return np.column_stack((p_fake, p_true))


def prever_reglog(textos):
    return modelos['Regressão Logística'].predict_proba(vetorizador.transform(textos))


def prever_rf(textos):
    return modelos['Random Forest'].predict_proba(vetorizador.transform(textos))


def prever_SVM(textos):
    return chances_svm(vetorizador.transform(textos))


funcoes_prever = {
    'Regressão Logística': prever_reglog,
    'Random Forest':       prever_rf,
    'SVM':                 prever_SVM,
}

print("\nConvertendo a pontuação do SVM em probabilidade (sigmoide)...")
classe_sigmoide = np.where(chances_svm(matriz_teste)[:, 1] > 0.5, 'true', 'fake')
diferentes = (classe_sigmoide != previsoes['SVM']).sum()
print(f"  Notícias em que a classe difere do SVM original: {diferentes}")


def grafico_barras(palavras, pesos, titulo, arquivo):
    """Barras horizontais: azul = empurra para true, vermelho = empurra para fake."""
    palavras = list(palavras)[::-1]
    pesos = np.array(pesos)[::-1]
    cores = []
    for peso in pesos:
        if peso > 0:
            cores.append(COR_TRUE)
        else:
            cores.append(COR_FAKE)

    figura, eixo = plt.subplots(figsize=(8, 0.35 * len(palavras) + 1.5))
    eixo.barh(palavras, pesos, color=cores)
    eixo.axvline(0, color='black', linewidth=0.8)
    eixo.set_xlabel('Peso LIME  (← fake | true →)')
    eixo.set_title(titulo)
    plt.tight_layout()
    plt.savefig(arquivo, dpi=150, bbox_inches='tight')
    plt.close()


prob_reglog = modelos['Regressão Logística'].predict_proba(matriz_teste)[:, 1]
classe_reglog = previsoes['Regressão Logística']

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


# Explicador novo por notícia, com a mesma semente: o resultado não depende da ordem.


if TESTE_INTEIRO:
    amostra = posicoes
    para_explicar = posicoes
else:
    amostra = pegar_amostra(rotulos_teste, N_AMOSTRA_GLOBAL // 2, excluir=escolhidas)
    para_explicar = np.concatenate([list(escolhidas), amostra])

pd.DataFrame({
    'noticia': textos_teste.index[amostra],
    'real': real[amostra],
}).to_csv(os.path.join(pasta_global, 'amostra_global.csv'), index=False)

explicacoes = {}

for nome, prever in funcoes_prever.items():
    if TESTE_INTEIRO:
        print(f"\nCalculando LIME: {nome} (teste inteiro, {len(para_explicar)} notícias; demora bastante)")
    else:
        print(f"\nCalculando LIME: {nome} ({len(para_explicar)} notícias: {len(amostra)} da amostra "
              f"estratificada + {len(escolhidas)} escolhidas; pode levar alguns minutos)")
    explicacoes[nome] = {}
    contador = 0
    for posicao in para_explicar:
        explicador = LimeTextExplainer(class_names=['fake', 'true'], random_state=SEMENTE)
        explicacoes[nome][posicao] = explicador.explain_instance(
            textos_teste.iloc[posicao], prever,
            labels=(1,),   # classe true
            num_features=N_PALAVRAS_EXPLICACAO,
            num_samples=N_PERTURBACOES,
        )
        contador += 1
        if contador % 25 == 0:
            print(f"  {contador}/{len(para_explicar)}")


# Fidelidade = R² da regressão do LIME (perto de 1 = imitou bem o modelo).


linhas_locais = []
for posicao, motivo in escolhidas.items():
    numero = textos_teste.index[posicao]
    for nome in funcoes_prever:
        explicacao = explicacoes[nome][posicao]
        pesos_palavras = explicacao.as_list(label=1)
        prob_true = explicacao.predict_proba[1]
        fidelidade = explicacao.score
        arquivo = f'noticia_{numero}_{nome_arquivo(nome)}'

        palavras = []
        pesos = []
        for palavra, peso in pesos_palavras:
            palavras.append(palavra)
            pesos.append(peso)

        grafico_barras(
            palavras, pesos,
            f'Notícia {numero} ({motivo}) - {nome}\n'
            f'real: {real[posicao]} | previsto: {previsoes[nome][posicao]} | '
            f'P(true): {prob_true:.3f} | fidelidade: {fidelidade:.2f}',
            os.path.join(pasta_local, f'{arquivo}.png')
        )
        explicacao.save_to_file(os.path.join(pasta_local, f'{arquivo}.html'), labels=(1,))

        for palavra, peso in pesos_palavras:
            linhas_locais.append({
                'noticia': numero,
                'motivo': motivo,
                'real': real[posicao],
                'previsto': previsoes[nome][posicao],
                'modelo': nome,
                'prob_true': round(prob_true, 4),
                'fidelidade': round(fidelidade, 4),
                'palavra': palavra,
                'peso_lime': round(peso, 4),
                'empurra_para': 'true' if peso > 0 else 'fake',
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


# Importância = soma do |peso| da palavra nas explicações, dividida pelo número
# de notícias da amostra (notícia em que a palavra não apareceu conta como 0).
# Direção = sinal da média dos pesos nas notícias em que a palavra apareceu.

tabelasGlobais = {}

for nome in funcoes_prever:
    linhas = []
    for posicao in amostra:
        for palavra, peso in explicacoes[nome][posicao].as_list(label=1):
            linhas.append({'palavra': palavra, 'peso': peso})
    pesos = pd.DataFrame(linhas)
    pesos['peso_abs'] = pesos['peso'].abs()

    tabela = pesos.groupby('palavra').agg(
        soma_abs=('peso_abs', 'sum'),
        peso_medio_quando_aparece=('peso', 'mean'),
        noticias_em_que_aparece=('peso', 'size'),
    ).reset_index()
    tabela['importancia_media'] = tabela['soma_abs'] / len(amostra)
    tabela = tabela[['palavra', 'importancia_media', 'peso_medio_quando_aparece', 'noticias_em_que_aparece']].copy()
    tabela['direcao'] = np.where(tabela['peso_medio_quando_aparece'] > 0, 'true', 'fake')
    tabela = tabela.sort_values('importancia_media', ascending=False).reset_index(drop=True)
    tabela.insert(0, 'posicao', tabela.index + 1)
    tabelasGlobais[nome] = tabela

    arquivo = nome_arquivo(nome)
    tabela.head(100).to_csv(os.path.join(pasta_global, f'importancia_{arquivo}.csv'), index=False)

    topo = tabela.head(N_PALAVRAS_GRAFICO)
    sinal = np.where(topo['direcao'] == 'true', 1, -1)
    grafico_barras(
        topo['palavra'],
        sinal * topo['importancia_media'],
        f'Importância global LIME ({len(amostra)} notícias) - {nome}\n(azul = true, vermelho = fake)',
        os.path.join(pasta_global, f'barras_{arquivo}.png')
    )

    print(f"\nTop 10 palavras - {nome}:")
    for _, linha in tabela.head(10).iterrows():
        print(f"  {linha['posicao']:>2}. {linha['palavra']:<15} importância {linha['importancia_media']:.4f} -> {linha['direcao']}")

colunas = {}
for nome, tabela in tabelasGlobais.items():
    coluna = []
    for palavra, direcao in zip(tabela['palavra'].head(20), tabela['direcao'].head(20)):
        coluna.append(f"{palavra} ({direcao})")
    colunas[nome] = coluna
comparacao = pd.DataFrame(colunas)
comparacao.index = comparacao.index + 1
comparacao.to_csv(os.path.join(pasta_global, 'comparacao_top20_modelos.csv'), index_label='posicao')

print("\nPalavras em comum no top 20:")
nomes_modelos = list(tabelasGlobais)
for i in range(len(nomes_modelos)):
    for j in range(i + 1, len(nomes_modelos)):
        modelo_a, modelo_b = nomes_modelos[i], nomes_modelos[j]
        top20_a = set(tabelasGlobais[modelo_a]['palavra'].head(20))
        top20_b = set(tabelasGlobais[modelo_b]['palavra'].head(20))
        print(f"  {modelo_a} x {modelo_b}: {len(top20_a & top20_b)} de 20")


rotulos_amostra = real[amostra]
dados_fidelidade = []
for nome in funcoes_prever:
    fidelidades = []
    for posicao in amostra:
        fidelidades.append(explicacoes[nome][posicao].score)
    fidelidades = np.array(fidelidades)

    dados_fidelidade.append({
        'Modelo': nome,
        'Fidelidade média (todas)': round(fidelidades.mean(), 3),
        'Fidelidade média (fakes)': round(fidelidades[rotulos_amostra == 'fake'].mean(), 3),
        'Fidelidade média (trues)': round(fidelidades[rotulos_amostra == 'true'].mean(), 3),
    })
tabFidelidade = pd.DataFrame(dados_fidelidade)
tabFidelidade.to_csv(os.path.join(pasta_global, 'fidelidade_lime.csv'), index=False)
print("\nFidelidade do LIME (R² da regressão local; 1 = imita perfeitamente):")
print(tabFidelidade.to_string(index=False))

print(f"\nArquivos salvos em: {pasta_saida}")
print("  global/  -> gráficos de barras, importâncias, comparação e fidelidade")
print("  local/   -> gráficos e páginas HTML por notícia e modelo, contribuições e textos")
