import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score

from preparacao import (carregarDados, criarModelos, preparar_experimento,
                        MAX_PALAVRAS, SEMENTE, PASTA_RESULTADOS)

N_PARTES = 5

pasta_saida = os.path.join(PASTA_RESULTADOS, 'validacao_cruzada')
os.makedirs(pasta_saida, exist_ok=True)


dados = carregarDados()
textos, rotulos = dados['preprocessed_news'], dados['label']

divisor = StratifiedKFold(n_splits=N_PARTES, shuffle=True, random_state=SEMENTE)

metricas = {
    'Acurácia': 'accuracy',
    'Precisão': 'precision_weighted',
    'Recall':   'recall_weighted',
    'F1-Score': 'f1_weighted',
}


linhas_rodadas = []
dados_resumo = []

for nome, modelo in criarModelos().items():
    print(f"Validação cruzada: {nome} ({N_PARTES} rodadas)")

    # TF-IDF refeito em cada rodada (Pipeline).
    pipeline = make_pipeline(TfidfVectorizer(max_features=MAX_PALAVRAS), modelo)
    resultado = cross_validate(pipeline, textos, rotulos, cv=divisor, scoring=metricas)

    for rodada in range(N_PARTES):
        linha = {'Modelo': nome, 'Rodada': rodada + 1}
        for metrica in metricas:
            linha[metrica] = round(resultado[f'test_{metrica}'][rodada], 4)
        linhas_rodadas.append(linha)

    resumo = {'Modelo': nome}
    for metrica in metricas:
        valores = resultado[f'test_{metrica}']
        resumo[f'{metrica} média'] = round(valores.mean(), 4)
        resumo[f'{metrica} desvio'] = round(valores.std(), 4)
    resumo['F1 mínimo'] = round(resultado['test_F1-Score'].min(), 4)
    resumo['F1 máximo'] = round(resultado['test_F1-Score'].max(), 4)
    dados_resumo.append(resumo)

tabela_rodadas = pd.DataFrame(linhas_rodadas)
tabResumo = pd.DataFrame(dados_resumo)


print("\nCalculando o resultado da divisão única (mesma do main.py) para comparar...")
experimento = preparar_experimento(mostrar=False)
f1_unico = {}
for nome, modelo in experimento['modelos'].items():
    previsto = modelo.predict(experimento['matriz_teste'])
    f1_unico[nome] = round(f1_score(experimento['rotulos_teste'], previsto, average='weighted'), 4)

coluna_divisao_unica = []
for nome in tabResumo['Modelo']:
    coluna_divisao_unica.append(f1_unico[nome])
tabResumo.insert(1, 'F1 divisão única (main.py)', coluna_divisao_unica)

dados_texto = []
for _, linha in tabResumo.iterrows():
    dados_texto.append({
        'Modelo': linha['Modelo'],
        'F1 divisão única': f"{linha['F1 divisão única (main.py)']:.4f}",
        'Acurácia (média ± desvio)': f"{linha['Acurácia média']:.4f} ± {linha['Acurácia desvio']:.4f}",
        'F1 (média ± desvio)': f"{linha['F1-Score média']:.4f} ± {linha['F1-Score desvio']:.4f}",
        'F1 mín – máx': f"{linha['F1 mínimo']:.4f} – {linha['F1 máximo']:.4f}",
    })
tabTextos = pd.DataFrame(dados_texto)

tabela_rodadas.to_csv(os.path.join(pasta_saida, 'cv_por_rodada.csv'), index=False)
tabResumo.to_csv(os.path.join(pasta_saida, 'cv_resumo.csv'), index=False)
tabTextos.to_csv(os.path.join(pasta_saida, 'cv_tabela_para_texto.csv'), index=False)

print("\n" + "=" * 60)
print("VALIDAÇÃO CRUZADA - RESUMO")
print("=" * 60)
print(tabTextos.to_string(index=False))


figura, eixo = plt.subplots(figsize=(8, 5))
nomes = list(tabResumo['Modelo'])
for i, nome in enumerate(nomes):
    f1_das_rodadas = tabela_rodadas.loc[tabela_rodadas['Modelo'] == nome, 'F1-Score'].values
    media = f1_das_rodadas.mean()

    primeiro = (i == 0)
    eixo.scatter([i] * len(f1_das_rodadas), f1_das_rodadas, color='#4C72B0', alpha=0.7,
                 label='Rodadas' if primeiro else None)
    eixo.errorbar(i + 0.15, media, yerr=f1_das_rodadas.std(), fmt='o', color='#C44E52', capsize=6,
                  label='Média ± desvio' if primeiro else None)
    eixo.scatter(i - 0.15, f1_unico[nome], marker='D', color='#55A868',
                 label='Divisão única (main.py)' if primeiro else None)
eixo.set_xticks(range(len(nomes)))
eixo.set_xticklabels(nomes)
eixo.set_ylabel('F1-Score (weighted)')
eixo.set_title(f'Validação cruzada ({N_PARTES} partes)')
eixo.legend(loc='lower right')
eixo.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(pasta_saida, 'cv_f1_por_modelo.png'), dpi=150)
plt.close()

print(f"\nResultados salvos em: {pasta_saida}")
