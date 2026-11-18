import os
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from preparacao import preparar_experimento, nome_arquivo, PASTA_RESULTADOS

pasta_saida = os.path.join(PASTA_RESULTADOS, 'modelos')
os.makedirs(pasta_saida, exist_ok=True)


experimento = preparar_experimento()
modelos = experimento['modelos']
matriz_teste = experimento['matriz_teste']
rotulos_teste = experimento['rotulos_teste']


relatorios = {}

for nome, modelo in modelos.items():
    print('=' * 60)
    print(f"Avaliando: {nome}")
    print('=' * 60)

    previsto = modelo.predict(matriz_teste)

    relatorio = classification_report(rotulos_teste, previsto, output_dict=True)
    relatorios[nome] = relatorio
    print(classification_report(rotulos_teste, previsto))

    matriz_confusao = confusion_matrix(rotulos_teste, previsto, labels=['fake', 'true'])
    figura, eixo = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        matriz_confusao,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=['fake', 'true'],
        yticklabels=['fake', 'true'],
        ax=eixo
    )
    eixo.set_title(f'Matriz de Confusão\n{nome}')
    eixo.set_ylabel('Real')
    eixo.set_xlabel('Predito')
    plt.tight_layout()
    arquivo = nome_arquivo(nome)
    plt.savefig(os.path.join(pasta_saida, f'confusion_matrix_{arquivo}.png'), dpi=150)
    plt.close()
    print(f"Matriz de confusão salva: confusion_matrix_{arquivo}.png\n")


print('=' * 60)
print("COMPARATIVO ENTRE MODELOS")
print('=' * 60)

linhas = []
for nome, relatorio in relatorios.items():
    media_pond = relatorio['weighted avg']
    linhas.append({
        'Modelo':    nome,
        'Acurácia':  f"{relatorio['accuracy']:.4f}",
        'Precisão':  f"{media_pond['precision']:.4f}",
        'Recall':    f"{media_pond['recall']:.4f}",
        'F1-Score':  f"{media_pond['f1-score']:.4f}",
    })

tabela_comparativa = pd.DataFrame(linhas)
print(tabela_comparativa.to_string(index=False))

tabela_comparativa.to_csv(os.path.join(pasta_saida, 'resultados_modelos.csv'), index=False)
print(f"\nTabela comparativa salva em: {pasta_saida}")


nomes = []
valores_f1 = []
for nome, relatorio in relatorios.items():
    nomes.append(nome)
    valores_f1.append(float(f"{relatorio['weighted avg']['f1-score']:.4f}"))

figura, eixo = plt.subplots(figsize=(8, 5))
barras = eixo.bar(nomes, valores_f1, color=['#4C72B0', '#55A868', '#C44E52'])
eixo.set_ylim(0.8, 1.0)
eixo.set_ylabel('F1-Score (weighted avg)')
eixo.set_title('Comparativo de F1-Score entre os Modelos')
for barra, valor in zip(barras, valores_f1):
    eixo.text(barra.get_x() + barra.get_width() / 2, barra.get_height() + 0.002,
              f'{valor:.4f}', ha='center', va='bottom', fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(pasta_saida, 'comparativo_f1score.png'), dpi=150)
plt.close()
print("Gráfico comparativo salvo em: comparativo_f1score.png")
