import os
import sys
import time
import subprocess
import pandas as pd

PASTA_DO_CODIGO = os.path.dirname(os.path.abspath(__file__))
PASTA_RESULTADOS = os.path.join(PASTA_DO_CODIGO, 'resultados')


def executar(script, teste_inteiro=False):
    comando = [sys.executable, script]
    if teste_inteiro:
        comando.append('completo')

    print('\n' + '=' * 70)
    print(f"Rodando: {' '.join(comando[1:])}")
    print('=' * 70)
    inicio = time.time()
    resultado = subprocess.run(comando, cwd=PASTA_DO_CODIGO)
    minutos = (time.time() - inicio) / 60

    if resultado.returncode == 0:
        print(f"\n>> {script} terminou em {minutos:.1f} minutos.")
        return True
    print(f"\n>> {script} parou com erro (veja a mensagem acima).")
    return False


def ver_amostra(metodo, tempo_150, tempo_inteiro):
    print(f"\n{metodo}: qual conjunto de notícias usar na visão global?")
    print(f"  1 - Amostra de 150 notícias (configuração do TCC, {tempo_150})  [Enter]")
    print(f"  2 - Teste inteiro, 1.440 notícias ({tempo_inteiro}, pasta separada)")
    escolha = input("Escolha: ").strip()
    if escolha == '2':
        confirmar = input(f"Isso leva {tempo_inteiro}. Continuar? (s/n): ").strip().lower()
        if confirmar == 's':
            return True
        print("Usando a amostra de 150.")
    return False


def mostrarTabela(titulo, caminho):
    print(f"\n--- {titulo} ---")
    if os.path.exists(caminho):
        print(pd.read_csv(caminho).to_string(index=False))
    else:
        print("(ainda não gerado)")


def comparar_palavras(pasta_shap, arquivo_shap, pasta_lime):
    caminho_shap = os.path.join(PASTA_RESULTADOS, pasta_shap, 'global', arquivo_shap)
    caminho_lime = os.path.join(PASTA_RESULTADOS, pasta_lime, 'global', 'comparacao_top20_modelos.csv')
    print(f"\n--- Top 20: palavras em comum entre SHAP e LIME ({pasta_shap} x {pasta_lime}) ---")
    if not (os.path.exists(caminho_shap) and os.path.exists(caminho_lime)):
        print("(ainda não gerado)")
        return
    top_shap = pd.read_csv(caminho_shap)
    top_lime = pd.read_csv(caminho_lime)
    for modelo in ['Regressão Logística', 'Random Forest', 'SVM']:
        direcao_shap = {}
        for celula in top_shap[modelo]:
            palavra, direcao = celula.split(' (')
            direcao_shap[palavra] = direcao.rstrip(')')
        direcao_lime = {}
        for celula in top_lime[modelo]:
            palavra, direcao = celula.split(' (')
            direcao_lime[palavra] = direcao.rstrip(')')
        em_comum = set(direcao_shap) & set(direcao_lime)
        trocadas = [p for p in em_comum if direcao_shap[p] != direcao_lime[p]]
        print(f"  {modelo:<20} {len(em_comum)} de 20 em comum | direção diferente: {len(trocadas)}")


def mostrarResultados():
    r = PASTA_RESULTADOS
    mostrarTabela("Treino e teste (divisão 80/20)",
                   os.path.join(r, 'modelos', 'resultados_modelos.csv'))
    mostrarTabela("Validação cruzada (5 partes)",
                   os.path.join(r, 'validacao_cruzada', 'cv_tabela_para_texto.csv'))
    mostrarTabela("SHAP: fração da explicação que vem de palavras ausentes",
                   os.path.join(r, 'shap', 'global', 'fracao_shap_palavras_ausentes.csv'))
    mostrarTabela("LIME: fidelidade",
                   os.path.join(r, 'lime', 'global', 'fidelidade_lime.csv'))
    comparar_palavras('shap', 'comparacao_top20_modelos_amostra150.csv', 'lime')
    mostrarTabela("Tamanho igualado: texto completo x tamanho igualado",
                   os.path.join(r, 'tamanho_igualado', 'tabela_para_texto.csv'))

    if os.path.isdir(os.path.join(r, 'shap_tamanho_igualado')):
        mostrarTabela("[Extra] SHAP: palavras ausentes, texto completo x tamanho igualado",
                       os.path.join(r, 'shap_tamanho_igualado', 'fracao_palavras_ausentes.csv'))
        mostrarTabela("[Extra] SHAP: top 20, texto completo x tamanho igualado",
                       os.path.join(r, 'shap_tamanho_igualado', 'resumo_top20.csv'))
    if os.path.isdir(os.path.join(r, 'shap_teste_inteiro')):
        mostrarTabela("[Extra] SHAP no teste inteiro: palavras ausentes",
                       os.path.join(r, 'shap_teste_inteiro', 'global', 'fracao_shap_palavras_ausentes.csv'))
    if os.path.isdir(os.path.join(r, 'lime_teste_inteiro')):
        mostrarTabela("[Extra] LIME no teste inteiro: fidelidade",
                       os.path.join(r, 'lime_teste_inteiro', 'global', 'fidelidade_lime.csv'))
        comparar_palavras('shap_teste_inteiro', 'comparacao_top20_modelos.csv', 'lime_teste_inteiro')

    print(f"\nTodos os arquivos ficam em: {r}")


def rodar_tudo():
    print("\nRodando tudo na configuração do TCC (cerca de 25 minutos).")
    for script in ['main.py', 'avaliacao_cv.py', 'shap_explicabilidade.py',
                   'lime_explicabilidade.py', 'tamanho_igualado.py']:
        if not executar(script):
            print("\nParei aqui por causa do erro. As etapas seguintes não foram rodadas.")
            return
    print("\nTudo pronto.")
    mostrarResultados()


def menu():
    while True:
        print('\n' + '=' * 70)
        print("TCC FAKE NEWS - O QUE VOCÊ QUER RODAR?")
        print('=' * 70)
        print("  1 - Treino e teste dos modelos (main.py)            ~1 min")
        print("  2 - Validação cruzada, 5 partes (avaliacao_cv.py)   ~3 min")
        print("  3 - SHAP (shap_explicabilidade.py)                  ~2 min")
        print("  4 - LIME (lime_explicabilidade.py)                  ~12 min")
        print("  5 - Tamanho igualado (tamanho_igualado.py)          ~5 min")
        print("  6 - Rodar tudo na configuração do TCC (1 a 5)       ~25 min")
        print("  7 - Mostrar os resultados principais")
        print("  8 - SHAP: texto completo x tamanho igualado         ~5 min")
        print("  0 - Sair")
        print("(os tempos são aproximados e dependem do computador)")
        escolha = input("Escolha: ").strip()

        if escolha == '1':
            executar('main.py')
        elif escolha == '2':
            executar('avaliacao_cv.py')
        elif escolha == '3':
            teste_inteiro = ver_amostra('SHAP do Random Forest', '~2 min', 'cerca de 10 minutos')
            executar('shap_explicabilidade.py', teste_inteiro)
        elif escolha == '4':
            teste_inteiro = ver_amostra('LIME', '~12 min', 'cerca de 2 horas')
            executar('lime_explicabilidade.py', teste_inteiro)
        elif escolha == '5':
            executar('tamanho_igualado.py')
        elif escolha == '6':
            rodar_tudo()
        elif escolha == '7':
            mostrarResultados()
        elif escolha == '8':
            executar('shap_tamanho_igualado.py')
        elif escolha == '0':
            print("Até mais.")
            break
        else:
            print("Opção inválida. Digite um número de 0 a 8.")


if __name__ == '__main__':
    menu()
