import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import re
import os

# --- Configuração ---
ARQUIVO_CSV_ENTRADA = "resultados_com_scores_ragas copy.csv"
ARQUIVO_IMAGEM_SAIDA = "heatmap_correlacao_ragas.png"
# --------------------

def analisar_correlacoes(file_path):
    """
    Carrega o CSV de resultados, calcula a correlação entre as métricas Ragas
    e salva um heatmap.
    """
    if not os.path.exists(file_path):
        print(f"ERRO: Arquivo não encontrado no caminho: {file_path}")
        return

    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"ERRO ao ler o arquivo CSV: {e}")
        return

    # 1. Identifica automaticamente as colunas de métricas Ragas
    metricas_base = [
        'faithfulness', 
        'answer_relevancy', 
        'context_recall', 
        'context_precision', 
        'semantic_similarity' # Inclui a métrica que estava faltando no seu CSV de amostra
    ]
    
    # Encontra todas as colunas no CSV que *começam* com um dos nomes das métricas base
    colunas_scores = [col for col in df.columns for metrica in metricas_base if col.startswith(metrica)]

    if not colunas_scores:
        print(f"ERRO: Nenhuma coluna de score Ragas foi encontrada no arquivo.")
        print(f"O script procurou por colunas começando com: {metricas_base}")
        return

    print(f"Métricas Ragas encontradas para análise: {colunas_scores}")

    # 2. Prepara os dados
    # Converte colunas para numérico, tratando erros.
    for col in colunas_scores:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Remove linhas onde qualquer métrica Ragas esteja nula (NaN)
    # A correlação só pode ser calculada em dados completos
    df_scores = df.dropna(subset=colunas_scores)
    
    total_linhas_originais = len(df)
    total_linhas_analise = len(df_scores)
    
    if total_linhas_analise == 0:
        print("ERRO: Nenhum dado válido para calcular a correlação (todas as linhas continham NaN).")
        return
        
    print(f"Calculando correlações usando {total_linhas_analise} linhas completas (de {total_linhas_originais} totais).")

    # 3. Calcula e imprime a matriz de correlação
    correlation_matrix = df_scores[colunas_scores].corr()
    
    print("\n--- Matriz de Correlação (Valores de -1 a 1) ---")
    print(correlation_matrix.to_string())

    # 4. Gera o Heatmap (Gráfico)
    try:
        # Limpa os nomes das colunas para o gráfico (ex: 'faithfulness_gpt-4o' -> 'faithfulness')
        nomes_limpos_mapa = {col: re.match(r"^[a-zA-Z_]+", col).group(0) for col in colunas_scores}
        df_heatmap = df_scores[colunas_scores].rename(columns=nomes_limpos_mapa)
        
        # Recalcula a correlação com os nomes limpos para o gráfico
        correlation_matrix_clean = df_heatmap.corr()

        plt.figure(figsize=(12, 9))
        sns.heatmap(
            correlation_matrix_clean, 
            annot=True,     # Mostra os números nos quadrados
            cmap='coolwarm',# Esquema de cores: Vermelho (positivo), Azul (negativo)
            fmt='.2f',      # Formata os números para 2 casas decimais
            linewidths=.5,  # Linhas finas entre os quadrados
            vmin=-1,        # Define o limite mínimo do eixo de cor
            vmax=1          # Define o limite máximo do eixo de cor
        )
        plt.title('Heatmap de Correlação das Métricas Ragas', fontsize=16, pad=20)
        plt.xticks(rotation=45, ha='right', fontsize=10)
        plt.yticks(rotation=0, fontsize=10)
        plt.tight_layout() # Ajusta o layout para evitar cortes
        
        # Salva a imagem
        plt.savefig(ARQUIVO_IMAGEM_SAIDA)
        print(f"\n--- SUCESSO! ---")
        print(f"Um heatmap de correlação foi salvo como: {ARQUIVO_IMAGEM_SAIDA}")

    except Exception as e:
        print(f"\nAVISO: A matriz de correlação foi impressa, mas falhou ao gerar o gráfico de imagem.")
        print(f"Erro do gráfico: {e}")

# --- Ponto de Entrada Principal ---
if __name__ == "__main__":
    analisar_correlacoes(ARQUIVO_CSV_ENTRADA)