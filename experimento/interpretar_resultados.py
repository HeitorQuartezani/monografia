# ==================================================================================================
# SCRIPT DE ANÁLISE E INTERPRETAÇÃO (v2.0 - Análise Focada e Flexível)
#
# DESCRIÇÃO:
# - Permite filtrar o conjunto de dados para realizar análises focadas e
#   comparações "2 a 2".
# - Permite selecionar facilmente qual conjunto de métricas (de qual LLM avaliador)
#   deve ser usado na análise.
# ==================================================================================================
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import logging

# --- 1. Configuração da Análise ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

ARQUIVO_ENTRADA = "resultados_brutos_experimento.csv"
PASTA_GRAFICOS = "analise_graficos_focada"

# !!! IMPORTANTE: CONFIGURE A SUA ANÁLISE AQUI !!!

# 1. Escolha o LLM avaliador cujas métricas você quer analisar
EVALUATOR_LLM_NAME = "gpt-4o-mini" 

# 2. Defina filtros para focar a análise. Deixe o dicionário vazio para analisar tudo.
# Exemplo 1: Analisar apenas o 'search_type' híbrido
# FILTROS = {"search_type": "hibrida"}
# Exemplo 2: Analisar apenas top_k = 5 e a estratégia de chunking de 1000
# FILTROS = {"top_k": 5, "chunking_strategy": "recursive_1000_200"}
FILTROS = {}

# 3. Defina as variáveis que você quer comparar nos gráficos e tabelas
VARIAVEIS_PARA_ANALISAR = [
    'chunking_strategy',
    'top_k',
    'search_type'
]

# --- 2. Lógica do Script ---

# Gera os nomes das colunas de métricas com base no LLM avaliador escolhido
METRICAS_RAGAS = [f"{m}_{EVALUATOR_LLM_NAME}" for m in ['faithfulness', 'answer_relevancy', 'context_recall', 'context_precision', 'answer_semantic_similarity']]

def aplicar_filtros(df: pd.DataFrame, filtros: dict) -> pd.DataFrame:
    """Aplica os filtros definidos ao DataFrame."""
    if not filtros:
        return df
    
    query_parts = []
    for key, value in filtros.items():
        if isinstance(value, str):
            query_parts.append(f"`{key}` == '{value}'")
        else:
            query_parts.append(f"`{key}` == {value}")
            
    query = " & ".join(query_parts)
    logging.info(f"Aplicando filtro: {query}")
    return df.query(query).copy()

def analisar_e_visualizar(df: pd.DataFrame):
    """Realiza a análise estatística e gera os gráficos para o DataFrame fornecido."""
    print("\n" + "="*80)
    print("== ANÁLISE DE DESEMPENHO MÉDIO (COM FILTROS APLICADOS) ==")
    print("="*80)

    for var in VARIAVEIS_PARA_ANALISAR:
        if var in FILTROS: continue # Não agrupar por uma variável que já foi filtrada
        print(f"\n--- Analisando por '{var}' ---")
        analise = df.groupby(var)[METRICAS_RAGAS].mean().sort_values(by=METRICAS_RAGAS[2], ascending=False)
        print(analise)

    if not os.path.exists(PASTA_GRAFICOS): os.makedirs(PASTA_GRAFICOS)
    sns.set_theme(style="whitegrid")

    for metrica in METRICAS_RAGAS:
        for var in VARIAVEIS_PARA_ANALISAR:
            if var in FILTROS: continue
            plt.figure(figsize=(10, 6))
            sns.boxplot(data=df, x=var, y=metrica)
            plt.title(f"Métrica '{metrica}' por '{var}'\nFiltros: {FILTROS if FILTROS else 'Nenhum'}")
            plt.tight_layout()
            plt.savefig(f"{PASTA_GRAFICOS}/{metrica}_por_{var}.png")
            plt.close()

    logging.info(f"Análise concluída. Gráficos salvos em '{PASTA_GRAFICOS}'.")


if __name__ == "__main__":
    if not os.path.exists(ARQUIVO_ENTRADA):
        logging.error(f"ERRO: Arquivo '{ARQUIVO_ENTRADA}' não encontrado.")
    else:
        df_completo = pd.read_csv(ARQUIVO_ENTRADA)
        
        # Verifica se as colunas de métricas necessárias existem
        if not all(col in df_completo.columns for col in METRICAS_RAGAS):
            logging.error(f"ERRO: As colunas de métricas para o avaliador '{EVALUATOR_LLM_NAME}' não foram encontradas.")
            logging.error("Execute o 'analise_metricas_v2.py' primeiro com essa configuração.")
        else:
            df_filtrado = aplicar_filtros(df_completo, FILTROS)
            if df_filtrado.empty:
                logging.warning("Nenhum dado encontrado após aplicar os filtros. Nenhuma análise será executada.")
            else:
                analisar_e_visualizar(df_filtrado)