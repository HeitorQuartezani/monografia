# ==================================================================================================
# SCRIPT DE ANÁLISE DE MÉTRICAS RAGAS (v2.3 - Final e Robusto)
#
# DESCRIÇÃO:
# - CORREÇÃO: Usa a coluna 'full_context_sent' para avaliações precisas.
# - CORREÇÃO: Adicionado tratamento para valores nulos (NaN) para evitar ValidationError.
# - CORREÇÃO: Atualizada a métrica 'answer_semantic_similarity' para 'answer_correctness'.
# - ROBUSTO: Adicionado mecanismo de retentativas (max_retries) para o LLM avaliador.
# - É incremental e configurável.
# ==================================================================================================
import os
import logging
import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_recall, context_precision, answer_correctness 
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from dotenv import load_dotenv

# --- 1. Configuração ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
load_dotenv()

# !!! IMPORTANTE: CONFIGURE AQUI O LLM QUE SERÁ O "JUIZ" DA AVALIAÇÃO !!!
# Recomenda-se usar um modelo poderoso como gpt-4o para avaliações mais confiáveis
EVALUATOR_LLM_DEPLOYMENT = os.getenv("AZURE_OPENAI_GPT4O_DEPLOYMENT") 
EVALUATOR_LLM_NAME = "gpt-4o"

ARQUIVO_DADOS = "resultados_brutos_experimento_v4.csv"

# --- 2. Função Principal de Análise ---

def analisar_resultados_com_ragas(caminho_arquivo: str, llm_juiz_deployment: str, llm_juiz_nome: str):
    if not os.path.exists(caminho_arquivo):
        logging.error(f"ERRO: Arquivo '{caminho_arquivo}' não encontrado.")
        return

    logging.info(f"Carregando resultados de '{caminho_arquivo}'...")
    df = pd.read_csv(caminho_arquivo)

    metricas_obj = [faithfulness, answer_relevancy, context_recall, context_precision, answer_correctness]
    nomes_colunas_score = [f"{m.name}_{llm_juiz_nome}" for m in metricas_obj]

    primeira_coluna_score = nomes_colunas_score[0]
    if primeira_coluna_score in df.columns:
        df_para_avaliar = df[df[primeira_coluna_score].isnull()].copy()
    else:
        df_para_avaliar = df.copy()

    if df_para_avaliar.empty:
        logging.info(f"Nenhuma linha nova para avaliar com o modelo '{llm_juiz_nome}'. Processo concluído.")
        return

    logging.info(f"Encontradas {len(df_para_avaliar)} linhas para avaliar com '{llm_juiz_nome}'.")

    azure_llm = AzureChatOpenAI(
        openai_api_version="2024-05-01-preview",
        azure_deployment=llm_juiz_deployment,
        api_key=os.getenv("AZURE_OPENAI_KEY"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        temperature=0,
        max_retries=3, # CORREÇÃO: Adiciona robustez contra falhas intermitentes da API
    )
    azure_embeddings = AzureOpenAIEmbeddings(
        openai_api_version="2023-05-15",
        azure_deployment=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT"),
        api_key=os.getenv("AZURE_OPENAI_KEY"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    )
    
    # CORREÇÃO: Trata valores ausentes (NaN) antes de passar para o Ragas
    colunas_texto_obrigatorias = ["question", "bot_answer", "expected_answer", "full_context_sent"]
    for col in colunas_texto_obrigatorias:
        if col not in df_para_avaliar.columns:
            logging.error(f"ERRO: A coluna obrigatória '{col}' não foi encontrada no CSV. Abortando.")
            return
        df_para_avaliar[col] = df_para_avaliar[col].fillna('')

    # Prepara os dados no formato que o Ragas espera
    dataset_dict = {
        "question": list(df_para_avaliar["question"]),
        "answer": list(df_para_avaliar["bot_answer"]),
        "contexts": [[str(ctx)] for ctx in df_para_avaliar["full_context_sent"]], # CORREÇÃO: Usa o contexto completo
        "ground_truth": list(df_para_avaliar["expected_answer"])
    }
    
    dataset = Dataset.from_dict(dataset_dict)

    logging.info("Iniciando a avaliação com Ragas... (Isso pode levar muito tempo)")
    resultado_avaliacao = evaluate(dataset, metrics=metricas_obj, llm=azure_llm, embeddings=azure_embeddings)
    df_scores = resultado_avaliacao.to_pandas()

    df_scores.rename(columns={m.name: f"{m.name}_{llm_juiz_nome}" for m in metricas_obj}, inplace=True)
    
    # Adiciona os scores de volta ao DataFrame original com segurança
    for col in nomes_colunas_score:
        if col not in df.columns:
            df[col] = pd.NA
        # Garante que os índices estão alinhados antes de atualizar
        df.loc[df_para_avaliar.index, col] = df_scores[col].values

    df.to_csv(caminho_arquivo, index=False, encoding='utf-8-sig')
    logging.info(f"--- SUCESSO! ---")
    logging.info(f"Arquivo '{caminho_arquivo}' foi atualizado com os scores do avaliador '{llm_juiz_nome}'.")


if __name__ == "__main__":
    analisar_resultados_com_ragas(ARQUIVO_DADOS, EVALUATOR_LLM_DEPLOYMENT, EVALUATOR_LLM_NAME)