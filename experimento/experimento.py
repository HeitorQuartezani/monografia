# ==================================================================================================
# SCRIPT DE EXECUÇÃO DE EXPERIMENTO (v4.1 - Salva Contexto Completo)
#
# DESCRIÇÃO:
# - ADICIONADO: Salva o texto completo do contexto na coluna 'full_context_sent' para
#   permitir o cálculo correto de todas as métricas Ragas (especialmente faithfulness).
# - REFEITO: Script reestruturado para usar uma API de duas fases (Recuperação -> Geração).
# - OTIMIZADO: Implementa a busca com 'max_k' e simula 'top_k'.
# - ROBUSTO: Ambas as fases são resumíveis.
# ==================================================================================================
import json
import time
import pandas as pd
from itertools import product
from tqdm import tqdm
import requests
import os
import csv

# --- 1. Configuração do Experimento ---
API_URL_BASE = "http://127.0.0.1:8000"
API_URL_RECUPERACAO = f"{API_URL_BASE}/recuperar_contexto"
API_URL_GERACAO = f"{API_URL_BASE}/gerar_resposta"

ARQUIVO_GABARITO = "experimento/perguntas.json"
ARQUIVO_INTERMEDIARIO_JSON = "resultados_recuperacao.json"
ARQUIVO_SAIDA_CSV = "resultados_brutos_experimento_v4.csv"

# Parâmetros que afetam a fase de RECUPERAÇÃO
PARAMETROS_RECUPERACAO = {
    "chunking_strategy": ["recursive_1000_200", "recursive_500_100", "semantic_percentile_95"],
    "search_type": ["vetorial", "textual", "hibrida"],
}

# Parâmetros que afetam a fase de GERAÇÃO
PARAMETROS_GERACAO = {
    "model": ["gpt-4o-mini", "gpt-4o"],
    "temperature": [0.1, 0.2]
}

# Valores de top_k a serem testados (simulados na Fase 2)
TOP_K_VALORES = [5, 10, 15]
MAX_K = max(TOP_K_VALORES) # O valor máximo que a API de recuperação buscará

# Parâmetros FIXOS para todo o experimento
PARAMETROS_FIXOS = {
    "embedding_model": "text-embedding-3-small",
}

# --- 2. Funções Auxiliares ---

def gerar_plano_experimental(parametros_variaveis: dict) -> list[dict]:
    chaves = parametros_variaveis.keys()
    valores = parametros_variaveis.values()
    combinacoes = list(product(*valores))
    plano = [dict(zip(chaves, combo)) for combo in combinacoes]
    return plano

# --- Funções para a FASE 1: RECUPERAÇÃO ---

def carregar_recuperacoes_existentes(caminho_arquivo: str) -> set:
    if not os.path.exists(caminho_arquivo):
        return set()
    
    configuracoes_existentes = set()
    with open(caminho_arquivo, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line)
                chave = (
                    data['question'],
                    data['config_recuperacao']['chunking_strategy'],
                    data['config_recuperacao']['search_type']
                )
                configuracoes_existentes.add(chave)
            except json.JSONDecodeError:
                continue
    print(f"Encontradas {len(configuracoes_existentes)} recuperações já executadas em '{caminho_arquivo}'.")
    return configuracoes_existentes

def chamar_api_recuperacao(config_recuperacao: dict, pergunta_str: str):
    payload = {
        "pergunta": pergunta_str,
        "max_k": MAX_K,
        **config_recuperacao
    }
    try:
        response = requests.post(API_URL_RECUPERACAO, json=payload, timeout=120)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"\nERRO ao chamar API de recuperação: {e}")
        return {"documentos_ranqueados": []}

def fase_1_recuperacao():
    print("\n--- INICIANDO FASE 1: RECUPERAÇÃO DE CONTEXTO ---")
    GABARITO = pd.read_json(ARQUIVO_GABARITO).to_dict(orient="records")
    plano_recuperacao = gerar_plano_experimental(PARAMETROS_RECUPERACAO)
    
    recuperacoes_ja_feitas = carregar_recuperacoes_existentes(ARQUIVO_INTERMEDIARIO_JSON)
    
    with open(ARQUIVO_INTERMEDIARIO_JSON, 'a', encoding='utf-8') as f:
        pbar_configs = tqdm(plano_recuperacao, desc="Progresso das Configurações de Recuperação")
        for config_rec in pbar_configs:
            config_completa_rec = {**PARAMETROS_FIXOS, **config_rec}
            
            for item_gabarito in tqdm(GABARITO, desc=f"Config Rec: {str(config_rec)}", leave=False):
                chave_teste_atual = (
                    item_gabarito['question'],
                    config_completa_rec['chunking_strategy'],
                    config_completa_rec['search_type']
                )
                
                if chave_teste_atual in recuperacoes_ja_feitas:
                    continue
                
                resposta_api = chamar_api_recuperacao(config_completa_rec, item_gabarito["question"])
                
                resultado_recuperacao = {
                    "question_id": item_gabarito.get("id", item_gabarito["question"]),
                    "question": item_gabarito["question"],
                    "expected_answer": item_gabarito.get("expected_answer"),
                    "config_recuperacao": config_completa_rec,
                    "documentos_ranqueados": resposta_api.get("documentos_ranqueados", [])
                }
                
                f.write(json.dumps(resultado_recuperacao, ensure_ascii=False) + '\n')
    
    print("--- FASE 1: RECUPERAÇÃO DE CONTEXTO CONCLUÍDA ---")

# --- Funções para a FASE 2: GERAÇÃO ---

def carregar_resultados_existentes_csv(caminho_arquivo: str, colunas_chave: list) -> set:
    if not os.path.exists(caminho_arquivo):
        return set()
    try:
        df = pd.read_csv(caminho_arquivo, on_bad_lines='skip')
        if df.empty: return set()
        
        colunas_validas = [col for col in colunas_chave if col in df.columns]
        if not colunas_validas: return set()

        configuracoes_existentes = set(df[colunas_validas].itertuples(index=False, name=None))
        print(f"Encontradas {len(configuracoes_existentes)} configurações de GERAÇÃO já executadas em '{caminho_arquivo}'.")
        return configuracoes_existentes
    except Exception as e:
        print(f"AVISO: Não foi possível ler o arquivo CSV. Erro: {e}. A Fase 2 pode re-executar testes.")
        return set()

def construir_contexto_local(documentos: list[dict]) -> str:
    if not documentos:
        return "Com base nos documentos consultados, não encontrei informações sobre este assunto."
    c_str = "Contexto para a resposta (use a informação em 'Fonte' para a citação):\n"
    for res in documentos:
        fonte = res.get('fonte_documento', 'Fonte não informada')
        c_str += f"\n---\nFonte: {fonte}\nConteúdo: {res.get('texto')}\n"
    return c_str

def chamar_api_geracao(pergunta_str: str, contexto_str: str, config_geracao: dict):
    payload = {
        "pergunta": pergunta_str,
        "contexto": contexto_str,
        **config_geracao
    }
    try:
        response = requests.post(API_URL_GERACAO, json=payload, timeout=180)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"\nERRO ao chamar API de geração: {e}")
        return {"resposta": f"ERRO_API: {e}", "tokens_usados": 0}

def fase_2_geracao():
    print("\n--- INICIANDO FASE 2: GERAÇÃO DE RESPOSTAS ---")
    
    if not os.path.exists(ARQUIVO_INTERMEDIARIO_JSON):
        print("ERRO: Arquivo intermediário de recuperação não encontrado. Execute a Fase 1 primeiro.")
        return

    plano_geracao = gerar_plano_experimental(PARAMETROS_GERACAO)
    colunas_config_chave = sorted(list(PARAMETROS_RECUPERACAO.keys()) + list(PARAMETROS_GERACAO.keys()) + list(PARAMETROS_FIXOS.keys()) + ["top_k", "question"])
    geracoes_ja_feitas = carregar_resultados_existentes_csv(ARQUIVO_SAIDA_CSV, colunas_config_chave)
    novos_resultados_finais = []

    with open(ARQUIVO_INTERMEDIARIO_JSON, 'r', encoding='utf-8') as f:
        linhas_recuperacao = list(f)
        pbar_recuperacao = tqdm(linhas_recuperacao, desc="Progresso dos Contextos Recuperados")

        for line in pbar_recuperacao:
            try:
                resultado_rec = json.loads(line)
            except json.JSONDecodeError:
                continue

            for k in TOP_K_VALORES:
                documentos_para_contexto = resultado_rec["documentos_ranqueados"][:k]
                contexto_str = construir_contexto_local(documentos_para_contexto)
                retrieved_context_json = json.dumps([doc['fonte_documento'] for doc in documentos_para_contexto], ensure_ascii=False)

                for config_gen in plano_geracao:
                    config_completa = {**resultado_rec["config_recuperacao"], **config_gen, "top_k": k, "question": resultado_rec["question"]}
                    chave_teste_atual = tuple(config_completa[col] for col in colunas_config_chave)
                    
                    if chave_teste_atual in geracoes_ja_feitas:
                        continue
                        
                    resposta_api_geracao = chamar_api_geracao(resultado_rec["question"], contexto_str, config_gen)
                    
                    resultado_final_completo = {
                        **config_completa,
                        "expected_answer": resultado_rec.get("expected_answer"),
                        "bot_answer": resposta_api_geracao.get("resposta"),
                        "retrieved_context_sources": retrieved_context_json,
                        "full_context_sent": contexto_str # <-- CORREÇÃO: SALVANDO O CONTEXTO COMPLETO
                    }
                    novos_resultados_finais.append(resultado_final_completo)

    if not novos_resultados_finais:
        print("\nNenhum novo resultado de geração para adicionar. O experimento já está completo.")
        return
        
    print(f"\nFase 2 finalizada. {len(novos_resultados_finais)} novos resultados foram gerados.")
    
    df_novos = pd.DataFrame(novos_resultados_finais)
    
    colunas_info = ["question", "expected_answer", "bot_answer", "retrieved_context_sources", "full_context_sent"]
    colunas_parametros = sorted(df_novos.columns.drop(colunas_info, errors='ignore'))
    df_novos = df_novos[colunas_info + colunas_parametros]
    
    df_novos.to_csv(
        ARQUIVO_SAIDA_CSV, 
        mode='a', 
        header=not os.path.exists(ARQUIVO_SAIDA_CSV),
        index=False, 
        encoding='utf-8-sig',
        quoting=csv.QUOTE_ALL
    )
    print(f"Resultados foram adicionados com sucesso ao arquivo '{ARQUIVO_SAIDA_CSV}'.")

# --- 3. Script Principal ---

def main():
    print("Iniciando experimento v4.1 (Arquitetura de Duas Fases com Contexto Completo)")
    fase_1_recuperacao()
    fase_2_geracao()
    print("\nExperimento concluído.")

if __name__ == "__main__":
    main()