# --- 1. CONFIGURAÇÃO CENTRAL ---
import os
import time
import logging
import pandas as pd
import requests
from bs4 import BeautifulSoup
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import AzureOpenAIEmbeddings
from dotenv import load_dotenv
from openai import AzureOpenAI, OpenAIError
from tqdm import tqdm
import chromadb
import tiktoken

# Lógica de .env simplificada:
# 1. Procura o .env no diretório atual (C:\monografia)
# 2. Se não achar, procura na pasta pai (caso o script seja rodado de dentro de /src)
project_root = os.path.abspath(os.getcwd()) # Começa em C:\monografia
dotenv_path = os.path.join(project_root, '.env')

if not os.path.exists(dotenv_path):
    # Fallback: Se estiver rodando de dentro de /src, CWD será C:\monografia\src
    # Então, procuramos um nível acima.
    project_root = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
    dotenv_path = os.path.join(project_root, '.env')

# Carrega o .env encontrado
load_dotenv(dotenv_path=dotenv_path)

# Configuração do Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(funcName)s] - %(message)s')
if os.path.exists(dotenv_path):
    logging.info(f"Arquivo .env encontrado e carregado de: {dotenv_path}")
else:
    logging.warning(f"Arquivo .env NÃO encontrado. (Procurado em {dotenv_path})")
    # Mesmo se não encontrar, o script continua para mostrar os erros de 'None'

BASE_COLLECTION_NAME = "portarias_mpes"
CHROMA_DATA_PATH = "chroma_db"

EMBEDDING_MODELS = [
    {
        "model_name": "text-embedding-3-small",
        "azure_deployment": os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT"),
        "max_tokens": 8191
    },
]
DOCUMENT_PROCESSING_BATCH_SIZE = 100
EMBEDDING_API_BATCH_SIZE = 16
CHROMA_UPSERT_BATCH_SIZE = 2048

# --- 2. VALIDAÇÃO, CHUNKING, COLETA ---
def validate_configurations() -> bool:
    logging.info("Iniciando verificação de pré-voo das configurações...")
    all_configs_valid = True
    for model_config in EMBEDDING_MODELS:
        model_name, azure_deployment = model_config["model_name"], model_config["azure_deployment"]
        logging.info(f"Validando config para o modelo: '{model_name}' (deployment: {azure_deployment})...")
        if not azure_deployment:
            logging.critical(f"FALHA: Deployment para '{model_name}' não configurado.")
            all_configs_valid = False
            continue
        try:
            AzureOpenAI(api_key=os.getenv("AZURE_OPENAI_KEY"), api_version="2023-05-15", azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"))
            AzureOpenAIEmbeddings(api_key=os.getenv("AZURE_OPENAI_KEY"), azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"), azure_deployment=azure_deployment, openai_api_version="2023-05-15")
            logging.info(f"Config para '{model_name}' é VÁLIDA.")
        except (OpenAIError, ValueError) as e:
            logging.critical(f"FALHA na validação para '{model_name}': {e}")
            all_configs_valid = False
    if all_configs_valid: logging.info("Verificação de pré-voo concluída.")
    else: logging.critical("Configurações inválidas. Pipeline abortado.")
    return all_configs_valid

def split_by_recursive_char(text: str, params: dict) -> list[str]:
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=params.get("chunk_size", 1000), chunk_overlap=params.get("chunk_overlap", 200), length_function=len)
    return text_splitter.split_text(text)

def split_by_semantic(text: str, params: dict, langchain_embeddings: AzureOpenAIEmbeddings) -> list[str]:
    if not langchain_embeddings: return []
    try:
        text_splitter = SemanticChunker(embeddings=langchain_embeddings, breakpoint_threshold_type=params.get("breakpoint_threshold_type", "percentile"), breakpoint_threshold_amount=params.get("breakpoint_threshold_amount", 95))
        return text_splitter.split_text(text)
    except Exception as e:
        logging.warning(f"Falha no SemanticChunker, retornando texto original como um único chunk. Erro: {e}")
        return [text]

CHUNKING_STRATEGIES = [{"name": "recursive_1000_200", "function": split_by_recursive_char, "params": {"chunk_size": 1000, "chunk_overlap": 200}},
                        {"name": "recursive_500_100", "function": split_by_recursive_char, "params": {"chunk_size": 500, "chunk_overlap": 100}}, 
                        {"name": "semantic_percentile_75", "function": split_by_semantic, "params": {"breakpoint_threshold_type": "percentile", "breakpoint_threshold_amount": 75}},
                        {"name": "semantic_percentile_95", "function": split_by_semantic, "params": {"breakpoint_threshold_type": "percentile", "breakpoint_threshold_amount": 95}}]

def gera_tabela_legislacoes() -> pd.DataFrame:
    BASE_URL = "https://mpes.legislacaocompilada.com.br/consulta-legislacao.aspx?situacao=1&interno=0"
    HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"}
    session = requests.Session()
    all_legislacoes = []
    try:
        logging.info("Iniciando requisição à página principal...")
        r = session.get(BASE_URL, headers=HEADERS, timeout=30)
        r.raise_for_status()
        soup = BeautifulSoup(r.content, "html.parser")
        viewstate = soup.select_one('input[name="__VIEWSTATE"]')['value']
        viewstategen = soup.select_one('input[name="__VIEWSTATEGENERATOR"]')['value']
        eventvalidation = soup.select_one('input[name="__EVENTVALIDATION"]')['value']
        data = {"__EVENTTARGET": "ctl00$ContentPlaceHolder1$ddl_ItensExibidos", "__EVENTARGUMENT": "", "__LASTFOCUS": "", "__VIEWSTATE": viewstate, "__VIEWSTATEGENERATOR": viewstategen, "__EVENTVALIDATION": eventvalidation, "ctl00$ContentPlaceHolder1$ddl_ItensExibidos": "100"}
        logging.info("Requisitando visualização com 100 itens por página...")
        r = session.post(BASE_URL, headers=HEADERS, data=data, timeout=30)
        r.raise_for_status()
        soup = BeautifulSoup(r.content, "html.parser")
        page = 1
        while True:
            logging.info(f"Coletando metadados da página {page}...")
            legislacoes_pagina = []

            # --- INÍCIO DA LÓGICA MODIFICADA ---
            # Itera sobre cada item de legislação na página
            for item in soup.select('.kt-widget5__item'):
                
                # Primeiro, garante que o item tem um título antes de prosseguir
                titulo_el = item.select_one('.kt-widget5__title')
                if not titulo_el:
                    continue

                # Seleciona todos os botões de link dentro do item
                botoes_links = item.select('.btn-label-info')
                link_final_el = None

                # Prioridade 1: Buscar por 'Texto Compilado'
                link_compilado = next((a for a in botoes_links if "TEXTO COMPILADO" in a.get_text(strip=True)), None)
                if link_compilado:
                    link_final_el = link_compilado
                else:
                    # Prioridade 2 (Fallback): Buscar por 'Texto Completo' se 'Compilado' não existir
                    link_completo = next((a for a in botoes_links if "TEXTO COMPLETO" in a.get_text(strip=True)), None)
                    if link_completo:
                        link_final_el = link_completo

                # Se um link válido (compilado ou completo) foi encontrado, extrai as informações
                if link_final_el:
                    href = link_final_el['href']
                    link_url = "https://mpes.legislacaocompilada.com.br" + href if not href.startswith('http') else href
                    
                    legislacoes_pagina.append({
                        "titulo_portaria": titulo_el.get_text(strip=True), 
                        # A chave do dicionário continua a mesma para não quebrar o resto do código
                        "link_texto_completo": link_url 
                    })
            # --- FIM DA LÓGICA MODIFICADA ---

            if not legislacoes_pagina: break
            all_legislacoes.extend(legislacoes_pagina)
            btn_next = soup.select_one('a#ContentPlaceHolder1_lbNext')
            if not btn_next or 'aspNetDisabled' in btn_next.get('class', []): break
            viewstate, viewstategen, eventvalidation = soup.select_one('input[name="__VIEWSTATE"]')['value'], soup.select_one('input[name="__VIEWSTATEGENERATOR"]')['value'], soup.select_one('input[name="__EVENTVALIDATION"]')['value']
            data.update({"__EVENTTARGET": "ctl00$ContentPlaceHolder1$lbNext", "__VIEWSTATE": viewstate, "__VIEWSTATEGENERATOR": viewstategen, "__EVENTVALIDATION": eventvalidation})
            logging.info(f"Requisitando página {page + 1}...")
            r = session.post(BASE_URL, headers=HEADERS, data=data, timeout=30)
            r.raise_for_status()
            soup = BeautifulSoup(r.content, "html.parser")
            page += 1
            time.sleep(1)
        df = pd.DataFrame(all_legislacoes)
        if not df.empty: df.drop_duplicates(subset=['link_texto_completo'], keep='last', inplace=True)
        logging.info(f"Coleta concluída. Documentos únicos: {len(df)}")
        return df
    except Exception as e:
        logging.error(f"Erro na coleta: {e}", exc_info=True)
        return pd.DataFrame()

def get_processed_ids(collection: chromadb.Collection) -> set:
    try:
        if collection.count() == 0: return set()
        all_metadata = collection.get(include=["metadatas"])['metadatas']
        processed_ids = {meta['source_document_id'] for meta in all_metadata if 'source_document_id' in meta}
        logging.info(f"Encontrados {len(processed_ids)} documentos processados na coleção '{collection.name}'.")
        return processed_ids
    except Exception as e:
        logging.error(f"Erro ao obter IDs processados da coleção '{collection.name}': {e}")
        return set()

def delete_stale_documents(ids_to_delete: set, collection: chromadb.Collection, model_name: str):
    if not ids_to_delete: return
    logging.warning(f"[{model_name}] Removendo {len(ids_to_delete)} documentos órfãos da coleção '{collection.name}'...")
    try:
        collection.delete(where={"source_document_id": {"$in": list(ids_to_delete)}})
        logging.info(f"[{model_name}] Remoção de documentos órfãos concluída.")
    except Exception as e:
        logging.error(f"[{model_name}] Erro ao remover documentos órfãos: {e}")

# --- 3. FUNÇÃO DE PROCESSAMENTO EM LOTES ---
def process_documents_in_batches(
    df_docs_to_add: pd.DataFrame, 
    collection: chromadb.Collection, 
    model_config: dict, 
    openai_client: AzureOpenAI,
    langchain_embeddings: AzureOpenAIEmbeddings,
    tokenizer: tiktoken.Encoding
) -> bool:
    model_name, max_tokens = model_config['model_name'], model_config['max_tokens']
    if df_docs_to_add.empty:
        logging.info(f"[{model_name}] Nenhum documento novo para processar.")
        return True

    overall_success = True
    for i in tqdm(range(0, len(df_docs_to_add), DOCUMENT_PROCESSING_BATCH_SIZE), desc=f"[{model_name}] Processando Lotes de Documentos"):
        
        df_batch = df_docs_to_add.iloc[i:i+DOCUMENT_PROCESSING_BATCH_SIZE].copy()
        
        session = requests.Session()
        session.headers.update({"User-Agent": "Mozilla/5.0"})
        def extract_clean_text(url):
            try:
                response = session.get(url, timeout=20)
                response.raise_for_status()
                return BeautifulSoup(response.content, 'html.parser').get_text(separator='\n', strip=True)
            except requests.RequestException: return None
        df_batch['texto_completo'] = df_batch['link_texto_completo'].apply(extract_clean_text)
        df_batch.dropna(subset=['texto_completo'], inplace=True)
        if df_batch.empty: continue
            
        all_chunks_texts, all_chunks_metadatas, all_chunks_ids = [], [], []
        
        for _, doc in df_batch.iterrows():
            texto = doc.get('texto_completo', '')
            if not texto: continue
            for strategy in CHUNKING_STRATEGIES:
                initial_chunks = strategy["function"](texto, strategy["params"], langchain_embeddings) if strategy['function'] == split_by_semantic else strategy["function"](texto, strategy["params"])
                
                final_chunks = []
                for chunk in initial_chunks:
                    num_tokens = len(tokenizer.encode(chunk))
                    if num_tokens > max_tokens:
                        logging.warning(f"Chunk grande detectado (estratégia: {strategy['name']}, tokens: {num_tokens}). Re-dividindo para caber no limite de {max_tokens} tokens.")
                        token_splitter = RecursiveCharacterTextSplitter(
                            chunk_size=max_tokens,
                            chunk_overlap=max_tokens // 10,
                            length_function=lambda text: len(tokenizer.encode(text))
                        )
                        sub_chunks = token_splitter.split_text(chunk)
                        final_chunks.extend(sub_chunks)
                    else:
                        final_chunks.append(chunk)

                for j, chunk_text in enumerate(final_chunks):
                    chunk_id = f"{doc['link_texto_completo']}|{strategy['name']}|{model_name}|{j}"
                    
                    metadata = {
                        "source_document_id": str(doc["link_texto_completo"]),
                        "documento_origem": str(doc.get("titulo_portaria")),
                        "titulo_portaria": str(doc.get("titulo_portaria")),
                        "chunking_strategy": str(strategy['name']),
                        "embedding_model": str(model_name)
                    }
                    all_chunks_texts.append(chunk_text)
                    all_chunks_metadatas.append(metadata)
                    all_chunks_ids.append(chunk_id)

        if not all_chunks_texts: continue

        all_embeddings = []
        for j in tqdm(range(0, len(all_chunks_texts), EMBEDDING_API_BATCH_SIZE), desc="Gerando Embeddings", leave=False):
            try:
                batch_texts = all_chunks_texts[j:j+EMBEDDING_API_BATCH_SIZE]
                response = openai_client.embeddings.create(input=batch_texts, model=model_config['azure_deployment'])
                all_embeddings.extend([item.embedding for item in response.data])
            except Exception as e:
                logging.error(f"[{model_name}] Falha na API de embeddings no lote {j}: {e}")
                overall_success = False; break
        if not overall_success: break
        
        try:
            for j in range(0, len(all_chunks_ids), CHROMA_UPSERT_BATCH_SIZE):
                collection.upsert(ids=all_chunks_ids[j:j+CHROMA_UPSERT_BATCH_SIZE], embeddings=all_embeddings[j:j+CHROMA_UPSERT_BATCH_SIZE], documents=all_chunks_texts[j:j+CHROMA_UPSERT_BATCH_SIZE], metadatas=all_chunks_metadatas[j:j+CHROMA_UPSERT_BATCH_SIZE])
            logging.info(f"Lote de documentos {i//DOCUMENT_PROCESSING_BATCH_SIZE + 1} ({len(all_chunks_ids)} chunks) ingerido com sucesso.")
        except Exception as e:
            logging.critical(f"[{model_name}] Falha crítica ao inserir no ChromaDB: {e}", exc_info=True)
            overall_success = False; break

    return overall_success

# --- 4. EXECUÇÃO PRINCIPAL ---
if __name__ == "__main__":
    print("=" * 60)
    print("== INICIANDO PIPELINE DE SINCRONIZAÇÃO (v5.7) ==")
    
    pipeline_start_time = time.time()
    
    if not validate_configurations(): exit(1)

    print("-" * 60)
    logging.info("Iniciando coleta de metadados da fonte...")
    df_scraped = gera_tabela_legislacoes()
    if df_scraped.empty:
        logging.critical("Coleta de dados falhou ou fonte está vazia. Abortando.")
        exit(1)
    
    print("-" * 60)
    logging.info("Coleta concluída. Iniciando processamento por modelo...")
    overall_status = True
    chroma_client = chromadb.PersistentClient(path=CHROMA_DATA_PATH)
    
    try:
        tokenizer = tiktoken.get_encoding("cl100k_base")
    except Exception as e:
        logging.critical(f"Não foi possível carregar o tokenizer 'cl100k_base': {e}. Abortando.")
        exit(1)

    for model_config in EMBEDDING_MODELS:
        model_name, azure_deployment = model_config["model_name"], model_config["azure_deployment"]
        collection_name = f"{BASE_COLLECTION_NAME}_{model_name}"
        
        print("-" * 60); print(f"PROCESSANDO: Modelo='{model_name}' | Coleção='{collection_name}'")
        
        try:
            current_openai_client = AzureOpenAI(api_key=os.getenv("AZURE_OPENAI_KEY"), api_version="2023-05-15", azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"))
            current_langchain_embeddings = AzureOpenAIEmbeddings(api_key=os.getenv("AZURE_OPENAI_KEY"), azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"), azure_deployment=azure_deployment, openai_api_version="2023-05-15")
            
            collection = chroma_client.get_or_create_collection(name=collection_name, metadata={"hnsw:space": "cosine"})
            
            scraped_ids, processed_ids = set(df_scraped['link_texto_completo']), get_processed_ids(collection)
            new_ids, stale_ids = scraped_ids - processed_ids, processed_ids - scraped_ids
            
            if not new_ids and not stale_ids:
                print(f"[{model_name}] Nenhuma alteração detectada. Coleção já está sincronizada.")
                continue

            delete_stale_documents(stale_ids, collection, model_name)
            
            df_to_add = df_scraped[df_scraped['link_texto_completo'].isin(new_ids)]
            
            model_pipeline_status = process_documents_in_batches(df_to_add, collection, model_config, current_openai_client, current_langchain_embeddings, tokenizer)
            
            if not model_pipeline_status: overall_status = False
        
        except Exception as e:
            logging.critical(f"ERRO CRÍTICO NO PIPELINE PARA '{model_name}': {e}", exc_info=True)
            overall_status = False

    print("=" * 60)
    if overall_status:
        print("== PIPELINE DE SINCRONIZAÇÃO CONCLUÍDO COM SUCESSO! ==")
    else:
        print("== !!! PIPELINE CONCLUÍDO COM FALHAS. VERIFIQUE OS LOGS. !!! ==")
    print(f"== Tempo total de execução: {time.time() - pipeline_start_time:.2f} segundos ==")
    
    if not overall_status: exit(1)