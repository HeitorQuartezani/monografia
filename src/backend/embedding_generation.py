import logging
import os
import time
from typing import List, Dict, Tuple, Optional, Any
from pymongo import MongoClient, UpdateOne
from pymongo.errors import PyMongoError
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
    RetryError
)
import requests
from dotenv import load_dotenv

# Configuração básica
load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Constantes
MAX_EMBEDDING_BATCH_UPDATES = 100
MAX_DOCS_PER_EMBEDDING_PASS = 50
EMBEDDING_API_TIMEOUT = 60
EMBEDDING_API_MAX_RETRIES = 5
EMBEDDING_DB_MAX_RETRIES = 3
EMBEDDING_MAIN_LOOP_MAX_ATTEMPTS = 10
EMBEDDING_MAIN_LOOP_NO_PROGRESS_THRESHOLD = 3
EMBEDDING_MODEL_DIMENSION = 1536

class EmbeddingProcessingError(Exception):
    pass

class DatabaseOperationError(Exception):
    pass

def setup_database_embedding() -> Tuple[MongoClient, Any]: # Removido o tipo de retorno não usado
    mongo_uri = os.getenv('MONGO_URI', 'mongodb://localhost:27017/')
    db_name = os.getenv('MONGO_DB_NAME', 'legislacao_mpes')
    client = MongoClient(mongo_uri, maxPoolSize=50)
    db = client[db_name]

    portarias_col = db['portarias']
    
    pending_embedding_index_name = "idx_pending_embeddings_simple"
    index_info = portarias_col.index_information()

    # Opcional: Remover índice antigo se ele foi criado com erro.
    # old_problematic_index_name = "idx_pending_embeddings" 
    # if old_problematic_index_name in index_info:
    #     try:
    #         logging.info(f"Tentando remover índice potencialmente problemático: {old_problematic_index_name}")
    #         portarias_col.drop_index(old_problematic_index_name)
    #         logging.info(f"Índice '{old_problematic_index_name}' antigo removido.")
    #         index_info = portarias_col.index_information() # Atualizar info após drop
    #     except Exception as e_drop:
    #         logging.warning(f"Não foi possível remover o índice antigo '{old_problematic_index_name}': {e_drop}")


    if pending_embedding_index_name not in index_info:
        try:
            portarias_col.create_index(
                [("texto_blocos.embedding", 1)],
                name=pending_embedding_index_name,
                partialFilterExpression={
                    "texto_blocos.embedding": {"$in": [None, []]}
                },
                background=True
            )
            logging.info(f"Índice parcial '{pending_embedding_index_name}' criado ou já existente.")
        except PyMongoError as e:
            logging.warning(f"Não foi possível criar/verificar índice parcial '{pending_embedding_index_name}': {e}. Isso pode ser devido à versão do MongoDB ou tipo de erro. Tentando índice não parcial como fallback.")
            # Fallback para um índice não parcial
            non_partial_index_name = "idx_tb_embedding_basic_v2" # Nome diferente para evitar conflito se o outro já existe e é diferente
            if non_partial_index_name not in index_info: # Re-verificar index_info pode ser necessário se o create_index acima falhou mas ainda criou algo
                try:
                    portarias_col.create_index([("texto_blocos.embedding", 1)], name=non_partial_index_name, background=True)
                    logging.info(f"Índice básico '{non_partial_index_name}' em 'texto_blocos.embedding' criado como fallback.")
                except PyMongoError as e2:
                    logging.error(f"Falha ao criar índice básico de fallback em 'texto_blocos.embedding': {e2}. O script continuará, mas as queries de embedding podem ser lentas.")
            else:
                logging.info(f"Índice básico de fallback '{non_partial_index_name}' já existe.")
    else:
        logging.info(f"Índice parcial '{pending_embedding_index_name}' já existe.")
        
    return client, portarias_col

@retry(
    stop=stop_after_attempt(EMBEDDING_API_MAX_RETRIES),
    wait=wait_exponential(multiplier=1, min=4, max=60),
    retry=retry_if_exception_type((requests.HTTPError, requests.Timeout, requests.ConnectionError, EmbeddingProcessingError)),
    before_sleep=before_sleep_log(logging, logging.WARNING)
)
def get_embedding_from_api(text_to_embed: str) -> List[float]:
    api_key = os.getenv('AZURE_OPENAI_KEY')
    api_endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')

    if not api_key or not api_endpoint:
        logging.error("AZURE_OPENAI_KEY ou AZURE_OPENAI_ENDPOINT não configurados.")
        raise EmbeddingProcessingError("Credenciais da API não configuradas.")

    headers = {
        "Content-Type": "application/json",
        "api-key": api_key
    }
    if len(text_to_embed) == 0: # Adicionado para evitar enviar texto vazio
        logging.warning("Tentativa de gerar embedding para texto vazio. Retornando erro.")
        raise EmbeddingProcessingError("Não é possível gerar embedding para texto vazio.")

    if len(text_to_embed) > 20000: 
        logging.warning(f"Texto muito longo ({len(text_to_embed)} chars) para embedding, truncando para 20000 chars.")
        text_to_embed = text_to_embed[:20000]

    try:
        response = requests.post(
            api_endpoint,
            headers=headers,
            json={"input": text_to_embed},
            timeout=EMBEDDING_API_TIMEOUT
        )
        response.raise_for_status()

        data = response.json()
        embedding_data_list = data.get("data")
        if not embedding_data_list or not isinstance(embedding_data_list, list) or not embedding_data_list[0]:
            logging.error(f"Resposta da API não contém 'data' ou primeiro elemento esperado: {data}")
            raise EmbeddingProcessingError("Estrutura de resposta da API inesperada.")

        embedding_obj = embedding_data_list[0]
        embedding_vector = embedding_obj.get("embedding")

        if not isinstance(embedding_vector, list) or len(embedding_vector) != EMBEDDING_MODEL_DIMENSION:
            logging.error(f"Embedding inválido recebido. Dimensão: {len(embedding_vector) if isinstance(embedding_vector, list) else 'N/A'}. Esperado: {EMBEDDING_MODEL_DIMENSION}")
            raise EmbeddingProcessingError(f"Dimensão do embedding inválida.")

        return embedding_vector

    except requests.JSONDecodeError as e:
        resp_text = response.text[:200] if response else "No response object"
        logging.error(f"Erro ao decodificar JSON da API: {e}. Resposta: {resp_text}")
        raise EmbeddingProcessingError("Formato de resposta inválido (JSONDecodeError)")
    except requests.RequestException as e:
        logging.error(f"Erro na requisição à API de embedding: {e}")
        raise

@retry(
    stop=stop_after_attempt(EMBEDDING_DB_MAX_RETRIES),
    wait=wait_exponential(multiplier=1, min=2, max=20),
    retry=retry_if_exception_type(PyMongoError),
    before_sleep=before_sleep_log(logging, logging.WARNING)
)
def _execute_embedding_bulk_write(collection, operations: List[UpdateOne]):
    if not operations:
        return None
    try:
        result = collection.bulk_write(operations, ordered=False)
        return result
    except PyMongoError as e:
        logging.error(f"Erro durante bulk_write de embeddings: {e}")
        raise DatabaseOperationError(f"Falha no bulk_write de embeddings: {e}")


def process_document_embeddings(portarias_col) -> Dict[str, int]:
    stats = {
        'documentos_verificados':0,
        'blocos_considerados_para_embedding': 0,
        'embeddings_gerados_sucesso': 0,
        'falhas_geracao_embedding': 0,
        'blocos_ja_tinham_embedding_valido': 0,
        'erros_salvamento_db': 0
    }
    
    query = {
        "chunk_params.status": {"$nin": ["ERRO_NA_COLETA", "TEXTO_VAZIO", "ERRO_NO_CHUNKING"]},
        "texto_blocos": { 
            "$elemMatch": { 
                "texto": {"$exists": True, "$ne": ""},       
                "texto_hash": {"$exists": True, "$ne": ""}, 
                "$or": [ 
                    {"embedding": {"$exists": False}},
                    {"embedding": None},
                    {"embedding": {"$size": 0}}
                ]
            }
        }
    }
    
    documentos_cursor = portarias_col.find(query, no_cursor_timeout=False).batch_size(MAX_DOCS_PER_EMBEDDING_PASS)
    all_updates_for_bulk = []

    for documento in documentos_cursor:
        stats['documentos_verificados'] +=1
        doc_id = documento['_id']
        blocos_do_documento = documento.get('texto_blocos', [])
        if not blocos_do_documento:
            continue

        for bloco_idx, bloco in enumerate(blocos_do_documento): # Adicionado enumerate para logging
            # stats['blocos_considerados_para_embedding'] +=1 # Movido para dentro do if de processamento

            current_embedding = bloco.get('embedding')
            if isinstance(current_embedding, list) and len(current_embedding) == EMBEDDING_MODEL_DIMENSION:
                # stats['blocos_ja_tinham_embedding_valido'] +=1 # Contar apenas uma vez
                continue # Pula se já tem embedding válido
            
            texto_do_bloco = bloco.get('texto', '').strip()
            hash_do_bloco = bloco.get('texto_hash')

            # Só considera para embedding se realmente precisa e é válido
            if not texto_do_bloco or not hash_do_bloco:
                logging.debug(f"Bloco {bloco_idx} no doc {doc_id} sem texto ou hash válido. Pulando para embedding.")
                continue
            
            # Agora sim, este bloco será contado como considerado para embedding
            stats['blocos_considerados_para_embedding'] +=1
            if isinstance(current_embedding, list) and len(current_embedding) == EMBEDDING_MODEL_DIMENSION: # Dupla checagem (já feita acima)
                 stats['blocos_ja_tinham_embedding_valido'] +=1
                 continue


            try:
                logging.debug(f"Gerando embedding para bloco {hash_do_bloco} (idx: {bloco_idx}) do doc {doc_id}...")
                novo_embedding = get_embedding_from_api(texto_do_bloco)
                
                all_updates_for_bulk.append(UpdateOne(
                    {'_id': doc_id, 'texto_blocos.texto_hash': hash_do_bloco},
                    {'$set': {'texto_blocos.$.embedding': novo_embedding, 'texto_blocos.$.embedding_timestamp': time.time()}}
                ))
                stats['embeddings_gerados_sucesso'] += 1
                
                if len(all_updates_for_bulk) >= MAX_EMBEDDING_BATCH_UPDATES:
                    logging.info(f"Executando bulk_write com {len(all_updates_for_bulk)} operações de embedding...")
                    try:
                        result = _execute_embedding_bulk_write(portarias_col, all_updates_for_bulk)
                        if result:
                            logging.debug(f"Bulk write de embeddings executado. Modificados: {result.modified_count}")
                        all_updates_for_bulk = [] 
                    except (DatabaseOperationError, RetryError) as e_db:
                        stats['erros_salvamento_db'] += len(all_updates_for_bulk)
                        logging.error(f"Falha persistente ao salvar lote de embeddings: {e_db}")
                        all_updates_for_bulk = [] 
                    except Exception as e_db_unexpected:
                        stats['erros_salvamento_db'] += len(all_updates_for_bulk)
                        logging.error(f"Erro inesperado ao salvar lote de embeddings: {e_db_unexpected}", exc_info=True)
                        all_updates_for_bulk = []

            except EmbeddingProcessingError as e_embed_proc:
                stats['falhas_geracao_embedding'] += 1
                logging.error(f"Falha de processamento ao gerar embedding para bloco {hash_do_bloco} (doc: {doc_id}): {e_embed_proc}")
            except RetryError as e_retry_api: 
                stats['falhas_geracao_embedding'] += 1
                logging.error(f"Falha persistente (API) para bloco {hash_do_bloco} (doc: {doc_id}): {e_retry_api}")
            except Exception as e_general:
                stats['falhas_geracao_embedding'] += 1
                logging.error(f"Erro inesperado ao processar bloco {hash_do_bloco} (doc: {doc_id}): {e_general}", exc_info=True)
        
    if all_updates_for_bulk:
        logging.info(f"Executando bulk_write final com {len(all_updates_for_bulk)} operações de embedding...")
        try:
            result = _execute_embedding_bulk_write(portarias_col, all_updates_for_bulk)
            if result:
                logging.debug(f"Bulk write final de embeddings. Modificados: {result.modified_count}")
        except (DatabaseOperationError, RetryError) as e_db_final:
            stats['erros_salvamento_db'] += len(all_updates_for_bulk)
            logging.error(f"Falha persistente ao salvar lote final de embeddings: {e_db_final}")
        except Exception as e_db_final_unexpected:
            stats['erros_salvamento_db'] += len(all_updates_for_bulk)
            logging.error(f"Erro inesperado ao salvar lote final de embeddings: {e_db_final_unexpected}", exc_info=True)

    logging.info(f"Resumo da passagem de processamento de embeddings:")
    for key, value in stats.items():
        logging.info(f"  {key.replace('_', ' ').capitalize()}: {value}")
    
    return stats

def get_pending_embeddings_count_agg(portarias_col) -> int:
    try:
        pipeline = [
            {"$match": { 
                "chunk_params.status": {"$nin": ["ERRO_NA_COLETA", "TEXTO_VAZIO", "ERRO_NO_CHUNKING"]},
                 # O índice parcial (idx_pending_embeddings_simple) deve ajudar aqui
                 "texto_blocos.embedding": {"$in": [None, []]} 
            }},
            {"$unwind": "$texto_blocos"},
            {"$match": {
                "texto_blocos.texto": {"$exists": True, "$ne": ""},
                "texto_blocos.texto_hash": {"$exists": True, "$ne": ""},
                "$or": [
                    {"texto_blocos.embedding": {"$exists": False}},
                    {"texto_blocos.embedding": None},
                    {"texto_blocos.embedding": {"$size": 0}}
                ]
            }},
            {"$count": "pendentes"}
        ]
        result = list(portarias_col.aggregate(pipeline))
        return result[0]['pendentes'] if result else 0
    except PyMongoError as e:
        logging.error(f"Erro na contagem de embeddings pendentes: {e}")
        raise
    except Exception as e:
        logging.error(f"Erro inesperado na contagem de pendentes: {e}", exc_info=True)
        raise

def main_embedding_loop():
    start_time_main = time.time()
    client_main = None
    success_overall = False
    try:
        client_main, portarias_col_main = setup_database_embedding()
    except PyMongoError as e_db_setup:
        logging.critical(f"Falha crítica ao configurar DB para embeddings: {e_db_setup}. Encerrando.")
        return False
    except Exception as e_setup_unexpected:
        logging.critical(f"Falha inesperada ao configurar DB para embeddings: {e_setup_unexpected}. Encerrando.", exc_info=True)
        return False

    overall_stats_accumulator_main = {'embeddings_gerados_sucesso': 0, 'falhas_geracao_embedding': 0, 'erros_salvamento_db': 0}
    previous_pending_count_main = float('inf')
    consecutive_no_progress_count_main = 0
    current_pending_count_main = float('inf') # Inicializar

    for attempt_main in range(1, EMBEDDING_MAIN_LOOP_MAX_ATTEMPTS + 1):
        logging.info(f"\n--- Tentativa Geral de Embedding {attempt_main}/{EMBEDDING_MAIN_LOOP_MAX_ATTEMPTS} ---")
        try:
            # Verificar se há algo a fazer antes de processar
            current_pending_count_main = get_pending_embeddings_count_agg(portarias_col_main)
            if current_pending_count_main == 0:
                logging.info("Nenhum bloco pendente de embedding encontrado no início da tentativa.")
                success_overall = True
                break 

            current_run_stats_main = process_document_embeddings(portarias_col_main)
            for key in overall_stats_accumulator_main:
                if key in current_run_stats_main:
                    overall_stats_accumulator_main[key] += current_run_stats_main[key]

            # Re-verificar após o processamento
            current_pending_count_main = get_pending_embeddings_count_agg(portarias_col_main)
            logging.info(f"Total de blocos ainda pendentes de embedding após processamento: {current_pending_count_main}")

            if current_pending_count_main == 0:
                logging.info("\n✅ Todos os blocos elegíveis foram processados com embeddings!")
                success_overall = True
                break 

            if current_pending_count_main < previous_pending_count_main:
                consecutive_no_progress_count_main = 0 
            elif current_run_stats_main.get('falhas_geracao_embedding', 0) > 0 or current_run_stats_main.get('erros_salvamento_db', 0) > 0:
                consecutive_no_progress_count_main += 1
                logging.warning(f"Sem progresso na redução de embeddings pendentes ({current_pending_count_main}) E houve falhas. Contagem 'sem progresso': {consecutive_no_progress_count_main}/{EMBEDDING_MAIN_LOOP_NO_PROGRESS_THRESHOLD}")

            if consecutive_no_progress_count_main >= EMBEDDING_MAIN_LOOP_NO_PROGRESS_THRESHOLD:
                logging.error(f"\n❌ Processamento de embedding estagnado por {EMBEDDING_MAIN_LOOP_NO_PROGRESS_THRESHOLD} tentativas com falhas. Verifique erros persistentes.")
                break 

            previous_pending_count_main = current_pending_count_main
        
        except (DatabaseOperationError, PyMongoError) as e_db_loop:
            logging.error(f"Erro de DB na tentativa de embedding {attempt_main}: {e_db_loop}. Tentando novamente após backoff.")
        except requests.RequestException as e_req_loop:
             logging.error(f"Erro de rede na tentativa de embedding {attempt_main}: {e_req_loop}. Tentando novamente após backoff.")
        except Exception as e_crit_loop:
            logging.critical(f"Erro crítico inesperado na tentativa de embedding {attempt_main}: {e_crit_loop}", exc_info=True)
        
        if attempt_main < EMBEDDING_MAIN_LOOP_MAX_ATTEMPTS and current_pending_count_main > 0 :
            sleep_duration = min(60, 2 ** attempt_main)
            logging.info(f"Aguardando {sleep_duration}s antes da próxima tentativa de embedding...")
            time.sleep(sleep_duration)
    
    logging.info("\n--- Resumo Final do Processamento de Embedding ---")
    for key, value in overall_stats_accumulator_main.items():
        logging.info(f"  Total {key.replace('_', ' ').capitalize()}: {value}")
    
    # Final check on pending count
    if 'portarias_col_main' in locals(): # Verificar se a variável foi definida
        final_pending_count_main = get_pending_embeddings_count_agg(portarias_col_main)
        if final_pending_count_main == 0 and success_overall: # Se já era sucesso ou se tornou agora
            logging.info("✅ Processamento de embedding concluído com sucesso para todos os blocos elegíveis.")
            success_overall = True
        else:
            logging.error(f"❌ Processamento de embedding finalizado, mas {final_pending_count_main} blocos elegíveis ainda estão pendentes.")
            success_overall = False
    else:
        logging.error("Não foi possível realizar a contagem final de pendentes (coleção não inicializada).")
        success_overall = False


    total_time_main = time.time() - start_time_main
    logging.info(f"Tempo total de execução do script de embedding: {total_time_main:.2f} segundos.")
    
    return success_overall

if __name__ == "__main__":
    if main_embedding_loop():
        exit(0)
    else:
        exit(1)