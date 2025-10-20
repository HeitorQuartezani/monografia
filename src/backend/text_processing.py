import logging
import time
from pymongo import MongoClient
import hashlib
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Configuração básica de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def setup_database():
    mongo_uri = 'mongodb://localhost:27017/'
    db_name = 'legislacao_mpes'
    client = MongoClient(mongo_uri)
    db = client[db_name]
    return client, db['portarias']

def split_text_into_blocks(chunk_size: int = 1000, chunk_overlap: int = 200):
    client, portarias_col = setup_database()
    
    index_info = portarias_col.index_information()
    chunk_params_hash_index_exists = any(
        'chunk_params.hash' in idx_def['key'] for idx_def in index_info.values()
    )
    if not chunk_params_hash_index_exists:
        try:
            portarias_col.create_index([("chunk_params.hash", 1)], background=True)
            logging.info("Índice em 'chunk_params.hash' criado ou já existente.")
        except Exception as e:
            logging.warning(f"Não foi possível criar índice em 'chunk_params.hash': {e}.")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False
    )

    processed_docs_count = 0
    skipped_docs_count = 0
    docs_with_empty_text_count = 0
    errors_processing_doc_count = 0

    try:
        # Query Simplificada:
        # 1. Documentos com 'texto_completo' mas sem 'chunk_params' (nunca processados).
        # 2. Documentos que já têm 'chunk_params' (para re-verificação de hash ou params no código).
        #    Filtraremos mais precisamente no loop Python.
        query = {
            "texto_completo": {"$exists": True, "$ne": ""}, # Deve ter texto_completo não vazio
            "$or": [
                {"chunk_params": {"$exists": False}}, # Nunca processado
                {"chunk_params": {"$exists": True}}   # Já processado, verificar no código
            ]
        }
        # Poderíamos também adicionar:
        # {"chunk_params.status": "ERRO_NO_CHUNKING"} para tentar reprocessar os que falharam.
        # query["$or"].append({"chunk_params.status": "ERRO_NO_CHUNKING"})


        for doc in portarias_col.find(query, no_cursor_timeout=False).batch_size(50):
            doc_id = doc['_id']
            try:
                texto_completo_original = doc.get('texto_completo', '') # Já garantido pela query que existe e não é ""

                # Lógica de erro de coleta
                if texto_completo_original.startswith("ERRO_") or texto_completo_original == "Conteúdo não encontrado":
                    # (código para lidar com ERRO_NA_COLETA como antes)
                    logging.warning(f"Documento {doc_id}: 'texto_completo' indica erro de coleta ('{texto_completo_original[:30]}...'). Limpando blocos e pulando chunking.")
                    current_content_hash_for_error_doc = hashlib.sha256(texto_completo_original.encode('utf-8')).hexdigest()
                    # Verificar se já está no estado correto para evitar update desnecessário
                    # chunk_params_error = doc.get('chunk_params', {})
                    # if not (chunk_params_error.get('status') == 'ERRO_NA_COLETA' and \
                    #         chunk_params_error.get('hash') == current_content_hash_for_error_doc and \
                    #         not doc.get('texto_blocos')): # Verifica se blocos já estão vazios
                    portarias_col.update_one(
                        {'_id': doc_id},
                        {
                            '$set': {
                                'texto_blocos': [],
                                'chunk_params': {
                                    'hash': current_content_hash_for_error_doc,
                                    'chunk_size': chunk_size,
                                    'chunk_overlap': chunk_overlap,
                                    'status': 'ERRO_NA_COLETA',
                                    'ultima_atualizacao_chunk': time.time()
                                }
                            }
                        }
                    )
                    docs_with_empty_text_count +=1
                    continue

                texto_completo_limpo = texto_completo_original.strip()

                # Lógica de texto vazio após strip
                if not texto_completo_limpo:
                    # (código para lidar com TEXTO_VAZIO como antes)
                    logging.warning(f"Documento {doc_id}: 'texto_completo' está vazio ou contém apenas espaços. Limpando blocos.")
                    current_content_hash_for_empty_doc = hashlib.sha256("".encode('utf-8')).hexdigest()
                    # chunk_params_empty = doc.get('chunk_params', {})
                    # if not (chunk_params_empty.get('status') == 'TEXTO_VAZIO' and \
                    #         chunk_params_empty.get('hash') == current_content_hash_for_empty_doc and \
                    #         not doc.get('texto_blocos')):
                    portarias_col.update_one(
                        {'_id': doc_id},
                        {
                            '$set': {
                                'texto_blocos': [],
                                'chunk_params': {
                                    'hash': current_content_hash_for_empty_doc,
                                    'chunk_size': chunk_size,
                                    'chunk_overlap': chunk_overlap,
                                    'status': 'TEXTO_VAZIO',
                                    'ultima_atualizacao_chunk': time.time()
                                }
                            }
                        }
                    )
                    docs_with_empty_text_count +=1
                    continue

                # Agora, a lógica de verificação de reprocessamento
                current_content_hash = hashlib.sha256(texto_completo_limpo.encode('utf-8')).hexdigest()
                chunk_params = doc.get('chunk_params', {})
                
                needs_reprocessing = False
                if not chunk_params: # Nunca processado antes
                    needs_reprocessing = True
                    logging.info(f"Documento {doc_id}: Processando pela primeira vez (sem chunk_params).")
                else:
                    if chunk_params.get('hash') != current_content_hash:
                        needs_reprocessing = True
                        logging.info(f"Documento {doc_id}: Reprocessando - hash do conteúdo mudou.")
                    elif chunk_params.get('chunk_size') != chunk_size:
                        needs_reprocessing = True
                        logging.info(f"Documento {doc_id}: Reprocessando - chunk_size mudou (script: {chunk_size}, doc: {chunk_params.get('chunk_size')}).")
                    elif chunk_params.get('chunk_overlap') != chunk_overlap:
                        needs_reprocessing = True
                        logging.info(f"Documento {doc_id}: Reprocessando - chunk_overlap mudou (script: {chunk_overlap}, doc: {chunk_params.get('chunk_overlap')}).")
                    elif chunk_params.get('status') in ['ERRO_NO_CHUNKING', 'ERRO_NA_COLETA', 'TEXTO_VAZIO']:
                        # Se o status anterior era um tipo de erro, e o texto agora é válido, reprocessar.
                        needs_reprocessing = True
                        logging.info(f"Documento {doc_id}: Reprocessando - status anterior era '{chunk_params.get('status')}' e texto agora parece válido.")

                if not needs_reprocessing:
                    skipped_docs_count += 1
                    logging.debug(f"Documento {doc_id} já processado com os mesmos parâmetros e conteúdo. Pulando.")
                    continue

                # (Restante do código de processamento do documento, split, criação de blocos, etc., como antes)
                logging.info(f"Processando documento {doc_id} para chunking (needs_reprocessing=True).")
                
                existing_blocks_embeddings = {}
                if chunk_params and chunk_params.get('hash') == current_content_hash: # Conteúdo é o mesmo, mas params de chunking mudaram
                    for old_block in doc.get('texto_blocos', []):
                        if old_block.get('texto_hash') and old_block.get('embedding'):
                            existing_blocks_embeddings[old_block['texto_hash']] = old_block['embedding']
                    if existing_blocks_embeddings:
                         logging.debug(f"Documento {doc_id}: {len(existing_blocks_embeddings)} embeddings de blocos antigos carregados para possível reutilização.")

                split_texts = text_splitter.split_text(texto_completo_limpo)
                novos_blocos_para_db = []
                for i, texto_bloco_segmento in enumerate(split_texts):
                    texto_bloco_limpo_segmento = texto_bloco_segmento.strip()
                    if not texto_bloco_limpo_segmento:
                        continue
                    block_content_hash = hashlib.sha256(texto_bloco_limpo_segmento.encode('utf-8')).hexdigest()
                    retrieved_embedding = existing_blocks_embeddings.get(block_content_hash, None)
                    novos_blocos_para_db.append({
                        'texto': texto_bloco_limpo_segmento,
                        'tamanho': len(texto_bloco_limpo_segmento),
                        'texto_hash': block_content_hash,
                        'embedding': retrieved_embedding,
                        'indice_bloco': i,
                    })
                
                if not novos_blocos_para_db:
                    logging.warning(f"Documento {doc_id}: Nenhum bloco de texto gerado após o split de '{texto_completo_limpo[:50]}...'.")
                
                update_result = portarias_col.update_one(
                    {'_id': doc_id},
                    {'$set': {
                        'texto_blocos': novos_blocos_para_db,
                        'chunk_params': {
                            'hash': current_content_hash,
                            'chunk_size': chunk_size,
                            'chunk_overlap': chunk_overlap,
                            'numero_blocos': len(novos_blocos_para_db),
                            'status': 'PROCESSADO_OK',
                            'ultima_atualizacao_chunk': time.time()
                        }
                    }}
                )
                
                if update_result.modified_count > 0:
                    processed_docs_count += 1
                    logging.info(f"Documento {doc_id} atualizado com {len(novos_blocos_para_db)} blocos.")
                elif update_result.matched_count > 0:
                    skipped_docs_count += 1
                    logging.info(f"Documento {doc_id} encontrado, mas não modificado.")
                else:
                    logging.warning(f"Documento {doc_id} não encontrado para atualização.")


            except Exception as e:
                errors_processing_doc_count += 1
                logging.error(f"Erro ao processar documento {doc_id} para chunking: {e}", exc_info=True)
                try:
                    portarias_col.update_one(
                        {'_id': doc_id},
                        {'$set': {'chunk_params.status': 'ERRO_NO_CHUNKING', 'chunk_params.error_message': str(e)[:500]}}
                    )
                except Exception as dbe:
                    logging.error(f"Falha ao marcar documento {doc_id} com erro de chunking no DB: {dbe}")
                continue

        logging.info("Processamento de chunking concluído.")
        logging.info(f"  Documentos efetivamente processados/atualizados: {processed_docs_count}")
        logging.info(f"  Documentos com texto vazio/erro de coleta: {docs_with_empty_text_count}")
        logging.info(f"  Documentos pulados (já atualizados ou sem mudanças): {skipped_docs_count}")
        logging.info(f"  Erros ao processar documentos individuais: {errors_processing_doc_count}")
        
        return processed_docs_count

    except Exception as e:
        logging.critical(f"Erro geral catastrófico no script de chunking: {e}", exc_info=True)
        return 0
    finally:
        if 'client' in locals() and client:
            client.close()
            logging.info("Conexão com MongoDB (script 2) fechada.")

if __name__ == "__main__":
    CS = 500
    CO = 150
    logging.info(f"Iniciando script de divisão de texto com chunk_size={CS}, chunk_overlap={CO}")
    num_docs_atualizados = split_text_into_blocks(chunk_size=CS, chunk_overlap=CO)
    logging.info(f"Número total de documentos cujos blocos foram atualizados: {num_docs_atualizados}")