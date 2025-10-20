import logging
import os
import time
from typing import List, Dict

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pymongo import MongoClient
from tqdm import tqdm

# --- 1. Configuração ---
load_dotenv()

# Configuração do Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configurações do Pipeline
MONGO_URI = os.getenv('MONGO_URI')
MONGO_DB_NAME = os.getenv('MONGO_DB_NAME')
MONGO_COLLECTION_NAME = 'legislacao_vetorizada' # Coleção final com os blocos e vetores

AZURE_OPENAI_KEY = os.getenv('AZURE_OPENAI_KEY')
AZURE_OPENAI_ENDPOINT = os.getenv('AZURE_OPENAI_ENDPOINT')

SCRAPER_BASE_URL = "https://mpes.legislacaocompilada.com.br"
SCRAPER_START_URL = f"{SCRAPER_BASE_URL}/consulta-legislacao.aspx?situacao=1&interno=0"

# --- 2. Funções do Pipeline ---

def scrape_and_extract() -> List[Dict]:
    """
    Etapa 1: Coleta os metadados e o texto completo de todas as legislações.
    Navega pelas páginas usando requests e extrai o conteúdo com BeautifulSoup.
    """
    logger.info(">>> Iniciando Etapa 1: Coleta e Extração de Dados...")
    documentos = []
    
    with requests.Session() as session:
        session.headers.update({"User-Agent": "Mozilla/5.0"})

        # Pega os parâmetros da página inicial
        r = session.get(SCRAPER_START_URL)
        soup = BeautifulSoup(r.content, "html.parser")
        
        # Configura para 100 itens por página
        viewstate = soup.select_one('input[name="__VIEWSTATE"]')['value']
        viewstategen = soup.select_one('input[name="__VIEWSTATEGENERATOR"]')['value']
        eventvalidation = soup.select_one('input[name="__EVENTVALIDATION"]')['value']
        data = {
            "__EVENTTARGET": "ctl00$ContentPlaceHolder1$ddl_ItensExibidos", "__VIEWSTATE": viewstate,
            "__VIEWSTATEGENERATOR": viewstategen, "__EVENTVALIDATION": eventvalidation,
            "ctl00$ContentPlaceHolder1$ddl_ItensExibidos": "100"
        }
        r = session.post(SCRAPER_START_URL, data=data)
        soup = BeautifulSoup(r.content, "html.parser")

        # Itera pelas páginas
        page_num = 1
        while True:
            logger.info(f"Coletando metadados da página {page_num}...")
            
            items = soup.find_all('div', class_='kt-widget5__item')
            if not items:
                break
            
            for item in items:
                titulo_tag = item.find('a', class_='kt-widget5__title')
                link_tag = next((a for a in item.select('.btn-label-info') if "TEXTO COMPLETO" in a.text.upper()), None)

                if not titulo_tag or not link_tag:
                    continue
                
                link_url = SCRAPER_BASE_URL + link_tag['href']
                
                # Extrai texto completo da URL específica
                try:
                    text_response = session.get(link_url, timeout=15)
                    text_soup = BeautifulSoup(text_response.content, 'html.parser')
                    content_div = text_soup.find('div', class_='kt-portlet__body')
                    texto_completo = content_div.get_text(separator='\n', strip=True) if content_div else ""
                except Exception as e:
                    logger.warning(f"Falha ao extrair texto de {link_url}: {e}")
                    texto_completo = ""

                documentos.append({
                    'titulo': titulo_tag.get_text(strip=True),
                    'descricao': (item.find('a', class_='kt-widget5__desc').get_text(strip=True) 
                                 if item.find('a', class_='kt-widget5__desc') else ""),
                    'link_origem': link_url,
                    'texto_completo': texto_completo,
                })

            # Vai para a próxima página
            btn_next = soup.select_one('a#ContentPlaceHolder1_lbNext')
            if 'aspNetDisabled' in btn_next.get('class', []):
                break # Fim
            
            viewstate = soup.select_one('input[name="__VIEWSTATE"]')['value']
            viewstategen = soup.select_one('input[name="__VIEWSTATEGENERATOR"]')['value']
            eventvalidation = soup.select_one('input[name="__EVENTVALIDATION"]')['value']
            data = {
                "__EVENTTARGET": "ctl00$ContentPlaceHolder1$lbNext", "__VIEWSTATE": viewstate,
                "__VIEWSTATEGENERATOR": viewstategen, "__EVENTVALIDATION": eventvalidation,
                "ctl00$ContentPlaceHolder1$ddl_ItensExibidos": "100"
            }
            r = session.post(SCRAPER_START_URL, data=data)
            soup = BeautifulSoup(r.content, "html.parser")
            page_num += 1
            time.sleep(0.5)

    logger.info(f"Coleta concluída. Total de {len(documentos)} documentos extraídos.")
    return documentos

def chunk_documents(documentos: List[Dict]) -> List[Dict]:
    """
    Etapa 2: Pega os documentos e divide o 'texto_completo' em blocos (chunks).
    """
    logger.info(">>> Iniciando Etapa 2: Divisão de Documentos em Blocos (Chunking)...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    
    blocos_finais = []
    for doc in tqdm(documentos, desc="Dividindo documentos"):
        if not doc.get('texto_completo'):
            continue
            
        chunks = text_splitter.split_text(doc['texto_completo'])
        
        for i, chunk_text in enumerate(chunks):
            bloco = {
                'id_bloco': f"{doc['link_origem']}_{i}",
                'titulo_origem': doc['titulo'],
                'link_origem': doc['link_origem'],
                'bloco_texto': chunk_text,
                'embedding': None # Será preenchido na próxima etapa
            }
            blocos_finais.append(bloco)
            
    logger.info(f"Divisão concluída. {len(blocos_finais)} blocos gerados.")
    return blocos_finais

def generate_embeddings(blocos: List[Dict]) -> List[Dict]:
    """
    Etapa 3: Gera os vetores (embeddings) para cada bloco de texto usando a API do Azure.
    """
    logger.info(">>> Iniciando Etapa 3: Geração de Embeddings...")
    
    if not AZURE_OPENAI_KEY or not AZURE_OPENAI_ENDPOINT:
        logger.error("Credenciais da API Azure não encontradas. Pulando etapa de embedding.")
        return blocos

    headers = {"Content-Type": "application/json", "api-key": AZURE_OPENAI_KEY}

    for bloco in tqdm(blocos, desc="Gerando Embeddings"):
        try:
            response = requests.post(
                AZURE_OPENAI_ENDPOINT,
                headers=headers,
                json={"input": bloco['bloco_texto']},
                timeout=20
            )
            response.raise_for_status()
            embedding_data = response.json()['data'][0]['embedding']
            bloco['embedding'] = embedding_data
        except Exception as e:
            logger.error(f"Falha ao gerar embedding para o bloco {bloco['id_bloco']}: {e}")
            # O embedding continuará como None
    
    logger.info("Geração de embeddings concluída.")
    return blocos

def save_to_mongodb(data: List[Dict]):
    """
    Etapa 4: Salva os dados finais no MongoDB.
    Limpa a coleção e insere os novos dados.
    """
    logger.info(f">>> Iniciando Etapa 4: Salvando dados no MongoDB...")
    
    if not MONGO_URI:
        logger.error("MONGO_URI não definida. Não é possível salvar os dados.")
        return

    try:
        client = MongoClient(MONGO_URI)
        db = client[MONGO_DB_NAME]
        collection = db[MONGO_COLLECTION_NAME]

        # Estratégia simples: apaga tudo e insere os novos dados
        logger.info(f"Limpando a coleção '{MONGO_COLLECTION_NAME}'...")
        collection.delete_many({})

        logger.info(f"Inserindo {len(data)} novos documentos...")
        collection.insert_many(data)
        
        # Opcional: Criar um índice vetorial (para MongoDB Atlas)
        # collection.create_search_index(...) 

        logger.info("Dados salvos com sucesso no MongoDB!")
    except Exception as e:
        logger.critical(f"Falha crítica ao salvar no MongoDB: {e}")
    finally:
        if 'client' in locals():
            client.close()

# --- 3. Execução Principal ---

def main():
    """
    Orquestra a execução de todas as etapas do pipeline.
    """
    logger.info("=" * 30)
    logger.info("INICIANDO PIPELINE DE PROCESSAMENTO DE LEGISLAÇÃO")
    logger.info("=" * 30)
    
    start_time = time.time()

    # Etapa 1
    documentos_originais = scrape_and_extract()
    if not documentos_originais:
        logger.error("Nenhum documento foi coletado. Abortando pipeline.")
        return

    # Etapa 2
    blocos_de_texto = chunk_documents(documentos_originais)
    if not blocos_de_texto:
        logger.error("Nenhum bloco de texto foi gerado. Abortando pipeline.")
        return

    # Etapa 3
    blocos_com_vetores = generate_embeddings(blocos_de_texto)

    # Etapa 4
    save_to_mongodb(blocos_com_vetores)
    
    end_time = time.time()
    logger.info("-" * 30)
    logger.info(f"Pipeline concluído com sucesso em {end_time - start_time:.2f} segundos.")
    logger.info("-" * 30)


if __name__ == "__main__":
    main()