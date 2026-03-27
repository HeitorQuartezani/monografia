# ==================================================================================================
# PIPELINE DE EXTRAÇÃO E SALVAMENTO DE TEXTOS LEGISLATIVOS EM TXT (v1.1 - Corrigido)
#
# INSTRUÇÕES:
# 1. Instale as dependências: pip install pandas requests beautifulsoup4 tqdm
# 2. Crie uma pasta chamada "documentos_txt" no mesmo diretório do script, ou o script
#    a criará para você.
# 3. Execute o script. Ele vai raspar o site do MPES e salvar cada legislação
#    em um arquivo .txt separado na pasta de saída.
#
# Changelog v1.1:
# - Corrigida a função `sanitize_filename` para remover quebras de linha e outros
#   espaçamentos excessivos dos títulos, evitando o erro "[Errno 22] Invalid argument".
# ==================================================================================================

import os
import time
import logging
import pandas as pd
import requests
import re
from bs4 import BeautifulSoup
from tqdm import tqdm

# --- 1. CONFIGURAÇÃO CENTRAL ---

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Pasta onde os arquivos .txt serão salvos
OUTPUT_DIR = "experimento/documentos_txt"

# --- 2. LÓGICA DE COLETA DE DADOS ---

def gera_tabela_legislacoes() -> pd.DataFrame:
    """Raspa os metadados das legislações do site do MPES, retornando um DataFrame."""
    BASE_URL = "https://mpes.legislacaocompilada.com.br/consulta-legislacao.aspx?situacao=1&interno=0"
    HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"}
    session = requests.Session()
    all_legislacoes = []
    
    try:
        logging.info("Acessando a página inicial da legislação...")
        r = session.get(BASE_URL, headers=HEADERS)
        r.raise_for_status()
        soup = BeautifulSoup(r.content, "html.parser")
        
        # Lógica para mostrar 100 itens por página
        viewstate = soup.select_one('input[name="__VIEWSTATE"]')['value']
        viewstategen = soup.select_one('input[name="__VIEWSTATEGENERATOR"]')['value']
        eventvalidation = soup.select_one('input[name="__EVENTVALIDATION"]')['value']
        data = {
            "__EVENTTARGET": "ctl00$ContentPlaceHolder1$ddl_ItensExibidos", "__EVENTARGUMENT": "", "__LASTFOCUS": "",
            "__VIEWSTATE": viewstate, "__VIEWSTATEGENERATOR": viewstategen, "__EVENTVALIDATION": eventvalidation,
            "ctl00$ContentPlaceHolder1$ddl_ItensExibidos": "100"
        }
        r = session.post(BASE_URL, headers=HEADERS, data=data)
        r.raise_for_status()
        soup = BeautifulSoup(r.content, "html.parser")

        page = 1
        while True:
            logging.info(f"Coletando metadados da página {page}...")
            legislacoes_pagina = []
            for item in soup.select('.kt-widget5__item'):
                titulo_tag = item.select_one('.kt-widget5__title')
                titulo = titulo_tag.get_text(strip=True) if titulo_tag else ""
                
                link_el = next((a for a in item.select('.btn-label-info') if "TEXTO COMPLETO" in a.text), None)
                if not titulo or not link_el: continue
                
                link = link_el['href']
                if not link.startswith('http'): link = "https://mpes.legislacaocompilada.com.br" + link
                
                legislacoes_pagina.append({"titulo_portaria": titulo, "link_texto_completo": link})
            
            if not legislacoes_pagina: break
            all_legislacoes.extend(legislacoes_pagina)

            btn_next = soup.select_one('a#ContentPlaceHolder1_lbNext')
            if not btn_next or 'aspNetDisabled' in btn_next.get('class', []): break

            viewstate = soup.select_one('input[name="__VIEWSTATE"]')['value']
            viewstategen = soup.select_one('input[name="__VIEWSTATEGENERATOR"]')['value']
            eventvalidation = soup.select_one('input[name="__EVENTVALIDATION"]')['value']
            data.update({
                "__EVENTTARGET": "ctl00$ContentPlaceHolder1$lbNext", "__VIEWSTATE": viewstate,
                "__VIEWSTATEGENERATOR": viewstategen, "__EVENTVALIDATION": eventvalidation
            })
            r = session.post(BASE_URL, headers=HEADERS, data=data)
            r.raise_for_status()
            soup = BeautifulSoup(r.content, "html.parser")
            page += 1
            time.sleep(1) # Cortesia para não sobrecarregar o servidor
            
    except Exception as e:
        logging.error(f"Falha durante a coleta de metadados: {e}")

    df = pd.DataFrame(all_legislacoes)
    df.drop_duplicates(subset=['link_texto_completo'], keep='last', inplace=True)
    logging.info(f"Coleta de metadados concluída. Total de documentos únicos encontrados: {len(df)}")
    return df

def download_and_save_texts(df_docs: pd.DataFrame, output_folder: str):
    """Baixa o texto completo de cada documento e o salva em um arquivo .txt."""
    if df_docs.empty:
        logging.info("Nenhum documento para processar.")
        return

    logging.info(f"Iniciando o download e salvamento de {len(df_docs)} documentos...")
    
    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0"})

    # Cria a pasta de saída se ela não existir
    os.makedirs(output_folder, exist_ok=True)

    def sanitize_filename(filename: str) -> str:
        """Limpa e sanitiza uma string para ser usada como um nome de arquivo seguro."""
        # 1. Substitui uma ou mais quebras de linha, tabulações ou espaços por um único espaço.
        filename = re.sub(r'\s+', ' ', filename).strip()

        # 2. Remove caracteres que são inválidos em nomes de arquivo na maioria dos sistemas.
        filename = re.sub(r'[\\/*?:"<>|]', "", filename)
        
        # 3. Garante que o nome do arquivo não seja excessivamente longo (opcional, mas boa prática)
        max_len = 200
        if len(filename) > max_len:
            name, ext = os.path.splitext(filename)
            filename = name[:max_len - len(ext)] + ext

        return filename

    for index, doc in tqdm(df_docs.iterrows(), total=df_docs.shape[0], desc="Processando Documentos"):
        url = doc['link_texto_completo']
        title = doc['titulo_portaria']

        try:
            # Baixar o conteúdo da página
            response = session.get(url, timeout=30)
            response.raise_for_status()
            
            # Extrair texto limpo com BeautifulSoup
            soup = BeautifulSoup(response.content, 'html.parser')
            clean_text = soup.get_text(separator='\n', strip=True)

            if not clean_text:
                logging.warning(f"Texto vazio para o documento: {title} ({url})")
                continue

            # Criar nome de arquivo e salvar
            safe_filename = sanitize_filename(title) + ".txt"
            filepath = os.path.join(output_folder, safe_filename)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(f"FONTE: {url}\n")
                f.write(f"TÍTULO: {title}\n")
                f.write("="*80 + "\n\n")
                f.write(clean_text)
            
        except requests.RequestException as e:
            logging.error(f"Falha ao baixar {title} ({url}): {e}")
        except Exception as e:
            logging.error(f"Erro inesperado ao processar {title}: {e}", exc_info=False) # exc_info=False para não poluir o log com tracebacks

# --- 3. EXECUÇÃO PRINCIPAL ---
if __name__ == "__main__":
    print("=" * 60)
    print("== INICIANDO PIPELINE DE EXTRAÇÃO DE TEXTOS ==")
    print("=" * 60)
    
    pipeline_start_time = time.time()
    
    try:
        # 1. Obter a lista de todos os documentos do site
        df_legislacoes = gera_tabela_legislacoes()
        
        # 2. Baixar e salvar cada documento como TXT
        download_and_save_texts(df_legislacoes, OUTPUT_DIR)

        print("\nPIPELINE DE EXTRAÇÃO CONCLUÍDO COM SUCESSO!")
        print(f"Os arquivos foram salvos na pasta: '{OUTPUT_DIR}'")

    except Exception as e:
        logging.critical(f"!!!!!!   ERRO CRÍTICO NO PIPELINE: {e}   !!!!!!", exc_info=True)

    print("=" * 60)
    print(f"== Tempo total de execução: {time.time() - pipeline_start_time:.2f} segundos ==")
    print("=" * 60)