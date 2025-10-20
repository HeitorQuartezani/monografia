import logging # Adicionado para logging
import time
import requests
from pymongo import MongoClient
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.firefox.options import Options as FirefoxOptions # Para headless, se necessário

# Configuração básica de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def setup_database():
    # Idealmente, pegar de variáveis de ambiente
    mongo_uri = 'mongodb://localhost:27017/'
    db_name = 'legislacao_mpes'
    client = MongoClient(mongo_uri)
    db = client[db_name]
    paginas_col = db['paginas_html']
    portarias_col = db['portarias']
    # Adicionar índices para otimizar queries de sincronização
    portarias_col.create_index("link_texto_completo", unique=True, background=True)
    return client, paginas_col, portarias_col

def save_page_html(paginas_col, page_num, html_content):
    """Salva o HTML de uma página da lista de portarias."""
    try:
        paginas_col.update_one(
            {'pagina_num': page_num},
            {'$set': {
                'html': html_content,
                'timestamp': time.time()
            }},
            upsert=True
        )
        logging.info(f"HTML da página {page_num} salvo/atualizado no MongoDB.")
    except Exception as e:
        logging.error(f"Erro ao salvar HTML da página {page_num}: {e}")

def parse_and_extract_portaria_details(page_num, html_content, base_url, processed_links_in_current_run):
    """
    Analisa o HTML de uma página da lista, extrai informações de cada portaria,
    e busca o texto completo.
    """
    soup = BeautifulSoup(html_content, 'html.parser')
    portaria_items = soup.find_all('div', class_='kt-widget5__item')
    
    extracted_portarias = []
    
    if not portaria_items:
        logging.warning(f"Nenhum item 'kt-widget5__item' encontrado na página {page_num}.")
        return extracted_portarias

    for item_idx, item in enumerate(portaria_items):
        try:
            title_tag = item.find('a', class_='kt-widget5__title')
            portaria_titulo = title_tag.get_text(strip=True) if title_tag else 'N/A'
            
            desc_tag = item.find('a', class_='kt-widget5__desc')
            descricao = desc_tag.get_text(strip=True) if desc_tag else 'N/A'
            
            data_publicacao = 'N/A'
            info_divs = item.find_all('div', class_='kt-widget5__info')
            
            for div in info_divs:
                text_content = div.get_text(strip=True)
                if 'Data:' in text_content:
                    data_span = div.find('span', class_='kt-font-info')
                    data_publicacao = data_span.get_text(strip=True) if data_span else 'N/A'
                    break # Assumindo que há apenas uma data por item
            
            autores = []
            autores_div = next((div for div in info_divs if 'Autor(es) da Norma:' in div.get_text()), None)
            if autores_div:
                autores_links = autores_div.find_all('a')
                autores = [a.get_text(strip=True) for a in autores_links]
            
            link_texto_completo_url = None
            texto_links_tags = item.find_all('a', class_='btn-label-info')
            for link_tag in texto_links_tags:
                if 'TEXTO COMPLETO' in link_tag.get_text(strip=True).upper(): # Mais robusto
                    relative_link = link_tag.get('href', '')
                    if relative_link:
                        link_texto_completo_url = f"{base_url}{relative_link}" if relative_link.startswith('/') else relative_link
                    break
            
            if not link_texto_completo_url:
                logging.warning(f"Link 'TEXTO COMPLETO' não encontrado para item na página {page_num}, título: {portaria_titulo[:50]}")
                continue

            # Evitar reprocessar o mesmo link dentro desta execução (não consulta o DB aqui por performance)
            if link_texto_completo_url in processed_links_in_current_run:
                logging.debug(f"Link {link_texto_completo_url} já processado nesta execução. Pulando.")
                continue
            
            processed_links_in_current_run.add(link_texto_completo_url)
            
            texto_completo_content = ''
            try:
                # Usar headers para simular um navegador pode ajudar
                headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
                response = requests.get(link_texto_completo_url, timeout=20, headers=headers) # Timeout aumentado
                response.raise_for_status() # Levanta erro para status 4xx/5xx
                response.encoding = response.apparent_encoding # Tenta detectar o encoding correto
                
                soup_texto = BeautifulSoup(response.text, 'html.parser')

                # Remover tags de script, style, etc. antes de procurar o conteúdo
                for s_tag in soup_texto(['script', 'style', 'nav', 'header', 'footer', 'aside', 'form']):
                    s_tag.decompose()

                content_div = soup_texto.find('div', class_='kt-portlet__body')
                
                if content_div:
                    # Tentar extrair de <p> e <div> primeiro, filtrando os vazios
                    elements = content_div.find_all(['p', 'div', 'span', 'li', 'td', 'th']) # Mais abrangente
                    
                    text_parts = []
                    for elem in elements:
                        # Evitar pegar texto de sub-elementos já processados ou indesejados
                        # Esta é uma heurística e pode precisar de ajuste.
                        # A ideia é pegar o texto "mais direto" do elemento.
                        elem_text = elem.get_text(separator=' ', strip=True)
                        if elem_text:
                            text_parts.append(elem_text)
                    
                    if text_parts:
                         texto_completo_content = "\n".join(text_parts)
                    else:
                        # Fallback: pegar todo o texto diretamente da content_div se não houver elementos internos com texto
                        texto_completo_content = content_div.get_text(separator='\n', strip=True)
                        if not texto_completo_content.strip(): # Se ainda estiver vazio
                             logging.warning(f"Div 'kt-portlet__body' encontrada, mas sem texto extraível de subelementos ou dela mesma para {link_texto_completo_url}")
                             texto_completo_content = "" # Garantir que é uma string vazia se nada for extraído

                elif soup_texto.body:
                    logging.warning(f"Div 'kt-portlet__body' não encontrada em {link_texto_completo_url}. Usando soup_texto.body como fallback.")
                    # Fallback mais amplo - CUIDADO, pode pegar muito lixo.
                    # Priorizar áreas de conteúdo se conhecidas (ex: <article>, <main>)
                    main_content_area = soup_texto.find('article') or soup_texto.find('main') or soup_texto.body
                    texto_completo_content = main_content_area.get_text(separator='\n', strip=True)
                else:
                    logging.error(f"Nem 'kt-portlet__body' nem 'body' encontrados em {link_texto_completo_url}. HTML pode estar malformado.")
                    texto_completo_content = "ERRO_ESTRUTURA_HTML_VAZIA" # Sinaliza problema grave

                # Limpeza final do texto extraído
                if texto_completo_content:
                    # Remover múltiplas quebras de linha e espaços redundantes
                    lines = [line.strip() for line in texto_completo_content.splitlines() if line.strip()]
                    texto_completo_content = "\n".join(lines)

                if not texto_completo_content.strip() and not texto_completo_content.startswith("ERRO_"): # Se ficou vazio após limpeza
                    logging.warning(f"Texto completo resultou vazio após extração e limpeza para o link: {link_texto_completo_url}. Verificar parser ou estrutura da página de origem.")
                
            except requests.exceptions.HTTPError as e:
                logging.error(f"Erro HTTP ao baixar texto de {link_texto_completo_url}: {e}")
                texto_completo_content = f"ERRO_HTTP_{e.response.status_code}"
            except requests.exceptions.RequestException as e:
                logging.error(f"Erro de requisição ao baixar texto de {link_texto_completo_url}: {e}")
                texto_completo_content = "ERRO_REQUISICAO_DOWNLOAD"
            except Exception as e:
                logging.error(f"Erro inesperado ao processar texto de {link_texto_completo_url}: {e}", exc_info=True)
                texto_completo_content = "ERRO_INESPERADO_PROCESSAMENTO_TEXTO"

            extracted_portarias.append({
                '_id': link_texto_completo_url, # Usar o link como ID é uma boa prática se for único
                'pagina_num_origem': page_num,
                'titulo_portaria': portaria_titulo,
                'descricao': descricao,
                'data_publicacao': data_publicacao,
                'autores': autores,
                'link_texto_completo': link_texto_completo_url, # Campo redundante se _id é o link, mas pode ser útil
                'texto_completo': texto_completo_content,
                'timestamp_extracao': time.time()
            })
        except Exception as e:
            logging.error(f"Erro ao processar item {item_idx} na página {page_num}: {e}", exc_info=True)
            continue # Pula para o próximo item na página

    return extracted_portarias

def navigate_and_sync_portarias():
    client, paginas_col, portarias_col = setup_database()
    
    # Configurar Selenium para modo headless (opcional, mas bom para servidores)
    firefox_options = FirefoxOptions()
    # firefox_options.add_argument("--headless") # Descomente para rodar sem interface gráfica
    
    # É recomendável usar um gerenciador de drivers como webdriver-manager
    # from selenium.webdriver.firefox.service import Service as FirefoxService
    # from webdriver_manager.firefox import GeckoDriverManager
    # driver = webdriver.Firefox(service=FirefoxService(GeckoDriverManager().install()), options=firefox_options)
    driver = webdriver.Firefox(options=firefox_options) # Simples, se o geckodriver estiver no PATH

    base_url = "https://mpes.legislacaocompilada.com.br"
    start_url = f"{base_url}/consulta-legislacao.aspx?situacao=1&interno=0"
    
    try:
        all_extracted_portarias_dict = {} # Usar dicionário para fácil atualização por link (ID)
        processed_links_in_current_run = set() # Para evitar re-buscar o mesmo texto completo na mesma rodada
        
        logging.info(f"Acessando URL inicial: {start_url}")
        driver.get(start_url)
        
        # Configurar para mostrar 100 itens por página
        try:
            select_itens_por_pagina = WebDriverWait(driver, 20).until(
                EC.presence_of_element_located((By.NAME, "ctl00$ContentPlaceHolder1$ddl_ItensExibidos"))
            )
            select_itens_por_pagina.send_keys("100")
            logging.info("Configurado para 100 itens por página. Aguardando carregamento...")
            # Aguardar um pouco para a página recarregar com 100 itens.
            # Uma espera mais inteligente seria verificar se um elemento mudou ou se o número de itens aumentou.
            time.sleep(5) # Aumentar se necessário
        except Exception as e:
            logging.error(f"Não foi possível configurar 100 itens por página: {e}. Continuando com o padrão.")

        current_page_num = 1
        max_page_navigation_attempts = 3
        
        while True:
            logging.info(f"Processando página da lista: {current_page_num}...")
            try:
                # Aguardar que os itens da lista estejam presentes
                WebDriverWait(driver, 20).until(
                    EC.presence_of_all_elements_located((By.CLASS_NAME, "kt-widget5__item"))
                )
                page_html_content = driver.page_source
                save_page_html(paginas_col, current_page_num, page_html_content)
                
                portarias_from_page = parse_and_extract_portaria_details(
                    current_page_num, page_html_content, base_url, processed_links_in_current_run
                )
                for portaria_data in portarias_from_page:
                    all_extracted_portarias_dict[portaria_data['_id']] = portaria_data 
                logging.info(f"Página {current_page_num}: {len(portarias_from_page)} portarias extraídas (ou texto completo buscado).")

            except Exception as e:
                logging.error(f"Erro crítico ao processar conteúdo da página {current_page_num}: {e}", exc_info=True)
                # Poderia tentar recarregar a página ou pular para a próxima
                # Por ora, vamos tentar ir para a próxima página se houver botão

            # Navegar para a próxima página da lista
            page_nav_attempt = 0
            navigated_to_next = False
            while page_nav_attempt < max_page_navigation_attempts:
                try:
                    next_button = WebDriverWait(driver, 10).until(
                        EC.element_to_be_clickable((By.ID, "ContentPlaceHolder1_lbNext")) # element_to_be_clickable é melhor
                    )
                    
                    if "aspNetDisabled" in next_button.get_attribute("class"):
                        logging.info("Botão 'Próximo' está desabilitado. Última página da lista alcançada.")
                        navigated_to_next = False # Sinaliza para sair do loop principal
                        break 
                    
                    # Scroll para o botão e click via JavaScript podem ser mais robustos
                    driver.execute_script("arguments[0].scrollIntoView(true);", next_button)
                    time.sleep(0.5) # Pequena pausa antes do clique
                    # next_button.click() # Click direto pode falhar por overlay
                    driver.execute_script("arguments[0].click();", next_button)

                    # Esperar que a página realmente mude.
                    # Uma boa forma é esperar que um elemento da página anterior se torne 'stale'
                    # ou que um elemento da nova página apareça (ex: o indicador de página atual).
                    # Por simplicidade, uma pausa, mas idealmente usar EC.staleness_of ou similar.
                    time.sleep(5) # Aumentar se a navegação for lenta
                    
                    current_page_num += 1
                    navigated_to_next = True
                    break # Sucesso na navegação
                except Exception as e:
                    page_nav_attempt += 1
                    logging.warning(f"Tentativa {page_nav_attempt}/{max_page_navigation_attempts} de clicar no botão 'Próximo' falhou: {e}")
                    if page_nav_attempt < max_page_navigation_attempts:
                        time.sleep(3) # Espera antes de tentar novamente
                        driver.refresh() # Tenta recarregar a página
                        time.sleep(5) # Espera o refresh
                    else:
                        logging.error("Número máximo de tentativas de navegação para próxima página excedido. Interrompendo.")
                        navigated_to_next = False # Sinaliza para sair do loop principal
                        break
            
            if not navigated_to_next: # Se não conseguiu navegar ou era a última página
                break
        
        # Sincronização com o banco de dados
        if all_extracted_portarias_dict:
            logging.info(f"Total de {len(all_extracted_portarias_dict)} portarias únicas extraídas nesta execução.")
            
            # Obter todos os IDs (_id é o link_texto_completo) do DB para comparação
            # Isto pode ser custoso se a coleção for muito grande. Alternativas:
            # 1. Processar em lotes.
            # 2. Ter um campo 'last_seen_in_crawl' e atualizar.
            db_ids = set(p['_id'] for p in portarias_col.find({}, {'_id': 1}))
            
            current_run_ids = set(all_extracted_portarias_dict.keys())
            
            ids_to_add_or_update = current_run_ids
            ids_to_remove = db_ids - current_run_ids # IDs no DB que não estão na coleta atual

            # Adicionar novos ou atualizar existentes
            # Usar bulk_write para eficiência
            from pymongo import UpdateOne, InsertOne # Mover import para o topo se usado em mais lugares
            
            bulk_operations = []
            updated_count = 0
            added_count = 0

            for doc_id in ids_to_add_or_update:
                portaria_doc = all_extracted_portarias_dict[doc_id]
                # Verificar se o texto completo mudou para decidir se atualiza
                # Para isso, precisaríamos do hash do texto_completo antigo.
                # Por simplicidade, vamos atualizar se o doc_id já existe, ou inserir se é novo.
                # Uma lógica mais sofisticada poderia comparar `timestamp_extracao` ou um hash do `texto_completo`
                
                # Se o documento já existe, fazemos um UpdateOne
                # Se não, um InsertOne.
                # A forma mais simples de fazer upsert em massa é com UpdateOne e upsert=True,
                # mas precisamos garantir que os campos corretos sejam definidos em cada caso.
                # Aqui, vamos assumir que se está em current_run_ids, deve existir ou ser criado/atualizado.
                
                # Se _id (link) existe no DB, atualizamos. Senão, inserimos.
                # O campo _id já está em portaria_doc
                if doc_id in db_ids:
                     # Apenas atualizamos se o texto_completo realmente mudou,
                     # ou se outros campos relevantes mudaram.
                     # Para simplificar, vamos atualizar se o documento foi re-extraído.
                     # Uma lógica mais robusta compararia o conteúdo.
                    bulk_operations.append(
                        UpdateOne({'_id': doc_id}, {'$set': portaria_doc}) # Sobrescreve com os novos dados
                    )
                    updated_count+=1
                else:
                    bulk_operations.append(
                        InsertOne(portaria_doc)
                    )
                    added_count+=1

            if bulk_operations:
                try:
                    result = portarias_col.bulk_write(bulk_operations, ordered=False)
                    logging.info(f"Bulk write para adicionar/atualizar: {result.inserted_count} inseridos, {result.modified_count} modificados.")
                    # Note: result.modified_count pode ser menor que updated_count se os dados não mudaram.
                    # result.upserted_count também seria relevante se usássemos upsert=True em UpdateOne.
                except Exception as e:
                    logging.error(f"Erro durante bulk_write de adição/atualização: {e}")

            if added_count > 0:
                 logging.info(f"{added_count} novas portarias preparadas para inserção/atualização.")
            if updated_count > 0:
                 logging.info(f"{updated_count} portarias existentes preparadas para atualização.")


            # Remover documentos que não foram encontrados na coleta atual
            if ids_to_remove:
                logging.info(f"Removendo {len(ids_to_remove)} portarias que não existem mais.")
                try:
                    delete_result = portarias_col.delete_many({'_id': {'$in': list(ids_to_remove)}})
                    logging.info(f"Portarias removidas: {delete_result.deleted_count}")
                except Exception as e:
                    logging.error(f"Erro ao remover portarias antigas: {e}")
            
            # Remover campo 'situacao' se existir (conforme código original)
            # Esta operação pode ser pesada se feita em toda a coleção.
            # Considere se é realmente necessária a cada execução.
            # if portarias_col.count_documents({"situacao": {"$exists": True}}) > 0:
            #     logging.info("Removendo campo 'situacao' de documentos onde existe...")
            #     try:
            #         # update_result_situacao = portarias_col.update_many(
            #         #     {"situacao": {"$exists": True}}, # Apenas em documentos que têm o campo
            #         #     {"$unset": {"situacao": ""}}
            #         # )
            #         # logging.info(f"Campo 'situacao' removido de {update_result_situacao.modified_count} documentos.")
            #         pass # Desabilitado temporariamente, pois pode não ser o foco do problema
            #     except Exception as e:
            #         logging.error(f"Erro ao remover campo 'situacao': {e}")

        else:
            logging.info("Nenhuma portaria extraída nesta execução.")
        
        logging.info("Sincronização concluída!")
    
    except Exception as e:
        logging.critical(f"Erro fatal no processo de navegação e sincronização: {e}", exc_info=True)
    finally:
        if 'driver' in locals() and driver:
            driver.quit()
            logging.info("Driver do Selenium finalizado.")
        if 'client' in locals() and client:
            client.close()
            logging.info("Conexão com MongoDB fechada.")

if __name__ == "__main__":
    navigate_and_sync_portarias()