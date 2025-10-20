# chatbot.py

import os
import logging
from collections import deque
from datetime import datetime
import numpy as np
from pymongo import MongoClient
from dotenv import load_dotenv
from openai import AzureOpenAI
from bson.objectid import ObjectId

# Carrega as variáveis de ambiente do arquivo .env
load_dotenv()

# Configura o logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ChatbotMPES:
    def __init__(self):
        # --- Clientes e Configurações ---
        try:
            # Cliente Azure OpenAI para Chat (GPT-4o mini)
            self.chat_client = AzureOpenAI(
                azure_endpoint=os.getenv("ENDPOINT_URL_GPT4OMINI"),
                api_key=os.getenv("AZURE_OPENAI_KEY"),
                api_version="2024-05-01-preview"
            )
            self.chat_deployment = os.getenv("DEPLOYMENT_NAME_GPT4OMINI")

            # Cliente Azure OpenAI para Embeddings
            self.embedding_client = AzureOpenAI(
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"), # Endpoint do serviço de embeddings
                api_key=os.getenv("AZURE_OPENAI_KEY"),
                api_version="2023-05-15" # Versão de API estável para embeddings
            )
            self.embedding_deployment = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")

            # Configuração do MongoDB (usando a collection 'portarias')
            self.mongo_client = MongoClient(os.getenv("MONGO_URI", "mongodb://localhost:27017/"))
            self.db = self.mongo_client[os.getenv("MONGO_DB_NAME", "Fabi")] # Corrigido para "Fabi"
            self.portarias_col = self.db['portarias']
            self.conversas_col = self.db['conversas']
            
            self._verificar_indices()

            self.historico = deque(maxlen=3)
            logging.info("ChatbotMPES inicializado com sucesso.")

        except Exception as e:
            logging.error(f"Falha na inicialização do Chatbot: {e}")
            raise

    def __del__(self):
        if hasattr(self, 'mongo_client'):
            self.mongo_client.close()

    def _verificar_indices(self):
        """Verifica e cria os índices de texto e vetorial necessários."""
        try:
            # Índice Textual
            index_name = "idx_texto_blocos"
            if index_name not in self.portarias_col.index_information():
                self.portarias_col.create_index(
                    [("texto_blocos.texto", "text")],
                    name=index_name,
                    default_language="portuguese"
                )
                logging.info(f"Índice textual '{index_name}' criado com sucesso.")
            
            # Nota sobre o índice vetorial
            logging.info("Lembre-se de criar o Índice Vetorial no MongoDB Atlas para a busca semântica funcionar.")
            logging.info("Nome sugerido para o índice: 'idx_vetorial_blocos'")

        except Exception as e:
            logging.error(f"Erro ao configurar índices do MongoDB: {e}")
            raise

    def _gerar_embedding(self, text: str) -> list[float] | None:
        """Gera embedding para um texto usando o cliente Azure OpenAI."""
        try:
            response = self.embedding_client.embeddings.create(
                input=[text], 
                model=self.embedding_deployment
            )
            return response.data[0].embedding
        except Exception as e:
            logging.error(f"Erro ao gerar embedding: {e}")
            return None

    def _buscar_hibrido(self, pergunta: str, top_n: int = 5) -> list[dict]:
        """
        Realiza uma busca híbrida combinando busca vetorial (semântica) e textual (palavra-chave).
        Esta função depende de índices no MongoDB Atlas para performance.
        """
        resultados_unicos = {}

        # --- 1. Busca Vetorial (Semântica) com Atlas Vector Search ---
        query_embedding = self._gerar_embedding(pergunta)
        if query_embedding:
            try:
                pipeline_vetorial = [
                    {
                        "$vectorSearch": {
                            "index": "idx_vetorial_blocos", # NOME DO SEU ÍNDICE VETORIAL NO ATLAS
                            "path": "texto_blocos.embedding",
                            "queryVector": query_embedding,
                            "numCandidates": 100,
                            "limit": top_n
                        }
                    },
                    {
                        "$project": {
                            "score_vetorial": {"$meta": "vectorSearchScore"},
                            "titulo": 1,
                            "link_texto_completo": 1,
                            "texto_blocos": 1,
                            "_id": 0
                        }
                    }
                ]
                cursor_vetorial = self.portarias_col.aggregate(pipeline_vetorial)
                for doc in cursor_vetorial:
                    for bloco in doc.get("texto_blocos", []):
                        # A chave única combina o link do documento e o número do bloco
                        chave = (doc.get("link_texto_completo"), bloco.get("numero_bloco"))
                        if chave not in resultados_unicos and bloco.get("texto"):
                            resultados_unicos[chave] = {
                                "vetorial": doc.get("score_vetorial", 0.0),
                                "textual": 0.0,
                                "texto": bloco["texto"],
                                "fonte": doc.get("titulo", "N/A"),
                                "bloco_numero": bloco.get("numero_bloco", 0),
                                "link": doc.get("link_texto_completo", "")
                            }
            except Exception as e:
                logging.warning(f"Busca vetorial falhou. Verifique o índice no Atlas. Erro: {e}")

        # --- 2. Busca Textual (Palavra-chave) ---
        try:
            cursor_textual = self.portarias_col.find(
                {"$text": {"$search": pergunta}},
                {
                    "score_textual": {"$meta": "textScore"},
                    "titulo": 1,
                    "link_texto_completo": 1,
                    "texto_blocos": 1,
                    "_id": 0
                }
            ).sort([("score_textual", {"$meta": "textScore"})]).limit(top_n)

            for doc in cursor_textual:
                texto_score = doc.get("score_textual", 0)
                for bloco in doc.get("texto_blocos", []):
                    # Adiciona/atualiza apenas se o bloco for relevante
                    if pergunta.lower().split()[0] in bloco.get("texto", "").lower():
                        chave = (doc.get("link_texto_completo"), bloco.get("numero_bloco"))
                        if chave not in resultados_unicos:
                             resultados_unicos[chave] = {
                                "vetorial": 0.0,
                                "textual": texto_score,
                                "texto": bloco["texto"],
                                "fonte": doc.get("titulo", "N/A"),
                                "bloco_numero": bloco.get("numero_bloco", 0),
                                "link": doc.get("link_texto_completo", "")
                            }
                        else:
                            # Se já existe (da busca vetorial), apenas adiciona o score textual
                            resultados_unicos[chave]["textual"] = max(resultados_unicos[chave]["textual"], texto_score)
        
        except Exception as e:
            logging.warning(f"Busca textual falhou. Verifique o índice textual. Erro: {e}")

        return list(resultados_unicos.values())

    def _construir_contexto(self, resultados: list[dict]) -> tuple[str, list]:
        if not resultados:
            return "Nenhum documento relevante encontrado na base de dados.", []

        # Normalizar scores para uma combinação mais justa (escala 0-1)
        max_vetorial = max(r['vetorial'] for r in resultados) if any(r['vetorial'] for r in resultados) else 1
        max_textual = max(r['textual'] for r in resultados) if any(r['textual'] for r in resultados) else 1

        for res in resultados:
            norm_vetorial = res['vetorial'] / max_vetorial
            norm_textual = res['textual'] / max_textual
            # Pontuação híbrida (pode ajustar os pesos)
            res['score_hibrido'] = (0.6 * norm_vetorial) + (0.4 * norm_textual)

        # Ordenar pelo score híbrido final
        resultados_ordenados = sorted(resultados, key=lambda x: x['score_hibrido'], reverse=True)

        contexto_str = "Contexto para a resposta:\n"
        contexto_cru = []

        for i, res in enumerate(resultados_ordenados, 1):
            contexto_str += (
                f"\n--- Bloco de Informação {i} ---\n"
                f"Fonte: {res['fonte']} (Bloco {res['bloco_numero']})\n"
                f"Conteúdo: {res['texto']}\n"
                f"Link para o documento completo: {res['link']}\n"
            )
            contexto_cru.append({
                'fonte': res['fonte'],
                'bloco': res['bloco_numero'],
                'texto': res['texto'],
                'score_hibrido': res['score_hibrido'],
                'link': res['link']
            })

        return contexto_str, contexto_cru

    def _gerar_resposta(self, contexto: str, pergunta: str) -> tuple[str, int, list]:
        """Gera a resposta final usando o modelo de chat com o contexto e histórico."""
        try:
            messages = [
                {"role": "system", "content": (
                    "Você é um assistente especializado na legislação do Ministério Público do Estado do Espírito Santo (MPES). Siga estas regras rigorosamente:\n"
                    "1. Fundamente TODAS as suas respostas exclusivamente no 'Contexto para a resposta' fornecido.\n"
                    "2. Se a informação não estiver no contexto, responda de forma clara: 'Com base nos documentos consultados, não encontrei informações sobre este assunto.' Não invente respostas.\n"
                    "3. Responda de forma objetiva e em formato técnico-jurídico.\n"
                    "4. Ao usar uma informação, cite a fonte e o link no final da frase ou parágrafo. Formato: [Fonte: NOME DA LEI, Bloco NÚMERO, Link: URL_DO_DOCUMENTO]."
                )}
            ]
            # Adicionar histórico da conversa
            for hist_pergunta, hist_resposta in self.historico:
                messages.append({"role": "user", "content": hist_pergunta})
                messages.append({"role": "assistant", "content": hist_resposta})

            # Adicionar contexto e pergunta atual
            messages.append({"role": "user", "content": f"{contexto}\n\nCom base no contexto acima, responda à seguinte pergunta: {pergunta}"})

            response = self.chat_client.chat.completions.create(
                model=self.chat_deployment,
                messages=messages,
                temperature=0.1, # Temperatura baixa para respostas mais factuais
                max_tokens=1024,
                top_p=0.95
            )

            conteudo = response.choices[0].message.content
            total_tokens = response.usage.total_tokens if response.usage else 0
            return conteudo, total_tokens, messages

        except Exception as e:
            logging.error(f"Erro ao chamar a API de chat: {e}")
            return "Ocorreu um erro ao tentar gerar a resposta.", 0, []

    def _registrar_conversa(self, pergunta: str, resposta: str, blocos: list[dict], tokens: int, messages: list):
        """Salva a interação completa no MongoDB."""
        try:
            doc = {
                "pergunta": pergunta,
                "resposta": resposta,
                "blocos_contexto": blocos,
                "prompt_completo": messages,
                "timestamp": datetime.utcnow(),
                "diagnostico": {
                    "chat_deployment": self.chat_deployment,
                    "embedding_deployment": self.embedding_deployment,
                    "total_tokens": tokens
                },
                "historico_utilizado": [{"pergunta": p, "resposta": r} for p, r in self.historico]
            }
            result = self.conversas_col.insert_one(doc)
            logging.info(f"Conversa registrada com ID: {result.inserted_id}. Tokens: {tokens}")
            return str(result.inserted_id)
        except Exception as e:
            logging.error(f"Erro ao registrar conversa: {e}")
            return None
    
    def update_feedback(self, interaction_id: str, feedback_value: int):
        """Atualiza uma conversa com o feedback do usuário (ex: 1 para útil, -1 para não útil)."""
        try:
            self.conversas_col.update_one(
                {"_id": ObjectId(interaction_id)},
                {"$set": {"feedback": feedback_value, "feedback_timestamp": datetime.utcnow()}}
            )
            logging.info(f"Feedback {feedback_value} registrado para interação {interaction_id}")
        except Exception as e:
            logging.error(f"Erro ao registrar feedback para {interaction_id}: {e}")

    def responder(self, pergunta: str) -> tuple[str, str | None]:
        """Método principal para orquestrar a resposta a uma pergunta."""
        logging.info(f"Nova pergunta recebida: '{pergunta}'")
        
        resultados_busca = self._buscar_hibrido(pergunta)
        contexto_formatado, blocos_usados = self._construir_contexto(resultados_busca)
        
        logging.info(f"Contexto construído com {len(blocos_usados)} blocos de informação.")
        
        resposta, tokens, messages = self._gerar_resposta(contexto_formatado, pergunta)
        
        # Adiciona ao histórico ANTES de registrar para que o registro contenha o histórico correto
        self.historico.append((pergunta, resposta))
        
        interaction_id = self._registrar_conversa(pergunta, resposta, blocos_usados, tokens, messages)
        
        return resposta, interaction_id

# --- Bloco de Execução para Teste via Linha de Comando ---
if __name__ == "__main__":
    print("Iniciando Chatbot MPES (CLI)... Digite 'sair' para encerrar.")
    try:
        chatbot = ChatbotMPES()
        while True:
            pergunta_usuario = input("\nVocê: ")
            if pergunta_usuario.lower() in ['sair', 'exit', 'quit']:
                print("Encerrando o chatbot. Até logo!")
                break
            
            resposta_assistente, _ = chatbot.responder(pergunta_usuario)
            print("\nAssistente MPES:", resposta_assistente)
    except Exception as e:
        print(f"\nERRO CRÍTICO: Não foi possível iniciar o chatbot. Detalhes: {e}")