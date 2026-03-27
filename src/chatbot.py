# ==================================================================================================
# API DO CHATBOT MPES (v2.10 - Aceita Prompt de Geração)
#
# DESCRIÇÃO:
# - ADICIONADO: O endpoint '/gerar_resposta' (Fase 2 do Experimento) agora aceita
#   um 'system_prompt_override' para permitir testes de engenharia de prompt.
# - Mantém a busca textual (keyword) com o algoritmo BM25.
# - Mantém a autenticação por Chave de API (X-API-Key).
# ==================================================================================================

import os
import logging
import string
import secrets 
from collections import deque
from datetime import datetime
from contextlib import asynccontextmanager
from enum import Enum
from typing import Literal, Tuple, Dict, List

import uvicorn
import chromadb
import nltk
import numpy as np
from rank_bm25 import BM25Okapi
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from fastapi import FastAPI, HTTPException, Security, Depends, status
from fastapi.security import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from openai import AzureOpenAI

# --- 1. Configuração Inicial ---
load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(funcName)s] - %(message)s')

# --- Configuração de Autenticação ---
API_KEY = os.getenv("API_KEY")
if not API_KEY:
    logging.warning("API_KEY não definida no .env. A API ficará desprotegida.")
    
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

async def get_api_key(key: str = Security(api_key_header)):
    """Dependência que valida a chave de API no cabeçalho."""
    if not API_KEY:
        return True 
    if not key or not secrets.compare_digest(key, API_KEY):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Chave de API inválida ou ausente."
        )
    return key

# --- Funções de pré-processamento de texto para BM25 ---
def _setup_nltk_data():
    """Baixa os pacotes necessários do NLTK se não estiverem presentes."""
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('corpora/stopwords')
        logging.info("Pacotes NLTK ('punkt', 'stopwords') já estão baixados.")
    except LookupError:
        logging.info("Baixando pacotes NLTK ('punkt', 'stopwords')...")
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        logging.info("Download do NLTK concluído.")

def _preprocess_text(text: str) -> list[str]:
    """Processa o texto para a busca BM25: lowercase, remove pontuação e stopwords."""
    try:
        stop_words = set(stopwords.words('portuguese'))
        translator = str.maketrans('', '', string.punctuation)
        text_lower = text.lower()
        text_no_punct = text_lower.translate(translator)
        tokens = word_tokenize(text_no_punct, language='portuguese')
        return [word for word in tokens if word.isalpha() and word not in stop_words]
    except Exception as e:
        logging.warning(f"Erro ao pré-processar texto: {e}. Retornando tokens brutos.")
        return text.lower().split()

# --- Enums para tipos de API controlados ---
class EmbeddingModelType(str, Enum):
    text_embedding_3_small = "text-embedding-3-small"

class SearchType(str, Enum):
    hibrida = "hibrida"
    vetorial = "vetorial"
    textual = "textual"

class ModelType(str, Enum):
    gpt4o_mini = "gpt-4o-mini"
    gpt4o = "gpt-4o"

class ChunkingStrategy(str, Enum):
    recursive_1000_200 = "recursive_1000_200"
    recursive_500_100 = "recursive_500_100"
    semantic_percentile_75 = "semantic_percentile_75"
    semantic_percentile_95 = "semantic_percentile_95"

# --- 2. Classe de Lógica do Chatbot ---
class ChatbotMPES:
    DEFAULT_SYSTEM_PROMPT = "Você é um assistente especialista na legislação do Ministério Público do Estado do Espírito Santo (MPES). Sua função é responder perguntas baseando-se estritamente no contexto fornecido. Seja objetivo e conciso. Ao final da sua resposta, cite o nome do documento de origem de onde a informação foi extraída, usando o formato (Fonte: NOME_DO_DOCUMENTO). Se a informação não estiver no contexto, responda 'Com base nos documentos consultados, não encontrei informações sobre este assunto.'. Não invente informações e não se refira a 'blocos' ou 'trechos'."
    BASE_COLLECTION_NAME = "portarias_mpes"

    def __init__(self):
        logging.info("Inicializando a instância do ChatbotMPES...")
        _setup_nltk_data()
        try:
            azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
            api_key = os.getenv("AZURE_OPENAI_KEY")
            if not all([azure_endpoint, api_key]):
                raise ValueError("AZURE_OPENAI_ENDPOINT e AZURE_OPENAI_KEY devem ser definidos no .env")

            self.chat_client = AzureOpenAI(azure_endpoint=azure_endpoint, api_key=api_key, api_version="2024-05-01-preview")
            self.embedding_client = AzureOpenAI(azure_endpoint=azure_endpoint, api_key=api_key, api_version="2023-05-15")
            
            self.chat_deployments = {
                ModelType.gpt4o_mini: os.getenv("AZURE_OPENAI_GPT4OMINI_DEPLOYMENT"),
                ModelType.gpt4o: os.getenv("AZURE_OPENAI_GPT4O_DEPLOYMENT")
            }
            self.embedding_deployments = {
                EmbeddingModelType.text_embedding_3_small: os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
            }
            if not all(self.chat_deployments.values()) or not all(self.embedding_deployments.values()):
                raise ValueError("Deployments para embeddings e todos os modelos de chat devem ser definidos no .env")

            self.chroma_data_path = os.getenv("CHROMA_DATA_PATH", "chroma_db")
            logging.info(f"Conectando ao ChromaDB em: {self.chroma_data_path}")
            self.chroma_client = chromadb.PersistentClient(path=self.chroma_data_path)
            
            self.collections: Dict[str, chromadb.Collection] = {}
            for model_type in EmbeddingModelType:
                collection_name = f"{self.BASE_COLLECTION_NAME}_{model_type.value}"
                try:
                    collection = self.chroma_client.get_collection(name=collection_name)
                    self.collections[model_type.value] = collection
                    logging.info(f"Conectado com sucesso à coleção '{collection_name}' com {collection.count()} documentos.")
                except Exception as e:
                    logging.error(f"Não foi possível conectar à coleção '{collection_name}'. Erro: {e}")
            
            if not self.collections:
                raise RuntimeError("Nenhuma coleção do ChromaDB foi carregada com sucesso.")

            logging.info("Iniciando construção dos índices BM25 em memória...")
            self.bm25_indices: Dict[str, BM25Okapi] = {}
            self.doc_store_by_strategy: Dict[str, List[Dict]] = {}
            main_collection = self.collections.get(EmbeddingModelType.text_embedding_3_small.value)
            
            if main_collection:
                for strategy in ChunkingStrategy:
                    strategy_name = strategy.value
                    logging.info(f"Construindo índice BM25 para a estratégia: '{strategy_name}'...")
                    results = main_collection.get(where={"chunking_strategy": strategy_name}, include=["metadatas", "documents"])
                    
                    if not results or not results['ids']:
                        logging.warning(f"Nenhum documento encontrado para a estratégia '{strategy_name}'. Índice BM25 ficará vazio.")
                        continue

                    corpus_texts = results['documents']
                    processed_corpus = [_preprocess_text(doc) for doc in corpus_texts]
                    
                    doc_store_list = []
                    for i, doc_id in enumerate(results['ids']):
                        doc_store_list.append({
                            "_id": doc_id,
                            "texto": results['documents'][i],
                            "fonte_documento": results['metadatas'][i].get("documento_origem", "N/A"),
                        })
                    
                    self.bm25_indices[strategy_name] = BM25Okapi(processed_corpus)
                    self.doc_store_by_strategy[strategy_name] = doc_store_list
                    logging.info(f"Índice BM25 para '{strategy_name}' construído com {len(doc_store_list)} documentos.")

            self.historico = {}
            logging.info("ChatbotMPES inicializado com sucesso (com índices BM25).")
        
        except Exception as e:
            logging.critical(f"FALHA CRÍTICA na inicialização do Chatbot: {e}", exc_info=True)
            raise

    def _get_or_create_history(self, session_id: str, max_len: int) -> deque:
        if session_id not in self.historico: self.historico[session_id] = deque(maxlen=max_len)
        elif self.historico[session_id].maxlen != max_len: self.historico[session_id] = deque(list(self.historico[session_id]), maxlen=max_len)
        return self.historico[session_id]

    def _gerar_embedding(self, text: str, embedding_model: EmbeddingModelType) -> list[float] | None:
        deployment_name = self.embedding_deployments.get(embedding_model)
        if not deployment_name: return None
        try:
            r = self.embedding_client.embeddings.create(input=[text], model=deployment_name)
            return r.data[0].embedding
        except Exception as e:
            logging.error(f"Erro ao gerar embedding: {e}")
            return None

    def _format_chroma_results(self, results: dict) -> list[dict]:
        formatted_list = []
        if not results or not results.get('ids'): return []
        ids, docs, metas = results.get('ids', [[]])[0], results.get('documents', [[]])[0], results.get('metadatas', [[]])[0]
        dists = results.get('distances', [[]])[0] if results.get('distances') else [None] * len(ids)
        for i, doc_id in enumerate(ids):
            formatted_list.append({
                "_id": doc_id, "texto": docs[i],
                "fonte_documento": metas[i].get("documento_origem", "N/A"),
                "score": 1 - dists[i] if dists[i] is not None else 0.0
            })
        return formatted_list

    def _reciprocal_rank_fusion(self, results_lists: list[list[dict]], final_k: int, k: int = 60) -> list[dict]:
        ranked_results = {}
        for results in results_lists:
            for rank, result in enumerate(results):
                doc_id = result["_id"]
                if doc_id not in ranked_results:
                    ranked_results[doc_id] = {"score": 0.0, "doc": result}
                ranked_results[doc_id]["score"] += 1.0 / (k + rank + 1)
        
        sorted_results = sorted(ranked_results.values(), key=lambda x: x["score"], reverse=True)
        final_list = [item["doc"] for item in sorted_results]
        return final_list[:final_k]

    def _executar_busca_vetorial(self, collection: chromadb.Collection, pergunta: str, embedding_model: EmbeddingModelType, top_k: int, where_filter: dict) -> list[dict]:
        query_embedding = self._gerar_embedding(pergunta, embedding_model)
        if not query_embedding: return []
        vector_results = collection.query(
            query_embeddings=[query_embedding], 
            n_results=top_k, 
            where=where_filter,
            include=["metadatas", "documents", "distances"]
        )
        return self._format_chroma_results(vector_results)

    def _executar_busca_textual(self, pergunta: str, top_k: int, strategy: ChunkingStrategy) -> list[dict]:
        strategy_name = strategy.value
        bm25_index = self.bm25_indices.get(strategy_name)
        doc_store = self.doc_store_by_strategy.get(strategy_name)
        if not bm25_index or not doc_store:
            logging.warning(f"Índice BM25 para a estratégia '{strategy_name}' não encontrado ou vazio.")
            return []
        processed_query = _preprocess_text(pergunta)
        doc_scores = bm25_index.get_scores(processed_query)
        top_indices = np.argsort(doc_scores)[-top_k:][::-1]
        final_results = []
        for i in top_indices:
            if doc_scores[i] <= 0: continue
            doc = doc_store[i].copy() 
            doc['score'] = doc_scores[i] 
            final_results.append(doc)
        return final_results

    def _buscar_documentos_chroma(self, pergunta: str, search_type: SearchType, max_k: int, strategy: ChunkingStrategy, embedding_model: EmbeddingModelType) -> list[dict]:
        logging.info(f"Executando busca '{search_type.value}' (estratégia: {strategy.value}) com max_k={max_k}")
        collection = self.collections.get(embedding_model.value)
        if not collection:
            logging.error(f"Coleção para o modelo '{embedding_model.value}' não encontrada.")
            return []
        where_filter = {"chunking_strategy": strategy.value}

        if search_type == SearchType.vetorial:
            return self._executar_busca_vetorial(collection, pergunta, embedding_model, max_k, where_filter)
        elif search_type == SearchType.textual:
            return self._executar_busca_textual(pergunta, max_k, strategy)
        elif search_type == SearchType.hibrida:
            vector_results, textual_results = [], []
            try:
                vector_results = self._executar_busca_vetorial(collection, pergunta, embedding_model, max_k, where_filter)
            except Exception as e:
                logging.error(f"Erro na busca vetorial da híbrida: {e}", exc_info=True)
            try:
                textual_results = self._executar_busca_textual(pergunta, max_k, strategy)
            except Exception as e:
                logging.error(f"Erro na busca textual (BM25) da híbrida: {e}", exc_info=True)
            if not vector_results and not textual_results: return []
            return self._reciprocal_rank_fusion([vector_results, textual_results], max_k)
        else:
            logging.warning(f"Tipo de busca desconhecido: '{search_type}'. Usando busca vetorial como padrão.")
            return self._executar_busca_vetorial(collection, pergunta, embedding_model, max_k, where_filter)

    def _construir_contexto(self, resultados: list[dict]) -> tuple[str, list]:
        if not resultados:
            return "Com base nos documentos consultados, não encontrei informações sobre este assunto.", []
        c_str = "Contexto para a resposta (use a informação em 'Fonte' para a citação):\n"
        for res in resultados:
            fonte = res.get('fonte_documento', 'Fonte não informada')
            c_str += f"\n---\nFonte: {fonte}\nConteúdo: {res.get('texto')}\n"
        return c_str, resultados

    def _gerar_resposta(self, contexto: str, pergunta: str, historico_conversa: deque, model: ModelType, temperature: float, system_prompt: str) -> tuple[str, int, list]:
        deployment_name = self.chat_deployments.get(model, self.chat_deployments[ModelType.gpt4o_mini])
        msgs = [{"role": "system", "content": system_prompt}]
        for hp, hr in historico_conversa: msgs.extend([{"role": "user", "content": hp}, {"role": "assistant", "content": hr}])
        msgs.append({"role": "user", "content": f"{contexto}\n\nPergunta: {pergunta}"})
        try:
            resp = self.chat_client.chat.completions.create(model=deployment_name, messages=msgs, temperature=temperature, max_tokens=2048)
            return resp.choices[0].message.content, resp.usage.total_tokens if resp.usage else 0, msgs
        except Exception as e:
            logging.error(f"Erro na API de chat: {e}")
            return "Desculpe, ocorreu um erro ao tentar gerar a resposta.", 0, []

    def _registrar_conversa(self, params: 'PerguntaRequest', resposta: str, blocos: list, tokens: int, messages: list):
        return str(int(datetime.utcnow().timestamp() * 1000))

    def responder(self, request: 'PerguntaRequest') -> Tuple[str, str, str | None]:
        historico_sessao = self._get_or_create_history(request.session_id, request.history_length)
        resultados_da_busca = self._buscar_documentos_chroma(
            pergunta=request.pergunta, 
            search_type=request.search_type, 
            max_k=request.top_k,
            strategy=request.chunking_strategy, 
            embedding_model=request.embedding_model
        )
        contexto, blocos_usados = self._construir_contexto(resultados_da_busca)
        system_prompt = request.system_prompt_override or self.DEFAULT_SYSTEM_PROMPT
        resposta, tokens, messages = self._gerar_resposta(contexto, request.pergunta, historico_sessao, request.model, request.temperature, system_prompt)
        if tokens > 0: historico_sessao.append((request.pergunta, resposta))
        interaction_id = self._registrar_conversa(request, resposta, blocos_usados, tokens, messages)
        return resposta, contexto, interaction_id

# --- 3. Configuração da API FastAPI ---
chatbot_instance = None
@asynccontextmanager
async def lifespan(app: FastAPI):
    global chatbot_instance
    chatbot_instance = ChatbotMPES()
    yield

app = FastAPI(title="Chatbot MPES API", version="2.10.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# --- 4. Modelos de Dados da API (Pydantic) ---
class PerguntaRequest(BaseModel):
    pergunta: str = Field(..., min_length=3, max_length=500)
    session_id: str = Field(default="default-session")
    embedding_model: EmbeddingModelType = Field(EmbeddingModelType.text_embedding_3_small)
    search_type: SearchType = Field(SearchType.hibrida)
    top_k: int = Field(5, ge=1, le=15)
    chunking_strategy: ChunkingStrategy = Field(ChunkingStrategy.recursive_1000_200)
    model: ModelType = Field(ModelType.gpt4o_mini)
    temperature: float = Field(0.1, ge=0.0, le=2.0)
    system_prompt_override: str | None = Field(None, max_length=4000)
    history_length: int = Field(3, ge=0, le=10)
    debug: bool = Field(False)

class RespostaResponse(BaseModel):
    resposta: str
    contexto: str | None
    interaction_id: str | None

class RecuperacaoRequest(BaseModel):
    pergunta: str = Field(..., min_length=3, max_length=500)
    embedding_model: EmbeddingModelType = Field(EmbeddingModelType.text_embedding_3_small)
    search_type: SearchType = Field(SearchType.hibrida)
    max_k: int = Field(15, ge=1, le=50)
    chunking_strategy: ChunkingStrategy = Field(ChunkingStrategy.recursive_1000_200)

class DocumentoRecuperado(BaseModel):
    _id: str
    texto: str
    fonte_documento: str
    score: float

class RecuperacaoResponse(BaseModel):
    documentos_ranqueados: List[DocumentoRecuperado]

# --- ALTERAÇÃO AQUI ---
class GeracaoRequest(BaseModel):
    pergunta: str = Field(..., min_length=3, max_length=500)
    contexto: str = Field(..., description="Contexto completo já formatado, obtido da fase de recuperação.")
    session_id: str = Field(default="default-session-generation")
    model: ModelType = Field(ModelType.gpt4o_mini)
    temperature: float = Field(0.1, ge=0.0, le=2.0)
    history_length: int = Field(0, ge=0, le=10)
    system_prompt_override: str | None = Field(None, max_length=4000) # <-- 1. CAMPO ADICIONADO

class GeracaoResponse(BaseModel):
    resposta: str
    tokens_usados: int

# --- 5. Endpoints da API (Protegidos) ---
@app.post("/responder", response_model=RespostaResponse, tags=["Chat"],
          dependencies=[Depends(get_api_key)])
async def responder_endpoint(request: PerguntaRequest):
    if not chatbot_instance: raise HTTPException(status_code=503, detail="Serviço indisponível.")
    try:
        resposta, contexto, interaction_id = chatbot_instance.responder(request)
        return RespostaResponse(resposta=resposta, contexto=contexto, interaction_id=interaction_id)
    except Exception as e:
        logging.critical(f"Erro não tratado no endpoint /responder: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Ocorreu um erro interno inesperado.")

@app.post("/recuperar_contexto", response_model=RecuperacaoResponse, tags=["Experimento"],
          dependencies=[Depends(get_api_key)])
async def recuperar_contexto_endpoint(request: RecuperacaoRequest):
    if not chatbot_instance:
        raise HTTPException(status_code=503, detail="Serviço indisponível.")
    try:
        resultados_ranqueados = chatbot_instance._buscar_documentos_chroma(
            pergunta=request.pergunta,
            search_type=request.search_type,
            max_k=request.max_k,
            strategy=request.chunking_strategy,
            embedding_model=request.embedding_model
        )
        documentos_response = [DocumentoRecuperado(**doc) for doc in resultados_ranqueados]
        return RecuperacaoResponse(documentos_ranqueados=documentos_response)
    except Exception as e:
        logging.error(f"Erro no endpoint /recuperar_contexto: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Erro interno na recuperação de contexto.")

# --- ALTERAÇÃO AQUI ---
@app.post("/gerar_resposta", response_model=GeracaoResponse, tags=["Experimento"],
          dependencies=[Depends(get_api_key)])
async def gerar_resposta_endpoint(request: GeracaoRequest):
    """
    Executa apenas a fase de geração de resposta a partir de um contexto fornecido.
    Ideal para experimentos em duas fases.
    """
    if not chatbot_instance:
        raise HTTPException(status_code=503, detail="Serviço indisponível.")
    try:
        historico_sessao = chatbot_instance._get_or_create_history(request.session_id, request.history_length)
        
        # --- 2. LÓGICA DO PROMPT ALTERADA ---
        # Usa o prompt do request, se fornecido; senão, usa o padrão.
        system_prompt_final = request.system_prompt_override or chatbot_instance.DEFAULT_SYSTEM_PROMPT
        
        resposta, tokens, _ = chatbot_instance._gerar_resposta(
            contexto=request.contexto,
            pergunta=request.pergunta,
            historico_conversa=historico_sessao,
            model=request.model,
            temperature=request.temperature,
            system_prompt=system_prompt_final # <-- Passa o prompt correto
        )
        return GeracaoResponse(resposta=resposta, tokens_usados=tokens)
    except Exception as e:
        logging.error(f"Erro no endpoint /gerar_resposta: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Erro interno na geração da resposta.")

# --- 6. Bloco de Execução ---
if __name__ == "__main__":
    print("=" * 80)
    print("== INICIANDO SERVIDOR DA API DO CHATBOT MPES (v2.10 - BM25 + Auth + Prompt) ==")
    if not API_KEY:
        print("== AVISO: API_KEY não definida no .env. A API ESTÁ DESPROTEGIDA. ==")
    else:
        print("== Autenticação por Chave de API ATIVADA. ==")
    print("== Acesse a documentação interativa: http://127.0.0.1:8000/docs ==")
    print("=" * 80)
    uvicorn.run("chatbot:app", host="0.0.0.0", port=8000, reload=True)